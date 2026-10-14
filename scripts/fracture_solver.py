"""Mixed physical residuals consumed by the fracture neural solver.

Bulk plastic history, interface separation history and water inventories are
committed together after an accepted loading step. Pressure is continuous across
the ideal zero-skin-resistance crack; its two displacement traces are independent.
"""
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve
import torch

from face_capacity_pinn import elastic_constants, saturation, pressure_relative_permeability
from fracture_physics import CrackMesh, mohr_coulomb_rankine, cohesive_contact, fracture_water, interface_kinematics
from finite_strain import material_response, swept_face_increment


class FractureSystem:
    def __init__(self, b, spacing=.025, control='displacement', normal_spacing=None, tangent_spacing=None, interface_order=2):
        self.b, self.control = b, control
        self.mesh = m = CrackMesh(b, spacing, normal_spacing, tangent_spacing)
        self.n = len(m.xy)
        self.constants = elastic_constants(m, b)
        if b.bulk_kinematics == 'finite_strain':
            self.elastic_metric_old = torch.diag_embed(torch.exp(2*self.constants[2]))
        else:
            self.plastic_old = torch.zeros((len(m.tri), 3, 3))
        self.water_old = torch.full((len(m.tri), 3), b.porosity0*b.saturation0)
        self.maximum_old = torch.zeros((len(m.crack_cells), interface_order))
        self.slip_old = torch.zeros_like(self.maximum_old)
        self.crack_water_old = torch.zeros_like(self.maximum_old)
        # Rigid shield contact is unilateral. Soil can lift off its roof;
        # clamping that surface would prevent the observed uplifted block.
        x, z = m.points_np.T
        roof = np.flatnonzero((x < -1.e-10) & np.isclose(z, 0.))
        roof = np.r_[roof, m.crack_minus[0]]
        floor = np.flatnonzero((x <= 1.e-10) & np.isclose(z, -b.shield_diameter_m))
        self.contact = np.r_[roof, floor]
        self.contact_sign = np.r_[np.ones(len(roof)), -np.ones(len(floor))]
        contact_weights = []
        for ids in [roof, floor]:
            lengths = np.diff(x[ids])
            weights = np.zeros(len(ids))
            weights[:-1] += lengths/2
            weights[1:] += lengths/2
            contact_weights.extend(weights)
        self.contact_weights = np.array(contact_weights)
        self.contact_initial = b.density_kg_m3*b.gravity_m_s2*(b.cover_m-z[self.contact])
        m.free[self.contact, 1] = True
        points, weights = np.polynomial.legendre.leggauss(interface_order)
        self.crack_fraction = (points+1)/2
        self.crack_weights = torch.tensor(weights/2)
        self.shape = torch.tensor(np.c_[1-self.crack_fraction, self.crack_fraction])
        self.crack_nodes = torch.tensor(m.crack_cells)
        self.bulk_dofs = (3*m.triangles_np[:, :, None]+np.arange(3)).reshape(-1, 9)
        self.crack_dofs = (3*m.crack_cells[:, :, None]+np.arange(3)).reshape(-1, 12)
        z = torch.einsum('qn,en->eq', self.shape, m.xy[self.crack_nodes[:, :2], 1])
        sv = -b.density_kg_m3*b.gravity_m_s2*(b.cover_m-z)
        sh = b.k0*sv
        initial = torch.diag_embed(torch.stack([sh, sv], -1))
        initial += b.saturation0*b.pressure0*torch.eye(2)
        normal = m.crack_frame[:, 0]
        traction = torch.einsum('eqij,ej->eqi', initial, normal)
        self.traction0 = torch.einsum('eji,eqi->eqj', m.crack_frame, traction)
        self.initial_face = b.k0*b.density_kg_m3*b.gravity_m_s2*(b.cover_m-m.xy[:, 1])
        self.initial_face_mean = float(m.face_weights@self.initial_face/m.face_weights.sum())
        self.face_nodes = torch.tensor(m.face[np.argsort(m.points_np[m.face, 1])])
        self.face_edges_local = torch.tensor(np.c_[np.arange(len(m.face)-1), np.arange(1, len(m.face))])
        self.face_position_old = m.xy[self.face_nodes].clone()
        pe = self.initial_face[m.face_edges].numpy()
        ef = m.face_lengths.numpy()[:, None]/6*np.stack([2*pe[:, 0]+pe[:, 1], pe[:, 0]+2*pe[:, 1]], 1)
        self.initial_force = np.zeros(self.n)
        np.add.at(self.initial_force, m.face_edges.numpy().ravel(), ef.ravel())
        # Eliminate fixed displacements and identify the two pressure traces.
        independent = np.c_[m.free.numpy(), m.parent == np.arange(self.n)].ravel()
        self.free = np.flatnonzero(independent)
        lookup = np.full(3*self.n, -1)
        lookup[self.free] = np.arange(len(self.free))
        for duplicate in m.crack_plus:
            lookup[3*duplicate+2] = lookup[3*m.parent[duplicate]+2]
        valid = lookup >= 0
        self.expansion = coo_matrix((np.ones(valid.sum()), (np.flatnonzero(valid), lookup[valid])),
                                   shape=(3*self.n, len(self.free))).tocsc()
        self.lookup = lookup
        self.top = np.unique(m.parent[m.top])
        self.nfield = len(self.free)
        self.ndof = self.nfield+len(self.top)+(control != 'pressure')
        scale = np.c_[np.repeat((b.pressure_scale_pa*np.sqrt(m.weights.numpy()))[:, None], 2, 1),
                      m.weights.numpy()*b.porosity0].ravel()
        self.scale = np.asarray(self.expansion.T@scale).ravel()
        self.qscale = b.pressure_scale_pa/b.filter_resistance_pa_s_m*spacing
        self.state = np.zeros(self.ndof)
        self.target = 0.
        self.time_s = 0.
        self.face_volume_old = 0.
        self.injection_old = 0.

    def unpack(self, state):
        fields = np.asarray(self.expansion@state[:self.nfield]).reshape(self.n, 3)
        u = fields[:, :2]*self.b.displacement_scale_m
        p = self.b.pressure0+fields[:, 2]*self.b.pressure_scale_pa
        q = state[self.nfield:self.nfield+len(self.top)]*self.qscale
        load = state[-1]*self.b.pressure_scale_pa if self.control != 'pressure' else self.target
        return fields, u, p, q, load

    def bulk(self, values, details=False):
        b, m = self.b, self.mesh
        u = values[:, :, :2]*b.displacement_scale_m
        p = b.pressure0+values[:, :, 2]*b.pressure_scale_pa
        gradient = torch.einsum('tni,tnj->tij', u, m.grad)
        lam, mu, eps0, _, _ = self.constants
        finite = b.bulk_kinematics == 'finite_strain'
        if finite:
            deformation = torch.eye(3).expand(len(gradient), 3, 3).clone()
            deformation[:, :2, :2] += gradient
            material = material_response(deformation, self.elastic_metric_old, lam, mu, eps0.sum(1), b)
            stress, j = material['stress'], material['J']
        else:
            j = 1+gradient[:, 0, 0]+gradient[:, 1, 1]
            strain = torch.diag_embed(eps0)
            strain[:, :2, :2] += (gradient+gradient.transpose(1, 2))/2
            stress, plastic, shear_multiplier, tensile_multiplier = mohr_coulomb_rankine(
                strain, self.plastic_old, lam, mu, b.phi_deg, b.dilation_deg, b.effective_cohesion_pa, b.tensile_strength_pa)
            material = {'plastic': plastic, 'plastic_multiplier': shear_multiplier+tensile_multiplier,
                        'shear_multiplier': shear_multiplier, 'tensile_multiplier': tensile_multiplier}
        pb = (saturation(p, b)*p).mean(1)
        total = stress[:, :2, :2]-pb[:, None, None]*torch.eye(2)
        if finite:
            inverse = torch.linalg.inv(deformation[:, :2, :2])
            piola = j[:, None, None]*total@inverse.transpose(-1, -2)
        else:
            piola = total
        force = m.area[:, None, None]*torch.einsum('tij,tnj->tni', piola, m.grad)
        force[:, :, 1] += m.area[:, None]/3*b.density_kg_m3*b.gravity_m_s2
        porosity = 1-(1-b.porosity0)/j
        permeability = b.permeability_m2*(porosity/b.porosity0)**3*((1-b.porosity0)/(1-porosity))**2
        mobility = permeability/b.viscosity_pa_s*pressure_relative_permeability(p, b).mean(1)
        elevation = m.xy[m.tri, 1]+u[:, :, 1] if finite else m.xy[m.tri, 1]
        head = p+1000*b.gravity_m_s2*elevation
        head_gradient = torch.einsum('tn,tnj->tj', head, m.grad)
        if finite:
            flow_metric = j[:, None, None]*inverse@inverse.transpose(-1, -2)
            head_gradient = torch.einsum('tij,tj->ti', flow_metric, head_gradient)
        flow = b.step_time_s*m.area[:, None]*mobility[:, None]*torch.einsum('tni,ti->tn', m.grad, head_gradient)
        water = (j[:, None]-1+b.porosity0)*saturation(p, b)+b.porosity0*b.water_compressibility_pa_inv*(p-b.pressure0)
        mass = m.area[:, None]/3*(water-self.water_old)+flow
        residual = torch.cat([force, mass[:, :, None]], 2).reshape(-1, 9)
        if details:
            return {**material, 'water': water, 'J': j, 'gradient': gradient,
                    'stress': stress, 'total_stress': total, 'first_piola': piola}
        return residual

    def face_boundary(self, values, load):
        """Current-face pressure, filtration and accepted-path swept volume.

        Keep the prescribed depth gradient and center it on the current
        length-weighted height. Thus the scalar load still specifies the mean
        total pressure exactly, even when the face bends or changes length.
        """
        b = self.b
        position = self.mesh.xy[self.face_nodes]+values[:, :2]*b.displacement_scale_m
        edges = self.face_edges_local
        nodes = position[edges]
        direction = nodes[:, 1]-nodes[:, 0]
        length = torch.linalg.vector_norm(direction, dim=1)
        normal_measure = torch.stack([direction[:, 1], -direction[:, 0]], -1)
        zmean = (length*nodes[:, :, 1].mean(1)).sum()/length.sum()
        pressure = self.initial_face_mean+load+b.k0*b.density_kg_m3*b.gravity_m_s2*(zmean-position[:, 1])
        pe = pressure[edges]
        integrated = torch.stack([2*pe[:, 0]+pe[:, 1], pe[:, 0]+2*pe[:, 1]], 1)/6
        force = -integrated[:, :, None]*normal_measure[:, None, :]
        pore = b.pressure0+values[:, 2]*b.pressure_scale_pa
        difference = (pore-pressure)[edges]
        filtration = length[:, None]/(6*b.filter_resistance_pa_s_m)*torch.stack(
            [2*difference[:, 0]+difference[:, 1], difference[:, 0]+2*difference[:, 1]], 1)
        local = torch.cat([force, (b.step_time_s*filtration)[:, :, None]], -1)
        residual = torch.zeros_like(values).index_add(0, edges.flatten(), local.reshape(-1, 3))
        volume = self.face_volume_old+swept_face_increment(position, self.face_position_old, edges)
        return {'residual': residual, 'swept_volume': volume, 'inflow': -filtration.sum(),
                'pressure': pressure, 'length': length, 'position': position}

    def valid_deformation(self, fields):
        """Check pore-space collapse before evaluating a constitutive trial."""
        u = fields[self.mesh.triangles_np, :2]*self.b.displacement_scale_m
        gradient = np.einsum('tni,tnj->tij', u, self.mesh.gradients_np)
        j = np.linalg.det(np.eye(2)+gradient) if self.b.bulk_kinematics == 'finite_strain' else 1+np.trace(gradient, axis1=1, axis2=2)
        return bool(np.isfinite(j).all() and np.min(j) > 1-self.b.porosity0)

    def interface(self, values, details=False):
        b, m = self.b, self.mesh
        u = values[:, :, :2]*b.displacement_scale_m
        p = b.pressure0+values[:, :, 2]*b.pressure_scale_pa
        geometry = interface_kinematics(u, m.xy[self.crack_nodes], self.shape, b.interface_geometry == 'corotated')
        jump_local = geometry['jump']
        length = geometry['length']
        stretch = length/m.crack_length
        cohesive, width, damage, maximum, slip = cohesive_contact(jump_local, self.traction0, self.maximum_old, self.slip_old, b)
        pnodes = (p[:, :2]+p[:, 2:])/2
        pressure = torch.einsum('qn,en->eq', self.shape, pnodes)
        water, conductance, sf = fracture_water(pressure, width, b)
        water = stretch[:, None]*water  # Inventory per reference length.
        # Fluid traction and cohesive traction act on the same cavity faces.
        # At zero aperture the retention law recovers the matrix Bishop term.
        pb = sf*pressure
        traction = torch.stack([cohesive[:, :, 0]-stretch[:, None]*pb, cohesive[:, :, 1]], -1)
        force = m.crack_length[:, None, None]*torch.einsum('q,eqani,eqa->eni', self.crack_weights, geometry['derivative'], traction)
        storage = m.crack_length[:, None]*torch.einsum('q,qn,eq->en', self.crack_weights, self.shape, water-self.crack_water_old)
        dh = (pnodes[:, 1]-pnodes[:, 0])/length+1000*b.gravity_m_s2*geometry['frame'][:, 1, 1]
        flux = (conductance*self.crack_weights).sum(1)*dh
        mass = storage+b.step_time_s*flux[:, None]*torch.tensor([-1., 1.])
        mass = torch.cat([mass/2, mass/2], 1)
        residual = torch.cat([force, mass[:, :, None]], 2).reshape(-1, 12)
        if details:
            return {'water': water, 'aperture': width, 'damage': damage, 'maximum': maximum,
                    'slip': slip, 'traction': traction, 'pressure': pressure, 'flux': -flux,
                    'jump': jump_local, 'saturation': sf, 'current_length': length,
                    'frame': geometry['frame']}
        return residual

    def residual(self, state, jacobian=False):
        b, m = self.b, self.mesh
        fields, u, p, q, load = self.unpack(state)
        finite = b.bulk_kinematics == 'finite_strain'
        full = np.zeros(3*self.n)
        matrices = []
        storage_sum = 0.
        for function, nodes, dofs in [(self.bulk, m.triangles_np, self.bulk_dofs),
                                      (self.interface, m.crack_cells, self.crack_dofs)]:
            values = torch.tensor(fields[nodes], requires_grad=bool(jacobian))
            local = function(values)
            dimension = local.shape[1]
            if jacobian:
                active_rows = list(range(2, dimension, 3)) if jacobian == 'hydraulic' else list(range(dimension))
                derivative = np.zeros((len(local), dimension, dimension))
                for index, k in enumerate(active_rows):
                    derivative[:, k] = torch.autograd.grad(local[:, k].sum(), values,
                        retain_graph=index < len(active_rows)-1)[0].reshape(-1, dimension).numpy()
                rows = np.broadcast_to(dofs[:, :, None], derivative.shape).ravel()
                cols = np.broadcast_to(dofs[:, None, :], derivative.shape).ravel()
                matrices.append(coo_matrix((derivative.ravel(), (rows, cols)), shape=(3*self.n, 3*self.n)).tocsc())
            local_np = local.detach().numpy()
            np.add.at(full, dofs.ravel(), local_np.ravel())
            storage_sum += local_np[:, 2::3].sum()
        if finite:
            face_dofs = (3*self.face_nodes[:, None]+torch.arange(3)).numpy().ravel()
            face_values = torch.tensor(np.r_[fields[self.face_nodes].ravel(), load/b.pressure_scale_pa], requires_grad=bool(jacobian))

            def face_residual(v):
                return self.face_boundary(v[:-1].reshape(-1, 3), v[-1]*b.pressure_scale_pa)['residual'].flatten()

            full[face_dofs] += face_residual(face_values).detach().numpy()
            if jacobian:
                face_derivative = torch.autograd.functional.jacobian(face_residual, face_values, vectorize=True).numpy()
                rr = np.broadcast_to(face_dofs[:, None], face_derivative[:, :-1].shape).ravel()
                cc = np.broadcast_to(face_dofs[None, :], face_derivative[:, :-1].shape).ravel()
                matrices.append(coo_matrix((face_derivative[:, :-1].ravel(), (rr, cc)), shape=(3*self.n, 3*self.n)).tocsc())
        else:
            full[0::3] -= self.initial_force+m.face_weights.numpy()*load
        gap = self.contact_sign*u[self.contact, 1]
        support = np.maximum(self.contact_initial-b.interface_stiffness_pa_m*gap, 0.)
        full[3*self.contact+1] -= self.contact_weights*self.contact_sign*support
        if not finite:
            factor = b.step_time_s*m.face_weights.numpy()/b.filter_resistance_pa_s_m
            full[2::3] += factor*(p-self.initial_face.numpy()-load)
        full[3*self.top+2] += b.step_time_s*q
        weak = np.asarray(self.expansion.T@full).ravel()/self.scale
        a, c = -p[self.top]/b.pressure_scale_pa, q/self.qscale
        radius = np.sqrt(a*a+c*c+1.e-30)
        rows = [weak, radius-a-c]
        if self.control == 'displacement':
            rows.append(np.array([(m.face_weights.numpy()@u[:, 0]/float(m.face_weights.sum())-self.target)/b.displacement_scale_m]))
        elif self.control == 'mean_pressure':
            rows.append(np.array([(self.initial_face_mean+load-self.target)/b.pressure_scale_pa]))
        elif self.control == 'injection':
            if finite:
                volume = self.injection(state)-(self.target-self.injection_old)
            else:
                swept = m.face_weights.numpy()@u[:, 0]-self.face_volume_old
                volume = swept+storage_sum+b.step_time_s*q.sum()-(self.target-self.injection_old)
            rows.append(np.array([volume/(b.shield_diameter_m*b.displacement_scale_m)]))
        result = np.concatenate(rows)
        if not jacobian:
            return result
        matrix = sum(matrices)
        contact_dof = 3*self.contact+1
        stiffness = self.contact_weights*b.interface_stiffness_pa_m*b.displacement_scale_m*(support > 0)
        matrix += coo_matrix((stiffness, (contact_dof, contact_dof)), shape=matrix.shape).tocsc()
        if not finite:
            volume_derivative = np.asarray(matrix[2::3].sum(0)).ravel()
            volume_derivative[0::3] += m.face_weights.numpy()*b.displacement_scale_m
            pdof = 3*np.arange(self.n)+2
            matrix += coo_matrix((factor*b.pressure_scale_pa, (pdof, pdof)), shape=matrix.shape).tocsc()
        matrix = diags(1/self.scale)@self.expansion.T@matrix@self.expansion
        extra = np.zeros((self.nfield, self.ndof-self.nfield))
        extra[self.lookup[3*self.top+2], np.arange(len(self.top))] = b.step_time_s*self.qscale/self.scale[self.lookup[3*self.top+2]]
        if self.control != 'pressure':
            derivative = np.zeros(3*self.n)
            if finite:
                derivative[face_dofs] = face_derivative[:, -1]
            else:
                derivative[0::3] = -m.face_weights.numpy()*b.pressure_scale_pa
                derivative[2::3] = -factor*b.pressure_scale_pa
            extra[:, -1] = np.asarray(self.expansion.T@derivative).ravel()/self.scale
        matrix = hstack([matrix, csc_matrix(extra)], format='csc')
        ids = np.arange(len(self.top))
        boundary = coo_matrix((1-a/radius, (ids, self.lookup[3*self.top+2])), shape=(len(ids), self.ndof)).tocsc()
        boundary += coo_matrix((c/radius-1, (ids, self.nfield+ids)), shape=boundary.shape).tocsc()
        matrix = vstack([matrix, boundary], format='csc')
        if self.control == 'displacement':
            row = np.zeros(self.ndof)
            np.add.at(row, self.lookup[3*m.face], m.face_weights.numpy()[m.face]/float(m.face_weights.sum()))
            matrix = vstack([matrix, csc_matrix(row[None])], format='csc')
        elif self.control == 'mean_pressure':
            row = np.zeros(self.ndof)
            row[-1] = 1.
            matrix = vstack([matrix, csc_matrix(row[None])], format='csc')
        elif self.control == 'injection':
            row = self.injection(state, True)[1] if finite else np.r_[self.expansion.T@volume_derivative, np.full(len(self.top), b.step_time_s*self.qscale), 0.]
            row /= b.shield_diameter_m*b.displacement_scale_m
            matrix = vstack([matrix, csc_matrix(row[None])], format='csc')
        return result, matrix

    def solve(self, max_iterations=70):
        """Physical residual correction for kernel checks and neural training."""
        state = self.state.copy()
        for iteration in range(max_iterations):
            residual, jacobian = self.residual(state, True)
            if np.max(np.abs(residual)) < 1.e-8:
                self.state = state
                return iteration
            change = spsolve(jacobian, -residual)
            if not np.isfinite(change).all():
                raise ArithmeticError('Singular physical tangent')
            norm = np.linalg.norm(residual)
            fraction = 1.
            while fraction > 2**-18:
                trial = state+fraction*change
                fields = self.unpack(trial)[0]
                if self.valid_deformation(fields):
                    trial_norm = np.linalg.norm(self.residual(trial))
                    if np.isfinite(trial_norm) and trial_norm < norm:
                        state = trial
                        break
                fraction /= 2
            else:
                raise ArithmeticError(f'Physical correction stalled, residual {norm:.5g}')
        raise ArithmeticError(f'Physical correction did not converge, residual {norm:.5g}')

    def injection(self, state, jacobian=False):
        """Incremental chamber supply: swept face volume plus filtered water."""
        b, m = self.b, self.mesh
        fields, u, p, _, load = self.unpack(state)
        if b.bulk_kinematics == 'finite_strain':
            values = torch.tensor(np.r_[fields[self.face_nodes].ravel(), load/b.pressure_scale_pa], requires_grad=jacobian)
            face = self.face_boundary(values[:-1].reshape(-1, 3), values[-1]*b.pressure_scale_pa)
            amount = face['swept_volume']-self.face_volume_old+b.step_time_s*face['inflow']
            if not jacobian:
                return float(amount)
            local = torch.autograd.grad(amount, values)[0].numpy()
            derivative = np.zeros(self.ndof)
            dofs = self.lookup[(3*self.face_nodes[:, None]+torch.arange(3)).numpy().ravel()]
            active = dofs >= 0
            np.add.at(derivative, dofs[active], local[:-1][active])
            if self.control != 'pressure':
                derivative[-1] = local[-1]
            return float(amount.detach()), derivative
        weights = m.face_weights.numpy()
        amount = weights@u[:, 0]-self.face_volume_old
        amount += b.step_time_s*weights@(self.initial_face.numpy()+load-p)/b.filter_resistance_pa_s_m
        if not jacobian:
            return float(amount)
        derivative = np.zeros(self.ndof)
        face = m.face
        np.add.at(derivative, self.lookup[3*face], weights[face]*b.displacement_scale_m)
        np.add.at(derivative, self.lookup[3*face+2], -b.step_time_s*weights[face]*b.pressure_scale_pa/b.filter_resistance_pa_s_m)
        if self.control != 'pressure':
            derivative[-1] = b.step_time_s*weights.sum()*b.pressure_scale_pa/b.filter_resistance_pa_s_m
        return float(amount), derivative

    def inspect(self):
        b, m = self.b, self.mesh
        fields, u, p, q, load = self.unpack(self.state)
        bulk = self.bulk(torch.tensor(fields[m.triangles_np]), True)
        crack = self.interface(torch.tensor(fields[m.crack_cells]), True)
        delta_bulk = float((m.area[:, None]/3*(bulk['water']-self.water_old)).sum())
        delta_crack = float((m.crack_length[:, None]*self.crack_weights*(crack['water']-self.crack_water_old)).sum())
        inflow = float(m.face_weights.numpy()@(self.initial_face.numpy()+load-p)/b.filter_resistance_pa_s_m)
        swept = float(m.face_weights.numpy()@u[:, 0])
        if b.bulk_kinematics == 'finite_strain':
            face = self.face_boundary(torch.tensor(fields[self.face_nodes]), load)
            inflow, swept = float(face['inflow']), float(face['swept_volume'])
        residual = self.residual(self.state)
        physical = residual[:self.nfield]*self.scale
        mass = physical[self.free % 3 == 2]
        force = physical[self.free % 3 != 2]
        width = crack['aperture'].detach().numpy()
        damage = crack['damage'].detach().numpy()
        row = {'time_s': self.time_s+b.step_time_s, 'face_surcharge_kpa': load/1000,
            'face_mean_total_pressure_kpa': (self.initial_face_mean+load)/1000,
            'mean_face_displacement_mm': float(m.face_weights.numpy()@u[:, 0]/float(m.face_weights.sum()))*1000,
            'face_swept_volume_ml_per_m': swept*1.e6,
            'maximum_surface_uplift_mm': float(u[m.top, 1].max())*1000,
            'max_aperture_mm': float(width.max())*1000,
            'near_surface_quadrature_aperture_mm': float(width[-1, -1])*1000,
            'maximum_cohesive_damage': float(damage.max()),
            'plastic_area_m2': float(m.area[bulk['plastic_multiplier'] > 1.e-12].sum()),
            'tensile_plastic_area_m2': float(m.area[bulk['tensile_multiplier'] > 1.e-12].sum()),
            'maximum_principal_effective_stress_pa': float(torch.linalg.eigvalsh(bulk['stress'])[:, -1].max()),
            'bulk_water_change_ml_per_m': delta_bulk*1.e6,
            'fracture_water_change_ml_per_m': delta_crack*1.e6,
            'fracture_water_ml_per_m': float((m.crack_length[:, None]*self.crack_weights*crack['water']).sum())*1.e6,
            'inflow_ml_per_m_s': inflow*1.e6, 'outflow_ml_per_m_s': float(q.sum())*1.e6,
            'mass_defect_ml_per_m': (delta_bulk+delta_crack-b.step_time_s*(inflow-q.sum()))*1.e6,
            'local_mass_residual_l1_ml_per_m': float(np.abs(mass).sum())*1.e6,
            'force_residual_max_n_per_m': float(np.abs(force).max()),
            'min_J': float(bulk['J'].min()), 'max_J': float(bulk['J'].max()),
            'maximum_displacement_gradient': float(bulk['gradient'].abs().max()),
            'max_pore_pressure_kpa': float(p.max())/1000}
        saved = {'xy': m.points_np, 'triangles': m.triangles_np, 'u_m': u, 'pressure_pa': p,
                 'crack_cells': m.crack_cells, 'crack_length_m': m.crack_length.numpy(),
                 'crack_quadrature_fraction': self.crack_fraction, 'outflow_m2_s': q,
                 'interface_geometry': np.array(b.interface_geometry),
                 'bulk_kinematics': np.array(b.bulk_kinematics)}
        if b.bulk_kinematics == 'finite_strain':
            saved['face_swept_volume_m2'] = np.array(swept)
            saved['face_pressure_pa'] = face['pressure'].numpy()
            row['crown_total_pressure_kpa'] = float(face['pressure'][-1])/1000
        saved.update({'bulk_'+key: value.detach().numpy() for key, value in bulk.items()})
        saved.update({'crack_'+key: value.detach().numpy() for key, value in crack.items()})
        return row, saved

    def accept(self, saved):
        if str(saved.get('interface_geometry', 'reference')) != self.b.interface_geometry:
            raise ValueError('Interface geometry differs from accepted history; restart from the initial state')
        if str(saved.get('bulk_kinematics', 'small_strain')) != self.b.bulk_kinematics:
            raise ValueError('Bulk kinematics differs from accepted history; restart from the initial state')
        if self.b.bulk_kinematics == 'finite_strain':
            self.elastic_metric_old = torch.tensor(saved['bulk_elastic_reference_metric'])
            self.face_volume_old = float(saved['face_swept_volume_m2'])
            self.face_position_old = self.mesh.xy[self.face_nodes]+torch.tensor(saved['u_m'][self.face_nodes])
        else:
            self.plastic_old = torch.tensor(saved['bulk_plastic'])
            self.face_volume_old = float(self.mesh.face_weights.numpy()@saved['u_m'][:, 0])
        self.water_old = torch.tensor(saved['bulk_water'])
        self.maximum_old = torch.tensor(saved['crack_maximum'])
        self.slip_old = torch.tensor(saved['crack_slip'])
        self.crack_water_old = torch.tensor(saved['crack_water'])
        self.time_s += self.b.step_time_s
        if self.control == 'injection':
            self.injection_old = self.target
