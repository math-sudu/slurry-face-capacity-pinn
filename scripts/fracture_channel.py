"""Subcell pressure in an opening crack, with paired matrix-wall exchange.

Each integration point has its own pressure and inventory. An open upstream
part can receive water while its closed downstream neighbour blocks the tip.
"""
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, vstack
import torch

from fracture_physics import cohesive_contact, fracture_water, fracture_retention, interface_kinematics
from fracture_solver import FractureSystem
from fracture_transport import OpeningHistory


class ChannelSystem(FractureSystem):
    def __init__(self, b, spacing=.03, control='displacement', normal_spacing=None, tangent_spacing=None, interface_order=2):
        if control not in ('displacement', 'mean_pressure'):
            raise ValueError('The channel branch requires displacement or mean_pressure control')
        super().__init__(b, spacing, control, normal_spacing, tangent_spacing, interface_order)
        self.nchannel = interface_order*len(self.mesh.crack_cells)
        self.pf_indices = np.arange(self.ndof-1, self.ndof-1+self.nchannel)
        self.ndof += self.nchannel
        self.state = np.zeros(self.ndof)
        self.aperture_old = torch.zeros_like(self.crack_water_old)
        length = self.mesh.crack_length.numpy()
        self.channel_volume = (self.mesh.crack_length[:, None]*self.crack_weights).flatten()
        self.channel_scale = self.channel_volume.numpy()*(2*b.fracture_energy_j_m2/b.tensile_strength_pa)
        z = self.mesh.xy[self.crack_nodes[:, :2], 1]
        normal = self.mesh.crack_frame[:, 0]
        self.normal_traction_endpoints = (-b.density_kg_m3*b.gravity_m_s2*(b.cover_m-z)
            *(b.k0*normal[:, :1]**2+normal[:, 1:]**2)+b.saturation0*b.pressure0)
        self.opening_history = OpeningHistory(length, self.crack_fraction,
            self.normal_traction_endpoints.numpy()/b.interface_stiffness_pa_m)
        indices = self.lookup[self.crack_dofs]
        if np.any(indices < 0):
            raise ValueError('Crack channel requires free displacement traces')
        self.active = np.unique(np.r_[indices.ravel(), self.pf_indices, self.ndof-1])
        if b.bulk_kinematics == 'finite_strain':
            face_indices = self.lookup[(3*self.face_nodes[:, None]+torch.arange(3)).numpy()]
            if np.any(face_indices < 0):
                raise ValueError('Moving face requires free displacement and pressure nodes')
            self.active = np.union1d(self.active, face_indices.ravel())
            self.local_face = np.searchsorted(self.active, face_indices)
        self.local_indices = np.searchsorted(self.active, indices).reshape(-1, 4, 3)
        self.local_pf = np.searchsorted(self.active, self.pf_indices)
        self.local_load = np.searchsorted(self.active, self.ndof-1)
        self.local_scale = np.ones(len(self.active))
        field = self.active < self.nfield
        self.local_scale[field] = self.scale[self.active[field]]
        self.local_scale[self.local_pf] = self.channel_scale
        pressure_dofs = self.lookup[3*self.mesh.crack_cells+2]
        self.local_water = np.unique(np.r_[np.searchsorted(self.active, pressure_dofs).ravel(), self.local_pf])

    def interface(self, values, details=False):
        if details:
            return self.channel(torch.tensor(self.state[self.active]), True)
        return values.reshape(-1, 12)*0

    def channel(self, values, details=False):
        b, m = self.b, self.mesh
        nodal = values[self.local_indices]
        u = nodal[:, :, :2]*b.displacement_scale_m
        pm = b.pressure0+nodal[:, :, 2]*b.pressure_scale_pa
        pf = b.pressure0+values[self.local_pf]*b.pressure_scale_pa
        load = values[self.local_load]*b.pressure_scale_pa
        geometry = interface_kinematics(u, m.xy[self.crack_nodes], self.shape, b.interface_geometry == 'corotated')
        jump = geometry['jump']
        length = geometry['length']
        stretch = length/m.crack_length
        cohesive, width, damage, maximum, slip = cohesive_contact(jump, self.traction0, self.maximum_old, self.slip_old, b)
        matrix_pressure = torch.einsum('qn,en->eq', self.shape, (pm[:, :2]+pm[:, 2:])/2)
        pressure = pf.reshape_as(width)
        water, _, sf = fracture_water(pressure, width, b)
        water = stretch[:, None]*water  # Stored and differenced per reference length.
        pb = sf*pressure
        # The cohesive law is nominal. Cauchy fluid pressure is pulled back by
        # the current/reference line stretch; both act through the full gap variation.
        traction = torch.stack([cohesive[:, :, 0]-stretch[:, None]*pb, cohesive[:, :, 1]], -1)
        force = m.crack_length[:, None, None]*torch.einsum('q,eqani,eqa->eni', self.crack_weights, geometry['derivative'], traction)
        _, relative = fracture_retention(pressure, width, b)
        effective_endpoints = (geometry['jump_nodes'][..., 0]
                               +self.normal_traction_endpoints/b.interface_stiffness_pa_m)
        transmissibility, end_transmissibility = self.opening_history.transmissibility(effective_endpoints, b, length)
        # A newly opened cavity exposes its wall area. Its small volume does
        # not shrink that area; the converged pressure jump tends to zero as
        # the opening vanishes. A fully closed interval exchanges no water.
        wet_wall = (width > 0) | (self.aperture_old > 0)
        exchange = torch.where(wet_wall, 2*(pressure-matrix_pressure)/b.fracture_wall_resistance_pa_s_m, 0.)
        nodal_exchange = length[:, None]*torch.einsum('q,qn,eq->en', self.crack_weights, self.shape, exchange)
        matrix_mass = -torch.cat([nodal_exchange/2, nodal_exchange/2], 1)*b.step_time_s
        # Independent inventory in each subcell prevents the closed tip
        # from blocking chamber water from the already open upstream part.
        relative = relative.flatten()
        z = geometry['midline'][..., 1]
        zcenter = torch.einsum('qn,en->eq', self.shape, z).flatten()
        head = pf+1000*b.gravity_m_s2*zcenter
        mouth = self.initial_face[m.crack_minus[0]]+load
        if b.bulk_kinematics == 'finite_strain':
            mouth = self.face_boundary(values[self.local_face], load)['pressure'][-1]
        mouth_head = mouth+1000*b.gravity_m_s2*z[0, 0]
        # Upwind relative mobility by hydraulic potential. Harmonic resistance
        # uses only the geometric cubic aperture; a dry receiving cell must not
        # impose its mobility on the saturated pressure boundary.
        inlet_relative = fracture_retention(mouth, width.flatten()[0], b)[1]
        inlet_relative = torch.where(mouth_head >= head[0], inlet_relative, relative[0])
        inlet = transmissibility[0]*inlet_relative*(mouth_head-head[0])
        upstream = torch.where(head[:-1] >= head[1:], relative[:-1], relative[1:])
        middle = transmissibility[1:-1]*upstream*(head[:-1]-head[1:])
        outlet = (transmissibility[-1]*relative[-1]*(head[-1]-1000*b.gravity_m_s2*z[-1, -1])).clamp_min(0)
        edges = torch.cat([inlet[None], middle, outlet[None]])
        storage = self.channel_volume*(water-self.crack_water_old).flatten()
        wall_volume = (length[:, None]*self.crack_weights).flatten()
        mass = storage+b.step_time_s*(wall_volume*exchange.flatten()+edges[1:]-edges[:-1])
        local = torch.cat([force, matrix_mass[:, :, None]], 2)
        result = torch.zeros_like(values).index_add(0, torch.tensor(self.local_indices.ravel()), local.flatten())
        empty = (width.flatten() == 0) & (self.aperture_old.flatten() == 0)
        reference = (pf-matrix_pressure.flatten())/b.pressure_scale_pa*torch.tensor(self.channel_scale)
        result = result.index_add(0, torch.tensor(self.local_pf), torch.where(empty, reference, mass))
        if details:
            return {'water': water, 'aperture': width, 'damage': damage, 'maximum': maximum,
                    'slip': slip, 'traction': traction, 'pressure': pressure, 'flux': (edges[1:]+edges[:-1])/2,
                    'edge_flux': edges, 'jump': jump, 'saturation': sf, 'exchange': exchange,
                    'storage_change': storage, 'physical_mass': mass,
                    'effective_endpoints': effective_endpoints,
                    'current_length': length, 'frame': geometry['frame'],
                    'edge_transmissibility': transmissibility,
                    'endpoint_edge_transmissibility': end_transmissibility}
        return result/torch.tensor(self.local_scale)

    def residual(self, state, jacobian=False):
        base = super().residual(state, jacobian)
        if jacobian:
            base, matrix = base
        values = torch.tensor(state[self.active], requires_grad=bool(jacobian))
        local = self.channel(values)
        result = np.r_[base[:-1], np.zeros(self.nchannel), base[-1]]
        result[self.active] += local.detach().numpy()
        if not jacobian:
            return result
        rows = self.local_water if jacobian == 'hydraulic' else np.arange(len(self.active))
        derivative = torch.autograd.functional.jacobian(lambda v: self.channel(v)[rows], values, vectorize=True).numpy()
        rr = np.broadcast_to(self.active[rows, None], derivative.shape).ravel()
        cc = np.broadcast_to(self.active[None, :], derivative.shape).ravel()
        channel_matrix = coo_matrix((derivative.ravel(), (rr, cc)), shape=(self.ndof, self.ndof)).tocsc()
        matrix = vstack([matrix[:-1], csc_matrix((self.nchannel, self.ndof)), matrix[-1]], format='csc')
        return result, matrix+channel_matrix

    def inspect(self):
        row, fields = super().inspect()
        inflow, outflow = fields['crack_edge_flux'][[0, -1]]
        row['channel_inflow_ml_per_m_s'] = float(inflow)*1.e6
        row['channel_outflow_ml_per_m_s'] = float(outflow)*1.e6
        row['inflow_ml_per_m_s'] += float(inflow)*1.e6
        row['outflow_ml_per_m_s'] += float(outflow)*1.e6
        row['mass_defect_ml_per_m'] -= self.b.step_time_s*(inflow-outflow)*1.e6
        row['max_crack_pressure_kpa'] = float(fields['crack_pressure'].max())/1000
        fields['channel_pressure_pa'] = self.b.pressure0+self.state[self.pf_indices]*self.b.pressure_scale_pa
        fields.update(self.opening_history.saved(fields['crack_effective_endpoints']))
        return row, fields

    def injection(self, state, jacobian=False):
        base = super().injection(state, jacobian)
        values = torch.tensor(state[self.active], requires_grad=jacobian)
        inlet = self.channel(values, True)['edge_flux'][0]*self.b.step_time_s
        if not jacobian:
            return base+float(inlet)
        amount, derivative = base
        derivative[self.active] += torch.autograd.grad(inlet, values)[0].numpy()
        return amount+float(inlet.detach()), derivative

    def accept(self, saved):
        super().accept(saved)
        self.aperture_old = torch.tensor(saved['crack_aperture'])
        self.opening_history.accept(saved)
