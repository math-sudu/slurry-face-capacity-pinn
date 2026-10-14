"""Constrained, jump-enriched neural residual solver for a fluid-filled crack.

Gaussian hidden radii and displacement weights are trained on force residuals.
A differentiable hydraulic projection enforces the matrix/crack inventory and
seepage equations inside training. Its tangent is obtained by implicit
differentiation. Loading continuation is encoded exactly in the neural output.
This is a mixed neural/constraint solver, with pressure as an auxiliary field.
No solved displacement is used as a training label or neural initialization.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve, lstsq, svd
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
import torch

from face_capacity_pinn import ROOT, segments
from fracture_physics import FractureProblem, crack_coordinates, graded_distances
from fracture_solver import FractureSystem


def save_result(output, result):
    temporary = output/'result.next.json'
    temporary.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    deadline = time.monotonic()+5.
    while True:
        try:
            temporary.replace(output/'result.json')
            return
        except PermissionError:
            # Windows readers can briefly hold the current index without
            # delete sharing. Keep the completed step and retry that rename.
            if time.monotonic() >= deadline:
                raise
            time.sleep(.1)


class JumpNetwork:
    def __init__(self, system, center_spacing=.07, learn_radii=True, volume_control=False, trace_enrichment=False, local_center_spacing=None, normal_center_spacing=None):
        self.system, self.learn_radii = system, learn_radii
        self.volume_control = volume_control
        self.pressure_control = system.control == 'mean_pressure'
        if volume_control and self.pressure_control:
            raise ValueError('Supply and mean face pressure cannot both be prescribed')
        self.free_face = volume_control or self.pressure_control
        # The final scalar is the pressure reaction for displacement control,
        # but the free mean displacement for prescribed mean pressure.
        self.face_parameter = -2 if volume_control else -1
        m, b = system.mesh, system.b
        if normal_center_spacing is not None and (local_center_spacing is None or not 0 < normal_center_spacing <= local_center_spacing):
            raise ValueError('Normal neural spacing requires a larger or equal local center spacing')
        xs = segments([-b.shield_length_m, 0., b.box_width_m-b.shield_length_m], center_spacing)
        zs = segments([b.cover_m-b.box_height_m, -b.shield_diameter_m, 0., b.cover_m], center_spacing)
        xx, zz = np.meshgrid(xs, zs)
        seeds = np.c_[xx.ravel(), zz.ravel()]
        valid = ~((seeds[:, 0] < 0) & (seeds[:, 1] < 0) & (seeds[:, 1] > -b.shield_diameter_m))
        # Local neural features resolve the loaded face and its two corners;
        # the rest of the box retains a much smaller smooth trial space.
        if local_center_spacing is None:
            indices = np.unique(cKDTree(m.points_np).query(seeds[valid])[1])
            x, z = m.points_np.T
            near = (x > -.07) & (x < .10) & (z > -b.shield_diameter_m-.05)
            indices = np.unique(np.r_[indices, np.flatnonzero(near)])
            self.centers = np.unique(m.points_np[indices], axis=0)
        else:
            if local_center_spacing <= 0:
                raise ValueError('Local neural center spacing must be positive')
            if normal_center_spacing is None:
                local_x = segments([-.07, 0., .10], local_center_spacing)
            else:
                local_x = np.r_[-graded_distances(.07, normal_center_spacing, local_center_spacing)[:0:-1],
                                graded_distances(.10, normal_center_spacing, local_center_spacing)]
            xx, zz = np.meshgrid(local_x,
                segments([-b.shield_diameter_m-.05, -b.shield_diameter_m, 0., b.cover_m], local_center_spacing))
            local = np.c_[xx.ravel(), zz.ravel()]
            valid_local = ~((local[:, 0] < 0) & (local[:, 1] < 0) & (local[:, 1] > -b.shield_diameter_m))
            # Continuous centers remain identical when only quadrature changes.
            reference_centers = np.unique(np.r_[seeds[valid], local[valid_local]], axis=0)
            self.centers = crack_coordinates(reference_centers, b)
            order = np.lexsort((self.centers[:, 1], self.centers[:, 0]))
            self.centers, reference_centers = self.centers[order], reference_centers[order]
        x, z = m.points_np.T
        branch = (z >= -1.e-12) & (x <= b.crack_slope*z+1.e-12)
        branch[m.crack_plus] = False
        self.branch = branch.astype(float)
        selected = (self.centers[:, 1] >= -1.e-12) & (self.centers[:, 0] <= b.crack_slope*self.centers[:, 1]+1.e-12)
        candidates = self.centers[selected]
        self.branch_centers = candidates
        self.distance = ((m.points_np[:, None]-self.centers)**2).sum(-1)
        self.branch_distance = ((m.points_np[:, None]-candidates)**2).sum(-1)
        self.nc, self.nb = len(self.centers), len(candidates)
        local = (self.centers[:, 0] > -.07) & (self.centers[:, 0] < .10) & (self.centers[:, 1] > -b.shield_diameter_m-.05)
        local_radius = m.spacing if local_center_spacing is None else local_center_spacing
        self.radius = 1.3*np.where(local, local_radius, center_spacing)
        local_branch = (candidates[:, 0] > -.07) & (candidates[:, 0] < .10)
        self.branch_radius = 1.3*np.where(local_branch, local_radius, center_spacing)
        if normal_center_spacing is not None:
            # Thin wetting layers need narrow normal features without making
            # their tangential support equally small. Hidden log radii still
            # train each metric; centers and aspect ratios are mesh-independent.
            cx, cz = reference_centers.T
            local = (cx >= -.07) & (cx <= .10) & (cz >= -b.shield_diameter_m-.05)
            tangent = np.where(local, local_center_spacing, center_spacing)
            normal = np.where(local, np.minimum(local_center_spacing, normal_center_spacing+.35*np.abs(cx)), center_spacing)
            delta = m.reference_points_np[:, None]-reference_centers
            self.distance = delta[:, :, 0]**2*(tangent/normal)**2+delta[:, :, 1]**2
            self.branch_distance = self.distance[:, selected]
            self.radius = 1.3*tangent
            self.branch_radius = self.radius[selected]
        self.jump_stop = 2*self.nc+2*self.nb
        # Local trace functions give the crack banks and loaded face their own
        # virtual-work tests. Bulk displacement remains a learned neural field.
        self.trace_nodes = np.unique(np.r_[m.crack_minus, m.crack_plus, m.face]) if trace_enrichment else np.empty(0, dtype=int)
        self.outputs = self.jump_stop+2*len(self.trace_nodes)
        self.hidden = self.nc+self.nb if learn_radii else 0
        self.extra = 2 if volume_control else 1
        self.nparameter = self.outputs+self.hidden+self.extra
        self.parameters = np.zeros(self.nparameter)
        self.mask = m.free.numpy()
        left, right = -b.shield_length_m, b.box_width_m-b.shield_length_m
        lift = (m.points_np[:, 0]-left)*(right-m.points_np[:, 0])
        self.face_average = m.face_weights.numpy()/float(m.face_weights.sum())
        self.lift = lift/(self.face_average@lift)

    @staticmethod
    def basis(distance, radius):
        squared = distance/radius[None]**2
        raw = np.exp(-squared)
        basis = raw/raw.sum(1)[:, None]
        return basis, 2*squared*basis

    def evaluate(self, parameters, derivative=False):
        s = self.system
        w = parameters[:2*self.nc].reshape(self.nc, 2)
        v = parameters[2*self.nc:self.jump_stop].reshape(self.nb, 2)
        logs = parameters[self.outputs:self.outputs+self.hidden] if self.learn_radii else np.zeros(self.nc+self.nb)
        base, db = self.basis(self.distance, self.radius*np.exp(logs[:self.nc]))
        jump, dj = self.basis(self.branch_distance, self.branch_radius*np.exp(logs[self.nc:]))
        bulk = base@w
        enrichment = jump@v
        displacement = bulk+self.branch[:, None]*enrichment
        displacement[self.trace_nodes] += parameters[self.jump_stop:self.outputs].reshape(-1, 2)
        displacement *= self.mask
        target = parameters[self.face_parameter] if self.free_face else s.target/s.b.displacement_scale_m
        displacement[:, 0] += self.lift*(target-self.face_average@displacement[:, 0])
        state = s.state.copy()
        fields = np.zeros((s.n, 3))
        fields[:, :2] = displacement
        mechanical = s.free % 3 != 2
        state[:s.nfield][mechanical] = fields.ravel()[s.free][mechanical]
        state[-1] = (s.target-s.initial_face_mean)/s.b.pressure_scale_pa if self.pressure_control else parameters[-1]
        if not derivative:
            return state
        jac = np.zeros((3*s.n, self.nparameter))
        for component in range(2):
            jac[component::3, component:2*self.nc:2] = base
            jac[component::3, 2*self.nc+component:self.jump_stop:2] = self.branch[:, None]*jump
            jac[3*self.trace_nodes+component, self.jump_stop+2*np.arange(len(self.trace_nodes))+component] = 1.
        if self.learn_radii:
            for component in range(2):
                jac[component::3, self.outputs:self.outputs+self.nc] = db*(w[:, component][None]-bulk[:, component, None])
            for component in range(2):
                jac[component::3, self.outputs+self.nc:self.outputs+self.hidden] = self.branch[:, None]*dj*(v[:, component][None]-enrichment[:, component, None])
        mask = np.c_[self.mask, np.zeros(s.n)].ravel()
        jac *= mask[:, None]
        jac[0::3] -= self.lift[:, None]*(self.face_average@jac[0::3])[None]
        if self.free_face:
            jac[0::3, self.face_parameter] = self.lift
        result = np.zeros((s.ndof, self.nparameter))
        result[:s.nfield] = jac[s.free]
        if not self.pressure_control:
            result[-1, -1] = 1.
        return state, result

    @property
    def output_parameters(self):
        return np.r_[np.arange(self.outputs), np.arange(self.nparameter-self.extra, self.nparameter)]


class ConservativePINN:
    def __init__(self, system, center_spacing=.07, learn_radii=True, volume_control=False, trace_enrichment=False, local_center_spacing=None, normal_center_spacing=None):
        self.system = system
        self.network = JumpNetwork(system, center_spacing, learn_radii, volume_control, trace_enrichment, local_center_spacing, normal_center_spacing)
        self.mechanical = np.flatnonzero(system.free % 3 != 2)
        self.hydraulic = np.r_[np.flatnonzero(system.free % 3 == 2), np.arange(system.nfield, system.ndof-1)]
        self.volume_control = volume_control
        self.volume_factor = system.b.pressure_scale_pa/system.b.displacement_scale_m
        self.force_scale = system.scale[self.mechanical]
        if volume_control:
            self.force_scale = np.r_[self.force_scale, system.b.pressure_scale_pa*system.b.shield_diameter_m]

    def forces(self, state, residual):
        force = residual[self.mechanical]*self.system.scale[self.mechanical]
        if self.volume_control:
            s = self.system
            volume = s.injection(state)-(s.target-s.injection_old)
            force = np.r_[force, volume*self.volume_factor]
        return force

    def project(self, state, tangent=True):
        """Solve auxiliary inventory equations for this neural displacement."""
        state = state.copy()
        ids, s = self.hydraulic, self.system
        for iteration in range(40):
            residual, jacobian = s.residual(state, 'hydraulic')
            r = residual[ids]
            factor = splu(jacobian[ids][:, ids].tocsc())
            if np.max(np.abs(r)) < 1.e-9:
                if tangent:
                    residual, jacobian = s.residual(state, True)
                return state, residual, jacobian, factor
            change = factor.solve(-r)
            norm = np.linalg.norm(r)
            fraction = 1.
            while fraction > 2**-16:
                trial = state.copy()
                trial[ids] += fraction*change
                rr = s.residual(trial)[ids]
                if np.isfinite(rr).all() and np.linalg.norm(rr) < norm:
                    state = trial
                    break
                fraction /= 2
            else:
                raise ArithmeticError(f'Hydraulic projection stalled ({norm:.4g})')
        raise ArithmeticError(f'Hydraulic projection did not settle ({norm:.4g})')

    def train(self, max_iterations=120, verbose=False):
        """Variable projection: equilibrate outputs, then learn hidden radii.

        Output weights satisfy neural virtual work, including a load-reaction
        test. Hidden features minimize the force left outside that trial space,
        differentiating through both water balance and output equilibration.
        """
        net, s = self.network, self.system
        parameters = net.parameters.copy()
        parameters, data, count = self.equilibrate(parameters)
        force, derivative, tests, transform = data
        scale = self.force_scale
        loss = float((force/scale)@(force/scale))
        initial_loss = loss
        accepted_features = 0
        # Feature optimization is an inexact inner iteration; loading advances
        # only from a re-equilibrated neural field and a converged water layer.
        for outer in range(max_iterations if net.learn_radii else 0):
            if verbose and outer % 5 == 0:
                print(f'Feature iteration {outer}: force_loss={loss:.6g}', flush=True)
            hidden = np.arange(net.outputs, net.outputs+net.hidden)
            output = net.output_parameters
            test_derivative = self.test_radius_derivative(parameters, force)
            rhs = tests.T@derivative[:, hidden]+test_derivative
            reduced_matrix = self.virtual_work_tangent(tests, derivative, transform)
            output_response = transform@lstsq(reduced_matrix, -transform.T@rhs,
                                             cond=1.e-12, lapack_driver='gelsy')[0]
            reduced = (derivative[:, hidden]+derivative[:, output]@output_response)/scale[:, None]
            norms = np.maximum(np.linalg.norm(reduced, axis=0), 1.e-5)
            jj = reduced/norms
            change = solve(jj.T@jj+.05*np.eye(len(hidden)), -jj.T@(force/scale), assume_a='pos')/norms
            # Bound the proposed radius change before backtracking. The search
            # fraction is relative to this step, so a weakly sensitive feature
            # cannot make the first trial smaller than an absolute cutoff.
            change *= min(1., .2/max(np.max(np.abs(change)), .2))
            fraction = 1.
            accepted = False
            while fraction >= 1.e-4:
                trial = self.feature_predictor(parameters, change, output_response@change, fraction)
                try:
                    trial, trial_data, iterations = self.equilibrate(trial)
                except ArithmeticError:
                    fraction /= 2
                    continue
                trial_loss = float((trial_data[0]/scale)@(trial_data[0]/scale))
                if trial_loss < loss:
                    accepted = True
                    break
                fraction /= 2
            if not accepted:
                break
            improvement = (loss-trial_loss)/max(loss, 1.e-20)
            parameters, data = trial, trial_data
            force, derivative, tests, transform = data
            loss = trial_loss
            count += iterations
            accepted_features += 1
            if improvement < 1.e-5:
                break
        net.parameters = parameters
        s.state, residual, _, _ = self.project(net.evaluate(parameters), tangent=False)
        return {'neural_iterations': count, 'feature_updates': accepted_features,
                'neural_loss': loss, 'force_loss_before_feature_learning': initial_loss,
                'constraint_max': float(np.max(np.abs(residual[self.hydraulic]))),
                'optimizer_status': 'neural_virtual_work_equilibrated'}

    def feature_predictor(self, parameters, hidden_change, output_change, fraction):
        """Retract a linearized neural-field update into the changed basis.

        Large, cancelling output coefficients make their first-order update a
        poor finite-step predictor when radii move. Fit a correction to that
        update, preserving existing weak modes and the zero-step identity.
        The target is the network's own linearized displacement.
        """
        net = self.network
        output = net.output_parameters
        hidden = np.arange(net.outputs, net.outputs+net.hidden)
        state, derivative = net.evaluate(parameters, True)
        change = derivative[:, hidden]@hidden_change+derivative[:, output]@output_change
        desired = (state+fraction*change)[self.mechanical]
        trial = parameters.copy()
        trial[hidden] += fraction*hidden_change
        trial[output] += fraction*output_change
        displacement_outputs = output if net.pressure_control else output[:-1]
        baseline, basis = net.evaluate(trial, True)
        trial[displacement_outputs] += lstsq(
            basis[self.mechanical][:, displacement_outputs], desired-baseline[self.mechanical],
            cond=1.e-9, lapack_driver='gelsy')[0]
        return trial

    def tangent(self, parameters):
        s, net = self.system, self.network
        state, neural = net.evaluate(parameters, True)
        state, residual, physical, factor = self.project(state)
        s.state = state
        derivative = physical@neural
        dp = -factor.solve(derivative[self.hydraulic])
        self.pressure_tangent = dp
        derivative = derivative[self.mechanical]+physical[self.mechanical][:, self.hydraulic]@dp
        scale = s.scale[self.mechanical]
        derivative *= scale[:, None]
        force = self.forces(state, residual)
        output = net.output_parameters
        tests = neural[self.mechanical][:, output].copy()
        if self.volume_control:
            _, volume_derivative = s.injection(state, True)
            state_derivative = neural.copy()
            state_derivative[self.hydraulic] = dp
            derivative = np.vstack([derivative, (volume_derivative@state_derivative)*self.volume_factor])
            tests = np.vstack([tests, np.zeros((1, len(output)))])
            tests[-1, -1] = 1.
        elif not net.pressure_control:
            nodal_lift = np.zeros(3*s.n)
            nodal_lift[0::3] = net.lift
            tests[:, -1] = nodal_lift[s.free[self.mechanical]]
        return force, derivative, tests

    def virtual_work_tangent(self, tests, derivative, transform):
        # Contract the independent field directions before the two sides
        # meet. Transforming an already assembled coefficient matrix loses
        # digits when nearly dependent neural features cancel.
        return (tests@transform).T@(derivative[:, self.network.output_parameters]@transform)

    def equilibrate(self, parameters):
        net, s = self.network, self.system
        parameters = parameters.copy()
        output = net.output_parameters
        for iteration in range(45):
            data = self.tangent(parameters)
            force, derivative, tests = data
            if iteration == 0:
                # Remove coefficient gauges before solving the physical tangent.
                # Rank is selected from the neural displacement map, not from
                # the material stiffness (which could remove a soft mode).
                _, singular, right = svd(tests[:, :-1], full_matrices=False)
                rank = np.count_nonzero(singular > singular[0]*1.e-9)
                transform = np.zeros((len(output), rank+1))
                transform[:-1, :rank] = right[:rank].T/singular[:rank]
                transform[-1, -1] = 1.
                active_tests = tests@transform
                norm_test = np.maximum(np.linalg.norm(active_tests, axis=0), 1.e-12)
            work = active_tests.T@force/norm_test
            if np.max(np.abs(work)) < .1:
                return parameters, (*data, transform), iteration+1
            active_matrix = self.virtual_work_tangent(tests, derivative, transform)/norm_test[:, None]
            change = transform@lstsq(active_matrix, -work, cond=1.e-12, lapack_driver='gelsy')[0]
            fraction = 1.
            norm = np.linalg.norm(work)
            while fraction > 2**-16:
                trial = parameters.copy()
                trial[output] += fraction*change
                state = net.evaluate(trial)
                state[self.hydraulic] += fraction*(self.pressure_tangent[:, output]@change)
                fields = s.unpack(state)[0]
                if not s.valid_deformation(fields):
                    fraction /= 2
                    continue
                try:
                    state, residual, _, _ = self.project(state, tangent=False)
                except ArithmeticError:
                    fraction /= 2
                    continue
                new_force = self.forces(state, residual)
                if np.linalg.norm(active_tests.T@new_force/norm_test) < norm:
                    parameters = trial
                    s.state = state
                    break
                fraction /= 2
            else:
                raise ArithmeticError(f'Neural virtual work stalled ({norm:.5g} N/m)')
        raise ArithmeticError(f'Neural virtual work did not settle ({norm:.5g} N/m)')

    def test_radius_derivative(self, parameters, force):
        """Contraction of d(test functions)/d(log radius) with physical force."""
        net, s = self.network, self.system
        nodal = np.zeros(3*s.n)
        nodal[s.free[self.mechanical]] = force[:-1] if self.volume_control else force
        f = nodal.reshape(s.n, 3)[:, :2].copy()
        f[:, 0] -= net.face_average*(net.lift@f[:, 0])
        f *= net.mask
        logs = parameters[net.outputs:net.outputs+net.hidden]
        base, db = net.basis(net.distance, net.radius*np.exp(logs[:net.nc]))
        jump, dj = net.basis(net.branch_distance, net.branch_radius*np.exp(logs[net.nc:]))
        result = np.zeros((len(net.output_parameters), net.hidden))
        for component in range(2):
            bf = f[:, component]
            result[component:2*net.nc:2, :net.nc] = np.diag(bf@db)-base.T@(bf[:, None]*db)
            jf = bf*net.branch
            result[2*net.nc+component:net.jump_stop:2, net.nc:] = np.diag(jf@dj)-jump.T@(jf[:, None]*dj)
        return result


def run(b, output, steps, spacing, center_spacing, increment, dt, resume=False, learn_radii=True, max_iterations=120, channel=False, injection_step_ml=None, trace_enrichment=False, normal_spacing=None, tangent_spacing=None, local_center_spacing=None, normal_center_spacing=None, interface_order=2, pressure_step_pa=None, hold_time_s=60.):
    """Integrate displacement, supplied-volume or total-pressure loading.

    A pressure run holds P_mean = stage * pressure_step_pa for hold_time_s.
    The existing depth profile is retained: P(z) = K0*rho*g*(cover-z)+load,
    so load = P_mean-P_initial_mean and P_crown = P_mean-K0*rho*g*D/2.
    Supply is the computed face sweep plus filtered and crack-inlet water.
    """
    torch.set_num_threads(2)
    pressure_control = pressure_step_pa is not None
    volume_control = injection_step_ml is not None
    if pressure_control:
        if volume_control:
            raise ValueError('Supply and mean face pressure cannot both be prescribed')
        if not np.isfinite([pressure_step_pa, hold_time_s, dt]).all() or min(pressure_step_pa, hold_time_s, dt) <= 0:
            raise ValueError('Pressure steps and hold/integration times must be positive')
    control = 'mean_pressure' if pressure_control else 'displacement'
    if channel:
        from fracture_channel import ChannelSystem
        s = ChannelSystem(b, spacing, control=control, normal_spacing=normal_spacing, tangent_spacing=tangent_spacing, interface_order=interface_order)
    else:
        s = FractureSystem(b, spacing, control=control, normal_spacing=normal_spacing, tangent_spacing=tangent_spacing, interface_order=interface_order)
    model = ConservativePINN(s, center_spacing, learn_radii, volume_control, trace_enrichment, local_center_spacing, normal_center_spacing)
    output.mkdir(parents=True, exist_ok=True)
    settings = {'spacing_m': spacing, 'center_spacing_m': center_spacing,
                'opening_step_m': increment, 'dt_s': dt, 'hydraulic_projection': True,
                'learn_radii': learn_radii, 'max_iterations': max_iterations,
                'fracture_retention': b.fracture_retention,
                'bulk_constitutive': 'mohr_coulomb_rankine',
                'interface_shear_cohesion': 'effective_cohesion'}
    for name, value in [('normal_spacing_m', normal_spacing), ('tangent_spacing_m', tangent_spacing),
                        ('local_center_spacing_m', local_center_spacing), ('normal_center_spacing_m', normal_center_spacing)]:
        if value is not None:
            settings[name] = value
    if learn_radii:
        settings['feature_predictor'] = 'neural_field_retraction'
    if interface_order != 2:
        settings['interface_order'] = interface_order
    if channel:
        settings['hydraulic_model'] = 'subcell_fracture_pressure'
        settings['channel_flux'] = 'upstream_relative_mobility'
        settings['channel_transport'] = 'continuous_history_series_slit'
    if volume_control:
        settings.pop('opening_step_m')
        settings['injection_step_ml_per_m'] = injection_step_ml
        increment = injection_step_ml*1.e-6
    if pressure_control:
        settings.pop('opening_step_m')
        settings.update(control='mean_pressure', pressure_step_pa=pressure_step_pa,
                        hold_time_s=hold_time_s, pressure_profile='initial_geostatic_plus_uniform_surcharge')
    if trace_enrichment:
        settings['trace_enrichment'] = True
    result = {'kind': 'conservative_jump_neural_fracture', 'problem': asdict(b), 'settings': settings,
              'quadrature': {'nodes': s.n, 'physical_dofs': s.ndof},
              'neural': {'parameters': model.network.nparameter, 'base_centers': model.network.nc,
                         'branch_centers': model.network.nb, 'trace_nodes': len(model.network.trace_nodes)}, 'history': [], 'status': 'running'}
    result_path = output/'result.next.json'
    if not result_path.exists():
        result_path = output/'result.json'
    if result_path.exists():
        if not resume:
            raise FileExistsError(output)
        # A completed checkpoint can await the Windows-locked index rename.
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result['settings'].get('bulk_constitutive') != settings['bulk_constitutive']:
            raise ValueError('Saved mechanics predates MC/Rankine; start the revised material from its initial state')
        saved_problem = FractureProblem(**result['problem'])
        if asdict(saved_problem) != asdict(b) or result['settings'] != settings:
            raise ValueError('Resume differs from saved model/training settings')
        # Earlier accepted paths used one restart archive. New paths resume
        # from the immutable accepted-step file named by result.json.
        with np.load(output/result.get('checkpoint', 'restart.npz')) as archive:
            checkpoint = {key: archive[key].copy() for key in archive.files}
        model.network.parameters = checkpoint['parameters'].copy()
        s.state = checkpoint['state'].copy()
        s.accept(checkpoint)
        s.time_s = result['history'][-1]['time_s']
        if volume_control or pressure_control:
            s.injection_old = result['history'][-1]['injected_volume_ml_per_m']*1.e-6
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        target = (s.injection_old if volume_control else result['history'][-1]['mean_face_displacement_mm']/1000) if result['history'] else 0.
        advance = min(increment, 2*result['history'][-1]['step_time_s']/dt*increment) if result['history'] else increment
        if pressure_control:
            target = s.time_s
            advance = min(dt, 2*result['history'][-1]['step_time_s']) if result['history'] else dt
        finish = steps*(hold_time_s if pressure_control else increment)
        while target < finish-1.e-12:
            index = len(result['history'])+1
            if pressure_control:
                stage = int(np.floor((s.time_s+1.e-10)/hold_time_s))+1
                s.target = stage*pressure_step_pa
                # A failed solve reduces the integration time at the same
                # plateau pressure; it must not silently turn a step into a ramp.
                s.b = replace(b, step_time_s=min(advance, stage*hold_time_s-target))
            else:
                s.target = min(target+advance, finish)
                s.b = replace(b, step_time_s=dt*(s.target-target)/increment)
            saved_state = s.state.copy()
            saved_parameters = model.network.parameters.copy()
            try:
                training = model.train(max_iterations, verbose=True)
            except ArithmeticError as error:
                s.state = saved_state
                model.network.parameters = saved_parameters
                advance = s.b.step_time_s/2 if pressure_control else advance/2
                if advance < (dt if pressure_control else increment)/1024:
                    result.update(status='unaccepted_trial', error=str(error))
                    if pressure_control:
                        result['unaccepted_trial'] = {'start_time_s': s.time_s,
                            'step_time_s': s.b.step_time_s, 'target_mean_pressure_pa': s.target}
                    save_result(output, result)
                    raise
                print(f'Retry from accepted history: increment={advance:.6g}, {error}', flush=True)
                continue
            except MemoryError as error:
                result.update(status='resource_interrupted', error=str(error))
                save_result(output, result)
                raise
            row, fields = s.inspect()
            row.update(training)
            row['step'] = index
            row['step_time_s'] = s.b.step_time_s
            if volume_control:
                row['injected_volume_ml_per_m'] = s.target*1.e6
                row['volume_control_defect_ml_per_m'] = (s.injection(s.state)-(s.target-s.injection_old))*1.e6
            if pressure_control:
                supplied = s.injection(s.state)
                row.update(pressure_stage=stage, target_mean_pressure_kpa=s.target/1000,
                           crown_total_pressure_kpa=row.get('crown_total_pressure_kpa', float(s.initial_face[s.mesh.crack_minus[0]]+s.unpack(s.state)[-1])/1000),
                           injected_volume_ml_per_m=(s.injection_old+supplied)*1.e6,
                           injection_increment_ml_per_m=supplied*1.e6,
                           supply_rate_ml_per_m_s=supplied/s.b.step_time_s*1.e6)
            print(json.dumps(row), flush=True)
            result['history'].append(row)
            checkpoint_name = f'fields_{index:03d}.npz'
            np.savez_compressed(output/checkpoint_name, parameters=model.network.parameters, state=s.state, **fields)
            result['checkpoint'] = checkpoint_name
            s.accept(fields)
            if volume_control:
                s.injection_old = s.target
            elif pressure_control:
                s.injection_old += supplied
            target = s.time_s if pressure_control else s.target
            advance = min(2*advance, dt if pressure_control else increment)
            result.pop('error', None)
            result.pop('unaccepted_trial', None)
            result.update(status='complete' if target >= finish-1.e-12 else 'running', elapsed_s=time.perf_counter()-start)
            save_result(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover', type=float, default=.09)
    parser.add_argument('--spacing', type=float, default=.03)
    parser.add_argument('--centers', type=float, default=.075)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--opening-step', type=float, default=.00002)
    parser.add_argument('--dt', type=float, default=60.)
    parser.add_argument('--max-iterations', type=int, default=120)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fixed-radii', action='store_true')
    parser.add_argument('--channel', action='store_true')
    parser.add_argument('--effective-cohesion', type=float, default=1000.)
    parser.add_argument('--initial-suction', type=float, default=FractureProblem().initial_suction_pa,
                        help='Independent initial suction in Pa; default retains the previous hydraulic scenario')
    parser.add_argument('--fracture-wall-resistance', type=float, default=2.e9,
                        help='Fracture-wall resistance in Pa s/m, independent of the face filter')
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument('--injection-step-ml', type=float)
    loading.add_argument('--pressure-step-pa', type=float,
                        help='Absolute mean total face pressure is stage times this increment (Pa)')
    parser.add_argument('--hold-time', type=float, default=60.,
                        help='Duration of each pressure plateau in seconds; --dt bounds integration steps')
    parser.add_argument('--trace-enrichment', action='store_true')
    parser.add_argument('--normal-spacing', type=float)
    parser.add_argument('--tangent-spacing', type=float)
    parser.add_argument('--local-centers', type=float)
    parser.add_argument('--normal-centers', type=float)
    parser.add_argument('--interface-order', type=int, default=2)
    parser.add_argument('--interface-geometry', choices=['reference', 'corotated'], default='corotated',
                        help='Use current interface kinematics; reference reproduces earlier paths')
    parser.add_argument('--bulk-kinematics', choices=['small_strain', 'finite_strain'], default='finite_strain',
                        help='Use multiplicative bulk plasticity; small_strain reproduces earlier paths')
    parser.add_argument('--fracture-retention', choices=['matrix_limit_plus_gap', 'capillary_bridge'], default='capillary_bridge',
                        help='Aperture-limited bridge retention; matrix_limit_plus_gap reproduces pre-repair archives')
    parser.add_argument('--output', type=Path, default=ROOT/'results/ideation/fracture_neural_c09')
    args = parser.parse_args()
    run(FractureProblem(cover_m=args.cover, step_time_s=args.dt, effective_cohesion_pa=args.effective_cohesion,
        initial_suction_pa=args.initial_suction, fracture_wall_resistance_pa_s_m=args.fracture_wall_resistance,
        interface_geometry=args.interface_geometry, bulk_kinematics=args.bulk_kinematics,
        fracture_retention=args.fracture_retention), args.output, args.steps, args.spacing,
        args.centers, args.opening_step, args.dt, args.resume, not args.fixed_radii, args.max_iterations, args.channel, args.injection_step_ml, args.trace_enrichment,
        args.normal_spacing, args.tangent_spacing, args.local_centers, args.normal_centers, args.interface_order,
        args.pressure_step_pa, args.hold_time)
