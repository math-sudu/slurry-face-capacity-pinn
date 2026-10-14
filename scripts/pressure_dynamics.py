"""Evolve a prescribed pressure jump with physical inertia in neural space.

Backward Euler advances momentum and the existing conservative water layer at
the same time. The mass is the reported initial bulk density times the nodal
volume; no mass scaling or material damping is introduced. Plasticity, crack
history, velocity and water are committed only after a converged coupled step.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import lstsq
import torch
from threadpoolctl import threadpool_limits

from fracture_channel import ChannelSystem
from fracture_physics import FractureProblem, cohesive_normal
from fracture_pinn import ConservativePINN, save_result


class DynamicPINN(ConservativePINN):
    def __init__(self, system, **settings):
        if system.control != 'mean_pressure':
            raise ValueError('Dynamic pressure loading requires mean_pressure control')
        super().__init__(system, **settings)
        nodes = system.free[self.mechanical]//3
        self.mass = system.b.density_kg_m3*system.mesh.weights.numpy()[nodes]
        self.displacement_old = system.state[self.mechanical].copy()*system.b.displacement_scale_m
        self.velocity_old = np.zeros(len(self.mechanical))

    def kinematics(self, state):
        dt = self.system.b.step_time_s
        displacement = state[self.mechanical]*self.system.b.displacement_scale_m
        velocity = (displacement-self.displacement_old)/dt
        acceleration = (velocity-self.velocity_old)/dt
        return velocity, acceleration

    def predict_displacement(self):
        """Fit the constant-velocity trial in the current neural output space."""
        net, s = self.network, self.system
        state, basis = net.evaluate(net.parameters, True)
        target = (self.displacement_old + s.b.step_time_s * self.velocity_old) / s.b.displacement_scale_m
        output = net.output_parameters
        net.parameters[output] += lstsq(basis[self.mechanical][:, output],
            target - state[self.mechanical], cond=1.e-9, lapack_driver='gelsy')[0]

    def forces(self, state, residual):
        return super().forces(state, residual)+self.mass*self.kinematics(state)[1]

    def tangent(self, parameters):
        force, derivative, tests = super().tangent(parameters)
        _, neural = self.network.evaluate(parameters, True)
        factor = self.mass*self.system.b.displacement_scale_m/self.system.b.step_time_s**2
        derivative += factor[:, None]*neural[self.mechanical]
        return force, derivative, tests

    def inspect_dynamics(self):
        s = self.system
        force, _, tests = self.tangent(self.network.parameters)
        velocity, acceleration = self.kinematics(s.state)
        inertial = self.mass*acceleration
        displacement_increment = s.state[self.mechanical]*s.b.displacement_scale_m-self.displacement_old
        kinetic = .5*np.dot(self.mass, velocity**2)
        old_kinetic = .5*np.dot(self.mass, self.velocity_old**2)
        dissipation = .5*np.dot(self.mass, (velocity-self.velocity_old)**2)
        work = np.dot(inertial, displacement_increment)
        velocity_field = np.zeros(3*s.n)
        acceleration_field = np.zeros(3*s.n)
        velocity_field[s.free[self.mechanical]] = velocity
        acceleration_field[s.free[self.mechanical]] = acceleration
        row = {
            'kinetic_energy_j_per_m': float(kinetic),
            'backward_euler_dissipation_j_per_m': float(dissipation),
            'inertial_work_j_per_m': float(work),
            'kinetic_identity_defect_j_per_m': float(work-(kinetic-old_kinetic)-dissipation),
            'max_velocity_m_s': float(np.abs(velocity).max()),
            'max_acceleration_m_s2': float(np.abs(acceleration).max()),
            'momentum_residual_max_n_per_m': float(np.abs(force).max()),
            'neural_momentum_work_max_n_per_m': float(np.max(np.abs(tests.T@force)/
                np.maximum(np.linalg.norm(tests, axis=0), 1.e-12))),
        }
        fields = {'velocity_m_s': velocity_field.reshape(s.n, 3)[:, :2],
                  'acceleration_m_s2': acceleration_field.reshape(s.n, 3)[:, :2]}
        return row, fields

    def accept(self, fields):
        s = self.system
        velocity = np.c_[fields['velocity_m_s'], np.zeros(s.n)].ravel()
        self.velocity_old = velocity[s.free[self.mechanical]].copy()
        self.displacement_old = s.state[self.mechanical].copy()*s.b.displacement_scale_m
        s.accept(fields)


def restore_dynamic_model(directory, checkpoint, dt):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    settings = result['settings']
    if settings.get('control') != 'mean_pressure':
        raise ValueError('The source must be an accepted mean-pressure path')
    index = int(Path(checkpoint).stem.rsplit('_', 1)[1])
    row = next(row for row in result['history'] if row['step'] == index)
    problem = replace(FractureProblem(**result['problem']), step_time_s=dt)
    system = ChannelSystem(problem, settings['spacing_m'], control='mean_pressure',
        normal_spacing=settings.get('normal_spacing_m'),
        tangent_spacing=settings.get('tangent_spacing_m'),
        interface_order=settings.get('interface_order', 2))
    system.target = row['face_mean_total_pressure_kpa']*1000
    model = DynamicPINN(system, center_spacing=settings['center_spacing_m'],
        learn_radii=settings['learn_radii'],
        trace_enrichment=settings.get('trace_enrichment', False),
        local_center_spacing=settings.get('local_center_spacing_m'),
        normal_center_spacing=settings.get('normal_center_spacing_m'))
    with np.load(directory/checkpoint) as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    system.state = fields['state'].copy()
    model.network.parameters = fields['parameters'].copy()
    if result['kind'] == 'dynamic_pressure_neural_fracture':
        model.accept(fields)
    else:
        # A held quasi-static checkpoint starts at rest. The displacement
        # accumulated during its static initialization is not a velocity.
        system.accept(fields)
        model.displacement_old = system.state[model.mechanical].copy()*problem.displacement_scale_m
    system.time_s = row['time_s']
    system.injection_old = row['injected_volume_ml_per_m']*1.e-6
    return model, result


def run_dynamic(directory, step, output, pressure_pa, duration_s, dt,
                max_iterations=1, resume=False, stop_surface_opening=False,
                predict_velocity=False):
    if not np.isfinite([pressure_pa, duration_s, dt]).all() or min(pressure_pa, duration_s, dt) <= 0:
        raise ValueError('Pressure, duration and time step must be positive')
    directory, output = Path(directory), Path(output)
    torch.set_num_threads(2)
    source = {'directory': str(directory.resolve()), 'checkpoint': f'fields_{step:03d}.npz'}
    if resume:
        result = json.loads((output/'result.json').read_text(encoding='utf-8'))
        if result['source'] != source or result['settings']['target_pressure_pa'] != pressure_pa or result['settings']['dt_s'] != dt or result['settings']['max_iterations'] != max_iterations:
            raise ValueError('Resume differs from saved source or evolution settings')
        if result['settings'].get('predict_velocity', False) != predict_velocity:
            raise ValueError('Resume differs from the saved displacement predictor')
        if stop_surface_opening and result.get('stop_reason') == 'surface_opening':
            return result
        model, _ = restore_dynamic_model(output, result['checkpoint'], dt)
    else:
        model, original = restore_dynamic_model(directory, source['checkpoint'], dt)
        output.mkdir(parents=True, exist_ok=False)
        settings = original['settings'].copy()
        settings.update(time_integrator='backward_euler', mass='initial_bulk_density_lumped',
            target_pressure_pa=pressure_pa, dt_s=dt, max_iterations=max_iterations,
            stop_surface_opening=stop_surface_opening)
        settings['predict_velocity'] = predict_velocity
        settings.pop('hold_time_s', None)
        settings.pop('pressure_step_pa', None)
        result = {'kind': 'dynamic_pressure_neural_fracture', 'source': source,
            'problem': original['problem'], 'settings': settings,
            'start_time_s': model.system.time_s,
            'initial_velocity': 'source_checkpoint' if original['kind'] == 'dynamic_pressure_neural_fracture' else 'rest',
            'history': [], 'status': 'running'}
    s, net = model.system, model.network
    finish = result['start_time_s']+duration_s
    advance = result.get('next_step_time_s',
        dt if not result['history'] else min(dt, 2*result['history'][-1]['step_time_s']))
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        while s.time_s < finish-1.e-12:
            s.target = pressure_pa
            s.b = replace(s.b, step_time_s=min(advance, finish-s.time_s))
            saved_state, saved_parameters = s.state.copy(), net.parameters.copy()
            try:
                if predict_velocity:
                    model.predict_displacement()
                training = model.train(max_iterations)
            except ArithmeticError as error:
                s.state, net.parameters = saved_state, saved_parameters
                advance = s.b.step_time_s/2
                result.update(status='retrying', error=str(error), next_step_time_s=advance)
                save_result(output, result)
                print(f'Retry t={s.time_s:.9g}, dt={advance:.6g}: {error}', flush=True)
                if s.time_s+advance == s.time_s:
                    raise ArithmeticError('Dynamic time increment reached floating-point resolution') from error
                continue
            dynamics, dynamic_fields = model.inspect_dynamics()
            row, fields = s.inspect()
            row.update(training)
            row.update(dynamics)
            fields.update(dynamic_fields)
            supplied = s.injection(s.state)
            index = len(result['history'])+1
            row.update(step=index, step_time_s=s.b.step_time_s,
                injected_volume_ml_per_m=(s.injection_old+supplied)*1.e6,
                injection_increment_ml_per_m=supplied*1.e6,
                target_mean_pressure_kpa=pressure_pa/1000)
            checkpoint = f'fields_{index:03d}.npz'
            np.savez_compressed(output/checkpoint, parameters=net.parameters, state=s.state, **fields)
            model.accept(fields)
            s.injection_old += supplied
            result['history'].append(row)
            result['checkpoint'] = checkpoint
            result.pop('error', None)
            result.pop('next_step_time_s', None)
            surface_open = False
            if stop_surface_opening:
                effective = torch.as_tensor(fields['crack_effective_endpoints'][-1, 1])
                maximum = torch.as_tensor(fields['crack_history_maximum'][-1])
                aperture = cohesive_normal(effective, torch.zeros_like(effective), maximum, s.b)[1]
                surface_open = bool(aperture > 0)
                if surface_open:
                    result['stop_reason'] = 'surface_opening'
            result.update(status='complete' if surface_open or s.time_s >= finish-1.e-12 else 'running',
                          elapsed_s=time.perf_counter()-start)
            save_result(output, result)
            print(json.dumps(row), flush=True)
            if surface_open:
                break
            advance = min(2*advance, dt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pressure-pa', type=float, required=True)
    parser.add_argument('--duration', type=float, required=True)
    parser.add_argument('--dt', type=float, required=True)
    parser.add_argument('--max-iterations', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-surface-opening', action='store_true')
    parser.add_argument('--predict-velocity', action='store_true',
                        help='Initialize output equilibrium with a constant-velocity neural field')
    args = parser.parse_args()
    run_dynamic(args.directory, args.step, args.output, args.pressure_pa,
                args.duration, args.dt, args.max_iterations, args.resume,
                args.stop_surface_opening, args.predict_velocity)


if __name__ == '__main__':
    main()
