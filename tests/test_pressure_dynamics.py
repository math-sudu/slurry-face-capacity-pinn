import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fracture_channel import ChannelSystem
from fracture_physics import FractureProblem
from fracture_pinn import run
from fracture_observations import path_observations
from pressure_dynamics import DynamicPINN, restore_dynamic_model, run_dynamic


class PressureDynamicsTests(unittest.TestCase):
    def test_inertia_tangent_includes_hidden_features_and_has_physical_mass(self):
        torch.set_num_threads(2)
        s = ChannelSystem(FractureProblem(step_time_s=.002), .12, control='mean_pressure')
        s.target = 1300.
        model = DynamicPINN(s, center_spacing=.18, trace_enrichment=True)
        net = model.network
        rng = np.random.default_rng(921)
        net.parameters[:] = rng.normal(0, .0001, net.nparameter)
        model.velocity_old[:] = rng.normal(0, .001, len(model.mechanical))
        with threadpool_limits(limits=2):
            force, derivative, tests = model.tangent(net.parameters)
            base_state = s.state.copy()
            pressure_tangent = model.pressure_tangent.copy()
            direction = rng.normal(size=net.nparameter)
            h = 1.e-8

            def forces(parameters):
                state = net.evaluate(parameters)
                state[model.hydraulic] = base_state[model.hydraulic]+pressure_tangent@(parameters-net.parameters)
                state, residual, _, _ = model.project(state)
                return model.forces(state, residual)

            finite = (forces(net.parameters+h*direction)-forces(net.parameters-h*direction))/(2*h)
            np.testing.assert_allclose(derivative@direction, finite, rtol=3.e-5, atol=.002)
            hidden = np.arange(net.outputs, net.outputs+net.hidden)
            radius_direction = np.zeros(net.nparameter)
            radius_direction[hidden] = direction[hidden]

            def work(parameters):
                _, jac = net.evaluate(parameters, True)
                return jac[model.mechanical][:, net.output_parameters].T@force

            finite = (work(net.parameters+h*radius_direction)-work(net.parameters-h*radius_direction))/(2*h)
            np.testing.assert_allclose(model.test_radius_derivative(net.parameters, force)@direction[hidden],
                                       finite, rtol=2.e-5, atol=2.e-5)
        self.assertAlmostEqual(s.mesh.weights.sum(), s.mesh.area.sum())
        nodes = s.free[model.mechanical]//3
        np.testing.assert_allclose(model.mass, s.b.density_kg_m3*s.mesh.weights.numpy()[nodes])

    def test_dynamic_trial_keeps_history_until_acceptance_and_balances_kinetic_work(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), threadpool_limits(limits=2):
            source = Path(directory)/'source'
            run(FractureProblem(step_time_s=.2), source, 1, .12, .18, .00002, .2,
                learn_radii=False, channel=True, pressure_step_pa=1100., hold_time_s=.4)
            model, _ = restore_dynamic_model(source, 'fields_002.npz', .002)
            s = model.system
            self.assertEqual(np.count_nonzero(model.velocity_old), 0)
            history = [s.plastic_old.clone(), s.water_old.clone(), s.maximum_old.clone()]
            old_displacement = model.displacement_old.copy()
            old_time = s.time_s
            s.target = 2200.
            model.train(0)
            dynamic_row, dynamic_fields = model.inspect_dynamics()
            row, fields = s.inspect()
            fields.update(dynamic_fields)
            self.assertEqual(s.time_s, old_time)
            for actual, previous in zip([s.plastic_old, s.water_old, s.maximum_old], history):
                np.testing.assert_array_equal(actual, previous)
            np.testing.assert_array_equal(model.displacement_old, old_displacement)
            self.assertGreater(dynamic_row['kinetic_energy_j_per_m'], 0.)
            self.assertLess(abs(dynamic_row['kinetic_identity_defect_j_per_m']), 1.e-12)
            self.assertLess(dynamic_row['neural_momentum_work_max_n_per_m'], .1)
            self.assertLess(abs(row['mass_defect_ml_per_m']), 1.e-5)
            model.accept(fields)
            self.assertAlmostEqual(s.time_s, old_time+.002)
            self.assertGreater(np.linalg.norm(model.velocity_old), 0.)

    def test_resume_preserves_velocity_and_coupled_history(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            source = root/'source'
            run(FractureProblem(step_time_s=.2, fracture_retention='capillary_bridge'), source, 1, .12, .18, .00002, .2,
                learn_radii=False, channel=True, pressure_step_pa=1100., hold_time_s=.4)
            kwargs = dict(directory=source, step=2, pressure_pa=2200., dt=.002, max_iterations=0)
            first = run_dynamic(output=root/'resumed', duration_s=.002, **kwargs)
            checkpoint = (root/'resumed'/first['checkpoint']).read_bytes()
            resumed = run_dynamic(output=root/'resumed', duration_s=.004, resume=True, **kwargs)
            self.assertEqual(resumed['problem']['fracture_retention'], 'capillary_bridge')
            self.assertEqual(resumed['settings']['fracture_retention'], 'capillary_bridge')
            whole = run_dynamic(output=root/'whole', duration_s=.004, **kwargs)
            self.assertEqual(checkpoint, (root/'resumed'/first['checkpoint']).read_bytes())
            with np.load(root/'resumed'/resumed['checkpoint']) as actual, np.load(root/'whole'/whole['checkpoint']) as expected:
                for key in ['u_m', 'velocity_m_s', 'bulk_plastic', 'bulk_water', 'crack_maximum', 'channel_pressure_pa']:
                    np.testing.assert_allclose(actual[key], expected[key], atol=1.e-10)
            rows = resumed['history']
            np.testing.assert_allclose([r['time_s'] for r in rows], [.402, .404])
            np.testing.assert_allclose([r['face_mean_total_pressure_kpa'] for r in rows], [2.2, 2.2])
            self.assertAlmostEqual(rows[-1]['injected_volume_ml_per_m'],
                                   whole['history'][-1]['injected_volume_ml_per_m'])
            observations = path_observations(root/'resumed')
            self.assertEqual(len(observations), 2)
            self.assertAlmostEqual(observations[-1]['bulk_tensile_plastic_volume_ml_per_m']+
                                   observations[-1]['bulk_shear_dilation_ml_per_m'],
                                   observations[-1]['bulk_plastic_volume_ml_per_m'], places=8)
            branch = run_dynamic(root/'whole', 1, root/'branch', 2200., .002, .002, max_iterations=0)
            self.assertEqual(branch['initial_velocity'], 'source_checkpoint')
            with np.load(root/'branch'/branch['checkpoint']) as actual, np.load(root/'whole'/whole['checkpoint']) as expected:
                np.testing.assert_allclose(actual['velocity_m_s'], expected['velocity_m_s'], atol=1.e-10)
            branch_observations = path_observations(root/'branch')
            for key in ['bulk_tensile_plastic_volume_ml_per_m', 'bulk_shear_dilation_ml_per_m',
                        'connected_open_tip_z_m', 'sensor_2_uplift_mm']:
                self.assertAlmostEqual(branch_observations[-1][key], observations[-1][key])


if __name__ == '__main__':
    unittest.main()
