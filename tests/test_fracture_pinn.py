import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from scipy.linalg import svd
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fracture_physics import FractureProblem
from fracture_solver import FractureSystem
from fracture_channel import ChannelSystem
from fracture_pinn import ConservativePINN, run


class NeuralFractureTests(unittest.TestCase):
    def test_feature_learning_tries_a_small_sensitivity_direction(self):
        class LinearResidualPINN(ConservativePINN):
            def __init__(self):
                self.network = SimpleNamespace(
                    parameters=np.zeros(2), outputs=1, hidden=1,
                    learn_radii=True, output_parameters=np.array([0]),
                    evaluate=lambda parameters: parameters.copy())
                self.system = SimpleNamespace(state=np.zeros(2))
                self.force_scale = np.ones(2)
                self.hydraulic = np.array([0])

            def equilibrate(self, parameters):
                parameters = parameters.copy()
                parameters[0] = 0.
                force = np.array([0., 1.+1.e-5*parameters[1]])
                derivative = np.diag([1., 1.e-5])
                tests = np.array([[1.], [0.]])
                return parameters, (force, derivative, tests, np.ones((1, 1))), 1

            def test_radius_derivative(self, parameters, force):
                return np.zeros((1, 1))

            def feature_predictor(self, parameters, hidden_change, output_change, fraction):
                return parameters+fraction*np.r_[output_change, hidden_change]

            def project(self, state, tangent=True):
                return state, np.zeros(2), None, None

        model = LinearResidualPINN()
        result = model.train(1)
        self.assertEqual(result['feature_updates'], 1)
        self.assertLess(result['neural_loss'], result['force_loss_before_feature_learning'])
        self.assertAlmostEqual(model.network.parameters[1], -.2)

    def test_prescribed_mean_pressure_tangent_and_response_supply(self):
        torch.set_num_threads(2)
        s = ChannelSystem(FractureProblem(), .06, control='mean_pressure')
        s.target = 1600.
        model = ConservativePINN(s, .1, trace_enrichment=True)
        net = model.network
        net.parameters[-1] = .02
        state, jac = net.evaluate(net.parameters, True)
        self.assertAlmostEqual(s.residual(state)[-1], 0., places=12)
        self.assertEqual(np.count_nonzero(jac[-1]), 0)
        self.assertAlmostEqual(s.unpack(state)[-1]+s.initial_face_mean, s.target)
        direction = np.random.default_rng(124).normal(size=net.nparameter)
        h = 1.e-7
        finite = (net.evaluate(net.parameters+h*direction)-net.evaluate(net.parameters-h*direction))/(2*h)
        np.testing.assert_allclose(jac@direction, finite, rtol=2.e-6, atol=1.e-8)
        with threadpool_limits(limits=2):
            force, derivative, tests = model.tangent(net.parameters)
            np.testing.assert_allclose(tests, jac[model.mechanical][:, net.output_parameters])
            def forces(parameters):
                trial, residual, _, _ = model.project(net.evaluate(parameters))
                return model.forces(trial, residual)
            finite = (forces(net.parameters+h*direction)-forces(net.parameters-h*direction))/(2*h)
            np.testing.assert_allclose(derivative@direction, finite, rtol=2.e-4, atol=2.e-4)
            hidden = np.arange(net.outputs, net.outputs+net.hidden)
            radius_direction = np.zeros(net.nparameter)
            radius_direction[hidden] = direction[hidden]
            def work(parameters):
                _, trial_jac = net.evaluate(parameters, True)
                return trial_jac[model.mechanical][:, net.output_parameters].T@force
            finite = (work(net.parameters+h*radius_direction)-work(net.parameters-h*radius_direction))/(2*h)
            np.testing.assert_allclose(model.test_radius_derivative(net.parameters, force)@direction[hidden],
                                       finite, rtol=2.e-5, atol=2.e-6)
            model.train(1)
        row, _ = s.inspect()
        self.assertAlmostEqual(row['face_mean_total_pressure_kpa'], 1.6)
        self.assertLess(abs(row['mass_defect_ml_per_m']), 1.e-5)
        supplied = (row['face_swept_volume_ml_per_m']+s.b.step_time_s*row['inflow_ml_per_m_s'])*1.e-6
        self.assertAlmostEqual(s.injection(s.state), supplied, places=12)

    def test_pressure_plateau_resume_preserves_history_and_supply(self):
        b = FractureProblem(step_time_s=.2)
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory)
            settings = dict(b=b, output=output, spacing=.12, center_spacing=.18,
                            increment=.00002, dt=.2, learn_radii=False,
                            channel=True, pressure_step_pa=1100., hold_time_s=.4)
            first = run(steps=1, **settings)
            checkpoint = (output/first['checkpoint']).read_bytes()
            resumed = run(steps=2, resume=True, **settings)
            self.assertEqual(checkpoint, (output/first['checkpoint']).read_bytes())
        rows = resumed['history']
        self.assertEqual(resumed['status'], 'complete')
        np.testing.assert_allclose([r['time_s'] for r in rows], [.2, .4, .6, .8])
        np.testing.assert_allclose([r['face_mean_total_pressure_kpa'] for r in rows], [1.1, 1.1, 2.2, 2.2])
        self.assertAlmostEqual(rows[-1]['injected_volume_ml_per_m'], sum(r['injection_increment_ml_per_m'] for r in rows))
        inventory = sum(r['bulk_water_change_ml_per_m']+r['fracture_water_change_ml_per_m']
                        +r['step_time_s']*r['outflow_ml_per_m_s'] for r in rows)
        self.assertAlmostEqual(rows[-1]['injected_volume_ml_per_m'],
                               inventory+rows[-1]['face_swept_volume_ml_per_m'], places=6)

    def test_virtual_work_derivative_in_nearly_dependent_retained_directions(self):
        torch.set_num_threads(2)
        s = ChannelSystem(FractureProblem(), .03, normal_spacing=.002, tangent_spacing=.01, interface_order=8)
        model = ConservativePINN(s, .075, trace_enrichment=True,
                                 local_center_spacing=.02, normal_center_spacing=.003)
        net = model.network
        output = net.output_parameters
        with threadpool_limits(limits=2):
            _, derivative, tests = model.tangent(net.parameters)
            _, singular, right = svd(tests[:, :-1], full_matrices=False)
            retained = np.flatnonzero(singular > singular[0]*1.e-9)[-2:]
            transform = np.zeros((len(output), 2))
            transform[:-1] = right[retained].T/singular[retained]
            active_tests = tests@transform
            direction = np.array([.4, -.7])
            change = transform@direction
            exact = model.virtual_work_tangent(tests, derivative, transform)@direction

            def work(fraction):
                parameters = net.parameters.copy()
                parameters[output] += fraction*change
                state, residual, _, _ = model.project(net.evaluate(parameters), tangent=False)
                return active_tests.T@model.forces(state, residual)

            h = 1.e-5
            finite = (work(h)-work(-h))/(2*h)
        np.testing.assert_allclose(exact, finite, rtol=1.e-6, atol=1.e-5)

    def test_feature_predictor_preserves_linearized_neural_field(self):
        s = ChannelSystem(FractureProblem(), .06)
        s.target = .00003
        model = ConservativePINN(s, .1, trace_enrichment=True)
        net = model.network
        rng = np.random.default_rng(723)
        parameters = rng.normal(0, .01, net.nparameter)
        hidden_change = rng.normal(0, .1, net.hidden)
        output_change = rng.normal(0, .01, len(net.output_parameters))
        state, jac = net.evaluate(parameters, True)
        hidden = np.arange(net.outputs, net.outputs+net.hidden)
        direction = jac[:, hidden]@hidden_change+jac[:, net.output_parameters]@output_change
        errors = []
        for fraction in [.01, .005]:
            trial = model.feature_predictor(parameters, hidden_change, output_change, fraction)
            error = net.evaluate(trial)[model.mechanical]-(state+fraction*direction)[model.mechanical]
            errors.append(np.linalg.norm(error))
        self.assertLess(errors[1], .3*errors[0])

    def test_zero_feature_step_preserves_existing_weak_neural_modes(self):
        s = ChannelSystem(FractureProblem(), .03, control='mean_pressure',
                          normal_spacing=.002, tangent_spacing=.01, interface_order=8)
        s.target = 5000.
        model = ConservativePINN(s, .075, trace_enrichment=True,
                                 local_center_spacing=.02, normal_center_spacing=.003)
        net = model.network
        parameters = net.parameters.copy()
        with threadpool_limits(limits=2):
            _, jacobian = net.evaluate(parameters, True)
            basis = jacobian[model.mechanical][:, net.output_parameters]
            _, singular, right = svd(basis, full_matrices=False)
            weak = np.flatnonzero((singular < singular[0]*1.e-9)
                                 & (singular > singular[0]*1.e-11))[0]
            parameters[net.output_parameters] = .01*right[weak]/singular[weak]
            before = net.evaluate(parameters)[model.mechanical]
            self.assertGreater(np.linalg.norm(before), .009)
            trial = model.feature_predictor(parameters, np.zeros(net.hidden),
                                            np.zeros(len(net.output_parameters)), 0.)
            after = net.evaluate(trial)[model.mechanical]
        np.testing.assert_array_equal(after, before)

    def test_local_traces_balance_crack_and_face_virtual_work(self):
        torch.set_num_threads(2)
        s = ChannelSystem(FractureProblem(), .06, normal_spacing=.015, tangent_spacing=.03)
        s.target = .00002
        model = ConservativePINN(s, .1, trace_enrichment=True, local_center_spacing=.06, normal_center_spacing=.02)
        net = model.network
        direction = np.random.default_rng(718).normal(size=net.nparameter)
        state, jac = net.evaluate(net.parameters, True)
        h = 1.e-7
        finite = (net.evaluate(net.parameters+h*direction)-net.evaluate(net.parameters-h*direction))/(2*h)
        np.testing.assert_allclose(jac@direction, finite, rtol=2.e-6, atol=1.e-8)
        with threadpool_limits(limits=2):
            model.train(1)
        force = s.residual(s.state)[:s.nfield]*s.scale
        ids = s.lookup[(3*net.trace_nodes[:, None]+np.arange(2)).ravel()]
        self.assertLess(np.max(np.abs(force[ids[ids >= 0]])), .11)
        row, _ = s.inspect()
        self.assertLess(abs(row['mass_defect_ml_per_m']), 1.e-5)

    def test_volume_control_implicit_tangent_and_supply_budget(self):
        torch.set_num_threads(2)
        s = ChannelSystem(FractureProblem(), .06)
        s.target = 30.e-6
        model = ConservativePINN(s, .1, volume_control=True)
        net = model.network
        net.parameters[-2:] = [.03, .1]
        with threadpool_limits(limits=2):
            _, derivative, _ = model.tangent(net.parameters)
            direction = np.random.default_rng(915).normal(size=net.nparameter)
            h = 1.e-7
            def force(parameters):
                state, residual, _, _ = model.project(net.evaluate(parameters))
                return model.forces(state, residual)
            finite = (force(net.parameters+h*direction)-force(net.parameters-h*direction))/(2*h)
            np.testing.assert_allclose(derivative@direction, finite, rtol=2.e-4, atol=2.e-4)
            model.train(1)
        row, _ = s.inspect()
        self.assertLess(abs(row['mass_defect_ml_per_m']), 1.e-5)
        self.assertLess(abs(s.injection(s.state)-s.target), 1.e-8)
        physical_supply = (row['face_swept_volume_ml_per_m']+s.b.step_time_s*row['inflow_ml_per_m_s'])*1.e-6
        self.assertAlmostEqual(s.injection(s.state), physical_supply, places=12)

    def test_hidden_derivative_and_exact_displacement_control(self):
        torch.set_num_threads(2)
        s = FractureSystem(FractureProblem(), .05)
        model = ConservativePINN(s, .09)
        net = model.network
        s.target = .00002
        rng = np.random.default_rng(92)
        parameters = rng.normal(0, .001, net.nparameter)
        direction = rng.normal(size=net.nparameter)
        state, jac = net.evaluate(parameters, True)
        h = 1.e-7
        fd = (net.evaluate(parameters+h*direction)-net.evaluate(parameters-h*direction))/(2*h)
        np.testing.assert_allclose(jac@direction, fd, rtol=2.e-6, atol=1.e-8)
        self.assertAlmostEqual(s.residual(state)[-1], 0., places=12)
        self.assertLess(net.nparameter, len(model.mechanical))

    def test_hydraulic_projection_and_implicit_neural_tangent(self):
        s = FractureSystem(FractureProblem(), .06)
        model = ConservativePINN(s, .1)
        net = model.network
        s.target = .00002
        state, network_jac = net.evaluate(net.parameters, True)
        state, residual, jac, factor = model.project(state)
        s.state = state
        row, _ = s.inspect()
        self.assertLess(abs(row['mass_defect_ml_per_m']), 1.e-5)
        rng = np.random.default_rng(7)
        direction = rng.normal(size=net.nparameter)
        partial = jac@network_jac@direction
        dh = -factor.solve(partial[model.hydraulic])
        expected = partial[model.mechanical]+jac[model.mechanical][:, model.hydraulic]@dh
        h = 1.e-7
        plus = model.project(net.evaluate(net.parameters+h*direction))[1][model.mechanical]
        minus = model.project(net.evaluate(net.parameters-h*direction))[1][model.mechanical]
        np.testing.assert_allclose(expected, (plus-minus)/(2*h), rtol=2.e-4, atol=2.e-5)

    def test_feature_derivative_preserves_the_virtual_work_test(self):
        s = FractureSystem(FractureProblem(), .06)
        model = ConservativePINN(s, .1)
        net = model.network
        rng = np.random.default_rng(84)
        force = rng.normal(size=len(model.mechanical))
        direction = np.zeros(net.nparameter)
        direction[net.outputs:net.outputs+net.hidden] = rng.normal(size=net.hidden)
        output = np.r_[np.arange(net.outputs), net.nparameter-1]
        def test_work(parameters):
            _, jac = net.evaluate(parameters, True)
            return jac[model.mechanical][:, output].T@force
        h = 1.e-6
        fd = (test_work(net.parameters+h*direction)-test_work(net.parameters-h*direction))/(2*h)
        exact = model.test_radius_derivative(net.parameters, force)@direction[net.outputs:net.outputs+net.hidden]
        np.testing.assert_allclose(exact, fd, rtol=1.e-6, atol=1.e-8)


if __name__ == '__main__':
    unittest.main()
