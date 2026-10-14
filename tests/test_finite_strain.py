import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from finite_strain import material_response, symmetric_log, swept_face_increment
from fracture_physics import FractureProblem
from fracture_solver import FractureSystem
from fracture_channel import ChannelSystem
from fracture_pinn import run
from fracture_observations import path_observations
from pressure_dynamics import run_dynamic


class FiniteStrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.b = FractureProblem(interface_geometry='corotated', bulk_kinematics='finite_strain')

    def response(self, deformation, old=None, b=None):
        count = len(deformation)
        if old is None:
            old = torch.eye(3).expand(count, 3, 3).clone()
        return material_response(deformation, old, torch.full((count,), 2.e6),
                                 torch.full((count,), 1.e6), torch.zeros(count), b or self.b)

    def test_spectral_log_derivative_at_repeated_stretches(self):
        matrix = torch.stack([torch.eye(3), torch.diag(torch.tensor([.8, .8, 1.3]))]).requires_grad_()
        rng = np.random.default_rng(119)
        direction = torch.tensor(rng.normal(size=matrix.shape))
        direction = (direction+direction.transpose(-1, -2))/2
        weight = torch.tensor(rng.normal(size=matrix.shape))
        derivative = torch.autograd.grad((symmetric_log(matrix)*weight).sum(), matrix)[0]
        h = 1.e-6
        finite = ((symmetric_log(matrix+h*direction)-symmetric_log(matrix-h*direction))*weight).sum()/(2*h)
        self.assertAlmostEqual(float((derivative*direction).sum()), float(finite.detach()), places=8)

    def test_material_objectivity_plastic_volume_and_unloading(self):
        deformation = torch.tensor([[[1.2, .06, 0.], [0., .94, 0.], [0., 0., 1.]]])
        original = self.response(deformation)
        angle = .71
        rotation = torch.tensor([[np.cos(angle), -np.sin(angle), 0.],
                                 [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        turned = self.response(rotation@deformation)
        np.testing.assert_allclose(turned['stress'], rotation@original['stress']@rotation.T, atol=1.e-8)
        np.testing.assert_allclose(turned['elastic_reference_metric'], original['elastic_reference_metric'], atol=1.e-12)
        trace = torch.diagonal(original['plastic_log_increment'], dim1=-2, dim2=-1).sum(-1)
        np.testing.assert_allclose(torch.log(original['plastic_J']), trace, atol=1.e-13)
        npsi = (1+np.sin(np.deg2rad(self.b.dilation_deg)))/(1-np.sin(np.deg2rad(self.b.dilation_deg)))
        np.testing.assert_allclose(trace, original['tensile_multiplier']+(npsi-1)*original['shear_multiplier'], atol=1.e-13)
        repeated = self.response(rotation@deformation, original['elastic_reference_metric'])
        np.testing.assert_allclose(repeated['stress'], turned['stress'], atol=5.e-8)
        np.testing.assert_allclose(repeated['elastic_reference_metric'], original['elastic_reference_metric'], atol=1.e-12)
        self.assertLess(float(repeated['plastic_multiplier'].max()), 1.e-12)
        # An isotropic elastic unloading increment leaves the plastic metric fixed.
        unloaded = self.response(.999999*deformation, original['elastic_reference_metric'])
        np.testing.assert_allclose(unloaded['elastic_reference_metric'], original['elastic_reference_metric'], atol=1.e-12)
        self.assertGreater(float((original['kirchhoff_stress']*original['plastic_log_increment']).sum()), 0.)

    def test_cauchy_strength_and_piola_tangent_include_volume_change(self):
        deformation = torch.tensor([[[1.2, .06, 0.], [0., .94, 0.], [0., 0., 1.]],
                                    [[.96, .07, 0.], [0., .95, 0.], [0., 0., 1.]],
                                    [[1.00001, 0., 0.], [0., 1.00001, 0.], [0., 0., 1.00001]]], requires_grad=True)
        result = self.response(deformation)
        eigen = torch.linalg.eigvalsh(result['stress']).detach().numpy()
        nphi = (1+np.sin(np.deg2rad(self.b.phi_deg)))/(1-np.sin(np.deg2rad(self.b.phi_deg)))
        self.assertLess(float((-eigen[:, 0]+nphi*eigen[:, -1]-2000*np.sqrt(nphi)).max()), 1.e-7)
        self.assertLess(float(eigen[:, -1].max()), 100.+1.e-7)
        rng = np.random.default_rng(442)
        direction = torch.tensor(rng.normal(size=deformation.shape))
        weight = torch.tensor(rng.normal(size=deformation.shape))

        def piola(f):
            return self.response(f)['kirchhoff_stress']@torch.linalg.inv(f).transpose(-1, -2)

        derivative = torch.autograd.grad((piola(deformation)*weight).sum(), deformation)[0]
        h = 1.e-8
        finite = ((piola(deformation+h*direction)-piola(deformation-h*direction))*weight).sum()/(2*h)
        self.assertAlmostEqual(float(((derivative*direction).sum()/finite).detach()), 1., places=6)
        rate = direction@torch.linalg.inv(deformation.detach())
        np.testing.assert_allclose((piola(deformation)*direction).sum((-2, -1)).detach(),
                                  (result['kirchhoff_stress']*rate).sum((-2, -1)).detach(), rtol=1.e-12)

    def test_elastic_energy_is_conjugate_to_first_piola(self):
        b = replace(self.b, effective_cohesion_pa=1.e9, tensile_strength_pa=1.e9)
        deformation = torch.tensor([[[1.14, .12, 0.], [0., .96, 0.], [0., 0., 1.]]], requires_grad=True)
        result = self.response(deformation, b=b)
        derivative = torch.autograd.grad(result['elastic_energy_density'].sum(), deformation)[0]
        piola = result['kirchhoff_stress']@torch.linalg.inv(deformation).transpose(-1, -2)
        np.testing.assert_allclose(derivative, piola.detach(), rtol=1.e-11, atol=1.e-8)

    def test_initial_stress_and_coupled_tangents(self):
        rng = np.random.default_rng(483)
        for system in [FractureSystem, ChannelSystem]:
            with self.subTest(system=system.__name__):
                s = system(self.b, .09, control='mean_pressure', interface_order=4)
                s.target = s.initial_face_mean
                mechanical = s.residual(s.state)[:s.nfield][s.free % 3 != 2]
                self.assertLess(float(np.abs(mechanical).max()), 1.e-8)
                state = rng.normal(0, .5, s.ndof)
                direction = rng.normal(size=s.ndof)
                _, derivative = s.residual(state, True)
                h = 1.e-6
                finite = (s.residual(state+h*direction)-s.residual(state-h*direction))/(2*h)
                np.testing.assert_allclose(derivative@direction, finite, rtol=2.e-4, atol=2.e-5)
                _, supply_derivative = s.injection(state, True)
                finite_supply = (s.injection(state+h*direction)-s.injection(state-h*direction))/(2*h)
                self.assertAlmostEqual(float(supply_derivative@direction), finite_supply, places=8)

    def test_current_inventory_darcy_metric_and_hydrostatic_rest(self):
        s = FractureSystem(self.b, .1)
        values = torch.zeros((len(s.mesh.tri), 3, 3))
        xy = s.mesh.xy[s.mesh.tri]
        deformation = torch.tensor([[1.12, .18], [-.07, .93]])
        position = xy@deformation.T
        values[:, :, :2] = (position-xy)/self.b.displacement_scale_m
        pressure = -2000.-1000*self.b.gravity_m_s2*position[:, :, 1]
        values[:, :, 2] = (pressure-self.b.pressure0)/self.b.pressure_scale_pa
        detail = s.bulk(values, True)
        np.testing.assert_allclose(detail['J'], torch.linalg.det(deformation), atol=1.e-13)
        residual = s.bulk(values).reshape(-1, 3, 3)[:, :, 2]
        storage = s.mesh.area[:, None]/3*(detail['water']-s.water_old)
        np.testing.assert_allclose(residual, storage, atol=1.e-13)
        # Constant pressure isolates gravity, testing the pulled-back flux.
        values[:, :, 2] = (-2000.-self.b.pressure0)/self.b.pressure_scale_pa
        detail = s.bulk(values, True)
        residual = s.bulk(values).reshape(-1, 3, 3)[:, :, 2]
        from face_capacity_pinn import pressure_relative_permeability
        j = torch.linalg.det(deformation)
        porosity = 1-(1-self.b.porosity0)/j
        mobility = self.b.permeability_m2/self.b.viscosity_pa_s*(porosity/self.b.porosity0)**3*((1-self.b.porosity0)/(1-porosity))**2
        mobility *= pressure_relative_permeability(torch.tensor(-2000.), self.b)
        reference_flux = j*torch.linalg.inv(deformation)@torch.tensor([0., 1000*self.b.gravity_m_s2])*mobility
        expected_flow = self.b.step_time_s*s.mesh.area[:, None]*torch.einsum('tni,i->tn', s.mesh.grad, reference_flux)
        storage = s.mesh.area[:, None]/3*(detail['water']-s.water_old)
        np.testing.assert_allclose(residual-storage, expected_flow, rtol=1.e-10, atol=1.e-13)

    def test_swept_volume_pressure_work_and_current_mean(self):
        s = FractureSystem(self.b, .05, control='mean_pressure')
        values = torch.zeros((len(s.face_nodes), 3))
        values[:, 0] = torch.linspace(.2, 5., len(values))
        values[:, 1] = torch.linspace(-.5, 2., len(values))
        face = s.face_boundary(values, 3000.)
        pressure = face['pressure'][s.face_edges_local]
        mean = (face['length']*pressure.mean(1)).sum()/face['length'].sum()
        self.assertAlmostEqual(float(mean), s.initial_face_mean+3000., places=10)
        # A linear path split into substeps gives the same swept volume.
        old = s.face_position_old
        middle = (old+face['position'])/2
        split = swept_face_increment(middle, old, s.face_edges_local)+swept_face_increment(face['position'], middle, s.face_edges_local)
        self.assertAlmostEqual(float(split), float(face['swept_volume']), places=14)
        # At zero increment, uniform follower pressure is conjugate to sweep.
        s.face_position_old = face['position'].clone()
        s.b = replace(self.b, gravity_m_s2=0.)
        variable = values.clone().requires_grad_()
        equal = s.face_boundary(variable, 3000.)
        derivative = torch.autograd.grad(equal['swept_volume'], variable)[0][:, :2]/self.b.displacement_scale_m
        np.testing.assert_allclose(equal['residual'][:, :2].detach(),
                                  -(s.initial_face_mean+3000.)*derivative, rtol=1.e-12, atol=1.e-12)

    def test_saved_history_rejects_old_kinematics_and_reloads_finite_metrics(self):
        revised = ChannelSystem(self.b, .1)
        legacy = ChannelSystem(replace(self.b, bulk_kinematics='small_strain'), .1)
        with self.assertRaisesRegex(ValueError, 'restart from the initial state'):
            revised.accept(legacy.inspect()[1])
        rng = np.random.default_rng(887)
        revised.state = rng.normal(0, .3, revised.ndof)
        fields = revised.inspect()[1]
        revised.accept(fields)
        restored = ChannelSystem(self.b, .1)
        restored.accept(fields)
        restored.state = revised.state.copy()
        before, after = revised.inspect()[1], restored.inspect()[1]
        for key in ['bulk_elastic_reference_metric', 'bulk_plastic_J', 'bulk_water', 'bulk_stress', 'face_swept_volume_m2']:
            np.testing.assert_array_equal(before[key], after[key])
        np.testing.assert_allclose(before['bulk_elastic_reference_metric'], fields['bulk_elastic_reference_metric'], atol=1.e-12)

    def test_finite_dynamic_resume_water_and_plastic_history(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), threadpool_limits(limits=2):
            root = Path(directory)
            source = root/'source'
            run(replace(self.b, step_time_s=.2), source, 1, .12, .18, .00002, .2,
                learn_radii=False, channel=True, pressure_step_pa=1100., hold_time_s=.4)
            kwargs = dict(directory=source, step=2, pressure_pa=3000., dt=.002, max_iterations=0)
            first = run_dynamic(output=root/'resumed', duration_s=.002, **kwargs)
            resumed = run_dynamic(output=root/'resumed', duration_s=.004, resume=True, **kwargs)
            whole = run_dynamic(output=root/'whole', duration_s=.004, **kwargs)
            with np.load(root/'resumed'/resumed['checkpoint']) as actual, np.load(root/'whole'/whole['checkpoint']) as expected:
                for key in ['u_m', 'velocity_m_s', 'bulk_elastic_reference_metric', 'bulk_plastic_J', 'bulk_water', 'face_swept_volume_m2']:
                    np.testing.assert_allclose(actual[key], expected[key], atol=1.e-10)
            for row in resumed['history']:
                self.assertLess(abs(row['mass_defect_ml_per_m']), 1.e-5)
                self.assertLess(abs(row['kinetic_identity_defect_j_per_m']), 1.e-12)
            observation = path_observations(root/'resumed')[-1]
            self.assertAlmostEqual(observation['bulk_tensile_plastic_volume_ml_per_m']+observation['bulk_shear_dilation_ml_per_m'],
                                   observation['bulk_plastic_volume_ml_per_m'], places=7)


if __name__ == '__main__':
    unittest.main()
