import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from face_capacity_pinn import saturation
from fracture_physics import FractureProblem, fracture_water, cohesive_contact
from fracture_channel import ChannelSystem
from fracture_retention_diagnostic import install_traction_intervention


class CapillaryBridgeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.b = FractureProblem(fracture_retention='capillary_bridge',
            interface_geometry='corotated', bulk_kinematics='finite_strain')

    def test_closed_wall_matching_preserves_initial_total_traction(self):
        pressure = torch.tensor([-1.e7, -20000., self.b.pressure0, -1000., 0., 3000.])
        water, flow, covered = fracture_water(pressure, torch.zeros_like(pressure), self.b)
        np.testing.assert_allclose(covered, saturation(pressure, self.b), rtol=1.e-14)
        np.testing.assert_array_equal(water, 0)
        np.testing.assert_array_equal(flow, 0)
        s = ChannelSystem(self.b, .09, control='mean_pressure', interface_order=4)
        s.target = s.initial_face_mean
        force = s.residual(s.state)[:s.nfield][s.free % 3 != 2]
        self.assertLess(float(np.abs(force).max()), 1.e-8)
        with self.assertRaisesRegex(ValueError, 'pre-repair'):
            install_traction_intervention(s)

    def test_open_cavity_has_no_residual_floor_and_bounded_suction_traction(self):
        b = replace(self.b, water_compressibility_pa_inv=0)
        pressure = -torch.logspace(3, 12, 40)
        for width in [2.e-5, 2.e-4, 2.e-3]:
            water, _, covered = fracture_water(pressure, torch.full_like(pressure, width), b)
            self.assertTrue(bool((torch.diff(covered) < 0).all()))
            self.assertLess(float(covered[-1]), 1.e-7)
            self.assertTrue(bool((water <= width).all()))
            # Young-Laplace scale bounds the water-pressure resultant per wall
            # area, even as suction increases without bound at fixed open gap.
            self.assertLess(float((-covered*pressure).max()), 2*b.water_surface_tension_n_m/width)
        width = torch.logspace(-7, 0, 50)
        water, _, _ = fracture_water(torch.full_like(width, -1.e4), width, b)
        self.assertTrue(bool((torch.diff(water) > 0).all()))
        self.assertLess(float(water[-1]), 2*b.water_surface_tension_n_m/1.e4)

    def test_positive_pressure_retains_full_slit_storage_and_flow(self):
        width = torch.tensor([0., 2.e-5, 2.e-4])
        pressure = torch.tensor([0., 3000., 2.e4])
        water, flow, covered = fracture_water(pressure, width, self.b)
        np.testing.assert_array_equal(covered, 1)
        np.testing.assert_allclose(water, width*(1+self.b.water_compressibility_pa_inv*pressure))
        np.testing.assert_allclose(flow, width**3/(12*self.b.viscosity_pa_s))

    def test_isolated_opening_receives_only_its_conserved_wall_supply(self):
        b, dt = self.b, .000125
        widths = [2.e-5, 2.e-4]
        pressures = []
        for width in widths:
            def balance(p):
                water = float(fracture_water(torch.tensor(p), torch.tensor(width), b)[0])
                return water+dt*2*(p-b.pressure0)/b.fracture_wall_resistance_pa_s_m
            pressure = brentq(balance, -1.e8, b.pressure0)
            water, _, covered = fracture_water(torch.tensor(pressure), torch.tensor(width), b)
            self.assertAlmostEqual(float(water), dt*2*(b.pressure0-pressure)/b.fracture_wall_resistance_pa_s_m, places=16)
            self.assertLess(float(covered), b.residual_saturation)
            pressures.append(pressure)
        # A tenfold gap increase no longer demands tenfold residual inventory.
        self.assertLess(abs(pressures[1]/pressures[0]), 2)

    def test_storage_and_traction_share_area_and_consistent_joint_derivative(self):
        s = ChannelSystem(self.b, .09, control='mean_pressure', interface_order=4)
        rng = np.random.default_rng(912)
        state = rng.normal(0, .08, s.ndof)
        state[s.pf_indices] = (-2.e5-self.b.pressure0)/self.b.pressure_scale_pa
        for node in s.mesh.crack_plus:
            for component in range(2):
                state[s.lookup[3*node+component]] += .06*float(s.mesh.crack_frame[0, 0, component])
        detail = s.channel(torch.tensor(state[s.active]), True)
        cohesive = cohesive_contact(detail['jump'], s.traction0, s.maximum_old, s.slip_old, s.b)[0]
        stretch = detail['current_length']/s.mesh.crack_length
        covered = detail['saturation']
        np.testing.assert_allclose(detail['water'], stretch[:, None]*detail['aperture']*covered*
                                   (1+self.b.water_compressibility_pa_inv*detail['pressure']))
        np.testing.assert_allclose(detail['traction'][..., 0], cohesive[..., 0]-
                                   stretch[:, None]*covered*detail['pressure'])
        direction = rng.normal(size=s.ndof)
        _, tangent = s.residual(state, True)
        h = 1.e-7
        finite = (s.residual(state+h*direction)-s.residual(state-h*direction))/(2*h)
        np.testing.assert_allclose(tangent@direction, finite, rtol=3.e-4, atol=2.e-5)


if __name__ == '__main__':
    unittest.main()
