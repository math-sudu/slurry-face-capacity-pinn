import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fracture_physics import FractureProblem, cohesive_normal
from fracture_transport import OpeningHistory


class ContinuousTransportTests(unittest.TestCase):
    def setUp(self):
        self.b = FractureProblem()
        self.history = OpeningHistory([.1], [.25, .75], np.zeros((1, 2)))

    def accept(self, effective):
        effective = np.asarray(effective)
        saved = self.history.saved(effective)
        saved['crack_effective_endpoints'] = effective
        self.history.accept(saved)
        return saved

    def test_linear_open_taper_has_exact_series_resistance(self):
        left, right = 2.e-4, 4.e-4
        trans, final = self.history.transmissibility(torch.tensor([[left, right]]), self.b)
        expected = left**2*right**2/(6*self.b.viscosity_pa_s*.1*(left+right))
        self.assertAlmostEqual(float(1/(1/trans).sum())/expected, 1., places=12)
        np.testing.assert_allclose(trans, final, rtol=1.e-14, atol=0.)

    def test_accepted_damage_at_both_ends_does_not_bridge_intact_middle(self):
        self.accept([[3.e-7, -1.e-6]])
        _, final = self.history.transmissibility(torch.tensor([[-1.e-6, 3.e-7]]), self.b)
        np.testing.assert_array_equal(final, 0.)
        saved = self.accept([[-1.e-6, 3.e-7]])
        effective = torch.full((1, 2), 5.e-8)
        self.accept(effective.numpy())
        trans, _ = self.history.transmissibility(effective, self.b)
        np.testing.assert_array_equal(trans, 0.)
        self.assertGreater(len(saved['crack_history_fraction']), 2)
        # Reopening above the cohesive onset connects the same history.
        trans, _ = self.history.transmissibility(torch.full((1, 2), 2.e-6), self.b)
        self.assertTrue(bool((trans > 0).all()))

    def test_uniform_closure_integrates_cubic_drainage_then_stops(self):
        effective = torch.full((1, 2), 1.e-5)
        old = cohesive_normal(effective, torch.zeros_like(effective), torch.zeros_like(effective), self.b)[1]
        self.accept(effective.numpy())
        closed = torch.full((1, 2), -1.e-8)
        trans, final = self.history.transmissibility(closed, self.b)
        expected = float(old[0, 0]**3)/(48*self.b.viscosity_pa_s*.1)
        self.assertAlmostEqual(float(1/(1/trans).sum())/expected, 1., places=12)
        np.testing.assert_array_equal(final, 0.)
        self.accept(closed.numpy())
        trans, _ = self.history.transmissibility(closed, self.b)
        np.testing.assert_array_equal(trans, 0.)

    def test_history_reload_preserves_unloading_transport_and_derivative(self):
        self.accept([[2.e-5, 4.e-6]])
        saved = self.accept([[3.e-6, 1.e-5]])
        restored = OpeningHistory([.1], [.25, .75], np.zeros((1, 2)))
        restored.accept(saved)
        effective = torch.tensor([[5.e-6, 3.e-6]], requires_grad=True)
        original = self.history.transmissibility(effective, self.b)[0]
        np.testing.assert_array_equal(original.detach(), restored.transmissibility(effective, self.b)[0].detach())
        direction = torch.tensor([[.3, -.7]])
        jac = torch.autograd.functional.jacobian(lambda e: restored.transmissibility(e, self.b)[0], effective)
        h = 1.e-10
        finite = (restored.transmissibility(effective+h*direction, self.b)[0]
                  -restored.transmissibility(effective-h*direction, self.b)[0])/(2*h)
        np.testing.assert_allclose(torch.einsum('eij,ij->e', jac, direction), finite.detach(), rtol=1.e-7, atol=1.e-17)

    def test_mixed_opening_closure_switch_is_continuous(self):
        self.accept([[2.e-4, 2.e-4]])
        # The left endpoint crosses from opening to closing while the rest of
        # the inlet link is opening. A link-wide time-rule switch would jump.
        before = self.history.transmissibility(torch.tensor([[2.e-4+1.e-12, 4.e-4]]), self.b)[0]
        after = self.history.transmissibility(torch.tensor([[2.e-4-1.e-12, 4.e-4]]), self.b)[0]
        np.testing.assert_allclose(before, after, rtol=1.e-7, atol=0.)


if __name__ == '__main__':
    unittest.main()
