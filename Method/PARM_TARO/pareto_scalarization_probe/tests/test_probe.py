import unittest
import torch
from PARM_TARO.pareto_scalarization_probe.config import ALPHA_GRID
from PARM_TARO.pareto_scalarization_probe.geometry import nondominated, support_interval
from PARM_TARO.pareto_scalarization_probe.manifest import prompt_hash
from PARM_TARO.pareto_scalarization_probe.scalarization import augmented_tchebycheff, linear_scalarization

class ProbeTests(unittest.TestCase):
    def test_grid(self):
        self.assertEqual(len(ALPHA_GRID),21); self.assertEqual(ALPHA_GRID[0],(0.,1.)); self.assertEqual(ALPHA_GRID[-1],(1.,0.))
        self.assertTrue(all(abs(sum(x)-1)<1e-9 for x in ALPHA_GRID))
    def test_prompt_hash_stable(self): self.assertEqual(prompt_hash("x"),prompt_hash("x")); self.assertNotEqual(prompt_hash("x"),prompt_hash("y"))
    def test_pareto_and_support(self):
        points=[(0.,1.),(.5,.5),(1.,0.),(.2,.2)]; self.assertEqual(nondominated(points),[True,True,True,False])
        self.assertIsNotNone(support_interval(1,points)); self.assertIsNone(support_interval(3,points))
    def test_linear(self):
        self.assertAlmostEqual(float(linear_scalarization(torch.tensor([2.,4.]),torch.tensor([.25,.75]))),3.5)
    def test_tchebycheff_finite_and_differentiable(self):
        losses=torch.tensor([2.,4.],requires_grad=True); value=augmented_tchebycheff(losses,torch.tensor([.5,.5]),reference=torch.zeros(2),scales=torch.ones(2))
        value.backward(); self.assertTrue(torch.isfinite(value)); self.assertTrue(torch.isfinite(losses.grad).all())
    def test_bad_alpha_rejected(self):
        with self.assertRaises(ValueError): linear_scalarization(torch.ones(2),torch.tensor([.8,.8]))
