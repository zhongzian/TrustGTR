#!/usr/bin/env python3
"""CPU algebra, gradient isolation and compatibility checks for gate risk."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from models.TrustGTR import Model, ForecastLoss, counterfactual_risk_loss, resolved_options, utility_oracle


def model(mode):
    return Model(SimpleNamespace(seq_len=12, pred_len=6, enc_in=2, cycle=4,
        d_model=16, dropout=0.5, use_revin=1, individual=0,
        trustgtr=dict(use_reliability_gate=True, gate_loss_mode=mode)))


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2024)

    def test_exact_normalized_mse_difference_and_gate_only_gradient(self):
        local = torch.randn(4, 6, 3, dtype=torch.float64, requires_grad=True)
        delta = torch.randn(4, 6, 3, dtype=torch.float64, requires_grad=True)
        target = torch.randn(4, 6, 3, dtype=torch.float64, requires_grad=True)
        gate = torch.rand(4, 1, 3, dtype=torch.float64, requires_grad=True)
        actual = counterfactual_risk_loss(local, delta, gate, target)
        error = (target - local).detach()
        correction = delta.detach()
        denominator = correction.square().sum(1, keepdim=True) + error.square().sum(1, keepdim=True)
        direct = ((error - gate * correction).square().sum(1, keepdim=True) -
                  error.square().sum(1, keepdim=True)) / denominator
        self.assertTrue(torch.allclose(actual, direct.mean(), atol=1e-12, rtol=1e-12))
        expected_gradient = torch.autograd.grad(direct.mean(), gate, retain_graph=True)[0]
        actual.backward()
        self.assertTrue(torch.allclose(gate.grad, expected_gradient, atol=1e-12, rtol=1e-12))
        self.assertIsNone(local.grad)
        self.assertIsNone(delta.grad)
        self.assertIsNone(target.grad)

    def test_expected_risk_optimum_differs_from_mean_clipped_oracles(self):
        local = torch.zeros(2, 1, 1, dtype=torch.float64)
        delta = torch.ones_like(local)
        target = torch.tensor([2., -2.], dtype=torch.float64).reshape_as(local)
        clipped, meaningful = utility_oracle(local, delta, target)
        self.assertTrue(meaningful.all())
        self.assertEqual(float(clipped.mean()), 0.5)
        # Both samples have the same P+E=5. Their joint objective is g^2/5,
        # so its optimum is 0, even though clipped per-sample labels average .5.
        grid = torch.linspace(0, 1, 101, dtype=torch.float64)
        losses = torch.stack([counterfactual_risk_loss(local, delta,
                  value.expand(2, 1, 1), target) for value in grid])
        self.assertEqual(int(losses.argmin()), 0)
        gate = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        loss = counterfactual_risk_loss(local, delta, gate.expand(2, 1, 1), target)
        loss.backward()
        self.assertAlmostEqual(float(gate.grad), 0.2)

    def test_perfect_zero_and_negative_corrections_are_finite_and_bounded(self):
        for error_value, correction_value in ((0., 1.), (1., 1.), (1., 0.), (1., -1.), (2., 1.), (-2., 1.)):
            local = torch.zeros(3, 6, 2)
            delta = torch.full_like(local, correction_value)
            target = torch.full_like(local, error_value)
            gate = torch.full((3, 1, 2), 0.4, requires_grad=True)
            loss = counterfactual_risk_loss(local, delta, gate, target)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(float(loss), -1.)
            self.assertLessEqual(float(loss), 2.)
            loss.backward()
            self.assertTrue(torch.isfinite(gate.grad).all())
            if correction_value == 0:
                self.assertEqual(float(loss), 0.)
                self.assertTrue(torch.equal(gate.grad, torch.zeros_like(gate)))
            if error_value == 0 or correction_value < 0:
                self.assertTrue((gate.grad > 0).all())

    def test_mixed_empty_channels_are_excluded(self):
        local = torch.zeros(1, 6, 2)
        delta = torch.ones_like(local)
        delta[..., 0] = 0
        target = torch.ones_like(local)
        gate = torch.full((1, 1, 2), 0.5, requires_grad=True)
        loss = counterfactual_risk_loss(local, delta, gate, target)
        self.assertAlmostEqual(float(loss), -0.375)
        loss.backward()
        self.assertEqual(float(gate.grad[..., 0]), 0.)

    def test_eval_disables_auxiliary_risk_and_forward_never_needs_target(self):
        candidate = model('counterfactual_risk')
        self.assertEqual(candidate.options['gate_loss_weight'], 0.05)
        x, origin = torch.randn(4, 12, 2), torch.tensor([0, 1, 2, 3])
        prediction = candidate(x, origin)
        target = torch.randn_like(prediction)
        auxiliary = candidate.auxiliary_losses(target)['gate']
        self.assertTrue(torch.isfinite(auxiliary))
        ForecastLoss(candidate)(prediction, target).backward()
        for parameter in candidate.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        candidate.eval()
        actual = candidate(x, origin)
        self.assertEqual(candidate.auxiliary_losses(target), {})
        self.assertIsNone(candidate._retrieval_context)
        self.assertTrue(torch.equal(ForecastLoss(candidate)(actual, target), F.mse_loss(actual, target)))

    def test_legacy_losses_and_gate_gradients_are_exact(self):
        local, delta, target = torch.randn(4, 6, 2), torch.randn(4, 6, 2), torch.randn(4, 6, 2)
        delta[0, :, 0] = 0
        for mode in ('uniform', 'power_weighted'):
            candidate = model(mode)
            gate = torch.rand(4, 1, 2, requires_grad=True)
            candidate._retrieval_context = dict(local=local, delta=delta, gate=gate)
            actual = candidate.auxiliary_losses(target)['gate']
            oracle, meaningful = utility_oracle(local, delta, target)
            if mode == 'uniform':
                expected = F.mse_loss(gate[meaningful], oracle[meaningful])
            else:
                power = delta.square().sum(1, keepdim=True)
                weights = (power / power.mean().clamp_min(1e-8)).clamp(max=10) * meaningful
                expected = (weights * (gate - oracle).square()).sum() / weights.sum().clamp_min(1e-8)
            self.assertTrue(torch.equal(actual, expected))
            actual_grad = torch.autograd.grad(actual, gate)[0]
            expected_grad = torch.autograd.grad(expected, gate)[0]
            self.assertTrue(torch.equal(actual_grad, expected_grad))
        self.assertEqual(resolved_options({}, 4)['gate_loss_mode'], 'uniform')
        self.assertEqual(resolved_options(dict(gate_loss_mode='counterfactual_risk'), 4)['gate_loss_mode'],
                         'counterfactual_risk')
        with self.assertRaises(ValueError):
            resolved_options(dict(gate_loss_mode='unknown'), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
