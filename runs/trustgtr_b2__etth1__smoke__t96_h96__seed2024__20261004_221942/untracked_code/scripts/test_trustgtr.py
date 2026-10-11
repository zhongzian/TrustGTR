#!/usr/bin/env python3
"""Small correctness checks before GPU smoke; no formal metrics here."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models.GTR import Model as Official
from models.TrustGTR import Model, ForecastLoss, utility_oracle


def args(options=None):
    return SimpleNamespace(seq_len=12, pred_len=6, enc_in=2, cycle=4,
                           d_model=16, dropout=0.5, use_revin=1, individual=0,
                           trustgtr=options or {})


class Checks(unittest.TestCase):
    def test_off_matches_original_parameters_forward_backward(self):
        torch.manual_seed(2024)
        original = Official(args())
        torch.manual_seed(2024)
        candidate = Model(args())
        self.assertEqual(set(original.state_dict()), set(candidate.state_dict()))
        x, cycle = torch.randn(3, 12, 2), torch.tensor([0, 1, 3])
        for training in (True, False):
            original.train(training); candidate.train(training)
            torch.manual_seed(17); baseline = original(x, cycle)
            torch.manual_seed(17); actual = candidate(x, cycle)
            self.assertTrue(torch.equal(baseline, actual))
        baseline.square().mean().backward(); actual.square().mean().backward()
        for left, right in zip(original.parameters(), candidate.parameters()):
            self.assertTrue(torch.equal(left.grad, right.grad))

    def test_oracle_known_optimum_and_bounds(self):
        local, delta = torch.randn(3, 6, 2), torch.randn(3, 6, 2)
        gate, valid = utility_oracle(local, delta, local + 0.25 * delta)
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(gate, torch.full_like(gate, 0.25), atol=1e-6))
        self.assertTrue(torch.equal(utility_oracle(local, delta, local - delta)[0], torch.zeros_like(gate)))

    def test_residual_initialization_preserves_baseline_rng_and_forecast(self):
        torch.manual_seed(2024)
        original = Official(args())
        original_rng = torch.get_rng_state().clone()
        torch.manual_seed(2024)
        candidate = Model(args(dict(use_multi_cycle=True, memory_mode='base_residual',
                                    preserve_initialization_rng=True)))
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        x, origins = torch.randn(3, 12, 2), torch.tensor([0, 1, 3])
        original.eval(); candidate.eval()
        self.assertTrue(torch.equal(original(x, origins), candidate(x, origins)))

    def test_all_modules_train_and_eval_without_target_access(self):
        options = dict(use_multi_cycle=True, use_reliability_gate=True, use_regime_contrast=True)
        model = Model(args(options))
        origins = torch.arange(8) * 24
        x, target = torch.randn(8, 12, 2), torch.randn(8, 6, 2)
        output = model(x, origins)
        loss = ForecastLoss(model)(output, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(model.diagnostics['contrast_valid_anchors'], 0)
        model.eval()
        prediction = model(x, origins)
        self.assertIsNone(model._retrieval_context)
        self.assertTrue(torch.equal(ForecastLoss(model)(prediction, target), torch.nn.functional.mse_loss(prediction, target)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
