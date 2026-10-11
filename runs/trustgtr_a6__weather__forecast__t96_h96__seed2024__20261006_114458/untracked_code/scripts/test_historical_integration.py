#!/usr/bin/env python3
"""Scientific invariants of the causal retrieval branch before GPU smoke."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.GTR import Model as Official
from models.TrustGTR import Model, ForecastLoss


def args(gate=False, contrast=False, strength_coordinates='raw'):
    return SimpleNamespace(seq_len=12, pred_len=6, enc_in=2, cycle=4,
        d_model=16, dropout=0.5, use_revin=1, individual=0,
        trustgtr=dict(use_multi_cycle=True, use_reliability_gate=gate,
            use_regime_contrast=contrast, memory_mode='historical',
            preserve_initialization_rng=True, contrast_loss_weight=0.002,
            gate_descriptor='forecast', detach_gate_descriptor=True,
            gate_loss_mode='power_weighted',
            historical_strength_coordinates=strength_coordinates))


def series():
    return np.random.default_rng(92).normal(size=(320, 2)).astype('float32')


class IntegrationChecks(unittest.TestCase):
    def test_initial_forecast_and_rng_match_original(self):
        for gate, coordinates in ((False, 'raw'), (True, 'raw'),
                                  (False, 'bounded'), (True, 'bounded')):
            torch.manual_seed(2024)
            baseline = Official(args())
            rng = torch.get_rng_state().clone()
            torch.manual_seed(2024)
            model = Model(args(gate=gate, strength_coordinates=coordinates))
            model.historical_memory.attach_training_series(series())
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            x, origin = torch.randn(4, 12, 2), torch.tensor([12, 70, 130, 230])
            for training in (False, True):
                baseline.train(training); model.train(training)
                torch.manual_seed(84); reference = baseline(x, origin)
                torch.manual_seed(84); output = model(x, origin)
                self.assertTrue(torch.equal(reference, output))

    def test_training_auxiliary_loss_gradients_and_checkpoint(self):
        torch.manual_seed(2024)
        model = Model(args(gate=True, contrast=True))
        model.historical_memory.attach_training_series(series())
        with torch.no_grad():
            model.historical_strength.fill_(0.3)
        x, target = torch.randn(8, 12, 2), torch.randn(8, 6, 2)
        origin = torch.arange(8) * 16 + 100
        loss = ForecastLoss(model)(model(x, origin), target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(model.diagnostics['contrast_valid_anchors'], 0)
        restored = Model(args(gate=True, contrast=True))
        restored.historical_memory.attach_training_series(series())
        restored.load_state_dict(model.state_dict())
        model.eval(); restored.eval()
        prediction = model(x, origin)
        self.assertTrue(torch.equal(prediction, restored(x, origin)))
        self.assertIsNone(model._retrieval_context)
        self.assertTrue(torch.equal(ForecastLoss(model)(prediction, target),
                                   torch.nn.functional.mse_loss(prediction, target)))

    def test_empty_causal_bank_falls_back_and_remains_finite(self):
        model = Model(args(gate=True, contrast=True))
        model.historical_memory.attach_training_series(series())
        model.eval()
        with torch.no_grad():
            model.historical_strength.fill_(1)
        x, origins = torch.randn(2, 12, 2), torch.tensor([12, 13])
        reference = Official.forward(model, x, origins)
        self.assertTrue(torch.equal(model(x, origins), reference))
        model.train()
        loss = ForecastLoss(model)(model(x, origins), torch.randn(2, 6, 2))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
