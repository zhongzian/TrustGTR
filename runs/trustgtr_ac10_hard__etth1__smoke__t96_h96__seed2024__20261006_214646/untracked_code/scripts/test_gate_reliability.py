#!/usr/bin/env python3
"""CPU checks for causal retrieval reliability inputs and legacy behavior."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from models.GTR import Model as Official
from models.HistoricalMemory import HistoricalMemory
from models.TrustGTR import Model, ForecastLoss, resolved_options


def configuration(options):
    return SimpleNamespace(seq_len=24, pred_len=12, enc_in=3, cycle=4,
                           d_model=16, dropout=0.5, use_revin=1, individual=0,
                           trustgtr=options)


def historical_options(**updates):
    return dict(use_multi_cycle=True, use_reliability_gate=True,
                memory_mode='historical', preserve_initialization_rng=True,
                gate_descriptor='retrieval', gate_loss_mode='power_weighted',
                historical_strength_coordinates='bounded', **updates)


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.train = np.random.default_rng(2024).normal(size=(320, 3)).astype('float32')

    def memory(self, collect=True, router='global_topk'):
        memory = HistoricalMemory(24, 12, 4, 3, [4, 28, 120],
                                  router_mode=router, collect_reliability=collect)
        memory.attach_training_series(self.train)
        return memory

    def query(self, origins):
        return torch.from_numpy(np.stack([self.train[t - 24:t].T for t in origins]))

    def test_collection_preserves_predictions_gradients_and_rng(self):
        for router in ('global_topk', 'per_bank'):
            torch.manual_seed(2024)
            original = self.memory(False, router)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(2024)
            collected = self.memory(True, router)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            origins = torch.tensor([270, 288, 310])
            query = self.query(origins.tolist())
            left, original_state = original(query, origins)
            right, collected_state = collected(query, origins)
            self.assertTrue(torch.equal(left, right))
            self.assertNotIn('reliability_features', original_state)
            self.assertFalse(collected_state['reliability_features'].requires_grad)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            left.square().mean().backward()
            right.square().mean().backward()
            for old, new in zip(original.parameters(), collected.parameters()):
                self.assertTrue(torch.equal(old.grad, new.grad))

    def test_empty_single_and_partial_banks_are_finite_and_causal(self):
        for router in ('global_topk', 'per_bank'):
            memory = self.memory(True, router)
            origins = torch.tensor([24, 25, 36, 310])
            prediction, state = memory(self.query(origins.tolist()), origins)
            features = state['reliability_features']
            self.assertEqual(features.shape, (4, 3, 8))
            self.assertTrue(torch.isfinite(features).all())
            self.assertFalse(features.requires_grad)
            self.assertTrue(torch.equal(features[:2], torch.zeros_like(features[:2])))
            self.assertTrue(torch.equal(features[2, :, 1:3], torch.zeros(3, 2)))
            self.assertTrue(torch.allclose(features[2, :, 5], torch.full((3,), 1 / 3)))
            for batch, origin in enumerate(origins):
                candidate_origins = state['base_candidate_origin'][batch][state['base_candidate_mask'][batch]]
                self.assertTrue(torch.all(candidate_origins + 12 <= origin))
            prediction.square().mean().backward()
            for parameter in memory.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_future_and_bank_disagreement_have_known_response(self):
        memory = self.memory()
        distances = torch.ones(1, 3, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        banks = torch.tensor([[[0, 1]]]).expand(1, 3, 2)
        weights = torch.full((1, 3, 2), 0.5)
        selected = torch.tensor([[[0, 1]]]).expand(1, 3, 2)
        futures = torch.zeros(1, 3, 2, 12)
        consistent = memory._reliability_features(distances, mask, banks, weights,
                                                  futures, selected, torch.tensor([100]))
        futures[:, :, 0] = 1
        futures[:, :, 1] = -1
        disagree = memory._reliability_features(distances, mask, banks, weights,
                                                futures, selected, torch.tensor([100]))
        self.assertTrue(torch.equal(consistent[..., 3:5], torch.zeros(1, 3, 2)))
        self.assertTrue(torch.allclose(disagree[..., 3:5], torch.full((1, 3, 2), np.log(2))))
        self.assertTrue(torch.allclose(disagree[..., 2], torch.ones(1, 3)))
        self.assertTrue(torch.allclose(disagree[..., 5], torch.full((1, 3), 2 / 3)))
        single_bank = memory._reliability_features(distances, mask, torch.zeros_like(banks), weights,
                                                  futures, selected, torch.tensor([100]))
        self.assertTrue(torch.equal(single_bank[..., 4], torch.zeros(1, 3)))
        self.assertTrue(torch.allclose(single_bank[..., 5], torch.full((1, 3), 1 / 3)))

    def test_new_gate_has_only_128_extra_parameters_and_detached_inputs(self):
        torch.manual_seed(2024)
        old_options = historical_options()
        old_options['gate_descriptor'] = 'forecast'
        old = Model(configuration(old_options))
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(2024)
        new = Model(configuration(historical_options()))
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in new.parameters()) -
                         sum(p.numel() for p in old.parameters()), 128)
        self.assertFalse(old.historical_memory.collect_reliability)
        self.assertTrue(new.historical_memory.collect_reliability)
        for model in (old, new):
            model.historical_memory.attach_training_series(self.train)
            with torch.no_grad():
                model.historical_strength.fill_(0.08)
        origins = torch.tensor([270, 288, 310])
        x = self.query(origins.tolist()).transpose(1, 2)
        torch.manual_seed(17)
        left = old(x, origins)
        captured = []
        handle = new.gate.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
        torch.manual_seed(17)
        right = new(x, origins)
        handle.remove()
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(captured[0].shape, (3, 3, 18))
        self.assertFalse(captured[0].requires_grad)
        self.assertTrue(torch.isfinite(captured[0]).all())
        before = new._retrieval_context['state']['reliability_features'].clone()
        target = torch.randn_like(right)
        loss = ForecastLoss(new)(right, target)
        loss.backward()
        self.assertTrue(torch.equal(before, new._retrieval_context['state']['reliability_features']))
        for parameter in new.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        new.eval()
        with torch.no_grad():
            prediction = new(x, origins)
        self.assertIsNone(new._retrieval_context)
        self.assertTrue(torch.equal(ForecastLoss(new)(prediction, target),
                                    torch.nn.functional.mse_loss(prediction, target)))

    def test_disabling_gate_avoids_collection_and_all_off_preserves_baseline(self):
        options = historical_options()
        options['use_reliability_gate'] = False
        disabled = Model(configuration(options))
        self.assertFalse(disabled.historical_memory.collect_reliability)
        self.assertIsNone(disabled.gate)
        torch.manual_seed(2024)
        original = Official(configuration({}))
        rng = torch.get_rng_state().clone()
        torch.manual_seed(2024)
        off = Model(configuration({}))
        self.assertEqual(set(original.state_dict()), set(off.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        x, origin = torch.randn(3, 24, 3), torch.tensor([0, 1, 3])
        torch.manual_seed(17)
        left = original(x, origin)
        torch.manual_seed(17)
        right = off(x, origin)
        self.assertTrue(torch.equal(left, right))
        left.square().mean().backward()
        right.square().mean().backward()
        for old, new in zip(original.parameters(), off.parameters()):
            self.assertTrue(torch.equal(old.grad, new.grad))

    def test_descriptor_validation(self):
        with self.assertRaises(ValueError):
            resolved_options(dict(gate_descriptor='retrieval'), 4)
        with self.assertRaises(ValueError):
            HistoricalMemory(24, 12, 4, 3, [4, 28, 120], collect_reliability='yes')


if __name__ == '__main__':
    unittest.main(verbosity=2)
