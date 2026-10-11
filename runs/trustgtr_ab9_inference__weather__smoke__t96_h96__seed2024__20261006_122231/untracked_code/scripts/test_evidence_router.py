#!/usr/bin/env python3
"""CPU evidence-routing invariants and frozen-source compatibility checks."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from models.HistoricalMemory import HistoricalMemory
from models.TrustGTR import Model, ForecastLoss, resolved_options

FROZEN_SOURCE = (ROOT / 'runs/trustgtr_a6__etth1__forecast__t96_h96__seed2024__20261005_160105'
                 / 'untracked_code/models/HistoricalMemory.py')
FROZEN_SHA256 = '3d8f84247121808d30a1690b7f835b4519424e4cea2ae50703a1e54ba546d6c0'


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2024)
        self.train = np.random.default_rng(2024).normal(size=(320, 3)).astype('float32')

    def memory(self, router='per_bank_evidence', collect=False, cls=HistoricalMemory):
        memory = cls(24, 12, 4, 3, [4, 28, 120], router_mode=router,
                     collect_reliability=collect)
        memory.attach_training_series(self.train)
        return memory

    def query(self, origins):
        return torch.from_numpy(np.stack([self.train[t - 24:t].T for t in origins]))

    def test_absolute_matching_quality_reweights_prior_without_count_bias(self):
        distances = torch.tensor([[[[0.01, 0.02, 0.03], [1., 2., 3.],
                                    [float('inf')] * 3]]], dtype=torch.float64)
        priors = torch.tensor([[[0.2, 0.8, 0.]]], dtype=torch.float64)
        weights, evidence, posterior, counts = HistoricalMemory._evidence_weights(distances, priors)
        first = sum(1 / (value + 0.001) for value in (0.01, 0.02, 0.03)) / 3
        second = sum(1 / (value + 0.001) for value in (1., 2., 3.)) / 3
        expected = 0.2 * first / (0.2 * first + 0.8 * second)
        self.assertAlmostEqual(float(posterior[0, 0, 0]), expected, places=12)
        self.assertGreater(float(posterior[0, 0, 0]), 0.95)
        self.assertTrue(torch.equal(counts, torch.tensor([[[3, 3, 0]]])))
        self.assertTrue(torch.allclose(weights.sum(-1), posterior))
        self.assertTrue(torch.equal(evidence[..., 2], torch.zeros_like(evidence[..., 2])))
        # One available match must have equal evidence to three equally good
        # matches. Padding with 1/inf=0 must not reduce that bank's mean.
        partial = torch.tensor([[[[1., float('inf'), float('inf')], [1., 1., 1.]]]])
        _, evidence, posterior, counts = HistoricalMemory._evidence_weights(partial, torch.full((1, 1, 2), 0.5))
        self.assertTrue(torch.equal(counts, torch.tensor([[[1, 3]]])))
        self.assertTrue(torch.equal(evidence[..., 0], evidence[..., 1]))
        self.assertTrue(torch.equal(posterior, torch.full_like(posterior, 0.5)))

    def test_inf_nan_zero_distance_and_tiny_evidence_remain_finite(self):
        for values in ((0., 0., float('inf')), (1e30, 2e30, float('inf')),
                       (float('inf'), float('nan'), float('inf'))):
            distances = torch.tensor(values).reshape(1, 1, 3, 1).requires_grad_()
            logits = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
            priors = logits.softmax(-1).reshape(1, 1, 3)
            weights, evidence, posterior, counts = HistoricalMemory._evidence_weights(distances, priors)
            for tensor in (weights, evidence, posterior):
                self.assertTrue(torch.isfinite(tensor).all())
            self.assertEqual(int(counts[..., 2]), 0)
            if counts.any():
                self.assertTrue(torch.allclose(weights.sum((-2, -1)), torch.ones(1, 1)))
            else:
                self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))
            (weights * torch.arange(3).reshape(1, 1, 3, 1)).sum().backward()
            self.assertTrue(torch.isfinite(distances.grad).all())
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_short_history_empty_fallback_and_diagnostic_semantics(self):
        memory = self.memory(collect=True)
        origins = torch.tensor([24, 25, 36, 79, 310])
        query = self.query(origins.tolist())
        prediction, state = memory(query, origins)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.equal(prediction[:2], query[:2].mean(-1).unsqueeze(1).expand(-1, 12, -1)))
        self.assertFalse(state['valid'][:2].any())
        self.assertTrue(torch.equal(state['cycle_weights'], state['bank_prior']))
        self.assertTrue(torch.allclose(state['bank_posterior'], state['bank_effective_weights']))
        self.assertTrue(torch.equal(state['bank_posterior'][2], torch.tensor([[1., 0., 0.]]).expand(3, -1)))
        self.assertTrue(torch.equal(state['bank_posterior'][:2], torch.zeros_like(state['bank_posterior'][:2])))
        self.assertTrue(torch.equal(state['bank_selected_valid_count'][:2],
                                    torch.zeros_like(state['bank_selected_valid_count'][:2])))
        self.assertTrue(torch.isfinite(state['reliability_features']).all())
        self.assertFalse(state['reliability_features'].requires_grad)
        for batch, origin in enumerate(origins):
            candidates = state['base_candidate_origin'][batch][state['base_candidate_mask'][batch]]
            self.assertTrue(torch.all(candidates + 12 <= origin))
            self.assertTrue(torch.all(candidates.remainder(4) == origin % 4))
        for key, field in (('bank_prior_sum', 'bank_prior'),
                           ('bank_posterior_sum', 'bank_posterior'),
                           ('bank_effective_weight_sum', 'bank_effective_weights')):
            self.assertTrue(torch.allclose(torch.tensor(memory.diagnostics[key]),
                                           state[field].sum((0, 1))))
        self.assertEqual(memory.diagnostics['bank_routing_count'], 15)
        self.assertEqual(memory.metadata['router_mode'], 'per_bank_evidence')

    def test_real_forecast_gradients_reach_router_and_key_encoder(self):
        memory = self.memory()
        with torch.no_grad():
            memory.key_encoder[-1].weight.fill_(0.001)
        origins = torch.tensor([270, 288, 310])
        prediction, _ = memory(self.query(origins.tolist()), origins)
        prediction.square().mean().backward()
        for name, parameter in memory.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(memory.cycle_router.weight.grad.abs().sum()), 1e-6)
        self.assertGreater(float(memory.key_encoder[-1].weight.grad.abs().sum()), 1e-6)

    def test_old_modes_predictions_gradients_parameters_and_rng_exact(self):
        self.assertTrue(FROZEN_SOURCE.is_file(), 'Frozen v6 source is required for exact compatibility')
        self.assertEqual(hashlib.sha256(FROZEN_SOURCE.read_bytes()).hexdigest(), FROZEN_SHA256)
        spec = importlib.util.spec_from_file_location('frozen_v6_historical_memory', FROZEN_SOURCE)
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        for router in ('global_topk', 'per_bank'):
            for collect in (False, True):
                with self.subTest(router=router, collect=collect):
                    torch.manual_seed(2024)
                    reference = self.memory(router, collect, frozen.HistoricalMemory)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(2024)
                    current = self.memory(router, collect)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    self.assertEqual(set(current.state_dict()), set(reference.state_dict()))
                    self.assertEqual(current.metadata, reference.metadata)
                    origins = torch.tensor([24, 36, 79, 310])
                    query = self.query(origins.tolist())
                    left, old_state = reference(query, origins)
                    right, new_state = current(query, origins)
                    self.assertTrue(torch.equal(left, right))
                    self.assertEqual(set(old_state), set(new_state))
                    for key in old_state:
                        self.assertTrue(torch.equal(old_state[key], new_state[key]), key)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    left.square().mean().backward()
                    right.square().mean().backward()
                    for old, new in zip(reference.parameters(), current.parameters()):
                        self.assertTrue(torch.equal(old.grad, new.grad))
                    self.assertEqual(current.diagnostics, reference.diagnostics)

    def test_mode_adds_no_parameters_and_full_model_save_load(self):
        options = dict(use_multi_cycle=True, memory_mode='historical',
                       preserve_initialization_rng=True, historical_strength_coordinates='bounded',
                       historical_router_mode='per_bank_evidence', use_regime_contrast=True,
                       contrast_loss_weight=0.002)
        configuration = SimpleNamespace(seq_len=24, pred_len=12, enc_in=3, cycle=4,
                                        d_model=16, dropout=0.5, use_revin=1,
                                        individual=0, trustgtr=options)
        candidate = Model(configuration)
        candidate.historical_memory.attach_training_series(self.train)
        original = self.memory('per_bank')
        self.assertEqual(set(candidate.historical_memory.state_dict()), set(original.state_dict()))
        self.assertEqual(sum(p.numel() for p in candidate.historical_memory.parameters()),
                         sum(p.numel() for p in original.parameters()))
        with torch.no_grad():
            candidate.historical_strength.fill_(0.08)
        origins = torch.tensor([270, 288, 310])
        x = self.query(origins.tolist()).transpose(1, 2)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=0.001)
        loss = ForecastLoss(candidate)(candidate(x, origins), torch.randn(3, 12, 3))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        restored = Model(configuration)
        restored.historical_memory.attach_training_series(self.train)
        restored.load_state_dict(candidate.state_dict())
        candidate.eval(); restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(candidate(x, origins), restored(x, origins)))
        self.assertIsNone(candidate._retrieval_context)
        self.assertEqual(resolved_options(options, 4)['historical_router_mode'], 'per_bank_evidence')


if __name__ == '__main__':
    unittest.main(verbosity=2)
