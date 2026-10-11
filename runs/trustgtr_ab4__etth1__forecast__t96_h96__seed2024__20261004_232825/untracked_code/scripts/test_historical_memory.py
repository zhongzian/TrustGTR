#!/usr/bin/env python3
"""CPU checks against the independently established retrieval diagnostic."""
import importlib.util
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.HistoricalMemory import HistoricalMemory


spec = importlib.util.spec_from_file_location('fixed_retrieval_diagnostic', ROOT / 'scripts/diagnose_historical_retrieval.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2024)
        self.train = np.random.default_rng(2024).normal(size=(320, 3)).astype('float32')
        self.memory = HistoricalMemory(24, 12, 4, 3, [4, 28, 120])
        self.memory.attach_training_series(self.train)

    def query(self, origins):
        return torch.from_numpy(np.stack([self.train[t - 24:t].T for t in origins]))

    def test_initial_prediction_and_keys_match_fixed_diagnostic(self):
        origins = [210, 279, 320]
        query = self.query(origins)
        prediction, state = self.memory(query, torch.tensor(origins))
        bank = diagnostic.history_bank(self.train, 24, 12)
        self.assertTrue(np.array_equal(bank['keys'], self.memory.history_keys.numpy()))
        for batch, origin in enumerate(origins):
            expected, _ = diagnostic.retrieve(query[batch].T.numpy(), origin, self.train, bank, 24, 12, 4, True)
            np.testing.assert_allclose(prediction[batch].detach().numpy(), expected, rtol=2e-5, atol=2e-6)
            candidates, _ = diagnostic.age_coverage_candidates(bank['origins'], origin, 12, 4, True)
            actual = state['base_candidate_index'][batch][state['base_candidate_mask'][batch]].numpy()
            self.assertTrue(np.array_equal(candidates, actual))

    def test_causality_phase_age_cap_and_no_empty_bank_nan(self):
        origins = [24, 36, 79, 310]
        prediction, state = self.memory(self.query(origins), torch.tensor(origins))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertFalse(state['valid'][0])
        self.assertTrue(torch.equal(prediction[0], self.query(origins)[0].mean(-1).expand(12, -1)))
        for batch, origin in enumerate(origins):
            mask = state['base_candidate_mask'][batch]
            candidates = state['base_candidate_origin'][batch][mask]
            self.assertTrue(torch.all(candidates + 12 <= origin))
            self.assertTrue(torch.all(candidates.remainder(4) == origin % 4))
            self.assertLessEqual(len(candidates), 192)
            provenance = state['candidate_bank_index'][batch][mask]
            for bank in range(3):
                self.assertLessEqual((provenance == bank).sum().item(), 64)
        all_empty, empty_state = self.memory(self.query([24, 25]), torch.tensor([24, 25]))
        self.assertTrue(torch.isfinite(all_empty).all())
        self.assertFalse(empty_state['valid'].any())
        self.assertEqual(empty_state['base_candidate_index'].shape, (2, 1))
        self.assertTrue((empty_state['base_candidate_index'] == -1).all())
        self.assertFalse(empty_state['base_candidate_mask'].any())

    def test_encoder_and_cycle_prior_receive_forecast_gradients(self):
        origins = [270, 288, 310]
        prediction, _ = self.memory(self.query(origins), torch.tensor(origins))
        prediction.square().mean().backward()
        for parameter in self.memory.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.memory.key_encoder[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.memory.cycle_router.weight.grad.abs().sum().item(), 0)

    def test_nonpersistent_buffers_reattach_checkpoint_alignment(self):
        with torch.no_grad():
            self.memory.key_encoder[-1].weight.fill_(0.001)
            self.memory.cycle_router.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
        payload = self.memory.state_dict()
        self.assertEqual(set(payload), {name for name, _ in self.memory.named_parameters()})
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        replica = HistoricalMemory(24, 12, 4, 3, [4, 28, 120])
        replica.load_state_dict(torch.load(stream, map_location='cpu'))
        rng_before = torch.get_rng_state().clone()
        replica.attach_training_series(self.train)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertEqual(replica.metadata['sha256'], self.memory.metadata['sha256'])
        self.assertEqual(replica.metadata['train_rows'], 320)
        query, origin = self.query([288, 310]), torch.tensor([288, 310])
        self.assertTrue(torch.equal(self.memory(query, origin)[0], replica(query, origin)[0]))

    def test_bank_is_static_and_rejects_bad_training_assets(self):
        digest = self.memory.metadata['sha256']
        original = self.memory.series.clone()
        self.memory(self.query([310]), torch.tensor([310]))
        self.assertTrue(torch.equal(original, self.memory.series))
        self.assertEqual(digest, self.memory.metadata['sha256'])
        with self.assertRaises(ValueError):
            self.memory.attach_training_series(np.ones((40, 4), dtype='float32'))
        with self.assertRaises(ValueError):
            self.memory.attach_training_series(np.full((40, 3), np.nan, dtype='float32'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
