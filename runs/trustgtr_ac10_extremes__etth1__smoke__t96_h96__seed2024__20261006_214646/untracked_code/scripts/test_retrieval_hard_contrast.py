#!/usr/bin/env python3
"""CPU checks for the single-factor, actual-retrieval C negative sampler."""
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
from torch.nn import functional as F

from models.GTR import Model as Official
from models.HistoricalMemory import HistoricalMemory
from models.TrustGTR import Model, ForecastLoss, resolved_options

FROZEN_RUN = 'trustgtr_ab9_train__etth1__forecast__t96_h96__seed2024__20261006_122231'
FROZEN_SHA = {
    'TrustGTR.py': '59fef362c5e61905bcd4db9f866ef9f30454ab63f511c797c72eff28eb540970',
    'HistoricalMemory.py': 'c05182901b603689b4852fa4b569d41f9ba4db938e3d3fcd462f5dfd3f90b6b9',
}


def options(**updates):
    result = dict(use_multi_cycle=True, use_regime_contrast=True,
                  memory_mode='historical', preserve_initialization_rng=True,
                  historical_router_mode='per_bank', contrast_metric='euclidean',
                  contrast_temperature=3., contrast_loss_weight=0.002,
                  positive_max_distance=1., regime_distance_gap=0.25)
    result.update(updates)
    return result


def configuration(opts, revin=1, individual=0):
    return SimpleNamespace(seq_len=12, pred_len=6, enc_in=2, cycle=4,
                           d_model=16, dropout=0.5, use_revin=revin,
                           individual=individual, trustgtr=opts)


def load_archived():
    modules = {}
    for filename, digest in FROZEN_SHA.items():
        source = ROOT / 'runs' / FROZEN_RUN / 'untracked_code/models' / filename
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise AssertionError('Archived source hash changed: ' + str(source))
        spec = importlib.util.spec_from_file_location('step65_frozen_' + source.stem, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[source.stem] = module
    # Compare both frozen science files; do not silently import today's bank.
    modules['TrustGTR'].HistoricalMemory = modules['HistoricalMemory'].HistoricalMemory
    return modules['TrustGTR']


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2024)
        self.series = np.random.default_rng(92).normal(size=(320, 2)).astype('float32')
        self.origins = torch.tensor([12, 40, 130, 280])
        self.x = torch.from_numpy(np.stack([self.series[t - 12:t] for t in self.origins.tolist()]))
        self.target = torch.randn(4, 6, 2)

    def model(self, opts=None, cls=Model, **config):
        model = cls(configuration(options() if opts is None else opts, **config))
        if model.historical_memory is not None:
            model.historical_memory.attach_training_series(self.series)
            with torch.no_grad():
                model.historical_strength.fill_(0.17)
        if model.gate is not None:
            with torch.no_grad():
                model.gate[-1].weight.fill_(0.02)
        return model

    def equal_gradients(self, old, new):
        self.assertEqual(set(dict(old.named_parameters())), set(dict(new.named_parameters())))
        for (name, left), (_, right) in zip(old.named_parameters(), new.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad), name)

    def test_defaults_scope_and_inactive_switch(self):
        self.assertEqual(resolved_options({}, 4)['contrast_sampling'], 'regime_extremes')
        with self.assertRaisesRegex(ValueError, 'Unknown contrastive sampling'):
            resolved_options(dict(contrast_sampling='other'), 4)
        with self.assertRaisesRegex(ValueError, 'requires historical'):
            resolved_options(dict(use_multi_cycle=True, use_regime_contrast=True,
                                  contrast_sampling='retrieval_hard'), 4)
        resolved_options(dict(contrast_sampling='retrieval_hard'), 4)
        with self.assertRaisesRegex(ValueError, 'collect_contrast must be a bool'):
            HistoricalMemory(12, 6, 4, 2, [4, 28, 120], collect_contrast=1)

    def test_all_off_exact_official_initialization_forward_backward_rng(self):
        torch.manual_seed(2024)
        old = Official(configuration({}))
        initial_rng = torch.get_rng_state().clone()
        torch.manual_seed(2024)
        new = self.model(dict(contrast_sampling='retrieval_hard'))
        self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        torch.manual_seed(53)
        left = old(self.x, self.origins)
        F.mse_loss(left, self.target).backward()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(53)
        right = new(self.x, self.origins)
        ForecastLoss(new)(right, self.target).backward()
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.equal_gradients(old, new)

    def test_default_exact_frozen_c_losses_and_all_routers(self):
        frozen = load_archived()
        cases = [options(historical_router_mode=router, contrast_metric=metric,
                         use_reliability_gate=gate, gate_descriptor='retrieval',
                         gate_loss_mode='power_weighted', gate_calibration_context=context)
                 for router in ('global_topk', 'per_bank', 'per_bank_evidence')
                 for metric in ('cosine', 'euclidean')
                 for gate, context in ((False, 'train'), (True, 'train'), (True, 'inference'))]
        for index, opts in enumerate(cases):
            with self.subTest(opts=opts):
                torch.manual_seed(2024)
                old = self.model(opts, cls=frozen.Model, revin=index % 2)
                initial_rng = torch.get_rng_state().clone()
                torch.manual_seed(2024)
                new = self.model({**opts, 'contrast_sampling': 'regime_extremes'}, revin=index % 2)
                self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
                for name in old.state_dict():
                    self.assertTrue(torch.equal(old.state_dict()[name], new.state_dict()[name]), name)
                torch.manual_seed(53)
                left = old(self.x, self.origins)
                old_loss = frozen.ForecastLoss(old)(left, self.target)
                old_loss.backward()
                after = torch.get_rng_state().clone()
                torch.manual_seed(53)
                right = new(self.x, self.origins)
                new_loss = ForecastLoss(new)(right, self.target)
                new_loss.backward()
                self.assertTrue(torch.equal(left, right))
                self.assertTrue(torch.equal(old_loss, new_loss))
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                self.equal_gradients(old, new)
                self.assertEqual(old.diagnostics, new.diagnostics)
                self.assertEqual(old.historical_memory.diagnostics, new.historical_memory.diagnostics)
                old.eval(); new.eval()
                torch.manual_seed(53)
                left = old(self.x, self.origins)
                after = torch.get_rng_state().clone()
                torch.manual_seed(53)
                right = new(self.x, self.origins)
                self.assertTrue(torch.equal(left, right))
                self.assertTrue(torch.equal(after, torch.get_rng_state()))

    def test_sampling_changes_only_aux_not_main_forward_or_gradients(self):
        for router in ('global_topk', 'per_bank', 'per_bank_evidence'):
            with self.subTest(router=router):
                opts = options(historical_router_mode=router)
                torch.manual_seed(2024)
                old = self.model(opts)
                initial_rng = torch.get_rng_state().clone()
                torch.manual_seed(2024)
                new = self.model({**opts, 'contrast_sampling': 'retrieval_hard'})
                self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
                self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                torch.manual_seed(53)
                left = old(self.x, self.origins)
                F.mse_loss(left, self.target).backward()
                after = torch.get_rng_state().clone()
                torch.manual_seed(53)
                right = new(self.x, self.origins)
                F.mse_loss(right, self.target).backward()
                self.assertTrue(torch.equal(left, right))
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                self.equal_gradients(old, new)

    def test_c_off_no_collector_or_aux_side_effects(self):
        for router in ('global_topk', 'per_bank', 'per_bank_evidence'):
            with self.subTest(router=router):
                opts = options(use_regime_contrast=False, historical_router_mode=router)
                torch.manual_seed(2024)
                old = self.model(opts)
                torch.manual_seed(2024)
                new = self.model({**opts, 'contrast_sampling': 'retrieval_hard'})
                self.assertFalse(new.historical_memory.collect_contrast)
                torch.manual_seed(53)
                left = old(self.x, self.origins)
                ForecastLoss(old)(left, self.target).backward()
                after = torch.get_rng_state().clone()
                torch.manual_seed(53)
                right = new(self.x, self.origins)
                ForecastLoss(new)(right, self.target).backward()
                self.assertTrue(torch.equal(left, right))
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                self.equal_gradients(old, new)
                self.assertEqual(new.auxiliary_losses(self.target), {})
                self.assertFalse(any(name.startswith('selected_candidate_')
                                     for name in new._retrieval_context['state']))

    def test_actual_collector_matches_differentiable_retrieval_all_routers(self):
        for router in ('global_topk', 'per_bank', 'per_bank_evidence'):
            with self.subTest(router=router):
                model = self.model(options(contrast_sampling='retrieval_hard', historical_router_mode=router))
                model(self.x, self.origins)
                state = model._retrieval_context['state']
                positions = state['selected_candidate_positions']
                distance = state['selected_candidate_distances']
                weight = state['selected_candidate_weights']
                self.assertFalse(distance.requires_grad)
                self.assertFalse(weight.requires_grad)
                self.assertTrue(state['base_candidate_key_embeddings'].requires_grad)
                self.assertTrue(state['query_embedding'].requires_grad)
                self.assertTrue(torch.equal(weight, state['selected_weights'].detach()))
                full = (state['base_candidate_key_embeddings'] - state['query_embedding'].unsqueeze(2)).square().sum(-1)
                actual = full.gather(-1, positions.clamp_min(0))
                valid = positions >= 0
                self.assertTrue(torch.equal(actual[valid], distance[valid]))
                self.assertTrue(torch.isinf(distance[~valid]).all())
                self.assertTrue(torch.equal(weight[~valid], torch.zeros_like(weight[~valid])))
                selected_origin = state['base_candidate_origin'].unsqueeze(1).expand_as(full).gather(-1, positions.clamp_min(0))
                origin = self.origins[:, None, None].expand_as(selected_origin)
                self.assertTrue((selected_origin[valid] % 4 == origin[valid] % 4).all())
                self.assertTrue((selected_origin[valid] + model.pred_len <= origin[valid]).all())
                model.eval()
                _, evaluation = model.historical_memory(self.x.permute(0, 2, 1), self.origins)
                self.assertFalse(any(name.startswith('selected_candidate_') for name in evaluation))
                model(self.x, self.origins)
                self.assertIsNone(model._retrieval_context)
                self.assertEqual(model.auxiliary_losses(self.target), {})

    def synthetic_cache(self):
        model = self.model(options(contrast_sampling='retrieval_hard'))
        query = torch.tensor([[[0., 0.]]], requires_grad=True)
        candidates = torch.tensor([[[[0.2, 0.], [0.4, 0.], [0.1, 0.],
                                    [0.3, 0.], [0.05, 0.], [0.5, 0.]]]], requires_grad=True)
        state = dict(base_candidate_index=torch.arange(6).view(1, 6),
                     base_candidate_mask=torch.ones(1, 6, dtype=torch.bool),
                     base_candidate_origin=torch.tensor([[20, 24, 28, 32, 36, 96]]),
                     regime=torch.tensor([[[0., 0.]], [[0.6, 0.]], [[0.8, 0.]],
                                          [[3., 0.]], [[5., 0.]], [[9., 0.]]]),
                     query_embedding=query, base_candidate_key_embeddings=candidates,
                     selected_candidate_positions=torch.tensor([[[0, 1, 2, 3, 5, -1]]]),
                     selected_candidate_distances=torch.tensor([[[0.04, 0.16, 0.01, 0.09, 0.25, float('inf')]]]),
                     selected_candidate_weights=torch.tensor([[[0.1, 0.2, 0.4, 0.2, 0.1, 0.]]]))
        cache = dict(state=state, origin=torch.tensor([100]), history_mean=torch.zeros(1, 1),
                     history_std=torch.ones(1, 1))
        return model, cache, query, candidates

    def test_negative_is_actual_nearest_eligible_not_global_regime_farthest(self):
        model, cache, query, candidates = self.synthetic_cache()
        target = torch.zeros(1, 6, 1, requires_grad=True)
        loss = model.historical_contrastive_loss(target, cache)
        # Positive slot 0; actual key-nearest eligible negative is slot 2.
        # Slot 4 is a closer key but was never retrieved. Slot 5 overlaps.
        logits = torch.stack((-(query - candidates[:, :, 0]).square().sum(-1),
                              -(query - candidates[:, :, 2]).square().sum(-1)), -1) / 3
        expected = F.cross_entropy(logits.reshape(-1, 2), torch.zeros(1, dtype=torch.long))
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(query.grad.abs().sum().item(), 0)
        self.assertGreater(candidates.grad[:, :, 0].abs().sum().item(), 0)
        self.assertGreater(candidates.grad[:, :, 2].abs().sum().item(), 0)
        self.assertEqual(candidates.grad[:, :, [1, 3, 4, 5]].abs().sum().item(), 0)
        self.assertEqual(model.diagnostics['contrast_valid_anchors'], 1)

    def test_zero_weights_gap_overlap_sentinel_and_ties(self):
        model, cache, _, _ = self.synthetic_cache()
        state = cache['state']
        regime = torch.tensor([[[0., 0.6, 0.8, 3., 5., 9.]]])
        mask = torch.tensor([[True, True, True, True, True, False]])
        near, positive = torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.long)
        # Remove nearest slot 2 by actual zero mass; slot 3 then wins.
        state['selected_candidate_weights'][..., 2] = 0
        negative, distance = model.retrieval_hard_negative(regime, mask, near, positive, state)
        self.assertEqual(negative.item(), 3)
        self.assertEqual(distance.item(), 3.)
        # Equal eligible keys retain recorded slot order (slot 1 before 3).
        state['selected_candidate_distances'][..., 1] = 0.09
        negative, _ = model.retrieval_hard_negative(regime, mask, near, positive, state)
        self.assertEqual(negative.item(), 1)
        # A future-regime near match is not a negative even if key-nearest.
        regime[..., 1] = 0.1
        negative, _ = model.retrieval_hard_negative(regime, mask, near, positive, state)
        self.assertEqual(negative.item(), 3)
        state['selected_candidate_weights'].zero_()
        _, distance = model.retrieval_hard_negative(regime, mask, near, positive, state)
        self.assertTrue(torch.isneginf(distance).all())

    def test_cosine_formula_and_explicit_empty_selection(self):
        model, cache, _, _ = self.synthetic_cache()
        model.options['contrast_metric'] = 'cosine'
        query = torch.tensor([[[0.8, 0.6]]], requires_grad=True)
        candidates = cache['state']['base_candidate_key_embeddings']
        cache['state']['query_embedding'] = query
        loss = model.historical_contrastive_loss(torch.zeros(1, 6, 1), cache)
        similarity = (F.normalize(query, dim=-1).unsqueeze(2) *
                      F.normalize(candidates, dim=-1)).sum(-1) / 3
        expected = F.cross_entropy(similarity[..., [0, 2]].reshape(1, 2),
                                   torch.zeros(1, dtype=torch.long))
        self.assertTrue(torch.equal(loss, expected))
        loss.backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(candidates.grad).all())
        # Direct private zero-width retrieval retains a typed empty collector;
        # public _candidates instead uses its masked sentinel for reductions.
        bank = model.historical_memory
        context = {}
        output = bank._retrieve_bank(torch.zeros(1, 2, 15), torch.zeros(1, 2),
                                     torch.ones(1, 2), torch.empty(1, 0, dtype=torch.long),
                                     torch.empty(1, 0, dtype=torch.long), torch.zeros(1, 2, 3),
                                     selection_state=context)
        self.assertEqual(output[0].shape, (1, 6, 2))
        self.assertEqual(context['selected_candidate_positions'].dtype, torch.long)
        self.assertEqual(context['selected_candidate_positions'].shape, (1, 2, 0))
        state = {name: torch.empty(1, 1, 0, dtype=torch.long if name.endswith('positions') else torch.float)
                 for name in context}
        _, distance = model.retrieval_hard_negative(torch.zeros(1, 1, 1),
            torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1, 1),
            torch.zeros(1, 1, dtype=torch.long), state)
        self.assertTrue(torch.isneginf(distance).all())

    def test_positive_threshold_and_missing_pairs_return_differentiable_zero(self):
        model, cache, query, candidates = self.synthetic_cache()
        cache['state']['regime'] += 2
        loss = model.historical_contrastive_loss(torch.zeros(1, 6, 1), cache)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(query.grad, torch.zeros_like(query)))
        self.assertTrue(torch.equal(candidates.grad, torch.zeros_like(candidates)))
        self.assertEqual(model.diagnostics['contrast_valid_anchors'], 0)
        for router in ('global_topk', 'per_bank', 'per_bank_evidence'):
            with self.subTest(router=router):
                model = self.model(options(contrast_sampling='retrieval_hard', historical_router_mode=router))
                origins = torch.tensor([12, 13])
                prediction = model(self.x[:2], origins)
                loss = model.auxiliary_losses(self.target[:2])['contrast']
                self.assertEqual(loss.item(), 0)
                self.assertTrue(loss.requires_grad)
                loss.backward()
                self.assertEqual(model.diagnostics['contrast_valid_anchors'], 0)
                self.assertTrue(torch.isfinite(prediction).all())

    def test_training_future_labels_only_auxiliary_and_memory_immutable(self):
        model = self.model(options(contrast_sampling='retrieval_hard', positive_max_distance=100.))
        snapshot = {name: tensor.clone() for name, tensor in model.historical_memory.named_buffers()}
        model(self.x, self.origins)
        state = model._retrieval_context['state']
        target = self.target.clone().requires_grad_()
        before = {name: value.clone() for name, value in state.items() if isinstance(value, torch.Tensor)}
        rng = torch.get_rng_state().clone()
        loss = model.auxiliary_losses(target)['contrast']
        loss.backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertGreater(model.diagnostics['contrast_valid_anchors'], 0)
        self.assertLessEqual(model.diagnostics['contrast_valid_anchors'], model.diagnostics['contrast_total_anchors'])
        gradient = model.historical_memory.key_encoder[-1].weight.grad
        self.assertGreater(gradient.abs().sum().item(), 0)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, state[name]), name)
        for name, tensor in model.historical_memory.named_buffers():
            self.assertTrue(torch.equal(snapshot[name], tensor), name)

    def test_missing_or_malformed_collector_fails_clearly(self):
        model, cache, _, _ = self.synthetic_cache()
        del cache['state']['selected_candidate_positions']
        with self.assertRaisesRegex(RuntimeError, 'training retrieval collector'):
            model.historical_contrastive_loss(torch.zeros(1, 6, 1), cache)
        model, cache, _, _ = self.synthetic_cache()
        cache['state']['selected_candidate_weights'] = torch.ones(1, 1, 2)
        with self.assertRaisesRegex(ValueError, 'Mismatched retrieval selection'):
            model.historical_contrastive_loss(torch.zeros(1, 6, 1), cache)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
