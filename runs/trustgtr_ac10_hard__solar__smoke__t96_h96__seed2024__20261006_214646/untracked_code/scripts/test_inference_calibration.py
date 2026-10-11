#!/usr/bin/env python3
"""CPU invariants of lazy inference-context gate calibration, without GPUs."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F

from models.GTR import Model as Official
from models.TrustGTR import Model, ForecastLoss, resolved_options

FROZEN_RUNS = [
    'trustgtr_a8__etth1__forecast__t96_h96__seed2024__20261005_221654',
    'trustgtr_a8__solar__forecast__t96_h96__seed2024__20261005_221654',
]
FROZEN_SHA256 = '58196668bdc9b4ccbb49d9290ede3521848a17ddd04ab9f4de006f1466bb9a48'


def configuration(options, revin=1, individual=0, dropout=0.5):
    return SimpleNamespace(seq_len=24, pred_len=12, enc_in=3, cycle=4,
        d_model=16, dropout=dropout, use_revin=revin,
        individual=individual, trustgtr=options)


def historical_options(**updates):
    result = dict(use_multi_cycle=True, use_reliability_gate=True,
        memory_mode='historical', preserve_initialization_rng=True,
        gate_descriptor='retrieval', gate_loss_mode='power_weighted',
        historical_strength_coordinates='bounded', historical_router_mode='per_bank')
    result.update(updates)
    return result


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2024)
        self.series = np.random.default_rng(2024).normal(size=(320, 3)).astype('float32')
        self.origins = torch.tensor([24, 36, 140, 280])
        self.x = torch.from_numpy(np.stack([self.series[t - 24:t] for t in self.origins.tolist()]))
        self.target = torch.randn(4, 12, 3)

    def candidate(self, options=None, cls=Model, **config):
        model = cls(configuration(historical_options() if options is None else options, **config))
        if model.historical_memory is not None:
            model.historical_memory.attach_training_series(self.series)
            with torch.no_grad():
                model.historical_strength.fill_(0.08)
        if model.gate is not None:
            with torch.no_grad():
                model.gate[-1].weight.fill_(0.02)
        return model

    def assert_gradients_equal(self, left, right):
        self.assertEqual(set(dict(left.named_parameters())), set(dict(right.named_parameters())))
        for (name, old), (_, new) in zip(left.named_parameters(), right.named_parameters()):
            if old.grad is None:
                self.assertIsNone(new.grad, name)
            else:
                self.assertTrue(torch.equal(old.grad, new.grad), name)

    def test_default_validation_and_supported_scope(self):
        self.assertEqual(resolved_options({}, 4)['gate_calibration_context'], 'train')
        self.assertEqual(resolved_options(historical_options(gate_calibration_context='inference'), 4)
                         ['gate_calibration_context'], 'inference')
        with self.assertRaisesRegex(ValueError, 'Unknown gate calibration'):
            resolved_options(dict(gate_calibration_context='bad'), 4)
        with self.assertRaisesRegex(ValueError, 'requires historical'):
            resolved_options(dict(use_reliability_gate=True, gate_calibration_context='inference'), 4)
        # The inactive switch must not prevent exact all-off baseline behavior.
        resolved_options(dict(gate_calibration_context='inference'), 4)

    def test_all_off_exact_official_parameters_rng_prediction_and_gradient(self):
        for context in ('train', 'inference'):
            with self.subTest(context=context):
                torch.manual_seed(2024)
                old = Official(configuration({}))
                initial_rng = torch.get_rng_state().clone()
                torch.manual_seed(2024)
                new = self.candidate(dict(gate_calibration_context=context))
                self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
                self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                torch.manual_seed(53)
                left = old(self.x, self.origins)
                left.square().mean().backward()
                after = torch.get_rng_state().clone()
                torch.manual_seed(53)
                right = new(self.x, self.origins)
                right.square().mean().backward()
                self.assertTrue(torch.equal(left, right))
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                self.assert_gradients_equal(old, new)

    def test_train_context_exact_archived_source_for_all_losses_descriptors_and_modes(self):
        for run in FROZEN_RUNS:
            source = ROOT / 'runs' / run / 'untracked_code/models/TrustGTR.py'
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), FROZEN_SHA256)
        spec = importlib.util.spec_from_file_location('frozen_a8_trustgtr',
            ROOT / 'runs' / FROZEN_RUNS[0] / 'untracked_code/models/TrustGTR.py')
        frozen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(frozen)
        cases = [historical_options(gate_descriptor=descriptor, gate_loss_mode=loss,
                                   historical_router_mode=router)
                 for descriptor in ('feature', 'forecast', 'retrieval')
                 for loss in ('uniform', 'power_weighted', 'counterfactual_risk')
                 for router in ('global_topk', 'per_bank', 'per_bank_evidence')]
        cases += [dict(use_multi_cycle=True, use_reliability_gate=True, memory_mode=mode,
                       gate_descriptor=descriptor, gate_loss_mode=loss)
                  for mode in ('independent', 'base_residual', 'factorized_residual')
                  for descriptor in ('feature', 'forecast')
                  for loss in ('uniform', 'power_weighted', 'counterfactual_risk')]
        cases += [historical_options(use_regime_contrast=True, gate_loss_mode='power_weighted')]
        for index, options in enumerate(cases):
            with self.subTest(options=options):
                revin = index % 2
                torch.manual_seed(2024)
                old = self.candidate(options, cls=frozen.Model, revin=revin)
                initial_rng = torch.get_rng_state().clone()
                torch.manual_seed(2024)
                new = self.candidate({**options, 'gate_calibration_context': 'train'}, revin=revin)
                self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
                self.assertEqual(set(old.state_dict()), set(new.state_dict()))
                for key in old.state_dict():
                    self.assertTrue(torch.equal(old.state_dict()[key], new.state_dict()[key]), key)
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
                self.assert_gradients_equal(old, new)
                self.assertEqual(old.diagnostics, new.diagnostics)
                old.eval(); new.eval()
                self.assertTrue(torch.equal(old(self.x, self.origins), new(self.x, self.origins)))

    def test_inference_switch_leaves_main_output_gradients_and_rng_exact(self):
        for revin in (0, 1):
            for descriptor in ('feature', 'forecast', 'retrieval'):
                with self.subTest(revin=revin, descriptor=descriptor):
                    torch.manual_seed(2024)
                    old = self.candidate(historical_options(gate_descriptor=descriptor), revin=revin)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(2024)
                    new = self.candidate(historical_options(gate_descriptor=descriptor,
                        gate_calibration_context='inference'), revin=revin)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
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
                    self.assert_gradients_equal(old, new)
                    self.assertIn('inference_calibration', new._retrieval_context)
                    self.assertNotIn('gate_calibration', new._retrieval_context)
                    self.assertEqual(old.diagnostics, new.diagnostics)

    def test_auxiliary_experts_and_gate_exact_actual_eval_inference(self):
        for revin in (0, 1):
            for individual in (0, 1):
                for descriptor in ('feature', 'forecast', 'retrieval'):
                    with self.subTest(revin=revin, individual=individual, descriptor=descriptor):
                        model = self.candidate(historical_options(gate_descriptor=descriptor,
                            gate_calibration_context='inference'), revin=revin, individual=individual)
                        model(self.x, self.origins)
                        cache = model._retrieval_context
                        captured = []
                        handle = model.gate.register_forward_pre_hook(
                            lambda module, inputs: captured.append(inputs[0].detach().clone()))
                        model.auxiliary_losses(self.target)
                        aux = cache['gate_calibration']
                        infer_descriptor = captured[0]
                        model.eval()
                        with torch.no_grad():
                            predicted = model(self.x, self.origins)
                            baseline = Official.forward(model, self.x, self.origins)
                        handle.remove()
                        self.assertTrue(torch.equal(infer_descriptor, captured[1]))
                        self.assertTrue(torch.equal(aux['local'], baseline))
                        inputs = cache['inference_calibration']
                        correction = (inputs['retrieved'] - aux['local']) * inputs['valid']
                        self.assertTrue(torch.equal(aux['delta'], inputs['strength'] * correction))
                        # Preserve the official forward's multiplication order;
                        # g*(s*d) and (s*g)*d can differ by a floating-point ULP.
                        reconstructed = aux['local'] + inputs['strength'] * aux['gate'] * correction
                        self.assertTrue(torch.equal(reconstructed, predicted))
                        self.assertFalse(aux['local'].requires_grad)
                        self.assertFalse(aux['delta'].requires_grad)
                        self.assertTrue(aux['gate'].requires_grad)
                        self.assertIsNone(model._retrieval_context)
                        self.assertEqual(model.auxiliary_losses(self.target), {})

    def test_auxiliary_gradient_only_gate_and_no_input_target_or_expert_gradient(self):
        for mode in ('uniform', 'power_weighted', 'counterfactual_risk'):
            with self.subTest(mode=mode):
                model = self.candidate(historical_options(gate_calibration_context='inference',
                                                         gate_loss_mode=mode))
                x = self.x.clone().requires_grad_()
                target = self.target.clone().requires_grad_()
                model(x, self.origins)
                model.auxiliary_losses(target)['gate'].backward()
                gate_gradient = 0.
                for name, parameter in model.named_parameters():
                    if name.startswith('gate.'):
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        gate_gradient += float(parameter.grad.abs().sum())
                    else:
                        self.assertIsNone(parameter.grad, name)
                self.assertGreater(gate_gradient, 0.)
                self.assertIsNone(x.grad)
                self.assertIsNone(target.grad)

    def test_mixed_module_modes_rng_and_main_cache_preserved_by_aux(self):
        model = self.candidate(historical_options(gate_calibration_context='inference',
                                                 use_regime_contrast=True))
        model.GTR.linear.eval()
        model.output_proj[0].eval()
        model(self.x, self.origins)
        cache = model._retrieval_context
        original_state = cache['state']
        modes = {name: module.training for name, module in model.named_modules()}
        rng = torch.get_rng_state().clone()
        original_gate, original_local, original_delta = cache['gate'], cache['local'], cache['delta']
        losses = model.auxiliary_losses(self.target)
        self.assertEqual(set(losses), {'gate', 'contrast'})
        self.assertIs(cache['state'], original_state)
        self.assertIs(cache['gate'], original_gate)
        self.assertIs(cache['local'], original_local)
        self.assertIs(cache['delta'], original_delta)
        self.assertIs(model._retrieval_context, cache)
        self.assertEqual(modes, {name: module.training for name, module in model.named_modules()})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        gate_calls = []
        handle = model.gate.register_forward_hook(lambda *args: gate_calls.append(True))
        repeat = model.auxiliary_losses(self.target)
        handle.remove()
        self.assertTrue(torch.equal(losses['gate'], repeat['gate']))
        self.assertFalse(gate_calls, 'Repeated aux calls should reuse the current context')

    def test_joint_loss_keeps_all_non_gate_parameter_gradients_exact(self):
        for mode in ('uniform', 'power_weighted', 'counterfactual_risk'):
            with self.subTest(mode=mode):
                opts = historical_options(gate_loss_mode=mode, use_regime_contrast=True)
                torch.manual_seed(2024)
                old = self.candidate(opts)
                torch.manual_seed(2024)
                new = self.candidate({**opts, 'gate_calibration_context': 'inference'})
                torch.manual_seed(53)
                ForecastLoss(old)(old(self.x, self.origins), self.target).backward()
                after = torch.get_rng_state().clone()
                torch.manual_seed(53)
                ForecastLoss(new)(new(self.x, self.origins), self.target).backward()
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                for (name, left), (_, right) in zip(old.named_parameters(), new.named_parameters()):
                    if not name.startswith('gate.'):
                        self.assertTrue(torch.equal(left.grad, right.grad), name)

    def test_zero_gate_recovers_current_branch_and_restores_modes(self):
        model = self.candidate(historical_options(gate_calibration_context='inference'))
        with torch.no_grad():
            model.gate[-1].weight.zero_()
            model.gate[-1].bias.fill_(-1000)
        model.GTR.linear.eval()
        modes = {name: module.training for name, module in model.named_modules()}
        output = model(self.x, self.origins)
        cache = model._retrieval_context
        self.assertTrue(torch.equal(cache['gate'], torch.zeros_like(cache['gate'])))
        self.assertTrue(torch.equal(output, cache['local']))
        model.auxiliary_losses(self.target)
        self.assertTrue(torch.equal(cache['gate_calibration']['gate'], torch.zeros_like(cache['gate'])))
        self.assertEqual(modes, {name: module.training for name, module in model.named_modules()})

    def test_error_restores_flags_rng_and_main_cache(self):
        model = self.candidate(historical_options(gate_calibration_context='inference'))
        model.GTR.linear.eval()
        model(self.x, self.origins)
        cache = model._retrieval_context
        modes = {name: module.training for name, module in model.named_modules()}
        rng = torch.get_rng_state().clone()
        with patch.object(Official, 'forward', side_effect=RuntimeError('simulated expert failure')):
            with self.assertRaisesRegex(RuntimeError, 'simulated expert failure'):
                model.auxiliary_losses(self.target)
        self.assertEqual(modes, {name: module.training for name, module in model.named_modules()})
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(model._retrieval_context, cache)
        self.assertNotIn('gate_calibration', cache)

    def test_empty_candidates_and_zero_strength_have_zero_aux_gradient(self):
        for mode in ('uniform', 'power_weighted', 'counterfactual_risk'):
            for empty in (True, False):
                with self.subTest(mode=mode, empty=empty):
                    model = self.candidate(historical_options(gate_calibration_context='inference',
                                                             gate_loss_mode=mode))
                    if empty:
                        origins = torch.tensor([24, 25])
                        x = self.x[:2]
                    else:
                        origins, x = self.origins, self.x
                        with torch.no_grad():
                            model.historical_strength.zero_()
                    output = model(x, origins)
                    loss = model.auxiliary_losses(self.target[:len(origins)])['gate']
                    self.assertEqual(float(loss), 0.)
                    self.assertTrue(torch.isfinite(output).all())
                    loss.backward()
                    for parameter in model.gate.parameters():
                        self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter)))
                    self.assertIsNone(model.historical_strength.grad)

    def test_eval_zero_weight_and_disabled_gate_never_repeat_expert(self):
        for updates in ({'gate_loss_weight': 0}, {'use_reliability_gate': False}, {}):
            with self.subTest(updates=updates):
                model = self.candidate(historical_options(gate_calibration_context='inference', **updates))
                if not updates:
                    model.eval()
                with patch.object(Official, 'forward', autospec=True,
                                  side_effect=Official.forward) as official:
                    output = model(self.x, self.origins)
                    ForecastLoss(model)(output, self.target)
                    self.assertEqual(official.call_count, 1)
                if not updates:
                    self.assertIsNone(model._retrieval_context)
                else:
                    self.assertNotIn('inference_calibration', model._retrieval_context)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
