"""Switchable multi-cycle routing, benefit-calibrated gate and regime contrast."""
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from models.GTR import Model as OfficialModel
from models.HistoricalMemory import HistoricalMemory


DEFAULTS = dict(use_multi_cycle=False, use_reliability_gate=False,
                use_regime_contrast=False, periods=None, state_dim=16,
                gate_initial_probability=0.9, gate_loss_weight=0.05,
                contrast_loss_weight=0.01, contrast_temperature=0.1,
                positive_max_distance=1.0, regime_distance_gap=0.25,
                preserve_initialization_rng=False, memory_mode='independent',
                gate_descriptor='feature', detach_gate_descriptor=False,
                gate_loss_mode='uniform', gate_calibration_context='train', historical_topk=3,
                historical_candidate_cap=192, historical_blend_bound=0.2,
                historical_strength_coordinates='raw', historical_router_mode='global_topk',
                contrast_metric='cosine')


def resolved_options(options, cycle):
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown TrustGTR options: ' + ', '.join(sorted(unknown)))
    result = {**DEFAULTS, **options}
    result['periods'] = result['periods'] or [cycle, cycle * 7, cycle * 30]
    if result['periods'][0] != cycle or len(set(result['periods'])) != len(result['periods']):
        raise ValueError('Periods must be unique and start with the official base cycle')
    if any(not isinstance(p, int) or p < 1 for p in result['periods']):
        raise ValueError('Periods must be positive integers')
    if result['use_regime_contrast'] and not result['use_multi_cycle']:
        raise ValueError('Regime contrast must train the actual multi-cycle routing encoder')
    if not 0 < result['gate_initial_probability'] < 1 or result['contrast_temperature'] <= 0:
        raise ValueError('Invalid gate probability or contrast temperature')
    if result['memory_mode'] not in ('independent', 'base_residual', 'factorized_residual', 'historical'):
        raise ValueError('Unknown memory mode')
    if result['memory_mode'] != 'independent' and any(period % cycle for period in result['periods']):
        raise ValueError('Centered residual periods must be multiples of the base cycle')
    if result['gate_descriptor'] not in ('feature', 'forecast', 'retrieval') or result['gate_loss_mode'] not in ('uniform', 'power_weighted', 'counterfactual_risk'):
        raise ValueError('Unknown gate descriptor or calibration loss mode')
    if result['gate_descriptor'] == 'retrieval' and result['memory_mode'] != 'historical':
        raise ValueError('Retrieval reliability descriptors require historical memory')
    if result['gate_calibration_context'] not in ('train', 'inference'):
        raise ValueError('Unknown gate calibration context')
    if (result['gate_calibration_context'] == 'inference' and result['use_reliability_gate']
            and result['memory_mode'] != 'historical'):
        raise ValueError('Inference gate calibration requires historical memory')
    if result['memory_mode'] == 'historical':
        if not result['use_multi_cycle']:
            raise ValueError('Historical retrieval requires the multi-cycle module')
        if (result['historical_topk'] < 1 or result['historical_candidate_cap'] < result['historical_topk']
                or not 0 < result['historical_blend_bound'] <= 1):
            raise ValueError('Invalid historical retrieval capacity or blend bound')
    if result['historical_strength_coordinates'] not in ('raw', 'bounded'):
        raise ValueError('Unknown strength coordinates')
    if result['historical_router_mode'] not in ('global_topk', 'per_bank', 'per_bank_evidence'):
        raise ValueError('Unknown historical routing mode')
    if result['contrast_metric'] not in ('cosine', 'euclidean'):
        raise ValueError('Unknown contrastive metric')
    return result


def state_features(x):
    """Only observed input; x is B,C,T in dataset-standardized units."""
    mean = x.mean(-1)
    std = x.std(-1, unbiased=False)
    quarter = max(1, x.shape[-1] // 4)
    first, last = x[..., :quarter].mean(-1), x[..., -quarter:].mean(-1)
    axis = torch.linspace(-1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
    slope = ((x - mean.unsqueeze(-1)) * axis).sum(-1) / axis.square().sum().clamp_min(1e-6)
    return torch.stack((mean, torch.log1p(std), first, last, x[..., -1],
                        last - first, slope), -1).clamp(-10, 10)


def utility_oracle(local, delta, target):
    """Exact best scalar mixture per sample/channel; training targets only."""
    power = delta.square().sum(1, keepdim=True)
    optimum = (((target - local) * delta).sum(1, keepdim=True) /
               power.clamp_min(1e-8)).clamp(0, 1)
    return optimum.detach(), power > 1e-6


def counterfactual_risk_loss(local, delta, gate, target):
    """Normalized excess forecast risk; supervised labels train only the gate.

    For e=target-local and d=delta, the numerator is
    ||e-g*d||^2-||e||^2. Dividing by ||d||^2+||e||^2 gives a lower
    bound of -1 without clipping each sample's optimal gate first.
    """
    error = (target - local).detach()
    correction = delta.detach()
    power = correction.square().sum(1, keepdim=True)
    utility = (correction * error).sum(1, keepdim=True)
    error_energy = error.square().sum(1, keepdim=True)
    meaningful = power > 1e-6
    risk = (gate.square() * power - 2 * gate * utility) / (power + error_energy).clamp_min(1e-8)
    return risk[meaningful].mean() if meaningful.any() else gate.sum() * 0


class Model(OfficialModel):
    def __init__(self, configs):
        # Initialize the original parameters first. All switches off creates
        # exactly the original parameter set and executes its original forward.
        super().__init__(configs)
        self.options = resolved_options(getattr(configs, 'trustgtr', {}), self.cycle_len)
        opts = self.options
        self.extra_memory = nn.ParameterList()
        self.state_encoder = self.cycle_router = self.gate = None
        self.historical_memory = None
        with (torch.random.fork_rng(devices=[]) if opts['preserve_initialization_rng'] else nullcontext()):
            self.initialize_new_modules(opts)
        self._retrieval_context = None
        self.diagnostics = {'gate_sum': 0., 'gate_count': 0, 'contrast_calls': 0,
                            'contrast_valid_anchors': 0, 'contrast_total_anchors': 0,
                            'cycle_weight_sum': [0.] * len(opts['periods']), 'cycle_weight_count': 0}

    def initialize_new_modules(self, opts):
        if opts['memory_mode'] == 'historical':
            self.historical_memory = HistoricalMemory(self.seq_len, self.pred_len,
                self.cycle_len, self.enc_in, opts['periods'], state_dim=opts['state_dim'],
                topk=opts['historical_topk'], candidate_cap=opts['historical_candidate_cap'],
                router_mode=opts['historical_router_mode'],
                collect_reliability=opts['use_reliability_gate'] and opts['gate_descriptor'] == 'retrieval')
            # Start from the exact original forecast. This controls a small
            # signed correction; the backbone and its training budget stay intact.
            self.historical_strength = nn.Parameter(torch.zeros(()))
        elif opts['use_multi_cycle']:
            self.extra_memory.extend(nn.Parameter(torch.zeros(
                (period // self.cycle_len, self.enc_in, 3) if opts['memory_mode'] == 'factorized_residual'
                else (period, self.enc_in))) for period in opts['periods'][1:])
            dim = opts['state_dim']
            self.state_encoder = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim))
            self.cycle_router = nn.Linear(dim, len(opts['periods']))
            nn.init.zeros_(self.cycle_router.bias)
            with torch.no_grad():
                self.cycle_router.bias[0] = 2.0
        if opts['use_reliability_gate']:
            gate_inputs = 18 if opts['gate_descriptor'] == 'retrieval' else 10
            self.gate = nn.Sequential(nn.Linear(gate_inputs, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias,
                              math.log(opts['gate_initial_probability'] / (1 - opts['gate_initial_probability'])))

    def retrieve(self, raw_input, origin):
        offset = torch.arange(self.seq_len, device=origin.device).view(1, -1)
        if not self.options['use_multi_cycle']:
            indices = (origin.view(-1, 1) + offset) % self.cycle_len
            return self.Q[indices].permute(0, 2, 1), None
        state = self.state_encoder(state_features(raw_input))
        weights = self.cycle_router(state).softmax(-1)
        totals = weights.detach().sum((0, 1)).tolist()
        self.diagnostics['cycle_weight_sum'] = [previous + current for previous, current in
                                              zip(self.diagnostics['cycle_weight_sum'], totals)]
        self.diagnostics['cycle_weight_count'] += weights.shape[0] * weights.shape[1]
        contexts = []
        for bank_index, (memory, period) in enumerate(zip([self.Q, *self.extra_memory], self.options['periods'])):
            if bank_index and self.options['memory_mode'] == 'factorized_residual':
                coefficients = memory - memory.mean(0, keepdim=True)
                axis = torch.linspace(-1, 1, self.cycle_len, device=memory.device, dtype=memory.dtype).view(1, -1, 1)
                # Each cycle retrieves its level/trend/amplitude deviations.
                # Sharing the base phase profile limits sparse long-period overfit.
                memory = (coefficients[..., 0].unsqueeze(1) +
                          coefficients[..., 1].unsqueeze(1) * axis +
                          coefficients[..., 2].unsqueeze(1) * self.Q.unsqueeze(0)).reshape(period, self.enc_in)
            elif bank_index and self.options['memory_mode'] == 'base_residual':
                grouped = memory.reshape(period // self.cycle_len, self.cycle_len, self.enc_in)
                memory = (grouped - grouped.mean(0, keepdim=True)).reshape(period, self.enc_in)
            indices = (origin.view(-1, 1) + offset) % period
            contexts.append(memory[indices].permute(0, 2, 1))
        if self.options['memory_mode'] != 'independent':
            # The short-cycle gradient is preserved. Longer banks represent
            # deviations within a base phase, not copies of the base pattern.
            query = contexts[0] + (torch.stack(contexts[1:], -1) * weights[..., 1:].unsqueeze(2)).sum(-1)
        else:
            query = (torch.stack(contexts, -1) * weights.unsqueeze(2)).sum(-1)
        return query, state

    def split_convolution(self, local, query):
        mapped = self.GTR.linear(query)
        padding = self.GTR.period_len // 2
        if self.GTR.CI:
            local_parts, global_parts = [], []
            for index, conv in enumerate(self.GTR.ds_convs):
                local_parts.append(F.conv1d(local[:, index:index+1], conv.weight[:, :, 0, :], padding=padding))
                global_parts.append(F.conv1d(mapped[:, index:index+1], conv.weight[:, :, 1, :], padding=padding))
            return torch.cat(local_parts, 1), torch.cat(global_parts, 1)
        shape = local.shape
        local_part = F.conv1d(local.reshape(-1, 1, self.seq_len),
                             self.GTR.conv2d.weight[:, :, 0, :], padding=padding)
        global_part = F.conv1d(mapped.reshape(-1, 1, self.seq_len),
                              self.GTR.conv2d.weight[:, :, 1, :], padding=padding)
        return local_part.reshape(shape), global_part.reshape(shape)

    def hidden(self, fused):
        projected = self.input_proj(fused)
        return self.model(projected) + projected

    def forward(self, x, cycle_index):
        opts = self.options
        self._retrieval_context = None
        if opts['memory_mode'] == 'historical':
            return self.forward_historical(x, cycle_index)
        if not (opts['use_multi_cycle'] or opts['use_reliability_gate']):
            return super().forward(x, cycle_index)
        raw_input = x.permute(0, 2, 1)
        if self.use_revin:
            seq_mean = x.mean(1, keepdim=True)
            seq_var = x.var(1, keepdim=True) + 1e-5
            x = (x - seq_mean) / seq_var.sqrt()
        local = x.permute(0, 2, 1)
        query, state = self.retrieve(raw_input, cycle_index)
        cache = dict(state=state, origin=cycle_index,
                     history_mean=raw_input.mean(-1).detach(),
                     history_std=raw_input.std(-1, unbiased=False).detach().clamp_min(1e-3))
        if opts['use_reliability_gate']:
            local_part, global_part = self.split_convolution(local, query)
            # Both branches share their dropout masks: the oracle measures
            # the global contribution rather than unrelated dropout noise.
            fusion_mask = F.dropout(torch.ones_like(local_part), p=self.GTR.dropout.p, training=self.training)
            local_hidden = self.hidden(local + local_part * fusion_mask)
            global_hidden = self.hidden(local + (local_part + global_part) * fusion_mask)
            head_mask = F.dropout(torch.ones_like(global_hidden), p=self.dropout, training=self.training)
            local_prediction = self.output_proj[1](local_hidden * head_mask).permute(0, 2, 1)
            full_prediction = self.output_proj[1](global_hidden * head_mask).permute(0, 2, 1)
            if opts['gate_descriptor'] == 'forecast':
                left, right = local_prediction.permute(0, 2, 1), full_prediction.permute(0, 2, 1)
                if opts['detach_gate_descriptor']:
                    left, right = left.detach(), right.detach()
                agreement = F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)
                scale = left.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
                norm_ratio = torch.log1p((right - left).square().mean(-1, keepdim=True).sqrt() / scale)
                mismatch = (right - left).abs().mean(-1, keepdim=True) / scale
            else:
                compared = global_part.detach() if opts['detach_gate_descriptor'] else global_part
                agreement = F.cosine_similarity(local, compared, dim=-1).unsqueeze(-1)
                scale = local.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
                norm_ratio = torch.log1p(compared.square().mean(-1, keepdim=True).sqrt() / scale)
                mismatch = (local - compared).abs().mean(-1, keepdim=True) / scale
            descriptor = torch.cat((state_features(raw_input), agreement, norm_ratio, mismatch), -1).clamp(-10, 10)
            gate = self.gate(descriptor).sigmoid().permute(0, 2, 1)
            output = local_prediction + gate * (full_prediction - local_prediction)
            if self.use_revin:
                local_prediction = local_prediction * seq_var.sqrt() + seq_mean
                full_prediction = full_prediction * seq_var.sqrt() + seq_mean
            cache.update(gate=gate, local=local_prediction.detach(),
                         delta=(full_prediction - local_prediction).detach())
            self.diagnostics['gate_sum'] += gate.detach().sum().item()
            self.diagnostics['gate_count'] += gate.numel()
        else:
            fused = local + self.GTR(local, query)
            output = self.output_proj(self.hidden(fused)).permute(0, 2, 1)
        if self.use_revin:
            output = output * seq_var.sqrt() + seq_mean
        if self.training:
            self._retrieval_context = cache
        return output

    def forward_historical(self, x, origin):
        """Retrieve observed historical continuations and correct the original GTR."""
        baseline = super().forward(x, origin)
        raw_input = x.permute(0, 2, 1)
        retrieved, state = self.historical_memory(raw_input, origin)
        valid = state['valid'].view(-1, 1, 1)
        correction = (retrieved - baseline) * valid
        bound = self.options['historical_blend_bound']
        coordinate_scale = bound if self.options['historical_strength_coordinates'] == 'bounded' else 1.0
        strength = bound * (self.historical_strength / coordinate_scale).tanh()
        cache = dict(state=state, origin=origin, history_mean=state['query_mean'],
                     history_std=state['query_scale'])
        if self.options['use_reliability_gate']:
            # The gate sees only predictions and observed input. Detachment
            # prevents the calibration descriptor from reshaping the experts.
            left = baseline.detach().permute(0, 2, 1)
            right = retrieved.detach().permute(0, 2, 1)
            agreement = F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)
            scale = left.square().mean(-1, keepdim=True).sqrt().clamp_min(0.1)
            norm_ratio = torch.log1p((right - left).square().mean(-1, keepdim=True).sqrt() / scale)
            mismatch = (right - left).abs().mean(-1, keepdim=True) / scale
            descriptor = torch.cat((state_features(raw_input), agreement, norm_ratio, mismatch), -1).clamp(-10, 10)
            if self.options['gate_descriptor'] == 'retrieval':
                descriptor = torch.cat((descriptor, state['reliability_features']), -1).detach().clamp(-10, 10)
            gate = self.gate(descriptor).sigmoid().permute(0, 2, 1) * valid
            output = baseline + strength * gate * correction
            cache.update(gate=gate, local=baseline.detach(),
                         delta=(strength * correction).detach())
            if (self.training and self.options['gate_calibration_context'] == 'inference'
                    and self.options['gate_loss_weight'] != 0):
                # Lazy auxiliary context: never repeat the experts in the main
                # forward, or replace its official dropout/gate gradients.
                cache['inference_calibration'] = dict(input=x.detach(),
                    retrieved=retrieved.detach(), valid=valid.detach(),
                    strength=strength.detach())
            self.diagnostics['gate_sum'] += gate.detach().sum().item()
            self.diagnostics['gate_count'] += gate.numel()
        else:
            output = baseline + strength * correction
        totals = state['cycle_weights'].detach().sum((0, 1)).tolist()
        self.diagnostics['cycle_weight_sum'] = [previous + current for previous, current in
                                              zip(self.diagnostics['cycle_weight_sum'], totals)]
        self.diagnostics['cycle_weight_count'] += x.shape[0] * x.shape[2]
        self.diagnostics['historical_strength'] = float(strength.detach())
        self.diagnostics['historical_valid_fraction_last_batch'] = float(state['valid'].float().mean())
        if self.training:
            self._retrieval_context = cache
        return output

    def inference_calibration_context(self, cache):
        """Calibrate with current inference experts; gradients train only gate.

        The retrieval encoder has no dropout, so its same-pass result is reused.
        The original GTR branch is evaluated without dropout under no_grad. All
        module flags, including mixed train/eval flags, are restored on errors.
        Training future labels are deliberately absent from this context.
        """
        if 'gate_calibration' in cache:
            return cache['gate_calibration']
        inputs = cache['inference_calibration']
        modes = [(module, module.training) for module in self.modules()]
        try:
            for module, _ in modes:
                module.training = False
            with torch.no_grad():
                baseline = super().forward(inputs['input'], cache['origin'])
                retrieved = inputs['retrieved']
                left = baseline.permute(0, 2, 1)
                right = retrieved.permute(0, 2, 1)
                agreement = F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)
                scale = left.square().mean(-1, keepdim=True).sqrt().clamp_min(0.1)
                norm_ratio = torch.log1p((right - left).square().mean(-1, keepdim=True).sqrt() / scale)
                mismatch = (right - left).abs().mean(-1, keepdim=True) / scale
                descriptor = torch.cat((state_features(inputs['input'].permute(0, 2, 1)),
                                        agreement, norm_ratio, mismatch), -1).clamp(-10, 10)
                if self.options['gate_descriptor'] == 'retrieval':
                    descriptor = torch.cat((descriptor, cache['state']['reliability_features']), -1).clamp(-10, 10)
                delta = inputs['strength'] * ((retrieved - baseline) * inputs['valid'])
        finally:
            for module, training in modes:
                module.training = training
        # No new parameters or stochastic operations: the gate has no dropout.
        gate = self.gate(descriptor).sigmoid().permute(0, 2, 1) * inputs['valid']
        result = dict(local=baseline, delta=delta, gate=gate)
        cache['gate_calibration'] = result
        return result

    def historical_contrastive_loss(self, target, cache):
        """Same-phase hard negatives from the fixed, causal training bank."""
        state = cache['state']
        index = state['base_candidate_index']
        candidate_mask = state['base_candidate_mask'] & (
            cache['origin'][:, None] - state['base_candidate_origin'] >= self.seq_len + self.pred_len)
        # Static regimes use only completed training continuations. Query
        # future labels are used exclusively here, during supervised training.
        bank_regime = state['regime'][index.clamp_min(0)].permute(0, 2, 1, 3).clamp(-5, 5)
        future = target.detach().permute(0, 2, 1)
        quarter = max(1, future.shape[-1] // 4)
        level = (future.mean(-1) - cache['history_mean']) / cache['history_std']
        trend = (future[..., -quarter:].mean(-1) - future[..., :quarter].mean(-1)) / cache['history_std']
        regime = torch.stack((level, trend), -1).clamp(-5, 5)
        distance = (bank_regime - regime.unsqueeze(2)).square().sum(-1).sqrt()
        mask = candidate_mask.unsqueeze(1)
        near_distance, positive = distance.masked_fill(~mask, float('inf')).min(-1)
        far_distance, negative = distance.masked_fill(~mask, -float('inf')).max(-1)
        usable = ((candidate_mask.sum(-1) >= 2).unsqueeze(1) &
                  (near_distance <= self.options['positive_max_distance']) &
                  (far_distance - near_distance >= self.options['regime_distance_gap']))
        self.diagnostics['contrast_calls'] += 1
        self.diagnostics['contrast_valid_anchors'] += usable.sum().item()
        self.diagnostics['contrast_total_anchors'] += usable.numel()
        query = state['query_embedding']
        candidates = state['base_candidate_key_embeddings']
        if not usable.any():
            return query.sum() * 0 + candidates.sum() * 0
        if self.options['contrast_metric'] == 'euclidean':
            similarity = -(query.unsqueeze(2) - candidates).square().sum(-1)
        else:
            similarity = (F.normalize(query, dim=-1).unsqueeze(2) *
                          F.normalize(candidates, dim=-1)).sum(-1)
        similarity = similarity / self.options['contrast_temperature']
        pos_logits = similarity.gather(-1, positive.unsqueeze(-1)).squeeze(-1)
        neg_logits = similarity.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
        logits = torch.stack((pos_logits, neg_logits), -1)[usable]
        return F.cross_entropy(logits, torch.zeros(len(logits), device=logits.device, dtype=torch.long))

    def contrastive_loss(self, target, cache):
        state = cache['state']
        embeddings = F.normalize(state.permute(1, 0, 2), dim=-1)
        origin = cache['origin']
        same_phase = origin[:, None].remainder(self.cycle_len) == origin[None, :].remainder(self.cycle_len)
        disjoint = (origin[:, None] - origin[None, :]).abs() >= self.seq_len + self.pred_len
        candidates = same_phase & disjoint
        future = target.detach().permute(0, 2, 1)
        quarter = max(1, future.shape[-1] // 4)
        level = (future.mean(-1) - cache['history_mean']) / cache['history_std']
        trend = (future[..., -quarter:].mean(-1) - future[..., :quarter].mean(-1)) / cache['history_std']
        regime = torch.stack((level, trend), -1).clamp(-5, 5).permute(1, 0, 2)
        distance = torch.cdist(regime, regime)
        nearest = distance.masked_fill(~candidates, float('inf'))
        farthest = distance.masked_fill(~candidates, -float('inf'))
        near_distance, positive = nearest.min(-1)
        far_distance, negative = farthest.max(-1)
        usable = ((candidates.sum(-1) >= 2).unsqueeze(0) &
                  (near_distance <= self.options['positive_max_distance']) &
                  (far_distance - near_distance >= self.options['regime_distance_gap']))
        self.diagnostics['contrast_calls'] += 1
        self.diagnostics['contrast_valid_anchors'] += usable.sum().item()
        self.diagnostics['contrast_total_anchors'] += usable.numel()
        if not usable.any():
            return state.sum() * 0
        similarity = torch.bmm(embeddings, embeddings.transpose(1, 2)) / self.options['contrast_temperature']
        pos_logits = similarity.gather(-1, positive.unsqueeze(-1)).squeeze(-1)
        neg_logits = similarity.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
        logits = torch.stack((pos_logits, neg_logits), -1)[usable]
        return F.cross_entropy(logits, torch.zeros(len(logits), device=logits.device, dtype=torch.long))

    def auxiliary_losses(self, target):
        cache = self._retrieval_context
        if not self.training or cache is None:
            return {}
        losses = {}
        if self.options['use_reliability_gate']:
            gate_cache = (self.inference_calibration_context(cache)
                          if 'inference_calibration' in cache else cache)
            if self.options['gate_loss_mode'] == 'counterfactual_risk':
                losses['gate'] = counterfactual_risk_loss(gate_cache['local'], gate_cache['delta'], gate_cache['gate'], target)
            else:
                optimum, meaningful = utility_oracle(gate_cache['local'], gate_cache['delta'], target)
                if self.options['gate_loss_mode'] == 'power_weighted':
                    power = gate_cache['delta'].square().sum(1, keepdim=True)
                    weights = (power / power.mean().clamp_min(1e-8)).clamp(max=10) * meaningful
                    losses['gate'] = (weights * (gate_cache['gate'] - optimum).square()).sum() / weights.sum().clamp_min(1e-8)
                else:
                    losses['gate'] = (F.mse_loss(gate_cache['gate'][meaningful], optimum[meaningful])
                                      if meaningful.any() else gate_cache['gate'].sum() * 0)
        if self.options['use_regime_contrast']:
            losses['contrast'] = (self.historical_contrastive_loss(target, cache)
                                  if self.options['memory_mode'] == 'historical'
                                  else self.contrastive_loss(target, cache))
        return losses


class ForecastLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Criterion must not register or duplicate the model parameters.
        object.__setattr__(self, 'forecast_model', model)

    def forward(self, prediction, target):
        model = self.forecast_model
        forecast = F.mse_loss(prediction, target)
        result = forecast
        for name, loss in model.auxiliary_losses(target).items():
            weight = model.options['gate_loss_weight' if name == 'gate' else 'contrast_loss_weight']
            result = result + weight * loss
        return result
