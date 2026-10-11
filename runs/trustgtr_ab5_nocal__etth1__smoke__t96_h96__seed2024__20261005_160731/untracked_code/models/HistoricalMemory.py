"""Learned, causal retrieval from an explicitly attached training-only series.

This bank is static: forward never inserts queries or targets. The caller must
verify that ``attach_training_series`` receives the fixed normalized train split;
an unlabelled NumPy array cannot itself prove its split provenance.
"""
import hashlib
import math

import numpy as np
import torch
from torch import nn


def _descriptors(history, floor=0.1, bins=12):
    """The diagnostic's observed-history descriptor, in N,C,T order."""
    mean = history.mean(-1)
    scale = np.maximum(history.std(-1, ddof=0), floor)
    normalized = (history - mean[..., None]) / scale[..., None]
    quarter = max(1, history.shape[-1] // 4)
    trend = (history[..., -quarter:].mean(-1) - history[..., :quarter].mean(-1)) / scale
    stats = np.stack((mean, np.log(scale), trend), -1)
    shape = normalized.reshape(*normalized.shape[:-1], bins, history.shape[-1] // bins).mean(-1)
    return [value.astype('float32') for value in (mean, scale, stats, shape)]


class HistoricalMemory(nn.Module):
    """Shared learned keys plus priors over short/middle/long historical ages.

    ``periods`` retains the proposed configuration's provenance and determines
    the number of age banks. Every candidate matches ``origin % cycle``; this
    implementation does not apply separate modulo-168/720-style phase rules.
    ``global_topk`` reweights global top-k; ``per_bank`` first retrieves from
    every valid bank and then continuously mixes their independent predictions.
    """
    def __init__(self, seq_len, pred_len, cycle, channels, periods,
                 state_dim=16, topk=3, candidate_cap=192, router_mode='global_topk',
                 collect_reliability=False):
        super().__init__()
        if min(seq_len, pred_len, cycle, channels, state_dim, topk, candidate_cap) < 1:
            raise ValueError('Historical-memory dimensions and budgets must be positive')
        if seq_len % 12:
            raise ValueError('The fixed 12-bin diagnostic descriptor requires seq_len divisible by 12')
        periods = list(periods)
        if not periods or periods[0] != cycle or len(set(periods)) != len(periods):
            raise ValueError('Unique periods must start with the official base cycle')
        if any(not isinstance(period, int) or period < 1 for period in periods):
            raise ValueError('Periods must be positive integers')
        if router_mode not in ('global_topk', 'per_bank'):
            raise ValueError('router_mode must be global_topk or per_bank')
        if not isinstance(collect_reliability, bool):
            raise ValueError('collect_reliability must be a bool')
        self.seq_len, self.pred_len, self.cycle = seq_len, pred_len, cycle
        self.channels, self.periods = channels, periods
        self.topk, self.candidate_cap = topk, candidate_cap
        self.router_mode = router_mode
        self.collect_reliability = collect_reliability
        self.key_encoder = nn.Sequential(nn.Linear(15, state_dim), nn.GELU(), nn.Linear(state_dim, 15))
        nn.init.zeros_(self.key_encoder[-1].weight)
        nn.init.zeros_(self.key_encoder[-1].bias)
        self.cycle_router = nn.Linear(15, len(periods))
        nn.init.zeros_(self.cycle_router.weight)
        nn.init.zeros_(self.cycle_router.bias)
        shapes = dict(series=(0, channels), history_keys=(0, channels, 15),
                      history_mean=(0, channels), history_scale=(0, channels),
                      regime=(0, channels, 2), stat_center=(channels, 3),
                      stat_scale=(channels, 3))
        for name, shape in shapes.items():
            self.register_buffer(name, torch.empty(shape), persistent=False)
        self.register_buffer('origins', torch.empty(0, dtype=torch.long), persistent=False)
        self.metadata = {}
        self.diagnostics = dict(calls=0, empty_queries=0,
                                candidate_total_by_bank=[0] * len(periods))
        self._count = 0

    def attach_training_series(self, series):
        """Attach only verified train rows, normalized with train statistics.

        Data buffers are excluded from state_dict; reattach the identical series
        before checkpoint inference/resumption and verify ``metadata['sha256']``.
        """
        train = np.asarray(series, dtype='float32')
        if train.ndim != 2 or train.shape[1] != self.channels:
            raise ValueError('Expected the normalized training series with shape N,C')
        if not np.isfinite(train).all():
            raise ValueError('Training memory contains non-finite observations')
        count = len(train) - self.seq_len - self.pred_len + 1
        if count < 1:
            raise ValueError('Training memory needs at least one complete history/future pair')
        train = np.ascontiguousarray(train).copy()
        history = np.lib.stride_tricks.sliding_window_view(train, self.seq_len, axis=0)[:count]
        future = np.lib.stride_tricks.sliding_window_view(train, self.pred_len, axis=0)[self.seq_len:self.seq_len + count]
        means, scales, stats, shapes, regimes = [], [], [], [], []
        quarter = max(1, self.pred_len // 4)
        for offset in range(0, count, 512):
            mean, scale, stat, shape = _descriptors(history[offset:offset + 512])
            for destination, value in zip((means, scales, stats, shapes), (mean, scale, stat, shape)):
                destination.append(value)
            values = future[offset:offset + 512]
            level = (values.mean(-1) - mean) / scale
            trend = (values[..., -quarter:].mean(-1) - values[..., :quarter].mean(-1)) / scale
            regimes.append(np.stack((level, trend), -1).astype('float32'))
        mean, scale, stat, shape = [np.concatenate(parts) for parts in (means, scales, stats, shapes)]
        center = stat.mean(0)
        stat_scale = np.maximum(stat.std(0), 0.1)
        keys = np.concatenate(((stat - center) / stat_scale / math.sqrt(3),
                               shape / math.sqrt(12)), -1).astype('float32')
        device = self.stat_center.device
        tensors = dict(series=train, history_keys=keys, history_mean=mean,
                       history_scale=scale, regime=np.concatenate(regimes),
                       stat_center=center, stat_scale=stat_scale)
        for name, array in tensors.items():
            setattr(self, name, torch.from_numpy(np.ascontiguousarray(array)).to(device))
        self.origins = torch.arange(count, device=device, dtype=torch.long) + self.seq_len
        digest = hashlib.sha256(str(train.dtype).encode() + str(train.shape).encode() + train.tobytes()).hexdigest()
        self._count = count
        self.metadata = dict(sha256=digest, train_rows=len(train), channels=self.channels,
                             source_split='train', static_bank=True, candidate_windows=count,
                             first_origin=self.seq_len, last_origin=self.seq_len + count - 1,
                             seq_len=self.seq_len, pred_len=self.pred_len,
                             descriptor='train-scaled mean/log_std/trend + 12 normalized shape bins',
                             history_std_floor=0.1, stat_std_floor=0.1,
                             configured_periods=list(self.periods), phase_cycle=self.cycle,
                             bank_rule='same base phase; deterministic relative-age bands',
                             bank_roles=['short', 'middle', 'long'] if len(self.periods) == 3 else list(range(len(self.periods))),
                             candidate_cap=self.candidate_cap, topk=self.topk,
                             router_mode=self.router_mode,
                             causality='candidate origin j + horizon <= query origin t')
        if self.collect_reliability:
            self.metadata['reliability_features'] = [
                'nearest_key_distance_log1p', 'relative_nearest_distance_gap',
                'selected_weight_entropy', 'selected_future_dispersion_log1p',
                'selected_bank_dispersion_log1p', 'selected_bank_coverage',
                'selected_age_cycles_log1p', 'candidate_count_fraction']
        return dict(self.metadata)

    def encode(self, key):
        return key + self.key_encoder(key)

    def _query_key(self, history):
        mean = history.mean(-1)
        scale = history.std(-1, unbiased=False).clamp_min(0.1)
        normalized = (history - mean.unsqueeze(-1)) / scale.unsqueeze(-1)
        quarter = max(1, self.seq_len // 4)
        trend = (history[..., -quarter:].mean(-1) - history[..., :quarter].mean(-1)) / scale
        stats = torch.stack((mean, scale.log(), trend), -1)
        shape = normalized.reshape(*normalized.shape[:-1], 12, self.seq_len // 12).mean(-1)
        key = torch.cat(((stats - self.stat_center) / self.stat_scale / math.sqrt(3),
                         shape / math.sqrt(12)), -1)
        return self.encode(key), mean, scale

    def _candidates(self, query_origins):
        """Diagnostic-equivalent relative-age terciles, without scanning all rows."""
        rows, bank_rows, bank_counts = [], [], []
        banks = len(self.periods)
        quotas = [self.candidate_cap // banks + int(i < self.candidate_cap % banks) for i in range(banks)]
        for origin in query_origins:
            # Bank origin for index i is seq_len+i; inclusive causal bound.
            stop = min(self._count, int(origin) - self.pred_len - self.seq_len + 1)
            first = (int(origin) - self.seq_len) % self.cycle
            recent_first = np.arange(first, max(0, stop), self.cycle, dtype=np.int64)[::-1]
            selected, provenance, counts = [], [], []
            for bank, (band, quota) in enumerate(zip(np.array_split(recent_first, banks), quotas)):
                if len(band) and quota:
                    positions = np.unique(np.linspace(0, len(band) - 1, min(quota, len(band))).astype(int))
                    selected.append(band[positions])
                    provenance.append(np.full(len(positions), bank, dtype=np.int64))
                    counts.append(len(positions))
                else:
                    counts.append(0)
            rows.append(np.concatenate(selected) if selected else np.empty(0, dtype=np.int64))
            bank_rows.append(np.concatenate(provenance) if provenance else np.empty(0, dtype=np.int64))
            bank_counts.append(counts)
        # One masked sentinel keeps downstream contrastive min/max reductions
        # defined even when every query is earlier than the first reference.
        width = max(1, max((len(row) for row in rows), default=0))
        indices = np.full((len(rows), width), -1, dtype=np.int64)
        provenance = np.full_like(indices, -1)
        for batch, row in enumerate(rows):
            indices[batch, :len(row)] = row
            provenance[batch, :len(row)] = bank_rows[batch]
        return (torch.as_tensor(indices, device=self.series.device),
                torch.as_tensor(provenance, device=self.series.device), bank_counts)

    @torch.no_grad()
    def _reliability_features(self, distances, mask, selected_banks, weights,
                              standardized, selected, origin):
        """Small gate inputs from observed keys and completed historical futures.

        The bank disagreement covers selected banks only; coverage explicitly
        distinguishes one available opinion from agreement across age banks.
        No query targets or differentiable retrieval activations enter the gate.
        """
        batch, channels, width = distances.shape
        nearest = distances.topk(min(2, width), dim=-1, largest=False).values
        valid_first = torch.isfinite(nearest[..., 0])
        first = torch.where(valid_first, nearest[..., 0], 0)
        second = nearest[..., 1] if width > 1 else torch.full_like(first, float('inf'))
        valid_second = torch.isfinite(second)
        second = torch.where(valid_second, second, first)
        gap = (second - first) / (second + first + 1e-6)
        gap = torch.where(valid_first & valid_second, gap, 0)

        weights = weights.detach().masked_fill(selected_banks < 0, 0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        selected_count = (selected_banks >= 0).sum(-1).clamp_min(2).to(weights.dtype)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1) / selected_count.log()
        futures = standardized.detach()
        mean_future = (futures * weights.unsqueeze(-1)).sum(2)
        variance = ((futures - mean_future.unsqueeze(2)).square() *
                    weights.unsqueeze(-1)).sum(2).mean(-1)
        bank_variance = torch.zeros_like(first)
        bank_coverage = torch.zeros_like(first)
        for bank in range(len(self.periods)):
            bank_weights = weights.masked_fill(selected_banks != bank, 0)
            mass = bank_weights.sum(-1)
            bank_mean = ((futures * bank_weights.unsqueeze(-1)).sum(2) /
                         mass.clamp_min(1e-12).unsqueeze(-1))
            bank_variance += mass * (bank_mean - mean_future).square().mean(-1)
            bank_coverage += (mass > 0).to(weights.dtype)
        bank_coverage /= len(self.periods)
        ages = (origin[:, None, None] - self.origins[selected]).to(weights.dtype)
        age_cycles = (weights * ages).sum(-1).clamp_min(0) / self.cycle
        candidate_fraction = (mask.sum(-1).to(weights.dtype) / self.candidate_cap)
        candidate_fraction = candidate_fraction.unsqueeze(1).expand(batch, channels)
        features = torch.stack((first.clamp_min(0).log1p(), gap, entropy,
                                variance.clamp_min(0).sqrt().log1p(),
                                bank_variance.clamp_min(0).sqrt().log1p(),
                                bank_coverage, age_cycles.log1p(), candidate_fraction), -1)
        return features.masked_fill(~valid_first.unsqueeze(-1), 0)

    def _retrieve_bank(self, query, mean, scale, indices, provenance, priors, origin=None):
        batch, channels, _ = query.shape
        mask = indices >= 0
        width = indices.shape[1]
        if not width:
            prediction = mean.unsqueeze(1).expand(batch, self.pred_len, channels)
            embeddings = query.new_empty(batch, channels, 0, 15)
            selected_banks = indices.new_empty(batch, channels, 0)
            weights = query.new_empty(batch, channels, 0)
            reliability = query.new_zeros(batch, channels, 8) if self.collect_reliability else None
            return prediction, embeddings, selected_banks, weights, reliability
        safe = indices.clamp_min(0)
        embeddings = self.encode(self.history_keys[safe].permute(0, 2, 1, 3))
        distances = (embeddings - query.unsqueeze(2)).square().sum(-1)
        distances = distances.masked_fill(~mask.unsqueeze(1), float('inf'))
        reliability_distances = distances.detach() if self.collect_reliability else None
        k = min(self.topk, width)
        if self.router_mode == 'global_topk':
            distances, positions = distances.topk(k, dim=-1, largest=False)
            selected_banks = provenance.clamp_min(0).unsqueeze(1).expand(-1, channels, -1).gather(-1, positions)
            # Preserve the v4 diagnostic-equivalent path and operation order.
            weights = priors.gather(-1, selected_banks) / (distances + 1e-3)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        else:
            all_distances, all_positions, all_weights = [], [], []
            for bank in range(len(self.periods)):
                bank_distances = distances.masked_fill(~(provenance == bank).unsqueeze(1), float('inf'))
                chosen_distances, chosen_positions = bank_distances.topk(k, dim=-1, largest=False)
                within_bank = 1 / (chosen_distances + 1e-3)
                within_bank = within_bank / within_bank.sum(-1, keepdim=True).clamp_min(1e-12)
                all_distances.append(chosen_distances)
                all_positions.append(chosen_positions)
                all_weights.append(within_bank * priors[..., bank:bank + 1])
            distances = torch.cat(all_distances, -1)
            positions = torch.cat(all_positions, -1)
            weights = torch.cat(all_weights, -1)
        selected = safe.unsqueeze(1).expand(-1, channels, -1).gather(-1, positions)
        selected_banks = provenance.clamp_min(0).unsqueeze(1).expand(-1, channels, -1).gather(-1, positions)
        channel = torch.arange(channels, device=self.series.device).view(1, channels, 1)
        # Gather only selected futures: B,C,k,H (global) or B,C,K*k,H
        # (per-bank), never the full B,C,N,H candidate-future tensor.
        times = self.origins[selected].unsqueeze(-1) + torch.arange(self.pred_len, device=self.series.device)
        future = self.series[times, channel.unsqueeze(-1)]
        standardized = ((future - self.history_mean[selected, channel].unsqueeze(-1)) /
                        self.history_scale[selected, channel].unsqueeze(-1))
        transferred = (standardized * weights.unsqueeze(-1)).sum(2)
        prediction = (transferred * scale.unsqueeze(-1) + mean.unsqueeze(-1)).transpose(1, 2)
        selected_banks = selected_banks.masked_fill(~torch.isfinite(distances), -1)
        reliability = (self._reliability_features(reliability_distances, mask, selected_banks,
                       weights, standardized, selected, origin) if self.collect_reliability else None)
        return prediction, embeddings, selected_banks, weights, reliability

    def forward(self, raw_input, origin):
        """Return B,H,C prediction and differentiable retrieval context.

        ``cycle_weights`` is the B,C,K bank prior; ``selected_weights`` is the
        B,C,k effective inverse-distance mixture (global), or B,C,K*k
        within-bank weights multiplied by that prior (per-bank).
        ``base_candidate_*`` covers all age banks (B,M indices/origin/mask,
        B,C,M,15 embeddings); M is at least one with masked -1 sentinels.
        """
        if not self._count:
            raise RuntimeError('Attach the verified normalized train split before historical retrieval')
        if raw_input.ndim != 3 or raw_input.shape[1:] != (self.channels, self.seq_len):
            raise ValueError('Expected observed input with shape B,C,T')
        if origin.ndim != 1 or len(origin) != len(raw_input):
            raise ValueError('Expected one absolute forecast origin per input window')
        query, mean, scale = self._query_key(raw_input)
        query_origins = origin.detach().to(device='cpu', dtype=torch.long).tolist()
        indices, provenance, counts = self._candidates(query_origins)
        valid_banks = torch.tensor(counts, dtype=torch.bool, device=self.series.device)
        logits = self.cycle_router(query).masked_fill(~valid_banks.unsqueeze(1), -1e9)
        priors = logits.softmax(-1) * valid_banks.unsqueeze(1)
        priors = priors / priors.sum(-1, keepdim=True).clamp_min(1e-12)
        prediction, embeddings, selected_banks, selected_weights, reliability = self._retrieve_bank(
            query, mean, scale, indices, provenance, priors, origin)
        candidate_mask = indices >= 0
        candidate_origins = self.origins[indices.clamp_min(0)].masked_fill(~candidate_mask, -1)
        state = dict(query_embedding=query, query_mean=mean, query_std=scale, query_scale=scale,
                     regime=self.regime, query_origin=origin,
                     base_candidate_key_embeddings=embeddings,
                     base_candidate_index=indices, base_candidate_mask=candidate_mask,
                     base_candidate_origin=candidate_origins,
                     candidate_bank_index=provenance, selected_bank_index=selected_banks,
                     selected_weights=selected_weights)
        valid = valid_banks.any(-1)
        fallback = mean.unsqueeze(1).expand(-1, self.pred_len, -1)
        prediction = torch.where(valid[:, None, None], prediction, fallback)
        state.update(cycle_weights=priors, bank_valid=valid_banks, valid=valid)
        if self.collect_reliability:
            state['reliability_features'] = reliability
        self.diagnostics['calls'] += 1
        self.diagnostics['empty_queries'] += sum(not any(row) for row in counts)
        candidate_totals = [sum(row[bank] for row in counts) for bank in range(len(self.periods))]
        self.diagnostics['candidate_total_by_bank'] = [a + b for a, b in
            zip(self.diagnostics['candidate_total_by_bank'], candidate_totals)]
        return prediction, state
