#!/usr/bin/env python3
"""Streaming, validation-only archived A6/A8 mechanism diagnosis.

Every forced bank/router is a same-checkpoint inference counterfactual, never a
trained ablation. No test dataset, gradients, optimizer, or checkpoint write.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from models.GTR import Model as OriginalModel
import diagnose_evidence_router as archive
import diagnose_gate_v5 as helper


class Moment:
    def __init__(self):
        self.n = 0
        self.total = 0.
        self.square = 0.
        self.low = float('inf')
        self.high = -float('inf')

    def add(self, value):
        value = np.asarray(value, dtype='float64').ravel()
        value = value[np.isfinite(value)]
        if not len(value):
            return
        self.n += len(value)
        self.total += value.sum()
        self.square += np.square(value).sum()
        self.low = min(self.low, float(value.min()))
        self.high = max(self.high, float(value.max()))

    def result(self):
        if not self.n:
            return dict(count=0, mean=None, std=None, min=None, max=None)
        mean = self.total / self.n
        return dict(count=self.n, mean=float(mean),
                    std=float(max(0., self.square / self.n - mean ** 2) ** .5),
                    min=self.low, max=self.high)


class Correlation:
    def __init__(self):
        self.n = 0
        self.x = self.y = self.xx = self.yy = self.xy = 0.

    def add(self, x, y):
        x, y = np.asarray(x, dtype='float64').ravel(), np.asarray(y, dtype='float64').ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        self.n += len(x)
        self.x += x.sum()
        self.y += y.sum()
        self.xx += np.square(x).sum()
        self.yy += np.square(y).sum()
        self.xy += (x * y).sum()

    def result(self):
        if self.n < 2:
            return dict(count=self.n, pearson=None)
        cov = self.xy - self.x * self.y / self.n
        variance = max(0., self.xx - self.x ** 2 / self.n) * max(0., self.yy - self.y ** 2 / self.n)
        return dict(count=self.n, pearson=float(cov / variance ** .5) if variance > 1e-20 else None)


class Error:
    def __init__(self):
        self.n = 0
        self.square = 0.
        self.absolute = 0.

    def add(self, error, mask=None):
        error = error.double()
        if mask is not None:
            error = error.permute(0, 2, 1)[mask]
        self.n += error.numel()
        self.square += float(error.square().sum())
        self.absolute += float(error.abs().sum())

    def result(self):
        return dict(elements=self.n, mse=self.square / self.n if self.n else None,
                    mae=self.absolute / self.n if self.n else None)


def numpy(value):
    return value.cpu().numpy()


def retrieve_components(memory, state):
    """Reproduce both frozen routing formulas using the actual candidate state."""
    query = state['query_embedding']
    indices = state['base_candidate_index']
    provenance = state['candidate_bank_index']
    full_distance = (state['base_candidate_key_embeddings'] - query.unsqueeze(2)).square().sum(-1)
    full_distance = full_distance.masked_fill(~state['base_candidate_mask'].unsqueeze(1), float('inf'))
    bank_distances, bank_positions = [], []
    k = min(memory.topk, indices.shape[-1])
    for bank in range(len(memory.periods)):
        distance = full_distance.masked_fill(~(provenance == bank).unsqueeze(1), float('inf'))
        chosen = distance.topk(k, dim=-1, largest=False)
        bank_distances.append(chosen.values)
        bank_positions.append(chosen.indices)
    distance = torch.stack(bank_distances, -2)
    positions = torch.stack(bank_positions, -2)
    finite = torch.isfinite(distance)
    inverse = 1 / (distance + 1e-3)
    mass = inverse.sum(-1)
    within_old = inverse / mass.unsqueeze(-1).clamp_min(1e-12)
    within_new = inverse / torch.where(mass > 0, mass, 1).unsqueeze(-1)
    prior = state['cycle_weights']
    evidence = mass / finite.sum(-1).clamp_min(1)
    maximum = evidence.amax(-1, keepdim=True)
    scaled_evidence = evidence / torch.where(maximum > 0, maximum, 1)
    score = prior * scaled_evidence
    total = score.sum(-1, keepdim=True)
    posterior = score / torch.where(total > 0, total, 1)
    grouped_weights = {'per_bank': within_old * prior.unsqueeze(-1),
                       'per_bank_evidence': within_new * posterior.unsqueeze(-1)}
    channels = memory.channels
    selected = indices.clamp_min(0).unsqueeze(1).expand(-1, channels, -1).gather(-1, positions.flatten(-2))
    channel = torch.arange(channels).view(1, channels, 1)
    times = memory.origins[selected].unsqueeze(-1) + torch.arange(memory.pred_len)
    future = memory.series[times, channel.unsqueeze(-1)]
    standardized = ((future - memory.history_mean[selected, channel].unsqueeze(-1)) /
                    memory.history_scale[selected, channel].unsqueeze(-1))
    mean, scale = state['query_mean'], state['query_scale']
    def transfer(weights):
        transferred = (standardized * weights.flatten(-2).unsqueeze(-1)).sum(2)
        return (transferred * scale.unsqueeze(-1) + mean.unsqueeze(-1)).transpose(1, 2)
    route_forecasts = {key: transfer(value) for key, value in grouped_weights.items()}
    shaped = standardized.reshape(*query.shape[:2], len(memory.periods), k, memory.pred_len)
    bank_transfer = (shaped * within_old.unsqueeze(-1)).sum(-2)
    bank_forecasts = (bank_transfer * scale[:, :, None, None] + mean[:, :, None, None]).permute(0, 2, 3, 1)
    candidate_forecasts = standardized * scale[:, :, None, None] + mean[:, :, None, None]
    return dict(distance=distance, evidence=evidence, prior=prior, posterior=posterior,
                weights=grouped_weights, route_forecasts=route_forecasts,
                bank_forecasts=bank_forecasts, candidate_forecasts=candidate_forecasts)


def diagnose(run, sample_count, batch_size):
    started = time.monotonic()
    if json.loads((run / 'status.json').read_text())['status'] != 'SUCCESS':
        raise ValueError('Only successful runs: ' + run.name)
    config = json.loads((run / 'resolved_config.yaml').read_text())
    if config['trustgtr']['use_reliability_gate'] or config['trustgtr']['use_regime_contrast']:
        raise ValueError('Only A-only checkpoints')
    model_class, sources = archive.archived_model(run)
    args = SimpleNamespace(**config['official_args'])
    datasets, assets, verified = helper.verified_datasets(run, args)
    model = model_class(args).cpu().eval()
    checkpoint = run / 'checkpoints/best.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    memory = model.historical_memory
    memory.attach_training_series(datasets['train'].data_x)
    stored = json.loads((run / 'historical_memory.json').read_text())
    if stored['sha256'] != memory.metadata['sha256']:
        raise ValueError('Train memory hash mismatch')
    data = datasets['val']
    indices = np.arange(len(data)) if sample_count == 0 else np.unique(
        np.linspace(0, len(data) - 1, min(sample_count, len(data))).astype(int))
    bound = model.options['historical_blend_bound']
    coordinate_scale = bound if model.options['historical_strength_coordinates'] == 'bounded' else 1.
    strength = bound * (model.historical_strength / coordinate_scale).tanh()
    native_mode = model.options['historical_router_mode']
    alternate = 'per_bank' if native_mode == 'per_bank_evidence' else 'per_bank_evidence'
    banks = len(memory.periods)
    names = ['actual', 'same_checkpoint_gtr_branch', 'retrieval_alone', 'alternate_router',
             *['forced_bank_' + str(bank) for bank in range(banks)]]
    errors = {name: Error() for name in names}
    strata = {stratum: {name: Error() for name in names} for stratum in ('near_flat', 'non_flat')}
    counts = dict(channel_windows=0, near_flat_channel_windows=0, harmful_correction=0,
                  harmful_retrieval_alone=0, posterior_selects_wrong_bank=0)
    statistics = {name: Moment() for name in ('bank_max_weight', 'bank_entropy_normalized',
        'bank_weight_hhi', 'candidate_max_weight', 'candidate_entropy_normalized',
        'posterior_prior_mean_abs', 'actual_correction_mse_gain', 'retrieval_alone_mse_gain',
        'input_std', 'query_distance_nearest')}
    stratum_statistics = {key: {name: Moment() for name in statistics} for key in ('near_flat', 'non_flat')}
    bank_stats = [{name: Moment() for name in ('prior', 'posterior', 'effective', 'evidence',
        'nearest_distance', 'topk_mean_distance', 'correction_mse_gain', 'retrieval_alone_mse_gain')}
        for _ in range(banks)]
    correlations = {name: Correlation() for name in ('bank_evidence_vs_correction_gain',
        'bank_nearest_distance_vs_correction_gain', 'bank_evidence_vs_retrieval_gain',
        'bank_posterior_vs_correction_gain', 'selected_bank_evidence_vs_gain',
        'candidate_distance_vs_correction_gain', 'candidate_distance_vs_retrieval_gain')}
    stratum_correlations = {key: {name: Correlation() for name in ('bank_evidence_vs_correction_gain',
        'bank_nearest_distance_vs_correction_gain')} for key in ('near_flat', 'non_flat')}
    per_bank_correlations = [{name: Correlation() for name in ('evidence_vs_gain', 'distance_vs_gain')}
                             for _ in range(banks)]
    smoke_error = route_reconstruction_error = posterior_effective_error = 0.
    with torch.inference_mode():
        # Explicit first-four-window smoke, independent of profiling batch size.
        x, target, origin = helper.tensors(data, indices[:4], assets['val']['split'][0])
        actual = model(x, origin)
        local = OriginalModel.forward(model, x, origin)
        retrieved, state = memory(x.permute(0, 2, 1), origin)
        reconstructed = local + strength * (retrieved - local) * state['valid'][:, None, None]
        smoke_error = float((actual - reconstructed).abs().max())
        components = retrieve_components(memory, state)
        smoke_route_error = float((retrieved - components['route_forecasts'][native_mode]).abs().max())
        if max(smoke_error, smoke_route_error) > 1e-6:
            raise ValueError('First-four smoke reconstruction failed: ' + str((smoke_error, smoke_route_error)))
        print(json.dumps(dict(experiment_id=run.name, smoke_forward_max_error=smoke_error,
                              smoke_route_max_error=smoke_route_error)), flush=True)
        for offset in range(0, len(indices), batch_size):
            x, target, origin = helper.tensors(data, indices[offset:offset + batch_size], assets['val']['split'][0])
            local = OriginalModel.forward(model, x, origin)
            retrieved, state = memory(x.permute(0, 2, 1), origin)
            valid = state['valid'][:, None, None]
            actual = local + strength * (retrieved - local) * valid
            components = retrieve_components(memory, state)
            route_reconstruction_error = max(route_reconstruction_error,
                float((retrieved - components['route_forecasts'][native_mode]).abs().max()))
            effective = torch.stack([(state['selected_weights'] * (state['selected_bank_index'] == bank)).sum(-1)
                                      for bank in range(banks)], -1)
            posterior = components['posterior'] if native_mode == 'per_bank_evidence' else components['prior']
            posterior_effective_error = max(posterior_effective_error, float((posterior - effective).abs().max()))
            forecasts = dict(actual=actual, same_checkpoint_gtr_branch=local, retrieval_alone=retrieved,
                alternate_router=local + strength * (components['route_forecasts'][alternate] - local) * valid)
            for bank in range(banks):
                forecasts['forced_bank_' + str(bank)] = local + strength * (components['bank_forecasts'][:, bank] - local) * valid
            input_std = x.std(1, unbiased=False)
            flat = input_std <= .1  # Predeclared descriptor floor, no target-dependent grouping.
            base_mse = (local - target).square().mean(1)
            actual_gain = base_mse - (actual - target).square().mean(1)
            retrieved_gain = base_mse - (retrieved - target).square().mean(1)
            bank_gain = torch.stack([base_mse - (forecasts['forced_bank_' + str(bank)] - target).square().mean(1)
                                     for bank in range(banks)], -1)
            bank_retrieval_gain = base_mse.unsqueeze(-1) - (components['bank_forecasts'] - target[:, None]).square().mean(2).transpose(1, 2)
            nearest = components['distance'][..., 0]
            finite_count = torch.isfinite(components['distance']).sum(-1)
            topk_mean = components['distance'].masked_fill(~torch.isfinite(components['distance']), 0).sum(-1) / finite_count.clamp_min(1)
            for name, prediction in forecasts.items():
                error = prediction - target
                errors[name].add(error)
                strata['near_flat'][name].add(error, flat)
                strata['non_flat'][name].add(error, ~flat)
            counts['channel_windows'] += flat.numel()
            counts['near_flat_channel_windows'] += int(flat.sum())
            counts['harmful_correction'] += int((actual_gain < 0).sum())
            counts['harmful_retrieval_alone'] += int((retrieved_gain < 0).sum())
            chosen = posterior.argmax(-1)
            counts['posterior_selects_wrong_bank'] += int((chosen != bank_gain.argmax(-1)).sum())
            candidate_weight = state['selected_weights']
            quantities = dict(bank_max_weight=effective.amax(-1),
                bank_entropy_normalized=-(effective * effective.clamp_min(1e-12).log()).sum(-1) / np.log(banks),
                bank_weight_hhi=effective.square().sum(-1), candidate_max_weight=candidate_weight.amax(-1),
                candidate_entropy_normalized=-(candidate_weight * candidate_weight.clamp_min(1e-12).log()).sum(-1) / np.log(candidate_weight.shape[-1]),
                posterior_prior_mean_abs=(posterior - components['prior']).abs().mean(-1),
                actual_correction_mse_gain=actual_gain, retrieval_alone_mse_gain=retrieved_gain,
                input_std=input_std, query_distance_nearest=nearest.amin(-1))
            for name, value in quantities.items():
                statistics[name].add(numpy(value))
                stratum_statistics['near_flat'][name].add(numpy(value[flat]))
                stratum_statistics['non_flat'][name].add(numpy(value[~flat]))
            for bank in range(banks):
                values = dict(prior=components['prior'][..., bank], posterior=posterior[..., bank],
                    effective=effective[..., bank], evidence=components['evidence'][..., bank],
                    nearest_distance=nearest[..., bank], topk_mean_distance=topk_mean[..., bank],
                    correction_mse_gain=bank_gain[..., bank], retrieval_alone_mse_gain=bank_retrieval_gain[..., bank])
                for name, value in values.items():
                    bank_stats[bank][name].add(numpy(value))
                per_bank_correlations[bank]['evidence_vs_gain'].add(numpy(values['evidence']), numpy(values['correction_mse_gain']))
                per_bank_correlations[bank]['distance_vs_gain'].add(numpy(values['nearest_distance']), numpy(values['correction_mse_gain']))
            correlations['bank_evidence_vs_correction_gain'].add(numpy(components['evidence']), numpy(bank_gain))
            correlations['bank_nearest_distance_vs_correction_gain'].add(numpy(nearest), numpy(bank_gain))
            correlations['bank_evidence_vs_retrieval_gain'].add(numpy(components['evidence']), numpy(bank_retrieval_gain))
            correlations['bank_posterior_vs_correction_gain'].add(numpy(posterior), numpy(bank_gain))
            correlations['selected_bank_evidence_vs_gain'].add(numpy(components['evidence'].gather(-1, chosen.unsqueeze(-1))), numpy(bank_gain.gather(-1, chosen.unsqueeze(-1))))
            candidate_target = target.transpose(1, 2).unsqueeze(2)
            candidate_local = local.transpose(1, 2).unsqueeze(2)
            candidate_retrieval = components['candidate_forecasts']
            candidate_correction = candidate_local + strength * (candidate_retrieval - candidate_local)
            candidate_gain = base_mse.unsqueeze(-1) - (candidate_correction - candidate_target).square().mean(-1)
            candidate_retrieval_gain = base_mse.unsqueeze(-1) - (candidate_retrieval - candidate_target).square().mean(-1)
            correlations['candidate_distance_vs_correction_gain'].add(numpy(components['distance'].flatten(-2)), numpy(candidate_gain))
            correlations['candidate_distance_vs_retrieval_gain'].add(numpy(components['distance'].flatten(-2)), numpy(candidate_retrieval_gain))
            for label, mask in (('near_flat', flat), ('non_flat', ~flat)):
                stratum_correlations[label]['bank_evidence_vs_correction_gain'].add(numpy(components['evidence'][mask]), numpy(bank_gain[mask]))
                stratum_correlations[label]['bank_nearest_distance_vs_correction_gain'].add(numpy(nearest[mask]), numpy(bank_gain[mask]))
            if offset % (batch_size * 32) == 0:
                print(json.dumps(dict(experiment_id=run.name, windows=min(offset + batch_size, len(indices)),
                                      total=len(indices), elapsed_seconds=time.monotonic() - started)), flush=True)
    result = dict(experiment_id=run.name, dataset=config['experiment']['dataset'], router_mode=native_mode,
        alternate_router_mode=alternate, strength=float(strength), raw_strength=float(model.historical_strength),
        evaluated_validation_windows=len(indices), total_validation_windows=len(data), validation_indices=indices.tolist(),
        fixed_assets=verified, model_sources=sources, checkpoint=str(checkpoint), checkpoint_sha256=helper.file_hash(checkpoint),
        historical_memory_sha256=stored['sha256'], counts=counts,
        fractions={name: counts[name] / counts['channel_windows'] for name in ('near_flat_channel_windows',
            'harmful_correction', 'harmful_retrieval_alone', 'posterior_selects_wrong_bank')},
        scores={name: value.result() for name, value in errors.items()},
        strata={key: {name: value.result() for name, value in row.items()} for key, row in strata.items()},
        statistics={name: value.result() for name, value in statistics.items()},
        stratum_statistics={key: {name: value.result() for name, value in row.items()} for key, row in stratum_statistics.items()},
        stratum_correlations={key: {name: value.result() for name, value in row.items()} for key, row in stratum_correlations.items()},
        banks=[{name: value.result() for name, value in row.items()} for row in bank_stats],
        correlations={name: value.result() for name, value in correlations.items()},
        per_bank_correlations=[{name: value.result() for name, value in row.items()} for row in per_bank_correlations],
        smoke_forward_max_error=smoke_error, smoke_route_max_error=smoke_route_error,
        full_route_reconstruction_max_error=route_reconstruction_error,
        posterior_effective_max_error=posterior_effective_error,
        elapsed_seconds=time.monotonic() - started, process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if posterior_effective_error > 2e-6:
        raise ValueError('Native posterior/effective weight mismatch')
    print(json.dumps(dict(experiment_id=run.name, scores=result['scores'], strength=result['strength'],
                         fractions=result['fractions'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
    return result


def markdown(report):
    lines = ['# 5.6 固定验证集机制诊断', '',
        '1. 仅归档 best checkpoint、完整固定验证窗口、CPU 2 threads、流式 batch16；无 test、反向、优化器或 checkpoint 写入。',
        '2. 同 checkpoint GTR branch 是联合训练后的分支，并非独立原 GTR；强制单库和路由切换是推理反事实，不是重训消融。',
        '3. 输入近恒定 strata 事先定义为各通道输入 std≤0.1，来自记忆描述符尺度下限；效用定义为本 checkpoint GTR branch MSE 减去对应预测 MSE，正值表示帮助。',
        '4. 全窗口流式聚合不同于官方 shuffle/drop_last 的 early-stopping val 口径，不能混用。', '',
        '| dataset | mode | windows | actual MSE | branch MSE | retrieved MSE | switched MSE | strength | harmful correction | bank max weight |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['experiments']:
        scores = row['scores']
        lines.append('| {} | {} | {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} | {:.6f} | {:.2%} | {:.6f} |'.format(
            row['dataset'], row['router_mode'], row['evaluated_validation_windows'], scores['actual']['mse'],
            scores['same_checkpoint_gtr_branch']['mse'], scores['retrieval_alone']['mse'], scores['alternate_router']['mse'],
            row['strength'], row['fractions']['harmful_correction'], row['statistics']['bank_max_weight']['mean']))
    for row in report['experiments']:
        lines += ['', '## ' + row['experiment_id'], '',
            '5. prior / posterior / effective 各库均值：' + json.dumps({key: [bank[key]['mean'] for bank in row['banks']]
              for key in ('prior', 'posterior', 'effective')}, ensure_ascii=False),
            '6. 匹配质量与验证未来效用相关：' + json.dumps(row['correlations'], ensure_ascii=False),
            '7. 分层 MSE / MAE：', '', '| stratum | elements | actual MSE | actual MAE | branch MSE | switched MSE |',
            '|---|---:|---:|---:|---:|---:|']
        for name, values in row['strata'].items():
            fmt = lambda v: 'NA' if v is None else '{:.9f}'.format(v)
            lines.append('| {} | {} | {} | {} | {} | {} |'.format(name, values['actual']['elements'],
                fmt(values['actual']['mse']), fmt(values['actual']['mae']), fmt(values['same_checkpoint_gtr_branch']['mse']),
                fmt(values['alternate_router']['mse'])))
        lines += ['', '8. 每库诊断：', '', '| bank | nearest distance | evidence | correction gain | forced-bank MSE | forced-bank MAE |',
                  '|---|---:|---:|---:|---:|---:|']
        for bank, values in enumerate(row['banks']):
            score = row['scores']['forced_bank_' + str(bank)]
            lines.append('| {} | {:.6f} | {:.6f} | {:.9f} | {:.9f} | {:.9f} |'.format(bank,
                values['nearest_distance']['mean'], values['evidence']['mean'], values['correction_mse_gain']['mean'], score['mse'], score['mae']))
        lines += ['', '9. 首4窗口 forward / route 重建最大误差：{} / {}；全量 route 重建最大误差：{}。'.format(
            row['smoke_forward_max_error'], row['smoke_route_max_error'], row['full_route_reconstruction_max_error']),
            '10. 耗时 {:.1f}s；详细数据、sha256、固定资产核验和窗口索引见同名 JSON。'.format(row['elapsed_seconds'])]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--sample-count', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = ROOT / args.output
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('Refuse overwrite: ' + str(output))
    report = dict(scope='5.6 archived A6/A8 best; identical fixed validation windows; CPU 2 threads',
        no_test_dataset_constructed=True, no_test_targets_used=True, no_backward=True, no_optimizer_steps=True,
        archived_scientific_sources=True, forced_routes_are_same_checkpoint_counterfactuals=True,
        official_early_stopping_validation_uses_different_aggregation=True,
        near_flat_rule='input channel std <= 0.1 (historical descriptor floor), before observing targets',
        script_sha256=helper.file_hash(Path(__file__)), helper_sha256={name: helper.file_hash(ROOT / 'scripts' / name)
            for name in ('diagnose_evidence_router.py', 'diagnose_gate_v5.py', 'diagnose_historical_retrieval.py')},
        batch_size=args.batch_size, experiments=[])
    for name in args.runs:
        row = diagnose(ROOT / 'runs' / name, args.sample_count, args.batch_size)
        for prior in report['experiments']:
            if row['dataset'] == prior['dataset']:
                if row['validation_indices'] != prior['validation_indices'] or row['fixed_assets'] != prior['fixed_assets']:
                    raise ValueError('Comparison indices/assets mismatch')
        report['experiments'].append(row)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        output.with_suffix('.md').write_text(markdown(report))


if __name__ == '__main__':
    main()
