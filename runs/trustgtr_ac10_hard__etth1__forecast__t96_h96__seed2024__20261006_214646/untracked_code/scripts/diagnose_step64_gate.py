#!/usr/bin/env python3
"""6.4.5: streaming, archived-checkpoint, validation-only gate diagnostics.

No test dataset, gradients, optimizer steps, or checkpoint writes. Forced gates
are inference counterfactuals within a jointly trained checkpoint. Label-aware
oracle forecasts are validation upper bounds, never deployable forecasts.
"""
import argparse
import hashlib
import importlib.util
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
from torch.nn import functional as F


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def archived_sources(run):
    manifest = json.loads((run / 'code_manifest.json').read_text())['sha256']
    paths, verified = {}, {}
    for name in ('models/GTR.py', 'models/TrustGTR.py', 'models/HistoricalMemory.py',
                 'data_provider/data_loader.py', 'utils/metrics.py'):
        snapshot = run / 'untracked_code' / name
        path = snapshot if snapshot.exists() else ROOT / name
        observed = file_hash(path)
        if observed != manifest[name]:
            raise ValueError('Archived source hash mismatch: ' + str(path))
        paths[name] = path
        verified[name] = dict(path=str(path), sha256=observed)
    # Bind the exact archived parent and memory classes before loading TrustGTR.
    names = ('models.GTR', 'models.HistoricalMemory')
    previous = {name: sys.modules.get(name) for name in names}
    try:
        gtr = load_module('models.GTR', paths['models/GTR.py'])
        load_module('models.HistoricalMemory', paths['models/HistoricalMemory.py'])
        trust = load_module('archived_gate64_' + run.name, paths['models/TrustGTR.py'])
    finally:
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]
    loader = load_module('archived_loader64_' + run.name, paths['data_provider/data_loader.py'])
    return trust, gtr, loader, verified


def verified_datasets(run, config, loader):
    args = SimpleNamespace(**config['official_args'])
    assets = json.loads((run / 'evaluation_assets.json').read_text())
    class_names = dict(ETTh1='Dataset_ETT_hour', ETTh2='Dataset_ETT_hour',
                       ETTm1='Dataset_ETT_minute', ETTm2='Dataset_ETT_minute',
                       Solar='Dataset_Solar', custom='Dataset_Custom')
    cls = getattr(loader, class_names[args.data])
    data_path = Path(args.root_path) / args.data_path
    if file_hash(data_path) != config['data_asset']['sha256']:
        raise ValueError('Raw data hash mismatch')
    kwargs = dict(root_path=args.root_path, data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
        target=args.target, timeenc=int(args.embed == 'timeF'), freq=args.freq, cycle=args.cycle)
    datasets, verified = {}, {}
    for flag in ('train', 'val'):
        metadata_path = Path(assets[flag]['path'])
        if file_hash(metadata_path) != assets[flag]['sha256']:
            raise ValueError('Evaluation asset manifest mismatch: ' + flag)
        metadata = json.loads(metadata_path.read_text())
        data = cls(flag=flag, **kwargs)
        checks = dict(data_x=array_hash(data.data_x) == metadata['data_x_sha256'],
            data_y=array_hash(data.data_y) == metadata['data_y_sha256'],
            phase=array_hash(data.cycle_index) == metadata['cycle_index_sha256'],
            raw_data=metadata['data_sha256'] == config['data_asset']['sha256'],
            window_count=len(data) == metadata['window_count'],
            split=assets[flag]['split'] == metadata['absolute_data_slice'])
        for name in ('mean_', 'var_', 'scale_'):
            checks[name] = np.array_equal(getattr(data.scaler, name),
                                         np.asarray(metadata['normalization'][name]))
        if not all(checks.values()):
            raise ValueError('Fixed evaluation assets disagree: ' + str(checks))
        datasets[flag] = data
        verified[flag] = dict(path=str(metadata_path), sha256=file_hash(metadata_path), checks=checks)
    return args, datasets, assets, verified


def tensors(data, indices, border):
    items = [data[int(index)] for index in indices]
    x = torch.tensor(np.stack([item[0] for item in items]), dtype=torch.float32)
    target = torch.tensor(np.stack([item[1] for item in items]), dtype=torch.float32)
    target = target[:, -data.pred_len:]
    origin = torch.tensor(border + indices + data.seq_len, dtype=torch.long)
    return x, target, origin


def decomposition(model, trust, gtr, x, origin):
    """Rebuild inference experts and gate using input/history only, no teacher."""
    local = gtr.Model.forward(model, x, origin)
    raw = x.permute(0, 2, 1)
    retrieved, state = model.historical_memory(raw, origin)
    valid = state['valid'].view(-1, 1, 1)
    bound = model.options['historical_blend_bound']
    scale = bound if model.options['historical_strength_coordinates'] == 'bounded' else 1.
    strength = bound * (model.historical_strength / scale).tanh()
    correction = (retrieved - local) * valid
    delta = strength * correction
    left, right = local.detach().permute(0, 2, 1), retrieved.detach().permute(0, 2, 1)
    agreement = F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)
    rms = left.square().mean(-1, keepdim=True).sqrt().clamp_min(.1)
    norm_ratio = torch.log1p((right - left).square().mean(-1, keepdim=True).sqrt() / rms)
    mismatch = (right - left).abs().mean(-1, keepdim=True) / rms
    descriptor = torch.cat((trust.state_features(raw), agreement, norm_ratio, mismatch), -1).clamp(-10, 10)
    if model.options['gate_descriptor'] == 'retrieval':
        descriptor = torch.cat((descriptor, state['reliability_features']), -1).detach().clamp(-10, 10)
    gate = model.gate(descriptor).sigmoid().permute(0, 2, 1) * valid
    reconstructed = local + strength * gate * correction
    return local, retrieved, delta, gate, reconstructed, strength


class Error:
    def __init__(self):
        self.elements = 0
        self.square = self.absolute = 0.

    def add(self, prediction, target):
        error = (prediction - target).double()
        self.elements += error.numel()
        self.square += float(error.square().sum())
        self.absolute += float(error.abs().sum())

    def result(self):
        return dict(elements=self.elements, mse=self.square / self.elements,
                    mae=self.absolute / self.elements)


def distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, std=None, quantiles={})
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                quantiles={str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)})


def correlation(x, y):
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or x.std() < 1e-10 or y.std() < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def selected_mean(value, mask):
    return float(value[mask].mean()) if mask.any() else None


def diagnose(run, sample_count, batch_size):
    started = time.monotonic()
    if json.loads((run / 'status.json').read_text())['status'] != 'SUCCESS':
        raise ValueError('Only successful experiments: ' + run.name)
    config = json.loads((run / 'resolved_config.yaml').read_text())
    opts = config['trustgtr']
    if opts['memory_mode'] != 'historical' or not opts['use_reliability_gate']:
        raise ValueError('Historical B checkpoint required')
    trust, gtr, loader, sources = archived_sources(run)
    args, datasets, assets, verified = verified_datasets(run, config, loader)
    model = trust.Model(args).cpu().eval()
    checkpoint = run / 'checkpoints' / 'best.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    model.historical_memory.attach_training_series(datasets['train'].data_x)
    memory_record = json.loads((run / 'historical_memory.json').read_text())
    if memory_record['sha256'] != model.historical_memory.metadata['sha256']:
        raise ValueError('Train-only historical memory hash mismatch')
    data = datasets['val']
    indices = np.arange(len(data)) if sample_count == 0 else np.unique(
        np.linspace(0, len(data) - 1, min(sample_count, len(data))).astype(int))
    border = assets['val']['split'][0]
    errors = {name: Error() for name in ('actual_dynamic', 'same_checkpoint_gtr_branch',
        'fixed_g1', 'retrieval_alone', 'validation_label_oracle_upper_bound', 'constant_gate_at_dynamic_mean')}
    # Only horizon-reduced channel-window scalars persist between batches.
    collected = {name: [] for name in ('gate', 'oracle', 'power', 'g1_gain', 'actual_gain', 'oracle_gain')}
    max_reconstruction_error = 0.
    with torch.inference_mode():
        # Every new checkpoint must pass a separate four-window reconstruction.
        smoke_indices = indices[:min(4, len(indices))]
        x, _, origin = tensors(data, smoke_indices, border)
        *_, reconstruction, _ = decomposition(model, trust, gtr, x, origin)
        smoke_error = float((model(x, origin) - reconstruction).abs().max())
        if smoke_error > 2e-6:
            raise ValueError('Four-window reconstruction failed: ' + str(smoke_error))
        for offset in range(0, len(indices), batch_size):
            x, target, origin = tensors(data, indices[offset:offset + batch_size], border)
            actual = model(x, origin)
            local, retrieved, delta, gate, reconstruction, strength = decomposition(model, trust, gtr, x, origin)
            max_reconstruction_error = max(max_reconstruction_error, float((actual - reconstruction).abs().max()))
            if max_reconstruction_error > 2e-6:
                raise ValueError('Full-stream actual reconstruction failed: ' + str(max_reconstruction_error))
            oracle, _ = trust.utility_oracle(local, delta, target)
            full, optimal = local + delta, local + oracle * delta
            predictions = dict(actual_dynamic=actual, same_checkpoint_gtr_branch=local,
                               fixed_g1=full, retrieval_alone=retrieved,
                               validation_label_oracle_upper_bound=optimal)
            for name, prediction in predictions.items():
                errors[name].add(prediction, target)
            base_mse = (local - target).square().mean(1)
            scalars = dict(gate=gate[:, 0], oracle=oracle[:, 0], power=delta.square().sum(1),
                g1_gain=base_mse - (full - target).square().mean(1),
                actual_gain=base_mse - (actual - target).square().mean(1),
                oracle_gain=base_mse - (optimal - target).square().mean(1))
            for name, value in scalars.items():
                collected[name].append(value.numpy())
            if offset % (batch_size * 32) == 0:
                print(json.dumps(dict(experiment_id=run.name, pass_index=1,
                    windows=min(offset + batch_size, len(indices)), total=len(indices),
                    elapsed_seconds=time.monotonic() - started)), flush=True)
        values = {name: np.concatenate(parts).astype(np.float64, copy=False) for name, parts in collected.items()}
        constant = float(values['gate'].mean())
        # MAE of a global constant mixture has no sufficient-moment formula;
        # a second read-only stream avoids retaining full Solar/ETTm predictions.
        for offset in range(0, len(indices), batch_size):
            x, target, origin = tensors(data, indices[offset:offset + batch_size], border)
            local, _, delta, _, _, _ = decomposition(model, trust, gtr, x, origin)
            errors['constant_gate_at_dynamic_mean'].add(local + constant * delta, target)
            if offset % (batch_size * 64) == 0:
                print(json.dumps(dict(experiment_id=run.name, pass_index=2,
                    windows=min(offset + batch_size, len(indices)), total=len(indices),
                    elapsed_seconds=time.monotonic() - started)), flush=True)
    gate, oracle, power, gain = [values[name] for name in ('gate', 'oracle', 'power', 'g1_gain')]
    meaningful = power > 1e-6  # Exact utility_oracle definition, not an efficacy threshold.
    harmful, helpful = meaningful & (gain < 0), meaningful & (gain > 0)
    weights = np.clip(power / max(float(power.mean()), 1e-8), 0, 10) * meaningful
    denom = float(weights.sum())
    groups = []
    order = np.argsort(gate[meaningful], kind='stable')
    for number, group in enumerate(np.array_split(order, 5), 1):
        flat_values = {name: value[meaningful][group] for name, value in values.items()}
        groups.append(dict(quintile=number, count=len(group),
            gate=distribution(flat_values['gate']), oracle=distribution(flat_values['oracle']),
            fixed_g1_mse_gain=distribution(flat_values['g1_gain']),
            actual_mse_gain=distribution(flat_values['actual_gain']),
            harmful_g1_fraction=float((flat_values['g1_gain'] < 0).mean()) if len(group) else None))
    score_values = {name: accumulator.result() for name, accumulator in errors.items()}
    actual_score = score_values['actual_dynamic']
    result = dict(experiment_id=run.name, dataset=config['experiment']['dataset'],
        calibration_context=model.options.get('gate_calibration_context', 'train'),
        gate_descriptor=opts['gate_descriptor'], gate_loss_mode=opts['gate_loss_mode'],
        gate_loss_weight=opts['gate_loss_weight'], router_mode=opts['historical_router_mode'],
        dropout_probabilities={name: module.p for name, module in model.named_modules()
                               if isinstance(module, torch.nn.Dropout)},
        strength=float(strength), raw_strength=float(model.historical_strength),
        evaluated_validation_windows=len(indices), total_validation_windows=len(data),
        validation_indices=indices.tolist(), fixed_assets=verified, model_sources=sources,
        checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
        historical_memory_sha256=memory_record['sha256'],
        scores=score_values, gate=distribution(gate), meaningful_gate=distribution(gate[meaningful]),
        oracle=distribution(oracle[meaningful]), effective_coefficient=distribution(float(strength) * gate),
        constant_gate_probability=constant,
        counts=dict(channel_windows=gate.size, meaningful_channel_windows=int(meaningful.sum()),
                    helpful_g1=int(helpful.sum()), harmful_g1=int(harmful.sum())),
        helpful_g1_gate_mean=selected_mean(gate, helpful), harmful_g1_gate_mean=selected_mean(gate, harmful),
        helpful_g1_oracle_mean=selected_mean(oracle, helpful), harmful_g1_oracle_mean=selected_mean(oracle, harmful),
        oracle_zero_fraction=selected_mean((oracle == 0).astype(float), meaningful),
        oracle_one_fraction=selected_mean((oracle == 1).astype(float), meaningful),
        gate_gain_correlation=correlation(gate[meaningful], gain[meaningful]),
        gate_oracle_correlation=correlation(gate[meaningful], oracle[meaningful]),
        gate_power_weighted_calibration_mse=(float((weights * (gate - oracle) ** 2).sum()) / denom if denom else None),
        calibration_weight_normalization='global validation mean power; differs from per-training-batch auxiliary aggregation',
        fixed_g1_mse_gain=distribution(gain[meaningful]),
        actual_mse_gain=distribution(values['actual_gain'][meaningful]),
        oracle_mse_gain=distribution(values['oracle_gain'][meaningful]),
        gate_quintiles=groups,
        dynamic_vs_constant_mean_mse_gain=score_values['constant_gate_at_dynamic_mean']['mse'] - actual_score['mse'],
        dynamic_vs_constant_mean_mae_gain=score_values['constant_gate_at_dynamic_mean']['mae'] - actual_score['mae'],
        oracle_mse_headroom=actual_score['mse'] - score_values['validation_label_oracle_upper_bound']['mse'],
        smoke_forward_max_error=smoke_error, full_forward_reconstruction_max_error=max_reconstruction_error,
        inference_aux_teacher_called=False, elapsed_seconds=time.monotonic() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps(dict(experiment_id=run.name, scores=score_values, gate=result['gate'],
        dynamic_vs_constant_mean_mse_gain=result['dynamic_vs_constant_mean_mse_gain'],
        elapsed_seconds=result['elapsed_seconds'])), flush=True)
    return result


def markdown(report):
    lines = ['# 6.4.5 固定验证集可靠性诊断', '',
        '1. 归档 best checkpoint；train-only记忆；固定validation窗口；CPU 2 threads、流式两遍；无test、反向、优化器或checkpoint写入。',
        '2. g0/g1/constant门都是同一联合训练checkpoint的推理反事实。g0分支并非独立原GTR，oracle仅为使用验证标签的MSE上限。',
        '3. 门控有效性依据dynamic相对constant的收益、harmful/helpful排序及校准误差，不设置任意门值效果阈值。',
        '4. 全窗口口径不同于官方shuffle/drop_last early-stopping val；powerweighted诊断用全验证集mean power，不能冒充训练batch辅助loss。', '',
        '| dataset | context | windows | dynamic MSE | dynamic MAE | g0 MSE | g1 MSE | constant MSE | dynamic−constant收益 | gate mean | harmful/helpful gate |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    fmt = lambda number: 'NA' if number is None else '{:.9f}'.format(number)
    for row in report['experiments']:
        s = row['scores']
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.6f} | {}/{} |'.format(
            row['dataset'], row['calibration_context'], row['evaluated_validation_windows'],
            fmt(s['actual_dynamic']['mse']), fmt(s['actual_dynamic']['mae']),
            fmt(s['same_checkpoint_gtr_branch']['mse']), fmt(s['fixed_g1']['mse']),
            fmt(s['constant_gate_at_dynamic_mean']['mse']), fmt(row['dynamic_vs_constant_mean_mse_gain']),
            row['gate']['mean'], fmt(row['harmful_g1_gate_mean']), fmt(row['helpful_g1_gate_mean'])))
    for row in report['experiments']:
        lines += ['', '## ' + row['experiment_id'], '',
            '5. gate分布：' + json.dumps(row['gate'], ensure_ascii=False),
            '6. counts：' + json.dumps(row['counts'], ensure_ascii=False),
            '7. gate−gain相关 {}，gate−oracle相关 {}，powerweighted校准误差 {}，oracle MSE余量 {}。'.format(
                fmt(row['gate_gain_correlation']), fmt(row['gate_oracle_correlation']),
                fmt(row['gate_power_weighted_calibration_mse']), fmt(row['oracle_mse_headroom'])),
            '8. 按门值排序的五等分：', '',
            '| quintile | count | gate mean | oracle mean | g1 gain | dynamic gain | harmful fraction |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for group in row['gate_quintiles']:
            lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
                group['quintile'], group['count'], fmt(group['gate']['mean']), fmt(group['oracle']['mean']),
                fmt(group['fixed_g1_mse_gain']['mean']), fmt(group['actual_mse_gain']['mean']),
                fmt(group['harmful_g1_fraction'])))
        lines += ['', '9. 首4窗口/全窗口重建误差：{} / {}；耗时 {:.1f}s。资产、源码、checkpoint哈希及各反事实MAE见同名JSON。'.format(
            row['smoke_forward_max_error'], row['full_forward_reconstruction_max_error'], row['elapsed_seconds'])]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--sample-count', type=int, default=256, help='0: all fixed validation windows')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.sample_count < 0 or args.batch_size < 1:
        raise ValueError('Invalid diagnostic sample count or batch size')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    output = ROOT / args.output
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('Refuse overwrite: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(scope='6.4.5 archived best checkpoints; validation only; CPU 2 threads',
        no_test_dataset_constructed=True, no_test_targets_used=True, no_backward=True,
        no_optimizer_steps=True, no_checkpoint_writes=True, archived_scientific_sources=True,
        forced_gates_are_same_checkpoint_counterfactuals=True, oracle_uses_validation_targets=True,
        official_early_stopping_validation_uses_different_aggregation=True,
        script_sha256=file_hash(Path(__file__)), batch_size=args.batch_size,
        sample_count=args.sample_count, experiments=[])
    for name in args.runs:
        row = diagnose(ROOT / 'runs' / name, args.sample_count, args.batch_size)
        for previous in report['experiments']:
            if previous['dataset'] == row['dataset']:
                if previous['validation_indices'] != row['validation_indices'] or previous['fixed_assets'] != row['fixed_assets']:
                    raise ValueError('Paired validation indices/assets mismatch')
        report['experiments'].append(row)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        output.with_suffix('.md').write_text(markdown(report))


if __name__ == '__main__':
    main()
