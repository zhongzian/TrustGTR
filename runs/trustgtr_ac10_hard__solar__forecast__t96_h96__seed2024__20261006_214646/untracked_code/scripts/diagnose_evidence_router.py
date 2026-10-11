#!/usr/bin/env python3
"""Read-only archived-model comparison on identical validation windows."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import diagnose_gate_v5 as helper


def archived_model(run):
    manifest = json.loads((run / 'code_manifest.json').read_text())['sha256']
    sources = {}
    for name in ('models/GTR.py', 'data_provider/data_loader.py', 'utils/metrics.py',
                 'models/TrustGTR.py', 'models/HistoricalMemory.py'):
        archived = run / 'untracked_code' / name
        path = archived if archived.exists() else ROOT / name
        if helper.file_hash(path) != manifest[name]:
            raise ValueError('Source hash mismatch: ' + str(path))
        sources[name] = dict(path=str(path), sha256=manifest[name])
    previous = sys.modules.get('models.HistoricalMemory')
    try:
        path = sources['models/HistoricalMemory.py']['path']
        spec = importlib.util.spec_from_file_location('models.HistoricalMemory', path)
        memory = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = memory
        spec.loader.exec_module(memory)
        path = sources['models/TrustGTR.py']['path']
        spec = importlib.util.spec_from_file_location('archived_evidence_' + run.name, path)
        trust = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trust)
    finally:
        if previous is None:
            sys.modules.pop('models.HistoricalMemory', None)
        else:
            sys.modules['models.HistoricalMemory'] = previous
    return trust.Model, sources


def diagnose(run, sample_count, batch_size):
    started = time.monotonic()
    if json.loads((run / 'status.json').read_text())['status'] != 'SUCCESS':
        raise ValueError('Run must be successful: ' + run.name)
    config = json.loads((run / 'resolved_config.yaml').read_text())
    if config['trustgtr']['use_reliability_gate'] or config['trustgtr']['use_regime_contrast']:
        raise ValueError('A-only route comparison required')
    args = SimpleNamespace(**config['official_args'])
    model_class, sources = archived_model(run)
    datasets, assets, verified = helper.verified_datasets(run, args)
    model = model_class(args).cpu().eval()
    checkpoint = run / 'checkpoints/best.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    model.historical_memory.attach_training_series(datasets['train'].data_x)
    stored = json.loads((run / 'historical_memory.json').read_text())
    if stored['sha256'] != model.historical_memory.metadata['sha256']:
        raise ValueError('Historical train memory hash mismatch')
    data = datasets['val']
    indices = (np.arange(len(data)) if sample_count == 0 else
               np.unique(np.linspace(0, len(data) - 1, min(sample_count, len(data))).astype(int)))
    predictions, targets = [], []
    values = {key: [] for key in ('prior', 'posterior', 'effective', 'evidence')}
    reconstruction_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            x, target, origin = helper.tensors(data, indices[start:start + batch_size], assets['val']['split'][0])
            prediction = model(x, origin)
            _, state = model.historical_memory(x.permute(0, 2, 1), origin)
            prior = state['cycle_weights']
            banks, weights = state['selected_bank_index'], state['selected_weights']
            effective = torch.stack([(weights * (banks == bank)).sum(-1)
                                     for bank in range(len(model.options['periods']))], -1)
            posterior = state.get('bank_posterior', prior)
            reconstruction_error = max(reconstruction_error, float((posterior - effective).abs().max()))
            predictions.append(prediction.numpy())
            targets.append(target.numpy())
            for key, value in (('prior', prior), ('posterior', posterior), ('effective', effective)):
                values[key].append(value.numpy())
            if 'bank_evidence' in state:
                values['evidence'].append(state['bank_evidence'].numpy())
    if reconstruction_error > 2e-6:
        raise ValueError('Posterior disagrees with effective bank weights')
    combined = {key: np.concatenate(parts) for key, parts in values.items() if parts}
    result = dict(experiment_id=run.name, dataset=config['experiment']['dataset'],
        router_mode=model.options['historical_router_mode'],
        scores=helper.scores(np.concatenate(predictions), np.concatenate(targets)),
        evaluated_validation_windows=len(indices), total_validation_windows=len(data),
        validation_indices=indices.tolist(), fixed_assets=verified,
        model_sources=sources, checkpoint=str(checkpoint), checkpoint_sha256=helper.file_hash(checkpoint),
        historical_memory_sha256=stored['sha256'],
        bank_prior_mean=combined['prior'].mean((0, 1)).tolist(),
        bank_posterior_mean=combined['posterior'].mean((0, 1)).tolist(),
        bank_effective_weight_mean=combined['effective'].mean((0, 1)).tolist(),
        posterior_effective_max_error=reconstruction_error,
        posterior_prior_mean_absolute_difference=float(np.abs(combined['posterior'] - combined['prior']).mean()),
        elapsed_seconds=time.monotonic() - started)
    if 'evidence' in combined:
        result['bank_evidence_mean'] = combined['evidence'].mean((0, 1)).tolist()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--sample-count', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = ROOT / args.output
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('Refuse to overwrite ' + str(output))
    report = dict(scope='best checkpoint; identical deterministic validation windows; CPU 2 threads',
        no_test_dataset_constructed=True, no_test_targets_used=True,
        no_backward=True, no_optimizer_steps=True, archived_scientific_sources=True,
        official_early_stopping_validation_uses_different_aggregation=True,
        script_sha256=helper.file_hash(Path(__file__)), experiments=[])
    for run_name in args.runs:
        row = diagnose(ROOT / 'runs' / run_name, args.sample_count, args.batch_size)
        for previous in report['experiments']:
            if previous['dataset'] == row['dataset']:
                if previous['validation_indices'] != row['validation_indices']:
                    raise ValueError('Comparison windows differ')
                if previous['fixed_assets'] != row['fixed_assets']:
                    raise ValueError('Comparison assets differ')
        report['experiments'].append(row)
        print(json.dumps({key: row[key] for key in ('experiment_id', 'scores', 'bank_prior_mean',
              'bank_posterior_mean', 'elapsed_seconds')}), flush=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    lines = ['# v8 固定验证窗口路由诊断', '',
             '仅验证集、归档科学源码与 best checkpoint；没有 test、反向或优化器更新。', '',
             '| dataset | router | MSE | MAE | windows | posterior−prior mean abs |',
             '|---|---|---:|---:|---:|---:|']
    for row in report['experiments']:
        lines.append('| {} | {} | {:.9f} | {:.9f} | {} | {:.6f} |'.format(
            row['dataset'], row['router_mode'], row['scores']['mse'], row['scores']['mae'],
            row['evaluated_validation_windows'], row['posterior_prior_mean_absolute_difference']))
    lines += ['', '同数据集 indices 与资产逐项核对；此诊断的窗口聚合口径不同于官方早停验证损失，不能混为同一数值。',
              '匹配距离接近不保证未来更有用；先看同协议指标，再决定是否采用候选证据路由。']
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
