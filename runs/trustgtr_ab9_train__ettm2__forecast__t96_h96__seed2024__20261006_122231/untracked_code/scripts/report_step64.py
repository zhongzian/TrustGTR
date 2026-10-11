#!/usr/bin/env python3
"""Summarize persisted paired calibration runs without selecting on test."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', required=True)
    parser.add_argument('--output', default='reports/step_6_4_paired_results.json')
    args = parser.parse_args()
    queue = read(ROOT / args.queue)
    rows, pairs = [], {}
    for job in queue['jobs']:
        if job['is_smoke']:
            continue
        run = Path(job['run_dir'])
        config_path = Path(job['argv'][job['argv'].index('--config') + 1])
        config = read(config_path)
        metrics, status = read(run / 'metrics.json'), read(run / 'status.json')
        dataset = config['experiment']['dataset']
        context = config['trustgtr']['gate_calibration_context']
        if status.get('status') != 'SUCCESS' or metrics.get('status') != 'SUCCESS':
            raise RuntimeError('Formal run not complete: ' + job['id'])
        if status['exit_code'] != 0:
            raise RuntimeError('Formal exit code is nonzero: ' + job['id'])
        resolved = read(run / 'resolved_config.yaml')
        if resolved['smoke']['enabled']:
            raise ValueError('Smoke cannot enter formal comparison')
        baseline_id = config['experiment']['baseline_experiment_id']
        baseline = read(ROOT / 'runs' / baseline_id / 'metrics.json')
        source_id = config['experiment']['source_experiment_id']
        source = read(ROOT / 'runs' / source_id / 'metrics.json')
        if baseline.get('status') != 'SUCCESS' or source.get('status') != 'SUCCESS':
            raise ValueError('Reference not successful')
        if metrics['evaluation_assets'] != baseline['evaluation_assets'] or metrics['evaluation_assets'] != source['evaluation_assets']:
            raise ValueError('Evaluation assets differ')
        source_resolved = read(ROOT / 'runs' / source_id / 'resolved_config.yaml')
        excluded = {'model', 'model_id', 'checkpoints', 'trustgtr'}
        diffs = {key: (source_resolved['official_args'].get(key), value)
                 for key, value in resolved['official_args'].items()
                 if key not in excluded and source_resolved['official_args'].get(key) != value}
        if diffs:
            raise ValueError('Training protocol changed: ' + str(diffs))
        row = {key: metrics.get(key) for key in ('experiment_id', 'dataset', 'model', 'mse', 'mae',
            'runtime_seconds', 'training_time_seconds', 'inference_time_seconds', 'peak_gpu_memory_mib',
            'parameters', 'epochs_completed', 'best_epoch', 'checkpoint', 'trustgtr_diagnostics')}
        row.update(status='SUCCESS', exit_code=0, context=context, seed=metrics['seed'],
            baseline_experiment_id=baseline_id, source_experiment_id=source_id,
            baseline_scores={name: baseline[name] for name in ('mse', 'mae')},
            source_scores={name: source[name] for name in ('mse', 'mae')},
            relative_change_vs_gtr_pct={name: 100 * (metrics[name] - baseline[name]) / baseline[name] for name in ('mse', 'mae')},
            relative_change_vs_a6_pct={name: 100 * (metrics[name] - source[name]) / source[name] for name in ('mse', 'mae')},
            metrics_path=str(run / 'metrics.json'), metrics_sha256=digest(run / 'metrics.json'),
            checkpoint_sha256=digest(Path(metrics['checkpoint'])),
            resolved_config_path=str(run / 'resolved_config.yaml'), resolved_config_sha256=digest(run / 'resolved_config.yaml'),
            same_training_budget_and_evaluation_assets=True)
        rows.append(row)
        pairs.setdefault(dataset, {})[context] = (row, resolved)
    if len(rows) != 8 or len(pairs) != 4:
        raise ValueError('Expected eight formal runs across four datasets')
    comparisons = []
    for dataset, pair in pairs.items():
        train, a = pair['train']
        inference, b = pair['inference']
        filtered = {key: value for key, value in a['official_args'].items() if key not in excluded}
        if filtered != {key: value for key, value in b['official_args'].items() if key not in excluded}:
            raise ValueError('Pair official settings differ')
        differences = {key: (a['trustgtr'].get(key), value) for key, value in b['trustgtr'].items()
                       if a['trustgtr'].get(key) != value}
        if differences != {'gate_calibration_context': ('train', 'inference')}:
            raise ValueError('Pair must change only calibration context: ' + str(differences))
        if train['parameters'] != inference['parameters']:
            raise ValueError('Pair parameter counts differ')
        comparisons.append(dict(dataset=dataset,
            train_experiment_id=train['experiment_id'], inference_experiment_id=inference['experiment_id'],
            scores_train={name: train[name] for name in ('mse', 'mae')},
            scores_inference={name: inference[name] for name in ('mse', 'mae')},
            absolute_change_inference_vs_train={name: inference[name] - train[name] for name in ('mse', 'mae')},
            relative_change_inference_vs_train_pct={name: 100 * (inference[name] - train[name]) / train[name] for name in ('mse', 'mae')},
            independent_variable= differences))
    report = dict(step='6.4', formal_runs_complete=8, datasets=4, single_seed_exploratory=True,
        diagnostics_cumulative_train_val_test_are_not_best_validation=True,
        speed_not_strict_benchmark_due_to_shared_gpu_concurrency=True,
        experiments=rows, comparisons=comparisons)
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError('Refuse to overwrite final report ' + str(output))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 6.4 推理一致校准配对正式结果', '',
        '四数据集、T96/H96、seed2024、官方预算与固定资产；仅辅助校准上下文train→inference变化。负变化代表误差下降。', '',
        '| dataset | train MSE | train MAE | inference MSE | inference MAE | MSE change % | MAE change % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for pair in comparisons:
        a, b, d = pair['scores_train'], pair['scores_inference'], pair['relative_change_inference_vs_train_pct']
        lines.append('| {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} | {:.4f} | {:.4f} |'.format(
            pair['dataset'], a['mse'], a['mae'], b['mse'], b['mae'], d['mse'], d['mae']))
    lines += ['', '与原GTR/A6的比较、成本、完整配置/指标/checkpoint哈希见JSON。',
        '累计gate统计混合train/val/test，门控机制结论须另查best checkpoint固定val诊断。',
        '单seed只筛方向，不能宣称统计稳定或投稿等级；后续保留全部负结果，未按test重新选择checkpoint。']
    output.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(str(output))


if __name__ == '__main__':
    main()
