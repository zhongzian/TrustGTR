#!/usr/bin/env python3
"""Read persisted results, including failures; never infer absent metrics."""
import csv
import fcntl
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['experiment_id', 'dataset', 'model', 'stage', 'seed', 'seq_len', 'pred_len',
          'mse', 'mae', 'paper_mse', 'paper_mae', 'mse_delta_vs_paper', 'mae_delta_vs_paper',
          'mse_relative_change_pct', 'mae_relative_change_pct', 'runtime_seconds',
          'peak_memory_mib', 'parameters', 'status', 'error', 'same_evaluation_protocol',
          'git_commit', 'resolved_config_path', 'checkpoint_path',
          'baseline_experiment_id', 'baseline_mse', 'baseline_mae',
          'mse_delta_vs_baseline', 'mae_delta_vs_baseline', 'mse_reduction_pct', 'mae_reduction_pct']


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return {}


def summarize():
    rows = []
    for config_path in sorted((ROOT / 'configs').glob('*.json')):
        config = read_json(config_path)
        if 'experiment' not in config:
            continue
        exp = config['experiment']
        args = config['official_args']
        run_dir = ROOT / 'runs' / exp['id']
        metrics = read_json(run_dir / 'metrics.json')
        status = read_json(run_dir / 'status.json')
        queued = read_json(run_dir / 'queue_status.json')
        resources = read_json(run_dir / 'resource_usage.json')
        row = dict(experiment_id=exp['id'], dataset=exp['dataset'], model=exp['model'],
                   stage=exp['stage'], seed=args['random_seed'], seq_len=args['seq_len'],
                   pred_len=args['pred_len'], mse=metrics.get('mse'), mae=metrics.get('mae'),
                   paper_mse=exp.get('paper_mse'), paper_mae=exp.get('paper_mae'),
                   runtime_seconds=metrics.get('runtime_seconds', resources.get('elapsed_seconds')),
                   peak_memory_mib=metrics.get('peak_gpu_memory_mib', resources.get('peak_gpu_memory_mib')),
                   parameters=metrics.get('parameters', metrics.get('parameter_count')),
                   status=queued.get('status', status.get('status', 'pending')),
                   error=queued.get('last_error_excerpt') or status.get('error', ''),
                   same_evaluation_protocol=exp.get('same_evaluation_protocol', False),
                   git_commit=metrics.get('git_commit'),
                   resolved_config_path=str(run_dir / 'resolved_config.yaml'),
                   checkpoint_path=metrics.get('checkpoint', metrics.get('checkpoint_path', metrics.get('best_checkpoint'))),
                   baseline_experiment_id=exp.get('baseline_experiment_id'),
                   baseline_mse=exp.get('baseline_mse'), baseline_mae=exp.get('baseline_mae'))
        for name in ('mse', 'mae'):
            current, paper = row[name], row['paper_' + name]
            if exp['stage'] != 'smoke' and current is not None and paper is not None:
                row[name + '_delta_vs_paper'] = current - paper
                row[name + '_relative_change_pct'] = 100 * (current - paper) / paper
            baseline = row['baseline_' + name]
            if exp['stage'] != 'smoke' and current is not None and baseline is not None:
                row[name + '_delta_vs_baseline'] = current - baseline
                row[name + '_reduction_pct'] = 100 * (baseline - current) / baseline
        rows.append(row)
    rows.sort(key=lambda r: (r['dataset'], r['model'], r['seed'], r['stage'], r['experiment_id']))
    target = ROOT / 'reports'
    target.mkdir(exist_ok=True)
    csv_path = target / 'latest_results.csv'
    csv_tmp = csv_path.with_suffix('.csv.tmp')
    with csv_tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: 'NA' if row.get(key) is None else row.get(key) for key in FIELDS})
    csv_tmp.replace(csv_path)
    columns = ['experiment_id', 'dataset', 'stage', 'mse', 'mae', 'baseline_mse', 'baseline_mae',
               'mse_reduction_pct', 'mae_reduction_pct', 'paper_mse', 'paper_mae', 'status']
    lines = ['# Latest persisted results', '',
             'Paper values are H96 five-seed references; current runs use seed 2024. Smoke is not comparable.', '',
             '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(key, 'NA')).replace('|', '/')
                                      if row.get(key) is not None else 'NA' for key in columns) + ' |')
    content = '\n'.join(lines) + '\n'
    md_path = target / 'latest_results.md'
    md_tmp = md_path.with_suffix('.md.tmp')
    md_tmp.write_text(content)
    md_tmp.replace(md_path)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    archive = target / 'archive'
    archive.mkdir(exist_ok=True)
    (archive / f'{stamp}.md').write_text(content)
    (archive / f'{stamp}.csv').write_bytes(csv_path.read_bytes())
    print(f'Summarized {len(rows)} runs: {md_path}')


def main():
    (ROOT / 'runtime').mkdir(exist_ok=True)
    with (ROOT / 'runtime/report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summarize()


if __name__ == '__main__':
    main()
