#!/usr/bin/env python3
"""Prepare two official, single-seed baselines and their independent smoke jobs."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='configs/queue_baseline.json')
    cli = parser.parse_args()
    output = ROOT / cli.output
    if output.exists():
        raise SystemExit(f'Queue already exists: {output}; resume it instead of replacing IDs.')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    sources = {
        'etth1': dict(data='ETTh1', data_path='ETTh1.csv', enc_in=7, cycle=24,
                      batch_size=256, learning_rate=0.001, dropout=0.5, use_revin=1,
                      paper_mse=0.367, paper_mae=0.391,
                      expected_hash='f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066'),
        'solar': dict(data='Solar', data_path='solar_AL.txt', enc_in=137, cycle=144,
                      batch_size=64, learning_rate=0.003, dropout=0.0, use_revin=0,
                      paper_mse=0.176, paper_mae=0.235,
                      expected_hash='230327ef72d2abb387939d4a35d6fd34f1066071bc7c40ce7ecf5531a0122ac2'),
    }
    jobs = []
    assets = []
    for dataset, spec in sources.items():
        asset_path = ROOT / 'dataset' / spec['data_path']
        actual_hash = digest(asset_path)
        if actual_hash != spec['expected_hash']:
            raise SystemExit(f'DATA_ERROR: checksum mismatch for {asset_path}')
        asset = dict(path=str(asset_path), sha256=actual_hash, size_bytes=asset_path.stat().st_size,
                     provenance='Existing public-format asset copied without preprocessing changes')
        assets.append(dict(dataset=dataset, **asset))
        smoke_id = f'gtr__{dataset}__smoke__t96_h96__seed2024__{stamp}'
        formal_id = f'gtr__{dataset}__forecast__t96_h96__seed2024__{stamp}'
        for is_smoke, run_id in [(True, smoke_id), (False, formal_id)]:
            args = {key: spec[key] for key in ('data', 'data_path', 'enc_in', 'cycle',
                    'batch_size', 'learning_rate', 'dropout', 'use_revin')}
            args.update(is_training=1, root_path=str(ROOT / 'dataset') + '/',
                        model_id=run_id, model='GTR', features='M', seq_len=96,
                        pred_len=96, train_epochs=1 if is_smoke else 30,
                        patience=5, itr=1, random_seed=2024)
            run_dir = ROOT / 'runs' / run_id
            config = {
                'official_args': args,
                'experiment': {'id': run_id, 'dataset': dataset, 'model': 'GTR',
                               'stage': 'smoke' if is_smoke else 'baseline',
                               'protocol': 'official_gtr_t96_h96_seed2024',
                               'paper_mse': spec['paper_mse'], 'paper_mae': spec['paper_mae'],
                               'paper_reference': 'Table 7/10 H96; reported five-seed mean, seeds 2025-2029',
                               'paper_url': 'https://arxiv.org/html/2602.10847v1',
                               'same_evaluation_protocol': not is_smoke,
                               'comparison_note': 'Single seed 2024 versus paper multi-seed reference; not a statistical reproduction claim',
                               'official_script': f'scripts/GTR/{dataset}.sh',
                               'config_changes': ([{'key': 'train_epochs', 'old': 30, 'new': 1,
                                                   'reason': 'Independent smoke only, explicitly authorized'}]
                                                  if is_smoke else [])},
                'smoke': {'enabled': is_smoke, 'max_train_batches': 3 if is_smoke else None,
                          'max_eval_batches': 2 if is_smoke else None},
                'data_asset': asset,
            }
            config_path = ROOT / 'configs' / f'{run_id}.json'
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
            jobs.append({'id': run_id, 'argv': [str(ROOT / '.conda/bin/python'), '-u',
                        str(ROOT / 'scripts/run_experiment.py'), '--config', str(config_path),
                        '--run-dir', str(run_dir)], 'cwd': str(ROOT), 'run_dir': str(run_dir),
                        'required_memory_mib': None, 'is_smoke': is_smoke,
                        'smoke_passed': smoke_id if not is_smoke else False,
                        'depends_on': [smoke_id] if not is_smoke else [],
                        **({'memory_estimate_from': smoke_id} if not is_smoke else {})})
    queue = {'max_jobs': 3, 'poll_seconds': 3, 'oom_cooldown_seconds': 60,
             'on_change_argv': [str(ROOT / '.conda/bin/python'), str(ROOT / 'scripts/summarize_results.py')],
             'jobs': jobs}
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'artifacts/data_assets.json').write_text(json.dumps(assets, ensure_ascii=False, indent=2) + '\n')
    print(output)
    for job in jobs:
        print(job['id'], 'smoke' if job['is_smoke'] else 'official formal', 'depends_on', job['depends_on'])


if __name__ == '__main__':
    main()
