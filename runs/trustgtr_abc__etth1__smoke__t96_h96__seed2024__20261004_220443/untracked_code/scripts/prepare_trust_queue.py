#!/usr/bin/env python3
"""Single-seed, equal-budget screen; each configuration gets its own smoke."""
import copy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'configs/queue_trust_screen.json'
    if output.exists():
        raise SystemExit('Existing screen queue is immutable; resume it or use a fresh ID')
    baseline_queue = json.loads((ROOT / 'configs/queue_baseline_v3.json').read_text())
    # Retain completed baseline jobs in the single shared queue state. They
    # cannot be relaunched; subsequent jobs also verify their prerequisites.
    jobs = baseline_queue['jobs']
    source = next(job for job in jobs if not job['is_smoke'] and '__etth1__' in job['id'])
    baseline = json.loads(Path(source['argv'][source['argv'].index('--config') + 1]).read_text())
    baseline_metrics = json.loads((Path(source['run_dir']) / 'metrics.json').read_text())
    if baseline_metrics['status'] != 'SUCCESS':
        raise SystemExit('A comparable official ETTh1 baseline must complete first')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    recheck_id = f'gtr__etth1__smoke_recheck__t96_h96__seed2024__{stamp}'
    recheck = copy.deepcopy(baseline)
    recheck['official_args'].update(model_id=recheck_id, train_epochs=1)
    recheck['experiment'].update(id=recheck_id, stage='smoke', same_evaluation_protocol=False,
                                comparison_note='Recheck original baseline after innovation integration',
                                config_changes=[{'key': 'train_epochs', 'old': 30, 'new': 1, 'reason': 'smoke recheck only'}])
    recheck['smoke'] = dict(enabled=True, max_train_batches=3, max_eval_batches=2)
    recheck_path = ROOT / 'configs' / (recheck_id + '.json')
    recheck_path.write_text(json.dumps(recheck, ensure_ascii=False, indent=2) + '\n')
    recheck_run = ROOT / 'runs' / recheck_id
    jobs.append(dict(id=recheck_id, argv=[str(ROOT / '.conda/bin/python'), '-u',
                     str(ROOT / 'scripts/run_experiment.py'), '--config', str(recheck_path),
                     '--run-dir', str(recheck_run)], cwd=str(ROOT), run_dir=str(recheck_run),
                     required_memory_mib=None, is_smoke=True, smoke_passed=False,
                     depends_on=[source['id']], memory_estimate_from=source['depends_on'][0]))
    variants = [('b', dict(use_reliability_gate=True)),
                ('a', dict(use_multi_cycle=True)),
                ('ac', dict(use_multi_cycle=True, use_regime_contrast=True)),
                ('abc', dict(use_multi_cycle=True, use_reliability_gate=True, use_regime_contrast=True))]
    formal_ids = {}
    for variant, options in variants:
        smoke_id = f'trustgtr_{variant}__etth1__smoke__t96_h96__seed2024__{stamp}'
        formal_id = f'trustgtr_{variant}__etth1__forecast__t96_h96__seed2024__{stamp}'
        formal_ids[variant] = formal_id
        for smoke, run_id in [(True, smoke_id), (False, formal_id)]:
            config = copy.deepcopy(baseline)
            config['official_args'].update(model='TrustGTR', model_id=run_id, train_epochs=1 if smoke else 30)
            config['trustgtr'] = options
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                   max_eval_batches=2 if smoke else None)
            exp = config['experiment']
            exp.update(id=run_id, model='TrustGTR_' + variant, stage='smoke' if smoke else 'improvement',
                       baseline_experiment_id=source['id'],
                       baseline_mse=baseline_metrics['mse'], baseline_mae=baseline_metrics['mae'],
                       same_evaluation_protocol=not smoke,
                       comparison_note='Exact official seed-2024 baseline, same budget/assets/checkpoint selection',
                       config_changes=[{'key': 'model', 'old': 'GTR', 'new': 'TrustGTR', 'reason': variant},
                                       {'key': 'trustgtr', 'old': 'all modules disabled', 'new': options,
                                        'reason': 'B alone; A alone; C incremental on A; full A/B/C combination'}] +
                                      ([{'key': 'train_epochs', 'old': 30, 'new': 1, 'reason': 'independent smoke only'}] if smoke else []))
            if options.get('use_multi_cycle'):
                exp['config_changes'].append({'key': 'periods', 'old': [24], 'new': [24, 168, 720],
                                              'reason': 'sample/channel-conditioned multi-cycle retrieval; absolute origin preserves base modulo'})
            if options.get('use_reliability_gate'):
                exp['config_changes'].append({'key': 'gate_loss_weight', 'old': 0, 'new': 0.05,
                                              'reason': 'train-only clipped optimal mixture coefficient supervision'})
            if options.get('use_regime_contrast'):
                exp['config_changes'].append({'key': 'contrast_loss_weight', 'old': 0, 'new': 0.01,
                                              'reason': 'train-only same-phase, disjoint-window regime contrast'})
            config_path = ROOT / 'configs' / (run_id + '.json')
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
            run_dir = ROOT / 'runs' / run_id
            dependencies = [recheck_id] if smoke else [smoke_id]
            if not smoke and variant == 'ac':
                dependencies.append(formal_ids['a'])
            if not smoke and variant == 'abc':
                dependencies.extend([formal_ids['a'], formal_ids['b'], formal_ids['ac']])
            jobs.append(dict(id=run_id, argv=[str(ROOT / '.conda/bin/python'), '-u',
                        str(ROOT / 'scripts/run_trust_experiment.py'), '--config', str(config_path),
                        '--run-dir', str(run_dir)], cwd=str(ROOT), run_dir=str(run_dir),
                        required_memory_mib=None, is_smoke=smoke, smoke_passed=False,
                        depends_on=dependencies,
                        **({'memory_estimate_from': smoke_id} if not smoke else {})))
    baseline_queue['jobs'] = jobs
    output.write_text(json.dumps(baseline_queue, ensure_ascii=False, indent=2) + '\n')
    print(output)
    for variant, run_id in formal_ids.items():
        print(variant, run_id)


if __name__ == '__main__':
    main()
