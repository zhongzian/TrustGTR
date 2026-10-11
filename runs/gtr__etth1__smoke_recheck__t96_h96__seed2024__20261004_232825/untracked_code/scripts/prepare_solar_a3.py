#!/usr/bin/env python3
"""Prepare the missing Solar A3-only comparison; do not start experiments."""
import copy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'configs/queue_solar_a3.json'
A3_SOURCE = ROOT / 'configs/trustgtr_a3__etth1__forecast__t96_h96__seed2024__20261004_223023.json'


def write_new(path, payload):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def main():
    if OUTPUT.exists():
        raise SystemExit('Existing Solar A3 queue is immutable; use the scheduler resume command.')
    queue = json.loads((ROOT / 'configs/queue_baseline_v3.json').read_text())
    for job in queue['jobs']:
        metrics = json.loads((Path(job['run_dir']) / 'metrics.json').read_text())
        if metrics['status'] != 'SUCCESS':
            raise SystemExit('All retained baseline jobs must already have completed successfully.')
    source = next(job for job in queue['jobs'] if '__solar__forecast__' in job['id'])
    baseline = json.loads(Path(source['argv'][source['argv'].index('--config') + 1]).read_text())
    measured = json.loads((Path(source['run_dir']) / 'metrics.json').read_text())
    a3 = json.loads(A3_SOURCE.read_text())
    options = copy.deepcopy(a3['trustgtr'])
    if options != dict(use_multi_cycle=True, preserve_initialization_rng=True,
                       memory_mode='factorized_residual'):
        raise SystemExit('ETTh1 A3 source options changed; review before generating this comparison.')
    cycle = baseline['official_args']['cycle']
    options.update(periods=[cycle, cycle * 7, cycle * 30],
                   use_reliability_gate=False, use_regime_contrast=False)
    args = baseline['official_args']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    prefix = f"trustgtr_a3__solar__{{stage}}__t{args['seq_len']}_h{args['pred_len']}__seed{args['random_seed']}__{stamp}"
    smoke_id = prefix.format(stage='smoke')
    formal_id = prefix.format(stage='forecast')
    for smoke, run_id in [(True, smoke_id), (False, formal_id)]:
        config = copy.deepcopy(baseline)
        config['official_args'].update(model='TrustGTR', model_id=run_id)
        if smoke:
            config['official_args']['train_epochs'] = 1
        config['trustgtr'] = copy.deepcopy(options)
        config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                               max_eval_batches=2 if smoke else None)
        changes = [
            dict(key='model', old='GTR', new='TrustGTR', reason='Solar A3 single-factor attribution'),
            dict(key='trustgtr', old='all modules disabled', new=copy.deepcopy(options),
                 reason='Use the existing ETTh1 A3 method unchanged; B and C remain disabled'),
            dict(key='periods', old=[cycle], new=options['periods'],
                 reason='Apply the unchanged base/7x/30x period rule to Solar cycle=144'),
            dict(key='memory_mode', old='single official base-cycle bank', new='factorized_residual',
                 reason='Exactly the previously tested A3 level/trend/amplitude residual memory'),
            dict(key='preserve_initialization_rng', old=False, new=True,
                 reason='Preserve the official baseline data/dropout RNG stream after new module initialization'),
        ]
        if smoke:
            changes.append(dict(key='train_epochs', old=args['train_epochs'], new=1,
                                reason='Independent smoke only; formal budget remains identical to Solar baseline'))
        config['experiment'].update(
            id=run_id, model='TrustGTR_a3', stage='smoke' if smoke else 'improvement',
            same_evaluation_protocol=not smoke, baseline_experiment_id=source['id'],
            baseline_mse=measured['mse'], baseline_mae=measured['mae'],
            comparison_note='Missing Solar A3-only attribution: same seed, budget, data/assets and evaluation as completed baseline',
            config_changes=changes, method_source_config=str(A3_SOURCE))
        config_path = ROOT / 'configs' / (run_id + '.json')
        run_dir = ROOT / 'runs' / run_id
        if config_path.exists() or run_dir.exists():
            raise SystemExit('Experiment ID already exists; regenerate with a new timestamp.')
        write_new(config_path, config)
        queue['jobs'].append(dict(
            id=run_id, argv=[str(ROOT / '.conda/bin/python'), '-u',
                            str(ROOT / 'scripts/run_trust_experiment.py'), '--config',
                            str(config_path), '--run-dir', str(run_dir)],
            cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
            is_smoke=smoke, smoke_passed=False if smoke else smoke_id,
            depends_on=[source['id']] if smoke else [smoke_id],
            **({'memory_estimate_from': smoke_id} if not smoke else {})))
        print(run_id)
    write_new(OUTPUT, queue)
    print(OUTPUT)


if __name__ == '__main__':
    main()
