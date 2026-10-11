#!/usr/bin/env python3
"""Prepare router/C complementarity and append an unlaunched gate-risk test."""
import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'configs/queue_v7_complementarity.json'


def load_source(job):
    path = Path(job['argv'][job['argv'].index('--config') + 1])
    metrics_path = Path(job['run_dir']) / 'metrics.json'
    config = json.loads(path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if metrics.get('status') != 'SUCCESS':
        raise ValueError('Source is not successfully completed: ' + job['id'])
    provenance = dict(config_path=str(path), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metrics_path=str(metrics_path), metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest())
    return config, metrics, provenance


def prepare_complementarity():
    if OUTPUT.exists():
        raise SystemExit('Existing queue is immutable; resume it instead of overwriting')
    queue = json.loads((ROOT / 'configs/queue_v6_router.json').read_text())
    sources = [
        ('solar', 'a6', next(job for job in queue['jobs']
            if job['id'].startswith('trustgtr_a5__solar__forecast__'))),
        ('etth1', 'ac7', next(job for job in queue['jobs']
            if job['id'].startswith('trustgtr_ac5__etth1__forecast__')))]
    rechecks = {dataset: next(job for job in reversed(queue['jobs'])
        if job['id'].startswith(f'gtr__{dataset}__smoke_recheck__')) for dataset in ('solar', 'etth1')}
    for recheck in rechecks.values():
        load_source(recheck)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    generated = []
    for dataset, variant, source in sources:
        original, metrics, provenance = load_source(source)
        options = original['trustgtr']
        assert options['use_multi_cycle'] and not options['use_reliability_gate']
        assert options['use_regime_contrast'] == (dataset == 'etth1')
        assert options.get('historical_router_mode', 'global_topk') == 'global_topk'
        assert options.get('contrast_metric', 'cosine') == 'cosine'
        assert options.get('contrast_temperature', 0.1) == 0.1
        assert options['contrast_loss_weight'] == 0.002
        assert options['historical_strength_coordinates'] == 'bounded'
        smoke_id = f'trustgtr_{variant}__{dataset}__smoke__t96_h96__seed2024__{stamp}'
        formal_id = smoke_id.replace('__smoke__', '__forecast__')
        for smoke in (True, False):
            run_id = smoke_id if smoke else formal_id
            config = copy.deepcopy(original)
            config['official_args'].update(model_id=run_id,
                train_epochs=1 if smoke else original['official_args']['train_epochs'])
            config['trustgtr']['historical_router_mode'] = 'per_bank'
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                  max_eval_batches=2 if smoke else None)
            changes = copy.deepcopy(original['experiment']['config_changes'])
            changes.extend([
                dict(key='trustgtr.historical_router_mode', old='global_topk', new='per_bank',
                    reason=('single-factor Solar router transfer from A5; B/C stay disabled' if dataset == 'solar' else
                            'complete ETTh1 A5/AC5/A6/AC7 two-by-two router/C attribution; preserve original cosine C and its scale')),
                dict(key='official_args.model_id', old=original['official_args']['model_id'], new=run_id,
                     reason='unique follow-up experiment ID; preserve completed source')])
            if smoke:
                changes.extend([
                    dict(key='official_args.train_epochs', old=original['official_args']['train_epochs'], new=1,
                         reason='independent smoke only; full formal budget remains unchanged'),
                    dict(key='smoke', old=original['smoke'], new=config['smoke'],
                         reason='three train/two evaluation batches for smoke only')])
            config['experiment'].update(id=run_id, model='TrustGTR_' + variant,
                stage='smoke' if smoke else 'improvement', same_evaluation_protocol=not smoke,
                source_experiment_id=source['id'], source_mse=metrics['mse'], source_mae=metrics['mae'],
                source_provenance=provenance,
                fixed_evaluation_asset_source=original['experiment']['baseline_experiment_id'],
                baseline_recheck_experiment_id=rechecks[dataset]['id'],
                comparison_note='Single-factor per-bank router versus completed source; unchanged scientific code and original seed/budget/assets',
                config_changes=changes)
            path = ROOT / 'configs' / (run_id + '.json')
            run_dir = ROOT / 'runs' / run_id
            if path.exists() or run_dir.exists():
                raise ValueError('Experiment ID already exists: ' + run_id)
            job = dict(id=run_id,
                argv=[str(ROOT / '.conda/bin/python'), '-u', str(ROOT / 'scripts/run_trust_experiment.py'),
                      '--config', str(path), '--run-dir', str(run_dir)],
                cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
                is_smoke=smoke, smoke_passed=False,
                depends_on=[rechecks[dataset]['id'], source['id']] if smoke else [smoke_id])
            if not smoke:
                job['memory_estimate_from'] = smoke_id
            queue['jobs'].append(job)
            generated.append((path, config))
    ids = [job['id'] for job in queue['jobs']]
    assert queue['max_jobs'] == 3 and len(ids) == len(set(ids))
    assert all(dep in ids for job in queue['jobs'] for dep in job['depends_on'])
    for path, config in generated:
        with path.open('x') as file:
            file.write(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    with OUTPUT.open('x') as file:
        file.write(json.dumps(queue, indent=2) + '\n')
    print(OUTPUT)
    for _, config in generated:
        print(config['experiment']['id'])


def append_risk():
    queue = json.loads(OUTPUT.read_text())
    if any(job['id'].startswith('trustgtr_ab7__') for job in queue['jobs']):
        raise SystemExit('Risk jobs already exist; do not duplicate or overwrite them')
    for job in queue['jobs']:
        if job['id'].startswith(('trustgtr_ac7__', 'trustgtr_a6__solar__')) and Path(job['run_dir']).exists():
            raise SystemExit('The v7 queue has already launched; create a new queue instead')
    sources = {dataset: next(job for job in queue['jobs']
        if job['id'].startswith(f'trustgtr_ab6__{dataset}__forecast__')) for dataset in ('etth1', 'solar')}
    rechecks = {dataset: next(job for job in reversed(queue['jobs'])
        if job['id'].startswith(f'gtr__{dataset}__smoke_recheck__')) for dataset in sources}
    for recheck in rechecks.values():
        load_source(recheck)
    diagnostic_path = ROOT / 'reports/v6_fixed_full_val_20261005.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic['no_test_targets_used'] and diagnostic['no_test_dataset_constructed']
    diagnostic_hash = hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    generated = []
    for dataset, source in sources.items():
        original, metrics, provenance = load_source(source)
        options = original['trustgtr']
        assert options['use_reliability_gate'] and options['gate_descriptor'] == 'retrieval'
        assert options['gate_loss_mode'] == 'power_weighted'
        assert options.get('gate_loss_weight', 0.05) == 0.05
        assert options.get('gate_initial_probability', 0.9) == 0.9
        assert not options['use_regime_contrast']
        smoke_id = f'trustgtr_ab7__{dataset}__smoke__t96_h96__seed2024__{stamp}'
        formal_id = smoke_id.replace('__smoke__', '__forecast__')
        for smoke in (True, False):
            run_id = smoke_id if smoke else formal_id
            config = copy.deepcopy(original)
            config['official_args'].update(model_id=run_id,
                train_epochs=1 if smoke else original['official_args']['train_epochs'])
            config['trustgtr']['gate_loss_mode'] = 'counterfactual_risk'
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                  max_eval_batches=2 if smoke else None)
            changes = copy.deepcopy(original['experiment']['config_changes'])
            changes.extend([
                dict(key='trustgtr.gate_loss_mode', old='power_weighted', new='counterfactual_risk',
                    reason='single-factor B loss: detached baseline error e and correction d define normalized counterfactual excess MSE (P*g^2-2*U*g)/(E+P); clipped oracle supervision discards benefit magnitude; fixed full-val AB6 gate/gain correlation -0.0986 and no-calibration gate nearly always open motivate this test; this is not a baseline safety guarantee'),
                dict(key='official_args.model_id', old=original['official_args']['model_id'], new=run_id,
                     reason='unique follow-up experiment ID; preserve completed AB6 source')])
            if smoke:
                changes.extend([
                    dict(key='official_args.train_epochs', old=original['official_args']['train_epochs'], new=1,
                         reason='independent smoke only; full formal budget remains unchanged'),
                    dict(key='smoke', old=original['smoke'], new=config['smoke'],
                         reason='three train/two evaluation batches for smoke only')])
            config['experiment'].update(id=run_id, model='TrustGTR_ab7',
                stage='smoke' if smoke else 'improvement', same_evaluation_protocol=not smoke,
                source_experiment_id=source['id'], source_mse=metrics['mse'], source_mae=metrics['mae'],
                source_provenance=provenance,
                diagnostic_source=dict(path=str(diagnostic_path), sha256=diagnostic_hash,
                    scope='fixed full ETTh1 validation only; no test targets; motivates gate loss on both datasets'),
                fixed_evaluation_asset_source=original['experiment']['baseline_experiment_id'],
                baseline_recheck_experiment_id=rechecks[dataset]['id'],
                comparison_note='Single-factor gate loss versus completed AB6; same scalar gate, retrieval descriptor, initial probability, loss weight, seed/budget/assets; no safety guarantee',
                config_changes=changes)
            path = ROOT / 'configs' / (run_id + '.json')
            run_dir = ROOT / 'runs' / run_id
            if path.exists() or run_dir.exists():
                raise ValueError('Experiment ID already exists: ' + run_id)
            job = dict(id=run_id,
                argv=[str(ROOT / '.conda/bin/python'), '-u', str(ROOT / 'scripts/run_trust_experiment.py'),
                      '--config', str(path), '--run-dir', str(run_dir)],
                cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
                is_smoke=smoke, smoke_passed=False,
                depends_on=[rechecks[dataset]['id'], source['id']] if smoke else [smoke_id])
            if not smoke:
                job['memory_estimate_from'] = smoke_id
            queue['jobs'].append(job)
            generated.append((path, config))
    ids = [job['id'] for job in queue['jobs']]
    assert queue['max_jobs'] == 3 and len(ids) == len(set(ids))
    assert all(dep in ids for job in queue['jobs'] for dep in job['depends_on'])
    for path, config in generated:
        with path.open('x') as file:
            file.write(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    temporary = OUTPUT.with_suffix('.tmp')
    with temporary.open('x') as file:
        file.write(json.dumps(queue, indent=2) + '\n')
    temporary.replace(OUTPUT)
    print(OUTPUT)
    for _, config in generated:
        print(config['experiment']['id'])


def append_baseline_regression():
    queue = json.loads(OUTPUT.read_text())
    if any(job['id'].startswith('gtr__etth1__smoke_recheck_v7__') for job in queue['jobs']):
        raise SystemExit('Fresh v7 baseline smoke already exists; do not duplicate it')
    current_smokes = [job for job in queue['jobs'] if job['is_smoke'] and
        job['id'].startswith(('trustgtr_ac7__', 'trustgtr_ab7__', 'trustgtr_a6__solar__'))]
    if not current_smokes or any(Path(job['run_dir']).exists() for job in current_smokes):
        raise SystemExit('The v7 queue is missing model smokes or has already launched')
    baseline = next(job for job in queue['jobs']
        if job['id'].startswith('gtr__etth1__forecast__') and '20261004_214520' in job['id'])
    config, _, provenance = load_source(baseline)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    run_id = f'gtr__etth1__smoke_recheck_v7__t96_h96__seed2024__{stamp}'
    original_args = copy.deepcopy(config['official_args'])
    original_smoke = copy.deepcopy(config['smoke'])
    config['official_args'].update(model_id=run_id, train_epochs=1)
    config['smoke'] = dict(enabled=True, max_train_batches=3, max_eval_batches=2)
    config['experiment'].update(id=run_id, stage='smoke', same_evaluation_protocol=False,
        source_experiment_id=baseline['id'], source_provenance=provenance,
        comparison_note='Fresh original GTR regression smoke after v7 gate-risk scientific code change; not a formal result',
        config_changes=[
            dict(key='official_args.model_id', old=original_args['model_id'], new=run_id,
                 reason='unique v7 baseline regression smoke; preserve successful baseline'),
            dict(key='official_args.train_epochs', old=original_args['train_epochs'], new=1,
                 reason='independent smoke only; no change to any formal budget'),
            dict(key='smoke', old=original_smoke, new=config['smoke'],
                 reason='three train/two evaluation batches to recheck original baseline after innovation change')])
    path = ROOT / 'configs' / (run_id + '.json')
    run_dir = ROOT / 'runs' / run_id
    if path.exists() or run_dir.exists():
        raise ValueError('Experiment ID already exists: ' + run_id)
    job = dict(id=run_id,
        argv=[str(ROOT / '.conda/bin/python'), '-u', str(ROOT / 'scripts/run_experiment.py'),
              '--config', str(path), '--run-dir', str(run_dir)],
        cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
        is_smoke=True, smoke_passed=False, depends_on=[baseline['id']])
    queue['jobs'].append(job)
    for model_smoke in current_smokes:
        model_smoke['depends_on'].append(run_id)
    # Normalize legacy manifest fields; eligibility still checks recorded smoke dependencies.
    for existing in queue['jobs']:
        existing['smoke_passed'] = existing.get('smoke_passed') is True
    with path.open('x') as file:
        file.write(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    temporary = OUTPUT.with_suffix('.tmp')
    with temporary.open('x') as file:
        file.write(json.dumps(queue, indent=2) + '\n')
    temporary.replace(OUTPUT)
    print(OUTPUT)
    print(run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--add-risk', action='store_true',
                        help='append AB7 gate-risk smokes/formals to the unlaunched complementarity manifest')
    parser.add_argument('--add-baseline-regression', action='store_true',
                        help='append a fresh GTR smoke and require it before all current v7 model smokes')
    args = parser.parse_args()
    if args.add_risk and args.add_baseline_regression:
        parser.error('Append one stage at a time')
    if args.add_baseline_regression:
        append_baseline_regression()
    elif args.add_risk:
        append_risk()
    else:
        prepare_complementarity()


if __name__ == '__main__':
    main()
