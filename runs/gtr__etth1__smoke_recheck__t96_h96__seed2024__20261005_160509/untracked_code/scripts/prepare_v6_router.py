#!/usr/bin/env python3
"""Prepare v6 router and reliability mechanisms, leaving all v5 runs untouched."""
import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'configs/queue_v6_router.json'


def source_provenance(job):
    config_path = Path(job['argv'][job['argv'].index('--config') + 1])
    metrics_path = Path(job['run_dir']) / 'metrics.json'
    return dict(config_path=str(config_path),
                config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                metrics_path=str(metrics_path),
                metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest())


def prepare_router():
    if OUTPUT.exists():
        raise SystemExit('Existing queue is immutable; resume it instead of overwriting')
    source_queue = json.loads((ROOT / 'configs/queue_historical_v5.json').read_text())
    source_jobs = {job['id']: job for job in source_queue['jobs']}
    source = next(job for job in source_queue['jobs']
                  if job['id'].startswith('trustgtr_a5__etth1__forecast__'))
    metrics = json.loads((Path(source['run_dir']) / 'metrics.json').read_text())
    if metrics.get('status') != 'SUCCESS':
        raise ValueError('A5 source must be successfully completed')
    config_path = Path(source['argv'][source['argv'].index('--config') + 1])
    source_config = json.loads(config_path.read_text())
    assert source_config['trustgtr']['use_multi_cycle']
    assert not source_config['trustgtr']['use_reliability_gate']
    assert not source_config['trustgtr']['use_regime_contrast']
    assert source_config['trustgtr']['historical_strength_coordinates'] == 'bounded'
    assert source_config['trustgtr'].get('historical_router_mode', 'global_topk') == 'global_topk'

    # Preserve successful baseline jobs and the exact completed A5 dependency chain.
    queue = json.loads((ROOT / 'configs/queue_baseline_v3.json').read_text())
    retained = {job['id'] for job in queue['jobs']}
    def retain(identifier):
        if identifier in retained:
            return
        job = source_jobs[identifier]
        for dependency in job['depends_on']:
            retain(dependency)
        completed = json.loads((Path(job['run_dir']) / 'metrics.json').read_text())
        if completed.get('status') != 'SUCCESS':
            raise ValueError('Source dependency is not successful: ' + identifier)
        queue['jobs'].append(copy.deepcopy(job))
        retained.add(identifier)
    retain(source['id'])

    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    smoke_id = f'trustgtr_a6__etth1__smoke__t96_h96__seed2024__{stamp}'
    formal_id = smoke_id.replace('__smoke__', '__forecast__')
    generated = []
    for smoke in (True, False):
        run_id = smoke_id if smoke else formal_id
        config = copy.deepcopy(source_config)
        config['official_args'].update(model_id=run_id,
            train_epochs=1 if smoke else source_config['official_args']['train_epochs'])
        config['trustgtr']['historical_router_mode'] = 'per_bank'
        config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                              max_eval_batches=2 if smoke else None)
        changes = copy.deepcopy(source_config['experiment']['config_changes'])
        changes.extend([
            dict(key='trustgtr.historical_router_mode', old='global_topk', new='per_bank',
                 reason='single-factor A6: each age bank supplies top-k before learned bank mixing; all other A5 settings unchanged'),
            dict(key='official_args.model_id', old=source_config['official_args']['model_id'], new=run_id,
                 reason='unique experiment ID; preserve completed A5 source')])
        if smoke:
            changes.extend([
                dict(key='official_args.train_epochs', old=source_config['official_args']['train_epochs'], new=1,
                     reason='independent smoke only; full formal training budget remains unchanged'),
                dict(key='smoke', old=source_config['smoke'], new=config['smoke'],
                     reason='three train/two evaluation batches for smoke only')])
        config['experiment'].update(id=run_id, model='TrustGTR_a6',
            stage='smoke' if smoke else 'improvement', same_evaluation_protocol=not smoke,
            source_experiment_id=source['id'],
            source_mse=metrics['mse'], source_mae=metrics['mae'],
            fixed_evaluation_asset_source=source_config['experiment']['baseline_experiment_id'],
            comparison_note='Single-factor router change relative to successful A5; same official seed/budget/assets, no test tuning',
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
            depends_on=[source['id']] if smoke else [smoke_id])
        if not smoke:
            job['memory_estimate_from'] = smoke_id
        queue['jobs'].append(job)
        generated.append((path, config))
    assert queue['max_jobs'] == 3
    ids = [job['id'] for job in queue['jobs']]
    assert len(ids) == len(set(ids))
    assert all(dep in ids for job in queue['jobs'] for dep in job['depends_on'])
    for path, config in generated:
        with path.open('x') as file:
            file.write(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    with OUTPUT.open('x') as file:
        file.write(json.dumps(queue, indent=2) + '\n')
    print(OUTPUT)
    print(smoke_id)
    print(formal_id)


def append_reliability():
    """Append descriptor tests only while the v6 manifest has never launched."""
    queue = json.loads(OUTPUT.read_text())
    if any(job['id'].startswith('trustgtr_ab6__') for job in queue['jobs']):
        raise SystemExit('Reliability jobs already exist; do not duplicate or overwrite them')
    for job in queue['jobs']:
        if job['id'].startswith('trustgtr_a6__') and Path(job['run_dir']).exists():
            raise SystemExit('The v6 queue has already launched; create a new queue instead')
    manifests = [json.loads((ROOT / name).read_text()) for name in
                 ('configs/queue_historical_v5.json', 'configs/queue_v5_followup.json',
                  'configs/queue_baseline_v3.json')]
    source_jobs = {job['id']: job for manifest in manifests for job in manifest['jobs']}
    sources = {
        'etth1': next(job for job in source_jobs.values()
                      if job['id'].startswith('trustgtr_ab5__etth1__forecast__')),
        'solar': next(job for job in source_jobs.values()
                     if job['id'].startswith('trustgtr_a5__solar__forecast__'))}
    baselines = {dataset: next(job for job in source_jobs.values()
        if job['id'].startswith(f'gtr__{dataset}__forecast__') and '20261004_214520' in job['id'])
        for dataset in sources}
    retained = {job['id'] for job in queue['jobs']}

    def load(job):
        config = json.loads(Path(job['argv'][job['argv'].index('--config') + 1]).read_text())
        metrics = json.loads((Path(job['run_dir']) / 'metrics.json').read_text())
        if metrics.get('status') != 'SUCCESS':
            raise ValueError('Source is not successfully completed: ' + job['id'])
        return config, metrics

    def retain(identifier):
        if identifier in retained:
            return
        job = source_jobs[identifier]
        load(job)
        for dependency in job['depends_on']:
            retain(dependency)
        queue['jobs'].append(copy.deepcopy(job))
        retained.add(identifier)

    for source in sources.values():
        retain(source['id'])
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    generated = []

    def add(config, dependencies, entry, memory_source=None):
        run_id = config['experiment']['id']
        path = ROOT / 'configs' / (run_id + '.json')
        run_dir = ROOT / 'runs' / run_id
        if path.exists() or run_dir.exists():
            raise ValueError('Experiment ID already exists: ' + run_id)
        generated.append((path, config))
        job = dict(id=run_id,
            argv=[str(ROOT / '.conda/bin/python'), '-u', str(ROOT / 'scripts' / entry),
                  '--config', str(path), '--run-dir', str(run_dir)],
            cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
            is_smoke=config['smoke']['enabled'], smoke_passed=False, depends_on=dependencies)
        if memory_source:
            job['memory_estimate_from'] = memory_source
        queue['jobs'].append(job)

    rechecks = {}
    for dataset, baseline in baselines.items():
        config, _ = load(baseline)
        run_id = f'gtr__{dataset}__smoke_recheck__t96_h96__seed2024__{stamp}'
        original_args = copy.deepcopy(config['official_args'])
        original_smoke = copy.deepcopy(config['smoke'])
        config['official_args'].update(model_id=run_id, train_epochs=1)
        config['smoke'] = dict(enabled=True, max_train_batches=3, max_eval_batches=2)
        config['experiment'].update(id=run_id, stage='smoke', same_evaluation_protocol=False,
            source_experiment_id=baseline['id'],
            comparison_note='Original baseline smoke after retrieval reliability descriptor integration; not a formal result',
            config_changes=[
                dict(key='official_args.model_id', old=original_args['model_id'], new=run_id,
                     reason='unique smoke ID; preserve successful baseline source'),
                dict(key='official_args.train_epochs', old=original_args['train_epochs'], new=1,
                     reason='independent smoke only'),
                dict(key='smoke', old=original_smoke, new=config['smoke'],
                     reason='three train/two evaluation batches for smoke only')])
        add(config, [baseline['id']], 'run_experiment.py')
        rechecks[dataset] = run_id
    for job in queue['jobs']:
        if job['id'].startswith('trustgtr_a6__etth1__smoke__'):
            job['depends_on'].append(rechecks['etth1'])

    for dataset, source in sources.items():
        source_config, measured = load(source)
        assert source_config['trustgtr']['gate_descriptor'] == 'forecast'
        assert source_config['trustgtr'].get('gate_initial_probability', 0.9) == 0.9
        assert source_config['trustgtr'].get('gate_loss_weight', 0.05) == 0.05
        assert source_config['trustgtr']['gate_loss_mode'] == 'power_weighted'
        assert source_config['trustgtr']['historical_strength_coordinates'] == 'bounded'
        assert not source_config['trustgtr']['use_regime_contrast']
        assert source_config['trustgtr']['use_reliability_gate'] == (dataset == 'etth1')
        smoke_id = f'trustgtr_ab6__{dataset}__smoke__t96_h96__seed2024__{stamp}'
        formal_id = smoke_id.replace('__smoke__', '__forecast__')
        for smoke in (True, False):
            run_id = smoke_id if smoke else formal_id
            config = copy.deepcopy(source_config)
            config['official_args'].update(model_id=run_id,
                train_epochs=1 if smoke else source_config['official_args']['train_epochs'])
            config['trustgtr'].update(use_reliability_gate=True, gate_descriptor='retrieval')
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                  max_eval_batches=2 if smoke else None)
            changes = copy.deepcopy(source_config['experiment']['config_changes'])
            if dataset == 'solar':
                changes.append(dict(key='trustgtr.use_reliability_gate', old=False, new=True,
                    reason='first historical B reliability test on Solar: jointly enable B and its retrieval descriptor; not a descriptor-only ablation'))
            changes.extend([
                dict(key='trustgtr.gate_descriptor', old='forecast', new='retrieval',
                    reason='retrieval distance/dispersion/age-bank evidence is available to reliability gate; no target needed at inference'),
                dict(key='official_args.model_id', old=source_config['official_args']['model_id'], new=run_id,
                     reason='unique experiment ID; preserve completed source')])
            if smoke:
                changes.extend([
                    dict(key='official_args.train_epochs', old=source_config['official_args']['train_epochs'], new=1,
                         reason='independent smoke only; formal training budget remains unchanged'),
                    dict(key='smoke', old=source_config['smoke'], new=config['smoke'],
                         reason='three train/two evaluation batches for smoke only')])
            config['experiment'].update(id=run_id, model='TrustGTR_ab6',
                stage='smoke' if smoke else 'improvement', same_evaluation_protocol=not smoke,
                source_experiment_id=source['id'], source_mse=measured['mse'], source_mae=measured['mae'],
                source_provenance=source_provenance(source),
                fixed_evaluation_asset_source=source_config['experiment']['baseline_experiment_id'],
                comparison_note=('ETTh1: descriptor-only change versus AB5; ' if dataset == 'etth1' else
                                 'Solar: jointly enable B and retrieval descriptor versus A5; ') +
                                'same official seed/budget/assets; no test tuning', config_changes=changes)
            add(config, [rechecks[dataset]] if smoke else [smoke_id],
                'run_trust_experiment.py', smoke_id if not smoke else None)
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


def append_calibration():
    """Add gate-loss attribution and train-calibrated contrast metric tests."""
    queue = json.loads(OUTPUT.read_text())
    if any(job['id'].startswith(('trustgtr_ab5_nocal__', 'trustgtr_ac6__')) for job in queue['jobs']):
        raise SystemExit('Calibration jobs already exist; do not duplicate or overwrite them')
    for job in queue['jobs']:
        if job['id'].startswith(('trustgtr_a6__', 'trustgtr_ab6__')) and Path(job['run_dir']).exists():
            raise SystemExit('The v6 queue has already launched; create a new queue instead')
    source_jobs = {}
    for name in ('configs/queue_historical_v5.json', 'configs/queue_v5_followup.json'):
        source_jobs.update({job['id']: job for job in json.loads((ROOT / name).read_text())['jobs']})
    variants = [
        ('ab5_nocal', next(job for job in source_jobs.values()
                          if job['id'].startswith('trustgtr_ab5__etth1__forecast__')),
         {'gate_loss_weight': 0.0}),
        ('ac6', next(job for job in source_jobs.values()
                    if job['id'].startswith('trustgtr_ac5__etth1__forecast__')),
         {'contrast_metric': 'euclidean', 'contrast_temperature': 3.0})]
    retained = {job['id'] for job in queue['jobs']}

    def load(job):
        config = json.loads(Path(job['argv'][job['argv'].index('--config') + 1]).read_text())
        metrics = json.loads((Path(job['run_dir']) / 'metrics.json').read_text())
        if metrics.get('status') != 'SUCCESS':
            raise ValueError('Source is not successfully completed: ' + job['id'])
        return config, metrics

    def retain(identifier):
        if identifier in retained:
            return
        job = source_jobs[identifier]
        load(job)
        for dependency in job['depends_on']:
            retain(dependency)
        queue['jobs'].append(copy.deepcopy(job))
        retained.add(identifier)

    recheck = next(job for job in reversed(queue['jobs'])
                   if job['id'].startswith('gtr__etth1__smoke_recheck__'))
    diagnostic_path = ROOT / 'reports/gate_v5_diagnostic_20261005.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic['no_test_targets_used'] and diagnostic['no_test_dataset_constructed']
    diagnostic_hash = hashlib.sha256(diagnostic_path.read_bytes()).hexdigest()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    generated = []
    for variant, source, options in variants:
        retain(source['id'])
        source_config, measured = load(source)
        smoke_id = f'trustgtr_{variant}__etth1__smoke__t96_h96__seed2024__{stamp}'
        formal_id = smoke_id.replace('__smoke__', '__forecast__')
        for smoke in (True, False):
            run_id = smoke_id if smoke else formal_id
            config = copy.deepcopy(source_config)
            config['official_args'].update(model_id=run_id,
                train_epochs=1 if smoke else source_config['official_args']['train_epochs'])
            config['trustgtr'].update(options)
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                  max_eval_batches=2 if smoke else None)
            changes = copy.deepcopy(source_config['experiment']['config_changes'])
            if variant == 'ab5_nocal':
                changes.append(dict(key='trustgtr.gate_loss_weight', old=0.05, new=0.0,
                    reason='gate calibration attribution: ABC5 fixed train batch measured weighted aux/main=0.678 and cosine=-0.328; test whether this hypothesized conflict transfers to AB5; B stays enabled'))
                note = 'Single-factor gate auxiliary weight change versus AB5; gate remains enabled and forecast descriptor retained'
            else:
                changes.extend([
                    dict(key='trustgtr.contrast_metric', old='cosine', new='euclidean',
                         reason='align the C objective with actual historical retrieval geometry'),
                    dict(key='trustgtr.contrast_temperature', old=0.1, new=3.0,
                         reason='necessary paired scale calibration using fixed train256 gradients: AC5 euclidean weighted aux/main 21.19 at 0.1 versus 0.292 at 3.0; lambda remains 0.002; not a metric-only ablation')])
                note = 'Paired C metric and necessary temperature scale calibration versus AC5; not a metric-only ablation'
            changes.append(dict(key='official_args.model_id', old=source_config['official_args']['model_id'], new=run_id,
                                reason='unique experiment ID; preserve completed source'))
            if smoke:
                changes.extend([
                    dict(key='official_args.train_epochs', old=source_config['official_args']['train_epochs'], new=1,
                         reason='independent smoke only; full formal budget remains unchanged'),
                    dict(key='smoke', old=source_config['smoke'], new=config['smoke'],
                         reason='three train/two evaluation batches for smoke only')])
            config['experiment'].update(id=run_id, model='TrustGTR_' + variant,
                stage='smoke' if smoke else 'improvement', same_evaluation_protocol=not smoke,
                source_experiment_id=source['id'], source_mse=measured['mse'], source_mae=measured['mae'],
                source_provenance=source_provenance(source),
                diagnostic_source=dict(path=str(diagnostic_path), sha256=diagnostic_hash,
                                       scope='fixed training batch gradients only; no test targets'),
                fixed_evaluation_asset_source=source_config['experiment']['baseline_experiment_id'],
                comparison_note=note + '; same official seed/budget/assets', config_changes=changes)
            path = ROOT / 'configs' / (run_id + '.json')
            run_dir = ROOT / 'runs' / run_id
            if path.exists() or run_dir.exists():
                raise ValueError('Experiment ID already exists: ' + run_id)
            job = dict(id=run_id,
                argv=[str(ROOT / '.conda/bin/python'), '-u', str(ROOT / 'scripts/run_trust_experiment.py'),
                      '--config', str(path), '--run-dir', str(run_dir)],
                cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
                is_smoke=smoke, smoke_passed=False,
                depends_on=[recheck['id']] if smoke else [smoke_id])
            if not smoke:
                job['memory_estimate_from'] = smoke_id
            queue['jobs'].append(job)
            generated.append((path, config))
    # Add provenance metadata to this unlaunched batch's AB6 configs; never touch A6.
    for job in queue['jobs']:
        if job['id'].startswith('trustgtr_ab6__'):
            path = Path(job['argv'][job['argv'].index('--config') + 1])
            config = json.loads(path.read_text())
            config['experiment']['source_provenance'] = source_provenance(source_jobs[config['experiment']['source_experiment_id']])
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--add-reliability', action='store_true',
                        help='append baseline rechecks and AB6 to an unlaunched router manifest')
    parser.add_argument('--add-calibration', action='store_true',
                        help='append AB5 no-calibration and AC6 scale-calibrated contrast to unlaunched mechanism manifest')
    args = parser.parse_args()
    if args.add_reliability and args.add_calibration:
        parser.error('Append one stage at a time')
    if args.add_calibration:
        append_calibration()
    elif args.add_reliability:
        append_reliability()
    else:
        prepare_router()


if __name__ == '__main__':
    main()
