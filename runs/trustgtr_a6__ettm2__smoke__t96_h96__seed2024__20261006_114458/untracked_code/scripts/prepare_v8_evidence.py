#!/usr/bin/env python3
"""Prepare the single-factor A8 evidence router; never launch experiments."""
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE_QUEUE = ROOT / 'configs/queue_v7_complementarity.json'
OUTPUT = ROOT / 'configs/queue_v8_evidence.json'


def load_source(job):
    path = Path(job['argv'][job['argv'].index('--config') + 1])
    metrics_path = Path(job['run_dir']) / 'metrics.json'
    config = json.loads(path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if metrics.get('status') != 'SUCCESS':
        raise ValueError('Source is not successfully completed: ' + job['id'])
    provenance = dict(config_path=str(path),
        config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        metrics_path=str(metrics_path),
        metrics_sha256=hashlib.sha256(metrics_path.read_bytes()).hexdigest())
    return config, metrics, provenance


def make_job(run_id, smoke, dependencies, memory_source, memory_estimate=None):
    run_dir = ROOT / 'runs' / run_id
    path = ROOT / 'configs' / (run_id + '.json')
    if path.exists() or run_dir.exists():
        raise ValueError('Experiment ID already exists: ' + run_id)
    return dict(id=run_id,
        argv=[str(ROOT / '.conda/bin/python'), '-u', str(ROOT / 'scripts/run_trust_experiment.py'),
              '--config', str(path), '--run-dir', str(run_dir)],
        cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=memory_estimate,
        is_smoke=smoke, smoke_passed=False, depends_on=dependencies,
        memory_estimate_from=memory_source)


def main():
    if OUTPUT.exists():
        raise SystemExit('Existing queue is immutable; resume it instead of overwriting')
    previous = json.loads(SOURCE_QUEUE.read_text())
    source_jobs = {job['id']: job for job in previous['jobs']}
    runtime = Path(previous.get('runtime_dir', ROOT / 'runtime'))
    recorded = json.loads((runtime / 'queue_state.json').read_text())['jobs']
    sources = {dataset: next(job for job in previous['jobs']
        if job['id'].startswith(f'trustgtr_a6__{dataset}__forecast__'))
        for dataset in ('etth1', 'solar')}
    original_baseline = next(job for job in previous['jobs']
        if job['id'].startswith('gtr__etth1__forecast__') and '20261004_214520' in job['id'])
    queue = {key: copy.deepcopy(value) for key, value in previous.items() if key != 'jobs'}
    queue['jobs'] = []
    retained = set()

    def retain(identifier):
        if identifier in retained:
            return
        job = source_jobs[identifier]
        load_source(job)
        # This scheduler restores successes from its existing state, not metrics alone.
        # Refuse to regenerate an archived run if that state has been moved or reset.
        if recorded.get(identifier, {}).get('status') != 'SUCCESS':
            raise ValueError('Completed source is absent from successful scheduler state: ' + identifier)
        for dependency in job['depends_on']:
            retain(dependency)
        retained_job = copy.deepcopy(job)
        retained_job['smoke_passed'] = retained_job.get('smoke_passed') is True
        queue['jobs'].append(retained_job)
        retained.add(identifier)

    retain(original_baseline['id'])
    for source in sources.values():
        retain(source['id'])
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    generated = []

    # Exercise the changed TrustGTR file with all innovation modules disabled.
    baseline, baseline_metrics, provenance = load_source(original_baseline)
    baseline_smoke_id = f'gtr__etth1__smoke_recheck_v8__t96_h96__seed2024__{stamp}'
    baseline_smoke = copy.deepcopy(baseline)
    alloff = dict(use_multi_cycle=False, use_reliability_gate=False,
                  use_regime_contrast=False, memory_mode='independent',
                  preserve_initialization_rng=True)
    baseline_smoke['official_args'].update(model='TrustGTR', model_id=baseline_smoke_id, train_epochs=1)
    baseline_smoke['trustgtr'] = alloff
    baseline_smoke['smoke'] = dict(enabled=True, max_train_batches=3, max_eval_batches=2)
    baseline_smoke['experiment'].update(id=baseline_smoke_id, model='TrustGTR_alloff',
        stage='smoke', same_evaluation_protocol=False,
        source_experiment_id=original_baseline['id'], source_provenance=provenance,
        comparison_note='Original GTR regression through TrustGTR with A/B/C disabled; independent smoke, not a formal baseline',
        config_changes=[
            dict(key='official_args.model', old='GTR', new='TrustGTR',
                 reason='exercise all-off innovation entry point while restoring original GTR behavior'),
            dict(key='trustgtr', old=baseline.get('trustgtr', 'absent'), new=alloff,
                 reason='A/B/C all disabled; independent memory mode creates no historical bank'),
            dict(key='official_args.model_id', old=baseline['official_args']['model_id'], new=baseline_smoke_id,
                 reason='unique v8 regression smoke; preserve completed baseline'),
            dict(key='official_args.train_epochs', old=baseline['official_args']['train_epochs'], new=1,
                 reason='independent smoke only; no change to any formal budget'),
            dict(key='smoke', old=baseline['smoke'], new=baseline_smoke['smoke'],
                 reason='three training/two evaluation batches to verify the all-off path')])
    queue['jobs'].append(make_job(baseline_smoke_id, True, [original_baseline['id']],
        original_baseline['id'], baseline_metrics['peak_gpu_memory_mib']))
    generated.append((ROOT / 'configs' / (baseline_smoke_id + '.json'), baseline_smoke))

    for dataset, source in sources.items():
        original, metrics, provenance = load_source(source)
        options = original['trustgtr']
        args = original['official_args']
        assert options['use_multi_cycle'] and not options['use_reliability_gate']
        assert not options['use_regime_contrast'] and options['memory_mode'] == 'historical'
        assert options['historical_router_mode'] == 'per_bank'
        assert options['historical_strength_coordinates'] == 'bounded'
        assert args['seq_len'] == args['pred_len'] == 96 and args['random_seed'] == 2024
        assert args['train_epochs'] == 30 and args['patience'] == 5
        smoke_id = f'trustgtr_a8__{dataset}__smoke__t96_h96__seed2024__{stamp}'
        formal_id = smoke_id.replace('__smoke__', '__forecast__')
        for smoke in (True, False):
            run_id = smoke_id if smoke else formal_id
            config = copy.deepcopy(original)
            config['official_args'].update(model_id=run_id,
                train_epochs=1 if smoke else args['train_epochs'])
            config['trustgtr']['historical_router_mode'] = 'per_bank_evidence'
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                  max_eval_batches=2 if smoke else None)
            changes = copy.deepcopy(original['experiment']['config_changes'])
            changes.extend([
                dict(key='trustgtr.historical_router_mode', old='per_bank', new='per_bank_evidence',
                     reason='single-factor A8: preserve top-k coverage per age bank, update bank mass with mean inverse-distance candidate evidence before final normalization; query-only bank priors ignore absolute matching quality; no new trainable parameters or independent innovation claim'),
                dict(key='official_args.model_id', old=args['model_id'], new=run_id,
                     reason='unique A8 follow-up ID; preserve the completed A6 comparison')])
            if smoke:
                changes.extend([
                    dict(key='official_args.train_epochs', old=args['train_epochs'], new=1,
                         reason='independent smoke only; unchanged formal 30-epoch maximum and patience'),
                    dict(key='smoke', old=original['smoke'], new=config['smoke'],
                         reason='three training/two evaluation batches for smoke only')])
            config['experiment'].update(id=run_id, model='TrustGTR_a8',
                stage='smoke' if smoke else 'improvement', same_evaluation_protocol=not smoke,
                source_experiment_id=source['id'], source_mse=metrics['mse'], source_mae=metrics['mae'],
                source_provenance=provenance,
                fixed_evaluation_asset_source=original['experiment']['baseline_experiment_id'],
                baseline_recheck_experiment_id=baseline_smoke_id,
                comparison_note='Single-factor evidence-aware bank weighting versus completed A6; B/C disabled, fixed seed, official budget and evaluation assets; evidence is input similarity, not guaranteed future utility',
                config_changes=changes)
            job = make_job(run_id, smoke,
                [baseline_smoke_id, source['id']] if smoke else [smoke_id, source['id']],
                source['id'] if smoke else smoke_id, metrics['peak_gpu_memory_mib'])
            queue['jobs'].append(job)
            generated.append((ROOT / 'configs' / (run_id + '.json'), config))

    identifiers = [job['id'] for job in queue['jobs']]
    assert queue['max_jobs'] == 3 and len(identifiers) == len(set(identifiers))
    assert all(dep in identifiers for job in queue['jobs'] for dep in job['depends_on'])
    assert all(job['memory_estimate_from'] in identifiers for job in queue['jobs']
               if job.get('memory_estimate_from'))
    for path, config in generated:
        with path.open('x') as file:
            file.write(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    with OUTPUT.open('x') as file:
        file.write(json.dumps(queue, indent=2) + '\n')
    print(OUTPUT)
    print(f'Retained completed dependencies: {len(retained)}; new jobs: {len(generated)}')
    for _, config in generated:
        print(config['experiment']['id'])
    print('Prepared only. Freeze and verify scientific source hashes before launching this queue.')


if __name__ == '__main__':
    main()
