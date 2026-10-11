#!/usr/bin/env python3
"""Complete the v5 AC attribution and transfer A5 to Solar without budget changes."""
import copy
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'configs/queue_v5_followup.json'
V5_QUEUE = ROOT / 'configs/queue_historical_v5.json'


def load_job_config(job):
    return json.loads(Path(job['argv'][job['argv'].index('--config') + 1]).read_text())


def completed_metrics(job):
    metrics = json.loads((Path(job['run_dir']) / 'metrics.json').read_text())
    if metrics.get('status') != 'SUCCESS':
        raise ValueError('Source experiment is not completed: ' + job['id'])
    return metrics


def main():
    if OUTPUT.exists():
        raise SystemExit('Existing queue is immutable; resume it instead of overwriting')
    queue = json.loads((ROOT / 'configs/queue_baseline_v3.json').read_text())
    v5 = json.loads(V5_QUEUE.read_text())
    baselines = {dataset: next(job for job in queue['jobs']
        if f'__{dataset}__forecast__' in job['id']) for dataset in ('etth1', 'solar')}
    source_a = next(job for job in v5['jobs'] if job['id'].startswith('trustgtr_a5__etth1__forecast__'))
    source_abc = next(job for job in v5['jobs'] if job['id'].startswith('trustgtr_abc5__etth1__forecast__'))
    for job in [*baselines.values(), source_a, source_abc]:
        completed_metrics(job)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    generated = []

    def add(config, dependencies, entry, memory_source=None):
        run_id = config['experiment']['id']
        config_path = ROOT / 'configs' / (run_id + '.json')
        run_dir = ROOT / 'runs' / run_id
        if config_path.exists() or run_dir.exists():
            raise ValueError('Experiment ID already exists: ' + run_id)
        generated.append((config_path, config))
        job = dict(id=run_id, argv=[str(ROOT / '.conda/bin/python'), '-u',
            str(ROOT / 'scripts' / entry), '--config', str(config_path), '--run-dir', str(run_dir)],
            cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
            is_smoke=config['smoke']['enabled'], smoke_passed=False, depends_on=dependencies)
        if memory_source:
            job['memory_estimate_from'] = memory_source
        queue['jobs'].append(job)

    recheck_id = f'gtr__solar__smoke_recheck__t96_h96__seed2024__{stamp}'
    config = load_job_config(baselines['solar'])
    old_id = config['official_args']['model_id']
    config['official_args'].update(model_id=recheck_id, train_epochs=1)
    config['smoke'] = dict(enabled=True, max_train_batches=3, max_eval_batches=2)
    config['experiment'].update(id=recheck_id, stage='smoke', same_evaluation_protocol=False,
        source_experiment_id=baselines['solar']['id'],
        comparison_note='Independent original Solar baseline smoke before historical A5 transfer; not a formal result',
        config_changes=[dict(key='official_args.model_id', old=old_id, new=recheck_id,
                             reason='unique smoke experiment ID; preserve completed source'),
            dict(key='official_args.train_epochs', old=30, new=1, reason='independent smoke only'),
            dict(key='smoke', old=dict(enabled=False, max_train_batches=None, max_eval_batches=None),
                 new=config['smoke'], reason='three training/two evaluation batches for smoke only')])
    add(config, [baselines['solar']['id']], 'run_experiment.py')

    for dataset, variant, source in [('etth1', 'ac5', source_abc), ('solar', 'a5', source_a)]:
        baseline = load_job_config(baselines[dataset])
        measured = completed_metrics(baselines[dataset])
        source_config = load_job_config(source)
        formal_config = copy.deepcopy(source_config if dataset == 'etth1' else baseline)
        options = copy.deepcopy(source_config['trustgtr'])
        if dataset == 'etth1':
            options['use_reliability_gate'] = False
            changes = copy.deepcopy(source_config['experiment']['config_changes'])
            changes.append(dict(key='trustgtr.use_reliability_gate', old=True, new=False,
                reason='AC5 attribution: only disable B relative to completed ABC5; preserve A/C and training budget'))
        else:
            options['periods'] = [144, 1008, 4320]
            formal_config['official_args']['model'] = 'TrustGTR'
            changes = [dict(key='official_args.model', old='GTR', new='TrustGTR',
                reason='test validated A5 architecture on a second dataset without changing official Solar budget'),
                dict(key='trustgtr', old='disabled', new=options,
                    reason='copy completed ETTh1 A5 options; only periods scale to the official Solar base cycle'),
                dict(key='trustgtr.periods', old=[24, 168, 720], new=[144, 1008, 4320],
                    reason='same short/middle/long relative-age bank semantics with Solar cycle 144; not a changed phase rule')]
        formal_config['trustgtr'] = options
        smoke_id = f'trustgtr_{variant}__{dataset}__smoke__t96_h96__seed2024__{stamp}'
        formal_id = smoke_id.replace('__smoke__', '__forecast__')
        original_args = copy.deepcopy(formal_config['official_args'])
        for smoke in (True, False):
            run_id = smoke_id if smoke else formal_id
            config = copy.deepcopy(formal_config)
            config['official_args'].update(model_id=run_id, train_epochs=1 if smoke else original_args['train_epochs'])
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                  max_eval_batches=2 if smoke else None)
            run_changes = copy.deepcopy(changes)
            run_changes.append(dict(key='official_args.model_id', old=original_args['model_id'], new=run_id,
                                    reason='unique follow-up experiment ID; preserve completed source'))
            if smoke:
                run_changes.extend([
                    dict(key='official_args.train_epochs', old=original_args['train_epochs'], new=1,
                         reason='independent smoke only; full formal budget remains unchanged'),
                    dict(key='smoke', old=formal_config['smoke'], new=config['smoke'],
                         reason='three training/two evaluation batches for smoke only')])
            config['experiment'].update(id=run_id, model='TrustGTR_' + variant,
                stage='smoke' if smoke else 'improvement', source_experiment_id=source['id'],
                baseline_experiment_id=baselines[dataset]['id'],
                baseline_mse=measured['mse'], baseline_mae=measured['mae'],
                same_evaluation_protocol=not smoke,
                fixed_evaluation_asset_source=baselines[dataset]['id'],
                comparison_note='Same official dataset/seed/budget/evaluation assets; v5 follow-up without test tuning',
                config_changes=run_changes)
            unchanged = {'model_id', 'train_epochs'} if smoke else {'model_id'}
            assert all(config['official_args'][key] == value for key, value in original_args.items()
                       if key not in unchanged), 'Silent protocol change'
            assert config['data_asset'] == baseline['data_asset'], 'Data asset changed'
            dependencies = ([recheck_id] if dataset == 'solar' else [baselines[dataset]['id']]) if smoke else [smoke_id]
            add(config, dependencies, 'run_trust_experiment.py', smoke_id if not smoke else None)

    # Validate the full manifest before writing any new config.
    ids = [job['id'] for job in queue['jobs']]
    assert len(ids) == len(set(ids))
    assert queue['max_jobs'] == 3
    assert all(dep in ids for job in queue['jobs'] for dep in job['depends_on'])
    for config_path, config in generated:
        with config_path.open('x') as file:
            file.write(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    with OUTPUT.open('x') as file:
        file.write(json.dumps(queue, indent=2) + '\n')
    print(OUTPUT)
    for _, config in generated:
        print(config['experiment']['id'])


if __name__ == '__main__':
    main()
