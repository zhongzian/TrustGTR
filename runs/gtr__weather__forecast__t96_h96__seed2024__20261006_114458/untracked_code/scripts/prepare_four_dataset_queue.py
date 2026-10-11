#!/usr/bin/env python3
"""Prepare the two added datasets' official GTR/A6 jobs without launching GPUs.

The first two datasets already have completed baselines. This queue adds only
Weather and ETTm2, with independent smoke and formal jobs for each model.
Existing queues/configurations are immutable and safely reusable.
"""
import argparse
import ast
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'configs/trustgtr_a6__etth1__forecast__t96_h96__seed2024__20261005_160105.json'
SPECS = {
    'weather': dict(data='custom', data_path='weather.csv', enc_in=21, cycle=144,
                    batch_size=64, learning_rate=0.001, dropout=0.5, use_revin=1),
    'ettm2': dict(data='ETTm2', data_path='ETTm2.csv', enc_in=7, cycle=96,
                  batch_size=256, learning_rate=0.001, dropout=0.5, use_revin=1),
}
SCIENCE_FILES = ('run.py', 'models/GTR.py', 'models/TrustGTR.py',
                 'models/HistoricalMemory.py', 'scripts/run_experiment.py',
                 'scripts/run_trust_experiment.py', 'exp/exp_repro.py',
                 'data_provider/data_loader.py', 'data_provider/data_factory.py',
                 'utils/metrics.py', 'utils/tools.py')


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def official_defaults():
    defaults = {}
    for node in ast.walk(ast.parse((ROOT / 'run.py').read_text())):
        if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                or node.func.attr != 'add_argument' or not node.args):
            continue
        name = ast.literal_eval(node.args[0]).removeprefix('--')
        for keyword in node.keywords:
            if keyword.arg == 'default':
                defaults[name] = ast.literal_eval(keyword.value)
    expected = dict(use_revin=1, d_model=512, label_len=0, loss='mse',
                    lradj='type3', pct_start=0.3, use_amp=False, use_multi_gpu=False,
                    num_workers=10, model_type='mlp', target='OT', freq='h')
    if {name: defaults.get(name) for name in expected} != expected:
        raise SystemExit('Official critical defaults changed; inspect before preparing experiments')
    return defaults


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def asset_entries(manifest):
    """Accept a list of assets or an audit with an assets/datasets collection."""
    entries = manifest
    if isinstance(manifest, dict):
        entries = manifest.get('assets', manifest.get('datasets', manifest))
    if isinstance(entries, dict):
        return {name.lower(): dict(value, dataset=name.lower())
                for name, value in entries.items() if name.lower() in SPECS}
    return {entry['dataset'].lower(): entry for entry in entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', required=True, help='Verified new dataset asset manifest JSON')
    parser.add_argument('--output', default='configs/queue_four_datasets_v1.json')
    parser.add_argument('--audit', default='reports/four_dataset_protocol_audit.json')
    cli = parser.parse_args()
    output = ROOT / cli.output
    audit_path = ROOT / cli.audit
    if output.exists():
        queue = read_json(output)
        for name, expected in queue.get('preparation', {}).get('science_sha256', {}).items():
            if digest(ROOT / name) != expected:
                raise SystemExit('Scientific source changed since immutable queue preparation: ' + name)
        for job in queue['jobs']:
            config = read_json(job['argv'][job['argv'].index('--config') + 1])
            asset = config['data_asset']
            if digest(asset['path']) != asset['sha256']:
                raise SystemExit('DATA_ERROR: immutable queue asset changed: ' + asset['path'])
        print('Existing immutable queue retained; resume instead of creating duplicate IDs: ' + str(output))
        return
    if audit_path.exists():
        raise SystemExit('Audit already exists; choose new --output/--audit instead of overwriting')
    source = read_json(SOURCE)
    source_id = source['experiment']['id']
    metrics_path = ROOT / 'runs' / source_id / 'metrics.json'
    source_metrics = read_json(metrics_path)
    if source_metrics.get('status') != 'SUCCESS':
        raise SystemExit('A6 template source must have successful persisted metrics')
    options = copy.deepcopy(source['trustgtr'])
    assert options['use_multi_cycle'] and not options['use_reliability_gate']
    assert not options['use_regime_contrast'] and options['memory_mode'] == 'historical'
    assert options['historical_router_mode'] == 'per_bank'
    assert options['historical_strength_coordinates'] == 'bounded'
    source_provenance = dict(config_path=str(SOURCE), config_sha256=digest(SOURCE),
                             metrics_path=str(metrics_path), metrics_sha256=digest(metrics_path))
    manifest_path = Path(cli.assets).resolve()
    assets = asset_entries(read_json(manifest_path))
    if set(SPECS) - set(assets):
        raise SystemExit('DATA_ERROR: manifest must provide weather and ettm2')
    code_hashes = {name: digest(ROOT / name) for name in SCIENCE_FILES}
    defaults = official_defaults()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    queue = dict(max_jobs=3, poll_seconds=3, oom_cooldown_seconds=60,
        on_change_argv=[str(ROOT / '.conda/bin/python'), str(ROOT / 'scripts/summarize_results.py')],
        preparation=dict(step='6.7', datasets=['etth1', 'solar', 'weather', 'ettm2'],
            new_datasets=['weather', 'ettm2'], new_job_count=8,
            asset_manifest=str(manifest_path), asset_manifest_sha256=digest(manifest_path),
            science_sha256=code_hashes,
            scope='User explicitly authorized two added datasets; no multi-seed/external baseline expansion'),
        jobs=[])
    generated = []
    checks = []
    for dataset, spec in SPECS.items():
        provided = assets[dataset]
        asset_path = Path(provided['path']).resolve()
        expected_path = ROOT / 'dataset' / spec['data_path']
        if asset_path != expected_path.resolve() or digest(asset_path) != provided['sha256']:
            raise SystemExit('DATA_ERROR: verified official-format asset mismatch: ' + str(asset_path))
        asset = dict(path=str(asset_path), sha256=provided['sha256'],
            size_bytes=asset_path.stat().st_size,
            provenance=provided.get('provenance', provided.get('source_url', 'Verified public benchmark asset')),
            audit_manifest=str(manifest_path), audit_manifest_sha256=digest(manifest_path))
        for key in ('source_url', 'rows', 'feature_count', 'columns'):
            if key in provided:
                asset[key] = provided[key]
        ids = {name: f'{model}__{dataset}__{stage}__t96_h96__seed2024__{stamp}'
            for name, model, stage in (
                ('baseline_smoke', 'gtr', 'smoke'), ('baseline', 'gtr', 'forecast'),
                ('a6_smoke', 'trustgtr_a6', 'smoke'), ('a6', 'trustgtr_a6', 'forecast'))}
        configs = {}
        for name in ('baseline_smoke', 'baseline', 'a6_smoke', 'a6'):
            smoke = name.endswith('smoke')
            improvement = name.startswith('a6')
            run_id = ids[name]
            args = dict(spec, is_training=1, root_path=str(ROOT / 'dataset') + '/',
                model_id=run_id, model='TrustGTR' if improvement else 'GTR', features='M',
                seq_len=96, pred_len=96, train_epochs=1 if smoke else 30,
                patience=5, itr=1, random_seed=2024)
            changes = [dict(key='official_args.model_id', old='official script dataset_T_H identifier',
                new=run_id, reason='unique immutable ID; independent new dataset experiment')]
            if improvement:
                changes.extend([
                    dict(key='official_args.model', old='GTR', new='TrustGTR',
                         reason='single-factor A6 historical retrieval; original GTR forecast branch retained'),
                    dict(key='trustgtr', old='all innovation modules disabled', new=options,
                         reason='reuse successful prior A6 configuration, without selecting A8 from test results; B/C preserved but disabled in this initial attribution pair')])
            if smoke:
                changes.extend([
                    dict(key='official_args.train_epochs', old=30, new=1,
                         reason='independent smoke only; unchanged batch/model/protocol, formal epoch cap remains 30'),
                    dict(key='smoke.batch_limits', old=None, new=dict(train=3, evaluation=2),
                         reason='smoke forward/backward/optimizer/checkpoint/load/evaluation/metrics chain only')])
            experiment = dict(id=run_id, dataset=dataset,
                model='TrustGTR_a6' if improvement else 'GTR',
                stage='smoke' if smoke else 'improvement' if improvement else 'baseline',
                protocol='official_gtr_t96_h96_seed2024', same_evaluation_protocol=not smoke,
                official_script=f'scripts/GTR/{dataset}.sh', paper_mse=None, paper_mae=None,
                paper_reference='NA: no verified paper values supplied for this extension',
                paper_url='https://arxiv.org/html/2602.10847v1', step='6.7',
                config_changes=changes,
                comparison_note='Same per-dataset official seed/budget/loader/normalization/evaluation/checkpoint policy; first extension is GTR versus A6 only; A/B/C retained for subsequent innovation rounds',
                frozen_code_sha256=code_hashes)
            if improvement:
                experiment.update(baseline_experiment_id=ids['baseline'],
                    fixed_evaluation_asset_source=ids['baseline'],
                    baseline_mse=None, baseline_mae=None,
                    source_experiment_id=source_id, source_provenance=source_provenance,
                    source_scope='A6 options only; new dataset uses its own official baseline protocol, not ETTh1 data/hyperparameters')
            config = dict(official_args=args, experiment=experiment, data_asset=asset,
                smoke=dict(enabled=smoke, max_train_batches=3 if smoke else None,
                           max_eval_batches=2 if smoke else None))
            if improvement:
                config['trustgtr'] = copy.deepcopy(options)
            config_path = ROOT / 'configs' / (run_id + '.json')
            run_dir = ROOT / 'runs' / run_id
            if config_path.exists() or run_dir.exists():
                raise SystemExit('Experiment ID already exists: ' + run_id)
            dependencies = ([] if name == 'baseline_smoke' else [ids['baseline_smoke']]
                            if name in ('baseline', 'a6_smoke') else [ids['a6_smoke'], ids['baseline']])
            memory_source = (ids['baseline_smoke'] if name == 'baseline'
                             else ids['a6_smoke'] if name == 'a6' else None)
            job = dict(id=run_id, argv=[str(ROOT / '.conda/bin/python'), '-u',
                str(ROOT / 'scripts' / ('run_trust_experiment.py' if improvement else 'run_experiment.py')),
                '--config', str(config_path), '--run-dir', str(run_dir)],
                cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
                is_smoke=smoke, smoke_passed=False, depends_on=dependencies)
            if memory_source:
                job['memory_estimate_from'] = memory_source
            queue['jobs'].append(job)
            generated.append((config_path, config))
            configs[name] = config
        resolved_base = {**defaults, **configs['baseline']['official_args']}
        resolved_a6 = {**defaults, **configs['a6']['official_args']}
        base_args = configs['baseline']['official_args']
        improved_args = configs['a6']['official_args']
        difference = {key: [base_args[key], improved_args[key]] for key in base_args
                      if base_args[key] != improved_args[key]}
        assert set(difference) == {'model', 'model_id'}
        assert configs['baseline']['data_asset'] == configs['a6']['data_asset']
        assert configs['a6']['trustgtr'] == options
        assert {key: [resolved_base[key], resolved_a6[key]] for key in resolved_base
                if resolved_base[key] != resolved_a6[key]} == difference
        checks.append(dict(dataset=dataset, official_script=f'scripts/GTR/{dataset}.sh',
            script_sha256=digest(ROOT / f'scripts/GTR/{dataset}.sh'),
            official_formal_args=base_args, baseline_id=ids['baseline'], improvement_id=ids['a6'],
            inherited_official_defaults={key: defaults[key] for key in
                ('use_revin', 'd_model', 'label_len', 'loss', 'lradj', 'pct_start',
                 'use_amp', 'use_multi_gpu', 'num_workers', 'model_type', 'target', 'freq')},
            official_args_diff=difference, same_asset=True, same_training_budget=True,
            same_metric_and_loader=True, unchanged_a6_options=True,
            paper_values='NA until independently verified'))
    identifiers = [job['id'] for job in queue['jobs']]
    assert len(identifiers) == len(set(identifiers)) == 8
    assert all(dep in identifiers for job in queue['jobs'] for dep in job['depends_on'])
    audit = dict(prepared_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), step='6.7',
        queue_path=str(output), job_count=8, formal_count=4, smoke_count=4,
        datasets_added=['weather', 'ettm2'], max_jobs=3, launch_performed=False,
        scientific_code_sha256=code_hashes, source_provenance=source_provenance,
        asset_manifest=str(manifest_path), asset_manifest_sha256=digest(manifest_path),
        formal_protocol_checks=checks,
        memory_policy='baseline and A6 smoke start with unknown peak; no borrowing the no-history baseline peak for history retrieval; scheduler conservatively avoids same-GPU project co-location until profile is known; formal uses own successful smoke peak',
        scope_note='A8 is not a universal default after Solar degradation; full A/B/C research scope retained',
        compatibility_requirement='runner must record ETTm/custom absolute split metadata before historical A6 smoke; original loader/splits unchanged')
    for path, config in generated:
        write_new(path, config)
    write_new(audit_path, audit)
    write_new(output, queue)
    print(output)
    print('Prepared 8 independent jobs (4 smoke, 4 formal), max concurrency 3; no GPU launched.')
    for job in queue['jobs']:
        print(job['id'], 'depends_on', job['depends_on'])


if __name__ == '__main__':
    main()
