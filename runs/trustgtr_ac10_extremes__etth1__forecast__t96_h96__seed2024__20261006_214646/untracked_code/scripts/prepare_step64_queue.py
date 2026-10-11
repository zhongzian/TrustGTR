#!/usr/bin/env python3
"""Prepare step 6.4 paired gate calibration jobs; never launch training.

The train/inference pair changes only gate_calibration_context. A6 retrieval,
the new gate, its loss, and all official per-dataset settings are shared.
Existing GTR/A6 results are references, not jobs in this queue.
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
SOURCE_IDS = {
    'etth1': 'trustgtr_a6__etth1__forecast__t96_h96__seed2024__20261005_160105',
    'solar': 'trustgtr_a6__solar__forecast__t96_h96__seed2024__20261005_162347',
    'weather': 'trustgtr_a6__weather__forecast__t96_h96__seed2024__20261006_114458',
    'ettm2': 'trustgtr_a6__ettm2__forecast__t96_h96__seed2024__20261006_114458',
}
SPECS = {
    'etth1': dict(data='ETTh1', batch_size=256, learning_rate=0.001, dropout=0.5,
                  use_revin=1, cycle=24, enc_in=7, data_path='ETTh1.csv'),
    'solar': dict(data='Solar', batch_size=64, learning_rate=0.003, dropout=0.0,
                  use_revin=0, cycle=144, enc_in=137, data_path='solar_AL.txt'),
    'weather': dict(data='custom', batch_size=64, learning_rate=0.001, dropout=0.5,
                    use_revin=1, cycle=144, enc_in=21, data_path='weather.csv'),
    'ettm2': dict(data='ETTm2', batch_size=256, learning_rate=0.001, dropout=0.5,
                  use_revin=1, cycle=96, enc_in=7, data_path='ETTm2.csv'),
}
SCIENCE_FILES = (
    'run.py', 'models/GTR.py', 'models/TrustGTR.py', 'models/HistoricalMemory.py',
    'scripts/run_experiment.py', 'scripts/run_trust_experiment.py',
    'exp/exp_basic.py', 'exp/exp_main.py', 'exp/exp_repro.py',
    'data_provider/data_loader.py', 'data_provider/data_factory.py',
    'utils/metrics.py', 'utils/tools.py', 'utils/timefeatures.py',
)
COMMON_GATE = dict(use_reliability_gate=True, gate_descriptor='retrieval',
                   detach_gate_descriptor=True, gate_loss_mode='power_weighted',
                   gate_loss_weight=0.05, gate_initial_probability=0.9)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def defaults():
    result = {}
    for node in ast.walk(ast.parse((ROOT / 'run.py').read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument' and node.args):
            name = ast.literal_eval(node.args[0]).removeprefix('--')
            for keyword in node.keywords:
                if keyword.arg == 'default':
                    result[name] = ast.literal_eval(keyword.value)
    expected = dict(d_model=512, label_len=0, loss='mse', lradj='type3',
                    pct_start=0.3, use_amp=False, use_multi_gpu=False,
                    num_workers=10, model_type='mlp', target='OT', freq='h')
    if {key: result.get(key) for key in expected} != expected:
        raise SystemExit('Official critical defaults changed; inspect before preparing jobs')
    return result


def successful_metrics(identifier):
    path = ROOT / 'runs' / identifier / 'metrics.json'
    metrics = read(path)
    if metrics.get('status') != 'SUCCESS' or metrics.get('stage') == 'smoke':
        raise SystemExit('Successful formal reference required: ' + identifier)
    return metrics, dict(id=identifier, metrics_path=str(path), metrics_sha256=digest(path),
                         code_manifest_path=str(path.parent / 'code_manifest.json'),
                         code_manifest_sha256=digest(path.parent / 'code_manifest.json'))


def verified_sources(official_defaults):
    sources = {}
    for dataset, identifier in SOURCE_IDS.items():
        config_path = ROOT / 'configs' / (identifier + '.json')
        config = read(config_path)
        metrics, provenance = successful_metrics(identifier)
        baseline_id = config['experiment']['baseline_experiment_id']
        baseline, baseline_provenance = successful_metrics(baseline_id)
        options = config['trustgtr']
        if not (options['use_multi_cycle'] and not options['use_reliability_gate']
                and not options['use_regime_contrast']
                and options['historical_router_mode'] == 'per_bank'
                and options['memory_mode'] == 'historical'
                and options['historical_strength_coordinates'] == 'bounded'):
            raise SystemExit('A6 source changed: ' + identifier)
        resolved = {**official_defaults, **config['official_args']}
        expected = {**SPECS[dataset], 'train_epochs': 30, 'patience': 5,
                    'random_seed': 2024, 'seq_len': 96, 'pred_len': 96,
                    'label_len': 0, 'd_model': 512, 'use_amp': False,
                    'features': 'M', 'model': 'TrustGTR', 'itr': 1}
        if {key: resolved.get(key) for key in expected} != expected:
            raise SystemExit('A6 formal official protocol mismatch: ' + identifier)
        asset = config['data_asset']
        expected_path = ROOT / 'dataset' / config['official_args']['data_path']
        if Path(asset['path']).resolve() != expected_path.resolve() or digest(expected_path) != asset['sha256']:
            raise SystemExit('DATA_ERROR: reference dataset hash/path mismatch: ' + dataset)
        eval_assets = metrics['evaluation_assets']
        if eval_assets != baseline['evaluation_assets']:
            raise SystemExit('Fixed evaluation asset differs between A6/GTR: ' + dataset)
        evaluation_provenance = {}
        for flag, record in eval_assets.items():
            if digest(record['path']) != record['sha256']:
                raise SystemExit('DATA_ERROR: fixed evaluation asset changed: ' + record['path'])
            evaluation_provenance[flag] = {**record, 'scaler_sha256': digest(record['scaler'])}
        baseline_config = read(ROOT / 'configs' / (baseline_id + '.json'))
        base_resolved = {**official_defaults, **baseline_config['official_args']}
        actual_diff = {key: [base_resolved[key], resolved[key]] for key in resolved
                       if base_resolved.get(key) != resolved[key]}
        if set(actual_diff) != {'model', 'model_id'}:
            raise SystemExit('Baseline/A6 official budget mismatch: ' + dataset)
        provenance.update(config_path=str(config_path), config_sha256=digest(config_path),
                          evaluation_assets=evaluation_provenance,
                          baseline=baseline_provenance)
        sources[dataset] = dict(config=config, metrics=metrics, baseline=baseline,
                                provenance=provenance, resolved=resolved)
    return sources


def ready_model():
    # Compile the actual pure option resolver, avoiding torch/CUDA imports.
    tree = ast.parse((ROOT / 'models/TrustGTR.py').read_text())
    default_node = None
    resolver_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULTS' for t in node.targets):
            if isinstance(node.value, ast.Call):
                options = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
                if options.get('gate_calibration_context') == 'train':
                    default_node = node
        if isinstance(node, ast.FunctionDef) and node.name == 'resolved_options':
            resolver_node = node
    if default_node is not None and resolver_node is not None:
        namespace = {}
        module = ast.Module(body=[default_node, resolver_node], type_ignores=[])
        exec(compile(module, str(ROOT / 'models/TrustGTR.py'), 'exec'), namespace)
        return namespace['resolved_options']
    raise SystemExit('Model/runner implementation not frozen yet: new train-default gate_calibration_context required')


def existing_queue_check(output):
    queue = read(output)
    for name, expected in queue['preparation']['science_sha256'].items():
        if digest(ROOT / name) != expected:
            raise SystemExit('Immutable queue source changed: ' + name)
    for job in queue['jobs']:
        config_path = job['argv'][job['argv'].index('--config') + 1]
        config = read(config_path)
        if digest(config_path) != queue['preparation']['config_sha256'][config_path]:
            raise SystemExit('Immutable input config changed: ' + config_path)
        if digest(config['data_asset']['path']) != config['data_asset']['sha256']:
            raise SystemExit('DATA_ERROR: immutable queue asset changed')
    print('Existing immutable step 6.4 queue retained; resume rather than creating duplicate IDs: ' + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--queue', dest='output', default='configs/queue_step_6_4_ab9.json')
    parser.add_argument('--audit', default='reports/step_6_4_protocol_audit.json')
    parser.add_argument('--check-only', action='store_true', help='Verify completed references and defaults without writing or freezing code')
    cli = parser.parse_args()
    output = ROOT / cli.output
    audit_path = ROOT / cli.audit
    audit_md = audit_path.with_suffix('.md')
    if output.exists():
        existing_queue_check(output)
        return
    official_defaults = defaults()
    sources = verified_sources(official_defaults)
    if cli.check_only:
        print('Reference checks passed: 4 successful A6/GTR pairs; fixed assets and official budgets exact; no files written.')
        return
    resolve_options = ready_model()
    code_hashes = {name: digest(ROOT / name) for name in SCIENCE_FILES}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    alloff_id = f'trustgtr_alloff__etth1__smoke__t96_h96__seed2024__{stamp}'
    generated = []
    checks = []
    queue = dict(max_jobs=3, poll_seconds=3, oom_cooldown_seconds=60,
        runtime_dir=str(ROOT / 'runtime/step_6_4_ab9'),
        on_change_argv=[str(ROOT / '.conda/bin/python'), str(ROOT / 'scripts/summarize_results.py')],
        preparation=dict(step='6.4', datasets=list(SOURCE_IDS), new_job_count=17,
            scope='Single-seed four-dataset paired gate context test; A/C retained; no extra GTR formal reruns',
            science_sha256=code_hashes,
            source_provenance={dataset: source['provenance'] for dataset, source in sources.items()}), jobs=[])

    def add(config, dependencies, memory_source=None):
        run_id = config['experiment']['id']
        path = ROOT / 'configs' / (run_id + '.json')
        run_dir = ROOT / 'runs' / run_id
        if path.exists() or run_dir.exists():
            raise SystemExit('Unique experiment ID already exists: ' + run_id)
        job = dict(id=run_id, argv=[str(ROOT / '.conda/bin/python'), '-u',
            str(ROOT / 'scripts/run_trust_experiment.py'), '--config', str(path),
            '--run-dir', str(run_dir)], cwd=str(ROOT), run_dir=str(run_dir),
            required_memory_mib=None, is_smoke=config['smoke']['enabled'],
            smoke_passed=False, depends_on=dependencies)
        if memory_source:
            job['memory_estimate_from'] = memory_source
        queue['jobs'].append(job)
        generated.append((path, config))

    source = sources['etth1']
    alloff = copy.deepcopy(source['config'])
    alloff['official_args'].update(model_id=alloff_id, train_epochs=1)
    alloff['trustgtr'].update(use_multi_cycle=False, use_reliability_gate=False,
                             use_regime_contrast=False, memory_mode='independent',
                             gate_calibration_context='train')
    alloff['smoke'] = dict(enabled=True, max_train_batches=3, max_eval_batches=2)
    alloff['experiment'] = dict(id=alloff_id, dataset='etth1', model='TrustGTR_alloff',
        stage='smoke', step='6.4', protocol=source['config']['experiment']['protocol'],
        same_evaluation_protocol=False, source_experiment_id=SOURCE_IDS['etth1'],
        source_provenance=source['provenance'],
        baseline_experiment_id=source['config']['experiment']['baseline_experiment_id'],
        fixed_evaluation_asset_source=source['config']['experiment']['baseline_experiment_id'],
        frozen_code_sha256=code_hashes,
        comparison_note='Fresh all-modules-off regression smoke after gate calibration integration; not a formal result',
        config_changes=[dict(key='trustgtr.module_switches', old='A6 on; B/C off',
            new='A/B/C all off', reason='confirm unchanged original GTR behavior through TrustGTR entry'),
            dict(key='trustgtr.memory_mode', old='historical', new='independent',
                 reason='all-off regression smoke requires valid baseline options without a historical memory'),
            dict(key='official_args.train_epochs', old=30, new=1, reason='independent smoke only; formal budget untouched'),
            dict(key='smoke.batch_limits', old=None, new=dict(train=3, evaluation=2), reason='smoke chain only'),
            dict(key='official_args.model_id', old=SOURCE_IDS['etth1'], new=alloff_id, reason='unique immutable smoke ID')])
    add(alloff, [])

    for dataset, source in sources.items():
        source_cfg = source['config']
        baseline_id = source_cfg['experiment']['baseline_experiment_id']
        ids = {(context, smoke): f'trustgtr_ab9_{context}__{dataset}__{"smoke" if smoke else "forecast"}__t96_h96__seed2024__{stamp}'
               for context in ('train', 'inference') for smoke in (True, False)}
        pair = {}
        common = {**copy.deepcopy(source_cfg['trustgtr']), **COMMON_GATE}
        for context in ('train', 'inference'):
            for smoke in (True, False):
                identifier = ids[context, smoke]
                config = copy.deepcopy(source_cfg)
                config['official_args'].update(model_id=identifier, train_epochs=1 if smoke else 30)
                config['trustgtr'] = {**common, 'gate_calibration_context': context}
                config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                      max_eval_batches=2 if smoke else None)
                changes = [dict(key='official_args.model_id', old=SOURCE_IDS[dataset], new=identifier,
                                reason='new immutable experiment ID; retain original A6 formal result')]
                for key, new in COMMON_GATE.items():
                    old = source_cfg['trustgtr'].get(key, {'gate_loss_weight': 0.05, 'gate_initial_probability': 0.9}.get(key))
                    changes.append(dict(key='trustgtr.' + key, old=old, new=new,
                        reason='shared control and inference setting; B6 retrieval gate and power-weighted calibration held fixed'))
                changes.append(dict(key='trustgtr.gate_calibration_context', old='train', new=context,
                    reason='sole paired scientific factor: same training-mode pass versus additional deterministic inference-mode gate calibration pass; original forecast loss and training budget unchanged'))
                if smoke:
                    changes.extend([
                        dict(key='official_args.train_epochs', old=30, new=1, reason='independent smoke only'),
                        dict(key='smoke.batch_limits', old=None, new=dict(train=3, evaluation=2), reason='smoke chain only; official batch unchanged')])
                original_exp = source_cfg['experiment']
                experiment = dict(id=identifier, dataset=dataset, model='TrustGTR_ab9_' + context,
                    stage='smoke' if smoke else 'improvement', step='6.4',
                    protocol=original_exp['protocol'], same_evaluation_protocol=not smoke,
                    official_script=original_exp['official_script'],
                    baseline_experiment_id=baseline_id,
                    baseline_mse=None, baseline_mae=None,
                    fixed_evaluation_asset_source=baseline_id,
                    source_experiment_id=SOURCE_IDS[dataset],
                    source_mse=source['metrics']['mse'], source_mae=source['metrics']['mae'],
                    source_provenance=source['provenance'],
                    control_experiment_id=ids['train', False],
                    paired_control_experiment_id=ids['train', False],
                    paired_experiment_id=ids['inference' if context == 'train' else 'train', smoke],
                    paired_factor='trustgtr.gate_calibration_context',
                    frozen_code_sha256=code_hashes, config_changes=changes,
                    comparison_note='Four-dataset A6 per-bank history held fixed; B shared retrieval descriptor/loss/init; C disabled only for attribution; train versus inference context is the sole pair difference. Existing GTR/A6 results are referenced, never rescheduled. Inference adds a calibration forward and may increase compute; official epochs/batches/optimizer/evaluation unchanged.')
                for key in ('paper_mse', 'paper_mae', 'paper_reference', 'paper_url'):
                    experiment[key] = original_exp.get(key)
                config['experiment'] = experiment
                deps = [alloff_id] if smoke else [alloff_id, ids[context, True]]
                if not smoke and context == 'inference':
                    deps.append(ids['train', False])
                add(config, deps, ids[context, True] if not smoke else None)
                pair[context, smoke] = config
        args_diff = {key: [pair['train', False]['official_args'][key], pair['inference', False]['official_args'][key]]
                     for key in pair['train', False]['official_args']
                     if pair['train', False]['official_args'][key] != pair['inference', False]['official_args'][key]}
        options_diff = {key: [pair['train', False]['trustgtr'][key], pair['inference', False]['trustgtr'][key]]
                        for key in common.keys() | {'gate_calibration_context'}
                        if pair['train', False]['trustgtr'][key] != pair['inference', False]['trustgtr'][key]}
        assert set(args_diff) == {'model_id'}
        assert options_diff == {'gate_calibration_context': ['train', 'inference']}
        assert pair['train', False]['data_asset'] == pair['inference', False]['data_asset'] == source_cfg['data_asset']
        assert all(not pair[context, False]['smoke']['enabled']
                   and pair[context, False]['smoke']['max_train_batches'] is None
                   and pair[context, False]['smoke']['max_eval_batches'] is None for context in ('train', 'inference'))
        checks.append(dict(dataset=dataset, source_a6_id=SOURCE_IDS[dataset], baseline_gtr_id=baseline_id,
            train_control_id=ids['train', False], inference_id=ids['inference', False],
            official_args_diff=args_diff, trustgtr_options_diff=options_diff,
            inherited_resolved_official_args=source['resolved'],
            same_assets=True, same_official_training_budget=True, same_parameter_architecture=True,
            unchanged_A6_retrieval=True, C_disabled=True,
            expected_added_parameters_vs_A6=321,
            overhead_note='Inference context adds a deterministic calibration forward; elapsed time/peak memory must be reported separately, not claimed equal compute',
            paper_values='Inherited verified source values; Weather/ETTm2 remain NA'))

    identifiers = [job['id'] for job in queue['jobs']]
    resolved_option_checks = []
    for path, config in generated:
        resolved = resolve_options(config['trustgtr'], config['official_args']['cycle'])
        config['experiment']['prepared_resolved_trustgtr'] = resolved
        resolved_option_checks.append(dict(id=config['experiment']['id'],
                                           valid=True, resolved_trustgtr=resolved))
    for check in checks:
        prepared = {config['experiment']['id']: config['experiment']['prepared_resolved_trustgtr']
                    for _, config in generated}
        left = prepared[check['train_control_id']]
        right = prepared[check['inference_id']]
        resolved_diff = {key: [left[key], right[key]] for key in left if left[key] != right[key]}
        assert resolved_diff == {'gate_calibration_context': ['train', 'inference']}
        check['resolved_trustgtr_options_diff'] = resolved_diff
    assert len(identifiers) == len(set(identifiers)) == 17
    assert sum(job['is_smoke'] for job in queue['jobs']) == 9
    assert all(dep in identifiers for job in queue['jobs'] for dep in job['depends_on'])
    assert not any(identifier in SOURCE_IDS.values() for identifier in identifiers)
    # Confirm acyclic ordering without depending on the scheduler implementation.
    remaining = {job['id']: set(job['depends_on']) for job in queue['jobs']}
    satisfied = set()
    while remaining:
        ready = {identifier for identifier, dependencies in remaining.items() if dependencies <= satisfied}
        if not ready:
            raise SystemExit('CONFIG_ERROR: dependency cycle in prepared jobs')
        satisfied.update(ready)
        remaining = {identifier: dependencies for identifier, dependencies in remaining.items() if identifier not in ready}
    queue['preparation']['config_sha256'] = {
        str(path): hashlib.sha256((json.dumps(config, ensure_ascii=False, indent=2) + '\n').encode()).hexdigest()
        for path, config in generated}
    audit = dict(prepared_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), step='6.4',
        queue_path=str(output), new_job_count=17, smoke_count=9, formal_count=8,
        launch_performed=False, max_jobs=3, science_sha256=code_hashes,
        paired_factor='trustgtr.gate_calibration_context', common_gate_settings=COMMON_GATE,
        protocol_checks=checks, dependency_graph=[{key: job[key] for key in ('id', 'depends_on')} for job in queue['jobs']],
        resolved_option_checks=resolved_option_checks,
        source_provenance=queue['preparation']['source_provenance'],
        memory_policy='Every new smoke starts with an unknown memory peak and original official batch; every formal job uses its own successful smoke measurement. No old GTR peak is borrowed for historical gated retrieval. Existing scheduler provides UUID choice, flock/shared reservations, nonblocking poll, maximum three, OOM backoff/requeue, safe stop and resume.',
        reference_summary_policy='All GTR baseline metrics stay null in input and are read dynamically by summarize_results.py only from SUCCESS metrics.json; A6 and paired control IDs/provenance are retained for dedicated attribution.',
        training_policy='Same epoch cap30/patience5/seed2024/batch/LR/optimizer/loader/split/scaler/metric/checkpoint policy; no AMP/gradient accumulation; deterministic inference calibration is the documented new factor.',
        disk_estimate=dict(new_run_count=17, observed_per_run_mib='15-16',
                           expected_new_runs_gib=0.26, available_gib_at_planning=1.2,
                           policy='No dataset/source copies or deletions; no change to outputs or scientific protocol'),
        scope='A6 history retained; A8 and counterfactual_risk not used; A/B/C scope preserved; no multi-seed or outside-baseline expansion')
    md = '\n'.join([
        '# 6.4 配对实验协议审计', '',
        f'准备时间：{audit["prepared_at"]}。仅生成配置；未启动 GPU。', '',
        '6.4.1：四数据集各一组 AB9_train / AB9_inference，A 固定已完成 A6 的 per_bank 历史检索，C 关闭以归因。',
        '6.4.2：两组共同开启 B；retrieval 描述、detach=true、power_weighted、权重0.05、初始门值0.9完全一致。唯一科学变化是 gate_calibration_context: train → inference。',
        '6.4.3：9项 smoke（含一项先行 TrustGTR 全关闭回归）+8项正式；每项正式先通过自身 smoke。每个 inference 正式还依赖相应 train 正式成功。',
        '6.4.4：正式配置沿用各数据集原 batch、30 epoch上限、patience5、seed2024、输入/预测96、原优化与评估协议。新增确定性校准前向会增加计算，须单独比较耗时与峰值。',
        '6.4.5：已完成 GTR/A6 仅作为落盘引用，不进入新队列；Weather/ETTm2 的论文指标保持 NA。源配置、指标、代码清单、数据、固定划分/归一化资产均记录哈希。', '',
        '| 数据集 | batch | LR | dropout | 周期 | control | inference |',
        '|---|---:|---:|---:|---:|---|---|',
        *[f'| {check["dataset"]} | {SPECS[check["dataset"]]["batch_size"]} | {SPECS[check["dataset"]]["learning_rate"]} | {SPECS[check["dataset"]]["dropout"]} | {SPECS[check["dataset"]]["cycle"]} | {check["train_control_id"]} | {check["inference_id"]} |' for check in checks], '',
        '启动命令（本脚本不执行）：', '',
        f'`bash scripts/queue.sh start {output.relative_to(ROOT)}`', '',
        f'状态：`bash scripts/queue.sh status {output.relative_to(ROOT)}`；日志：`tail -f runtime/step_6_4_ab9/queue.log`；停止/续跑使用同一脚本 stop/resume。', '',
        '显存：未知 smoke 独立估峰；正式使用各自 smoke 峰值。现有统一调度器执行 UUID、短时锁、动态补位、最多三并发及 OOM 退避回队列，不改变正式配置。', '',
    ])
    # Preflight every target before the first mutation; never overwrite immutable results.
    for path in [path for path, _ in generated] + [output, audit_path, audit_md]:
        if path.exists():
            raise SystemExit('Target exists; choose fresh output/audit/ID instead of overwriting: ' + str(path))
    for path, config in generated:
        write_new(path, config)
    write_new(audit_path, audit)
    audit_md.parent.mkdir(parents=True, exist_ok=True)
    with audit_md.open('x') as handle:
        handle.write(md)
    write_new(output, queue)
    print(output)
    print('Prepared 17 new jobs: 9 smoke and 8 formal, maximum 3; no GPU launched.')
    for job in queue['jobs']:
        print(job['id'], 'depends_on', job['depends_on'])


if __name__ == '__main__':
    main()
