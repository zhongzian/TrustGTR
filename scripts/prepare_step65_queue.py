#!/usr/bin/env python3
"""Prepare step 6.5 paired regime-contrast sampling jobs; never launch training.

Both groups use the completed A6 historical router and Euclidean contrast at
temperature 3, weight .002. Only contrast_sampling changes between the groups.
Existing successful GTR/A6 experiments are references, not queued reruns.
"""
import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

from prepare_step64_queue import (ROOT, SOURCE_IDS, SPECS, SCIENCE_FILES, defaults,
                                  digest, read, ready_model, verified_sources, write_new)


SAMPLINGS = ('regime_extremes', 'retrieval_hard')
VARIANTS = dict(regime_extremes='ac10_extremes', retrieval_hard='ac10_hard')
COMMON_CONTRAST = dict(use_multi_cycle=True, use_reliability_gate=False,
                       use_regime_contrast=True, contrast_metric='euclidean',
                       contrast_temperature=3.0, contrast_loss_weight=0.002,
                       positive_max_distance=1.0, regime_distance_gap=0.25,
                       historical_router_mode='per_bank',
                       historical_strength_coordinates='bounded',
                       gate_calibration_context='train')


def existing_queue_check(output):
    queue = read(output)
    if queue['preparation']['step'] != '6.5':
        raise SystemExit('Wrong stage at requested immutable queue path')
    for name, expected in queue['preparation']['science_sha256'].items():
        if digest(ROOT / name) != expected:
            raise SystemExit('Immutable queue source changed: ' + name)
    for job in queue['jobs']:
        path = job['argv'][job['argv'].index('--config') + 1]
        if digest(path) != queue['preparation']['config_sha256'][path]:
            raise SystemExit('Immutable input config changed: ' + path)
        asset = read(path)['data_asset']
        if digest(asset['path']) != asset['sha256']:
            raise SystemExit('DATA_ERROR: immutable queue dataset changed')
    print('Existing immutable step 6.5 queue retained; resume rather than create duplicate IDs: ' + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--queue', dest='output', default='configs/queue_step_6_5_ac10.json')
    parser.add_argument('--audit', default='reports/step_6_5_protocol_audit.json')
    parser.add_argument('--check-only', action='store_true',
                        help='Verify completed references/defaults without freezing code or writing configurations')
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
        print('Step 6.5 references verified: four successful A6/GTR pairs, original official budgets, dataset/split/scaler hashes; no files written.')
        return
    resolve_options = ready_model()
    probe = resolve_options({**sources['etth1']['config']['trustgtr'], **COMMON_CONTRAST}, 24)
    if probe.get('contrast_sampling') != 'regime_extremes':
        raise SystemExit('Scientific implementation not ready: legacy-default contrast_sampling required')
    for sampling in SAMPLINGS:
        resolve_options({**probe, 'contrast_sampling': sampling}, 24)
    code_hashes = {name: digest(ROOT / name) for name in SCIENCE_FILES}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    alloff_id = f'trustgtr_alloff__etth1__smoke__t96_h96__seed2024__{stamp}'
    generated, checks = [], []
    queue = dict(max_jobs=3, poll_seconds=3, oom_cooldown_seconds=60,
        runtime_dir=str(ROOT / 'runtime/step_6_5_ac10'),
        on_change_argv=[str(ROOT / '.conda/bin/python'), str(ROOT / 'scripts/summarize_results.py')],
        preparation=dict(step='6.5', datasets=list(SOURCE_IDS), new_job_count=17,
            scope='Single-seed four-dataset paired C sampling test; A6 fixed, B off for attribution; GTR/A6 results referenced only',
            science_sha256=code_hashes,
            source_provenance={dataset: source['provenance'] for dataset, source in sources.items()}), jobs=[])

    def add(config, dependencies, memory_source=None):
        identifier = config['experiment']['id']
        path = ROOT / 'configs' / (identifier + '.json')
        run_dir = ROOT / 'runs' / identifier
        if path.exists() or run_dir.exists():
            raise SystemExit('Unique experiment ID already exists: ' + identifier)
        resolved = resolve_options(config['trustgtr'], config['official_args']['cycle'])
        config['trustgtr'] = resolved
        config['experiment']['prepared_resolved_trustgtr'] = copy.deepcopy(resolved)
        job = dict(id=identifier, argv=[str(ROOT / '.conda/bin/python'), '-u',
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
        gate_calibration_context='train', contrast_sampling='regime_extremes')
    alloff['smoke'] = dict(enabled=True, max_train_batches=3, max_eval_batches=2)
    alloff['experiment'] = dict(id=alloff_id, dataset='etth1', model='TrustGTR_alloff',
        stage='smoke', step='6.5', protocol=source['config']['experiment']['protocol'],
        same_evaluation_protocol=False, source_experiment_id=SOURCE_IDS['etth1'],
        source_provenance=source['provenance'],
        baseline_experiment_id=source['config']['experiment']['baseline_experiment_id'],
        fixed_evaluation_asset_source=source['config']['experiment']['baseline_experiment_id'],
        frozen_code_sha256=code_hashes,
        comparison_note='Fresh all-modules-off regression smoke after C sampling integration; not a formal result',
        config_changes=[
            dict(key='trustgtr.module_switches', old='A6 on; B/C off', new='A/B/C all off',
                 reason='confirm unchanged original GTR behavior through TrustGTR entry'),
            dict(key='trustgtr.memory_mode', old='historical', new='independent',
                 reason='valid all-off baseline without historical memory'),
            dict(key='official_args.train_epochs', old=30, new=1, reason='independent smoke only; formal budget unchanged'),
            dict(key='smoke.batch_limits', old=None, new=dict(train=3, evaluation=2), reason='smoke chain only'),
            dict(key='official_args.model_id', old=SOURCE_IDS['etth1'], new=alloff_id, reason='unique immutable smoke ID')])
    add(alloff, [])

    for dataset, source in sources.items():
        source_cfg = source['config']
        original_exp = source_cfg['experiment']
        baseline_id = original_exp['baseline_experiment_id']
        old_options = resolve_options(source_cfg['trustgtr'], source_cfg['official_args']['cycle'])
        common = {**old_options, **COMMON_CONTRAST}
        ids = {(sampling, smoke): f'trustgtr_{VARIANTS[sampling]}__{dataset}__{"smoke" if smoke else "forecast"}__t96_h96__seed2024__{stamp}'
               for sampling in SAMPLINGS for smoke in (True, False)}
        pair = {}
        for sampling in SAMPLINGS:
            for smoke in (True, False):
                identifier = ids[sampling, smoke]
                config = copy.deepcopy(source_cfg)
                config['official_args'].update(model_id=identifier, train_epochs=1 if smoke else 30)
                config['trustgtr'] = {**common, 'contrast_sampling': sampling}
                config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                      max_eval_batches=2 if smoke else None)
                changes = [dict(key='official_args.model_id', old=SOURCE_IDS[dataset], new=identifier,
                                reason='unique immutable experiment ID; preserve completed A6 result')]
                changes.extend(dict(key='trustgtr.' + key, old=old_options[key], new=new,
                    reason='shared C setting in both groups; fix loss geometry/scale, A6 router and B-off attribution')
                    for key, new in COMMON_CONTRAST.items() if old_options[key] != new)
                changes.append(dict(key='trustgtr.contrast_sampling', old='regime_extremes', new=sampling,
                    reason='sole paired scientific factor: raw regime extremes versus different-regime negatives among actual learned retrieval neighbors'))
                if smoke:
                    changes.extend([
                        dict(key='official_args.train_epochs', old=30, new=1, reason='independent smoke only'),
                        dict(key='smoke.batch_limits', old=None, new=dict(train=3, evaluation=2), reason='smoke chain only; official batch unchanged')])
                experiment = dict(id=identifier, dataset=dataset, model='TrustGTR_' + VARIANTS[sampling],
                    stage='smoke' if smoke else 'improvement', step='6.5',
                    protocol=original_exp['protocol'], same_evaluation_protocol=not smoke,
                    official_script=original_exp['official_script'],
                    baseline_experiment_id=baseline_id, baseline_mse=None, baseline_mae=None,
                    fixed_evaluation_asset_source=baseline_id,
                    source_experiment_id=SOURCE_IDS[dataset],
                    source_mse=source['metrics']['mse'], source_mae=source['metrics']['mae'],
                    source_provenance=source['provenance'],
                    control_experiment_id=ids['regime_extremes', False],
                    paired_control_experiment_id=ids['regime_extremes', False],
                    paired_experiment_id=ids['retrieval_hard' if sampling == 'regime_extremes' else 'regime_extremes', smoke],
                    paired_factor='trustgtr.contrast_sampling',
                    frozen_code_sha256=code_hashes, config_changes=changes,
                    comparison_note='Four-dataset A6 per-bank causal train history fixed; B off only for attribution. Both C groups share Euclidean metric/T3/weight .002/distance1/gap .25; only sampling differs. This fresh control is not the prior cosine/T0.1 AC7. Existing successful GTR/A6 only referenced; original official budget/evaluation/assets unchanged. A/B/C research scope retained.')
                for key in ('paper_mse', 'paper_mae', 'paper_reference', 'paper_url'):
                    experiment[key] = original_exp.get(key)
                config['experiment'] = experiment
                deps = [alloff_id] if smoke else [alloff_id, ids[sampling, True]]
                add(config, deps, ids[sampling, True] if not smoke else None)
                pair[sampling, smoke] = config
        left, right = pair['regime_extremes', False], pair['retrieval_hard', False]
        args_diff = {key: [left['official_args'][key], right['official_args'][key]]
                     for key in left['official_args'] if left['official_args'][key] != right['official_args'][key]}
        options_diff = {key: [left['trustgtr'][key], right['trustgtr'][key]]
                        for key in left['trustgtr'] if left['trustgtr'][key] != right['trustgtr'][key]}
        assert set(args_diff) == {'model_id'}
        assert options_diff == {'contrast_sampling': list(SAMPLINGS)}
        assert left['data_asset'] == right['data_asset'] == source_cfg['data_asset']
        assert all(not pair[sampling, False]['smoke']['enabled']
                   and pair[sampling, False]['smoke']['max_train_batches'] is None
                   and pair[sampling, False]['smoke']['max_eval_batches'] is None for sampling in SAMPLINGS)
        checks.append(dict(dataset=dataset, source_a6_id=SOURCE_IDS[dataset], baseline_gtr_id=baseline_id,
            extremes_control_id=ids['regime_extremes', False], hard_id=ids['retrieval_hard', False],
            official_args_diff=args_diff, trustgtr_options_diff=options_diff,
            inherited_resolved_official_args=source['resolved'],
            same_assets=True, same_official_training_budget=True, same_parameter_architecture=True,
            unchanged_A6_router=True, B_disabled=True, expected_added_parameters_vs_A6=0,
            overhead_note='C trains the existing retrieval encoder; report elapsed time/peak memory rather than claim equal compute',
            paper_values='Inherited verified source values; Weather/ETTm2 remain NA'))

    identifiers = [job['id'] for job in queue['jobs']]
    assert len(identifiers) == len(set(identifiers)) == 17
    assert sum(job['is_smoke'] for job in queue['jobs']) == 9
    assert all(dep in identifiers for job in queue['jobs'] for dep in job['depends_on'])
    assert not any(identifier in SOURCE_IDS.values() for identifier in identifiers)
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
    audit = dict(prepared_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), step='6.5',
        queue_path=str(output), new_job_count=17, smoke_count=9, formal_count=8,
        launch_performed=False, max_jobs=3, science_sha256=code_hashes,
        preparation_script_sha256=digest(__file__),
        paired_factor='trustgtr.contrast_sampling', common_contrast_settings=COMMON_CONTRAST,
        protocol_checks=checks,
        dependency_graph=[{key: job[key] for key in ('id', 'depends_on')} for job in queue['jobs']],
        resolved_option_checks=[dict(id=config['experiment']['id'], valid=True,
                                    resolved_trustgtr=config['trustgtr']) for _, config in generated],
        source_provenance=queue['preparation']['source_provenance'],
        memory_policy='Unknown smoke peaks with unchanged official batch; each formal uses its own successful smoke. Existing scheduler uses UUID/flock/shared reservations/max3/nonblocking poll/OOM backoff/requeue/safe stop/resume.',
        reference_summary_policy='Baseline metrics remain null in input and are dynamically read from SUCCESS metrics.json. A6 and paired C control provenance retained; no GTR/A6 training reruns.',
        training_policy='Same cap30/patience5/seed2024/batch/LR/optimizer/loader/split/scaler/metric/checkpoint policy; no AMP/gradient accumulation; only paired sampling differs.',
        disk_estimate=dict(new_run_count=17, observed_per_run_mib='15-16', expected_new_runs_gib=0.30,
                           available_gib_at_preparation=round(shutil.disk_usage(ROOT).free / 2**30, 3),
                           policy='No dataset/source copies or deletions; no protocol changes'),
        scope='A6 history retained; B off for attribution; full A/B/C scope preserved; no multi-seed or outside-baseline expansion')
    md = '\n'.join([
        '# 6.5 配对实验协议审计', '', f'准备时间：{audit["prepared_at"]}。仅生成配置；未启动 GPU。', '',
        '6.5.2：四数据集各一组 AC10_extremes / AC10_hard，A 固定已完成 A6 的 per_bank/bounded 历史检索，B 关闭以归因。',
        '两组共同开启 C：euclidean、temperature=3、weight=.002、positive_max_distance=1、regime_distance_gap=.25。唯一科学差异为 contrast_sampling: regime_extremes → retrieval_hard。',
        '此处为新建相同欧氏距离/温度的 control；不将历史 cosine/T=.1 的 AC7 冒充本轮 control。',
        '9项 smoke（含全模块关闭回归）+8项正式；每项正式只依赖全关闭回归和自身 smoke，不依赖对照正式实验，资源可用即可补位。',
        '正式沿用原 batch、30 epoch上限、patience5、seed2024、输入/预测96、原优化/评估协议。所有 TrustGTR 默认值均显式解析进入输入配置；启动 runner 再保存最终 resolved_config。',
        '已完成 GTR/A6 只引用，不重新训练。固定数据、划分、scaler、代码、源配置和成功指标均校验哈希。', '',
        '| 数据集 | batch | LR | head dropout | 周期 | extremes control | hard |',
        '|---|---:|---:|---:|---:|---|---|',
        *[f'| {check["dataset"]} | {SPECS[check["dataset"]]["batch_size"]} | {SPECS[check["dataset"]]["learning_rate"]} | {SPECS[check["dataset"]]["dropout"]} | {SPECS[check["dataset"]]["cycle"]} | {check["extremes_control_id"]} | {check["hard_id"]} |' for check in checks], '',
        '启动命令（本脚本不执行）：', '', f'`bash scripts/queue.sh start {output.relative_to(ROOT)}`', '',
        f'状态：`bash scripts/queue.sh status {output.relative_to(ROOT)}`；日志：`tail -f runtime/step_6_5_ac10/queue.log`；安全停止/续跑使用同脚本 stop/resume。', '',
        '显存：未知 smoke 单独估峰，正式使用各自 smoke。UUID、短时锁、动态补位、最多三并发、OOM退避回队列保持现有调度器；不改变任何正式参数。', '',
    ])
    for path in [path for path, _ in generated] + [output, audit_path, audit_md]:
        if path.exists():
            raise SystemExit('Target exists; choose fresh output/audit/ID rather than overwrite: ' + str(path))
    for path, config in generated:
        write_new(path, config)
    write_new(audit_path, audit)
    with audit_md.open('x') as handle:
        handle.write(md)
    write_new(output, queue)
    print(output)
    print('Prepared 17 new jobs: 9 smoke and 8 formal, maximum 3; no GPU launched.')
    for job in queue['jobs']:
        print(job['id'], 'depends_on', job['depends_on'])


if __name__ == '__main__':
    main()
