#!/usr/bin/env python3
"""Summarize persisted step 6.5 paired C runs; never select checkpoints on test."""
import argparse
import json
import math
from pathlib import Path

from prepare_step64_queue import ROOT, SOURCE_IDS, digest
from prepare_step65_queue import COMMON_CONTRAST, SAMPLINGS

EXCLUDED_ARGS = {'model', 'model_id', 'checkpoints', 'trustgtr'}


def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Required persisted artifact missing: ' + str(path))
    return json.loads(path.read_text())


def official_protocol(resolved):
    return {key: value for key, value in resolved['official_args'].items() if key not in EXCLUDED_ARGS}


def completed_run(identifier):
    run = ROOT / 'runs' / identifier
    metrics = read(run / 'metrics.json')
    status = read(run / 'status.json')
    resolved = read(run / 'resolved_config.yaml')
    if metrics.get('status') != 'SUCCESS' or status.get('status') != 'SUCCESS' or status.get('exit_code') != 0:
        raise RuntimeError('Formal experiment not successfully complete: ' + identifier)
    if metrics.get('experiment_id') != identifier or status.get('experiment_id') != identifier:
        raise ValueError('Artifact experiment ID mismatch: ' + identifier)
    if resolved['smoke']['enabled'] or metrics.get('stage') == 'smoke':
        raise ValueError('Smoke cannot enter formal comparison: ' + identifier)
    for name in ('mse', 'mae'):
        if not math.isfinite(metrics[name]) or metrics[name] < 0:
            raise ValueError('Invalid persisted metric: ' + identifier + '/' + name)
    if metrics['seed'] != 2024 or resolved['official_args']['random_seed'] != 2024:
        raise ValueError('Seed changed: ' + identifier)
    return run, metrics, resolved


def verified_assets(metrics):
    assets = metrics['evaluation_assets']
    if set(assets) != {'train', 'val', 'test'}:
        raise ValueError('Expected three fixed evaluation splits')
    result = {}
    for flag, record in assets.items():
        if digest(record['path']) != record['sha256']:
            raise ValueError('Fixed evaluation asset hash changed: ' + record['path'])
        result[flag] = {**record, 'scaler_sha256': digest(record['scaler'])}
    return result


def references(dataset):
    source_id = SOURCE_IDS[dataset]
    _, source, source_resolved = completed_run(source_id)
    source_config = read(ROOT / 'configs' / (source_id + '.json'))
    baseline_id = source_config['experiment']['baseline_experiment_id']
    _, baseline, baseline_resolved = completed_run(baseline_id)
    options = source_resolved['trustgtr']
    expected = dict(use_multi_cycle=True, use_reliability_gate=False,
                    use_regime_contrast=False, memory_mode='historical',
                    historical_router_mode='per_bank', historical_strength_coordinates='bounded')
    if {key: options[key] for key in expected} != expected:
        raise ValueError('Reference is not completed A6: ' + source_id)
    if official_protocol(source_resolved) != official_protocol(baseline_resolved):
        raise ValueError('GTR/A6 reference budget differs: ' + dataset)
    if source['evaluation_assets'] != baseline['evaluation_assets']:
        raise ValueError('GTR/A6 reference fixed assets differ: ' + dataset)
    asset = source_config['data_asset']
    if digest(asset['path']) != asset['sha256']:
        raise ValueError('Reference raw dataset changed: ' + dataset)
    asset_provenance = verified_assets(source)
    # Scaler hashes are anchored to the shared successful GTR/A6 references.
    baseline_assets = verified_assets(baseline)
    if asset_provenance != baseline_assets:
        raise ValueError('GTR/A6 scaler provenance differs: ' + dataset)
    return dict(source_id=source_id, baseline_id=baseline_id,
                source=source, baseline=baseline,
                source_resolved=source_resolved, baseline_resolved=baseline_resolved,
                data_asset=asset, evaluation_provenance=asset_provenance)


def differences(left, right):
    return {key: [left.get(key), right.get(key)] for key in left.keys() | right.keys()
            if left.get(key) != right.get(key)}


def changes(current, reference):
    return ({name: current[name] - reference[name] for name in ('mse', 'mae')},
            {name: 100 * (current[name] - reference[name]) / reference[name] for name in ('mse', 'mae')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', default='configs/queue_step_6_5_ac10.json')
    parser.add_argument('--output', default='reports/step_6_5_paired_results.json')
    parser.add_argument('--check-references', action='store_true',
                        help='Read/validate completed GTR/A6 reference artifacts; write no reports')
    args = parser.parse_args()
    refs = {dataset: references(dataset) for dataset in SOURCE_IDS}
    if args.check_references:
        print('Step 6.5 reference reads passed: eight SUCCESS formal artifacts, seed2024, identical GTR/A6 budgets, fixed data/split/scaler hashes; no output written.')
        return
    output = ROOT / args.output
    markdown = output.with_suffix('.md')
    for path in (output, markdown):
        if path.exists():
            raise FileExistsError('Refuse to overwrite final paired report: ' + str(path))
    queue_path = ROOT / args.queue
    queue = read(queue_path)
    if queue['preparation']['step'] != '6.5':
        raise ValueError('Expected immutable step 6.5 queue')
    rows, pairs = [], {}
    for job in queue['jobs']:
        if job['is_smoke']:
            continue
        run, metrics, resolved = completed_run(job['id'])
        if run.resolve() != Path(job['run_dir']).resolve():
            raise ValueError('Queue run directory mismatch')
        config_path = Path(job['argv'][job['argv'].index('--config') + 1])
        config = read(config_path)
        if digest(config_path) != queue['preparation']['config_sha256'][str(config_path)]:
            raise ValueError('Frozen config hash changed: ' + str(config_path))
        dataset = config['experiment']['dataset']
        reference = refs[dataset]
        sampling = resolved['trustgtr']['contrast_sampling']
        if sampling not in SAMPLINGS or sampling != config['trustgtr']['contrast_sampling']:
            raise ValueError('Resolved C sampling changed')
        if config['experiment']['source_experiment_id'] != reference['source_id'] or config['experiment']['baseline_experiment_id'] != reference['baseline_id']:
            raise ValueError('Unexpected A6/GTR reference')
        if resolved['trustgtr'] != config['trustgtr'] or resolved['trustgtr'] != config['experiment']['prepared_resolved_trustgtr']:
            raise ValueError('Actual resolved options differ from explicit frozen input')
        if {key: resolved['trustgtr'][key] for key in COMMON_CONTRAST} != COMMON_CONTRAST:
            raise ValueError('Shared A6/C setting changed: ' + job['id'])
        if official_protocol(resolved) != official_protocol(reference['source_resolved']) or official_protocol(resolved) != official_protocol(reference['baseline_resolved']):
            raise ValueError('Official budget differs from GTR/A6: ' + job['id'])
        if metrics['evaluation_assets'] != reference['source']['evaluation_assets'] or metrics['evaluation_assets'] != reference['baseline']['evaluation_assets']:
            raise ValueError('Formal fixed evaluation assets differ')
        if resolved['data_asset'] != reference['data_asset'] or config['data_asset'] != reference['data_asset']:
            raise ValueError('Raw data provenance differs')
        actual_assets = verified_assets(metrics)
        if actual_assets != reference['evaluation_provenance']:
            raise ValueError('Fixed split/scaler hashes differ')
        expected_assets = config['experiment']['source_provenance']['evaluation_assets']
        if actual_assets != expected_assets:
            raise ValueError('Fixed asset provenance differs from frozen source')
        if metrics['parameters'] != reference['source']['parameters']:
            raise ValueError('C must add zero parameters to A6')
        manifest = read(run / 'code_manifest.json')
        expected_science = queue['preparation']['science_sha256']
        if config['experiment']['frozen_code_sha256'] != expected_science:
            raise ValueError('Config and queue scientific hashes differ')
        for name, expected in expected_science.items():
            if manifest['sha256'].get(name) != expected:
                raise ValueError('Actual run scientific code changed: ' + name)
        checkpoint = Path(metrics['checkpoint'])
        if not checkpoint.is_file() or run.resolve() not in checkpoint.resolve().parents:
            raise ValueError('Best checkpoint missing or outside the run')
        if not 1 <= metrics['best_epoch'] <= metrics['epochs_completed'] <= resolved['official_args']['train_epochs']:
            raise ValueError('Invalid recorded best epoch/training budget')
        abs_gtr, rel_gtr = changes(metrics, reference['baseline'])
        abs_a6, rel_a6 = changes(metrics, reference['source'])
        row = {key: metrics.get(key) for key in ('experiment_id', 'dataset', 'model', 'mse', 'mae',
            'runtime_seconds', 'training_time_seconds', 'inference_time_seconds', 'inference_seconds_per_batch',
            'peak_gpu_memory_mib', 'parameters', 'trainable_parameters', 'epochs_completed', 'best_epoch',
            'checkpoint', 'trustgtr_diagnostics', 'gpu_uuid', 'git_commit')}
        row.update(status='SUCCESS', exit_code=0, sampling=sampling, seed=metrics['seed'],
            baseline_experiment_id=reference['baseline_id'], source_experiment_id=reference['source_id'],
            baseline_scores={name: reference['baseline'][name] for name in ('mse', 'mae')},
            source_scores={name: reference['source'][name] for name in ('mse', 'mae')},
            absolute_change_vs_gtr=abs_gtr, relative_change_vs_gtr_pct=rel_gtr,
            absolute_change_vs_a6=abs_a6, relative_change_vs_a6_pct=rel_a6,
            additional_parameters_vs_a6=0,
            additional_parameters_vs_gtr=metrics['parameters'] - reference['baseline']['parameters'],
            config_path=str(config_path), config_sha256=digest(config_path),
            metrics_path=str(run / 'metrics.json'), metrics_sha256=digest(run / 'metrics.json'),
            checkpoint_sha256=digest(checkpoint),
            resolved_config_path=str(run / 'resolved_config.yaml'), resolved_config_sha256=digest(run / 'resolved_config.yaml'),
            code_manifest_path=str(run / 'code_manifest.json'), code_manifest_sha256=digest(run / 'code_manifest.json'),
            scientific_code_sha256=expected_science, evaluation_assets=actual_assets,
            same_training_budget_and_evaluation_assets=True)
        if sampling in pairs.setdefault(dataset, {}):
            raise ValueError('Duplicate dataset/sampling formal result')
        pairs[dataset][sampling] = (row, resolved)
        rows.append(row)
    if len(rows) != 8 or set(pairs) != set(SOURCE_IDS):
        raise ValueError('Expected eight formal results across exactly four datasets')
    comparisons = []
    for dataset, pair in pairs.items():
        if set(pair) != set(SAMPLINGS):
            raise ValueError('Incomplete C pair: ' + dataset)
        control, left = pair['regime_extremes']
        hard, right = pair['retrieval_hard']
        if official_protocol(left) != official_protocol(right):
            raise ValueError('Actual pair official settings differ')
        diff = differences(left['trustgtr'], right['trustgtr'])
        if diff != {'contrast_sampling': list(SAMPLINGS)}:
            raise ValueError('Actual pair must change only contrast_sampling: ' + str(diff))
        if control['parameters'] != hard['parameters'] or control['additional_parameters_vs_a6'] != 0:
            raise ValueError('Actual pair parameter architecture differs from A6')
        absolute, relative = changes(hard, control)
        comparisons.append(dict(dataset=dataset,
            extremes_control_experiment_id=control['experiment_id'], hard_experiment_id=hard['experiment_id'],
            scores_extremes={name: control[name] for name in ('mse', 'mae')},
            scores_hard={name: hard[name] for name in ('mse', 'mae')},
            absolute_change_hard_vs_extremes=absolute, relative_change_hard_vs_extremes_pct=relative,
            independent_variable=diff))
    report = dict(step='6.5', formal_runs_complete=8, datasets=4, single_seed_exploratory=True,
        paired_factor='trustgtr.contrast_sampling', common_contrast_settings=COMMON_CONTRAST,
        legacy_ac7_is_not_this_control='Old AC7 uses cosine/T0.1; both new C groups use euclidean/T3. C10 versus old AC7 cannot isolate sampling.',
        cumulative_training_contrast_statistics_are_not_best_checkpoint_fixed_validation=True,
        cycle_statistics_mix_train_val_test=True,
        speed_not_strict_benchmark_due_to_shared_gpu_concurrency=True,
        best_checkpoint_selected_by_original_validation_only=True,
        queue_path=str(queue_path), queue_sha256=digest(queue_path),
        experiments=rows, comparisons=comparisons)
    lines = ['# 6.5 实际检索近邻困难负例配对正式结果', '',
        '四数据集、T96/H96、seed2024、原官方预算与固定资产。两组共同为 A6+C（B关闭），euclidean/T3/λ=.002，只有 contrast_sampling 改变。负变化表示误差下降。', '',
        '| dataset | extremes MSE | extremes MAE | hard MSE | hard MAE | MSE change % | MAE change % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for pair in comparisons:
        a, b, d = pair['scores_extremes'], pair['scores_hard'], pair['relative_change_hard_vs_extremes_pct']
        lines.append('| {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} | {:.4f} | {:.4f} |'.format(
            pair['dataset'], a['mse'], a['mae'], b['mse'], b['mae'], d['mse'], d['mae']))
    lines += ['', '原GTR/A6的绝对与相对差、耗时、峰值显存、参数、代码/配置/指标/checkpoint哈希见JSON；两组相对A6新增参数均为0。',
        '历史AC7是cosine/T=.1，不能将本轮C10与其差异归因于采样；本轮新建extremes control才用于采样归因。',
        'contrast统计为累计训练采样结果，并非best checkpoint固定val诊断；cycle统计混合train/val/test。',
        '实际耗时受共享GPU与并发影响，不能作为严格速度基准。单seed只用于方向筛选，不作统计稳定或投稿等级结论。',
        '只使用原验证集规则选出的best checkpoint，不按test重新选择或修改配置。', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with markdown.open('x') as handle:
        handle.write('\n'.join(lines))
    print(output)


if __name__ == '__main__':
    main()
