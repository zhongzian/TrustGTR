#!/usr/bin/env python3
"""Validated causal historical memory, gate and regime retrieval: equal budget."""
import copy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'configs/queue_historical_v4.json'
    if output.exists():
        raise SystemExit('Existing queue is immutable; resume it')
    queue = json.loads((ROOT / 'configs/queue_baseline_v3.json').read_text())
    source = next(job for job in queue['jobs'] if '__etth1__forecast__' in job['id'])
    baseline = json.loads(Path(source['argv'][source['argv'].index('--config') + 1]).read_text())
    measured = json.loads((Path(source['run_dir']) / 'metrics.json').read_text())
    assert measured['status'] == 'SUCCESS'
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')

    def add(config, run_id, dependencies, entry, memory_source=None):
        config_path = ROOT / 'configs' / (run_id + '.json')
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        run_dir = ROOT / 'runs' / run_id
        job = dict(id=run_id, argv=[str(ROOT / '.conda/bin/python'), '-u',
            str(ROOT / 'scripts' / entry), '--config', str(config_path), '--run-dir', str(run_dir)],
            cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None,
            is_smoke=config['smoke']['enabled'], smoke_passed=False, depends_on=dependencies)
        if memory_source:
            job['memory_estimate_from'] = memory_source
        queue['jobs'].append(job)

    recheck_id = f'gtr__etth1__smoke_recheck__t96_h96__seed2024__{stamp}'
    config = copy.deepcopy(baseline)
    config['official_args'].update(model_id=recheck_id, train_epochs=1)
    config['smoke'] = dict(enabled=True, max_train_batches=3, max_eval_batches=2)
    config['experiment'].update(id=recheck_id, stage='smoke', same_evaluation_protocol=False,
        comparison_note='Original baseline smoke after historical memory integration',
        config_changes=[dict(key='train_epochs', old=30, new=1, reason='independent smoke only')])
    add(config, recheck_id, [source['id']], 'run_experiment.py')

    formal_ids = {}
    for variant, gate, contrast in [('a4', False, False), ('ab4', True, False), ('abc4', True, True)]:
        options = dict(use_multi_cycle=True, use_reliability_gate=gate, use_regime_contrast=contrast,
            memory_mode='historical', preserve_initialization_rng=True,
            gate_descriptor='forecast', detach_gate_descriptor=True, gate_loss_mode='power_weighted',
            contrast_loss_weight=0.002, historical_topk=3,
            historical_candidate_cap=192, historical_blend_bound=0.2)
        smoke_id = f'trustgtr_{variant}__etth1__smoke__t96_h96__seed2024__{stamp}'
        formal_id = smoke_id.replace('__smoke__', '__forecast__')
        formal_ids[variant] = formal_id
        for smoke in (True, False):
            run_id = smoke_id if smoke else formal_id
            config = copy.deepcopy(baseline)
            config['official_args'].update(model='TrustGTR', model_id=run_id, train_epochs=1 if smoke else 30)
            config['trustgtr'] = options
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None,
                                  max_eval_batches=2 if smoke else None)
            changes = [dict(key='model', old='GTR', new='TrustGTR', reason='switchable historical retrieval variant'),
                dict(key='memory_mode', old='factorized_residual', new='historical',
                     reason='long calendar banks overfit; fixed full-val historical retrieval shows complementary content'),
                dict(key='retrieval_candidates', old='one learned value per calendar cell',
                     new='train-only completed continuations, same base phase, three relative-age banks, at most 64 each',
                     reason='input state chooses actual history candidates; candidate future end <= query origin'),
                dict(key='long_bank_phase_rule', old='origin modulo 7x/30x base cycle',
                     new='all candidates align base cycle; memory banks cover short/middle/long historical cycle ages',
                     reason='validated three-age candidate coverage; periods retained as legacy source identifiers only'),
                dict(key='retrieval_fusion', old='modify learned Q before convolution',
                     new='original GTR forecast + 0.2*tanh(trainable scalar initialized 0)*retrieved forecast correction',
                     reason='retrieval standalone is weaker; initial prediction and RNG exactly match baseline'),
                dict(key='historical_blend_bound', old=None, new=0.2,
                     reason='fixed validation diagnostic blends 0.1/0.2 improve both metrics; no test tuning'),
                dict(key='trustgtr', old='disabled', new=options,
                     reason='A alone, add B, then add C; same original training budget and fixed assets')]
            if gate:
                changes.append(dict(key='gate_fallback', old='local convolution/MLP branch',
                    new='original jointly trained GTR forecast branch', reason='reject historical correction using train-only utility oracle'))
            if contrast:
                changes.append(dict(key='contrast_candidate_source', old='same-batch same-phase samples',
                    new='fixed causal training history candidates with disjoint windows',
                    reason='train actual retrieval key encoder; avoid sparse Solar/batch positives without stale queue'))
            if smoke:
                changes.append(dict(key='train_epochs', old=30, new=1, reason='independent smoke only'))
            config['experiment'].update(id=run_id, model='TrustGTR_' + variant,
                stage='smoke' if smoke else 'improvement', baseline_experiment_id=source['id'],
                baseline_mse=measured['mse'], baseline_mae=measured['mae'], same_evaluation_protocol=not smoke,
                comparison_note='Same seed/budget/evaluation assets; validation-only method diagnosis, no test selection',
                config_changes=changes)
            deps = [recheck_id] if smoke else [smoke_id]
            if not smoke and variant != 'a4':
                deps.append(formal_ids['a4'])
            add(config, run_id, deps, 'run_trust_experiment.py', smoke_id if not smoke else None)

    output.write_text(json.dumps(queue, indent=2) + '\n')
    print(output)
    for variant, run_id in formal_ids.items():
        print(variant, run_id)


if __name__ == '__main__':
    main()
