#!/usr/bin/env python3
"""Check reliability and the complete method on the second existing baseline."""
import copy
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'configs/queue_solar_screen.json'
    if output.exists():
        raise SystemExit('Solar screen already exists; use resume')
    queue = json.loads((ROOT / 'configs/queue_baseline_v3.json').read_text())
    source = next(job for job in queue['jobs'] if '__solar__forecast__' in job['id'])
    baseline = json.loads(Path(source['argv'][source['argv'].index('--config') + 1]).read_text())
    measured = json.loads((Path(source['run_dir']) / 'metrics.json').read_text())
    assert measured['status'] == 'SUCCESS'
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    for variant, options in [('b3', dict(use_reliability_gate=True)),
                             ('abc3', dict(use_multi_cycle=True, use_reliability_gate=True, use_regime_contrast=True,
                                           memory_mode='factorized_residual', contrast_loss_weight=0.002))]:
        options.update(preserve_initialization_rng=True, gate_descriptor='forecast',
                       detach_gate_descriptor=True, gate_loss_mode='power_weighted')
        smoke_id = f'trustgtr_{variant}__solar__smoke__t96_h96__seed2024__{stamp}'
        for smoke in [True, False]:
            run_id = smoke_id if smoke else smoke_id.replace('__smoke__', '__forecast__')
            config = copy.deepcopy(baseline)
            config['official_args'].update(model='TrustGTR', model_id=run_id, train_epochs=1 if smoke else 30)
            config['trustgtr'] = options
            config['smoke'] = dict(enabled=smoke, max_train_batches=3 if smoke else None, max_eval_batches=2 if smoke else None)
            config['experiment'].update(id=run_id, model='TrustGTR_' + variant,
                stage='smoke' if smoke else 'improvement', baseline_experiment_id=source['id'],
                baseline_mse=measured['mse'], baseline_mae=measured['mae'], same_evaluation_protocol=not smoke,
                comparison_note='Second originally selected dataset, existing official baseline; no new seed or budget',
                config_changes=[{'key': 'model', 'old': 'GTR', 'new': 'TrustGTR', 'reason': variant},
                                {'key': 'trustgtr', 'old': 'modules disabled', 'new': options,
                                 'reason': 'unchanged round-3 method; test shorter-than-cycle Solar applicability'},
                                {'key': 'gate_loss_weight', 'old': 0, 'new': 0.05, 'reason': 'train-only utility calibration'}] +
                               ([{'key': 'periods', 'old': [144], 'new': [144,1008,4320], 'reason': 'same base/7x/30x rule'},
                                 {'key': 'contrast_loss_weight', 'old': 0, 'new': 0.002, 'reason': 'unchanged round-3 contrast weight'}]
                                if variant == 'abc3' else []) +
                               ([{'key': 'train_epochs', 'old': 30, 'new': 1, 'reason': 'independent smoke only'}] if smoke else []))
            config_path = ROOT / 'configs' / (run_id + '.json')
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
            run_dir = ROOT / 'runs' / run_id
            queue['jobs'].append(dict(id=run_id, argv=[str(ROOT / '.conda/bin/python'), '-u',
                str(ROOT / 'scripts/run_trust_experiment.py'), '--config', str(config_path), '--run-dir', str(run_dir)],
                cwd=str(ROOT), run_dir=str(run_dir), required_memory_mib=None, is_smoke=smoke,
                smoke_passed=False, depends_on=[source['id']] if smoke else [smoke_id],
                **({'memory_estimate_from': smoke_id} if not smoke else {})))
            print(run_id)
    output.write_text(json.dumps(queue, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
