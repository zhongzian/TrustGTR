#!/usr/bin/env python3
"""Archived B7 training-window gradients; CPU only, no optimizer or test data."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from torch.nn import functional as F
from diagnose_gate_v5 import verified_datasets, tensors, file_hash, correlation, distribution


def archived_model(run):
    paths = [run / 'untracked_code/models' / name for name in ('HistoricalMemory.py', 'TrustGTR.py')]
    history_spec = importlib.util.spec_from_file_location('models.HistoricalMemory', paths[0])
    history = importlib.util.module_from_spec(history_spec)
    previous = sys.modules.get('models.HistoricalMemory')
    sys.modules['models.HistoricalMemory'] = history
    try:
        history_spec.loader.exec_module(history)
        name = 'archived_B7_' + file_hash(paths[1])[:16]
        spec = importlib.util.spec_from_file_location(name, paths[1])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop('models.HistoricalMemory', None)
        else:
            sys.modules['models.HistoricalMemory'] = previous
    return module, {str(p.relative_to(run)): file_hash(p) for p in paths}


def gradient(loss, parameters):
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for p, g in zip(parameters, values)]).detach()


def diagnose(run):
    config = json.loads((run / 'resolved_config.yaml').read_text())
    args = SimpleNamespace(**config['official_args'])
    # This helper constructs train and validation only and checks their saved assets.
    datasets, assets, verified = verified_datasets(run, args)
    module, hashes = archived_model(run)
    torch.manual_seed(args.random_seed)
    model = module.Model(args).cpu()
    checkpoint = run / 'checkpoints/best.pt'
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    model.historical_memory.attach_training_series(datasets['train'].data_x)
    memory = json.loads((run / 'historical_memory.json').read_text())
    assert memory['sha256'] == model.historical_memory.metadata['sha256']
    rng = np.random.default_rng(args.random_seed)
    indices = rng.choice(len(datasets['train']), 64, replace=False)
    x, y, origin = tensors(datasets['train'], indices, assets['train']['split'][0])
    params = list(model.gate.parameters())
    model.train()
    torch.manual_seed(args.random_seed)
    pred = model(x, origin)
    train_cache = model._retrieval_context
    main = F.mse_loss(pred, y)
    aux = model.auxiliary_losses(y)['gate']
    main_grad, aux_grad = gradient(main, params), gradient(aux, params)
    weighted = config['trustgtr']['gate_loss_weight'] * aux_grad
    groups = {
        'gate': list(model.gate.parameters()),
        'strength': [model.historical_strength],
        'retrieval_encoder': list(model.historical_memory.key_encoder.parameters()),
        'backbone_head': list(model.output_proj.parameters()),
    }
    aux_norms = {name: float(gradient(aux, selected).norm()) for name, selected in groups.items()}
    # No parameter update; eval-mode experts only. Train wrapper caches the same
    # experts when dropout modules are off; exact equality to eval output is verified.
    model.eval()
    with torch.no_grad():
        eval_pred = model(x, origin)
    model.train()
    dropouts = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.Dropout)]
    for _, dropout in dropouts:
        dropout.eval()
    inferred = model(x, origin)
    eval_cache = model._retrieval_context
    eval_aux = model.auxiliary_losses(y)['gate']
    eval_aux_grad = gradient(eval_aux, params)
    err = float((inferred.detach() - eval_pred).abs().max())
    assert err == 0, ('Dropout-off cache differs from inference', err)

    def characterize(cache):
        p = cache['delta'].square().sum(1, keepdim=True)
        u = (cache['delta'] * (y - cache['local'])).sum(1, keepdim=True)
        e = (y - cache['local']).square().sum(1, keepdim=True)
        valid = p > 1e-6
        optimum, _ = module.utility_oracle(cache['local'], cache['delta'], y)
        scalar_derivative = 2 * (cache['gate'] * p - u) / (p + e).clamp_min(1e-8)
        if not valid.any():
            raise ValueError('No meaningful corrections in training sample')
        def array(v):
            return v.detach()[valid].cpu().numpy()
        return dict(valid_sample_channels=int(valid.sum()),
            local_mse=float(F.mse_loss(cache['local'], y)),
            fixed_g1_mse=float(F.mse_loss(cache['local'] + cache['delta'], y)),
            gate=distribution(array(cache['gate'])), oracle=distribution(array(optimum)),
            oracle_zero_fraction=float((optimum[valid] == 0).float().mean()),
            oracle_one_fraction=float((optimum[valid] == 1).float().mean()),
            scalar_open_push_fraction=float((scalar_derivative[valid] < 0).float().mean()),
            scalar_close_push_fraction=float((scalar_derivative[valid] > 0).float().mean()),
            mean_excess_risk=float(((cache['gate'].square() * p - 2 * cache['gate'] * u) /
                                   (p + e).clamp_min(1e-8))[valid].mean()))

    train_oracle, train_valid = module.utility_oracle(train_cache['local'], train_cache['delta'], y)
    infer_oracle, infer_valid = module.utility_oracle(eval_cache['local'], eval_cache['delta'], y)
    valid = train_valid & infer_valid
    return dict(experiment_id=run.name, scope='fixed train64; archived best checkpoint; no optimizer step',
        indices=indices.tolist(), seed=args.random_seed, checkpoint_sha256=file_hash(checkpoint),
        archived_sources=hashes, fixed_assets=verified,
        dropout_probabilities={name: dropout.p for name, dropout in dropouts},
        main_loss=float(main.detach()), aux_loss=float(aux.detach()),
        gate_main_gradient_norm=float(main_grad.norm()), gate_weighted_aux_gradient_norm=float(weighted.norm()),
        weighted_aux_to_main_ratio=float(weighted.norm() / main_grad.norm().clamp_min(1e-12)),
        main_aux_gradient_cosine=float(F.cosine_similarity(main_grad[None], aux_grad[None])),
        aux_gradient_norm_by_module=aux_norms,
        inference_aux_loss=float(eval_aux.detach()),
        inference_aux_gradient_norm=float(eval_aux_grad.norm()),
        train_aux_inference_aux_gradient_cosine=float(F.cosine_similarity(aux_grad[None], eval_aux_grad[None])),
        main_inference_aux_gradient_cosine=float(F.cosine_similarity(main_grad[None], eval_aux_grad[None])),
        inference_cache_matches_eval_max_abs_error=err,
        train_oracle_vs_inference_oracle_correlation=correlation(train_oracle.detach()[valid].numpy(), infer_oracle.detach()[valid].numpy()),
        oracle_zero_status_changed_fraction=float(((train_oracle == 0) != (infer_oracle == 0))[valid].float().mean()),
        oracle_one_status_changed_fraction=float(((train_oracle == 1) != (infer_oracle == 1))[valid].float().mean()),
        training_experts=characterize(train_cache), inference_experts_same_train_windows=characterize(eval_cache))


def main():
    torch.set_num_threads(2)
    names = ['trustgtr_ab7__etth1__forecast__t96_h96__seed2024__20261005_162802',
             'trustgtr_ab7__solar__forecast__t96_h96__seed2024__20261005_162802']
    output = ROOT / 'reports/b8_calibration_gradient_audit.json'
    if output.exists():
        raise FileExistsError(output)
    # Tiny tensors verify that risk does not force every gate to open.
    risk_module, _ = archived_model(ROOT / 'runs' / names[0])
    gate = torch.tensor([[[.9]], [[.9]]], requires_grad=True)
    loss = risk_module.counterfactual_risk_loss(torch.zeros(2, 1, 1), torch.ones(2, 1, 1), gate,
                                               torch.tensor([[[2.]], [[-1.]]]))
    signs = torch.autograd.grad(loss, gate)[0].flatten().tolist()
    assert signs[0] < 0 < signs[1]
    report = dict(cpu_threads=2, no_test_dataset=True, no_optimizer_steps=True, no_checkpoint_writes=True,
                  script_sha256=file_hash(Path(__file__)), scalar_risk_gradients_helpful_harmful=signs,
                  experiments=[])
    for name in names:
        row = diagnose(ROOT / 'runs' / name)
        report['experiments'].append(row)
        print(json.dumps({k: row[k] for k in ('experiment_id', 'weighted_aux_to_main_ratio',
                          'main_aux_gradient_cosine', 'train_aux_inference_aux_gradient_cosine',
                          'oracle_zero_status_changed_fraction')}), flush=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Saved ' + str(output), flush=True)


if __name__ == '__main__':
    main()
