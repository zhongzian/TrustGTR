#!/usr/bin/env python3
"""CPU best-checkpoint diagnostics on fixed validation and training windows.

No test dataset is constructed and no optimizer step or checkpoint write occurs.
Forced gates are posthoc counterfactuals within one jointly trained checkpoint,
not independently trained ablations. Validation-label oracles are upper bounds.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
from torch.nn import functional as F
from data_provider.data_loader import Dataset_ETT_hour, Dataset_Solar
from models.GTR import Model as OriginalModel
from models.TrustGTR import Model, state_features, utility_oracle
from diagnose_historical_retrieval import digest, file_hash, scores


def correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    usable = np.isfinite(a) & np.isfinite(b)
    a, b = a[usable], b[usable]
    if len(a) < 2 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def distribution(a):
    values = np.asarray(a).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                quantiles={str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)})


def strata(feature, utility, gate, oracle):
    order = np.argsort(np.asarray(feature).ravel(), kind='stable')
    rows = []
    for group, indices in enumerate(np.array_split(order, 5)):
        rows.append(dict(group=group, count=len(indices),
            feature_mean=float(np.asarray(feature).ravel()[indices].mean()),
            fixed_g1_mse_gain=float(np.asarray(utility).ravel()[indices].mean()),
            gate_mean=float(np.asarray(gate).ravel()[indices].mean()),
            oracle_mean=float(np.asarray(oracle).ravel()[indices].mean())))
    return rows


def verified_datasets(run, args):
    assets = json.loads((run / 'evaluation_assets.json').read_text())
    cls = Dataset_ETT_hour if args.data.startswith('ETT') else Dataset_Solar
    kwargs = dict(root_path=args.root_path, data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len], features=args.features,
        target=args.target, timeenc=int(args.embed == 'timeF'), freq=args.freq, cycle=args.cycle)
    datasets, verified = {}, {}
    for flag in ('train', 'val'):
        data = cls(flag=flag, **kwargs)
        metadata = json.loads(Path(assets[flag]['path']).read_text())
        checks = dict(data_x=digest(data.data_x) == metadata['data_x_sha256'],
            data_y=digest(data.data_y) == metadata['data_y_sha256'],
            phase=digest(data.cycle_index) == metadata['cycle_index_sha256'],
            mean=np.array_equal(data.scaler.mean_, np.asarray(metadata['normalization']['mean_'])),
            scale=np.array_equal(data.scaler.scale_, np.asarray(metadata['normalization']['scale_'])))
        if not all(checks.values()):
            raise ValueError(f'Fixed asset mismatch {run.name}/{flag}: {checks}')
        datasets[flag] = data
        verified[flag] = dict(path=assets[flag]['path'], sha256=file_hash(assets[flag]['path']), checks=checks)
    return datasets, assets, verified


def tensors(data, indices, border):
    batch = [data[int(i)] for i in indices]
    x = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    target = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32)
    target = target[:, -data.pred_len:]
    origins = torch.tensor(border + indices + data.seq_len, dtype=torch.long)
    return x, target, origins


def decomposition(model, x, origin):
    """Reconstruct the historical inference path without exposing targets."""
    local = OriginalModel.forward(model, x, origin)
    raw = x.permute(0, 2, 1)
    retrieved, context = model.historical_memory(raw, origin)
    valid = context['valid'].view(-1, 1, 1)
    bound = model.options['historical_blend_bound']
    scale = bound if model.options['historical_strength_coordinates'] == 'bounded' else 1.
    strength = bound * (model.historical_strength / scale).tanh()
    delta = strength * (retrieved - local) * valid
    if model.options['use_reliability_gate']:
        left, right = local.permute(0, 2, 1), retrieved.permute(0, 2, 1)
        agreement = F.cosine_similarity(left, right, dim=-1).unsqueeze(-1)
        rms = left.square().mean(-1, keepdim=True).sqrt().clamp_min(.1)
        norm_ratio = torch.log1p((right - left).square().mean(-1, keepdim=True).sqrt() / rms)
        mismatch = (right - left).abs().mean(-1, keepdim=True) / rms
        descriptor = torch.cat((state_features(raw), agreement, norm_ratio, mismatch), -1).clamp(-10, 10)
        gate = model.gate(descriptor).sigmoid().permute(0, 2, 1) * valid
    else:
        gate = valid.expand(-1, 1, x.shape[-1]).float()
    distance = (context['base_candidate_key_embeddings'] - context['query_embedding'].unsqueeze(2)).square().sum(-1)
    distance = distance.masked_fill(~context['base_candidate_mask'].unsqueeze(1), float('inf'))
    if model.options['historical_router_mode'] != 'global_topk':
        raise ValueError('Neighbor-dispersion diagnostic currently supports the frozen v5 global_topk path only')
    nearest = distance.topk(min(model.options['historical_topk'], distance.shape[-1]), dim=-1, largest=False)
    uncertainty = nearest.values.mean(-1)
    memory = model.historical_memory
    selected = context['base_candidate_index'].clamp_min(0).unsqueeze(1).expand(-1, model.enc_in, -1).gather(-1, nearest.indices)
    channel = torch.arange(model.enc_in).view(1, model.enc_in, 1)
    times = memory.origins[selected].unsqueeze(-1) + torch.arange(model.pred_len)
    future = memory.series[times, channel.unsqueeze(-1)]
    transferred = ((future - memory.history_mean[selected, channel].unsqueeze(-1)) /
                   memory.history_scale[selected, channel].unsqueeze(-1))
    transferred = transferred * context['query_scale'][:, :, None, None] + context['query_mean'][:, :, None, None]
    dispersion = ((transferred - retrieved.transpose(1, 2).unsqueeze(2)).square() *
                  context['selected_weights'].unsqueeze(-1)).sum(2).mean(-1).sqrt()
    return local, retrieved, delta, gate, context, uncertainty, strength, dispersion


def train_gradient_diagnostic(model, dataset, border, seed, count):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), min(count, len(dataset)), replace=False)
    x, target, origins = tensors(dataset, indices, border)
    torch.manual_seed(seed)
    model.train()
    prediction = model(x, origins)
    main_loss = F.mse_loss(prediction, target)
    parameters = list(model.historical_memory.key_encoder.parameters())
    names = [name for name, _ in model.historical_memory.key_encoder.named_parameters()]
    def gradient(loss, selected_parameters=parameters):
        values = torch.autograd.grad(loss, selected_parameters, retain_graph=True, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if value is None else value).ravel()
                          for p, value in zip(selected_parameters, values)]).detach()
    main_gradient = gradient(main_loss)
    old_metric = model.options['contrast_metric']
    old_temperature = model.options['contrast_temperature']
    results = {}
    for metric in ('cosine', 'euclidean'):
        model.options['contrast_metric'] = metric
        before_valid = model.diagnostics['contrast_valid_anchors']
        before_total = model.diagnostics['contrast_total_anchors']
        contrast = model.historical_contrastive_loss(target, model._retrieval_context)
        other = gradient(contrast)
        weight = model.options['contrast_loss_weight']
        cosine = float(F.cosine_similarity(main_gradient[None], other[None]))
        results[metric] = dict(loss=float(contrast.detach()), gradient_norm=float(other.norm()),
            weighted_gradient_norm=float(other.norm()) * weight,
            weighted_to_main_ratio=float(other.norm()) * weight / max(float(main_gradient.norm()), 1e-12),
            gradient_cosine_with_main=cosine,
            usable_anchors=model.diagnostics['contrast_valid_anchors'] - before_valid,
            total_anchors=model.diagnostics['contrast_total_anchors'] - before_total)
    temperature_grid = {}
    model.options['contrast_metric'] = 'euclidean'
    for temperature in (.3, 1., 3., 10.):
        model.options['contrast_temperature'] = temperature
        contrast = model.historical_contrastive_loss(target, model._retrieval_context)
        other = gradient(contrast)
        weight = model.options['contrast_loss_weight']
        temperature_grid[str(temperature)] = dict(loss=float(contrast.detach()),
            gradient_norm=float(other.norm()), weighted_gradient_norm=float(other.norm()) * weight,
            weighted_to_main_ratio=float(other.norm()) * weight / max(float(main_gradient.norm()), 1e-12),
            gradient_cosine_with_main=float(F.cosine_similarity(main_gradient[None], other[None])))
    model.options['contrast_metric'] = old_metric
    model.options['contrast_temperature'] = old_temperature
    gate_gradients = None
    if model.options['use_reliability_gate']:
        gate_parameters = list(model.gate.parameters())
        gate_main = gradient(main_loss, gate_parameters)
        calibration_loss = model.auxiliary_losses(target)['gate']
        gate_calibration = gradient(calibration_loss, gate_parameters)
        weight = model.options['gate_loss_weight']
        gate_gradients = dict(main_gradient_norm=float(gate_main.norm()),
            calibration_loss=float(calibration_loss.detach()), gate_loss_weight=weight,
            weighted_aux_gradient_norm=float(gate_calibration.norm()) * weight,
            weighted_aux_to_main_ratio=float(gate_calibration.norm()) * weight / max(float(gate_main.norm()), 1e-12),
            main_aux_gradient_cosine=float(F.cosine_similarity(gate_main[None], gate_calibration[None])))
    cache = model._retrieval_context
    query = cache['state']['query_embedding'].detach()
    candidates = cache['state']['base_candidate_key_embeddings'].detach()
    mask = cache['state']['base_candidate_mask'].unsqueeze(1)
    euclidean = (query.unsqueeze(2) - candidates).square().sum(-1).masked_fill(~mask, float('inf'))
    cosine = (1 - (F.normalize(query, dim=-1).unsqueeze(2) * F.normalize(candidates, dim=-1)).sum(-1)).masked_fill(~mask, float('inf'))
    first_e, first_c = euclidean.argmin(-1), cosine.argmin(-1)
    valid_query = mask.any(-1).expand_as(first_e)
    k = min(model.options['historical_topk'], euclidean.shape[-1])
    top_e, top_c = euclidean.topk(k, dim=-1, largest=False).indices, cosine.topk(k, dim=-1, largest=False).indices
    enough_candidates = (mask.sum(-1) >= k).expand_as(first_e)
    overlap = (top_e.unsqueeze(-1) == top_c.unsqueeze(-2)).any(-1).float().mean(-1)[enough_candidates].mean()
    training_oracle = training_gate = None
    if model.options['use_reliability_gate']:
        training_oracle, _ = utility_oracle(cache['local'], cache['delta'], target)
        training_gate = cache['gate']
    model.eval()
    with torch.inference_mode():
        local, _, delta, eval_gate, _, _, _, _ = decomposition(model, x, origins)
        eval_oracle, _ = utility_oracle(local, delta, target)
    return dict(scope='fixed training batch only; no optimizer step', indices=indices.tolist(),
        seed=seed, count=len(indices), main_mse=float(main_loss.detach()),
        key_encoder_parameters=names, main_gradient_norm=float(main_gradient.norm()),
        contrast_weight=model.options['contrast_loss_weight'], contrast_metrics=results,
        contrast_original_temperature=old_temperature, euclidean_temperature_grid=temperature_grid,
        nearest_candidate_agreement=float((first_e == first_c)[valid_query].float().mean()),
        topk_candidate_overlap=float(overlap),
        gate_auxiliary_gradient=gate_gradients,
        gate_calibration_train_mode=(dict(oracle=distribution(training_oracle.detach().numpy()),
                                         gate=distribution(training_gate.detach().numpy()))
                                     if training_oracle is not None else None),
        gate_calibration_eval_mode_same_train_windows=dict(oracle=distribution(eval_oracle.numpy()),
                                                          gate=distribution(eval_gate.numpy())),
        query_norm=distribution(query.norm(dim=-1).numpy()),
        eligible_candidate_norm=distribution(candidates.norm(dim=-1)[mask.expand_as(euclidean)].numpy()))


def diagnose(run, sample_count, batch_size, train_count):
    started = time.monotonic()
    config = json.loads((run / 'resolved_config.yaml').read_text())
    args = SimpleNamespace(**config['official_args'])
    if config['trustgtr']['memory_mode'] != 'historical':
        raise ValueError('Only historical v5 models are supported')
    datasets, assets, verified = verified_datasets(run, args)
    checkpoint = run / 'checkpoints/best.pt'
    model = Model(args).cpu().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    model.historical_memory.attach_training_series(datasets['train'].data_x)
    stored_memory = json.loads((run / 'historical_memory.json').read_text())
    if stored_memory['sha256'] != model.historical_memory.metadata['sha256']:
        raise ValueError('Training memory hash differs from experiment')
    count = len(datasets['val'])
    indices = np.arange(count) if sample_count == 0 else np.unique(np.linspace(0, count - 1, min(sample_count, count)).astype(int))
    collected = {name: [] for name in ('local', 'retrieved', 'delta', 'gate', 'oracle', 'target', 'distance', 'priors', 'effective_bank', 'dispersion')}
    reconstruction_error = 0.
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            x, target, origins = tensors(datasets['val'], indices[offset:offset + batch_size], assets['val']['split'][0])
            local, retrieved, delta, gate, context, distance, strength, dispersion = decomposition(model, x, origins)
            actual = model(x, origins)
            reconstruction_error = max(reconstruction_error, float((actual - (local + gate * delta)).abs().max()))
            oracle, _ = utility_oracle(local, delta, target)
            banks = context['selected_bank_index']
            weights = context['selected_weights']
            effective_bank = torch.stack([(weights * (banks == bank)).sum(-1) for bank in range(len(model.options['periods']))], -1)
            values = dict(local=local, retrieved=retrieved, delta=delta, gate=gate, oracle=oracle,
                          target=target, distance=distance, priors=context['cycle_weights'], effective_bank=effective_bank,
                          dispersion=dispersion)
            for name, value in values.items():
                collected[name].append(value.numpy())
    data = {name: np.concatenate(values) for name, values in collected.items()}
    local, delta, gate, oracle, target = [data[name] for name in ('local', 'delta', 'gate', 'oracle', 'target')]
    if reconstruction_error > 1e-6:
        raise ValueError(f'Decomposition failed to reproduce actual model: {reconstruction_error}')
    actual = local + gate * delta
    full = local + delta
    gain = np.square(local - target).mean(1) - np.square(full - target).mean(1)
    actual_gain = np.square(local - target).mean(1) - np.square(actual - target).mean(1)
    power = np.square(delta).sum(1)
    flat_gate, flat_oracle = gate[:, 0], oracle[:, 0]
    weights = np.clip(power / max(float(power.mean()), 1e-8), 0, 10) * (power > 1e-6)
    channels = []
    for channel in range(args.enc_in):
        channels.append(dict(channel=channel, local=scores(local[..., channel], target[..., channel]),
            dynamic=scores(actual[..., channel], target[..., channel]), fixed_g1=scores(full[..., channel], target[..., channel]),
            gate_mean=float(flat_gate[:, channel].mean()), oracle_mean=float(flat_oracle[:, channel].mean()),
            effective_correction_mean=float(strength) * float(flat_gate[:, channel].mean()),
            beneficial_g1_fraction=float((gain[:, channel] > 0).mean()),
            gate_oracle_correlation=correlation(flat_gate[:, channel], flat_oracle[:, channel]),
            gate_gain_correlation=correlation(flat_gate[:, channel], gain[:, channel]),
            distance_gain_correlation=correlation(data['distance'][:, channel], gain[:, channel])))
    result = dict(experiment_id=run.name, dataset=config['experiment']['dataset'],
        checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
        evaluated_validation_windows=len(indices), total_validation_windows=count,
        validation_indices=indices.tolist(), fixed_assets=verified,
        training_memory_sha256=model.historical_memory.metadata['sha256'],
        decomposition_max_absolute_error=reconstruction_error, strength=float(strength),
        scores=dict(actual_dynamic=scores(actual, target), fixed_g1=scores(full, target),
                    fixed_g0_jointly_trained_GTR=scores(local, target),
                    constant_gate_at_dynamic_mean=scores(local + float(gate.mean()) * delta, target),
                    retrieval_alone=scores(data['retrieved'], target),
                    target_aware_oracle_upper_bound=scores(local + oracle * delta, target)),
        gate=distribution(gate), oracle=distribution(oracle), effective_coefficient=distribution(float(strength) * gate),
        oracle_zero_fraction=float((oracle == 0).mean()), oracle_one_fraction=float((oracle == 1).mean()),
        oracle_interior_fraction=float(((oracle > 0) & (oracle < 1)).mean()),
        fixed_g1_mse_gain=distribution(gain), actual_mse_gain=distribution(actual_gain),
        gate_oracle_correlation=correlation(flat_gate, flat_oracle), gate_gain_correlation=correlation(flat_gate, gain),
        distance_gain_correlation=correlation(data['distance'], gain),
        distance_oracle_correlation=correlation(data['distance'], flat_oracle),
        neighbor_dispersion=distribution(data['dispersion']),
        neighbor_dispersion_oracle_correlation=correlation(data['dispersion'], flat_oracle),
        neighbor_dispersion_gain_correlation=correlation(data['dispersion'], gain),
        retrieval_disagreement_oracle_correlation=correlation(np.sqrt(np.square(data['retrieved'] - local).mean(1)), flat_oracle),
        beneficial_g1_fraction=float((gain > 0).mean()), harmful_g1_fraction=float((gain < 0).mean()),
        harmful_g1_gate_mean=float(flat_gate[gain < 0].mean()), helpful_g1_gate_mean=float(flat_gate[gain > 0].mean()),
        gate_power_weighted_calibration_mse=float((weights * (flat_gate - flat_oracle) ** 2).sum() / max(float(weights.sum()), 1e-8)),
        gate_quintiles=strata(flat_gate, gain, flat_gate, flat_oracle),
        candidate_distance_quintiles=strata(data['distance'], gain, flat_gate, flat_oracle),
        neighbor_dispersion_quintiles=strata(data['dispersion'], gain, flat_gate, flat_oracle),
        bank_prior_mean=data['priors'].mean((0, 1)).tolist(),
        actual_selected_bank_weight_mean=data['effective_bank'].mean((0, 1)).tolist(),
        channels=channels, interpretation='All fixed gates share the same checkpoint; this is not a retrained ablation. Oracle uses validation labels only for an upper-bound diagnostic.')
    if model.options['use_regime_contrast']:
        result['training_gradient_diagnostic'] = train_gradient_diagnostic(model, datasets['train'], assets['train']['split'][0], args.random_seed, train_count)
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--sample-count', type=int, default=256, help='0: full fixed validation set')
    parser.add_argument('--batch-size', type=int, default=16, help='Diagnostic CPU batching only')
    parser.add_argument('--train-count', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(f'Refuse to overwrite {output}')
    report = dict(scope='best checkpoint only; deterministic validation windows',
        no_test_dataset_constructed=True, no_test_targets_used=True, no_optimizer_steps=True,
        counterfactual_gate_is_not_retrained_ablation=True,
        oracle_is_target_aware_validation_upper_bound=True,
        cpu_threads=args.threads, script_sha256=file_hash(Path(__file__)), experiments=[])
    for name in args.runs:
        run = Path(name)
        if not run.is_absolute():
            run = ROOT / 'runs' / run
        result = diagnose(run, args.sample_count, args.batch_size, args.train_count)
        report['experiments'].append(result)
        print(json.dumps(dict(experiment_id=result['experiment_id'], scores=result['scores'],
            gate_mean=result['gate']['mean'], strength=result['strength'], elapsed_seconds=result['elapsed_seconds'])), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
