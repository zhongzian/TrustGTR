#!/usr/bin/env python3
"""6.5 CPU diagnostics: fixed train64 gradients and archived full validation.

Training-label regimes are inspected only in the train diagnostic. No optimizer,
test dataset, parameter updates, or checkpoint writes. The gradient probe uses
eight-window chunks, not the official training batch, and cannot select weights.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from torch.nn import functional as F
import diagnose_step64_gate as archive


def fixed_dataset(run, config, loader, flag):
    args = SimpleNamespace(**config['official_args'])
    assets = json.loads((run / 'evaluation_assets.json').read_text())
    source = assets[flag]
    meta_path = Path(source['path'])
    if archive.file_hash(meta_path) != source['sha256']:
        raise ValueError('Asset manifest hash mismatch: ' + flag)
    metadata = json.loads(meta_path.read_text())
    if archive.file_hash(Path(args.root_path) / args.data_path) != config['data_asset']['sha256']:
        raise ValueError('Raw data hash mismatch')
    classes = dict(ETTh1='Dataset_ETT_hour', ETTh2='Dataset_ETT_hour',
                   ETTm1='Dataset_ETT_minute', ETTm2='Dataset_ETT_minute',
                   Solar='Dataset_Solar', custom='Dataset_Custom')
    data = getattr(loader, classes[args.data])(root_path=args.root_path,
        data_path=args.data_path, flag=flag, size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features, target=args.target, timeenc=int(args.embed == 'timeF'),
        freq=args.freq, cycle=args.cycle)
    checks = dict(data_x=archive.array_hash(data.data_x) == metadata['data_x_sha256'],
        data_y=archive.array_hash(data.data_y) == metadata['data_y_sha256'],
        cycle_index=archive.array_hash(data.cycle_index) == metadata['cycle_index_sha256'],
        raw_data=metadata['data_sha256'] == config['data_asset']['sha256'],
        window_count=len(data) == metadata['window_count'],
        split=source['split'] == metadata['absolute_data_slice'])
    for key in ('mean_', 'var_', 'scale_'):
        checks[key] = np.array_equal(getattr(data.scaler, key), np.asarray(metadata['normalization'][key]))
    if not all(checks.values()):
        raise ValueError('Fixed dataset verification failed: ' + str(checks))
    return data, source['split'][0], dict(path=str(meta_path), sha256=source['sha256'], checks=checks)


def make_model(trust, config, checkpoint, train, memory_record, sampling=None):
    args = SimpleNamespace(**config['official_args'])
    args.trustgtr = dict(config['trustgtr'])
    if sampling is not None:
        args.trustgtr['contrast_sampling'] = sampling
    torch.manual_seed(args.random_seed)
    model = trust.Model(args).cpu()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    model.historical_memory.attach_training_series(train.data_x)
    if model.historical_memory.metadata['sha256'] != memory_record['sha256']:
        raise ValueError('Historical memory differs from saved train-only series')
    return model


def inspect_selection(model, target, cache):
    """Independent negative reconstruction; no training diagnostics mutation."""
    state = cache['state']
    index = state['base_candidate_index']
    origin = cache['origin']
    mask = state['base_candidate_mask'] & (origin[:, None] -
        state['base_candidate_origin'] >= model.seq_len + model.pred_len)
    future = target.detach().permute(0, 2, 1)
    q = max(1, future.shape[-1] // 4)
    regime = torch.stack(((future.mean(-1) - cache['history_mean']) / cache['history_std'],
        (future[..., -q:].mean(-1) - future[..., :q].mean(-1)) / cache['history_std']), -1).clamp(-5, 5)
    bank = state['regime'][index.clamp_min(0)].permute(0, 2, 1, 3).clamp(-5, 5)
    distance = (bank - regime.unsqueeze(2)).square().sum(-1).sqrt()
    near, positive = distance.masked_fill(~mask.unsqueeze(1), float('inf')).min(-1)
    key_distance = (state['query_embedding'].unsqueeze(2) -
                    state['base_candidate_key_embeddings']).square().sum(-1).detach()
    # The old far model intentionally does not collect selection. Independently
    # reconstruct its exact unchanged retrieval rule for coverage comparison.
    width = index.shape[-1]
    valid = state['base_candidate_mask'].unsqueeze(1)
    routing_distance = key_distance.masked_fill(~valid, float('inf'))
    k = min(model.options['historical_topk'], width)
    positions, weights = [], []
    if model.options['historical_router_mode'] != 'per_bank':
        raise ValueError('6.5 diagnostic requires frozen A6 per_bank router')
    for number in range(len(model.options['periods'])):
        ds = routing_distance.masked_fill(~(state['candidate_bank_index'] == number).unsqueeze(1), float('inf'))
        ds, pos = ds.topk(k, dim=-1, largest=False)
        within = 1 / (ds + 1e-3)
        within = within / within.sum(-1, keepdim=True).clamp_min(1e-12)
        positions.append(pos.masked_fill(~torch.isfinite(ds), -1))
        weights.append(within * state['cycle_weights'][..., number:number + 1])
    positions, weights = torch.cat(positions, -1), torch.cat(weights, -1).detach()
    reconstruction_errors = {}
    if 'selected_candidate_positions' in state:
        if not torch.equal(positions, state['selected_candidate_positions']):
            raise ValueError('Actual collected retrieval positions disagree with independent topk')
        reconstruction_errors['weights_max_error'] = float((weights - state['selected_candidate_weights']).abs().max())
        if reconstruction_errors['weights_max_error'] != 0:
            raise ValueError('Actual collected weights disagree with independent retrieval')
    selected = torch.zeros_like(key_distance, dtype=torch.bool)
    for slot in range(positions.shape[-1]):
        active = (positions[..., slot] >= 0) & (weights[..., slot] > 0)
        entry = torch.zeros_like(selected)
        entry.scatter_(-1, positions[..., slot:slot + 1].clamp_min(0), active.unsqueeze(-1))
        selected |= entry
    slots = torch.arange(width).view(1, 1, -1)
    eligible = (selected & mask.unsqueeze(1) & (slots != positive.unsqueeze(-1)) &
        (distance >= near.unsqueeze(-1) + model.options['regime_distance_gap']) & torch.isfinite(key_distance))
    if model.options['contrast_sampling'] == 'retrieval_hard':
        safe_slots = positions.clamp_min(0)
        slot_eligible = eligible.gather(-1, safe_slots) & (positions >= 0) & (weights > 0)
        slot_distances = key_distance.gather(-1, safe_slots).masked_fill(~slot_eligible, float('inf'))
        _, chosen_slot = slot_distances.min(-1)
        negative = safe_slots.gather(-1, chosen_slot.unsqueeze(-1)).squeeze(-1)
        far = distance.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
        far = far.masked_fill(~eligible.any(-1), -float('inf'))
        actual_neg, actual_far = model.retrieval_hard_negative(distance, mask, near, positive, state)
        if not torch.equal(negative, actual_neg) or not torch.equal(far, actual_far):
            raise ValueError('Actual hard-negative helper disagrees with independent reference')
    else:
        far, negative = distance.masked_fill(~mask.unsqueeze(1), -float('inf')).max(-1)
    usable = ((mask.sum(-1) >= 2).unsqueeze(1) &
        (near <= model.options['positive_max_distance']) &
        (far - near >= model.options['regime_distance_gap']))
    pos_key = key_distance.gather(-1, positive.unsqueeze(-1)).squeeze(-1)
    neg_key = key_distance.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
    selected_negative = selected.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
    selected_positive = selected.gather(-1, positive.unsqueeze(-1)).squeeze(-1)
    phase_bad = causal_bad = overlap_bad = 0
    for chosen in (positive, negative):
        candidate_origin = state['base_candidate_origin'].unsqueeze(1).expand(-1, target.shape[-1], -1)
        chosen_origin = candidate_origin.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        phase_bad += int(((chosen_origin % model.cycle_len != origin[:, None] % model.cycle_len) & usable).sum())
        causal_bad += int(((chosen_origin + model.pred_len > origin[:, None]) & usable).sum())
        overlap_bad += int(((chosen_origin + model.pred_len > origin[:, None] - model.seq_len) & usable).sum())
    if phase_bad or causal_bad or overlap_bad:
        raise ValueError('Chosen contrastive window violates phase/causality/nonoverlap')
    if model.options['contrast_sampling'] == 'retrieval_hard' and not selected_negative[usable].all():
        raise ValueError('Hard negative has zero actual retrieval mass')
    if model.options['contrast_metric'] != 'euclidean':
        raise ValueError('6.5 probes require common euclidean metric')
    logits = torch.stack((-pos_key, -neg_key), -1) / model.options['contrast_temperature']
    expected_loss = F.cross_entropy(logits[usable], torch.zeros(int(usable.sum()), dtype=torch.long)) if usable.any() else logits.sum() * 0
    counts = dict(channel_windows=usable.numel(), usable=int(usable.sum()),
        all_candidate_slots=int(state['base_candidate_mask'].sum()) * target.shape[-1],
        disjoint_candidate_slots=int(mask.sum()) * target.shape[-1],
        selected_nonzero_slots=int(selected.sum()),
        selected_disjoint_slots=int((selected & mask.unsqueeze(1)).sum()),
        selected_eligible_negative_slots=int(eligible.sum()),
        anchors_with_selected_eligible_negative=int(eligible.any(-1).sum()),
        negative_in_actual_retrieval=int(selected_negative[usable].sum()),
        positive_in_actual_retrieval=int(selected_positive[usable].sum()),
        no_two_disjoint_candidates=int((mask.sum(-1) < 2).sum()) * target.shape[-1],
        phase_violations=phase_bad, causal_violations=causal_bad, overlap_violations=overlap_bad)
    values = dict(positive_regime_distance=near[usable].numpy(), negative_regime_distance=far[usable].numpy(),
        regime_gap=(far - near)[usable].numpy(), positive_key_squared_distance=pos_key[usable].numpy(),
        negative_key_squared_distance=neg_key[usable].numpy(),
        candidate_count=mask.sum(-1).numpy(), selected_count=selected.sum(-1).numpy(),
        eligible_negative_count=eligible.sum(-1).numpy())
    return counts, values, float(expected_loss), reconstruction_errors


def grad_summary(model, main, auxiliary):
    parameters = list(model.named_parameters())
    main_grad = torch.autograd.grad(main, [p for _, p in parameters], retain_graph=True, allow_unused=True)
    aux_grad = torch.autograd.grad(auxiliary, [p for _, p in parameters], allow_unused=True)
    def vector(gradients, encoder):
        return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
            for (n, p), g in zip(parameters, gradients) if n.startswith('historical_memory.key_encoder.') == encoder])
    main_key, aux_key = vector(main_grad, True), vector(aux_grad, True)
    other_aux = vector(aux_grad, False)
    a, b = float(main_key.norm()), float(aux_key.norm())
    if float(other_aux.abs().max()) != 0:
        raise ValueError('Contrastive auxiliary gradient reached a nonencoder parameter')
    return dict(main_encoder_norm=a, weighted_aux_encoder_norm=b,
        weighted_aux_to_main_encoder_norm=b / a if a > 0 else None,
        main_aux_encoder_cosine=float(F.cosine_similarity(main_key, aux_key, dim=0)) if a > 0 and b > 0 else None,
        weighted_aux_nonencoder_max_abs=float(other_aux.abs().max()),
        main_all_parameter_norm=float(torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                                                for (_, p), g in zip(parameters, main_grad)]).norm()))


def train_probe(trust, config, checkpoint, train, border, memory_record, sample_count, batch_size):
    indices = np.unique(np.linspace(0, len(train) - 1, min(sample_count, len(train))).astype(int))
    rows = []
    predictions = {}
    for sampling in ('regime_extremes', 'retrieval_hard'):
        model = make_model(trust, config, checkpoint, train, memory_record, sampling).train()
        if model.options['use_reliability_gate'] or not model.options['use_regime_contrast']:
            raise ValueError('6.5 training probe requires B off, C on')
        counts, collected, batches, prediction_parts = {}, {}, [], []
        for offset in range(0, len(indices), batch_size):
            x, target, origin = archive.tensors(train, indices[offset:offset + batch_size], border)
            torch.manual_seed(config['official_args']['random_seed'] + 10000 + offset)
            prediction = model(x, origin)
            cache = model._retrieval_context
            with torch.no_grad():
                current_counts, values, expected_loss, errors = inspect_selection(model, target, cache)
            auxiliary = model.auxiliary_losses(target)['contrast']
            if abs(float(auxiliary.detach()) - expected_loss) > 3e-6 * max(1, abs(expected_loss)):
                raise ValueError('Actual contrastive loss disagrees with independent chosen pairs')
            main = F.mse_loss(prediction, target)
            gradients = grad_summary(model, main, model.options['contrast_loss_weight'] * auxiliary)
            for key, value in current_counts.items():
                counts[key] = counts.get(key, 0) + value
            for key, value in values.items():
                collected.setdefault(key, []).append(value.ravel())
            batches.append(dict(indices=indices[offset:offset + batch_size].tolist(),
                counts=current_counts,
                main_mse=float(main.detach()), contrast_ce=float(auxiliary.detach()),
                weighted_contrast=float(model.options['contrast_loss_weight'] * auxiliary.detach()),
                expected_contrast_ce=expected_loss, gradients=gradients,
                actual_selection_reconstruction=errors))
            prediction_parts.append(prediction.detach().numpy())
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError('autograd.grad unexpectedly accumulated parameter grads')
        predictions[sampling] = np.concatenate(prediction_parts)
        rows.append(dict(sampling=sampling, counts=counts,
            usable_fraction=counts['usable'] / counts['channel_windows'],
            negative_actual_retrieval_fraction=counts['negative_in_actual_retrieval'] / counts['usable'] if counts['usable'] else None,
            values={key: archive.distribution(np.concatenate(parts)) for key, parts in collected.items()},
            gradient_distributions={key: archive.distribution(np.array([batch['gradients'][key] for batch in batches
                if batch['gradients'][key] is not None])) for key in batches[0]['gradients']},
            batches=batches))
        del model
    if not np.array_equal(predictions['regime_extremes'], predictions['retrieval_hard']):
        raise ValueError('Sampling collector changed prediction or dropout RNG before training')
    return dict(training_indices=indices.tolist(), windows=len(indices), batch_size=batch_size,
        same_checkpoint_paired_prediction_exact=True, samplers=rows,
        interpretation='Microbatch diagnostics at a fixed checkpoint; not official training losses/gradients. No optimizer steps.',
        future_targets_in_prediction_signature=False)


def validation_probe(trust, gtr, config, checkpoint, train, val, border, memory_record, batch_size, sample_count):
    model = make_model(trust, config, checkpoint, train, memory_record).eval()
    indices = np.arange(len(val)) if sample_count == 0 else np.unique(
        np.linspace(0, len(val) - 1, min(sample_count, len(val))).astype(int))
    if model.options['use_reliability_gate']:
        raise ValueError('6.5 validation probe is defined for B off only')
    errors = {name: archive.Error() for name in ('actual', 'same_checkpoint_gtr_branch', 'retrieval_alone')}
    values = {name: [] for name in ('nearest_key_squared_distance', 'selected_weight_entropy',
        'retrieval_mse', 'actual_gain_mse', 'retrieved_forecast_regime_distance')}
    bank_prior_sum = np.zeros(len(model.options['periods']))
    count = valid_count = 0
    max_error = 0.
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            x, target, origin = archive.tensors(val, indices[offset:offset + batch_size], border)
            actual = model(x, origin)
            local = gtr.Model.forward(model, x, origin)
            retrieved, state = model.historical_memory(x.permute(0, 2, 1), origin)
            bound = model.options['historical_blend_bound']
            scale = bound if model.options['historical_strength_coordinates'] == 'bounded' else 1.
            strength = bound * (model.historical_strength / scale).tanh()
            reconstructed = local + strength * ((retrieved - local) * state['valid'].view(-1, 1, 1))
            max_error = max(max_error, float((actual - reconstructed).abs().max()))
            if max_error != 0:
                raise ValueError('B-off inference reconstruction is not exact')
            if model._retrieval_context is not None or 'selected_candidate_positions' in state:
                raise ValueError('Training C selection collector ran during validation prediction')
            for name, prediction in dict(actual=actual, same_checkpoint_gtr_branch=local, retrieval_alone=retrieved).items():
                errors[name].add(prediction, target)
            ds = (state['query_embedding'].unsqueeze(2) - state['base_candidate_key_embeddings']).square().sum(-1)
            nearest = ds.masked_fill(~state['base_candidate_mask'].unsqueeze(1), float('inf')).min(-1).values
            weights = state['selected_weights']
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
            retrieval_mse = (retrieved - target).square().mean(1)
            gain = (local - target).square().mean(1) - (actual - target).square().mean(1)
            # Both regimes use the query history scale. This is a diagnostic
            # discrepancy, not a loss, positive/negative choice, or forward input.
            q = max(1, model.pred_len // 4)
            def regime(pred):
                pred = pred.permute(0, 2, 1)
                return torch.stack(((pred.mean(-1) - state['query_mean']) / state['query_scale'],
                    (pred[..., -q:].mean(-1) - pred[..., :q].mean(-1)) / state['query_scale']), -1).clamp(-5, 5)
            regime_error = (regime(retrieved) - regime(target)).square().sum(-1).sqrt()
            for key, array in dict(nearest_key_squared_distance=nearest, selected_weight_entropy=entropy,
                                  retrieval_mse=retrieval_mse, actual_gain_mse=gain,
                                  retrieved_forecast_regime_distance=regime_error).items():
                values[key].append(array.numpy().ravel())
            bank_prior_sum += state['cycle_weights'].sum((0, 1)).numpy()
            count += len(x) * x.shape[-1]
            valid_count += int(state['valid'].sum())
            if offset % (batch_size * 64) == 0:
                print(json.dumps(dict(validation_windows=min(offset + batch_size, len(indices)),
                    total=len(indices), elapsed_seconds=time.monotonic() - started)), flush=True)
    flattened = {key: np.concatenate(parts) for key, parts in values.items()}
    return dict(validation_indices=indices.tolist(), windows=len(indices), total_windows=len(val),
        scores={key: error.result() for key, error in errors.items()},
        values={key: archive.distribution(value[np.isfinite(value)]) for key, value in flattened.items()},
        nearest_distance_retrieval_mse_correlation=archive.correlation(flattened['nearest_key_squared_distance'], flattened['retrieval_mse']),
        strength=float(strength), raw_strength=float(model.historical_strength),
        mean_bank_prior=(bank_prior_sum / count).tolist(), valid_windows=valid_count,
        inference_reconstruction_max_error=max_error, auxiliary_loss_called=False,
        training_selection_collector_called=False,
        label_regime_discrepancy_is_post_prediction_diagnostic=True,
        aggregation='All fixed windows; float32 residuals promoted to float64 before squares/sums; differs from official shuffle/drop_last early-stopping validation.')


def diagnose(run, args):
    started = time.monotonic()
    if json.loads((run / 'status.json').read_text())['status'] != 'SUCCESS':
        raise ValueError('Only successful reference/formal experiments supported')
    config = json.loads((run / 'resolved_config.yaml').read_text())
    trust, gtr, loader, sources = archive.archived_sources(run)
    scientific_reference = 'archived formal checkpoint'
    if args.sources:
        from models import TrustGTR, GTR
        trust, gtr = TrustGTR, GTR
        config['trustgtr'].update(use_reliability_gate=False, use_regime_contrast=True,
            contrast_loss_weight=0.002, contrast_temperature=3.0, contrast_metric='euclidean',
            contrast_sampling='regime_extremes')
        config['official_args']['trustgtr'] = config['trustgtr']
        scientific_reference = 'existing A6 best weights evaluated with current 6.5 model; diagnostic only'
    for name in ('use_multi_cycle', 'use_regime_contrast'):
        if not config['trustgtr'][name]:
            raise ValueError('6.5 requires A and C on')
    if config['trustgtr']['use_reliability_gate']:
        raise ValueError('6.5 requires B off')
    if config['trustgtr']['contrast_metric'] != 'euclidean' or config['trustgtr']['contrast_temperature'] != 3.0 or config['trustgtr']['contrast_loss_weight'] != 0.002:
        raise ValueError('Diagnostic common C settings differ from frozen 6.5 values')
    train, train_border, train_asset = fixed_dataset(run, config, loader, 'train')
    memory_record = json.loads((run / 'historical_memory.json').read_text())
    checkpoint = run / 'checkpoints' / 'best.pt'
    row = dict(experiment_id=run.name, dataset=config['experiment']['dataset'], reference=scientific_reference,
        checkpoint=str(checkpoint), checkpoint_sha256=archive.file_hash(checkpoint),
        source_resolved_config=str(run / 'resolved_config.yaml'),
        source_resolved_config_sha256=archive.file_hash(run / 'resolved_config.yaml'),
        model_sources=sources, train_asset=train_asset, options=config['trustgtr'],
        historical_memory_sha256=memory_record['sha256'])
    if args.sources:
        row['current_diagnostic_model_sources'] = {name: archive.file_hash(ROOT / name)
            for name in ('models/GTR.py', 'models/TrustGTR.py', 'models/HistoricalMemory.py')}
    if args.scope in ('train', 'both'):
        row['train_probe'] = train_probe(trust, config, checkpoint, train, train_border, memory_record, args.train_count, args.batch_size)
    if args.scope in ('val', 'both'):
        val, border, val_asset = fixed_dataset(run, config, loader, 'val')
        row['validation_asset'] = val_asset
        row['validation_probe'] = validation_probe(trust, gtr, config, checkpoint, train, val,
            border, memory_record, args.batch_size, args.val_count)
    row.update(elapsed_seconds=time.monotonic() - started,
               process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return row


def markdown(report):
    lines = ['# 6.5 C实际检索困难负例诊断', '',
        'CPU 2 threads；没有test、optimizer step、参数更新或checkpoint写入。train梯度由8窗口微批获取，不能冒充官方训练batch；预测不接收future target。',
        '两个采样版本共同使用euclidean/T=3/λ=0.002。旧AC7使用cosine/T=0.1，因此相对旧AC7的变化不能仅归因采样。', '',
        '| dataset/reference | sampler | train windows | usable | negative实际检索覆盖 | aux/main encoder norm均值 | encoder gradient cosine均值 |',
        '|---|---|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'NA' if x is None else '{:.7g}'.format(x)
    for row in report['experiments']:
        if 'train_probe' in row:
            for sampler in row['train_probe']['samplers']:
                g = sampler['gradient_distributions']
                lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(row['dataset'], sampler['sampling'],
                    row['train_probe']['windows'], fmt(sampler['usable_fraction']),
                    fmt(sampler['negative_actual_retrieval_fraction']),
                    fmt(g['weighted_aux_to_main_encoder_norm']['mean']), fmt(g['main_aux_encoder_cosine']['mean'])))
        if 'validation_probe' in row:
            v = row['validation_probe']
            lines += ['', row['experiment_id'] + ': 固定val {}窗口，actual MSE/MAE {}/{}，同checkpoint GTR分支 {}/{}，retrieval {}/{}，strength={}，重建误差={}。'.format(
                v['windows'], fmt(v['scores']['actual']['mse']), fmt(v['scores']['actual']['mae']),
                fmt(v['scores']['same_checkpoint_gtr_branch']['mse']), fmt(v['scores']['same_checkpoint_gtr_branch']['mae']),
                fmt(v['scores']['retrieval_alone']['mse']), fmt(v['scores']['retrieval_alone']['mae']),
                fmt(v['strength']), fmt(v['inference_reconstruction_max_error']))]
    lines += ['', 'train-only静态库、raw/scaler/split/phase/数组/源代码/配置/checkpoint哈希、计数/距离/梯度明细见JSON。初始strength=0时主预测对key encoder的梯度为0；本预诊断使用A6已训练权重才能测有意义的梯度冲突。',
        'full-val分支/检索分解均为同一checkpoint事后诊断，不是分别训练的消融；val regime距离只在完成预测后计算，未用于选样、前向、重选checkpoint或调参。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sources', nargs='+', help='A6 best references, current model train probe only')
    group.add_argument('--runs', nargs='+', help='New successful archived 6.5 formal best runs')
    parser.add_argument('--scope', choices=('train', 'val', 'both'), default='train')
    parser.add_argument('--train-count', type=int, default=64)
    parser.add_argument('--val-count', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.sources and args.scope != 'train':
        raise ValueError('Preflight source references support train-only probes')
    if args.train_count < 1 or args.val_count < 0 or args.batch_size < 1:
        raise ValueError('Invalid diagnostic sample/batch size')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    output = ROOT / args.output
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('Refuse overwrite: ' + str(output))
    report = dict(scope='6.5 CPU fixed training64 / best-checkpoint fixed full validation',
        cpu_threads=2, no_test_dataset=True, no_optimizer=True, no_parameter_updates=True,
        no_checkpoint_writes=True, gradients_train_only=True,
        script_sha256=archive.file_hash(Path(__file__)),
        archive_helper_sha256=archive.file_hash(Path(archive.__file__)),
        arguments=vars(args), experiments=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.sources or args.runs:
        row = diagnose(ROOT / 'runs' / name, args)
        for previous in report['experiments']:
            if previous['dataset'] == row['dataset']:
                for key in ('train_asset', 'validation_asset'):
                    if key in previous and key in row and previous[key] != row[key]:
                        raise ValueError('Paired fixed asset mismatch: ' + key)
                for key, field in (('train_probe', 'training_indices'), ('validation_probe', 'validation_indices')):
                    if key in previous and key in row and previous[key][field] != row[key][field]:
                        raise ValueError('Paired diagnostic indices mismatch: ' + key)
        report['experiments'].append(row)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        output.with_suffix('.md').write_text(markdown(report))
        print(json.dumps(dict(experiment_id=name, elapsed_seconds=row['elapsed_seconds'], completed=True)), flush=True)


if __name__ == '__main__':
    main()
