#!/usr/bin/env python3
"""CPU-only validation diagnostic; no training and no test targets are loaded."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from data_provider.data_loader import Dataset_ETT_hour, Dataset_Solar
from models.GTR import Model
from utils.metrics import MAE, MSE


def digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def windows(data, length):
    return np.lib.stride_tricks.sliding_window_view(data, length, axis=0)


def descriptors(history, floor=0.1, bins=12):
    # Input N,C,T. Statistics and shape are entirely observed-history features.
    mean = history.mean(-1)
    scale = np.maximum(history.std(-1, ddof=0), floor)
    normalized = (history - mean[..., None]) / scale[..., None]
    quarter = max(1, history.shape[-1] // 4)
    trend = (history[..., -quarter:].mean(-1) - history[..., :quarter].mean(-1)) / scale
    stats = np.stack((mean, np.log(scale), trend), -1)
    if history.shape[-1] % bins:
        raise ValueError('Diagnostic shape bins must divide input length')
    shape = normalized.reshape(*normalized.shape[:-1], bins, history.shape[-1] // bins).mean(-1)
    return mean.astype('float32'), scale.astype('float32'), stats.astype('float32'), shape.astype('float32')


def window_keys(history, stat_center, stat_scale):
    """Public observed-history key API: N,C,T -> keys N,C,15, mean/scale N,C."""
    mean, scale, stats, shape = descriptors(history)
    key = np.concatenate(((stats - stat_center) / stat_scale / math.sqrt(3),
                          shape / math.sqrt(shape.shape[-1])), -1)
    return key, mean, scale


def history_bank(train, seq, horizon):
    count = len(train) - seq - horizon + 1
    all_windows = windows(train, seq)[:count]
    means, scales, stats, shapes = [], [], [], []
    for offset in range(0, count, 512):
        values = descriptors(all_windows[offset:offset + 512])
        for destination, value in zip((means, scales, stats, shapes), values):
            destination.append(value)
    mean, scale, stats, shape = [np.concatenate(items) for items in (means, scales, stats, shapes)]
    # Training-only feature scaling; separate groups each contribute mean squared distance.
    stat_center = stats.mean(0)
    stat_scale = np.maximum(stats.std(0), 0.1)
    keys = np.concatenate(((stats - stat_center) / stat_scale / math.sqrt(3),
                           shape / math.sqrt(shape.shape[-1])), -1)
    return dict(origins=np.arange(count) + seq, mean=mean, scale=scale,
                keys=keys, stat_center=stat_center, stat_scale=stat_scale)


def age_coverage_candidates(origins, query_origin, horizon, cycle, same_phase, per_band=64):
    # Exclusive future end j+H <= t; the query knows observations only through t-1.
    eligible = origins + horizon <= query_origin
    if same_phase:
        eligible &= origins % cycle == query_origin % cycle
    candidates = np.flatnonzero(eligible)
    candidates = candidates[np.argsort(query_origin - origins[candidates], kind='stable')]
    # Relative-age terciles keep the finite train-only bank represented at all ages.
    bands = np.array_split(candidates, 3)
    selected = [band[np.unique(np.linspace(0, len(band) - 1, min(per_band, len(band))).astype(int))]
                for band in bands if len(band)]
    return (np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)), [len(band) for band in bands]


def retrieve(query, query_origin, train, bank, seq, horizon, cycle, same_phase, diagnose_banks=False):
    query_keys, mean, scale = window_keys(query[None].transpose(0, 2, 1), bank['stat_center'], bank['stat_scale'])
    key = query_keys[0]
    candidates, population = age_coverage_candidates(bank['origins'], query_origin, horizon, cycle, same_phase)
    if not len(candidates):
        raise ValueError('No causal training references')
    distance = np.square(bank['keys'][candidates] - key[None]).sum(-1)
    k = min(3, len(candidates))
    positions = np.argpartition(distance, k - 1, axis=0)[:k]
    channels = np.arange(query.shape[-1])[None]
    selected = candidates[positions]
    selected_distance = distance[positions, channels]
    weights = 1 / (selected_distance + 1e-3)
    weights /= weights.sum(0, keepdims=True)
    origins = bank['origins'][selected]
    if np.any(origins + horizon > query_origin):
        raise AssertionError('Causal retrieval bound violated')
    horizon_offsets = np.arange(horizon)[:, None, None]
    future = train[origins[None] + horizon_offsets, channels[None]]
    standardized = (future - bank['mean'][selected, channels][None]) / bank['scale'][selected, channels][None]
    prediction = (standardized * weights[None]).sum(1) * scale[0][None] + mean[0][None]
    ages = query_origin - origins
    source = dict(candidate_count=len(candidates), population_by_age_tercile=population,
                  selected_age_min=int(ages.min()), selected_age_max=int(ages.max()),
                  selected_age_sum=float(ages.sum()), selected_count=int(ages.size))
    if diagnose_banks:
        band_ids = np.repeat(np.arange(3), [min(64, count) for count in population])
        selected_bands = band_ids[positions]
        source['selected_weight_by_age_band'] = [float((weights * (selected_bands == band)).sum())
                                                for band in range(3)]
        source['bank_predictions'] = {}
        for band, label in enumerate(('short_relative_age', 'middle_relative_age', 'long_relative_age')):
            available = np.flatnonzero(band_ids == band)
            if not len(available):
                continue
            k_band = min(3, len(available))
            selected_positions = available[np.argpartition(distance[available], k_band - 1, axis=0)[:k_band]]
            references = candidates[selected_positions]
            bank_weights = 1 / (distance[selected_positions, channels] + 1e-3)
            bank_weights /= bank_weights.sum(0, keepdims=True)
            historical_values = train[bank['origins'][references][None] + horizon_offsets, channels[None]]
            normalized_values = (historical_values - bank['mean'][references, channels][None]) / bank['scale'][references, channels][None]
            source['bank_predictions'][label] = ((normalized_values * bank_weights[None]).sum(1) * scale[0][None] + mean[0][None]).astype('float32')
    return prediction.astype('float32'), source


def scores(prediction, target):
    return dict(mse=float(MSE(prediction, target)), mae=float(MAE(prediction, target)))


def dataset_diagnostic(name, sample_count, threads):
    started = time.monotonic()
    run = ROOT / 'runs' / f'gtr__{name}__forecast__t96_h96__seed2024__20261004_214520'
    config = json.loads((run / 'resolved_config.yaml').read_text())
    args = SimpleNamespace(**config['official_args'])
    assets = json.loads((run / 'evaluation_assets.json').read_text())
    cls = Dataset_ETT_hour if name == 'etth1' else Dataset_Solar
    kwargs = dict(root_path=args.root_path, data_path=args.data_path, size=[args.seq_len, args.label_len, args.pred_len],
                  features=args.features, target=args.target, timeenc=int(args.embed == 'timeF'), freq=args.freq, cycle=args.cycle)
    train_dataset, val_dataset = [cls(flag=flag, **kwargs) for flag in ('train', 'val')]
    verified = {}
    for flag, data in (('train', train_dataset), ('val', val_dataset)):
        metadata = json.loads(Path(assets[flag]['path']).read_text())
        checks = dict(data_x=digest(data.data_x) == metadata['data_x_sha256'],
                      data_y=digest(data.data_y) == metadata['data_y_sha256'],
                      phase=digest(data.cycle_index) == metadata['cycle_index_sha256'],
                      mean=np.array_equal(data.scaler.mean_, np.asarray(metadata['normalization']['mean_'])),
                      scale=np.array_equal(data.scaler.scale_, np.asarray(metadata['normalization']['scale_'])))
        if not all(checks.values()):
            raise ValueError(f'Fixed asset verification failed: {name}/{flag}/{checks}')
        verified[flag] = dict(path=assets[flag]['path'], sha256=file_hash(assets[flag]['path']), checks=checks)
    n = len(val_dataset)
    indices = np.arange(n) if sample_count == 0 or sample_count >= n else np.unique(np.linspace(0, n - 1, sample_count).astype(int))
    seq, horizon = args.seq_len, args.pred_len
    val_data = np.asarray(val_dataset.data_x, dtype='float32')
    train_data = np.asarray(train_dataset.data_x, dtype='float32')
    x = np.stack([val_data[index:index + seq] for index in indices])
    target = np.stack([val_data[index + seq:index + seq + horizon] for index in indices])
    origins = assets['val']['split'][0] + indices + seq
    checkpoint = run / 'checkpoints/best.pt'
    model = Model(args).cpu().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(indices), 8):
            inputs = torch.from_numpy(x[offset:offset + 8])
            phase = torch.tensor(origins[offset:offset + 8] % args.cycle, dtype=torch.long)
            predictions.append(model(inputs, phase).numpy())
    baseline = np.concatenate(predictions)
    print(f'{name}: baseline on {len(indices)} fixed validation windows {scores(baseline, target)}', flush=True)
    bank = history_bank(train_data, seq, horizon)
    modes = {}
    for same_phase in (True, False):
        label = 'same_phase' if same_phase else 'without_phase'
        results, source_stats = [], []
        bank_results = {label: [] for label in ('short_relative_age', 'middle_relative_age', 'long_relative_age')}
        for offset, (query, origin) in enumerate(zip(x, origins)):
            result, source = retrieve(query, int(origin), train_data, bank, seq, horizon, args.cycle, same_phase, diagnose_banks=True)
            results.append(result)
            for bank_label, bank_prediction in source.pop('bank_predictions').items():
                bank_results[bank_label].append(bank_prediction)
            source_stats.append(source)
        retrieved = np.stack(results)
        delta = retrieved - baseline
        power = np.square(delta).sum(1, keepdims=True)
        optimum = np.clip(((target - baseline) * delta).sum(1, keepdims=True) / np.maximum(power, 1e-8), 0, 1)
        oracle = baseline + optimum * delta
        blend = {str(weight): scores(baseline + weight * delta, target) for weight in (0.05, 0.1, 0.2)}
        baseline_sample_mse = np.square(baseline - target).mean((1, 2))
        retrieve_sample_mse = np.square(retrieved - target).mean((1, 2))
        # Also report a stricter one-coefficient-per-whole-window diagnostic.
        window_power = np.square(delta).sum((1, 2), keepdims=True)
        window_optimum = np.clip(((target - baseline) * delta).sum((1, 2), keepdims=True) / np.maximum(window_power, 1e-8), 0, 1)
        modes[label] = dict(retrieval=scores(retrieved, target), fixed_blends=blend,
                            oracle_sample_channel=scores(oracle, target),
                            oracle_sample_window=scores(baseline + window_optimum * delta, target),
                            oracle_label='target-aware diagnostic upper bound; never deploy or count as model performance',
                            oracle_gate_mean=float(optimum.mean()), oracle_gate_positive_fraction=float((optimum > 0).mean()),
                            retrieval_better_window_fraction=float((retrieve_sample_mse < baseline_sample_mse).mean()),
                            selected_age_min=min(row['selected_age_min'] for row in source_stats),
                            selected_age_max=max(row['selected_age_max'] for row in source_stats),
                            selected_age_mean=sum(row['selected_age_sum'] for row in source_stats) / sum(row['selected_count'] for row in source_stats),
                            candidate_count_min=min(row['candidate_count'] for row in source_stats),
                            candidate_count_max=max(row['candidate_count'] for row in source_stats),
                            candidate_population_age_terciles_first=source_stats[0]['population_by_age_tercile'],
                            selected_weight_by_age_band_mean=(np.asarray([row['selected_weight_by_age_band'] for row in source_stats]).sum(0) / (len(indices) * args.enc_in)).tolist(),
                            bank_decomposition={bank_label: dict(retrieval=scores(np.stack(values), target),
                                  fixed_blends={str(weight): scores(baseline + weight * (np.stack(values) - baseline), target)
                                                for weight in (0.05, 0.1, 0.2)})
                                                for bank_label, values in bank_results.items() if len(values) == len(indices)})
        print(f'{name}/{label}: {json.dumps(modes[label])}', flush=True)
    return dict(dataset=name, diagnostic_only=True, no_optimizer_steps=True,
                no_test_split_constructed=True, no_test_targets_used=True,
                validation_scope='all validation windows' if len(indices) == n else 'fixed evenly spaced validation subsample',
                validation_window_count=n, evaluated_windows=len(indices), validation_indices=indices.tolist(),
                baseline=scores(baseline, target), retrieval_modes=modes,
                baseline_run=str(run), checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                fixed_assets=verified, input_length=seq, horizon=horizon, cycle=args.cycle,
                elapsed_seconds=time.monotonic() - started, cpu_threads=threads,
                diagnostic_script_sha256=file_hash(Path(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=('etth1', 'solar'), default=['etth1', 'solar'])
    parser.add_argument('--sample-count', type=int, default=256, help='0 means all fixed validation windows')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', default='reports/retrieval_headroom_subsample.json')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(args.threads)
    torch.manual_seed(2024)
    np.random.seed(2024)
    report = dict(stage='read_only_validation_prediction_diagnostic', method={
        'candidates': 'all complete train windows only; j+H<=query forecast origin t (known through t-1)',
        'candidate_age_coverage': 'sort by t-j, split into three relative-age terciles, deterministic evenly spaced <=64 per tercile',
        'phase_ablation': 'same forecast-origin modulo official base cycle versus no phase restriction; same <=192 cap',
        'input_keys': 'window mean, log std, normalized last-minus-first-quarter trend plus 12 standardized block means',
        'key_scaling': 'training-only per-channel mean/std for 3 statistics; equal total weighting of statistics and window shape',
        'retrieval': 'per-channel k=3 nearest input keys; inverse squared-distance weights with additive 1e-3',
        'future_transfer': 'candidate future centered/scaled by candidate observed history, then query history mean/std; std floor 0.1 dataset-standardized units',
        'fixed_blends': [0.05, 0.1, 0.2], 'oracle': 'clipped analytic MSE optimum; target-aware diagnostic only',
        'validation_metric': 'unchanged official MSE/MAE functions on fixed samples; no shuffled drop_last, so not the training early-stopping aggregate',
        'selection_policy': 'fixed methods/weights stated before evaluation; no test target access or parameter search'}, datasets=[])
    for dataset in args.datasets:
        report['datasets'].append(dataset_diagnostic(dataset, args.sample_count, args.threads))
        path = ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Refuse to overwrite diagnostic report {path}')
        # Incremental unique backup if the second dataset fails; final destination once at end.
        partial = path.with_suffix('.partial.json')
        partial.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Saved {path}', flush=True)


if __name__ == '__main__':
    main()
