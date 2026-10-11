#!/usr/bin/env python3
"""Verify and extract two public benchmark CSVs; preserve every existing asset."""
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/mnt/data/zza/code/RegimeBridge/runtime/downloads/timebridge_official_datasets.zip')
ORIGIN_MANIFEST = Path('/mnt/data/zza/code/RegimeBridge/data/manifest.json')
ORIGIN_README = Path('/mnt/data/zza/code/RegimeBridge/upstream/TimeBridge/README.md')
ARCHIVE_HASH = '01d7016eaa3eb571b77542612219d116dfc6a9f5399586dea1f98261fc4d80c4'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads(ORIGIN_MANIFEST.read_text())
    if manifest['archive_sha256'] != ARCHIVE_HASH or digest(ARCHIVE) != ARCHIVE_HASH:
        raise RuntimeError('Public archive hash differs from its recorded provenance')
    source_url = manifest['source']
    if '1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot' not in ORIGIN_README.read_text():
        raise RuntimeError('Recorded source is absent from locally archived official README')
    specifications = {
        'weather': {
            'member': 'iTransformer_datasets/weather/weather.csv', 'file': 'weather.csv',
            'sha256': '34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63',
            'rows': 52696, 'variables': 21, 'cadence_seconds': 600,
            'loader': 'Dataset_Custom', 'data': 'custom', 'cycle': 144, 'batch_size': 64,
        },
        'ettm2': {
            'member': 'iTransformer_datasets/ETT-small/ETTm2.csv', 'file': 'ETTm2.csv',
            'sha256': 'db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd',
            'rows': 69680, 'variables': 7, 'cadence_seconds': 900,
            'loader': 'Dataset_ETT_minute', 'data': 'ETTm2', 'cycle': 96, 'batch_size': 256,
        },
    }
    report = {
        'step': '6.6', 'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'authorization': 'User explicitly requested two additional datasets after step 5.6',
        'selection_rule': 'Freeze Weather and ETTm2 by availability and domain/cadence coverage, without new validation/test result selection',
        'archive': {'path': str(ARCHIVE), 'sha256': ARCHIVE_HASH, 'size_bytes': ARCHIVE.stat().st_size,
                    'source_url': source_url, 'official_readme': str(ORIGIN_README),
                    'official_readme_sha256': digest(ORIGIN_README),
                    'parent_manifest': str(ORIGIN_MANIFEST), 'parent_manifest_sha256': digest(ORIGIN_MANIFEST)},
        'storage_policy': 'Independent read-only project copies; no hardlinks, source mutations, or deletion',
        'datasets': {},
    }
    with zipfile.ZipFile(ARCHIVE) as source:
        for name, spec in specifications.items():
            content = source.read(spec['member'])  # zipfile checks each member CRC.
            if hashlib.sha256(content).hexdigest() != spec['sha256']:
                raise RuntimeError(f'Member hash mismatch: {name}')
            target = ROOT / 'dataset' / spec['file']
            if target.exists():
                if digest(target) != spec['sha256']:
                    raise RuntimeError(f'Refusing to overwrite different existing asset: {target}')
            else:
                fd, temporary = tempfile.mkstemp(prefix=target.name + '.', suffix='.partial', dir=target.parent)
                with os.fdopen(fd, 'wb') as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, 0o444)
                # Atomic no-clobber publication, unlike replace() of a concurrent file.
                os.link(temporary, target)
                os.unlink(temporary)
            frame = pd.read_csv(target)
            timestamps = pd.to_datetime(frame['date'])
            values = frame.drop(columns='date').to_numpy()
            assert frame.shape == (spec['rows'], spec['variables'] + 1), (name, frame.shape)
            assert 'OT' in frame.columns and np.isfinite(values).all(), name
            assert timestamps.is_monotonic_increasing, name
            cadence = timestamps.diff().dropna().dt.total_seconds()
            anomaly_mask = timestamps.diff().dt.total_seconds().ne(spec['cadence_seconds']) & timestamps.diff().notna()
            timestamp_anomalies = [{'row': int(i), 'previous': str(timestamps.iloc[i - 1]),
                                    'time': str(timestamps.iloc[i]),
                                    'delta_seconds': float((timestamps.iloc[i] - timestamps.iloc[i - 1]).total_seconds())}
                                   for i in timestamps.index[anomaly_mask]]
            # Canonical Weather has one duplicate and one gap. Keep the exact
            # official row sequence instead of changing the evaluation protocol.
            if name == 'weather':
                assert int(timestamps.duplicated().sum()) == 1 and len(timestamp_anomalies) == 2
            else:
                assert timestamps.is_unique and not timestamp_anomalies, timestamp_anomalies
            if name == 'ettm2':
                bounds = {'train': [0, 34560], 'val': [34464, 46080], 'test': [45984, 57600]}
                split_rule = 'Official 12/4/4 nominal 30-day months, 96 samples/day; trailing 12080 rows unused'
            else:
                n = len(frame)
                train, test = int(n * 0.7), int(n * 0.2)
                val = n - train - test
                bounds = {'train': [0, train], 'val': [train - 96, train + val], 'test': [n - test - 96, n]}
                split_rule = 'Official Dataset_Custom: floor(70%) train, floor(20%) test, remaining val; 96-step input overlap only'
            official_args = {
                'data': spec['data'], 'data_path': spec['file'], 'enc_in': spec['variables'],
                'cycle': spec['cycle'], 'batch_size': spec['batch_size'], 'learning_rate': 0.001,
                'dropout': 0.5, 'use_revin': 1, 'features': 'M', 'seq_len': 96, 'pred_len': 96,
                'train_epochs': 30, 'patience': 5, 'random_seed': 2024,
                'label_len': 0, 'freq': 'h', 'embed': 'timeF', 'd_model': 512,
                'lradj': 'type3', 'pct_start': 0.3, 'loss': 'mse', 'use_amp': False,
            }
            script = ROOT / 'scripts' / 'GTR' / (name + '.sh')
            report['datasets'][name] = {
                'path': str(target), 'sha256': spec['sha256'], 'size_bytes': target.stat().st_size,
                'archive_member': spec['member'], 'rows': len(frame), 'variables': spec['variables'],
                'dataset': name, 'source_url': source_url,
                'provenance': 'Exact public archive member; independent copy, no preprocessing',
                'columns': frame.columns.tolist(), 'first_time': str(timestamps.iloc[0]), 'last_time': str(timestamps.iloc[-1]),
                'cadence_seconds': spec['cadence_seconds'], 'finite_values': True,
                'missing_value_count': int(frame.isna().sum().sum()),
                'unique_regular_timestamps': bool(timestamps.is_unique and not timestamp_anomalies),
                'timestamp_duplicate_rows': int(timestamps.duplicated().sum()),
                'cadence_histogram_seconds': {str(k): int(v) for k, v in cadence.value_counts().items()},
                'timestamp_anomalies': timestamp_anomalies,
                'timestamp_anomaly_policy': 'Preserve canonical row sequence, cycle index and official split; no deduplication or resampling',
                'official_loader': spec['loader'], 'official_script': str(script), 'official_script_sha256': digest(script),
                'split_rule': split_rule, 'absolute_data_slice': bounds,
                'window_count': {flag: end - begin - 96 - 96 + 1 for flag, (begin, end) in bounds.items()},
                'normalization': 'StandardScaler fitted on training slice only; same scaler used on all splits',
                'official_args': official_args,
                'frequency_note': 'Preserve official parser default freq=h even though CSV cadence is finer; GTR forward does not consume time marks. No silent override.',
                'evaluation_note': 'Official val shuffled/drop_last, test ordered/full; deterministic mechanism diagnostics separately labeled',
                'paper_mse': None, 'paper_mae': None,
                'paper_reference_note': 'Not yet re-audited for these datasets; no guessed paper values',
            }
    report['new_data_size_bytes'] = sum(item['size_bytes'] for item in report['datasets'].values())
    output = ROOT / 'reports' / 'two_dataset_expansion_plan.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    (ROOT / 'artifacts' / 'expanded_data_assets.json').write_text(json.dumps(report['datasets'], indent=2, ensure_ascii=False) + '\n')
    lines = [
        '# 6.6 两个新增数据集资产与协议', '',
        '用户明确授权后续扩到四个数据集。新增 Weather 与 ETTm2；只跑 seed=2024、T=H=96，先独立 smoke 再正式对照。选择依据为数据可用性和领域/采样间隔互补，没有依据新数据测试结果挑选。', '',
        '| 数据集 | 原始行数 / 变量 | 采样 | cycle | batch | LR | dropout | RevIN | 正式预算 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---|',
        '| Weather | 52696 / 21 | 10 min | 144 | 64 | 0.001 | 0.5 | 1 | 最多30 epoch，patience=5 |',
        '| ETTm2 | 69680 / 7 | 15 min | 96 | 256 | 0.001 | 0.5 | 1 | 最多30 epoch，patience=5 |', '',
        '协议直接来自保留的官方 scripts/GTR/weather.sh、ettm2.sh、run.py 默认值和原 data_loader；未改训练参数。官方默认 freq=h 原样保留，GTR 不读取时间特征。', '',
        '公开来源：本地归档 TimeBridge 官方 README 的 Google Drive 数据链接（https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link），复用已下载 ZIP。全包 SHA256 与既有 manifest 完全一致，两个 ZIP member 的 CRC、CSV SHA256、行列数、有限值与时间递增检查通过；ETTm2 采样严格规则，Weather 原生异常详见下文。无需下载、无预处理、原资产只读。', '',
        '在项目 dataset/ 新增独立只读副本共 16.13 MiB；没有删除任何文件，没有对其他项目文件 chmod，也不以硬链接绑定外部可变数据。来源、脚本和数据哈希详见同名 JSON；原 artifacts/data_assets.json 保留。', '',
        '| 数据集 | train切片 [start,end) | val切片 | test切片 | train/val/test 窗口数 |',
        '|---|---|---|---|---|',
    ]
    for name, item in report['datasets'].items():
        b, w = item['absolute_data_slice'], item['window_count']
        lines.append(f"| {name} | {b['train']} | {b['val']} | {b['test']} | {w['train']}/{w['val']}/{w['test']} |")
    lines.extend([
        '', 'ETTm2 按官方12/4/4 nominal 30-day月划分，使用前57600行；后12080行保留原文件但不进入官方训练/评估。Weather按70%/剩余10%/20%划分。val/test回看重叠只是输入历史，目标不跨入前一split。归一化只拟合训练集。', '',
        'Weather 的官方 CSV 原生含1次重复时间（0-based row19044，2020-05-12 06:00）与1次100min间隔（row21514，2020-05-29 09:30→11:10），不是本项目预处理造成。保持原始行顺序和官方行序相位，不去重、不补点；cycle144并不始终对应精确墙钟日相位。这一限制保留供机制分析。', '',
        '现成 loader 已支持 ETTm2/custom；runner 需要为其 evaluation_assets.json 补充上述 absolute_data_slice，供历史记忆读取正确的绝对预测时间。该修改只补资产元数据，不能改原 loader。', '',
        '本步骤无 GPU 训练，没有新增性能结果。下一步由统一调度队列生成正式GTR与固定A6（per_bank，不含B/C）配置；所有有效参数、代码与评估资产独立归档。原论文这两个数据集数值尚未重新核对，暂留 NA。',
    ])
    (ROOT / 'reports' / 'two_dataset_expansion_plan.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'VERIFIED', 'report': str(output), 'assets': {n: {'path': d['path'], 'sha256': d['sha256']} for n, d in report['datasets'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
