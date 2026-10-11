#!/usr/bin/env python3
"""Verify asset borders against real official loaders and absolute origins."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
import run_experiment as runner
from run_trust_experiment import AbsoluteOriginDataset


class ProtocolChecks(unittest.TestCase):
    def verify(self, data, filename, cls, cycle, batch_size, recorded=None):
        path = ROOT / 'dataset' / filename
        with path.open(newline='') as handle:
            reader = csv.reader(handle)
            next(reader)
            rows = sum(1 for record in reader if record)
        args = dict(data=data, data_path=filename, features='M', target='OT',
                    seq_len=96, label_len=0, pred_len=96, cycle=cycle, batch_size=batch_size)
        old_root = runner.ROOT
        try:
            with tempfile.TemporaryDirectory(prefix='trustgtr_protocol_') as directory:
                runner.ROOT = Path(directory)
                ctx = runner.Context.__new__(runner.Context)
                ctx.resolved = dict(official_args=args)
                ctx.data_sha256 = runner.sha256(path)
                ctx.data_rows = rows
                ctx.code_hashes = {name: runner.sha256(ROOT / name) for name in
                                  ('utils/metrics.py', 'data_provider/data_loader.py')}
                ctx.run_dir = runner.ROOT / 'run'
                ctx.run_dir.mkdir()
                ctx.asset_metadata = {}
                ctx.append_event = lambda *values: None
                for flag in ('train', 'val', 'test'):
                    dataset = cls(root_path=str(ROOT / 'dataset'), flag=flag,
                        size=[96, 0, 96], features='M', target='OT', cycle=cycle,
                        data_path=filename, timeenc=1, freq='h')
                    loader = SimpleNamespace(batch_size=batch_size, drop_last=flag != 'test')
                    ctx.record_evaluation_asset(flag, dataset, loader)
                    asset = ctx.asset_metadata[flag]
                    start, stop = asset['split']
                    self.assertEqual(stop - start, len(dataset.data_x))
                    wrapper = AbsoluteOriginDataset(dataset, start)
                    for index in (0, len(dataset) // 2, len(dataset) - 1):
                        original, transformed = dataset[index], wrapper[index]
                        self.assertEqual(int(transformed[-1]), start + index + 96)
                        self.assertEqual(int(transformed[-1]) % cycle, int(original[-1]))
                        for left, right in zip(original[:-1], transformed[:-1]):
                            self.assertTrue((left == right).all())
                    metadata = json.loads(Path(asset['path']).read_text())
                    self.assertEqual(metadata['window_count'], len(dataset))
                    if recorded:
                        reference = json.loads((ROOT / recorded / (flag + '.json')).read_text())
                        self.assertEqual(metadata, reference)
        finally:
            runner.ROOT = old_root

    def test_etth1_existing_asset_metadata_exact(self):
        self.verify('ETTh1', 'ETTh1.csv', Dataset_ETT_hour, 24, 256,
                    'artifacts/evaluation/bfe9623f632b1c68ce78')

    def test_ettm2_official_minute_splits_and_window_values(self):
        self.verify('ETTm2', 'ETTm2.csv', Dataset_ETT_minute, 96, 256)

    def test_weather_official_remainder_split_and_window_values(self):
        self.verify('custom', 'weather.csv', Dataset_Custom, 144, 64)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
