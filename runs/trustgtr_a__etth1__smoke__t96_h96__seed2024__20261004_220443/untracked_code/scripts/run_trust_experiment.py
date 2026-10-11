#!/usr/bin/env python3
"""Use the same official training/evaluation loop for switchable TrustGTR."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from exp import exp_repro
from models import TrustGTR
import run_experiment as runner


class AbsoluteOriginDataset(Dataset):
    def __init__(self, dataset, border):
        self.dataset = dataset
        self.border = border

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        original = self.dataset[index]
        origin = self.border + index + self.dataset.seq_len
        if origin % self.dataset.cycle != int(original[-1]):
            raise ValueError('Absolute origin does not preserve the official base phase')
        return (*original[:-1], torch.tensor(origin))


class TrustContext(runner.Context):
    def record_resolved(self, args):
        args.trustgtr = TrustGTR.resolved_options(self.config.get('trustgtr', {}), args.cycle)
        if args.model != 'TrustGTR' or args.features != 'M':
            raise ValueError('TrustGTR quick screen requires model=TrustGTR and features=M')
        super().record_resolved(args)
        extra = self.code_hashes['scripts/run_trust_experiment.py'] + self.code_hashes['models/GTR.py']
        self.science_fingerprint = hashlib.sha256((self.science_fingerprint + extra).encode()).hexdigest()
        self.resolved['trustgtr'] = args.trustgtr
        self.resolved['runner'].update(science_fingerprint=self.science_fingerprint,
                                      model_class='models.TrustGTR.Model',
                                      criterion='same forecast MSE; gate/contrast auxiliary losses on training only',
                                      phase_transform='absolute forecast origin when multi-cycle is enabled; modulo base cycle verified',
                                      gate_fallback='jointly trained local convolution/MLP branch; no baseline-performance guarantee')
        runner.json_write(self.run_dir / 'resolved_config.yaml', self.resolved)
        self.append_event('trustgtr_options', {'options': args.trustgtr, 'fingerprint': self.science_fingerprint})
        print('TrustGTR resolved options:', json.dumps(args.trustgtr, sort_keys=True))

    def metrics(self, status, error=None):
        result = super().metrics(status, error)
        experiment = self.experiment_instance
        if experiment is not None and hasattr(experiment, 'model'):
            diagnostics = dict(experiment.model.diagnostics)
            if diagnostics['gate_count']:
                diagnostics['gate_mean'] = diagnostics['gate_sum'] / diagnostics['gate_count']
            if diagnostics['contrast_total_anchors']:
                diagnostics['contrast_valid_fraction'] = diagnostics['contrast_valid_anchors'] / diagnostics['contrast_total_anchors']
            result['trustgtr_diagnostics'] = diagnostics
            result['trustgtr'] = experiment.model.options
        return result


class ExpTrust(exp_repro.ExpRepro):
    def __init__(self, args):
        self._absolute_loaders = {}
        super().__init__(args)

    def _build_model(self):
        return TrustGTR.Model(self.args).float()

    def _select_criterion(self):
        return TrustGTR.ForecastLoss(self.model)

    def _get_data(self, flag):
        original_data, original_loader = super()._get_data(flag)
        if not self.args.trustgtr['use_multi_cycle']:
            return original_data, original_loader
        if flag not in self._absolute_loaders:
            start = self.ctx.asset_metadata[flag]['split'][0]
            dataset = AbsoluteOriginDataset(original_data, start)
            is_limited = isinstance(original_loader, exp_repro.BatchLimit)
            base = original_loader.loader if is_limited else original_loader
            loader = DataLoader(dataset, batch_size=base.batch_size,
                                shuffle=flag != 'test', num_workers=base.num_workers,
                                drop_last=base.drop_last, pin_memory=base.pin_memory)
            if is_limited:
                loader = exp_repro.BatchLimit(loader, len(original_loader))
            self._absolute_loaders[flag] = original_data, loader
            self.ctx.append_event('absolute_phase_loader',
                                  {'flag': flag, 'border': start, 'unchanged_x_y': True,
                                   'rule': 'border + window_index + seq_len', 'base_modulo_verified_each_sample': True})
        return self._absolute_loaders[flag]


if __name__ == '__main__':
    runner.Context = TrustContext
    exp_repro.ExpRepro = ExpTrust
    raise SystemExit(runner.main())
