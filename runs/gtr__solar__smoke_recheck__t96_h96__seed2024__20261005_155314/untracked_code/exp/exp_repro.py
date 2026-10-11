"""Observation/checkpoint hooks around the unmodified official GTR loop.

The training function is compiled from the official source, with only an
epoch-boundary restore/save hook and an explicit resume start epoch inserted.
No loss, learning-rate, evaluation, batching, or model-selection logic is copied.
"""
import ast
import hashlib
import inspect
import itertools
import json
import os
import random
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

from exp import exp_main as official
from exp.exp_basic import Exp_Basic


OfficialExp = official.Exp_Main
OfficialEarlyStopping = official.EarlyStopping


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, map_location="cpu"):
    # Full training state intentionally contains NumPy/Python RNG objects.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_checkpoint(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    # Check the full payload before replacing any previously valid checkpoint.
    checked = load_checkpoint(temporary)
    if not isinstance(checked, dict) or not checked:
        raise RuntimeError("Checkpoint integrity check failed: empty payload")
    del checked
    os.replace(temporary, path)


class AtomicEarlyStopping(OfficialEarlyStopping):
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print("Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...".format(
                self.val_loss_min, val_loss))
        atomic_checkpoint(model.state_dict(), Path(path) / "checkpoint.pth")
        atomic_checkpoint(model.state_dict(), ExpRepro.context.run_dir / "checkpoints" / "best.pt")
        self.val_loss_min = val_loss


class BatchLimit:
    """Cap only the independent smoke run; preserve the original DataLoader."""
    def __init__(self, loader, limit):
        self.loader = loader
        self.limit = min(len(loader), int(limit))

    def __len__(self):
        return self.limit

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.limit)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def instrument_training():
    source = textwrap.dedent(inspect.getsource(OfficialExp.train))
    tree = ast.parse(source)
    function = tree.body[0]
    function.decorator_list = []
    found_scheduler = found_epoch = False
    new_body = []
    for statement in function.body:
        new_body.append(statement)
        if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "scheduler"
                for target in statement.targets):
            new_body.extend(ast.parse(
                'self._resume_epoch = self._restore_training(model_optim, scheduler, '
                'locals().get("scaler"), early_stopping)\n').body)
            found_scheduler = True
        if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name) and statement.target.id == "epoch":
            if ast.unparse(statement.iter) != "range(self.args.train_epochs)":
                raise RuntimeError("Official epoch range changed; audit instrumentation before running")
            statement.iter = ast.parse("range(self._resume_epoch, self.args.train_epochs)", mode="eval").body
            save_hook = ast.parse(
                'self._on_epoch_complete(epoch, model_optim, scheduler, locals().get("scaler"), '
                'early_stopping, train_loss, vali_loss, test_loss)\n').body[0]
            # Preserve the official early-stop branch (no LR adjustment there).
            for node in ast.walk(statement):
                if isinstance(node, ast.If) and ast.unparse(node.test) == "early_stopping.early_stop":
                    for index, item in enumerate(node.body):
                        if isinstance(item, ast.Break):
                            node.body.insert(index, ast.parse(ast.unparse(save_hook)).body[0])
                            break
            statement.body.append(save_hook)
            found_epoch = True
    if not (found_scheduler and found_epoch):
        raise RuntimeError("Could not identify official scheduler/epoch loop")
    function.body = new_body
    ast.fix_missing_locations(tree)
    namespace = dict(OfficialExp.train.__globals__)
    namespace["EarlyStopping"] = AtomicEarlyStopping
    exec(compile(tree, "<official_gtr_train_with_epoch_hooks>", "exec"), namespace)
    return namespace["train"], source, ast.unparse(tree)


TRAIN_FUNCTION, ORIGINAL_TRAIN_SOURCE, INSTRUMENTED_TRAIN_SOURCE = instrument_training()


class ExpRepro(OfficialExp):
    context = None

    def __init__(self, args):
        self.args = args
        self.ctx = self.context
        self.ctx.record_resolved(args)
        self._data_cache = {}
        self._measure_inference = False
        self._inference_start = None
        self.global_step = 0
        self.resume_information = {"mode": "from_scratch", "epoch": 0}
        self.checks = {"forward": False, "backward": False, "optimizer_step": False,
                       "checkpoint_save": False, "checkpoint_load": False, "evaluation": False}
        self.inference_seconds = 0.0
        self.inference_batches = 0
        self.train_seconds = 0.0
        self.epoch_history = []
        self.best_epoch = None
        self.ctx.experiment_instance = self
        # OfficialExp.__init__ uses super(Exp_Main, self). The entry-point
        # class replacement changes that global name and would recurse.
        # Its only operation is this same base constructor.
        Exp_Basic.__init__(self, args)
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.trainable_parameters = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        self.model.register_forward_pre_hook(self._before_forward)
        self.model.register_forward_hook(self._after_forward)
        (self.ctx.run_dir / "official_train_source.py").write_text(ORIGINAL_TRAIN_SOURCE, encoding="utf-8")
        (self.ctx.run_dir / "instrumented_train_source.py").write_text(INSTRUMENTED_TRAIN_SOURCE, encoding="utf-8")
        if self.args.use_gpu:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _acquire_device(self):
        # The official method overwrites UUID visibility with the numeric index.
        if self.args.use_gpu:
            if self.args.use_multi_gpu or self.args.gpu != 0:
                raise ValueError("Reproduction runner requires one UUID-visible GPU and logical gpu=0")
            device = torch.device("cuda:0")
            print("Use GPU: cuda:0; CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "")))
            if self.ctx.gpu_uuid and os.environ.get("CUDA_VISIBLE_DEVICES") != self.ctx.gpu_uuid:
                raise ValueError("GPU UUID visibility mismatch")
            torch.cuda.set_device(device)
            print("Visible device:", torch.cuda.get_device_name(device), torch.cuda.get_device_properties(device).total_memory)
            return device
        if self.ctx.gpu_uuid:
            raise RuntimeError("Requested GPU UUID is unavailable to this PyTorch environment")
        print("Use CPU (explicit CPU check or CUDA unavailable)")
        return torch.device("cpu")

    def _before_forward(self, module, arguments):
        if self._measure_inference:
            if self.args.use_gpu:
                torch.cuda.synchronize(self.device)
            self._inference_start = time.perf_counter()

    def _after_forward(self, module, arguments, output):
        self.checks["forward"] = True
        if self._measure_inference:
            if self.args.use_gpu:
                torch.cuda.synchronize(self.device)
            self.inference_seconds += time.perf_counter() - self._inference_start
            self.inference_batches += 1

    def _select_optimizer(self):
        optimizer = super()._select_optimizer()
        def after_step(optim, arguments, keywords):
            self.global_step += 1
            self.checks["optimizer_step"] = True
            self.checks["backward"] = any(parameter.grad is not None for parameter in self.model.parameters())
        optimizer.register_step_post_hook(after_step)
        return optimizer

    def _get_data(self, flag):
        if flag not in self._data_cache:
            dataset, loader = super()._get_data(flag)
            self.ctx.record_evaluation_asset(flag, dataset, loader)
            if self.ctx.smoke_enabled:
                limit = self.ctx.config["smoke"].get("max_train_batches" if flag == "train" else "max_eval_batches", 2)
                loader = BatchLimit(loader, limit)
                if not len(loader):
                    raise ValueError("Smoke loader has zero batches; keep batch size and use enough samples")
            self._data_cache[flag] = dataset, loader
        return self._data_cache[flag]

    def _restore_training(self, optimizer, scheduler, scaler, early_stopping):
        source = self.ctx.resume_checkpoint
        if not source:
            return 0
        source = Path(source)
        try:
            saved = load_checkpoint(source, self.device)
            if saved.get("schema_version") != 1:
                raise ValueError("Unsupported checkpoint schema")
            if saved.get("science_fingerprint") != self.ctx.science_fingerprint:
                raise ValueError("Scientific configuration/data/code fingerprint mismatch")
            if not saved.get("epoch_complete"):
                raise ValueError("Only complete epoch checkpoints can be resumed")
            completed = int(saved["epoch"])
            if completed < 0 or completed > self.args.train_epochs:
                raise ValueError("Checkpoint epoch is outside the unchanged training budget")
            # The selected best state is stored in last.pt as well, so a new run
            # can still evaluate it when no later epoch improves validation.
            if not saved.get("best_model_state"):
                raise ValueError("Checkpoint lacks the validation-selected best model")
            self.model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            if scaler is not None:
                scaler.load_state_dict(saved["scaler"])
            for key, value in saved["early_stopping"].items():
                setattr(early_stopping, key, value)
            self.global_step = saved["global_step"]
            self.epoch_history = saved.get("epoch_history", [])
            self.train_seconds = saved.get("training_time_seconds", 0.0)
            self.best_epoch = saved.get("best_epoch")
            self.checks.update(saved.get("checks", {}))
            best_path = self.ctx.run_dir / "checkpoints" / "best.pt"
            atomic_checkpoint(saved["best_model_state"], best_path)
            setting_path = Path(self.args.checkpoints) / self.ctx.setting
            atomic_checkpoint(saved["best_model_state"], setting_path / "checkpoint.pth")
            random.setstate(saved["rng"]["python"])
            np.random.set_state(saved["rng"]["numpy"])
            torch.set_rng_state(saved["rng"]["torch_cpu"].cpu())
            if self.args.use_gpu:
                cuda_states = saved["rng"]["torch_cuda"]
                if len(cuda_states) != torch.cuda.device_count():
                    raise ValueError("CUDA RNG topology mismatch")
                torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
            self.resume_information = {"mode": "resumed", "source_experiment_id": saved["experiment_id"],
                                       "checkpoint": str(source.resolve()), "sha256": file_hash(source), "epoch": completed}
            self.ctx.append_event("resume", self.resume_information)
            print("Resume complete epoch:", completed, self.resume_information)
            if early_stopping.early_stop:
                return self.args.train_epochs
            return completed
        except Exception as error:
            # Do not partially restore and silently continue after incompatibility.
            # Fail clearly; the queue may launch a fresh ID after explicit diagnosis.
            self.resume_information = {"mode": "restart_from_scratch_required", "checkpoint": str(source), "reason": str(error)}
            self.ctx.append_event("resume_rejected", self.resume_information)
            raise ValueError("Resume rejected; restart_from_scratch with a fresh experiment ID: " + str(error)) from error

    def _on_epoch_complete(self, epoch, optimizer, scheduler, scaler, early_stopping, train_loss, vali_loss, test_loss):
        if not all(np.isfinite(value) for value in (train_loss, vali_loss, test_loss)):
            raise FloatingPointError("Non-finite train/validation/test loss")
        completed = epoch + 1
        if early_stopping.counter == 0:
            self.best_epoch = completed
        history = {"epoch": completed, "train_loss": float(train_loss), "validation_loss": float(vali_loss),
                   "test_loss_observation_only": float(test_loss), "next_epoch_learning_rate": optimizer.param_groups[0]["lr"],
                   "global_step": self.global_step}
        self.epoch_history.append(history)
        self.ctx.append_event("epoch", history)
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
               "torch_cuda": torch.cuda.get_rng_state_all() if self.args.use_gpu else []}
        state = {"schema_version": 1, "experiment_id": self.ctx.experiment_id,
                 "science_fingerprint": self.ctx.science_fingerprint, "epoch_complete": True, "epoch": completed,
                 "global_step": self.global_step, "model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler is not None else None,
                 "early_stopping": dict(vars(early_stopping)), "best_epoch": self.best_epoch, "rng": rng,
                 "best_model_state": load_checkpoint(self.ctx.run_dir / "checkpoints" / "best.pt"),
                 "epoch_history": self.epoch_history, "training_time_seconds": self.train_seconds + time.perf_counter() - self._train_start,
                 "checks": dict(self.checks), "dataloader_state": "epoch_boundary; deterministic datasets; global torch RNG restored"}
        atomic_checkpoint(state, self.ctx.run_dir / "checkpoints" / "last.pt")
        self.checks["checkpoint_save"] = True
        self.ctx.save_resources()
        self.ctx.write_status("RUNNING", epoch=completed, global_step=self.global_step)

    def train(self, setting):
        self.ctx.setting = setting
        self._train_start = time.perf_counter()
        result = TRAIN_FUNCTION(self, setting)
        self.train_seconds += time.perf_counter() - self._train_start
        self.checks["checkpoint_load"] = True  # official loop loaded validation-selected checkpoint
        return result

    def test(self, setting, test=0):
        if test:
            raise ValueError("Use the runner's train/resume path; standalone official test paths are unsupported")
        previous_metric = official.metric
        def capture_metric(predictions, targets):
            values = previous_metric(predictions, targets)
            self.ctx.evaluation_metrics = {key: float(value) if np.ndim(value) == 0 else np.asarray(value).tolist() for key, value in
                                           zip(("mae", "mse", "rmse", "mape", "mspe", "rse", "corr"), values)}
            self.ctx.evaluation_shape = list(predictions.shape)
            self.checks["evaluation"] = True
            return values
        official.metric = capture_metric
        self._measure_inference = True
        try:
            return super().test(setting, test=0)
        finally:
            self._measure_inference = False
            official.metric = previous_metric
