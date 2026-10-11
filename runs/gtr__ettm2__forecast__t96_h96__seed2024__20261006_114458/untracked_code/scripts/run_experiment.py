#!/usr/bin/env python3
"""Run the official GTR entry point with durable reproduction observations."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def shell_output(arguments, cwd=ROOT):
    result = subprocess.run(arguments, cwd=cwd, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable: " + result.stderr.strip()[:500]


def gpu_snapshot():
    try:
        output = shell_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"])
        if output.startswith("unavailable:"):
            return {"time": timestamp(), "error": output, "gpus": []}
        rows = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                continue
            rows.append(dict(index=int(fields[0]), uuid=fields[1], name=fields[2],
                             total_mib=float(fields[3]), used_mib=float(fields[4]), free_mib=float(fields[5])))
        return {"time": timestamp(), "gpus": rows}
    except (OSError, ValueError) as error:
        return {"time": timestamp(), "error": str(error), "gpus": []}


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False


class Context:
    def __init__(self, options, config):
        self.options = options
        self.config = config
        self.run_dir = Path(options.run_dir).resolve()
        self.experiment_id = config.get("experiment", {}).get("id") or self.run_dir.name
        if self.experiment_id != self.run_dir.name:
            raise ValueError("experiment.id must equal the unique --run-dir basename")
        self.gpu_uuid = options.gpu_uuid or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if self.gpu_uuid and not self.gpu_uuid.startswith("GPU-"):
            raise ValueError("Use a GPU UUID in --gpu-uuid/CUDA_VISIBLE_DEVICES; numeric visibility is ambiguous")
        self.smoke_enabled = bool(config.get("smoke", {}).get("enabled", False))
        if self.smoke_enabled and "smoke" not in self.experiment_id.lower():
            raise ValueError("A smoke run requires an independent ID containing 'smoke'")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        previous = self.run_dir / "status.json"
        previous_status = json.loads(previous.read_text()) if previous.exists() else {}
        if previous_status.get("status") in ("SUCCESS", "succeeded", "success") or previous_status.get("state") == "succeeded":
            raise ValueError("Completed experiment IDs are immutable; choose a fresh timestamp/ID")
        self.attempt = int(previous_status.get("attempt", 0)) + 1
        self.resume_checkpoint = options.resume_checkpoint
        if not self.resume_checkpoint and (self.run_dir / "checkpoints" / "last.pt").exists():
            self.resume_checkpoint = str(self.run_dir / "checkpoints" / "last.pt")
        self.start_time = timestamp()
        self.start_clock = time.perf_counter()
        self.experiment_instance = None
        self.evaluation_metrics = {}
        self.evaluation_shape = None
        self.asset_metadata = {}
        self.resource = {"pid": os.getpid(), "experiment_id": self.experiment_id, "attempt": self.attempt,
                         "gpu_uuid": self.gpu_uuid or None, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                         "started_at": self.start_time, "before": gpu_snapshot(), "process_peak_gpu_memory_mib": 0.0}
        self._stop_monitor = threading.Event()
        self.monitor = None
        self.code_hashes = {}

    def append_event(self, event, payload):
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": timestamp(), "attempt": self.attempt, "event": event, "payload": payload}, ensure_ascii=False, allow_nan=False) + "\n")

    def write_status(self, status, **extra):
        state = "succeeded" if status == "SUCCESS" else "running" if status == "RUNNING" else "failed"
        payload = {"experiment_id": self.experiment_id, "id": self.experiment_id, "status": status, "state": state,
                   "attempt": self.attempt, "stage": self.config.get("experiment", {}).get("stage"),
                   "pid": os.getpid(), "gpu_uuid": self.gpu_uuid or None,
                   "started_at": self.start_time, "updated_at": timestamp(), **extra}
        json_write(self.run_dir / "status.json", payload)

    def snapshot_code(self):
        commands = (["git", "rev-parse", "HEAD"], ["git", "status", "--short"], ["git", "diff", "--stat"])
        summary = "\n\n".join("$ " + shlex.join(command) + "\n" + shell_output(command) for command in commands)
        (self.run_dir / "git_state.txt").write_text(summary + "\n", encoding="utf-8")
        self.git_commit = shell_output(["git", "rev-parse", "HEAD"])
        diff = subprocess.run(["git", "diff", "--binary"], cwd=ROOT, capture_output=True, check=False)
        (self.run_dir / "code.diff").write_bytes(diff.stdout)
        tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.split(b"\0")
        known = {item.decode() for item in tracked if item}
        # Explicit project code roots exclude Conda, datasets, checkpoints and
        # credentials. Snapshot new executable code too; git diff omits it.
        candidates = set()
        for folder in ("models", "layers", "exp", "data_provider", "utils", "scripts"):
            for path in (ROOT / folder).rglob("*"):
                if path.is_file() and path.suffix in (".py", ".sh"):
                    candidates.add(path)
        candidates.add(ROOT / "run.py")
        untracked = []
        snapshot_dir = self.run_dir / "untracked_code"
        for path in sorted(candidates):
            relative = path.relative_to(ROOT).as_posix()
            self.code_hashes[relative] = sha256(path)
            if relative not in known:
                target = snapshot_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                untracked.append({"path": relative, "sha256": self.code_hashes[relative], "snapshot": str(target)})
        json_write(self.run_dir / "code_manifest.json", {"git_commit": self.git_commit, "sha256": self.code_hashes, "untracked": untracked})

    def record_environment(self):
        import numpy as np
        import pandas as pd
        import sklearn
        import torch
        self.environment = {"recorded_at": timestamp(), "hostname": platform.node(), "platform": platform.platform(),
                            "python": sys.version, "executable": sys.executable, "conda_prefix": sys.prefix,
                            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                            "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(),
                            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                            "torch_num_threads": torch.get_num_threads(), "git_commit": self.git_commit,
                            "gpu_snapshot_before": self.resource["before"]}
        (self.run_dir / "env.txt").write_text(json.dumps(self.environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def record_resolved(self, args):
        official_args = dict(vars(args))
        scientific = {key: value for key, value in official_args.items() if key not in
                      ("checkpoints", "model_id", "gpu", "devices", "use_gpu", "use_multi_gpu", "des")}
        science_paths = {"run.py", "scripts/run_experiment.py", "exp/exp_basic.py", "exp/exp_main.py",
                         "exp/exp_repro.py", "data_provider/data_factory.py", "data_provider/data_loader.py",
                         "utils/metrics.py", "utils/tools.py", "utils/timefeatures.py",
                         "models/" + official_args["model"] + ".py"}
        identity = {"official_args": scientific, "smoke": self.config.get("smoke", {}),
                    "data_sha256": self.data_sha256,
                    "code_sha256": {key: value for key, value in self.code_hashes.items() if key in science_paths},
                    "environment_versions": {key: self.environment[key] for key in ("python", "torch", "numpy", "pandas", "scikit_learn")}}
        self.science_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.resolved = {"official_args": official_args, "experiment": self.config.get("experiment", {}),
                         "smoke": self.config.get("smoke", {}), "data_asset": {**self.config.get("data_asset", {}), "path": str(self.data_path), "sha256": self.data_sha256},
                         "runner": {"science_fingerprint": self.science_fingerprint, "entrypoint": "runpy.run_path(original run.py)",
                                    "resume_policy": "complete epoch only; identical science/data/code/environment fingerprint",
                                    "loader_protocol": {"train": {"shuffle": True, "drop_last": True}, "val": {"shuffle": True, "drop_last": True}, "test": {"shuffle": False, "drop_last": False}},
                                    "checkpoint_selection": "official validation EarlyStopping; per-epoch test_loss is observation only",
                                    "initial_lr_rule": "official OneCycleLR initialization: learning_rate / 25; type3 does not step OneCycleLR",
                                    "changes_from_config": self.runtime_overrides}}
        # JSON is valid YAML 1.2 and preserves exact defaults without a YAML dependency.
        json_write(self.run_dir / "resolved_config.yaml", self.resolved)
        self.append_event("resolved_config", {"fingerprint": self.science_fingerprint, "changes": self.runtime_overrides})

    def record_evaluation_asset(self, flag, dataset, loader):
        import numpy as np
        def array_digest(array):
            value = np.ascontiguousarray(array)
            return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()
        metadata = {"dataset_class": type(dataset).__name__, "data_sha256": self.data_sha256,
                    "seq_len": dataset.seq_len, "label_len": dataset.label_len, "pred_len": dataset.pred_len,
                    "cycle": dataset.cycle, "window_count": len(dataset), "data_slice_length": len(dataset.data_x),
                    "data_x_sha256": array_digest(dataset.data_x), "data_y_sha256": array_digest(dataset.data_y),
                    "cycle_index_sha256": array_digest(dataset.cycle_index),
                    "forecast_origin_rule": "cycle_index[s_begin + seq_len]; official loader unchanged",
                    "normalization": {key: getattr(dataset.scaler, key).tolist() for key in ("mean_", "var_", "scale_")},
                    "loader": {"batch_size": loader.batch_size, "shuffle": flag != "test", "drop_last": loader.drop_last},
                    "metric_source_sha256": self.code_hashes["utils/metrics.py"],
                    "data_loader_source_sha256": self.code_hashes["data_provider/data_loader.py"],
                    "mask": None, "mask_reason": "complete-observation forecasting; no imputation masks"}
        if self.resolved["official_args"]["data"] in ("ETTh1", "ETTh2"):
            seq_len = dataset.seq_len
            bounds = {"train": [0, 8640], "val": [8640 - seq_len, 11520], "test": [11520 - seq_len, 14400]}
            metadata["absolute_data_slice"] = bounds[flag]
        elif self.resolved["official_args"]["data"] == "Solar":
            train_end, test_length, val_length = int(self.data_rows * 0.7), int(self.data_rows * 0.2), int(self.data_rows * 0.1)
            bounds = {"train": [0, train_end], "val": [train_end - dataset.seq_len, train_end + val_length], "test": [self.data_rows - test_length - dataset.seq_len, self.data_rows]}
            metadata["absolute_data_slice"] = bounds[flag]
        elif self.resolved["official_args"]["data"] in ("ETTm1", "ETTm2"):
            # Mirror the unchanged official 12/4/4-month minute-loader borders.
            seq_len = dataset.seq_len
            bounds = {"train": [0, 34560], "val": [34560 - seq_len, 46080], "test": [46080 - seq_len, 57600]}
            metadata["absolute_data_slice"] = bounds[flag]
            if len(dataset.data_x) != bounds[flag][1] - bounds[flag][0]:
                raise ValueError("Minute dataset does not cover the official split")
        elif self.resolved["official_args"]["data"] == "custom":
            # Dataset_Custom assigns the rounding remainder to validation.
            train_end, test_length = int(self.data_rows * 0.7), int(self.data_rows * 0.2)
            bounds = {"train": [0, train_end], "val": [train_end - dataset.seq_len, self.data_rows - test_length],
                      "test": [self.data_rows - test_length - dataset.seq_len, self.data_rows]}
            metadata["absolute_data_slice"] = bounds[flag]
            if len(dataset.data_x) != bounds[flag][1] - bounds[flag][0]:
                raise ValueError("Custom CSV row count differs from the official loader")
        identity = {key: self.resolved["official_args"][key] for key in ("data", "data_path", "features", "target", "seq_len", "label_len", "pred_len", "cycle", "batch_size")}
        key = hashlib.sha256(json.dumps({"identity": identity, "data": self.data_sha256}, sort_keys=True).encode()).hexdigest()[:20]
        folder = ROOT / "artifacts" / "evaluation" / key
        folder.mkdir(parents=True, exist_ok=True)
        asset = folder / (flag + ".json")
        with (folder / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if asset.exists():
                if json.loads(asset.read_text()) != metadata:
                    raise ValueError("Fixed evaluation asset changed: " + str(asset))
            else:
                json_write(asset, metadata)
                normalization = metadata["normalization"]
                temporary = folder / (flag + ".scaler.tmp.npz")
                np.savez(temporary, **{key: np.asarray(value) for key, value in normalization.items()})
                os.replace(temporary, folder / (flag + ".scaler.npz"))
            # Reuse and verify the frozen normalization statistics, preserving
            # the exact official normalized arrays already checked above.
            with np.load(folder / (flag + ".scaler.npz")) as saved:
                for attribute in ("mean_", "var_", "scale_"):
                    expected = saved[attribute]
                    if not np.array_equal(expected, getattr(dataset.scaler, attribute)):
                        raise ValueError("Frozen normalization mismatch: " + attribute)
                    setattr(dataset.scaler, attribute, expected.copy())
            fcntl.flock(lock, fcntl.LOCK_UN)
        self.asset_metadata[flag] = {"path": str(asset), "sha256": sha256(asset), "scaler": str(folder / (flag + ".scaler.npz")), "split": metadata.get("absolute_data_slice"), "window_count": len(dataset)}
        json_write(self.run_dir / "evaluation_assets.json", self.asset_metadata)

    def _monitor_gpu(self):
        while not self._stop_monitor.wait(5):
            try:
                output = shell_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"])
                for line in output.splitlines():
                    fields = [value.strip() for value in line.split(",")]
                    if len(fields) == 3 and fields[0] == str(os.getpid()):
                        self.resource["process_peak_gpu_memory_mib"] = max(self.resource["process_peak_gpu_memory_mib"], float(fields[2]))
            except (OSError, ValueError):
                pass

    def save_resources(self, final=False, exit_code=None):
        import torch
        experiment = self.experiment_instance
        if experiment is not None and hasattr(experiment, "device") and experiment.args.use_gpu:
            self.resource["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated(experiment.device) / 1024 ** 2
            self.resource["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved(experiment.device) / 1024 ** 2
        self.resource["peak_gpu_memory_mib"] = max(self.resource.get("torch_peak_reserved_mib", 0), self.resource.get("process_peak_gpu_memory_mib", 0))
        self.resource["elapsed_seconds"] = time.perf_counter() - self.start_clock
        if final:
            self.resource.update(ended_at=timestamp(), after=gpu_snapshot(), exit_code=exit_code)
        json_write(self.run_dir / "resource_usage.json", self.resource)

    def metrics(self, status, error=None):
        import math
        result = {**self.config.get("experiment", {}), "experiment_id": self.experiment_id, "status": status,
                  "stage": "smoke" if self.smoke_enabled else self.config.get("experiment", {}).get("stage", "formal"),
                  "seed": self.resolved["official_args"]["random_seed"] if hasattr(self, "resolved") else None,
                  "runtime_seconds": time.perf_counter() - self.start_clock,
                  "peak_gpu_memory_mib": self.resource.get("peak_gpu_memory_mib", 0), "git_commit": self.git_commit,
                  "resolved_config": str(self.run_dir / "resolved_config.yaml"), "gpu_uuid": self.gpu_uuid or None,
                  "evaluation_assets": self.asset_metadata, "error": error}
        if self.experiment_instance is not None:
            exp = self.experiment_instance
            result.update(parameters=exp.parameters if hasattr(exp, "parameters") else None,
                          trainable_parameters=getattr(exp, "trainable_parameters", None),
                          training_time_seconds=exp.train_seconds,
                          inference_time_seconds=exp.inference_seconds,
                          inference_batches=exp.inference_batches,
                          inference_seconds_per_batch=exp.inference_seconds / max(1, exp.inference_batches),
                          best_epoch=exp.best_epoch, epochs_completed=len(exp.epoch_history), global_step=exp.global_step,
                          checkpoint=str(self.run_dir / "checkpoints" / "best.pt"),
                          last_checkpoint=str(self.run_dir / "checkpoints" / "last.pt"),
                          smoke_checks={**exp.checks, "metrics_write": True}, resume=exp.resume_information)
        def finite_value(value):
            if isinstance(value, list):
                return [finite_value(item) for item in value]
            return value if math.isfinite(value) else None
        result.update({key: finite_value(value) for key, value in self.evaluation_metrics.items()})
        for key in ("mse", "mae"):
            paper_value = self.config.get("experiment", {}).get("paper_" + key)
            value = result.get(key)
            if value is not None and paper_value is not None:
                result[key + "_difference_from_paper"] = value - paper_value
                result[key + "_relative_difference_from_paper"] = (value - paper_value) / paper_value if paper_value else None
        return result


def classify(error):
    message = str(error).lower()
    if "out of memory" in message and ("cuda" in message or "cublas" in message):
        return "OOM"
    if isinstance(error, InterruptedError):
        return "EXTERNAL_KILL"
    if isinstance(error, FloatingPointError):
        return "NAN_OR_DIVERGENCE"
    if isinstance(error, (FileNotFoundError, PermissionError)) or "evaluation asset" in message or "normalization mismatch" in message or "data sha256" in message:
        return "DATA_ERROR"
    if isinstance(error, ValueError):
        return "CONFIG_ERROR"
    return "CODE_ERROR" if isinstance(error, (RuntimeError, TypeError, ImportError, AttributeError)) else "UNKNOWN_ERROR"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--run-dir", required=True, help="runs/<unique experiment ID>")
    parser.add_argument("--gpu-uuid", help="Physical GPU UUID; process uses logical cuda:0")
    parser.add_argument("--resume-checkpoint", help="Verified complete epoch last.pt; same run automatically detects it")
    options = parser.parse_args()
    options.config = str(Path(options.config).resolve())
    config = json.loads(Path(options.config).read_text(encoding="utf-8"))
    if options.gpu_uuid:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu_uuid
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.path.insert(0, str(ROOT))
    ctx = Context(options, config)
    argv = [sys.executable, str(Path(__file__).resolve()), "--config", options.config, "--run-dir", str(ctx.run_dir)]
    if ctx.gpu_uuid:
        argv += ["--gpu-uuid", ctx.gpu_uuid]
    if options.resume_checkpoint:
        argv += ["--resume-checkpoint", str(Path(options.resume_checkpoint).resolve())]
    (ctx.run_dir / "command.sh").write_text("#!/bin/sh\ncd " + shlex.quote(str(ROOT)) + "\nexec " + shlex.join(argv) + "\n", encoding="utf-8")
    json_write(ctx.run_dir / "input_config.json", config)
    logfile = (ctx.run_dir / "train.log").open("a", encoding="utf-8", buffering=1)
    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(original_out, logfile), Tee(original_err, logfile)
    original_parse_args = argparse.ArgumentParser.parse_args
    exit_code = 1
    try:
        ctx.snapshot_code()
        ctx.write_status("RUNNING")
        print("Experiment", ctx.experiment_id, "attempt", ctx.attempt, "start", ctx.start_time)
        print("GPU snapshot before:", json.dumps(ctx.resource["before"], ensure_ascii=False))
        overrides = dict(config.get("official_args", {}))
        if int(overrides.get("is_training", 1)) != 1 or int(overrides.get("itr", 1)) != 1:
            raise ValueError("Runner requires is_training=1 and itr=1; use unique IDs for repetitions")
        if overrides.get("use_multi_gpu", False) or overrides.get("do_predict", False):
            raise ValueError("First reproduction runner supports one GPU and train/test only")
        ctx.runtime_overrides = []
        for name, value, reason in (("checkpoints", str(ctx.run_dir / "checkpoints"), "isolate checkpoint outputs"),
                                    ("gpu", 0, "UUID visibility maps the selected GPU to logical zero")):
            if overrides.get(name) != value:
                ctx.runtime_overrides.append({"parameter": name, "original": overrides.get(name, "official default"), "new": value, "reason": reason})
            overrides[name] = value
        if ctx.smoke_enabled:
            if int(overrides.get("train_epochs", 30)) != 1:
                ctx.runtime_overrides.append({"parameter": "train_epochs", "original": overrides.get("train_epochs", 30), "new": 1, "reason": "independent smoke ID only; formal configuration remains unchanged"})
            overrides["train_epochs"] = 1
            for name in ("max_train_batches", "max_eval_batches"):
                if int(config.get("smoke", {}).get(name, 2)) < 1:
                    raise ValueError("Smoke batch limits must be positive")
        elif config.get("smoke", {}).get("max_train_batches") or config.get("smoke", {}).get("max_eval_batches"):
            raise ValueError("Formal run must not inherit smoke batch caps")
        root_path = Path(overrides.get("root_path", "./data/ETT/"))
        if not root_path.is_absolute():
            root_path = ROOT / root_path
        overrides["root_path"] = str(root_path.resolve())
        ctx.data_path = root_path / overrides.get("data_path", "ETTh1.csv")
        ctx.data_sha256 = sha256(ctx.data_path)
        claimed_hash = config.get("data_asset", {}).get("sha256")
        if claimed_hash and claimed_hash != ctx.data_sha256:
            raise ValueError("Data SHA256 mismatch: configured source differs from actual input")
        configured_path = config.get("data_asset", {}).get("path")
        if configured_path and Path(configured_path).resolve() != ctx.data_path.resolve():
            raise ValueError("data_asset.path differs from actual official_args root_path/data_path")
        if overrides.get("data") in ("ETTm1", "ETTm2", "custom"):
            import csv
            with ctx.data_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                if "date" not in header:
                    raise ValueError("Expected the official forecasting CSV date header")
                ctx.data_rows = sum(1 for row in reader if row)
        else:
            # Preserve existing ETTh/Solar metadata and asset identities.
            with ctx.data_path.open("rb") as handle:
                ctx.data_rows = sum(1 for _ in handle)
        ctx.record_environment()
        if ctx.gpu_uuid and not ctx.environment["cuda_available"]:
            raise RuntimeError("Requested GPU UUID is unavailable to this PyTorch environment")
        from exp import exp_main
        from exp.exp_repro import ExpRepro
        ExpRepro.context = ctx
        exp_main.Exp_Main = ExpRepro
        def parse_official(original_parser, args=None, namespace=None):
            actions = {action.dest: action for action in original_parser._actions}
            unknown = set(overrides) - set(actions)
            if unknown:
                raise ValueError("Unknown official arguments: " + ", ".join(sorted(unknown)))
            values = {}
            for key, value in overrides.items():
                action = actions[key]
                if action.type is bool or isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    if value not in (True, False, 0, 1):
                        raise ValueError("Boolean config must be JSON true/false or 0/1: " + key)
                    values[key] = bool(value)
                else:
                    values[key] = action.type(value) if action.type else value
            original_parser.set_defaults(**values)
            required = []
            for action in original_parser._actions:
                if action.required:
                    required += [action.option_strings[0], str(values.get(action.dest, action.default))]
            resolved = original_parse_args(original_parser, required, namespace)
            return resolved
        argparse.ArgumentParser.parse_args = parse_official
        def interrupted(number, frame):
            raise InterruptedError("Experiment received signal " + str(number))
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        ctx.monitor = threading.Thread(target=ctx._monitor_gpu, daemon=True)
        ctx.monitor.start()
        # Original entry point writes its own result.txt/visuals inside this run.
        os.chdir(ctx.run_dir)
        runpy.run_path(str(ROOT / "run.py"), run_name="__main__")
        if not ctx.evaluation_metrics or not ctx.experiment_instance:
            raise RuntimeError("Official loop completed without captured evaluation metrics")
        if ctx.smoke_enabled and not all(ctx.experiment_instance.checks.values()):
            raise RuntimeError("Smoke verification incomplete: " + str(ctx.experiment_instance.checks))
        ctx.save_resources()
        json_write(ctx.run_dir / "metrics.json", ctx.metrics("SUCCESS"))
        ctx.write_status("SUCCESS", exit_code=0, ended_at=timestamp(), metrics_path=str(ctx.run_dir / "metrics.json"))
        ctx.append_event("complete", {"status": "SUCCESS", "mse": ctx.evaluation_metrics.get("mse"), "mae": ctx.evaluation_metrics.get("mae")})
        exit_code = 0
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code == 0:
            error = RuntimeError("Original entry point exited before required training/evaluation completed")
        kind = classify(error)
        traceback.print_exc()
        detail = {"type": type(error).__name__, "classification": kind, "message": str(error)}
        ctx.append_event("failure", detail)
        ctx.write_status(kind, error_type=kind, error=detail, exit_code=1, ended_at=timestamp())
        if hasattr(ctx, "git_commit"):
            try:
                ctx.save_resources()
                json_write(ctx.run_dir / "metrics.json", ctx.metrics(kind, detail))
            except Exception as secondary:
                print("Failure evidence could not be fully written:", secondary)
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
        ctx._stop_monitor.set()
        if ctx.monitor:
            ctx.monitor.join(timeout=2)
        try:
            ctx.save_resources(final=True, exit_code=exit_code)
        except Exception as error:
            print("Final resource snapshot unavailable:", error)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = original_out, original_err
        logfile.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
