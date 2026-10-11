#!/usr/bin/env python3
# Keep this filename distinct from Python's standard-library queue module.
"""GPU UUID queue; stdlib only. Training parameters belong to each job's argv.

queue.json: {"max_jobs":3,"poll_seconds":3,"jobs":[{"id":"unique_id",
"argv":["/path/python","scripts/run_experiment.py",...],"cwd":"/project",
"run_dir":"runs/unique_id","required_memory_mib":1500,"smoke_passed":true}]}
Use is_smoke:true for smoke jobs, and depends_on:["smoke_id"] for formal jobs.
required_memory_mib is a peak estimate, never a training configuration change.
An unknown estimate can be attempted alone on a GPU after smoke has passed.
"""
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
TERMINAL = {"SUCCESS", "DATA_ERROR", "CONFIG_ERROR", "CODE_ERROR",
            "NAN_OR_DIVERGENCE", "EXTERNAL_KILL", "UNKNOWN_ERROR", "STOPPED"}
ACTIVE = {"RUNNING", "CLEANING"}
RETRY = {"PENDING", "OOM", "WAITING_RESOURCE"}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temp.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


@contextlib.contextmanager
def lock(path, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        try:
            yield stream
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def proc_info(pid):
    """UID, process group, session and Linux start ticks defend against PID reuse."""
    try:
        path = Path("/proc") / str(int(pid))
        raw = (path / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "uid": path.stat().st_uid, "state": fields[0],
                "pgid": int(fields[2]), "sid": int(fields[3]),
                "start_ticks": int(fields[19])}
    except (OSError, ValueError, IndexError):
        return None


def same_process(identity):
    info = proc_info(identity.get("pid", -1))
    return bool(info and info["uid"] == os.getuid() and info["state"] != "Z"
                and info["start_ticks"] == identity.get("start_ticks"))


def group_members(record):
    pid = record.get("pid")
    start = record.get("start_ticks", 0)
    if not pid:
        return []
    result = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            info = proc_info(int(entry.name))
            if (info and info["pgid"] == pid and info["sid"] == pid
                    and info["uid"] == os.getuid() and info["start_ticks"] >= start
                    and info["state"] != "Z"):
                result.append(info)
    return result


def signal_group(record, sig):
    members = group_members(record)
    if members:
        try:
            os.killpg(record["pid"], sig)
        except ProcessLookupError:
            pass
    return bool(members)


def classify(returncode, log_text):
    text = log_text.lower()
    if returncode == 0:
        return "SUCCESS"
    if any(x in text for x in ("cuda out of memory", "cuda error: out of memory",
                               "torch.outofmemoryerror", "cudnn_status_alloc_failed")):
        return "OOM"
    if any(x in text for x in ("ncclsystemerror", "nccl system error", "nccl error", "resource temporarily unavailable")):
        return "OOM"
    if any(x in text for x in ("filenotfounderror", "emptydataerror", "parsererror", "badzipfile",
                               "no such file or directory", "dataset not found")):
        return "DATA_ERROR"
    if any(x in text for x in ("unrecognized arguments", "invalid choice", "configerror", "configuration error")):
        return "CONFIG_ERROR"
    if any(x in text for x in ("loss is nan", "loss=nan", "loss: nan", "non-finite", "divergence")):
        return "NAN_OR_DIVERGENCE"
    if any(x in text for x in ("traceback (most recent call last)", "syntaxerror", "modulenotfounderror",
                               "assertionerror", "runtimeerror", "typeerror", "valueerror")):
        return "CODE_ERROR"
    if returncode is None or returncode < 0 or returncode in (137, 143):
        return "EXTERNAL_KILL"
    return "UNKNOWN_ERROR"


def tail(path, size=32768):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def load_config(path):
    config_path = Path(path).resolve()
    config = read_json(config_path)
    if not isinstance(config, dict) or not isinstance(config.get("jobs"), list):
        raise ValueError("Queue config must contain jobs list")
    config["max_jobs"] = int(config.get("max_jobs", 3))
    if not 1 <= config["max_jobs"] <= 3:
        raise ValueError("max_jobs must be 1..3")
    config["poll_seconds"] = max(0.05, float(config.get("poll_seconds", 3)))
    config["runtime_dir"] = str(Path(config.get("runtime_dir", PROJECT / "runtime")).resolve())
    config["shared_lock_dir"] = str(Path(config.get("shared_lock_dir", PROJECT.parent.parent / "runtime")).resolve())
    config["config_path"] = str(config_path)
    seen = set()
    run_dirs = set()
    for job in config["jobs"]:
        identifier = job.get("id", "")
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", identifier) or identifier in seen:
            raise ValueError("Job IDs must be unique, safe file names")
        seen.add(identifier)
        if not isinstance(job.get("argv"), list) or not job["argv"] or not all(isinstance(x, str) for x in job["argv"]):
            raise ValueError(identifier + ": argv must be a nonempty string list")
        job["cwd"] = str(Path(job.get("cwd", PROJECT)).resolve())
        run_dir = Path(job.get("run_dir", "runs/" + identifier))
        job["run_dir"] = str((Path(job["cwd"]) / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve())
        if job["run_dir"] in run_dirs:
            raise ValueError("Each job needs a unique run_dir")
        run_dirs.add(job["run_dir"])
        job["required_memory_mib"] = max(0, float(job.get("required_memory_mib") or 0))
        job["depends_on"] = job.get("depends_on", [])
        if not isinstance(job["depends_on"], list):
            raise ValueError("depends_on must be a list")
    for job in config["jobs"]:
        if any(dep not in seen or dep == job["id"] for dep in job["depends_on"]):
            raise ValueError("Invalid dependency for " + job["id"])
        if job.get("memory_estimate_from") and job["memory_estimate_from"] not in seen:
            raise ValueError("Unknown memory_estimate_from job: " + job["id"])
    visiting, visited = set(), set()
    jobs = {job["id"]: job for job in config["jobs"]}
    def visit(identifier):
        if identifier in visiting:
            raise ValueError("Dependency cycle")
        if identifier in visited:
            return
        visiting.add(identifier)
        for dep in jobs[identifier]["depends_on"]:
            visit(dep)
        visiting.remove(identifier)
        visited.add(identifier)
    for identifier in jobs:
        visit(identifier)
    return config


def signature(job):
    stable = {k: job[k] for k in ("id", "argv", "cwd", "run_dir")}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def gpu_snapshot(config):
    if config.get("fake_gpu_file"):
        gpus = read_json(config["fake_gpu_file"], [])
        apps = read_json(config.get("fake_compute_file", "/nonexistent"), [])
        return gpus, apps
    output = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
    gpus = []
    for row in csv.reader(output.stdout.splitlines()):
        if len(row) == 6:
            index, uuid, name, total, used, free = [x.strip() for x in row]
            gpus.append({"index": int(index), "uuid": uuid, "name": name,
                         "total_mib": float(total), "used_mib": float(used), "free_mib": float(free)})
    apps = []
    try:
        output = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        for row in csv.reader(output.stdout.splitlines()):
            if len(row) == 3:
                uuid, pid, memory = [x.strip() for x in row]
                if memory not in ("[N/A]", "N/A", "[Not Supported]"):
                    apps.append({"uuid": uuid, "pid": int(pid), "used_mib": float(memory)})
    except (subprocess.SubprocessError, ValueError):
        pass  # Unknown per-process allocation means full peak reservation, safely conservative.
    allowed = set(config.get("gpu_uuids", []))
    return ([gpu for gpu in gpus if not allowed or gpu["uuid"] in allowed], apps)


def margin_mib(gpu):
    return min(6144, max(256, float(gpu["total_mib"]) * 0.03))


class Queue:
    def __init__(self, config):
        self.config = config
        self.runtime = Path(config["runtime_dir"])
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.path = self.runtime / "queue_state.json"
        self.state = read_json(self.path, {"jobs": {}, "history": []})
        self.jobs = {job["id"]: job for job in config["jobs"]}
        self.processes = {}
        self.stopping = False
        self.summary_process = None
        self.summary_dirty = False
        self.summary_started_at = 0
        self.summary_identity = {}
        self.gpus, self.apps = [], []
        self.gpu_error = None
        for identifier, job in self.jobs.items():
            record = self.state["jobs"].setdefault(identifier, {"status": "PENDING", "attempt": 0,
                                                              "oom_retry_count": 0, "peak_memory_mib": 0})
            old_signature = record.get("signature")
            if old_signature and old_signature != signature(job):
                raise ValueError(identifier + ": argv/cwd/run_dir changed; create a new experiment ID")
            record["signature"] = signature(job)
            record["run_dir"] = job["run_dir"]
            record["required_memory_estimate"] = max(record.get("required_memory_estimate", 0), job["required_memory_mib"])
        if any(record.get("status") in ACTIVE and identifier not in self.jobs
               for identifier, record in self.state["jobs"].items()):
            raise ValueError("Config omitted an active job; retain it until completed")
        self.state["config_path"] = config["config_path"]

    def event(self, kind, identifier=None, **fields):
        event = {"time": time.time(), "event": kind, **fields}
        if identifier:
            event["id"] = identifier
        self.state["history"].append(event)
        print(json.dumps(event, sort_keys=True), flush=True)

    def persist(self):
        self.state["updated_at"] = time.time()
        self.state["gpu_snapshot"] = self.gpus
        self.state["gpu_error"] = self.gpu_error
        self.state["counts"] = self.counts()
        atomic_json(self.path, self.state)
        for identifier in self.jobs:
            # Wrapper owns status.json/metrics.json; scheduler metadata has a separate writer.
            atomic_json(Path(self.jobs[identifier]["run_dir"]) / "queue_status.json", self.state["jobs"][identifier])

    def counts(self):
        result = {"pending": 0, "running": 0, "succeeded": 0, "retrying": 0, "failed": 0, "stopped": 0}
        for identifier in self.jobs:
            status = self.state["jobs"][identifier]["status"]
            key = ("running" if status in ACTIVE else "succeeded" if status == "SUCCESS" else
                   "pending" if status == "PENDING" else "retrying" if status in RETRY else
                   "stopped" if status == "STOPPED" else "failed")
            result[key] += 1
        return result

    def refresh_gpu(self):
        try:
            self.gpus, self.apps = gpu_snapshot(self.config)
            self.gpu_error = None
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.gpus, self.apps = [], []
            message = str(error)
            if self.gpu_error != message:
                self.event("GPU_QUERY_FAILED", error=message)
            self.gpu_error = message

    def registered(self):
        shared = Path(self.config["shared_lock_dir"])
        path = shared / "trustgtr_gpu_reservations.json"
        rows = read_json(path, {})
        live = {key: record for key, record in rows.items() if same_process(record) or group_members(record)}
        return path, live

    def update_registry(self, identifier=None, remove=False):
        # Caller holds the shared launch lock. Registry also protects multiple queue files.
        path, rows = self.registered()
        prefix = str(self.runtime) + ":"
        if identifier:
            if remove:
                rows.pop(prefix + identifier, None)
            else:
                rows[prefix + identifier] = dict(self.state["jobs"][identifier])
        atomic_json(path, rows)

    def allocation(self, record):
        pids = {member["pid"] for member in group_members(record)}
        return sum(app["used_mib"] for app in self.apps
                   if app["uuid"] == record.get("gpu_uuid") and int(app["pid"]) in pids)

    def monitor(self):
        now = time.time()
        for identifier in self.jobs:
            record = self.state["jobs"][identifier]
            if record["status"] not in ACTIVE:
                continue
            allocation = self.allocation(record)
            record["current_memory_mib"] = allocation
            record["peak_memory_mib"] = max(record.get("peak_memory_mib", 0), allocation)
            process = self.processes.get(identifier)
            returncode = process.poll() if process else None
            alive = same_process(record)
            if record["status"] == "RUNNING" and (returncode is not None or not alive):
                outcome = read_json(record["exit_file"], {})
                rc = outcome.get("returncode", returncode)
                record["exit_code"] = rc
                record["finished_at"] = outcome.get("finished_at", now)
                record["outcome"] = "STOPPED" if self.stopping else classify(rc, tail(record["log_path"]))
                record["status"] = "CLEANING"
                record["cleanup_deadline"] = now + float(self.config.get("cleanup_seconds", 10))
                signal_group(record, signal.SIGTERM)
            if record["status"] == "CLEANING":
                members = group_members(record)
                if members and now >= record["cleanup_deadline"]:
                    signal_group(record, signal.SIGKILL)
                if not members:
                    if process:
                        process.poll()  # Reap our worker without blocking other jobs.
                    self.finalize(identifier)
            elif self.stopping:
                if not record.get("stop_deadline"):
                    record["stop_deadline"] = now + float(self.config.get("stop_grace_seconds", 20))
                    signal_group(record, signal.SIGTERM)
                elif now >= record["stop_deadline"]:
                    signal_group(record, signal.SIGKILL)

    def finalize(self, identifier):
        record = self.state["jobs"][identifier]
        outcome = record.pop("outcome")
        now = time.time()
        record["last_error_excerpt"] = tail(record["log_path"], 4000) if outcome != "SUCCESS" else ""
        if outcome == "OOM":
            count = record.get("oom_retry_count", 0) + 1
            record["oom_retry_count"] = count
            record["last_oom_time"] = now
            gpu = next((gpu for gpu in self.gpus if gpu["uuid"] == record["gpu_uuid"]), {})
            # Compare the next attempt with memory available at the failed
            # launch, rather than the free memory after its CUDA context died.
            record["last_oom_free_mib"] = record["start_free_mib"]
            record["last_oom_post_cleanup_free_mib"] = gpu.get("free_mib")
            record["last_oom_gpu_total_mib"] = record["gpu_total_mib"]
            old_estimate = record["required_memory_estimate"]
            peak = record.get("peak_memory_mib", 0)
            record["required_memory_estimate"] = max(old_estimate * 1.2, peak * 1.1,
                                                       256 if not old_estimate and not peak else 0)
            delay = min(float(self.config.get("max_retry_seconds", 3600)),
                        float(self.config.get("base_retry_seconds", 60)) * 2 ** min(count - 1, 16))
            record["next_eligible_time"] = now + delay
            record["status"] = "WAITING_RESOURCE" if count >= 3 else "OOM"
        else:
            record["status"] = outcome
            record["required_memory_estimate"] = max(record["required_memory_estimate"], record.get("peak_memory_mib", 0))
        record["last_attempt_pid"] = record.get("pid")
        self.event("FINISHED", identifier, status=record["status"], exit_code=record["exit_code"],
                   peak_memory_mib=record.get("peak_memory_mib", 0))
        with lock(Path(self.config["shared_lock_dir"]) / "trustgtr_gpu_launch.lock"):
            self.update_registry(identifier, remove=True)
        self.processes.pop(identifier, None)
        self.summary_dirty = True

    def summarize(self):
        """Coalesce result updates without waiting on report generation in the GPU loop."""
        if self.summary_process is not None:
            if self.summary_process.poll() is None:
                if time.time() - self.summary_started_at > float(self.config.get("summary_timeout_seconds", 60)):
                    signal_group(self.summary_identity, signal.SIGKILL)
                return
            self.state["summary_exit_code"] = self.summary_process.returncode
            self.summary_process = None
        argv = self.config.get("on_change_argv")
        if not argv:
            self.summary_dirty = False
            return
        if self.summary_dirty and argv:
            self.summary_dirty = False
            try:
                with (self.runtime / "summary.log").open("a") as output:
                    self.summary_process = subprocess.Popen(argv, cwd=self.config.get("on_change_cwd", str(PROJECT)),
                                                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                self.summary_started_at = time.time()
                self.summary_identity = proc_info(self.summary_process.pid) or {}
            except OSError as error:
                self.state["summary_error"] = str(error)
                self.summary_process = None

    def refresh_estimate(self, identifier):
        job, record = self.jobs[identifier], self.state["jobs"][identifier]
        source = job.get("memory_estimate_from")
        if not source or self.state["jobs"][source]["status"] != "SUCCESS":
            return
        resource = read_json(Path(self.jobs[source]["run_dir"]) / "resource_usage.json", {})
        values = [self.state["jobs"][source].get("peak_memory_mib", 0)]
        for key in ("gpu_peak_reserved_mib", "gpu_peak_allocated_mib", "peak_memory_mib",
                    "torch_peak_reserved_mib", "torch_peak_allocated_mib", "peak_gpu_memory_mib"):
            value = resource.get(key, 0)
            if isinstance(value, (float, int)):
                values.append(value)
        estimate = max(values)
        if estimate > record["required_memory_estimate"]:
            record["required_memory_estimate"] = estimate
            record["memory_estimate_source"] = source
            self.event("MEMORY_ESTIMATE", identifier, source=source, required_memory_estimate=estimate)

    def eligible(self, identifier):
        job = self.jobs[identifier]
        record = self.state["jobs"][identifier]
        if record["status"] not in RETRY or time.time() < record.get("next_eligible_time", 0):
            return False
        deps = [self.state["jobs"][dep] for dep in job["depends_on"]]
        if not all(dep["status"] == "SUCCESS" for dep in deps):
            record["wait_reason"] = "DEPENDENCY"
            if any(dep["status"] in TERMINAL - {"SUCCESS", "STOPPED"} for dep in deps):
                record["status"] = "CODE_ERROR"
                record["last_error_excerpt"] = "Prerequisite failed; repair and pass smoke before requeueing."
                self.event("DEPENDENCY_FAILED", identifier, dependencies=job["depends_on"])
                self.summary_dirty = True
            return False
        smoke_ok = job.get("is_smoke") or job.get("smoke_passed") is True or any(
            self.jobs[dep].get("is_smoke") and self.state["jobs"][dep]["status"] == "SUCCESS"
            for dep in job["depends_on"])
        if not smoke_ok:
            record["wait_reason"] = "SMOKE_REQUIRED"
            return False
        self.refresh_estimate(identifier)
        if record.get("oom_retry_count", 0) >= 3 and time.time() < record.get("next_resource_check", 0):
            return False
        record["next_resource_check"] = time.time() + float(self.config.get("resource_wait_seconds", 30))
        return True

    def choose_gpu(self, identifier, rows):
        job, record = self.jobs[identifier], self.state["jobs"][identifier]
        required = record["required_memory_estimate"]
        allowed_uuid = job.get("gpu_uuid")
        options = []
        for gpu in self.gpus:
            if allowed_uuid and gpu["uuid"] != allowed_uuid:
                continue
            active = [other for other in rows.values() if other.get("gpu_uuid") == gpu["uuid"]]
            if (not required and active) or any(not other.get("required_memory_estimate") for other in active):
                continue  # Unknown peak never steals a running job's future allocation.
            reserved = sum(max(0, other.get("required_memory_estimate", 0) - self.allocation(other)) for other in active)
            available = float(gpu["free_mib"]) - reserved
            if available < required + margin_mib(gpu):
                continue
            if record.get("oom_retry_count"):
                improvement = min(256, max(32, gpu["total_mib"] * 0.01))
                more_free = gpu["free_mib"] > record.get("last_oom_free_mib", 0) + improvement
                larger_gpu = gpu["total_mib"] > record.get("last_oom_gpu_total_mib", 0)
                if not (more_free or larger_gpu):
                    continue
            options.append((available, gpu))
        return max(options, key=lambda option: option[0])[1] if options else None

    def launch(self, identifier):
        # Atomic fresh snapshot -> GPU selection -> reservation -> status -> spawn -> PID.
        with lock(Path(self.config["shared_lock_dir"]) / "trustgtr_gpu_launch.lock"):
            self.refresh_gpu()
            _, rows = self.registered()
            if len(rows) >= self.config["max_jobs"]:
                return False
            gpu = self.choose_gpu(identifier, rows)
            record, job = self.state["jobs"][identifier], self.jobs[identifier]
            if not gpu:
                record["wait_reason"] = "GPU_MEMORY_OR_OOM_IMPROVEMENT"
                record["status"] = "WAITING_RESOURCE"
                return False
            directory = Path(job["run_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            attempt = record["attempt"] + 1
            launch_path = directory / ("queue_attempt_%d_launch.json" % attempt)
            exit_path = directory / ("queue_attempt_%d_exit.json" % attempt)
            log_path = directory / ("train_attempt_%d.log" % attempt)
            if launch_path.exists() or exit_path.exists() or log_path.exists():
                raise ValueError("Attempt artifacts already exist: " + str(launch_path))
            environment = dict(job.get("env", {}))
            environment.update({"CUDA_VISIBLE_DEVICES": gpu["uuid"], "TRUSTGTR_RUN_DIR": job["run_dir"],
                                "TRUSTGTR_ATTEMPT": str(attempt), "PYTHONUNBUFFERED": "1"})
            launch = {"id": identifier, "argv": job["argv"], "cwd": job["cwd"], "env": environment,
                      "gpu": gpu, "exit_file": str(exit_path), "attempt": attempt,
                      "identity_file": str(directory / ("queue_attempt_%d_pid.json" % attempt))}
            atomic_json(launch_path, launch)
            old = dict(record)
            record.update({"status": "RUNNING", "attempt": attempt, "gpu_uuid": gpu["uuid"],
                           "gpu_total_mib": gpu["total_mib"], "start_free_mib": gpu["free_mib"],
                           "started_at": time.time(), "log_path": str(log_path), "exit_file": str(exit_path),
                           "current_memory_mib": 0, "wait_reason": None, "identity_file": launch["identity_file"]})
            self.persist()
            try:
                with log_path.open("x") as output:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker",
                                                "--launch-file", str(launch_path)], stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True, cwd=job["cwd"])
            except OSError as error:
                record.clear()
                record.update(old)
                record["status"] = "CODE_ERROR"
                record["launch_error"] = str(error)
                self.event("LAUNCH_FAILED", identifier, error=str(error))
                self.persist()
                self.update_registry(identifier, remove=True)
                return False
            identity = proc_info(process.pid)
            record.update({"pid": process.pid, "start_ticks": identity["start_ticks"] if identity else -1})
            record.pop("stop_deadline", None)
            self.processes[identifier] = process
            self.update_registry(identifier)
            self.event("STARTED", identifier, pid=process.pid, gpu=gpu, attempt=attempt,
                       required_memory_estimate=record["required_memory_estimate"])
            self.persist()
            return True

    def daemon(self):
        self.state["daemon"] = proc_info(os.getpid())
        self.state["daemon"]["started_at"] = time.time()
        self.state["daemon"]["status"] = "RUNNING"
        atomic_json(self.runtime / "queue.pid.json", self.state["daemon"])
        (self.runtime / "queue.pid").write_text(str(os.getpid()) + "\n")
        for identifier in self.jobs:
            record = self.state["jobs"][identifier]
            if record["status"] in ACTIVE and not same_process(record) and record.get("identity_file"):
                receipt = read_json(record["identity_file"], {})
                if same_process(receipt):
                    record.update({"pid": receipt["pid"], "start_ticks": receipt["start_ticks"]})
            if record["status"] in ACTIVE and same_process(record):
                self.event("ADOPTED", identifier, pid=record["pid"])
        def stop_handler(signum, frame):
            self.stopping = True
        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        self.event("DAEMON_STARTED", pid=os.getpid())
        self.persist()
        try:
            while True:
                started_loop = time.monotonic()
                self.refresh_gpu()
                self.monitor()
                if not self.stopping:
                    for identifier in self.jobs:
                        if sum(self.state["jobs"][name]["status"] in ACTIVE for name in self.jobs) >= self.config["max_jobs"]:
                            break
                        if self.eligible(identifier):
                            self.launch(identifier)
                self.persist()
                self.summarize()  # Reports observe the states persisted in this tick.
                active = any(self.state["jobs"][name]["status"] in ACTIVE for name in self.jobs)
                queued = any(self.state["jobs"][name]["status"] in RETRY for name in self.jobs)
                summary_active = self.summary_process is not None and self.summary_process.poll() is None
                if self.stopping and not active and not summary_active:
                    self.state["daemon"]["status"] = "STOPPED"
                    break
                if not active and not queued and not summary_active and not self.summary_dirty:
                    self.state["daemon"]["status"] = "COMPLETED"
                    break
                # Only the daemon's short tick waits; no wait() on active training jobs.
                time.sleep(max(0.01, self.config["poll_seconds"] - (time.monotonic() - started_loop)))
        except BaseException:
            # Unexpected daemon failure preserves training sessions for adoption.
            self.state["daemon"]["status"] = "CRASHED"
            raise
        finally:
            self.state["daemon"]["finished_at"] = time.time()
            self.persist()
            atomic_json(self.runtime / "queue.pid.json", self.state["daemon"])
        self.event("DAEMON_FINISHED", counts=self.counts())
        return 0 if not self.counts()["failed"] else 2


def worker(path):
    launch = read_json(path)
    atomic_json(launch["identity_file"], proc_info(os.getpid()))
    environment = os.environ.copy()
    environment.update(launch["env"])
    print(json.dumps({"event": "WORKER_START", "id": launch["id"], "attempt": launch["attempt"],
                      "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"], "gpu": launch["gpu"],
                      "argv": launch["argv"], "cwd": launch["cwd"]}), flush=True)
    try:
        command = subprocess.Popen(launch["argv"], cwd=launch["cwd"], env=environment)
        returncode = command.wait()
    except OSError as error:
        print("Queue launch failure: " + repr(error), flush=True)
        returncode = 127
    atomic_json(launch["exit_file"], {"returncode": returncode, "finished_at": time.time(),
                                     "worker_pid": os.getpid(), "attempt": launch["attempt"]})
    return returncode if returncode >= 0 else 128 - returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "resume", "status", "stop", "_worker"))
    parser.add_argument("--config", default=str(PROJECT / "configs" / "queue.json"))
    parser.add_argument("--launch-file")
    parser.add_argument("--retry-failed", action="append", default=[], metavar="ID",
                        help="After repair and fresh smoke only; CONFIG_ERROR needs a new ID")
    args = parser.parse_args()
    if args.command == "_worker":
        return worker(args.launch_file)
    config = load_config(args.config)
    runtime = Path(config["runtime_dir"])
    if args.command in ("status", "stop"):
        state = read_json(runtime / "queue_state.json", {})
        daemon = read_json(runtime / "queue.pid.json", {})
        if args.command == "status":
            print(json.dumps({"daemon_alive": same_process(daemon), "daemon": daemon,
                              "counts": state.get("counts", {}), "gpu_error": state.get("gpu_error"),
                              "jobs": {key: {field: record.get(field) for field in
                                             ("status", "attempt", "pid", "gpu_uuid", "wait_reason", "oom_retry_count",
                                              "next_eligible_time", "required_memory_estimate")}
                                       for key, record in state.get("jobs", {}).items()}}, indent=2))
        elif same_process(daemon):
            os.kill(daemon["pid"], signal.SIGTERM)
            print("Safe stop requested; running groups will terminate within configured grace.")
        else:
            print("Daemon is absent; run resume to adopt and stop live jobs safely.")
        return 0
    try:
        with lock(runtime / "queue_daemon.lock", nonblocking=True):
            queue = Queue(config)
            if args.command == "resume":
                for identifier in queue.jobs:
                    record = queue.state["jobs"][identifier]
                    if record["status"] == "STOPPED":
                        record["status"] = "PENDING"
                for identifier in args.retry_failed:
                    if identifier not in queue.jobs:
                        raise ValueError("Unknown retry ID: " + identifier)
                    record = queue.state["jobs"][identifier]
                    if record["status"] == "CONFIG_ERROR":
                        raise ValueError("CONFIG_ERROR requires a new experiment ID")
                    if record["status"] in TERMINAL - {"SUCCESS"}:
                        record["status"] = "PENDING"
                        queue.event("MANUAL_REQUEUE", identifier, reason="repair and smoke asserted by resume --retry-failed")
            return queue.daemon()
    except BlockingIOError:
        print("Queue daemon already running; no duplicate started.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print("Queue error: " + str(error), file=sys.stderr)
        sys.exit(2)
