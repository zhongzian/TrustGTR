#!/usr/bin/env python3
"""CPU-only queue integration tests. Fake GPU files never allocate CUDA memory."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "experiment_queue.py"
spec = importlib.util.spec_from_file_location("trustgtr_queue", QUEUE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OUTPUT = None


def event(path, kind, identifier):
    with Path(path).open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps({"event": kind, "id": identifier, "attempt": os.environ.get("TRUSTGTR_ATTEMPT"),
                                 "time": time.time(), "pid": os.getpid(),
                                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\n")
        stream.flush()


def fake_task(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--seconds", type=float, default=0.2)
    parser.add_argument("--mode", default="success")
    args = parser.parse_args(argv)
    event(args.events, "start", args.id)
    time.sleep(args.seconds)
    attempt = int(os.environ.get("TRUSTGTR_ATTEMPT", "1"))
    result = "success"
    if args.mode == "oom_once" and attempt == 1:
        print("torch.OutOfMemoryError: CUDA out of memory", flush=True)
        result = "oom"
    if args.mode == "code_error":
        print("Traceback (most recent call last):\nRuntimeError: deterministic simulated code failure", flush=True)
        result = "code_error"
    directory = Path(os.environ["TRUSTGTR_RUN_DIR"])
    if result == "success":
        module.atomic_json(directory / "resource_usage.json", {"gpu_peak_reserved_mib": 700})
        module.atomic_json(directory / "metrics.json", {"simulation": True, "MSE": None})
        module.atomic_json(directory / "status.json", {"status": "SUCCESS", "exit_code": 0})
    event(args.events, "finish", args.id)
    return 0 if result == "success" else 1


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.directory = OUTPUT / self._testMethodName
        self.directory.mkdir(parents=True)
        self.config_path = self.directory / "queue.json"
        self.gpus_path = self.directory / "gpus.json"
        self.events_path = self.directory / "events.jsonl"
        self.events_path.touch()
        self.processes = []
        self.logs = []
        self.set_free(2000)
        self.config = {"max_jobs": 3, "poll_seconds": 0.05, "oom_cooldown_seconds": 0.15,
                       "resource_wait_seconds": 0.05, "stop_grace_seconds": 0.2, "cleanup_seconds": 0.2,
                       "runtime_dir": str(self.directory / "runtime"),
                       "shared_lock_dir": str(self.directory / "shared"),
                       "fake_gpu_file": str(self.gpus_path), "jobs": []}

    def tearDown(self):
        # Only groups created by these tests are touched, and their sessions are isolated.
        state = self.state()
        for record in state.get("jobs", {}).values():
            module.signal_group(record, signal.SIGKILL)
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        for stream in self.logs:
            stream.close()

    def set_free(self, free):
        module.atomic_json(self.gpus_path, [{"index": 9, "uuid": "GPU-FAKE-UUID", "name": "CPU simulation",
                                             "total_mib": 10240, "used_mib": 10240 - free, "free_mib": free}])

    def add_job(self, identifier, seconds=0.2, mode="success", **fields):
        job = {"id": identifier, "argv": [sys.executable, str(Path(__file__).resolve()), "_task",
                                            "--events", str(self.events_path), "--id", identifier,
                                            "--seconds", str(seconds), "--mode", mode],
               "cwd": str(ROOT), "run_dir": str(self.directory / "runs" / identifier),
               "required_memory_mib": 700, "smoke_passed": True}
        job.update(fields)
        self.config["jobs"].append(job)
        return job

    def start(self, command="start", extra=None):
        module.atomic_json(self.config_path, self.config)
        stream = (self.directory / ("daemon_%d.log" % len(self.processes))).open("w")
        self.logs.append(stream)
        process = subprocess.Popen([sys.executable, str(QUEUE), command, "--config", str(self.config_path),
                                    *(extra or [])], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)
        return process

    def state(self):
        return module.read_json(self.directory / "runtime" / "queue_state.json", {})

    def wait_for(self, predicate, seconds=10):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.03)
        self.fail("Timed out; state=" + json.dumps(self.state())[:2500])

    def run_command(self, command):
        return subprocess.run([sys.executable, str(QUEUE), command, "--config", str(self.config_path)],
                              capture_output=True, text=True, timeout=5)

    def test_dynamic_fill_max3_no_duplicate_and_failure(self):
        self.set_free(1100)  # 700+307 margin fits one, not two reservations.
        self.add_job("a", seconds=0.7)
        self.add_job("b", seconds=0.5)
        self.add_job("c", seconds=0.5)
        self.add_job("d", seconds=0.12)
        self.add_job("broken", seconds=0.12, mode="code_error")
        process = self.start()
        self.wait_for(lambda: self.state().get("counts", {}).get("running") == 1)
        time.sleep(0.1)
        self.assertEqual(self.state()["counts"]["running"], 1)
        self.set_free(5000)
        self.wait_for(lambda: self.state().get("counts", {}).get("running") == 3)
        self.assertEqual(process.wait(timeout=10), 2)
        state = self.state()
        self.assertEqual(state["counts"]["succeeded"], 4)
        self.assertEqual(state["jobs"]["broken"]["status"], "CODE_ERROR")
        self.assertEqual(state["jobs"]["broken"]["attempt"], 1)
        active, maximum = set(), 0
        for row in self.events_path.read_text().splitlines():
            item = json.loads(row)
            if item["event"] == "start":
                self.assertNotIn(item["id"], active)
                self.assertEqual(item["gpu"], "GPU-FAKE-UUID")
                active.add(item["id"])
                maximum = max(maximum, len(active))
            else:
                active.remove(item["id"])
        self.assertEqual(maximum, 3)
        before = self.events_path.read_text()
        again = self.start("resume")
        self.assertEqual(again.wait(timeout=5), 2)
        self.assertEqual(before, self.events_path.read_text())
        self.assertEqual(self.run_command("status").returncode, 0)

    def test_oom_backoff_and_resource_improvement(self):
        self.add_job("oom", seconds=0.1, mode="oom_once")
        process = self.start()
        self.wait_for(lambda: self.state().get("jobs", {}).get("oom", {}).get("oom_retry_count") == 1)
        record = self.state()["jobs"]["oom"]
        self.assertEqual(record["attempt"], 1)
        self.assertGreater(record["next_eligible_time"], record["last_oom_time"])
        time.sleep(0.4)
        self.assertEqual(self.state()["jobs"]["oom"]["attempt"], 1)
        self.set_free(4000)
        self.assertEqual(process.wait(timeout=10), 0)
        record = self.state()["jobs"]["oom"]
        self.assertEqual(record["status"], "SUCCESS")
        self.assertEqual(record["attempt"], 2)
        self.assertGreater(record["required_memory_estimate"], 700)
        self.assertTrue((Path(record["run_dir"]) / "train_attempt_1.log").exists())
        self.assertTrue((Path(record["run_dir"]) / "train_attempt_2.log").exists())

    def test_dependencies_and_smoke_peak(self):
        self.set_free(5000)
        self.add_job("smoke", is_smoke=True, smoke_passed=False)
        self.add_job("formal", smoke_passed=False, required_memory_mib=0,
                     depends_on=["smoke"], memory_estimate_from="smoke")
        process = self.start()
        self.assertEqual(process.wait(timeout=10), 0)
        state = self.state()
        self.assertEqual(state["jobs"]["formal"]["required_memory_estimate"], 700)
        self.assertGreaterEqual(state["jobs"]["formal"]["started_at"], state["jobs"]["smoke"]["finished_at"])
        # The wrapper's own status.json remains untouched by the queue.
        status = module.read_json(Path(state["jobs"]["formal"]["run_dir"]) / "status.json")
        self.assertEqual(status, {"status": "SUCCESS", "exit_code": 0})

    def test_duplicate_daemon_and_crash_adoption(self):
        self.add_job("long", seconds=1.4)
        process = self.start()
        self.wait_for(lambda: self.state().get("jobs", {}).get("long", {}).get("pid"))
        duplicate = self.start()
        self.assertEqual(duplicate.wait(timeout=5), 3)
        original = self.state()["jobs"]["long"]["pid"]
        process.kill()
        process.wait(timeout=5)
        resumed = self.start("resume")
        self.assertEqual(resumed.wait(timeout=10), 0)
        record = self.state()["jobs"]["long"]
        self.assertEqual(record["pid"], original)
        self.assertEqual(record["attempt"], 1)
        self.assertTrue(any(item["event"] == "ADOPTED" for item in self.state()["history"]))

    def test_crash_adoption_restores_missing_reservation(self):
        self.set_free(1100)  # Only one 700 MiB task fits with the safety margin.
        self.add_job("long", seconds=1.5)
        process = self.start()
        self.wait_for(lambda: self.state().get("jobs", {}).get("long", {}).get("identity_file"))
        record = self.state()["jobs"]["long"]
        receipt = self.wait_for(lambda: module.read_json(record["identity_file"], {}).get("pid"))
        process.kill()
        process.wait(timeout=5)
        # Reproduce a crash after Popen but before PID/registry persistence.
        state = self.state()
        state["jobs"]["long"].pop("pid", None)
        state["jobs"]["long"].pop("start_ticks", None)
        module.atomic_json(self.directory / "runtime" / "queue_state.json", state)
        registry = self.directory / "shared" / "trustgtr_gpu_reservations.json"
        module.atomic_json(registry, {})
        self.add_job("pending", seconds=0.1)
        resumed = self.start("resume")
        rows = self.wait_for(lambda: module.read_json(registry, {}))
        self.assertTrue(any(row.get("pid") == receipt for row in rows.values()))
        self.assertEqual(resumed.wait(timeout=10), 0)
        state = self.state()
        self.assertEqual(state["jobs"]["long"]["pid"], receipt)
        self.assertEqual(state["jobs"]["long"]["attempt"], 1)
        self.assertGreaterEqual(state["jobs"]["pending"]["started_at"],
                                state["jobs"]["long"]["finished_at"])

    def test_cooldown_accepts_legacy_key_without_changing_value(self):
        self.config.pop("oom_cooldown_seconds")
        self.config["base_retry_seconds"] = 0.45
        module.atomic_json(self.config_path, self.config)
        self.assertEqual(module.load_config(self.config_path)["oom_cooldown_seconds"], 0.45)

    def test_safe_stop_resume(self):
        self.add_job("long", seconds=1.0)
        process = self.start()
        self.wait_for(lambda: self.state().get("jobs", {}).get("long", {}).get("pid"))
        self.assertEqual(self.run_command("stop").returncode, 0)
        self.assertEqual(process.wait(timeout=10), 0)
        record = self.state()["jobs"]["long"]
        self.assertEqual(record["status"], "STOPPED")
        self.assertFalse(module.group_members(record))
        resumed = self.start("resume")
        self.assertEqual(resumed.wait(timeout=10), 0)
        self.assertEqual(self.state()["jobs"]["long"]["attempt"], 2)

    def test_no_double_subtraction_of_used_memory(self):
        self.add_job("candidate", required_memory_mib=1700)
        module.atomic_json(self.config_path, self.config)
        queue = module.Queue(module.load_config(self.config_path))
        queue.gpus = [{"index": 9, "uuid": "GPU-FAKE-UUID", "name": "simulation", "total_mib": 10240,
                       "used_mib": 8000, "free_mib": 2240}]
        # Existing peak 2000 with current usage 1800 reserves only 200 more MiB.
        queue.allocation = lambda record: 1800
        rows = {"existing": {"gpu_uuid": "GPU-FAKE-UUID", "required_memory_estimate": 2000}}
        self.assertIsNotNone(queue.choose_gpu("candidate", rows))
        # Unknown peak blocks same-card placement instead of pretending it is zero.
        rows["existing"]["required_memory_estimate"] = 0
        self.assertIsNone(queue.choose_gpu("candidate", rows))

    def test_oom_compares_launch_free_not_cleanup_free(self):
        self.add_job("oom")
        module.atomic_json(self.config_path, self.config)
        queue = module.Queue(module.load_config(self.config_path))
        record = queue.state["jobs"]["oom"]
        logfile = self.directory / "oom.log"
        logfile.write_text("CUDA out of memory\n")
        record.update(outcome="OOM", log_path=str(logfile), gpu_uuid="GPU-FAKE-UUID",
                      gpu_total_mib=10240, start_free_mib=2000, exit_code=1)
        queue.refresh_gpu()
        queue.gpus[0]["free_mib"] = 5000  # Context release is not new external capacity.
        queue.finalize("oom")
        self.assertEqual(record["last_oom_free_mib"], 2000)
        self.assertEqual(record["last_oom_post_cleanup_free_mib"], 5000)
        queue.gpus[0]["free_mib"] = 3000
        self.assertIsNotNone(queue.choose_gpu("oom", {}))

    def test_failed_smoke_stops_dependent_job(self):
        self.add_job("bad_smoke", mode="code_error", is_smoke=True)
        self.add_job("formal", smoke_passed=False, depends_on=["bad_smoke"])
        process = self.start()
        self.assertEqual(process.wait(timeout=10), 2)
        self.assertEqual(self.state()["jobs"]["formal"]["status"], "CODE_ERROR")
        self.assertEqual(self.state()["jobs"]["formal"]["attempt"], 0)


def main():
    global OUTPUT
    if len(sys.argv) > 1 and sys.argv[1] == "_task":
        return fake_task(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "runtime" / ("queue_tests_" + time.strftime("%Y%m%d_%H%M%S"))))
    args = parser.parse_args()
    OUTPUT = Path(args.output_dir).resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QueueTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "elapsed_seconds": time.time() - started,
               "gpu_allocations": 0, "artifact_dir": str(OUTPUT)}
    module.atomic_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
