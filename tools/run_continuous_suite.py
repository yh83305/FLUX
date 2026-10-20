"""Durable model/scene queue with continuous Isaac episodes and bounded recovery."""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
import urllib.error


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(start_ticks=stat[19],
                    argv=Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1])
    except OSError:
        return None


def accepted_rows(folder, count):
    """Only committed CSV rows with a closed video and real guard bindings count."""
    folder = Path(folder)
    try:
        rows = list(csv.DictReader((folder / "metric.csv").open()))
        events = []
        for line in (folder / "offscreen_guard.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                # A crash during the next heartbeat must not erase already
                # committed metrics/videos from preceding episodes.
                continue
        guard = read_json(folder / "offscreen_guard_config.json", {})
        if not guard.get("enabled") or not any(e.get("event") == "bind" for e in events):
            return []
        if not any(e.get("callbacks", 0) > 0 for e in events):
            return []
        camera = read_json(folder / "camera_profile.json", {})
        if not camera:
            return []
        ids = [int(row["episode"]) for row in rows]
        if len(ids) != len(set(ids)) or any(i not in range(count) for i in ids):
            raise RuntimeError(f"Invalid/duplicate episode IDs: {folder}")
        result = []
        for row in rows:
            video = folder / f'fps_{int(row["episode"])}.mp4'
            if (int(row["timing_requests"]) > 0 and video.exists()
                    and video.stat().st_size > 1000
                    and all(math.isfinite(float(row[k]))
                            for k in ("success", "spl", "time_to_goal", "trajectory_length"))):
                result.append(row)
        return result
    except (OSError, ValueError, KeyError):
        return []


def reconcile(folder, count):
    done = {}
    for attempt in sorted(Path(folder).glob("continuous_*")):
        for row in accepted_rows(attempt, count):
            episode = int(row["episode"])
            if episode in done:
                raise RuntimeError(f"Duplicate accepted episode {folder}/{episode}")
            done[episode] = dict(result_dir=str(attempt), episode=episode, metric=row)
    for episode, record in done.items():
        target = Path(folder) / f"episode_{episode:03d}"
        target.mkdir(exist_ok=True)
        if read_json(target / "COMPLETE.json") != record:
            atomic_json(target / "COMPLETE.json", record)
    return done


def next_missing(done, count):
    missing = [i for i in range(count) if i not in done]
    if not missing:
        return count
    first = missing[0]
    if any(i in done for i in range(first, count)):
        raise RuntimeError("Non-contiguous completed IDs; manual audit required")
    return first


def tail(path, size=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, Path(path).stat().st_size - size))
            return stream.read().decode(errors="replace")
    except OSError:
        return ""


class ProgressWatch:
    def __init__(self, now):
        self.started = self.step_changed = self.plan_changed = self.episode_changed = now
        self.last_step = self.last_plan = self.last_episode = None

    def check(self, progress, now, startup=600, stale=180, episode_limit=1800):
        if not progress:
            return "startup_timeout" if now - self.started > startup else None
        step = (progress.get("episode"), progress.get("step"))
        plan = progress.get("plan_revision")
        episode = progress.get("episode")
        if step != self.last_step:
            self.last_step, self.step_changed = step, now
        if plan != self.last_plan:
            self.last_plan, self.plan_changed = plan, now
        if episode != self.last_episode:
            self.last_episode, self.episode_changed = episode, now
        if now - self.step_changed > stale:
            return "simulation_not_advancing"
        if now - self.plan_changed > stale:
            return "planning_not_advancing"
        if now - self.episode_changed > episode_limit:
            return "episode_walltime_limit"
        return None


class Suite:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg["output"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.count = cfg.get("episodes", 100)
        self.retry_limit = cfg.get("max_attempts_per_episode", 3)
        self.stop = False
        self.server = {}
        self.dependency_bad_health = {}

    def status(self, path, state, **fields):
        record = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), state=state, **fields)
        atomic_json(path / "status.json", record)
        print(json.dumps(record), flush=True)

    def healthy(self, model):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{model["port"]}/health', timeout=5) as response:
                health = json.load(response)
            if health.get("algo", health.get("algorithm")) != model["health_algorithm"]:
                return False
            return all(health.get(k) == v for k, v in model.get("required_health", {}).items())
        except urllib.error.HTTPError as exc:
            # Legacy official services have no /health route. An HTTP 404 proves
            # only web-server liveness; episode progress is checked separately.
            return bool(model.get("legacy_health_404_is_liveness")) and exc.code == 404
        except (OSError, ValueError):
            return False

    def start_server(self, name, model, gpu):
        self.stop_server(name)
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", model["port"])) == 0:
                raise RuntimeError(f"Refusing occupied port {model['port']}")
        output = self.root / name
        output.mkdir(exist_ok=True)
        logfile = (output / f"server_{time.time_ns()}.log").open("w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.Popen(model["server_command"], env=env, cwd=output,
                                stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        self.server[name] = (proc, logfile)
        identity = process_identity(proc.pid)
        atomic_json(output / "owned_server.json", dict(
            pid=proc.pid, start_ticks=identity["start_ticks"],
            argv=[x.decode() for x in identity["argv"]]))
        for _ in range(120):
            if self.stop or proc.poll() is not None:
                raise RuntimeError(f"{name} server exited or suite stopped")
            if self.healthy(model):
                return
            time.sleep(2)
        raise RuntimeError(f"{name} server startup timeout")

    def stop_server(self, name):
        item = self.server.pop(name, None)
        if item:
            proc, logfile = item
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            logfile.close()

    def ensure_server(self, name, model, gpu):
        for attempt in range(3):
            try:
                self.start_server(name, model, gpu)
                return
            except Exception as exc:
                self.stop_server(name)
                self.status(self.root / name, "server_recovery", attempt=attempt+1, error=str(exc))
                if self.stop or attempt == 2:
                    raise
                time.sleep(10 * (attempt+1))

    def stop_sim(self, folder):
        pattern = "^/isaac-sim/kit/python/bin/python3 .*--output_dir " + re.escape(str(folder)) + "$"
        for sig in ("TERM", "KILL"):
            subprocess.run(["docker", "exec", self.cfg["container"], "pkill", "-" + sig, "-f", pattern],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
            time.sleep(3)
            found = subprocess.run(["docker", "exec", self.cfg["container"], "pgrep", "-f", pattern],
                                   capture_output=True, timeout=20)
            if found.returncode == 1:
                return
        raise RuntimeError(f"Cannot stop owned simulator: {folder}")

    def gpu_free(self, gpu):
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        selected = None
        for row in rows.splitlines():
            index, uuid, memory = [v.strip() for v in row.split(",")]
            if int(index) == gpu:
                if int(memory) < self.cfg.get("min_free_mb", 18000):
                    return False
                selected = uuid
        if selected is None:
            return False
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        for row in rows.splitlines():
            uuid, pid = [v.strip() for v in row.split(",")]
            if uuid != selected:
                continue
            cmd = subprocess.run(["ps", "-p", pid, "-o", "args="],
                                 capture_output=True, text=True).stdout
            # Previously deployed official model services may remain resident.
            if cmd and not any(marker in cmd for marker in self.cfg.get("allowed_resident_commands", [])):
                return False
        return True

    def run_scene(self, name, model, gpu, scene):
        root = self.root / name / scene
        root.mkdir(parents=True, exist_ok=True)
        failures_path = root / "retries.json"
        failures = read_json(failures_path, {})
        while not self.stop:
            done = reconcile(root, self.count)
            start = next_missing(done, self.count)
            if start == self.count:
                self.status(root, "complete", completed=len(done))
                (root / "COMPLETE").touch()
                return
            if failures.get(str(start), 0) >= self.retry_limit:
                raise RuntimeError(f"{name}/{scene}/{start}: retry circuit open")
            if name not in self.server or not self.healthy(model):
                self.ensure_server(name, model, gpu)
            out = root / f"continuous_{start:03d}_{time.time_ns()}"
            out.mkdir()
            command = [
                "docker", "exec", "-e", f"CUDA_VISIBLE_DEVICES={gpu}", "-e", "PYTHONUNBUFFERED=1",
                "-e", "PYTHONPATH=" + self.cfg["code_root"],
                "-e", "FLUX_ISAAC_EXPERIENCE=/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit",
                "-w", "/workspace/FLUX", self.cfg["container"], "/isaac-sim/python.sh", "-u",
                self.cfg["code_root"] + "/runtime_fix/eval_render_safe.py",
                "--port", str(model["port"]), "--gpu_id", "0",
                "--scene_dir", self.cfg["dataset"] + "/scene_views/" + scene,
                "--scene_index", "0", "--scene_scale", "1", "--num_envs", "1",
                "--episode_start", str(start), "--num_episodes", str(self.count-start),
                "--speed", ".5", "--stop_threshold", "-3", "--output_dir", str(out)]
            atomic_json(out / "command.json", command)
            self.status(root, "running", model=name, scene=scene, gpu=gpu, completed=len(done),
                        next_episode=start, attempt=str(out))
            watch = ProgressWatch(time.monotonic())
            bad_health = 0
            reason = "process_exit"
            with (out / "eval.log").open("w") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None and not self.stop:
                        time.sleep(10)
                        done = reconcile(root, self.count)
                        progress = read_json(out / "progress.json")
                        self.status(root, "running", model=name, scene=scene, gpu=gpu,
                                    completed=len(done), next_episode=next_missing(done, self.count),
                                    progress=progress, attempt=str(out))
                        if len(done) == self.count:
                            reason = "all_artifacts_complete"
                            break
                        text = tail(out / "eval.log")
                        if any(x in text for x in ("Traceback (most recent call last)",
                               "CUDA error", "Nphase_Kernel fail", "fail to launch")):
                            reason = "fatal_log"
                            break
                        if "[INFO] Cleanup complete" in text:
                            reason = "early_cleanup"
                            break
                        bad_health = 0 if self.healthy(model) else bad_health + 1
                        if bad_health >= 3:
                            reason = "model_health_failed"
                            break
                        reason_check = watch.check(
                            progress, time.monotonic(),
                            self.cfg.get("startup_timeout", 600),
                            self.cfg.get("stale_timeout", 180),
                            self.cfg.get("episode_walltime_limit", 1800))
                        if reason_check:
                            reason = reason_check
                            break
                finally:
                    self.stop_sim(out)
                    proc.wait(timeout=30)
            done = reconcile(root, self.count)
            failed = next_missing(done, self.count)
            atomic_json(out / "termination.json", dict(reason=reason, exit_code=proc.returncode,
                                                       completed=len(done), next_episode=failed))
            if self.stop:
                self.status(root, "paused", completed=len(done))
                return
            if failed == self.count:
                continue
            failures[str(failed)] = failures.get(str(failed), 0) + 1
            atomic_json(failures_path, failures)
            self.status(root, "recovering", episode=failed, reason=reason,
                        failures=failures[str(failed)], completed=len(done))
            # Stop hung requests and reset model state before replaying an incomplete episode.
            self.stop_server(name)
            if failures[str(failed)] >= self.retry_limit:
                raise RuntimeError(f"{name}/{scene}/{failed}: {reason}; retry circuit open")
            time.sleep(min(60, 10 * failures[str(failed)]))

    def worker(self, gpu, tasks):
        while not self.stop:
            try:
                name = tasks.get_nowait()
            except queue.Empty:
                return
            output = self.root / name
            output.mkdir(exist_ok=True)
            try:
                while not self.stop and not self.gpu_free(gpu):
                    self.status(output, "waiting_for_gpu", gpu=gpu)
                    time.sleep(20)
                if self.stop:
                    return
                model = self.cfg["models"][name]
                for scene in self.cfg["scenes"]:
                    self.run_scene(name, model, gpu, scene)
                    if self.stop:
                        return
                self.status(output, "complete", episodes=self.count * len(self.cfg["scenes"]))
                (output / "COMPLETE").touch()
            except Exception as exc:
                self.status(output, "blocked", error=str(exc))
                atomic_json(output / "ALERT.json", dict(error=str(exc)))
            finally:
                self.stop_server(name)
                tasks.task_done()

    def check_dependencies(self):
        """Supervise the already-running Office queues without restarting healthy work."""
        for dependency in self.cfg.get("dependencies", []):
            output = Path(dependency["output"])
            if (output / "COMPLETE").exists():
                continue
            name = dependency["name"]
            process_rows = subprocess.check_output(["ps", "-eo", "pid,args"], text=True).splitlines()[1:]
            queues = []
            for row in process_rows:
                parts = shlex.split(row)
                if (len(parts) > 2 and "--output" in parts
                        and parts[parts.index("--output")+1] == str(output)
                        and dependency["queue_script"] in parts
                        and Path(parts[1]).name.startswith("python")):
                    queues.append(int(parts[0]))
            self.dependency_bad_health[name] = (
                0 if self.healthy(dependency) else self.dependency_bad_health.get(name, 0)+1)
            status = read_json(output / "status.json", {})
            stalled = ((output / "status.json").exists()
                       and time.time() - (output / "status.json").stat().st_mtime > 1800)
            reason = ("model_unhealthy" if self.dependency_bad_health[name] >= 3 else
                      "queue_missing" if not queues else
                      "queue_failed" if status.get("state") == "failed" else
                      "episode_walltime_limit" if stalled else None)
            if not reason:
                continue
            retry_path = self.root / f"dependency_{name}_retries.json"
            count = read_json(retry_path, {}).get("count", 0)
            if count >= 3:
                raise RuntimeError(f"Office {name} recovery circuit open: {reason}")
            atomic_json(retry_path, dict(count=count+1, reason=reason, prior_status=status))
            self.status(self.root, "recovering_office", model=name, reason=reason, attempt=count+1)
            for pid in queues:
                os.kill(pid, signal.SIGTERM)
            subprocess.run(["tmux", "kill-session", "-t", dependency["queue_session"]],
                           capture_output=True)
            rows = subprocess.check_output(
                ["docker", "top", self.cfg["container"], "-eo", "pid,args"], text=True)
            active = set()
            for row in rows.splitlines()[1:]:
                parts = shlex.split(row)
                if len(parts) > 2 and parts[1] == "/isaac-sim/kit/python/bin/python3" and "--output_dir" in parts:
                    folder = Path(parts[parts.index("--output_dir")+1])
                    if folder.is_relative_to(output):
                        active.add(folder)
            for folder in active:
                self.stop_sim(folder)
            if reason != "queue_missing" or not self.healthy(dependency):
                pattern = ("^/isaac-sim/kit/python/bin/python3 -u "
                           + re.escape(dependency["server_script"])
                           + r" .*--port " + str(dependency["port"]) + r"( |$)")
                for sig in ("TERM", "KILL"):
                    subprocess.run(["docker", "exec", self.cfg["container"], "pkill", "-"+sig, "-f", pattern],
                                   capture_output=True, timeout=20)
                    time.sleep(2)
                subprocess.run(["tmux", "kill-session", "-t", dependency["server_session"]],
                               capture_output=True)
                shell = shlex.join(dependency["server_command"]) + " >> " + shlex.quote(str(output.parent/f"{name}_server_recovery.log")) + " 2>&1"
                subprocess.run(["tmux", "new-session", "-d", "-s", dependency["server_session"], shell], check=True)
                for _ in range(60):
                    if self.healthy(dependency):
                        break
                    time.sleep(2)
                else:
                    # Next poll will retry; no evaluator against a known dead service.
                    continue
            shell = shlex.join(dependency["queue_command"]) + " >> " + shlex.quote(str(output.parent/f"{name}_continuous_queue.log")) + " 2>&1"
            subprocess.run(["tmux", "new-session", "-d", "-s", dependency["queue_session"], shell], check=True)
            self.dependency_bad_health[name] = 0

    def run(self):
        lock = (self.root / "suite.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        # After a supervisor crash, reclaim only exact recorded simulator paths
        # and server PIDs with matching kernel start times/arguments.
        for name in self.cfg["models"]:
            output = self.root / name
            for folder in output.glob("*/continuous_*"):
                if not (folder / "termination.json").exists():
                    self.stop_sim(folder)
            record = read_json(output / "owned_server.json")
            if record:
                actual = process_identity(record["pid"])
                if (actual and actual["start_ticks"] == record["start_ticks"]
                        and actual["argv"] == [s.encode() for s in record["argv"]]):
                    os.killpg(record["pid"], signal.SIGTERM)
                    time.sleep(3)
                    actual = process_identity(record["pid"])
                    if actual and actual["start_ticks"] == record["start_ticks"]:
                        os.killpg(record["pid"], signal.SIGKILL)
        manifest = read_json(Path(self.cfg["dataset"]) / "manifest.json")
        assert manifest["count_per_scene"] == self.count
        for record in manifest["records"]:
            path = Path(self.cfg["dataset"]) / record["scene"] / f'episode_{record["episode"]}.json'
            assert sha256(path) == record["sha256"], str(path)
        provenance = dict(config=self.cfg, dataset_manifest_sha256=sha256(Path(self.cfg["dataset"])/"manifest.json"),
                          code_commit=subprocess.check_output(
                              ["git", "-C", self.cfg["code_root"], "rev-parse", "HEAD"], text=True).strip())
        provenance["input_sha256"] = {path: sha256(path)
                                      for path in self.cfg.get("provenance_files", [])}
        for path, expected in self.cfg.get("required_sha256", {}).items():
            assert sha256(path) == expected, f"Changed pinned input: {path}"
        existing = read_json(self.root / "manifest.json")
        if existing is not None and existing != provenance:
            raise RuntimeError("Run provenance changed; use a new output directory")
        if existing is None:
            atomic_json(self.root / "manifest.json", provenance)
        while not self.stop:
            pending = [p for p in self.cfg.get("wait_for_complete", []) if not Path(p).exists()]
            if not pending:
                break
            try:
                self.check_dependencies()
            except RuntimeError as exc:
                self.status(self.root, "blocked_dependency", error=str(exc))
                atomic_json(self.root / "ALERT.json", dict(error=str(exc)))
                return
            self.status(self.root, "waiting_for_office", pending=pending)
            time.sleep(20)
        if self.stop:
            self.status(self.root, "paused")
            return
        tasks = queue.Queue()
        for name in self.cfg["models"]:
            if not (self.root / name / "COMPLETE").exists():
                tasks.put(name)
        self.status(self.root, "running", planned=len(self.cfg["models"]) * len(self.cfg["scenes"]) * self.count)
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cfg["gpus"])) as pool:
            list(pool.map(lambda gpu: self.worker(gpu, tasks), self.cfg["gpus"]))
        complete = all((self.root / name / "COMPLETE").exists() for name in self.cfg["models"])
        self.status(self.root, "complete" if complete else "paused" if self.stop else "blocked")
        if complete:
            (self.root / "COMPLETE").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    Suite(json.loads(args.config.read_text())).run()
