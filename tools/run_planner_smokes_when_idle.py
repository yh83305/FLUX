"""Run four baseline smoke episodes only after the current suite frees its GPUs."""
import argparse
import concurrent.futures
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--config", type=Path, required=True)
a = p.parse_args()
cfg = json.loads(a.config.read_text())
root = Path(cfg["output"])
root.mkdir(parents=True, exist_ok=True)
lock = (root/"smoke.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (root/"CLOSED_LOOP_COMPLETE").exists():
    raise SystemExit("Smoke already completed")


def state(stage, **fields):
    obj = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), stage=stage, **fields)
    (root/"closedloop_status.json").write_text(json.dumps(obj, indent=2))
    print(json.dumps(obj), flush=True)


def ready():
    current = Path(cfg["wait_for_run"])
    if not all((current/m/"COMPLETE").exists() for m in ("navdp", "flux")):
        return False
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    free = {int(line.split(",")[0]): int(line.split(",")[1]) for line in usage.splitlines()}
    if any(free.get(m["gpu"], 0) < 18000 for m in cfg["models"].values()):
        return False
    # Do not start a second simulator alongside another user's workload.
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    for pid in rows.splitlines():
        cmd = subprocess.run(["ps", "-p", pid.strip(), "-o", "args="],
                             text=True, capture_output=True).stdout
        if cmd and not ("/baselines/" in cmd and
                        ("--port 9010" in cmd or "--port 9011" in cmd)):
            return False
    return True


def stop_sim(folder):
    pattern = "^/isaac-sim/kit/python/bin/python3 .*--output_dir "+re.escape(str(folder))+"$"
    for sig in ("TERM", "KILL"):
        subprocess.run(["docker", "exec", cfg["container"], "pkill", "-"+sig, "-f", pattern],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        found = subprocess.run(["docker", "exec", cfg["container"], "pgrep", "-f", pattern],
                               stdout=subprocess.DEVNULL)
        if found.returncode == 1:
            return
    raise RuntimeError("Simulator did not stop: "+str(folder))


gpu_locks = {m["gpu"]: threading.Lock() for m in cfg["models"].values()}


def worker(name, m):
    # More models than GPUs must never create concurrent simulators on one GPU.
    with gpu_locks[m["gpu"]]:
        return worker_on_gpu(name, m)


def worker_on_gpu(name, m):
    output = root/name/"closedloop"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(m["gpu"]),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    results = []
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", m["port"])) == 0:
            raise RuntimeError(f'{name}: port {m["port"]} already occupied')
    with (output/"server.log").open("w") as log:
        server = subprocess.Popen(m["server_command"], cwd=output, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError(name+" server exited")
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{m["port"]}/health', timeout=2) as response:
                        health = json.load(response)
                        actual = health.get("algo", health.get("algorithm"))
                        assert actual == m.get("health_algorithm", name), (name, actual)
                        for key, expected in m.get("required_health", {}).items():
                            assert health.get(key) == expected, (key, health.get(key), expected)
                    break
                except OSError:
                    time.sleep(2)
            else:
                raise RuntimeError(name+" server startup timeout")
            for scene in cfg.get("scenes", ["Hospital", "Office"]):
                folder = output/scene
                folder.mkdir(exist_ok=False)
                cmd = ["docker", "exec", "-e", f'CUDA_VISIBLE_DEVICES={m["gpu"]}',
                       "-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONPATH="+cfg["code_root"],
                       "-e", "FLUX_ISAAC_EXPERIENCE=/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit",
                       "-w", "/workspace/FLUX", cfg["container"], "/isaac-sim/python.sh", "-u",
                       cfg["code_root"]+"/runtime_fix/eval_render_safe.py",
                       "--port", str(m["port"]), "--gpu_id", "0", "--scene_dir",
                       cfg["dataset"]+"/scene_views/"+scene, "--scene_index", "0",
                       "--num_envs", "1", "--num_episodes", "1", "--episode_start", "0",
                       "--speed", ".5", "--stop_threshold", "-3.0", "--output_dir", str(folder)]
                with (folder/"eval.log").open("w") as eval_log:
                    proc = subprocess.Popen(cmd, stdout=eval_log, stderr=subprocess.STDOUT)
                    started = time.monotonic()
                    while proc.poll() is None:
                        time.sleep(5)
                        text = (folder/"eval.log").read_text(errors="replace")
                        heartbeat = folder/"offscreen_guard.jsonl"
                        stalled = heartbeat.exists() and time.time()-heartbeat.stat().st_mtime > 180
                        if ("[INFO] Cleanup complete" in text or "fail to launch" in text
                                or "Traceback (most recent call last)" in text
                                or stalled or time.monotonic()-started > 900):
                            break
                    stop_sim(folder)
                    proc.wait(timeout=20)
                rows = list(csv.DictReader((folder/"metric.csv").open()))
                assert len(rows) == 1 and int(rows[0]["episode"]) == 0
                assert int(rows[0]["timing_requests"]) > 0
                assert (folder/"fps_0.mp4").stat().st_size > 1000
                assert "[INFO] Cleanup complete" in (folder/"eval.log").read_text(errors="replace")
                events = [json.loads(x) for x in (folder/"offscreen_guard.jsonl").read_text().splitlines()]
                assert any(x.get("callbacks", 0) > 0 for x in events)
                results.append(dict(scene=scene, metric=rows[0], passed=True))
                (output/"results.json").write_text(json.dumps(results, indent=2))
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
    return results


try:
    while not ready():
        state("waiting_for_existing_suite_and_free_gpus")
        time.sleep(30)
    state("running_smoke_episodes",
          episodes=len(cfg["models"])*len(cfg.get("scenes", ["Hospital", "Office"])))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(worker, name, model) for name, model in cfg["models"].items()}
        results = {name: future.result() for name, future in futures.items()}
    state("complete", results=results)
    (root/"CLOSED_LOOP_COMPLETE").touch()
except Exception as error:
    state("failed", error=str(error))
    raise
