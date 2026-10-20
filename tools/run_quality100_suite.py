"""Resumable native policy suite, fresh Isaac process per episode."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--model", required=True, choices=["navdp", "flux"])
p.add_argument("--gpu", required=True)
p.add_argument("--port", required=True)
p.add_argument("--container", required=True)
p.add_argument("--evaluator", required=True)
p.add_argument("--code-root", required=True, help="Pinned FLUX source path visible in container")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
lock = (a.output/"queue.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
manifest = json.loads((a.dataset/"manifest.json").read_text())
assert manifest["count_per_scene"] == 100
for r in manifest["records"]:
    file = a.dataset/r["scene"]/f'episode_{r["episode"]}.json'
    assert hashlib.sha256(file.read_bytes()).hexdigest() == r["sha256"]

def valid(out, episode):
    try:
        rows = list(csv.DictReader((out/"metric.csv").open()))
        guard = [json.loads(line) for line in (out/"offscreen_guard.jsonl").read_text().splitlines()]
        return (any(row.get("callbacks", 0) > 0 for row in guard)
                and any(row.get("event") == "bind" for row in guard)
                and len(rows) == 1 and int(rows[0]["episode"]) == episode
                and int(rows[0]["timing_requests"]) > 0
                and (out/f"fps_{episode}.mp4").stat().st_size > 1000
                and "[INFO] Cleanup complete" in (out/"eval.log").read_text(errors="replace")
                and json.loads((out/"offscreen_guard_config.json").read_text())["enabled"])
    except (OSError, ValueError, KeyError):
        return False

def stop(out):
    pattern = "^/isaac-sim/kit/python/bin/python3 .*--output_dir " + re.escape(str(out)) + "$"
    for sig in ["TERM", "KILL"]:
        subprocess.run(["docker", "exec", a.container, "pkill", "-"+sig, "-f", pattern],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
    alive = subprocess.run(["docker", "exec", a.container, "pgrep", "-f", pattern],
                           stdout=subprocess.PIPE)
    if alive.returncode != 1:
        raise RuntimeError(f"Cannot confirm simulator exit: {out}")

def status(**kw):
    row = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), model=a.model, **kw)
    (a.output/"status.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)

try:
    for scene in ["Hospital", "Office"]:
        for episode in range(100):
            base = a.output/scene/f"episode_{episode:03d}"
            done = False
            for attempt in range(1, 4):
                out = base/f"attempt_{attempt}"
                if valid(out, episode):
                    done = True
                    break
                if out.exists():
                    # An interrupted prior launch may have left a container process.
                    stop(out)
                    continue
                out.mkdir(parents=True)
                status(state="running", scene=scene, episode=episode, attempt=attempt)
                cmd = ["docker", "exec", "-e", f"CUDA_VISIBLE_DEVICES={a.gpu}",
                       "-e", "PYTHONUNBUFFERED=1", "-e", f"PYTHONPATH={a.code_root}",
                       "-e", "FLUX_ISAAC_EXPERIENCE=/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit",
                       "-w", "/workspace/FLUX", a.container, "/isaac-sim/python.sh",
                       "-u", a.evaluator, "--port", a.port, "--gpu_id", "0",
                       "--scene_dir", str(a.dataset/"scene_views"/scene),
                       "--scene_index", "0", "--scene_scale", "1", "--num_envs", "1",
                       "--num_episodes", "1", "--episode_start", str(episode),
                       "--speed", "0.5", "--stop_threshold", "-3.0", "--output_dir", str(out)]
                with (out/"eval.log").open("w") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    started = time.monotonic()
                    while proc.poll() is None:
                        time.sleep(5)
                        text = (out/"eval.log").read_text(errors="replace")
                        if ("[INFO] Cleanup complete" in text or "fail to launch" in text
                                or "Traceback (most recent call last)" in text
                                or time.monotonic()-started > 1200):
                            break
                    stop(out)
                    proc.wait(timeout=20)
                if valid(out, episode):
                    done = True
                    break
                status(state="retry", scene=scene, episode=episode, attempt=attempt)
                time.sleep(30)
            if not done:
                raise RuntimeError(f"{scene}/{episode} exhausted three attempts")
            (base/"COMPLETE.json").write_text(json.dumps({"result_dir":str(out)}))
            status(state="episode_complete", scene=scene, episode=episode)
    (a.output/"COMPLETE").touch()
    status(state="complete", episodes=200)
except Exception as exc:
    status(state="failed", error=str(exc))
    raise
