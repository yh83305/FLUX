"""Keep each scene alive; checkpoint closed episode artifacts and resume on faults."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def completed_rows(folder):
    """metric.csv is atomically replaced only after each episode video closes."""
    try:
        rows = list(csv.DictReader((folder/"metric.csv").open()))
        events = [json.loads(x) for x in (folder/"offscreen_guard.jsonl").read_text().splitlines()]
        if not any(x.get("callbacks", 0) > 0 for x in events):
            return []
        if not any(x.get("event") == "bind" for x in events):
            return []
        if not json.loads((folder/"offscreen_guard_config.json").read_text())["enabled"]:
            return []
        ids = [int(x["episode"]) for x in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate metric episode IDs")
        return [x for x in rows if int(x["timing_requests"]) > 0
                and (folder/f'fps_{int(x["episode"])}.mp4').stat().st_size > 1000]
    except (OSError, ValueError, KeyError):
        return []


def main():
    p = argparse.ArgumentParser()
    for name in ("dataset", "output", "model", "gpu", "port", "container", "code-root"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--progress-timeout", type=float, default=180.)
    a = p.parse_args()
    root, dataset, code = Path(a.output).resolve(), Path(a.dataset), Path(a.code_root)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/"queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((dataset/"manifest.json").read_text())
    assert manifest["count_per_scene"] == 100
    for row in manifest["records"]:
        file = dataset/row["scene"]/f'episode_{row["episode"]}.json'
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row["sha256"]

    def status(**kw):
        record = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), model=a.model, **kw)
        tmp = root/"status.tmp"
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(root/"status.json")
        print(json.dumps(record), flush=True)

    def processes():
        result = subprocess.run(["docker", "top", a.container, "-eo", "pid,args"],
                                text=True, capture_output=True, check=True)
        outputs = set()
        for line in result.stdout.splitlines()[1:]:
            parts = shlex.split(line)
            if len(parts) < 2 or parts[1] != "/isaac-sim/kit/python/bin/python3":
                continue
            if "--output_dir" in parts:
                path = Path(parts[parts.index("--output_dir")+1])
                if path.is_relative_to(root):
                    outputs.add(path)
        return outputs

    def stop(folder):
        pattern = "^/isaac-sim/kit/python/bin/python3 .*--output_dir "+re.escape(str(folder))+"$"
        for sig in ("TERM", "KILL"):
            subprocess.run(["docker", "exec", a.container, "pkill", "-"+sig, "-f", pattern],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            if folder not in processes():
                return
        raise RuntimeError(f"Cannot stop exact owned simulator {folder}")

    def reconcile(scene):
        accepted = {}
        scene_root = root/scene
        metrics = sorted(scene_root.glob("episode_*/attempt_*/metric.csv"))
        metrics += sorted(scene_root.glob("continuous_*/metric.csv"))
        for file in metrics:
            for row in completed_rows(file.parent):
                episode = int(row["episode"])
                if episode not in range(100):
                    raise ValueError("Out-of-range episode")
                accepted.setdefault(episode, file.parent)
        for episode, folder in accepted.items():
            base = scene_root/f"episode_{episode:03d}"
            base.mkdir(parents=True, exist_ok=True)
            tmp = base/"COMPLETE.tmp"
            tmp.write_text(json.dumps(dict(result_dir=str(folder), episode=episode)))
            tmp.replace(base/"COMPLETE.json")
        return accepted

    def unhealthy(folder):
        text = (folder/"eval.log").read_text(errors="replace") if (folder/"eval.log").exists() else ""
        heartbeat = folder/"offscreen_guard.jsonl"
        stale = heartbeat.exists() and time.time()-heartbeat.stat().st_mtime > a.progress_timeout
        boot_stale = not heartbeat.exists() and time.time()-folder.stat().st_mtime > 600
        return ("[INFO] Cleanup complete" in text or "fail to launch" in text
                or "Traceback (most recent call last)" in text or stale or boot_stale)

    try:
        # Adopt the current one-episode process without restarting healthy work.
        while processes():
            for scene in ("Hospital", "Office"):
                reconcile(scene)
            for folder in processes():
                if unhealthy(folder):
                    stop(folder)
            status(state="finishing_existing_episode")
            time.sleep(5)

        for scene in ("Hospital", "Office"):
            failures = {}
            while True:
                done = reconcile(scene)
                missing = [i for i in range(100) if i not in done]
                if not missing:
                    break
                start = missing[0]
                if any(i in done for i in range(start, 100)):
                    raise RuntimeError("Non-contiguous completed set; refusing duplicate rollouts")
                out = root/scene/f"continuous_{start:03d}_{time.time_ns()}"
                out.mkdir(parents=True)
                command = ["docker", "exec", "-e", f"CUDA_VISIBLE_DEVICES={a.gpu}",
                           "-e", "PYTHONUNBUFFERED=1", "-e", f"PYTHONPATH={code}",
                           "-e", "FLUX_ISAAC_EXPERIENCE=/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit",
                           "-w", "/workspace/FLUX", a.container, "/isaac-sim/python.sh", "-u",
                           str(code/"runtime_fix/eval_render_safe.py"), "--port", a.port,
                           "--gpu_id", "0", "--scene_dir", str(dataset/"scene_views"/scene),
                           "--scene_index", "0", "--scene_scale", "1", "--num_envs", "1",
                           "--num_episodes", str(100-start), "--episode_start", str(start),
                           "--speed", ".5", "--stop_threshold", "-3.0", "--output_dir", str(out)]
                status(state="running_continuous", scene=scene, episode=start, remaining=100-start)
                last_count = len(done)
                with (out/"eval.log").open("w") as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    while proc.poll() is None:
                        time.sleep(5)
                        done = reconcile(scene)
                        if len(done) != last_count:
                            status(state="running_continuous", scene=scene, completed=len(done),
                                   episode=min((i for i in range(100) if i not in done), default=100))
                            last_count = len(done)
                        if unhealthy(out):
                            stop(out)
                            break
                    if out in processes():
                        stop(out)
                    proc.wait(timeout=20)
                done = reconcile(scene)
                if len(done) == 100:
                    continue
                failed_episode = min(i for i in range(100) if i not in done)
                failures[failed_episode] = failures.get(failed_episode, 0)+1
                if failures[failed_episode] >= 3:
                    raise RuntimeError(f"{scene}/{failed_episode}: three failed attempts")
                status(state="restart_after_fault", scene=scene, episode=failed_episode)
                time.sleep(10)
        (root/"COMPLETE").touch()
        status(state="complete", episodes=200)
    except Exception as exc:
        status(state="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
