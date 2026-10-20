import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "continuous_suite", Path(__file__).parents[1] / "tools/run_continuous_suite.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_watch_startup_and_frozen_steps():
    watch = module.ProgressWatch(0)
    assert watch.check(None, 599) is None
    assert watch.check(None, 601) == "startup_timeout"
    progress = dict(episode=0, step=1, plan_revision=1)
    assert watch.check(progress, 602) is None
    # Log/RPC traffic alone must not keep a frozen simulator "healthy".
    progress["plan_revision"] = 99
    assert watch.check(progress, 783) == "simulation_not_advancing"


def test_watch_frozen_plans_and_long_episode():
    watch = module.ProgressWatch(0)
    assert watch.check(dict(episode=0, step=1, plan_revision=1), 1) is None
    assert watch.check(dict(episode=0, step=100, plan_revision=1), 182) == "planning_not_advancing"
    watch = module.ProgressWatch(0)
    watch.check(dict(episode=0, step=1, plan_revision=1), 1)
    assert watch.check(dict(episode=0, step=2, plan_revision=2), 1802) == "episode_walltime_limit"
    assert watch.check(dict(episode=1, step=0, plan_revision=3), 1803) is None


def write_attempt(folder):
    folder.mkdir()
    (folder / "metric.csv").write_text(
        "episode,success,spl,time_to_goal,trajectory_length,timing_requests\n0,0,0,3,1,10\n")
    (folder / "fps_0.mp4").write_bytes(b"x" * 2000)
    (folder / "offscreen_guard_config.json").write_text('{"enabled":true}')
    (folder / "camera_profile.json").write_text('{"profile":"test"}')
    (folder / "offscreen_guard.jsonl").write_text('{"event":"bind"}\n{"callbacks":50}\n')


def test_resume_only_complete_artifacts(tmp_path):
    folder = tmp_path / "continuous_000_1"
    write_attempt(folder)
    done = module.reconcile(tmp_path, 100)
    assert module.next_missing(done, 100) == 1
    assert (tmp_path / "episode_000/COMPLETE.json").exists()
    with (folder / "offscreen_guard.jsonl").open("a") as stream:
        stream.write('{"event":"heartbeat","call')
    assert module.next_missing(module.reconcile(tmp_path, 100), 100) == 1
    (folder / "fps_0.mp4").unlink()
    assert module.reconcile(tmp_path, 100) == {}


def test_duplicate_and_gap_refused(tmp_path):
    write_attempt(tmp_path / "continuous_000_1")
    write_attempt(tmp_path / "continuous_000_2")
    with pytest.raises(RuntimeError, match="Duplicate"):
        module.reconcile(tmp_path, 100)
    with pytest.raises(RuntimeError, match="Non-contiguous"):
        module.next_missing({0: {}, 2: {}}, 100)


def test_persistent_failure_format(tmp_path):
    path = tmp_path / "retries.json"
    module.atomic_json(path, {"7": 3})
    assert module.read_json(path)["7"] == 3
    assert module.next_missing(dict.fromkeys(range(100)), 100) == 100


def test_scene_keeps_two_episodes_in_one_process(tmp_path, monkeypatch):
    cfg = dict(output=str(tmp_path), episodes=2, dataset="/data", code_root="/code", container="test")
    suite = module.Suite(cfg)
    suite.server["model"] = True
    monkeypatch.setattr(suite, "healthy", lambda _: True)
    monkeypatch.setattr(suite, "stop_sim", lambda _: None)
    commands = []

    class Process:
        returncode = 0
        def poll(self):
            return 0
        def wait(self, timeout):
            return 0

    def spawn(command, **_):
        commands.append(command)
        folder = Path(command[command.index("--output_dir")+1])
        (folder / "metric.csv").write_text(
            "episode,success,spl,time_to_goal,trajectory_length,timing_requests\n"
            "0,0,0,3,1,10\n1,1,1,3,1,10\n")
        for episode in range(2):
            (folder / f"fps_{episode}.mp4").write_bytes(b"x"*2000)
        (folder / "offscreen_guard_config.json").write_text('{"enabled":true}')
        (folder / "camera_profile.json").write_text('{"profile":"test"}')
        (folder / "offscreen_guard.jsonl").write_text('{"event":"bind"}\n{"callbacks":50}\n')
        return Process()

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    (tmp_path / "model").mkdir()
    suite.run_scene("model", {"port": 1}, 0, "Hospital")
    assert len(commands) == 1
    assert commands[0][commands[0].index("--num_episodes")+1] == "2"
    assert (tmp_path / "model/Hospital/COMPLETE").exists()


def test_circuit_is_not_reset_by_scheduler_restart(tmp_path):
    cfg = dict(output=str(tmp_path), episodes=2)
    folder = tmp_path / "model/Hospital"
    folder.mkdir(parents=True)
    module.atomic_json(folder / "retries.json", {"0": 3})
    suite = module.Suite(cfg)
    with pytest.raises(RuntimeError, match="retry circuit open"):
        suite.run_scene("model", {}, 0, "Hospital")


def test_legacy_404_is_only_explicit_opt_in_liveness(tmp_path, monkeypatch):
    suite = module.Suite(dict(output=str(tmp_path)))
    def fail(*args, **kwargs):
        raise module.urllib.error.HTTPError("url", 404, "Not Found", {}, None)
    monkeypatch.setattr(module.urllib.request, "urlopen", fail)
    model = dict(port=9010, health_algorithm="navdp")
    assert not suite.healthy(model)
    model["legacy_health_404_is_liveness"] = True
    assert suite.healthy(model)


def test_healthy_existing_office_is_not_restarted(tmp_path, monkeypatch):
    output = tmp_path / "existing/navdp"
    output.mkdir(parents=True)
    module.atomic_json(output / "status.json", dict(state="running_continuous"))
    dep = dict(name="navdp", output=str(output), queue_script="/code/run_quality100_continuous.py")
    suite = module.Suite(dict(output=str(tmp_path / "new"), dependencies=[dep]))
    monkeypatch.setattr(suite, "healthy", lambda _: True)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw:
                        f"PID COMMAND\n123 python3 -u {dep['queue_script']} --output {output}\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Healthy dependency must not be mutated")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    suite.check_dependencies()
    assert not (tmp_path / "new/dependency_navdp_retries.json").exists()
