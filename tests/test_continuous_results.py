import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "continuous", Path(__file__).parents[1]/"tools/run_quality100_continuous.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(folder):
    (folder/"metric.csv").write_text("episode,timing_requests\n5,20\n6,30\n")
    (folder/"offscreen_guard_config.json").write_text('{"enabled":true}')
    (folder/"offscreen_guard.jsonl").write_text(
        '{"event":"bind"}\n{"event":"heartbeat","callbacks":1000}\n')
    for i in (5, 6):
        (folder/f"fps_{i}.mp4").write_bytes(b"x"*1024)


def test_completed_episodes_survive_later_process_crash(tmp_path):
    fixture(tmp_path)
    (tmp_path/"eval.log").write_text("episode 7 crashed; no cleanup")
    assert [int(r["episode"]) for r in module.completed_rows(tmp_path)] == [5, 6]


def test_no_acceptance_without_video_or_runtime_evidence(tmp_path):
    fixture(tmp_path)
    (tmp_path/"fps_6.mp4").write_bytes(b"x"*48)
    assert [int(r["episode"]) for r in module.completed_rows(tmp_path)] == [5]
    (tmp_path/"offscreen_guard.jsonl").write_text('{"event":"installed"}\n')
    assert module.completed_rows(tmp_path) == []


def test_duplicate_episode_rows_not_accepted(tmp_path):
    fixture(tmp_path)
    (tmp_path/"metric.csv").write_text("episode,timing_requests\n5,20\n5,30\n")
    assert module.completed_rows(tmp_path) == []
