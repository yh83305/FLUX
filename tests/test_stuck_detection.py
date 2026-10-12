import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "configs/tasks/stuck_detection.py"
)
SPEC = importlib.util.spec_from_file_location("stuck_detection", MODULE_PATH)
stuck_detection = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stuck_detection)
pose_window_is_stuck = stuck_detection.pose_window_is_stuck
wrapped_angle_distance = stuck_detection.wrapped_angle_distance


def test_stationary_pose_window_is_stuck():
    poses = np.zeros((30, 3), dtype=np.float64)
    assert pose_window_is_stuck(
        poses, translation_threshold=0.1, heading_threshold=0.1
    )


def test_translation_prevents_stuck():
    poses = np.zeros((30, 3), dtype=np.float64)
    poses[:, 0] = np.linspace(0.0, 0.2, len(poses))
    assert not pose_window_is_stuck(
        poses, translation_threshold=0.1, heading_threshold=0.1
    )


def test_in_place_rotation_prevents_stuck():
    poses = np.zeros((30, 3), dtype=np.float64)
    poses[:, 2] = np.linspace(0.0, 0.4, len(poses))
    assert not pose_window_is_stuck(
        poses, translation_threshold=0.1, heading_threshold=0.1
    )


def test_heading_wraparound_uses_shortest_distance():
    distance = wrapped_angle_distance(
        np.deg2rad(np.array([179.0])), np.deg2rad(-179.0)
    )
    assert np.allclose(distance, np.deg2rad(2.0))
