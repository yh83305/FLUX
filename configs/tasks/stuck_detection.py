"""Pure helpers for translation-and-heading stuck detection."""

from __future__ import annotations

import numpy as np


def wrapped_angle_distance(first, second):
    """Return the absolute shortest angular distance in radians."""
    delta = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    return np.abs(np.arctan2(np.sin(delta), np.cos(delta)))


def pose_window_is_stuck(
    poses,
    *,
    translation_threshold: float,
    heading_threshold: float,
) -> bool:
    """Check whether every prior XY-theta pose remains near the current pose."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("poses must have shape [N,3] containing x, y, theta")
    if len(values) < 2:
        return False
    if translation_threshold <= 0.0 or heading_threshold <= 0.0:
        raise ValueError("stuck thresholds must be positive")
    current = values[-1]
    translation = np.linalg.norm(values[:-1, :2] - current[:2], axis=1)
    heading = wrapped_angle_distance(values[:-1, 2], current[2])
    return bool(
        translation.max() < float(translation_threshold)
        and heading.max() < float(heading_threshold)
    )
