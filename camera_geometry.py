"""Pure camera geometry helpers; contains no benchmark configuration."""
from __future__ import annotations

import math
import torch


def horizontal_aperture(focal_length: float, horizontal_fov_degrees: float) -> float:
    return 2.0 * float(focal_length) * math.tan(
        math.radians(float(horizontal_fov_degrees)) / 2.0
    )


def level_follow_camera_pose(root_transforms_xyzw, ground_z, height: float):
    """Follow robot XY/yaw while discarding chassis height, roll, pitch and scale."""
    roots = torch.as_tensor(root_transforms_xyzw)
    if roots.ndim != 2 or roots.shape[1] != 7 or not bool(torch.isfinite(roots).all()):
        raise ValueError("Robot transforms must be finite [N,7] XYZW poses")
    x, y, z, w = roots[:, 3:].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    positions = roots[:, :3].clone()
    positions[:, 2] = torch.as_tensor(
        ground_z, device=roots.device, dtype=roots.dtype
    ) + float(height)
    orientations = torch.zeros((len(roots), 4), device=roots.device, dtype=roots.dtype)
    orientations[:, 0] = torch.cos(yaw / 2)
    orientations[:, 3] = torch.sin(yaw / 2)
    return positions, orientations
