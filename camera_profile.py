"""Versioned observation-camera profile shared by FLUX benchmark entrypoints."""
from __future__ import annotations
import math
import torch

CAMERA_PROFILE_ID = "world060_pitch0_hfov69_v1"
CAMERA_HEIGHT_M = 0.60
HORIZONTAL_FOV_DEGREES = 69.0
FOCAL_LENGTH = 1.4
HORIZONTAL_APERTURE = 2.0 * FOCAL_LENGTH * math.tan(math.radians(HORIZONTAL_FOV_DEGREES / 2.0))


def level_follow_pose(root_transforms_xyzw, ground_z, height=CAMERA_HEIGHT_M):
    """Follow XY/yaw only; discard chassis height, roll, pitch, and asset scale."""
    roots = torch.as_tensor(root_transforms_xyzw)
    if roots.ndim != 2 or roots.shape[1] != 7 or not bool(torch.isfinite(roots).all()):
        raise ValueError("Robot transforms must be finite [N,7] XYZW poses")
    x,y,z,w = roots[:, 3:].unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    positions = roots[:, :3].clone()
    positions[:, 2] = torch.as_tensor(ground_z,device=roots.device,dtype=roots.dtype)+float(height)
    orientations = torch.zeros((len(roots),4),device=roots.device,dtype=roots.dtype)
    orientations[:, 0] = torch.cos(yaw/2)
    orientations[:, 3] = torch.sin(yaw/2)
    return positions, orientations
