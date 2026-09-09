
import os
import numpy as np
import csv
import cv2
import torch
try:
    import open3d as o3d
except ImportError:
    o3d = None
from typing import Any, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class PlanningInput:
    current_goal: Optional[np.ndarray] = None
    current_image: Optional[np.ndarray] = None
    current_depth: Optional[np.ndarray] = None
    camera_pos: Optional[np.ndarray] = None
    camera_rot: Optional[np.ndarray] = None
    robot_state: Optional[np.ndarray] = None
    applied_command: Optional[np.ndarray] = None
    people_positions: Optional[np.ndarray] = None
    people_paths: Any = None
    people_valid: bool = False
    observation_frame_id: int = -1
    observation_time_s: float = 0.0

@dataclass
class PlanningOutput:
    trajectory_points_world: Optional[np.ndarray] = None
    all_trajectories_world: Optional[List[np.ndarray]] = None
    all_trajectories_camera: Optional[np.ndarray] = None
    point_goals_camera: Optional[np.ndarray] = None
    all_values_camera: Optional[np.ndarray] = None
    sub_pointgoal_pd: Optional[np.ndarray] = None
    mode_debug: Optional[List[dict]] = None
    is_planning: bool = False
    planning_error: Optional[str] = None
    episode_generation: int = -1
    plan_revision: int = 0
    source_image: Optional[np.ndarray] = None
    source_depth: Optional[np.ndarray] = None
    source_camera_pos: Optional[np.ndarray] = None
    source_camera_rot: Optional[np.ndarray] = None
    source_robot_state: Optional[np.ndarray] = None
    source_applied_command: Optional[np.ndarray] = None
    source_people_positions: Optional[np.ndarray] = None
    source_people_paths: Any = None
    source_people_valid: bool = False
    source_observation_frame_id: int = -1
    source_observation_time_s: float = 0.0


def pad_video_frame(image: np.ndarray, macro_block_size: int = 16) -> np.ndarray:
    """Pad RGB frames without rescaling so ffmpeg receives codec-safe shapes."""
    height, width = image.shape[:2]
    pad_height = (-height) % macro_block_size
    pad_width = (-width) % macro_block_size
    if not pad_height and not pad_width:
        return image
    return np.pad(
        image,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="constant",
    )

def find_usd_path(dir,task='pointgoal'):
    paths = os.listdir(dir)
    usd_path = ""
    init_path = ""
    for p in paths:
        if ".usd" in p and 'noMDL' not in p:
            usd_path = os.path.join(dir,p)
        if ".npy" in p and task in p:
            init_path = os.path.join(dir,p)
    return usd_path,init_path

def write_metrics(metrics, path="exploration.csv"):
    with open(path, mode="w", newline="") as csv_file:
        fieldnames = metrics[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)
        
def draw_box_with_text(image, x, y, width, height, text, 
                       box_color=(0, 255, 0), text_color=(255, 255, 255), 
                       thickness=2, font_scale=1.0):
    cv2.rectangle(image, (x, y), (x + width, y + height), box_color, thickness)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = x + (width - text_width) // 2
    text_y = y + (height + text_height) // 2
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, text_color, thickness)
    return image

def cpu_pointcloud_from_array(points,colors):
    if o3d is None:
        raise ImportError("cpu_pointcloud_from_array requires open3d")
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    pointcloud.colors = o3d.utility.Vector3dVector(colors)
    return pointcloud

def adjust_usd_scale(prim_path="/World/Scene/terrain",scale=1.0):
    import omni
    from pxr import UsdGeom, Usd, Sdf, Gf
    stage = omni.usd.get_context().get_stage()
    scene_prim = stage.GetPrimAtPath(prim_path)
    if scene_prim.IsValid():
        print(f"Directly setting scale for prim: <{scene_prim.GetPath()}>")
        # 1. Get or create the scale attribute and set its value.
        scale_attr = scene_prim.GetAttribute("xformOp:scale")
        if not scale_attr:
            scale_attr = scene_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3, False)
        scale_attr.Set(Gf.Vec3d(scale, scale, scale))

        # 2. Ensure 'xformOp:scale' is in the transformation order.
        order_attr = scene_prim.GetAttribute("xformOpOrder")
        if not order_attr.HasValue():
            # If order doesn't exist, create it with a default that includes scale.
            scene_prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, False).Set(["xformOp:translate", "xformOp:orient", "xformOp:scale"])
        else:
            order = list(order_attr.Get())
            if "xformOp:scale" not in order:
                order.append("xformOp:scale")
                order_attr.Set(order)
        print(f"Successfully set scale for prim <{scene_prim.GetPath()}>")
    else:
        print("Warning: Could not find prim at /World/Scene to apply scale.")
