"""
Social point-goal navigation evaluation with simulated people in Isaac Sim.

Point-goal planning via NavDP, MPC control, and social metrics over multi-episode JSON.
"""
import argparse

parser = argparse.ArgumentParser(description="Social navigation (point goal) evaluation")
parser.add_argument(
    "--scene_dir", type=str, default="/workspace/FLUX/assets/dynbench/isaacsim_scene",
    help="Directory containing scene folders",
)
parser.add_argument("--scene_index", type=int, default=0, help="Scene index within scene_dir")
parser.add_argument("--scene_scale", type=float, default=1.0, help="Scene scale factor")
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0,
    help="Stop threshold for exploration turn (critic value)",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episode rollouts to run")
parser.add_argument(
    "--episode_start", type=int, default=0,
    help="First episode JSON id (allows fresh-process batched evaluation)",
)
parser.add_argument("--speed", type=float, default=0.5, help="Desired linear speed (m/s)")
parser.add_argument("--port", type=int, default=9999, help="NavDP server port")
parser.add_argument("--gpu_id", type=int, default=0, help="CUDA device id when not using multi-GPU")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Optional directory for videos and metrics")
args_cli = parser.parse_args()

import os
import sys
import json

print(f"GPU {args_cli.gpu_id}, Scene {args_cli.scene_index}")
print(f"OMNI_USER_DATA_DIR: {os.environ.get('OMNI_USER_DATA_DIR', 'NOT SET')}")
print(f"CARB_APP_DATA_DIR: {os.environ.get('CARB_APP_DATA_DIR', 'NOT SET')}")

from isaaclab.app import AppLauncher

HEADLESS = True
MULTI_GPU = False
NUM_GPUS = 4

CUSTOM_APP_PATH = os.environ.get(
    "FLUX_ISAAC_EXPERIENCE",
    "/home/yhpang/ProtoMotions/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit",
)

launcher_kwargs = {
    "headless": HEADLESS,
    "enable_cameras": True,
    "experience": CUSTOM_APP_PATH,
}

if MULTI_GPU:
    launcher_kwargs["multi_gpu"] = True
else:
    launcher_kwargs["device"] = f"cuda:{args_cli.gpu_id}"

app_launcher = AppLauncher(**launcher_kwargs)
simulation_app = app_launcher.app

if MULTI_GPU:
    import carb
    settings = carb.settings.get_settings()
    settings.set("/renderer/multiGpu/enabled", True)
    settings.set("/renderer/multiGpu/maxGpuCount", NUM_GPUS)
    settings.set("/renderer/multiGpu/autoEnable", True)
    print(f"[INFO] Multi-GPU enabled with {NUM_GPUS} GPUs")
else:
    print(f"[INFO] Single GPU mode: GPU {args_cli.gpu_id}")

import omni
import cv2
import carb
import numpy as np
import imageio
import csv
import torch
import open3d as o3d
import asyncio
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeled_robots.controllers.differential_controller import DifferentialController
import time
import threading

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text, adjust_usd_scale
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_reset, pointgoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller
from utils_tasks.sim_utils import cleanup_simulation, register_signal_handlers, setup_all_lighting
from socialnav_metrics import (
    SocialMetricsTracker, get_people_positions, write_socialnav_summary,
)

planning_input = PlanningInput()
planning_output = PlanningOutput()
planning_input.episode_generation = 0
planning_output.episode_generation = -1
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mpc = None

MODE_DEBUG_ALGOS = {"flux_explicit_modes_rule16", "flux_direction5_speed3_rule16",
                    "flux_predicted_prototype_rule16", "flux_gt_factorized_rule16",
                    "flux_direction5_speed3_rule15", "flux_gt_factorized_rule15"}


def _number(value, signed=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def draw_mode_debug_panel(image, debug):
    """Append the explicit-mode candidate selection table."""
    rows = debug.get("candidate_debug", []) if isinstance(debug, dict) else []
    panel = np.full((image.shape[0], 620, 3), (18, 22, 28), dtype=np.uint8)
    cv2.putText(panel, f"MODE SELECT: {debug.get('selection_reason', 'unknown')}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)
    columns = ((8, "idx"), (45, "mode"), (92, "P"), (132, "safe"),
               (174, "ent"), (213, "goal"), (281, "esdf"), (351, "unk"),
               (395, "temp"), (458, "final"), (530, "filter"))
    for x, label in columns:
        cv2.putText(panel, label, (x, 52), cv2.FONT_HERSHEY_SIMPLEX,
                    .38, (190, 200, 210), 1, cv2.LINE_AA)
    for line, row in enumerate(rows):
        selected, safe = bool(row.get("selected")), bool(row.get("safe"))
        color = (20, 220, 255) if selected else ((80, 220, 100) if safe else (100, 110, 125))
        y = 78 + line * 24
        values = ((8, f"{row.get('index', -1):02d}"), (45, f"m{row.get('mode', -1)}"),
                  (92, _number(row.get("prior"))), (132, "Y" if safe else "N"),
                  (174, "Y" if row.get("entered_selection") else "N"),
                  (213, _number(row.get("goal_score"), True)),
                  (281, _number(row.get("minimum_esdf_clearance_m"), True)),
                  (351, _number(row.get("unknown_fraction"))),
                  (395, _number(row.get("temporal_cost"))),
                  (458, _number(row.get("final_score"), True)),
                  (530, str(row.get("filtered_reason") or ("SELECTED" if selected else "-"))))
        for x, value in values:
            cv2.putText(panel, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .36, color, 1, cv2.LINE_AA)
    return np.concatenate((image, panel), axis=1)


def draw_esdf_candidates(size, trajectories, goal, debug, candidate_values=None):
    """Render ESDF, all candidates and the selected solution in camera coordinates."""
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    meta = debug.get("esdf_debug", {}) if isinstance(debug, dict) else {}
    esdf = np.asarray(meta.get("slice", []), dtype=np.uint8)
    if esdf.ndim != 2 or not esdf.size:
        return canvas
    colored = cv2.cvtColor(cv2.applyColorMap(255 - esdf, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    observed = np.asarray(meta.get("observed_slice", []), dtype=bool)
    if observed.shape == esdf.shape:
        checker = np.indices(esdf.shape).sum(axis=0) % 2
        unknown_color = np.where(checker[..., None],
                                 np.array([60, 60, 66]),
                                 np.array([92, 92, 100])).astype(np.uint8)
        colored[~observed] = unknown_color[~observed]
    colored = cv2.resize(colored, (size, size), interpolation=cv2.INTER_NEAREST)
    canvas[:] = colored
    origin_r, origin_f = meta.get("grid_origin_right_forward_m", [-2., 0.])
    voxel = max(float(meta.get("voxel_size_m", .05)), 1e-6)
    sx, sy = size / esdf.shape[1], size / esdf.shape[0]
    rows = debug.get("candidate_debug", [])
    selected = int(debug.get("selected_index", -1))
    safety_margin = float(meta.get("safety_margin_m", 0.10))
    display_minimum = float(meta.get("display_minimum_m", -0.25))
    display_maximum = float(meta.get("display_maximum_m", 0.60))
    values = np.asarray(
        candidate_values if candidate_values is not None else
        [row.get("final_score", np.nan) for row in rows],
        dtype=np.float32,
    ).reshape(-1)
    safe_mask = np.asarray(
        [bool(row.get("safe", False)) for row in rows], dtype=bool
    )
    if len(safe_mask) != len(values):
        safe_mask = np.zeros(len(values), dtype=bool)
    valid_scores = safe_mask & np.isfinite(values) & (values > -1.0e5)
    safe_values = values[valid_scores]
    values_min = float(safe_values.min()) if safe_values.size else 0.0
    values_max = float(safe_values.max()) if safe_values.size else 0.0

    def relative_score_color(index):
        if index >= len(values) or not valid_scores[index]:
            return (128, 128, 128)
        spread = values_max - values_min
        relative = 0.5 if spread <= 1e-8 else float(np.clip(
            (values[index] - values_min) / spread, 0.0, 1.0
        ))
        # RGB: low=blue, middle=green, high=red; selected stays yellow.
        if relative < 0.5:
            return (0, int(510 * relative), int(255 * (1.0 - 2.0 * relative)))
        return (int(510 * (relative - 0.5)), int(510 * (1.0 - relative)), 0)
    for index, trajectory in enumerate(np.asarray(trajectories)):
        px = (-(trajectory[:, 1]) - float(origin_r)) / voxel * sx
        py = (esdf.shape[0] - 1 - (trajectory[:, 0] - float(origin_f)) / voxel) * sy
        points = np.rint(np.stack((px, py), 1)).astype(np.int32)
        valid = ((points[:, 0] >= 0) & (points[:, 0] < size) &
                 (points[:, 1] >= 0) & (points[:, 1] < size))
        points = points[valid]
        if not bool(valid.all()):
            last = np.rint(np.clip(
                np.stack((px, py), axis=1)[np.flatnonzero(~valid)[0]],
                [7, 7], [size - 8, size - 8],
            )).astype(np.int32)
            cv2.drawMarker(canvas, tuple(last), (255, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        if len(points) < 2:
            continue
        mode = int(rows[index].get("mode", -1)) if index < len(rows) else -1
        unknown = float(rows[index].get("unknown_fraction", 0.0)) if index < len(rows) else 0.0
        selected_safe = index == selected and index < len(valid_scores) and valid_scores[index]
        color = (255, 255, 0) if selected_safe else relative_score_color(index)
        cv2.polylines(canvas, [points], False, color, 5 if index == selected else 2, cv2.LINE_AA)
        cv2.putText(canvas, f"m{mode} u={unknown:.2f}", tuple(points[-1]), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, color, 1, cv2.LINE_AA)
    goal = np.asarray(goal).reshape(-1)
    if len(goal) >= 2:
        gx = int(round((-goal[1] - float(origin_r)) / voxel * sx))
        gy = int(round((esdf.shape[0] - 1 - (goal[0] - float(origin_f)) / voxel) * sy))
        cv2.drawMarker(canvas, (int(np.clip(gx, 8, size-8)), int(np.clip(gy, 8, size-8))),
                       (255, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    cv2.putText(canvas, "relative score: blue=low green=mid red=high yellow=selected", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    # Use the exact range supplied by the selector. The free-space end equals
    # clearance_cap_m, where its soft clearance bonus saturates.
    bar_x0, bar_x1 = size - 42, size - 24
    bar_y0, bar_y1 = 52, min(size - 52, 332)
    encoded = np.linspace(255, 0, bar_y1 - bar_y0, dtype=np.uint8)[:, None]
    bar = cv2.cvtColor(
        cv2.applyColorMap(encoded, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
    )
    canvas[bar_y0:bar_y1, bar_x0:bar_x1] = np.repeat(
        bar, bar_x1 - bar_x0, axis=1
    )
    cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1),
                  (255, 255, 255), 1)
    for distance, label in ((display_minimum, f"{display_minimum:+.2f}m"), (0.0, "0"),
                            (safety_margin, f"safe {safety_margin:.2f}"),
                            (display_maximum, f"cap {display_maximum:.2f}m")):
        fraction = np.clip(
            (distance - display_minimum) /
            max(display_maximum - display_minimum, 1e-8), 0.0, 1.0
        )
        y = int(round(bar_y0 + fraction * (bar_y1 - bar_y0)))
        cv2.line(canvas, (bar_x0 - 4, y), (bar_x1 + 4, y),
                 (255, 255, 255), 1)
        cv2.putText(canvas, label, (bar_x0 - 82, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1,
                    cv2.LINE_AA)
    cv2.putText(canvas, "ESDF", (bar_x0 - 12, bar_y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "magenta X=out of grid", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 0, 255), 1,
                cv2.LINE_AA)
    return canvas


def validate_episode_start_state(
    env, infos, scene_path, episode_id, tolerance=0.05
):
    """Fail fast unless pose and goal observation match the JSON episode."""
    episode_file = os.path.join(scene_path, f"episode_{episode_id}.json")
    with open(episode_file, "r", encoding="utf-8") as handle:
        expected = json.load(handle)["episode"]["robot"]
    robot = env.unwrapped.scene.articulations["robot"]
    actual_xy = robot.data.root_pos_w[0, :2].detach().cpu().numpy()
    quat_wxyz = robot.data.root_quat_w[0].detach().cpu().numpy()
    actual_yaw = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler("xyz")[2]
    expected_xy = np.asarray(expected["start_pos"][:2], dtype=np.float64)
    expected_yaw = float(expected["start_orientation"])
    expected_goal_world = np.asarray(expected["goal_pos"][:2], dtype=np.float64)
    world_delta = expected_goal_world - expected_xy
    expected_goal_local = np.array((
        np.cos(expected_yaw) * world_delta[0]
        + np.sin(expected_yaw) * world_delta[1],
        -np.sin(expected_yaw) * world_delta[0]
        + np.cos(expected_yaw) * world_delta[1],
    ))
    actual_goal_local = (
        infos["observations"]["goal_pose"][0, :2].detach().cpu().numpy()
    )
    position_error = float(np.linalg.norm(actual_xy - expected_xy))
    yaw_error = float(abs((actual_yaw - expected_yaw + np.pi) % (2*np.pi) - np.pi))
    goal_error = float(np.linalg.norm(actual_goal_local - expected_goal_local))
    print(
        f"[EPISODE STATE] id={episode_id} expected_xy={expected_xy.tolist()} "
        f"actual_xy={actual_xy.tolist()} position_error={position_error:.5f}m "
        f"expected_yaw={expected_yaw:.5f} actual_yaw={actual_yaw:.5f} "
        f"yaw_error={yaw_error:.5f}rad "
        f"expected_goal_world={expected_goal_world.tolist()} "
        f"expected_goal_local={expected_goal_local.tolist()} "
        f"actual_goal_local={actual_goal_local.tolist()} "
        f"goal_error={goal_error:.5f}m",
        flush=True,
    )
    if position_error > tolerance or yaw_error > tolerance or goal_error > tolerance:
        raise RuntimeError(
            f"episode {episode_id} reset state mismatch: "
            f"position={position_error:.4f}m yaw={yaw_error:.4f}rad "
            f"goal={goal_error:.4f}m"
        )


def refresh_camera_after_reset(env, simulation_app, update_count=2):
    """Render, refresh camera buffers, and recompute reset observations."""
    camera = env.unwrapped.scene.sensors["camera_sensor"]
    for _ in range(int(update_count)):
        simulation_app.update()
    camera.update(env.unwrapped.step_dt, force_recompute=True)
    # Materialize annotator outputs now so the first recorded frame cannot
    # reuse RGB/depth from the terminal pose of the previous episode.
    _ = camera.data.output["rgb"]
    _ = camera.data.output["distance_to_image_plane"]
    observations = env.unwrapped.observation_manager.compute()
    print("[CAMERA RESET] refreshed RGB/depth and observations", flush=True)
    return observations


register_signal_handlers(
    get_env=lambda: globals().get("env", None),
    get_simulation_app=lambda: globals().get("simulation_app", None),
    stop_event=stop_event,
    get_planning_thread_obj=lambda: globals().get("planning_thread_obj", None),
    get_fps_writer=lambda: globals().get("fps_writer", None),
)


def planning_thread(env, camera_intrinsic):
    """Background thread: NavDP point-goal planning, world-frame trajectories, MPC state."""
    global mpc
    while not stop_event.is_set():
        try:
            with input_lock:
                if planning_input.current_goal is None or planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue
                goal = planning_input.current_goal.copy()
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()
                episode_generation = planning_input.episode_generation
            with output_lock:
                planning_output.is_planning = True

            trajectory_points_camera, all_trajectories_camera, all_values_camera, debug_payload = pointgoal_step(
                goal, image, depth, port=args_cli.port, return_debug=True
            )
            mode_debug = debug_payload.get("selector_diagnostics")

            batch_optimal_points_world = []
            for idx in range(trajectory_points_camera.shape[0]):
                trajectory_points_world = []
                for i, point in enumerate(trajectory_points_camera[idx]):
                    if i < 0:
                        continue
                    point_local = np.array([point[0], point[1], 0.0])
                    point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                    trajectory_points_world.append(point_world[:2])
                trajectory_points_world = np.array(trajectory_points_world)
                batch_optimal_points_world.append(trajectory_points_world)
                mpc = MPC_Controller(
                    trajectory_points_world,
                    desired_v=args_cli.speed,
                    v_max=args_cli.speed,
                    w_max=args_cli.speed
                )
            batch_optimal_points_world = np.array(batch_optimal_points_world)

            batch_all_points_world = []
            for idx in range(all_trajectories_camera.shape[0]):
                all_trajectories_world = []
                for traj_camera in all_trajectories_camera[idx]:
                    traj_world = []
                    for point in traj_camera:
                        point_local = np.array([point[0], point[1], 0.0])
                        point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                        traj_world.append(point_world[:2])
                    all_trajectories_world.append(np.array(traj_world))
                batch_all_points_world.append(all_trajectories_world)
            batch_all_points_world = np.array(batch_all_points_world)

            # A slow mode-conditioned request may finish after the simulator
            # has reset into the next episode.  Check and publish under the
            # same lock order used by reset so an old result can never be
            # written back after the reset cleared planning_output.
            with input_lock:
                generation_is_current = (
                    planning_input.episode_generation == episode_generation
                )
                with output_lock:
                    if generation_is_current:
                        planning_output.trajectory_points_world = batch_optimal_points_world
                        planning_output.all_trajectories_world = batch_all_points_world
                        planning_output.all_trajectories_camera = all_trajectories_camera.copy()
                        planning_output.point_goals_camera = goal.copy()
                        planning_output.all_values_camera = all_values_camera
                        planning_output.mode_debug = mode_debug
                        planning_output.episode_generation = episode_generation
                        planning_output.planning_error = None
                    planning_output.is_planning = False

        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        time.sleep(0.1)


def main():
    if args_cli.episode_start < 0 or args_cli.num_episodes <= 0:
        raise ValueError("episode_start must be non-negative and num_episodes positive")
    episode_end = args_cli.episode_start + args_cli.num_episodes
    scene_list = os.listdir(args_cli.scene_dir)
    scene_list.sort()

    scene_name = scene_list[args_cli.scene_index]
    scene_path = os.path.join(args_cli.scene_dir, scene_name) + "/"
    usd_path, init_path = find_usd_path(scene_path, 'pointgoal')

    print(f"[INFO] Validating episode files in {scene_path}")
    episode_json_files = []
    # The reset event validates IDs from zero through its configured count.
    # Validate that complete prefix here while executing only the requested
    # [episode_start, episode_end) interval below.
    for episode_id in range(episode_end):
        episode_json_path = os.path.join(scene_path, f"episode_{episode_id}.json")
        if not os.path.exists(episode_json_path):
            raise RuntimeError(
                f"Missing episode file: {episode_json_path}\n"
            f"Expected episode files 0 through {episode_end - 1}"
            )
        episode_json_files.append(episode_json_path)

    print(
        f"[INFO] Found episode files 0 through {episode_end - 1} OK; "
        f"running [{args_cli.episode_start}, {episode_end})"
    )

    first_episode_path = episode_json_files[args_cli.episode_start]

    scene_config = SocialNavSceneCfg()
    scene_config.num_envs = args_cli.num_envs
    scene_config.env_spacing = 0.0
    scene_config.terrain = BENCH_TERRAIN_CFG
    scene_config.terrain.usd_path = usd_path
    scene_config.goal = GOAL_CFG
    scene_config.robot = DINGO_CFG
    scene_config.camera_sensor = DINGO_CameraCfg
    scene_config.contact_sensor = DINGO_ContactCfg

    scene_config.people_simulation = True
    scene_config.episode_json_path = first_episode_path

    env_config = DingoSocialNavCfg()
    env_config.scene = scene_config
    env_config.events.reset_pose.params = {
        "episode_json_dir": scene_path,
        "num_episodes": episode_end,
        'height_offset': 0.1,
        'robot_visible': True,
        'light_enabled': False
    }

    env = ManagerBasedRLEnv(env_config)
    env = RslRlVecEnvWrapper(env)
    adjust_usd_scale(scale=args_cli.scene_scale)

    episode_steps = np.zeros((scene_config.num_envs,), dtype=np.int64)

    if scene_name in ['Hospital', 'Jetracer', 'Office']:
        setup_all_lighting(env)
        for _ in range(5):
            simulation_app.update()

    PREHEAT_STEPS = 10
    for _ in range(PREHEAT_STEPS):
        action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
        obs, rewards, dones, infos = env.step(action)

    # Preheating must not consume or partially advance benchmark episode 0.
    # Force a fresh deterministic reset before opening its video/metrics row.
    set_benchmark_episode_ids(
        env.unwrapped,
        np.full(args_cli.num_envs, args_cli.episode_start, dtype=np.int64),
    )
    obs, infos = env.reset()
    infos["observations"] = refresh_camera_after_reset(env, simulation_app)
    validate_episode_start_state(
        env, infos, scene_path, args_cli.episode_start
    )

    camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

    planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
    planning_thread_obj.daemon = True
    planning_thread_obj.start()

    controller = DifferentialController(
        name="simple_control",
        wheel_radius=DINGO_WHEEL_RADIUS,
        wheel_base=DINGO_WHEEL_BASE
    )
    algo = navigator_reset(
        camera_intrinsic.cpu().numpy(),
        batch_size=scene_config.num_envs,
        stop_threshold=args_cli.stop_threshold,
        port=args_cli.port
    )

    if algo == "fallback_algo":
        print("[ERROR] Navigator server connection failed!")
        print(f"[INFO] Please start the server: python server.py --port {args_cli.port}")
        cleanup_simulation(
            env=env,
            simulation_app=simulation_app,
            stop_event=stop_event,
            planning_thread_obj=planning_thread_obj,
        )
        sys.exit(1)

    print(f"[INFO] Connected to navigator server, algorithm: {algo}")

    episode_num = 0
    evaluation_metrics = []
    current_episode_idx = args_cli.episode_start
    save_dir = args_cli.output_dir or "./metrics/socialgoal_%s_%s/%s/" % (
        algo, args_cli.scene_dir.split("/")[-1], scene_path.split("/")[-2])
    if not save_dir.endswith(os.sep):
        save_dir += os.sep
    os.makedirs(save_dir, exist_ok=True)

    euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1))
    fps_writer = [
        imageio.get_writer(
            save_dir + "fps_%d.mp4" % (args_cli.episode_start + i), fps=10
        )
        for i in range(scene_config.num_envs)
    ]

    trajectory_length = np.zeros((scene_config.num_envs))

    social_metrics_trackers = [SocialMetricsTracker() for _ in range(scene_config.num_envs)]

    camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
    camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
    camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
    camera_rot = R.from_quat(camera_rot_quat).as_matrix()

    for i in range(scene_config.num_envs):
        initial_pose = np.array([
            camera_pos[i, 0],
            camera_pos[i, 1],
            np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0])
        ])
        vis_manager[i].reset(initial_robot_pose=initial_pose)

    if scene_config.people_simulation:
        print("Waiting for NavMesh to be ready...")
        wait_count = 0
        while not env.unwrapped.scene.navmesh_ready and wait_count < 100:
            simulation_app.update()
            wait_count += 1
            if wait_count % 20 == 0:
                print(f"  Waiting... ({wait_count}/100)")

        if env.unwrapped.scene.navmesh_ready:
            print("NavMesh ready.")
        else:
            print("Warning: NavMesh not ready, continuing without people")

    print("[INFO] Waiting for people to respawn...")
    wait_count = 0
    max_wait = 500
    while env.unwrapped.scene._people_setup_in_progress and wait_count < max_wait:
        simulation_app.update()
        wait_count += 1
        if wait_count % 50 == 0:
            print(f"  Waiting... ({wait_count}/{max_wait})")

    if hasattr(env.env, '_recent_positions'):
        env.env._recent_positions.clear()
    frame_count = 0

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                goals = infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]
                images = infos['observations']['rgb'].cpu().numpy()[:, :, :, 0:3]
                depths = infos['observations']['depth'].cpu().numpy()[:, :, :]

                camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
                camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
                camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
                camera_rot = R.from_quat(camera_rot_quat).as_matrix()

                with input_lock:
                    planning_input.current_goal = goals.copy()
                    planning_input.current_image = images.copy()
                    planning_input.current_depth = depths.copy()
                    planning_input.camera_pos = camera_pos.copy()
                    planning_input.camera_rot = camera_rot.copy()
                    planning_input.episode_generation = current_episode_idx

                robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
                robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()

                x0 = np.stack([
                    camera_pos[:, 0], camera_pos[:, 1],
                    np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]),
                    [robot_vel], [robot_ang_vel]
                ], axis=-1)
                current_trajectory = None
                current_all_trajectories = None
                current_all_values = None
                current_all_trajectories_camera = None
                current_point_goals_camera = None
                current_mode_debug = None
                with output_lock:
                    if (
                        planning_output.trajectory_points_world is not None
                        and planning_output.episode_generation == current_episode_idx
                    ):
                        current_trajectory = planning_output.trajectory_points_world.copy()
                        current_all_trajectories = planning_output.all_trajectories_world.copy()
                        current_all_values = planning_output.all_values_camera.copy()
                        current_all_trajectories_camera = planning_output.all_trajectories_camera.copy()
                        current_point_goals_camera = planning_output.point_goals_camera.copy()
                        current_mode_debug = planning_output.mode_debug

                if current_trajectory is not None:
                    goal_world = camera_pos[0] + camera_rot[0] @ np.array([goals[0][0], goals[0][1], 0.0])
                    action_list = []
                    for i in range(args_cli.num_envs):
                        people_positions, people_char_paths, pos_get_flag = get_people_positions(env)

                        if pos_get_flag:
                            social_metrics_trackers[i].update(camera_pos[i], people_positions, env.unwrapped.step_dt)

                            vis_image = vis_manager[i].visualize_trajectory_global_with_people(
                                images[i], depths[i][:, :, None], camera_intrinsic.cpu().numpy(),
                                current_trajectory[i],
                                robot_pose=x0[i],
                                goal_position=goal_world[:2],
                                all_trajectories_points=current_all_trajectories[i],
                                all_trajectories_values=current_all_values[i],
                                people_positions=people_positions,
                                people_positions_dict=people_char_paths,
                            )
                        else:
                            vis_image = vis_manager[i].visualize_trajectory_global(
                                images[i], depths[i][:, :, None], camera_intrinsic.cpu().numpy(),
                                current_trajectory[i],
                                robot_pose=x0[i],
                                goal_position=goal_world[:2],
                                all_trajectories_points=current_all_trajectories[i],
                                all_trajectories_values=current_all_values[i]
                            )

                        use_mode_debug = (algo in MODE_DEBUG_ALGOS and current_mode_debug
                                          and i < len(current_mode_debug))
                        if use_mode_debug:
                            esdf_view = draw_esdf_candidates(
                                vis_image.shape[0], current_all_trajectories_camera[i],
                                current_point_goals_camera[i], current_mode_debug[i],
                                current_all_values[i])
                            # Keep the complete native SocialNav visualization.  Its
                            # lower-left global map is wider than the RGB camera pane;
                            # cropping to camera width hid the map beneath the ESDF.
                            vis_image = np.concatenate((vis_image, esdf_view), axis=1)
                            vis_image = draw_mode_debug_panel(vis_image, current_mode_debug[i])

                        if mpc is None:
                            continue
                        opt_u_controls, opt_x_states = mpc.solve(x0[i, :3])
                        v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                        action = torch.tensor([v, w], device="cuda:0")
                        action_cpu = action.cpu().numpy()
                        joint_velocities = controller.forward(action_cpu).joint_velocities
                        action_list.append(joint_velocities)

                        try:
                            vis_image = draw_box_with_text(vis_image, 0, 0, 430, 50, "cmd lin.:%.2f ang.:%.2f" % (v, w))
                            vis_image = draw_box_with_text(vis_image, 0, 50, 430, 50, "actual lin.:%.2f ang.:%.2f" % (robot_vel, robot_ang_vel))
                            if current_all_values is not None:
                                vis_image = draw_box_with_text(
                                    vis_image, 0, 770, 430, 50,
                                    "critic max:%.2f min:%.2f" % (np.max(current_all_values[i]), np.min(current_all_values[i]))
                                )
                            vis_image = draw_box_with_text(
                                vis_image, 0, 820, 430, 50,
                                "point goal:(%.2f, %.2f)" % (goals[i][0], goals[i][1])
                            )
                            if frame_count > 0:
                                if frame_count == 1:
                                    print(
                                        f"[VIDEO FIRST FRAME] episode={current_episode_idx} "
                                        f"goal_local={goals[i].tolist()}",
                                        flush=True,
                                    )
                                cv2.imwrite(f"frame_{algo}_socialnav_{scene_name}.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                                fps_writer[i].append_data(vis_image)
                            frame_count += 1
                        except Exception:
                            pass

                    action = torch.as_tensor(np.stack(action_list, axis=0), device="cuda:0")
                    set_benchmark_episode_ids(
                        env.unwrapped,
                        np.full(
                            args_cli.num_envs,
                            (current_episode_idx + 1) % episode_end,
                            dtype=np.int64,
                        ),
                    )
                    obs, rewards, dones, infos = env.step(action)
                    episode_steps += 1
                    trajectory_length += (infos['observations']['policy'][:, 0] * env.unwrapped.step_dt).cpu().numpy()
                else:
                    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
                    set_benchmark_episode_ids(
                        env.unwrapped,
                        np.full(
                            args_cli.num_envs,
                            (current_episode_idx + 1) % episode_end,
                            dtype=np.int64,
                        ),
                    )
                    obs, rewards, dones, infos = env.step(action)
                    episode_steps += 1
                    print("Trajectory not ready; zero action")

                for i in range(args_cli.num_envs):
                    if dones[i] == True:
                        episode_num += 1
                        navigator_reset(env_id=i, port=args_cli.port)
                        success_flag = float(np.sqrt(np.square(goals[i]).sum()) < 1.5)
                        spl = np.clip(euclidean[i] / max(trajectory_length[i], 0.01), 0, 1) * success_flag

                        social_metrics = social_metrics_trackers[i].get_metrics()

                        evaluation_metrics.append({
                            'episode': current_episode_idx,
                            'success': success_flag,
                            'spl': spl,
                            'time_to_goal': episode_steps[i] * env.env.step_dt,
                            'distance': euclidean[i],
                            'trajectory_length': trajectory_length[i],
                            'collision': social_metrics['collision'],
                            'collision_count': social_metrics['collision_count'],
                            'min_distance': social_metrics['min_distance'],
                            'avg_distance': social_metrics['avg_distance'],
                            'psi_count': social_metrics['psi_count'],
                            'psi_time': social_metrics['psi_time'],
                            'total_time': social_metrics['total_time'],
                            'PSC': social_metrics['PSC'],
                            'SC': social_metrics['SC'],
                        })

                        print(f"\n=== Metrics of Episode {current_episode_idx} in Scene {scene_name} ===")
                        for key, value in evaluation_metrics[-1].items():
                            if isinstance(value, float):
                                print(f"  {key}: {value:.3f}")
                            else:
                                print(f"  {key}: {value}")

                        fps_writer[i].close()
                        current_episode_idx += 1
                        write_metrics(evaluation_metrics, save_dir + "metric.csv")

                        if current_episode_idx >= episode_end:
                            print(
                                f"\n[INFO] Episodes {args_cli.episode_start} "
                                f"through {episode_end - 1} completed!"
                            )
                            print(f"[INFO] Final metrics saved to {save_dir}metric.csv")
                            summary = write_socialnav_summary(
                                evaluation_metrics, save_dir
                            )
                            print(
                                "[INFO] Final summary: "
                                f"success={summary['success_rate']:.3f} "
                                f"spl={summary['mean_spl']:.3f} "
                                f"collision={summary['collision_rate']:.3f} "
                                f"psi_time={summary['psi_time_s_total']:.3f}s",
                                flush=True,
                            )

                            # The finally block performs the complete cleanup
                            # exactly once.  Calling it here as well used to
                            # unload AnimationGraph/Replicator twice.
                            return

                        if hasattr(env.env, '_recent_positions'):
                            env.env._recent_positions.clear()
                        # Invalidate the old generation atomically with
                        # clearing its output.  An in-flight planner can then
                        # finish, but its result will be discarded.
                        with input_lock:
                            planning_input.episode_generation = current_episode_idx
                            with output_lock:
                                planning_output.trajectory_points_world = None
                                planning_output.all_trajectories_world = None
                                planning_output.all_trajectories_camera = None
                                planning_output.point_goals_camera = None
                                planning_output.all_values_camera = None
                                planning_output.mode_debug = None
                                planning_output.episode_generation = -1
                                planning_output.is_planning = False
                                planning_output.planning_error = None

                        new_episode_path = os.path.join(scene_path, f"episode_{current_episode_idx}.json")
                        env.unwrapped.scene.cfg.episode_json_path = new_episode_path

                        if env.unwrapped.scene.people is not None or env.unwrapped.scene._people_setup_in_progress:
                            while env.unwrapped.scene._people_setup_in_progress:
                                simulation_app.update()

                            asyncio.ensure_future(env.unwrapped.scene._reset_people_for_episode(new_episode_path))

                            wait_count = 0
                            while not env.unwrapped.scene._people_setup_in_progress and wait_count < 10:
                                simulation_app.update()
                                wait_count += 1

                            print("[INFO] Waiting for people to respawn...")
                            wait_count = 0
                            max_wait = 500
                            while env.unwrapped.scene._people_setup_in_progress and wait_count < max_wait:
                                simulation_app.update()
                                wait_count += 1
                                if wait_count % 50 == 0:
                                    print(f"  Waiting... ({wait_count}/{max_wait})")

                        # People setup advances Kit asynchronously and may
                        # leave observations from the terminal episode cached.
                        # Reset only after it has finished, then validate this
                        # final state before opening the next episode's writer.
                        set_benchmark_episode_ids(
                            env.unwrapped,
                            np.full(
                                args_cli.num_envs,
                                current_episode_idx,
                                dtype=np.int64,
                            ),
                        )
                        obs, infos = env.reset()
                        infos["observations"] = refresh_camera_after_reset(
                            env, simulation_app
                        )
                        validate_episode_start_state(
                            env, infos, scene_path, current_episode_idx
                        )
                        if hasattr(env.env, '_recent_positions'):
                            env.env._recent_positions.clear()

                        euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]).sum(axis=-1))[i]
                        fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4" % current_episode_idx, fps=10)
                        trajectory_length[i] = 0.0

                        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
                        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
                        camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
                        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
                        initial_pose = np.array([
                            camera_pos[i, 0], camera_pos[i, 1],
                            np.arctan2(camera_rot[i, 1, 0], camera_rot[i, 0, 0])
                        ])
                        vis_manager[i].reset(initial_robot_pose=initial_pose)
                        frame_count = 0
                        episode_steps[i] = 0
                        social_metrics_trackers[i].reset()

                        break

                if episode_num > args_cli.num_episodes:
                    break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_simulation(
            env=env if 'env' in locals() else None,
            simulation_app=simulation_app if 'simulation_app' in locals() else None,
            stop_event=stop_event,
            planning_thread_obj=(
                planning_thread_obj if 'planning_thread_obj' in locals() else None
            ),
            fps_writer=fps_writer if 'fps_writer' in locals() else None,
        )


if __name__ == "__main__":
    main()
