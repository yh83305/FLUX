"""Generate ROS-style 2-D occupancy maps from DynBench USD scenes."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label


SCENE_USD_NAMES = {
    "Full_Warehouse": "full_warehouse.usd",
    "Hospital": "hospital.usd",
    "Jetracer": "jetracer.usd",
    "Office": "office.usd",
    "Warehouse": "warehouse.usd",
    "Warehouse_multiple_shelves": "warehouse_multiple_shelves.usd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root", default="/data/DynBench/isaacsim_scene"
    )
    parser.add_argument(
        "--scene", action="append", dest="scenes",
        help="Scene folder name; repeat it. Defaults to all six scenes.",
    )
    parser.add_argument(
        "--output-root", default="/data/DynBench/occupancy_maps"
    )
    parser.add_argument(
        "--flat-output", action="store_true",
        help="Write files directly under output-root instead of a scene subfolder.",
    )
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--padding", type=float, default=2.0)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--min-height", type=float, default=0.15)
    parser.add_argument("--max-height", type=float, default=1.35)
    parser.add_argument("--height-step", type=float, default=0.20)
    parser.add_argument(
        "--multi-height", action="store_true",
        help="Rasterize all height layers; default uses only the first layer "
        "because repeated OMAP generation can terminate Isaac Sim.",
    )
    parser.add_argument(
        "--max-endpoint-snap", type=float, default=0.15,
        help="Maximum accepted start/goal distance to a free pixel.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args()


def load_episode_points(scene_dir: Path) -> tuple[np.ndarray, list[dict]]:
    episodes = []
    points = []
    paths = sorted(
        scene_dir.glob("episode_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"No episode_*.json under {scene_dir}")
    for path in paths:
        episode = json.loads(path.read_text(encoding="utf-8"))["episode"]
        robot = episode["robot"]
        start = np.asarray(robot["start_pos"][:2], dtype=np.float64)
        goal = np.asarray(robot["goal_pos"][:2], dtype=np.float64)
        episode_id = int(episode.get("episode_id", path.stem.split("_")[-1]))
        episodes.append({"episode": episode_id, "start": start, "goal": goal})
        points.extend((start, goal))
    return np.stack(points), episodes


def resolve_scene_usd(scene_dir: Path, scene_name: str) -> Path:
    preferred = scene_dir / SCENE_USD_NAMES.get(scene_name, "")
    if preferred.is_file():
        return preferred
    candidates = sorted(
        path for path in scene_dir.glob("*.usd")
        if "nomdl" not in path.name.lower()
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Cannot identify one scene USD under {scene_dir}: {candidates}"
        )
    return candidates[0]


def grid_bounds(points: np.ndarray, padding: float, resolution: float):
    lower = np.floor((points.min(axis=0) - padding) / resolution) * resolution
    upper = np.ceil((points.max(axis=0) + padding) / resolution) * resolution
    return lower, upper


def generate_physx_map(
    usd_path: Path,
    lower_xy: np.ndarray,
    upper_xy: np.ndarray,
    args: argparse.Namespace,
    app,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import omni.kit.app
    import omni.physx
    import omni.timeline
    import omni.usd
    from isaacsim.core.utils.stage import open_stage
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    print("[DynBench OMap] enabling omap extension", flush=True)
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.asset.gen.omap", True)
    app.update()
    from isaacsim.asset.gen.omap.bindings import _omap
    print("[DynBench OMap] opening USD", flush=True)
    opened = open_stage(str(usd_path))
    print(f"[DynBench OMap] USD opened={opened}", flush=True)
    if not opened:
        raise RuntimeError(f"Failed to open {usd_path}")
    context = omni.usd.get_context()
    loading_start = time.monotonic()
    while context.get_stage_loading_status()[2] > 0:
        if time.monotonic() - loading_start > 120.0:
            status = context.get_stage_loading_status()
            raise TimeoutError(f"Timed out waiting for USD dependencies: {status}")
        app.update()
    stage = context.get_stage()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    world_range = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    collision_count = 0
    geometry_count = 0
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
            continue
        geometry_count += 1
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        if prim.IsA(UsdGeom.Mesh):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            approximation = mesh_collision.GetApproximationAttr()
            if not approximation.HasAuthoredValueOpinion():
                mesh_collision.CreateApproximationAttr().Set("none")
        collision_count += 1
    print(
        f"[DynBench OMap] geometry={geometry_count} collisions={collision_count} "
        f"world_min={list(world_range.GetMin())} world_max={list(world_range.GetMax())}",
        flush=True,
    )
    for _ in range(4):
        app.update()
    if not stage.GetPrimAtPath("/World/physicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(12):
        app.update()

    if args.multi_height:
        heights = np.arange(
            float(args.min_height),
            float(args.max_height) + 0.5 * float(args.height_step),
            float(args.height_step),
        ) + float(args.floor_z)
    else:
        heights = np.asarray([float(args.min_height) + float(args.floor_z)])
    layers = []
    dimensions = None
    min_bound = max_bound = None
    for height in heights:
        # OMAP keeps internal state across generate() calls. Reacquiring the
        # interface per height avoids a silent Isaac Sim shutdown when a stage
        # is rasterized repeatedly at different z values.
        generator = _omap.acquire_omap_interface()
        try:
            generator.set_cell_size(float(args.resolution))
            generator.set_transform(
                (0.0, 0.0, float(height)),
                (float(lower_xy[0]), float(lower_xy[1]), 0.0),
                (float(upper_xy[0]), float(upper_xy[1]), 0.0),
            )
            generator.update()
            app.update()
            generator.generate()
            app.update()
            layer_dimensions = np.asarray(generator.get_dimensions(), dtype=np.int64)
            raw = np.asarray(generator.get_buffer(), dtype=np.float32).copy()
            layer_min_bound = np.asarray(
                generator.get_min_bound(), dtype=np.float64
            )[:2]
            layer_max_bound = np.asarray(
                generator.get_max_bound(), dtype=np.float64
            )[:2]
        finally:
            _omap.release_omap_interface(generator)
        expected = int(layer_dimensions[0] * layer_dimensions[1])
        if len(raw) != expected or expected == 0:
            raise RuntimeError(
                f"Occupancy generator returned {len(raw)} cells for "
                f"{layer_dimensions} at z={height:.3f}"
            )
        if dimensions is not None and not np.array_equal(
            dimensions, layer_dimensions
        ):
            raise RuntimeError("Occupancy dimensions changed between height layers")
        dimensions = layer_dimensions
        layers.append(raw)
        unique, counts = np.unique(raw, return_counts=True)
        print(
            f"[DynBench OMap] z={height:.3f} "
            f"buffer_values={dict(zip(unique.tolist(), counts.tolist()))}",
            flush=True,
        )
        min_bound = layer_min_bound
        max_bound = layer_max_bound
    stacked = np.stack(layers)
    raw = np.full(stacked.shape[1], 0.5, dtype=np.float32)
    raw[np.all(np.isclose(stacked, 0.0), axis=0)] = 0.0
    raw[np.any(np.isclose(stacked, 1.0), axis=0)] = 1.0
    # The generator buffer is row-major from min-y to max-y. Images grow
    # downward, so flip only Y to obtain standard map image coordinates.
    values = raw.reshape(int(dimensions[1]), int(dimensions[0]))
    free_image = np.flipud(np.isclose(values, 0.0)).copy()
    timeline.stop()
    return free_image, min_bound, max_bound


def world_to_pixel(
    xy: np.ndarray, origin_xy: np.ndarray, resolution: float, height: int,
) -> np.ndarray:
    px = np.rint((xy[..., 0] - origin_xy[0]) / resolution).astype(np.int64)
    bottom_y = np.rint((xy[..., 1] - origin_xy[1]) / resolution).astype(np.int64)
    py = height - 1 - bottom_y
    return np.stack((py, px), axis=-1)


def validate_endpoints(
    free: np.ndarray,
    origin_xy: np.ndarray,
    episodes: list[dict],
    resolution: float,
    maximum_snap: float,
) -> dict:
    if not bool(free.any()):
        raise RuntimeError("Generated occupancy contains no free pixels")
    distance, nearest = distance_transform_edt(~free, return_indices=True)
    components, _ = label(free, structure=np.ones((3, 3), dtype=np.uint8))
    height, width = free.shape
    rows = []
    valid_count = connected_count = 0
    for episode in episodes:
        result = {"episode": int(episode["episode"])}
        labels = []
        valid = True
        for key in ("start", "goal"):
            pixel = world_to_pixel(
                episode[key][None], origin_xy, resolution, height
            )[0]
            inside = bool(
                0 <= pixel[0] < height and 0 <= pixel[1] < width
            )
            if inside:
                snap_pixels = float(distance[pixel[0], pixel[1]])
                nearest_pixel = nearest[:, pixel[0], pixel[1]].astype(int)
                component = int(components[tuple(nearest_pixel)])
            else:
                snap_pixels = float("inf")
                nearest_pixel = np.asarray([-1, -1])
                component = 0
            snap_m = snap_pixels * resolution
            accepted = inside and snap_m <= maximum_snap and component > 0
            valid &= accepted
            labels.append(component)
            result[key] = {
                "world_xy": episode[key].tolist(),
                "pixel_yx": pixel.tolist(),
                "inside": inside,
                "snap_distance_m": snap_m,
                "nearest_free_pixel_yx": nearest_pixel.tolist(),
                "component": component,
                "valid": accepted,
            }
        connected = valid and labels[0] == labels[1]
        result["connected"] = connected
        valid_count += int(valid)
        connected_count += int(connected)
        rows.append(result)
    return {
        "episodes": len(episodes),
        "valid_endpoint_episodes": valid_count,
        "connected_episodes": connected_count,
        "all_endpoints_valid": valid_count == len(episodes),
        "all_pairs_connected": connected_count == len(episodes),
        "details": rows,
    }


def save_outputs(
    scene_name: str,
    usd_path: Path,
    free: np.ndarray,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    episodes: list[dict],
    args: argparse.Namespace,
) -> dict:
    output_dir = Path(args.output_root).expanduser().resolve()
    if not args.flat_output:
        output_dir /= scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # MapMetadata treats origin as the centre of the lower-left pixel.
    origin_xy = min_bound + 0.5 * float(args.resolution)
    image_path = output_dir / "occupancy.png"
    Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).save(image_path)
    yaml_path = output_dir / "occupancy.yaml"
    yaml_path.write_text(
        "\n".join((
            "image: occupancy.png",
            f"resolution: {float(args.resolution):.8f}",
            f"origin: [{origin_xy[0]:.8f}, {origin_xy[1]:.8f}, 0.0]",
            "negate: 0",
            "occupied_thresh: 0.65",
            "free_thresh: 0.196",
            "",
        )),
        encoding="utf-8",
    )
    validation = validate_endpoints(
        free, origin_xy, episodes, float(args.resolution),
        float(args.max_endpoint_snap),
    )
    report = {
        "schema": "dynbench_physx_occupancy_v1",
        "scene": scene_name,
        "scene_usd": str(usd_path),
        "resolution_m": float(args.resolution),
        "floor_z_m": float(args.floor_z),
        "height_range_m": [float(args.min_height), float(args.max_height)],
        "height_step_m": float(args.height_step),
        "origin_xy_m": origin_xy.tolist(),
        "generator_min_bound_xy_m": min_bound.tolist(),
        "generator_max_bound_xy_m": max_bound.tolist(),
        "image_shape_hw": list(free.shape),
        "free_fraction": float(free.mean()),
        "validation": validation,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    preview = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(preview)
    for episode in episodes:
        for key, color in (("start", "#00cc44"), ("goal", "#ff3344")):
            py, px = world_to_pixel(
                episode[key][None], origin_xy, float(args.resolution),
                free.shape[0],
            )[0]
            if 0 <= px < free.shape[1] and 0 <= py < free.shape[0]:
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    preview.save(output_dir / "preview.png")
    return report


def build_scene(scene_name: str, args: argparse.Namespace, app) -> dict:
    scene_dir = Path(args.scene_root).expanduser().resolve() / scene_name
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)
    usd_path = resolve_scene_usd(scene_dir, scene_name)
    points, episodes = load_episode_points(scene_dir)
    lower, upper = grid_bounds(points, float(args.padding), float(args.resolution))
    print(
        f"[DynBench OMap] scene={scene_name} usd={usd_path} "
        f"bounds={lower.tolist()}..{upper.tolist()}", flush=True,
    )
    free, min_bound, max_bound = generate_physx_map(
        usd_path, lower, upper, args, app
    )
    report = save_outputs(
        scene_name, usd_path, free, min_bound, max_bound, episodes, args
    )
    validation = report["validation"]
    print(
        f"[DynBench OMap] scene={scene_name} shape={free.shape} "
        f"free={report['free_fraction']:.3f} "
        f"valid={validation['valid_endpoint_episodes']}/{len(episodes)} "
        f"connected={validation['connected_episodes']}/{len(episodes)}",
        flush=True,
    )
    return report


def main() -> None:
    args = parse_args()
    if args.resolution <= 0 or args.padding < 0:
        raise ValueError("resolution must be positive and padding non-negative")
    if not args.min_height <= args.max_height:
        raise ValueError("min-height cannot exceed max-height")
    if args.height_step <= 0.0:
        raise ValueError("height-step must be positive")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": bool(args.headless)})
    try:
        scenes = args.scenes or list(SCENE_USD_NAMES)
        for scene in scenes:
            build_scene(scene, args, app)
    finally:
        app.close()


if __name__ == "__main__":
    main()
