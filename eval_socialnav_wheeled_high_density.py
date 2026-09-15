#!/usr/bin/env python3
"""Prepare deterministic high-density DynBench episodes, then run SocialNav.

This wrapper never edits the source DynBench dataset or the official
``eval_socialnav_wheeled.py`` evaluator.  It builds an isolated scene overlay
inside the requested output directory and then executes the official evaluator
against that overlay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="High-density wrapper for the official FLUX SocialNav evaluator"
    )
    parser.add_argument(
        "--scene_dir",
        default="/workspace/datasets/DynBench/isaacsim_scene",
    )
    parser.add_argument("--scene_index", type=int, default=0)
    parser.add_argument("--scene_scale", type=float, default=1.0)
    parser.add_argument("--stop_threshold", type=float, default=-3.0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=14116)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_pedestrians", type=int, default=30)
    parser.add_argument("--density_seed", type=int, default=20260912)
    parser.add_argument("--min_spawn_separation", type=float, default=0.8)
    parser.add_argument("--min_robot_clearance", type=float, default=1.5)
    parser.add_argument("--min_goal_clearance", type=float, default=1.0)
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_known_args()


def character_name(index: int) -> str:
    return "Character" if index == 0 else f"Character_{index:02d}"


def xy(position: Any) -> tuple[float, float]:
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        raise ValueError(f"Invalid position: {position!r}")
    return float(position[0]), float(position[1])


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_episode(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    episode = data.get("episode")
    if not isinstance(episode, dict):
        raise ValueError(f"Missing episode object in {path}")
    characters = episode.get("characters")
    if not isinstance(characters, dict):
        raise ValueError(f"Missing characters object in {path}")
    return data


def validate_commands(commands: Any) -> bool:
    if not isinstance(commands, list) or not commands:
        return False
    for command in commands:
        if not isinstance(command, dict) or not command.get("cmd"):
            return False
        if command.get("cmd") == "GoTo" and not command.get("path"):
            return False
    return True


def path_spawn_variants(
    spawn: dict[str, Any],
    commands: list[dict[str, Any]],
    rng: random.Random,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]:
    """Return the original spawn plus safe-to-loop positions on existing paths."""
    variants: list[
        tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]
    ] = [(copy.deepcopy(spawn), copy.deepcopy(commands), {"source": "spawn"})]

    path_variants: list[
        tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]
    ] = []
    for command_index, command in enumerate(commands):
        if command.get("cmd") != "GoTo":
            continue
        path = command.get("path")
        if not isinstance(path, list) or len(path) < 5:
            continue

        # Sample existing NavMesh path points rather than inventing translated
        # coordinates.  Keep at least two points in both travel directions.
        stride = max(4, len(path) // 10)
        indices = list(range(1, len(path) - 2, stride))
        rng.shuffle(indices)
        for path_index in indices:
            segment = copy.deepcopy(path[path_index:])
            if len(segment) < 3:
                continue
            forward_goal = segment[-1]
            backward = list(reversed(copy.deepcopy(segment)))
            backward_goal = backward[-1]
            patrol_commands = [
                {
                    "cmd": "GoTo",
                    "params": [str(value) for value in forward_goal[:3]],
                    "path": segment,
                },
                {
                    "cmd": "GoTo",
                    "params": [str(value) for value in backward_goal[:3]],
                    "path": backward,
                },
            ]
            derived_spawn = copy.deepcopy(spawn)
            derived_spawn["pos"] = copy.deepcopy(segment[0])
            next_point = segment[1]
            derived_spawn["rot"] = math.atan2(
                float(next_point[1]) - float(segment[0][1]),
                float(next_point[0]) - float(segment[0][0]),
            )
            path_variants.append(
                (
                    derived_spawn,
                    patrol_commands,
                    {
                        "source": "precomputed_path",
                        "command_index": command_index,
                        "path_index": path_index,
                    },
                )
            )

    rng.shuffle(path_variants)
    variants.extend(path_variants)
    return variants


def safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Refusing to replace symlink: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Refusing to replace existing path: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def build_overlay(
    source_scene_root: Path,
    source_scene: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    overlay_base = output_dir / "high_density_dataset"
    overlay_scene_root = overlay_base / "isaacsim_scene"
    overlay_scene = overlay_scene_root / source_scene.name
    overlay_scene.mkdir(parents=True, exist_ok=True)

    dynbench_root = source_scene_root.parent
    for entry in dynbench_root.iterdir():
        if entry.name != source_scene_root.name:
            safe_symlink(entry, overlay_base / entry.name)

    for entry in source_scene.iterdir():
        if not (entry.name.startswith("episode_") and entry.suffix == ".json"):
            safe_symlink(entry, overlay_scene / entry.name)

    return overlay_scene_root, overlay_scene


def generate_episode(
    base_id: int,
    source_episodes: dict[int, dict[str, Any]],
    target_count: int,
    seed: int,
    min_spawn_separation: float,
    min_robot_clearance: float,
    min_goal_clearance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generated = copy.deepcopy(source_episodes[base_id])
    episode = generated["episode"]
    characters = episode["characters"]
    spawn_positions = characters.get("spawn_positions")
    commands = characters.get("commands")
    if not isinstance(spawn_positions, dict) or not isinstance(commands, dict):
        raise ValueError(f"Episode {base_id} has invalid character dictionaries")

    original_count = int(characters.get("num_characters", -1))
    expected_names = [character_name(index) for index in range(original_count)]
    if original_count <= 0 or list(spawn_positions) != expected_names:
        raise ValueError(
            f"Episode {base_id} character names are not a continuous canonical prefix"
        )
    if set(commands) != set(expected_names):
        raise ValueError(f"Episode {base_id} command names do not match spawns")
    if target_count < original_count:
        raise ValueError(
            f"num_pedestrians={target_count} is below source count {original_count}"
        )

    selected_positions = [xy(spawn_positions[name]["pos"]) for name in expected_names]
    robot_position = xy(episode["robot"]["start_pos"])
    goal_position = xy(episode["robot"]["goal_pos"])
    provenance: list[dict[str, Any]] = []

    rng = random.Random(seed + base_id * 1_000_003)
    donor_ids = list(source_episodes)
    rng.shuffle(donor_ids)

    for donor_id in donor_ids:
        donor_characters = source_episodes[donor_id]["episode"]["characters"]
        donor_spawns = donor_characters.get("spawn_positions", {})
        donor_commands = donor_characters.get("commands", {})
        donor_names = list(donor_spawns)
        rng.shuffle(donor_names)

        for donor_name in donor_names:
            if len(spawn_positions) >= target_count:
                break
            spawn = donor_spawns.get(donor_name)
            command_list = donor_commands.get(donor_name)
            if not isinstance(spawn, dict) or not validate_commands(command_list):
                continue
            for candidate_spawn, candidate_commands, source_info in path_spawn_variants(
                spawn, command_list, rng
            ):
                candidate = xy(candidate_spawn.get("pos"))
                if distance(candidate, robot_position) < min_robot_clearance:
                    continue
                if distance(candidate, goal_position) < min_goal_clearance:
                    continue
                if any(
                    distance(candidate, existing) < min_spawn_separation
                    for existing in selected_positions
                ):
                    continue

                new_name = character_name(len(spawn_positions))
                spawn_positions[new_name] = candidate_spawn
                commands[new_name] = candidate_commands
                selected_positions.append(candidate)
                provenance.append(
                    {
                        "name": new_name,
                        "donor_episode": donor_id,
                        "donor_character": donor_name,
                        "spawn_xy": list(candidate),
                        **source_info,
                    }
                )
                break

        if len(spawn_positions) >= target_count:
            break

    if len(spawn_positions) != target_count:
        raise RuntimeError(
            f"Episode {base_id}: only found {len(spawn_positions)} safe pedestrians, "
            f"requested {target_count}"
        )

    characters["num_characters"] = target_count
    canonical_names = [character_name(index) for index in range(target_count)]
    if list(spawn_positions) != canonical_names or list(commands) != canonical_names:
        raise AssertionError(f"Episode {base_id}: generated names are not canonical")
    return generated, provenance


def write_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.episode_start < 0 or args.num_episodes <= 0:
        raise ValueError("episode_start must be non-negative and num_episodes positive")
    if args.num_pedestrians <= 0:
        raise ValueError("num_pedestrians must be positive")
    for field in (
        "min_spawn_separation",
        "min_robot_clearance",
        "min_goal_clearance",
    ):
        if getattr(args, field) < 0:
            raise ValueError(f"{field} must be non-negative")

    source_scene_root = Path(args.scene_dir).resolve()
    scene_names = sorted(path.name for path in source_scene_root.iterdir() if path.is_dir())
    try:
        scene_name = scene_names[args.scene_index]
    except IndexError as error:
        raise ValueError(
            f"scene_index {args.scene_index} is invalid for {scene_names}"
        ) from error
    source_scene = source_scene_root / scene_name
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_scene_root, overlay_scene = build_overlay(
        source_scene_root, source_scene, output_dir
    )

    available_paths = sorted(
        source_scene.glob("episode_*.json"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    source_episodes = {
        int(path.stem.split("_")[-1]): load_episode(path) for path in available_paths
    }
    episode_end = args.episode_start + args.num_episodes
    required_ids = range(episode_end)
    missing = [episode_id for episode_id in required_ids if episode_id not in source_episodes]
    if missing:
        raise FileNotFoundError(f"Missing source episode IDs: {missing}")

    manifest: dict[str, Any] = {
        "schema": "flux_socialnav_high_density_v1",
        "source_scene_dir": str(source_scene),
        "overlay_scene_dir": str(overlay_scene),
        "scene_name": scene_name,
        "source_pedestrians": int(
            source_episodes[args.episode_start]["episode"]["characters"]["num_characters"]
        ),
        "num_pedestrians": args.num_pedestrians,
        "density_seed": args.density_seed,
        "min_spawn_separation_m": args.min_spawn_separation,
        "min_robot_clearance_m": args.min_robot_clearance,
        "min_goal_clearance_m": args.min_goal_clearance,
        "episode_start": args.episode_start,
        "num_episodes": args.num_episodes,
        "episodes": {},
    }

    for episode_id in required_ids:
        generated, provenance = generate_episode(
            episode_id,
            source_episodes,
            args.num_pedestrians,
            args.density_seed,
            args.min_spawn_separation,
            args.min_robot_clearance,
            args.min_goal_clearance,
        )
        destination = overlay_scene / f"episode_{episode_id}.json"
        digest = write_json(destination, generated)
        generated_characters = generated["episode"]["characters"]
        manifest["episodes"][str(episode_id)] = {
            "sha256": digest,
            "num_characters": generated_characters["num_characters"],
            "spawn_count": len(generated_characters["spawn_positions"]),
            "command_count": len(generated_characters["commands"]),
            "added": provenance,
        }

    manifest_path = output_dir / "density_manifest.json"
    write_json(manifest_path, manifest)
    return overlay_scene_root, manifest_path


def main() -> None:
    args, unknown_args = parse_args()
    overlay_scene_root, manifest_path = prepare_dataset(args)
    print(
        f"[HIGH DENSITY] prepared {args.num_episodes} episode(s), "
        f"{args.num_pedestrians} pedestrians each",
        flush=True,
    )
    print(f"[HIGH DENSITY] manifest={manifest_path}", flush=True)
    if args.prepare_only:
        return

    official_evaluator = Path(__file__).resolve().with_name("eval_socialnav_wheeled.py")
    command = [
        sys.executable,
        str(official_evaluator),
        "--scene_dir",
        str(overlay_scene_root),
        "--scene_index",
        "0",
        "--scene_scale",
        str(args.scene_scale),
        "--stop_threshold",
        str(args.stop_threshold),
        "--num_envs",
        str(args.num_envs),
        "--num_episodes",
        str(args.num_episodes),
        "--episode_start",
        str(args.episode_start),
        "--speed",
        str(args.speed),
        "--port",
        str(args.port),
        "--gpu_id",
        str(args.gpu_id),
        "--output_dir",
        str(Path(args.output_dir).resolve()),
        *unknown_args,
    ]
    print(f"[HIGH DENSITY] exec={' '.join(command)}", flush=True)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
