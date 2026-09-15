#!/usr/bin/env python3
"""Generate an isolated, deterministic FLUX interaction stress-test overlay."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROBOT_SPEED_MPS = 0.5
PEDESTRIAN_SPEED_MPS = 1.0
PATH_GRID_M = 0.6
MAX_INTERSECTION_DISTANCE_M = 0.65
BACKGROUND_ROUTE_CLEARANCE_M = 1.2
BACKGROUND_SPAWN_SEPARATION_M = 0.8
ROBOT_ENDPOINT_CLEARANCE_M = 1.5
CRITICAL_IDLE_S = 10000.0


@dataclass(frozen=True)
class PathRecord:
    episode_id: int
    character: str
    command_index: int
    points: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class Intersection:
    robot: PathRecord
    pedestrian: PathRecord
    robot_index: int
    pedestrian_index: int
    distance: float
    angle_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 10-episode Full_Warehouse interaction golden suite"
    )
    parser.add_argument(
        "--scene_dir",
        default="/workspace/datasets/DynBench/isaacsim_scene",
    )
    parser.add_argument("--scene_name", default="Full_Warehouse")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--num_pedestrians", type=int, default=30)
    return parser.parse_args()


def character_name(index: int) -> str:
    return "Character" if index == 0 else f"Character_{index:02d}"


def point3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"invalid point: {value!r}")
    z = float(value[2]) if len(value) > 2 else 0.02103900909423828
    return float(value[0]), float(value[1]), z


def dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tangent(points: tuple[tuple[float, float, float], ...], index: int) -> tuple[float, float]:
    left = max(0, index - 2)
    right = min(len(points) - 1, index + 2)
    dx = points[right][0] - points[left][0]
    dy = points[right][1] - points[left][1]
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 1.0, 0.0
    return dx / norm, dy / norm


def undirected_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    degrees = math.degrees(math.acos(dot))
    return min(degrees, 180.0 - degrees)


def cumulative(points: Iterable[tuple[float, float, float]]) -> list[float]:
    points = list(points)
    result = [0.0]
    for previous, current in zip(points, points[1:]):
        result.append(result[-1] + dist(previous, current))
    return result


def index_before(points: tuple[tuple[float, float, float], ...], anchor: int, distance_m: float) -> int | None:
    travelled = 0.0
    for index in range(anchor, 0, -1):
        travelled += dist(points[index], points[index - 1])
        if travelled >= distance_m:
            return index - 1
    return None


def index_after(points: tuple[tuple[float, float, float], ...], anchor: int, distance_m: float) -> int | None:
    travelled = 0.0
    for index in range(anchor, len(points) - 1):
        travelled += dist(points[index], points[index + 1])
        if travelled >= distance_m:
            return index + 1
    return None


def reverse_record(record: PathRecord) -> PathRecord:
    return PathRecord(
        record.episode_id,
        record.character,
        record.command_index,
        tuple(reversed(record.points)),
    )


def load_source(scene: Path) -> dict[int, dict[str, Any]]:
    episodes: dict[int, dict[str, Any]] = {}
    for path in sorted(scene.glob("episode_*.json"), key=lambda item: int(item.stem.split("_")[-1])):
        episode_id = int(path.stem.split("_")[-1])
        episodes[episode_id] = json.loads(path.read_text(encoding="utf-8"))
    if not episodes:
        raise FileNotFoundError(f"no episode JSON files in {scene}")
    return episodes


def collect_paths(episodes: dict[int, dict[str, Any]]) -> list[PathRecord]:
    records: list[PathRecord] = []
    for episode_id, data in episodes.items():
        commands = data["episode"]["characters"]["commands"]
        for name, sequence in commands.items():
            combined: list[tuple[float, float, float]] = []
            combined_start = 0

            def flush() -> None:
                nonlocal combined
                if len(combined) >= 32 and cumulative(combined)[-1] >= 14.0:
                    records.append(
                        PathRecord(
                            episode_id,
                            name,
                            combined_start,
                            tuple(combined),
                        )
                    )
                combined = []

            for command_index, command in enumerate(sequence):
                if command.get("cmd") != "GoTo":
                    continue
                raw_path = command.get("path")
                if not isinstance(raw_path, list) or len(raw_path) < 2:
                    continue
                points = tuple(point3(point) for point in raw_path)
                if not combined:
                    combined_start = command_index
                    combined.extend(points)
                elif dist(combined[-1], points[0]) <= 1.0:
                    combined.extend(points[1:])
                else:
                    flush()
                    combined_start = command_index
                    combined.extend(points)
            flush()
    if len(records) < 50:
        raise RuntimeError(f"only found {len(records)} usable paths")
    return records


def intersection_candidates(records: list[PathRecord]) -> list[Intersection]:
    grid: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for record_index, record in enumerate(records):
        for point_index in range(4, len(record.points) - 4, 3):
            point = record.points[point_index]
            key = (round(point[0] / PATH_GRID_M), round(point[1] / PATH_GRID_M))
            grid.setdefault(key, []).append((record_index, point_index))

    best: dict[tuple[int, int], Intersection] = {}
    for entries in grid.values():
        if len(entries) > 240:
            stride = max(1, len(entries) // 240)
            entries = entries[::stride][:240]
        for left in range(len(entries)):
            left_record_index, left_point_index = entries[left]
            for right in range(left + 1, len(entries)):
                right_record_index, right_point_index = entries[right]
                if left_record_index == right_record_index:
                    continue
                a = records[left_record_index]
                b = records[right_record_index]
                if a.episode_id == b.episode_id and a.character == b.character:
                    continue
                separation = dist(a.points[left_point_index], b.points[right_point_index])
                if separation > MAX_INTERSECTION_DISTANCE_M:
                    continue
                angle = undirected_angle(tangent(a.points, left_point_index), tangent(b.points, right_point_index))
                key = tuple(sorted((left_record_index, right_record_index)))
                candidate = Intersection(a, b, left_point_index, right_point_index, separation, angle)
                if key not in best or separation < best[key].distance:
                    best[key] = candidate
    return list(best.values())


def oriented_slice(
    record: PathRecord,
    anchor: int,
    before_m: float,
    after_m: float,
) -> tuple[PathRecord, int, int, int] | None:
    start = index_before(record.points, anchor, before_m)
    end = index_after(record.points, anchor, after_m)
    if start is not None and end is not None:
        sliced = record.points[start : end + 1]
        return PathRecord(record.episode_id, record.character, record.command_index, sliced), 0, anchor - start, len(sliced) - 1

    reversed_record = reverse_record(record)
    reversed_anchor = len(record.points) - 1 - anchor
    start = index_before(reversed_record.points, reversed_anchor, before_m)
    end = index_after(reversed_record.points, reversed_anchor, after_m)
    if start is None or end is None:
        return None
    sliced = reversed_record.points[start : end + 1]
    return PathRecord(record.episode_id, record.character, record.command_index, sliced), 0, reversed_anchor - start, len(sliced) - 1


def closest_distance_to_route(point: tuple[float, float, float], route: tuple[tuple[float, float, float], ...]) -> float:
    return min(dist(point, route_point) for route_point in route[::2])


def commands_stay_clear(commands: list[dict[str, Any]], route: tuple[tuple[float, float, float], ...]) -> bool:
    for command in commands:
        path = command.get("path") if command.get("cmd") == "GoTo" else None
        if not isinstance(path, list):
            continue
        for raw_point in path[::4]:
            if closest_distance_to_route(point3(raw_point), route) < BACKGROUND_ROUTE_CLEARANCE_M:
                return False
    return True


def background_pool(episodes: dict[int, dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]:
    pool = []
    for episode_id, data in episodes.items():
        chars = data["episode"]["characters"]
        for name, spawn in chars["spawn_positions"].items():
            sequence = chars["commands"].get(name)
            if isinstance(spawn, dict) and isinstance(sequence, list) and sequence:
                pool.append(
                    (
                        spawn,
                        sequence,
                        {"episode": episode_id, "character": name, "source": "spawn"},
                    )
                )
                variants_added = 0
                for command_index, command in enumerate(sequence):
                    raw_path = command.get("path") if command.get("cmd") == "GoTo" else None
                    if not isinstance(raw_path, list) or len(raw_path) < 12:
                        continue
                    points = tuple(point3(point) for point in raw_path)
                    for fraction in (0.25, 0.5, 0.75):
                        start_index = min(len(points) - 3, max(1, int(len(points) * fraction)))
                        segment = points[start_index:]
                        if cumulative(segment)[-1] < 2.0:
                            continue
                        direction = tangent(segment, 0)
                        derived_spawn = {
                            "pos": list(segment[0]),
                            "rot": math.atan2(direction[1], direction[0]),
                        }
                        derived_commands = [
                            make_goto(segment),
                            make_goto(tuple(reversed(segment))),
                        ]
                        pool.append(
                            (
                                derived_spawn,
                                derived_commands,
                                {
                                    "episode": episode_id,
                                    "character": name,
                                    "source": "precomputed_path",
                                    "command_index": command_index,
                                    "path_index": start_index,
                                },
                            )
                        )
                        variants_added += 1
                    if variants_added >= 6:
                        break
    return pool


def select_backgrounds(
    pool: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
    count: int,
    robot_route: tuple[tuple[float, float, float], ...],
    robot_start: tuple[float, float, float],
    robot_goal: tuple[float, float, float],
    critical_spawns: list[tuple[float, float, float]],
    rng: random.Random,
) -> list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]]:
    candidates = pool.copy()
    rng.shuffle(candidates)
    selected = []
    positions = list(critical_spawns)
    for spawn, sequence, provenance in candidates:
        position = point3(spawn.get("pos"))
        if dist(position, robot_start) < ROBOT_ENDPOINT_CLEARANCE_M:
            continue
        if dist(position, robot_goal) < ROBOT_ENDPOINT_CLEARANCE_M:
            continue
        if any(dist(position, other) < BACKGROUND_SPAWN_SEPARATION_M for other in positions):
            continue
        if not commands_stay_clear(sequence, robot_route):
            continue
        selected.append((copy.deepcopy(spawn), copy.deepcopy(sequence), provenance))
        positions.append(position)
        if len(selected) == count:
            return selected
    raise RuntimeError(f"only found {len(selected)} route-clear backgrounds, need {count}")


def make_goto(points: tuple[tuple[float, float, float], ...]) -> dict[str, Any]:
    target = points[-1]
    return {
        "cmd": "GoTo",
        "params": [f"{target[0]:.6f}", f"{target[1]:.6f}", f"{target[2]:.6f}"],
        "path": [list(point) for point in points],
    }


def sha256_json(value: Any) -> tuple[str, str]:
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def safe_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to replace existing path: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def prepare_overlay(source_root: Path, source_scene: Path, output: Path) -> Path:
    overlay_base = output / "interaction_dataset"
    overlay_root = overlay_base / "isaacsim_scene"
    overlay_scene = overlay_root / source_scene.name
    overlay_scene.mkdir(parents=True, exist_ok=False)
    for entry in source_root.parent.iterdir():
        if entry.name != source_root.name:
            safe_symlink(entry, overlay_base / entry.name)
    for entry in source_scene.iterdir():
        if not (entry.name.startswith("episode_") and entry.suffix == ".json"):
            safe_symlink(entry, overlay_scene / entry.name)
    return overlay_scene


def scenario_spec(index: int) -> tuple[str, float, int]:
    specs = [
        ("crossing_90", 7.0, 1),
        ("crossing_90", 4.0, 1),
        ("head_on", 7.0, 1),
        ("head_on", 4.0, 1),
        ("side_merge", 7.0, 1),
        ("side_merge", 4.0, 1),
        ("opposing_corridor", 7.0, 1),
        ("opposing_corridor", 4.0, 1),
        ("group_crossing", 7.0, 3),
        ("group_crossing", 4.0, 3),
    ]
    return specs[index]


def choose_intersection(
    kind: str,
    candidates: list[Intersection],
    records: list[PathRecord],
    used: set[tuple[int, str, int]],
    rng: random.Random,
) -> Intersection:
    if kind in {"head_on", "opposing_corridor"}:
        options = [record for record in records if cumulative(record.points)[-1] >= 18.0]
        rng.shuffle(options)
        for record in options:
            key = (record.episode_id, record.character, record.command_index)
            if key in used:
                continue
            anchor = len(record.points) // 2
            return Intersection(record, reverse_record(record), anchor, len(record.points) - 1 - anchor, 0.0, 0.0)
        raise RuntimeError(f"no unused path for {kind}")

    if kind in {"crossing_90", "group_crossing"}:
        options = [candidate for candidate in candidates if 60.0 <= candidate.angle_deg <= 90.0]
    else:
        options = [candidate for candidate in candidates if 20.0 <= candidate.angle_deg <= 55.0]
    # Spatially closest candidates are often short fragments around a turn.
    # Shuffle the full valid-angle pool so retries explore paths with enough
    # room on both sides for the trigger and time-aligned approach.
    rng.shuffle(options)
    for candidate in options:
        key = (candidate.robot.episode_id, candidate.robot.character, candidate.robot.command_index)
        if key not in used:
            return candidate
    raise RuntimeError(f"no unused intersection for {kind}")


def build_episode(
    episode_id: int,
    kind: str,
    trigger_distance: float,
    critical_count: int,
    intersection: Intersection,
    template: dict[str, Any],
    pool: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]],
    num_pedestrians: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    robot_slice = oriented_slice(intersection.robot, intersection.robot_index, trigger_distance + 2.0, 5.0)
    if robot_slice is None:
        raise RuntimeError(f"episode {episode_id}: robot route lacks room around conflict")
    robot_record, _, robot_q_index, _ = robot_slice

    pedestrian_slice = oriented_slice(intersection.pedestrian, intersection.pedestrian_index, 4.0, 4.0)
    if pedestrian_slice is None:
        raise RuntimeError(f"episode {episode_id}: pedestrian route lacks timing distance")
    pedestrian_record, _, pedestrian_q_index, _ = pedestrian_slice

    robot_route = robot_record.points
    pedestrian_route = pedestrian_record.points
    conflict = (
        (robot_route[robot_q_index][0] + pedestrian_route[pedestrian_q_index][0]) / 2.0,
        (robot_route[robot_q_index][1] + pedestrian_route[pedestrian_q_index][1]) / 2.0,
    )
    trigger_local_index = index_before(robot_route, robot_q_index, trigger_distance)
    if trigger_local_index is None:
        raise RuntimeError(f"episode {episode_id}: cannot place trigger gate")
    trigger_point = robot_route[trigger_local_index]
    trigger_tangent = tangent(robot_route, trigger_local_index)
    trigger_normal = (-trigger_tangent[1], trigger_tangent[0])
    gate_half_width = 2.0
    gate = [
        [trigger_point[0] - trigger_normal[0] * gate_half_width, trigger_point[1] - trigger_normal[1] * gate_half_width],
        [trigger_point[0] + trigger_normal[0] * gate_half_width, trigger_point[1] + trigger_normal[1] * gate_half_width],
    ]

    critical_routes: list[tuple[tuple[float, float, float], ...]] = []
    for critical_index in range(critical_count):
        if critical_count == 1:
            shifted = pedestrian_route
        else:
            # Keep every group member exactly on the donor NavMesh path.  A
            # one-metre longitudinal offset produces a short stream without
            # inventing laterally translated coordinates.
            arc = cumulative(pedestrian_route)
            target_offset = critical_index * 1.0
            start_index = next(
                (index for index, value in enumerate(arc) if value >= target_offset),
                len(pedestrian_route) - 2,
            )
            shifted = pedestrian_route[start_index:]
        critical_routes.append(shifted)

    backgrounds = select_backgrounds(
        pool,
        num_pedestrians - critical_count,
        robot_route,
        robot_route[0],
        robot_route[-1],
        [route[0] for route in critical_routes],
        rng,
    )

    generated = copy.deepcopy(template)
    episode = generated["episode"]
    episode["episode_id"] = episode_id
    start_tangent = tangent(robot_route, 0)
    episode["robot"] = {
        "start_pos": [robot_route[0][0], robot_route[0][1], 0.0],
        "goal_pos": [robot_route[-1][0], robot_route[-1][1], 0.0],
        "start_orientation": math.atan2(start_tangent[1], start_tangent[0]),
    }
    spawn_positions: dict[str, Any] = {}
    commands: dict[str, Any] = {}
    background_provenance = []
    for index, (spawn, sequence, provenance) in enumerate(backgrounds):
        name = character_name(index)
        spawn_positions[name] = spawn
        commands[name] = sequence
        background_provenance.append({"name": name, **provenance})

    critical_names = []
    for critical_offset, route in enumerate(critical_routes):
        index = num_pedestrians - critical_count + critical_offset
        name = character_name(index)
        critical_names.append(name)
        direction = tangent(route, 0)
        spawn_positions[name] = {
            "pos": list(route[0]),
            "rot": math.atan2(direction[1], direction[0]),
        }
        commands[name] = [
            {"cmd": "Idle", "params": [f"{CRITICAL_IDLE_S:.1f}"]},
            make_goto(route),
        ]

    episode["characters"] = {
        "num_characters": num_pedestrians,
        "spawn_positions": spawn_positions,
        "commands": commands,
    }
    nominal_robot_time = cumulative(robot_route[: robot_q_index + 1])[-1] / ROBOT_SPEED_MPS
    nominal_pedestrian_time_after_trigger = cumulative(
        pedestrian_route[: pedestrian_q_index + 1]
    )[-1] / PEDESTRIAN_SPEED_MPS
    nominal_robot_time_after_trigger = trigger_distance / ROBOT_SPEED_MPS
    release_delay_after_trigger = max(
        0.01, nominal_robot_time_after_trigger - nominal_pedestrian_time_after_trigger
    )
    interaction = {
        "schema": "flux_interaction_episode_v1",
        "type": kind,
        "conflict_point": list(conflict),
        "robot_reference_path": [list(point) for point in robot_route],
        "pedestrian_reference_path": [list(point) for point in pedestrian_route],
        "critical_characters": critical_names,
        "trigger": {
            "type": "gate",
            "point": [trigger_point[0], trigger_point[1]],
            "tangent": [trigger_tangent[0], trigger_tangent[1]],
            "segment": gate,
            "half_width_m": gate_half_width,
            "distance_before_conflict_m": trigger_distance,
        },
        "assumed_robot_speed_mps": ROBOT_SPEED_MPS,
        "assumed_pedestrian_speed_mps": PEDESTRIAN_SPEED_MPS,
        "nominal_robot_arrival_from_start_s": nominal_robot_time,
        "release_delay_after_trigger_s": release_delay_after_trigger,
        "nominal_robot_arrival_after_trigger_s": nominal_robot_time_after_trigger,
        "nominal_pedestrian_arrival_after_trigger_s": (
            release_delay_after_trigger + nominal_pedestrian_time_after_trigger
        ),
        "nominal_time_gap_after_trigger_s": abs(
            nominal_robot_time_after_trigger
            - release_delay_after_trigger
            - nominal_pedestrian_time_after_trigger
        ),
        "source": {
            "robot_episode": intersection.robot.episode_id,
            "robot_character": intersection.robot.character,
            "pedestrian_episode": intersection.pedestrian.episode_id,
            "pedestrian_character": intersection.pedestrian.character,
            "spatial_separation_m": intersection.distance,
            "path_angle_deg": intersection.angle_deg,
        },
    }
    episode["interaction"] = interaction
    provenance = {
        "interaction": interaction,
        "backgrounds": background_provenance,
    }
    return generated, provenance


def main() -> None:
    args = parse_args()
    if args.num_pedestrians < 3:
        raise ValueError("num_pedestrians must be at least 3")
    source_root = Path(args.scene_dir).resolve()
    source_scene = source_root / args.scene_name
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    overlay_scene = prepare_overlay(source_root, source_scene, output)
    episodes = load_source(source_scene)
    records = collect_paths(episodes)
    candidates = intersection_candidates(records)
    pool = background_pool(episodes)
    rng = random.Random(args.seed)
    template = episodes[min(episodes)]
    used: set[tuple[int, str, int]] = set()
    manifest: dict[str, Any] = {
        "schema": "flux_interaction_suite_v1",
        "scene_name": args.scene_name,
        "seed": args.seed,
        "num_episodes": 10,
        "num_pedestrians": args.num_pedestrians,
        "source_scene_dir": str(source_scene),
        "overlay_scene_dir": str(overlay_scene),
        "episodes": {},
    }

    for episode_id in range(10):
        kind, trigger_distance, critical_count = scenario_spec(episode_id)
        built = None
        errors = []
        for _ in range(400):
            intersection = choose_intersection(kind, candidates, records, used, rng)
            key = (
                intersection.robot.episode_id,
                intersection.robot.character,
                intersection.robot.command_index,
            )
            used.add(key)
            try:
                built = build_episode(
                    episode_id,
                    kind,
                    trigger_distance,
                    critical_count,
                    intersection,
                    template,
                    pool,
                    args.num_pedestrians,
                    rng,
                )
                break
            except RuntimeError as error:
                errors.append(str(error))
        if built is None:
            raise RuntimeError(f"failed to build episode {episode_id}: {errors[-5:]}")
        generated, provenance = built
        payload, digest = sha256_json(generated)
        (overlay_scene / f"episode_{episode_id}.json").write_text(payload, encoding="utf-8")
        manifest["episodes"][str(episode_id)] = {
            "sha256": digest,
            **provenance,
        }

    manifest_payload, _ = sha256_json(manifest)
    manifest_path = output / "interaction_manifest.json"
    manifest_path.write_text(manifest_payload, encoding="utf-8")
    print(
        f"GENERATED scene={args.scene_name} episodes=10 pedestrians={args.num_pedestrians} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
