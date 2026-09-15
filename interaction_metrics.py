#!/usr/bin/env python3
"""Runtime gate triggering and interaction-specific metrics for FLUX SocialNav."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


INTERACTION_COLUMNS = [
    "episode",
    "interaction_type",
    "interaction_presented",
    "trigger_time_s",
    "critical_count",
    "nominal_time_gap_s",
    "min_critical_distance_m",
    "critical_personal_space_time_s",
    "critical_social_space_time_s",
    "robot_conflict_arrival_s",
    "pedestrian_conflict_arrival_s",
    "PET_s",
    "min_TTC_s",
    "robot_stop_time_s",
    "max_deceleration_mps2",
    "max_jerk_mps3",
]


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


class InteractionRunRecorder:
    def __init__(self, manifest_path: str, output_dir: str, episode_start: int = 0):
        self.manifest_path = Path(manifest_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.episode_start = episode_start
        self.rows: list[dict[str, Any]] = []
        self.csv_path = self.output_dir / "interaction_metric.csv"
        self.summary_path = self.output_dir / "interaction_summary.json"

    def metadata(self, episode_id: int) -> dict[str, Any]:
        try:
            return self.manifest["episodes"][str(episode_id)]["interaction"]
        except KeyError as error:
            raise KeyError(f"interaction manifest has no episode {episode_id}") from error

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=INTERACTION_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        self._write_summary()

    def _write_summary(self) -> None:
        def numeric(key: str) -> list[float]:
            values = []
            for row in self.rows:
                value = _finite(row.get(key))
                if value is not None:
                    values.append(value)
            return values

        presented = numeric("interaction_presented")
        summary = {
            "schema": "flux_interaction_summary_v1",
            "episodes_recorded": len(self.rows),
            "interaction_presented_count": int(sum(presented)),
            "interaction_presented_rate": mean(presented) if presented else None,
            "mean_nominal_time_gap_s": mean(numeric("nominal_time_gap_s")) if numeric("nominal_time_gap_s") else None,
            "mean_min_critical_distance_m": mean(numeric("min_critical_distance_m")) if numeric("min_critical_distance_m") else None,
            "mean_PET_s": mean(numeric("PET_s")) if numeric("PET_s") else None,
            "mean_min_TTC_s": mean(numeric("min_TTC_s")) if numeric("min_TTC_s") else None,
            "mean_robot_stop_time_s": mean(numeric("robot_stop_time_s")) if numeric("robot_stop_time_s") else None,
            "mean_max_deceleration_mps2": mean(numeric("max_deceleration_mps2")) if numeric("max_deceleration_mps2") else None,
            "mean_max_jerk_mps3": mean(numeric("max_jerk_mps3")) if numeric("max_jerk_mps3") else None,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class EpisodeInteractionState:
    def __init__(self, episode_id: int, metadata: dict[str, Any]):
        self.episode_id = episode_id
        self.metadata = metadata
        self.elapsed = 0.0
        self.triggered = False
        self.trigger_time: float | None = None
        self.robot_conflict_arrival: float | None = None
        self.pedestrian_conflict_arrival: float | None = None
        self.min_critical_distance = math.inf
        self.personal_time = 0.0
        self.social_time = 0.0
        self.min_ttc = math.inf
        self.stop_time = 0.0
        self.max_deceleration = 0.0
        self.max_jerk = 0.0
        self.previous_robot: np.ndarray | None = None
        self.previous_people: dict[str, np.ndarray] = {}
        self.previous_speed: float | None = None
        self.previous_acceleration: float | None = None

    @property
    def critical_names(self) -> list[str]:
        return list(self.metadata["critical_characters"])

    def _passed_gate(self, robot_xy: np.ndarray) -> bool:
        trigger = self.metadata["trigger"]
        point = np.asarray(trigger["point"], dtype=np.float64)
        direction = np.asarray(trigger["tangent"], dtype=np.float64)
        delta = robot_xy - point
        longitudinal = float(np.dot(delta, direction))
        lateral = abs(float(direction[0] * delta[1] - direction[1] * delta[0]))
        return longitudinal >= 0.0 and lateral <= float(trigger["half_width_m"])

    def _critical_positions(self) -> dict[str, np.ndarray]:
        from omni.anim.people.scripts.utils import Utils

        result = {}
        for name in self.critical_names:
            position = Utils.get_character_position_by_name(name)
            if position is not None:
                result[name] = np.asarray(position[:2], dtype=np.float64)
        return result

    def _release(self) -> None:
        from omni.anim.people.scripts.utils import Utils

        delay = max(
            0.01, float(self.metadata.get("release_delay_after_trigger_s", 0.01))
        )
        for name in self.critical_names:
            # Interrupt the long initial Idle.  The computed short Idle aligns
            # the pedestrian's remaining travel time with the robot's nominal
            # travel time from the trigger gate to the conflict point.
            Utils.runtime_inject_command(
                character_name=name,
                command_list=[f"{name} Idle {delay:.6f}"],
                force_inject=True,
                set_status=False,
            )
        self.triggered = True
        self.trigger_time = self.elapsed
        print(
            f"[INTERACTION TRIGGER] episode={self.episode_id} "
            f"time={self.elapsed:.3f}s delay={delay:.3f}s "
            f"characters={self.critical_names}",
            flush=True,
        )

    def update(self, robot_pos: np.ndarray, dt: float) -> None:
        robot_xy = np.asarray(robot_pos[:2], dtype=np.float64)
        dt = float(dt)
        self.elapsed += dt
        people = self._critical_positions()

        if not self.triggered and self._passed_gate(robot_xy):
            self._release()

        conflict = np.asarray(self.metadata["conflict_point"], dtype=np.float64)
        if self.robot_conflict_arrival is None and np.linalg.norm(robot_xy - conflict) <= 0.75:
            self.robot_conflict_arrival = self.elapsed
        if self.pedestrian_conflict_arrival is None and people:
            if min(np.linalg.norm(position - conflict) for position in people.values()) <= 0.75:
                self.pedestrian_conflict_arrival = self.elapsed

        distances = [np.linalg.norm(robot_xy - position) for position in people.values()]
        if distances:
            nearest = float(min(distances))
            self.min_critical_distance = min(self.min_critical_distance, nearest)
            if nearest < 1.2:
                self.personal_time += dt
            if nearest < 3.6:
                self.social_time += dt

        if self.previous_robot is not None and dt > 0:
            robot_velocity = (robot_xy - self.previous_robot) / dt
            speed = float(np.linalg.norm(robot_velocity))
            if speed < 0.1:
                self.stop_time += dt
            if self.previous_speed is not None:
                acceleration = (speed - self.previous_speed) / dt
                self.max_deceleration = max(self.max_deceleration, -acceleration)
                if self.previous_acceleration is not None:
                    jerk = abs((acceleration - self.previous_acceleration) / dt)
                    self.max_jerk = max(self.max_jerk, jerk)
                self.previous_acceleration = acceleration
            self.previous_speed = speed

            for name, position in people.items():
                previous = self.previous_people.get(name)
                if previous is None:
                    continue
                person_velocity = (position - previous) / dt
                relative_position = position - robot_xy
                relative_velocity = person_velocity - robot_velocity
                denominator = float(np.dot(relative_velocity, relative_velocity))
                if denominator <= 1e-8:
                    continue
                ttc = -float(np.dot(relative_position, relative_velocity)) / denominator
                if 0.0 < ttc <= 10.0:
                    miss = np.linalg.norm(relative_position + relative_velocity * ttc)
                    if miss < 0.5:
                        self.min_ttc = min(self.min_ttc, ttc)

        self.previous_robot = robot_xy.copy()
        self.previous_people = {name: value.copy() for name, value in people.items()}

    def row(self) -> dict[str, Any]:
        pet = None
        if self.robot_conflict_arrival is not None and self.pedestrian_conflict_arrival is not None:
            pet = abs(self.robot_conflict_arrival - self.pedestrian_conflict_arrival)
        return {
            "episode": self.episode_id,
            "interaction_type": self.metadata["type"],
            "interaction_presented": int(self.triggered),
            "trigger_time_s": _finite(self.trigger_time),
            "critical_count": len(self.critical_names),
            "nominal_time_gap_s": _finite(self.metadata.get("nominal_time_gap_after_trigger_s")),
            "min_critical_distance_m": _finite(self.min_critical_distance),
            "critical_personal_space_time_s": self.personal_time,
            "critical_social_space_time_s": self.social_time,
            "robot_conflict_arrival_s": _finite(self.robot_conflict_arrival),
            "pedestrian_conflict_arrival_s": _finite(self.pedestrian_conflict_arrival),
            "PET_s": _finite(pet),
            "min_TTC_s": _finite(self.min_ttc),
            "robot_stop_time_s": self.stop_time,
            "max_deceleration_mps2": self.max_deceleration,
            "max_jerk_mps3": self.max_jerk,
        }


def install_tracker_patch(
    socialnav_metrics_module: Any,
    manifest_path: str,
    output_dir: str,
    episode_start: int,
) -> InteractionRunRecorder:
    original_tracker = socialnav_metrics_module.SocialMetricsTracker
    recorder = InteractionRunRecorder(manifest_path, output_dir, episode_start)

    class InteractionSocialMetricsTracker:
        def __init__(self):
            self._base = original_tracker()
            self._episode_id = episode_start
            self._state = EpisodeInteractionState(
                self._episode_id, recorder.metadata(self._episode_id)
            )
            self._finalized = False

        def update(self, robot_pos: np.ndarray, people_positions: list[np.ndarray], dt: float):
            self._base.update(robot_pos, people_positions, dt)
            self._state.update(robot_pos, dt)

        def get_metrics(self) -> dict[str, Any]:
            result = self._base.get_metrics()
            if not self._finalized:
                row = self._state.row()
                recorder.append(row)
                print(f"[INTERACTION METRICS] {json.dumps(row, sort_keys=True)}", flush=True)
                self._finalized = True
            return result

        def reset(self):
            self._base.reset()
            if self._finalized:
                self._episode_id += 1
                self._state = EpisodeInteractionState(
                    self._episode_id, recorder.metadata(self._episode_id)
                )
                self._finalized = False

    socialnav_metrics_module.SocialMetricsTracker = InteractionSocialMetricsTracker
    return recorder
