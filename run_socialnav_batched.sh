#!/usr/bin/env bash
set -euo pipefail

# Run SocialNav in short-lived Isaac Sim processes. Recreating animated people
# repeatedly in one USD stage can corrupt native AnimationGraph/Fabric state.
port="${PORT:-9999}"
scene_dir="${SCENE_DIR:-/workspace/FLUX/assets/dynbench/isaacsim_scene}"
scene_index="${SCENE_INDEX:-0}"
total_episodes="${TOTAL_EPISODES:-100}"
episodes_per_process="${EPISODES_PER_PROCESS:-10}"
start_episode="${START_EPISODE:-0}"
output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR to an absolute container path}"

if (( start_episode < 0 || total_episodes <= 0 || episodes_per_process <= 0 )); then
    echo "START_EPISODE must be non-negative; episode counts must be positive" >&2
    exit 2
fi

mkdir -p "$output_dir"
episode_end=$((start_episode + total_episodes))
for ((start = start_episode; start < episode_end; start += episodes_per_process)); do
    count=$episodes_per_process
    if (( start + count > episode_end )); then
        count=$((episode_end - start))
    fi
    batch_dir=$(printf "%s/batch_%03d_%03d" "$output_dir" "$start" "$((start + count - 1))")
    mkdir -p "$batch_dir"
    echo "[BATCH] episodes [$start, $((start + count))) -> $batch_dir"
    /isaac-sim/python.sh -u /workspace/FLUX/eval_socialnav_wheeled.py \
        --port "$port" \
        --gpu_id 0 \
        --scene_dir "$scene_dir" \
        --scene_index "$scene_index" \
        --scene_scale 1.0 \
        --num_envs 1 \
        --episode_start "$start" \
        --num_episodes "$count" \
        --speed 0.5 \
        --stop_threshold -3.0 \
        --output_dir "$batch_dir"
done

/isaac-sim/python.sh - "$output_dir" <<'PY'
import csv
import pathlib
import sys
from socialnav_metrics import write_socialnav_summary

root = pathlib.Path(sys.argv[1])
rows = []
fieldnames = None
for path in sorted(root.glob("batch_*/metric.csv")):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = fieldnames or reader.fieldnames
        rows.extend(reader)
if fieldnames is None:
    raise RuntimeError("No batch metric.csv files were produced")
rows.sort(key=lambda row: int(row["episode"]))
with (root / "metric.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[BATCH] merged {len(rows)} episodes into {root / 'metric.csv'}")
summary = write_socialnav_summary(rows, root)
print(f"[BATCH] final summary: {summary}")
PY
