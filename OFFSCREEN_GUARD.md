# Offscreen pedestrian guard

Based on FLUX commit 73d246f959903837f2e7ce242d86a17568831898.
`eval_socialnav_wheeled.py` enables the guard by default; pass
`--disable_offscreen_guard` for the original benchmark behavior.
This is a modified environment, not directly comparable with unguarded results.

Only approaching pedestrians wholly outside the horizontal 69-degree frustum
(including a 0.30 m body-disc margin) are paused when their estimated one-second
motion would come within 0.80 m of the current robot position. Visible pedestrians
are never paused. No occlusion test is performed. The robot command and collision
metrics are unchanged. This does not guarantee safety if the robot itself drives
into a stopped pedestrian. Initial overlap is not repaired by teleportation.

The animation callback preserves the command queue, publishes the stationary
position, and resumes the command when the risk or visibility changes. Output
`offscreen_guard_config.json` identifies the environment variant;
`offscreen_guard.jsonl` records hold/release transitions by character path.
The runtime hook is scoped to this evaluator process and restored on cleanup.
Headless unit tests validate geometry and pause/release behavior; full Isaac
runtime validation is required before reporting protected benchmark results.
