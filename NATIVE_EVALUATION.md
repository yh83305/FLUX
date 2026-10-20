# Native NavDP / FLUX evaluation

`runtime_fix/eval_render_safe.py` and `baselines/navdp/` capture the Bench
`navdp_ready_20261019` deployment, previously maintained outside Git.
`runtime_fix/bench_people_clock.py` and `CrowdSim/utils/bench_*` are its runtime
dependencies. Both policy checkpoints remain external artifacts.

The guard now discovers actual `CharacterBehavior` classes through Kit's
ScriptManager, rather than importing a different copy of the behavior class.
It rescans on app updates to cover reloads and episode resets. Logs include
`bind`, `installed`, `heartbeat`, `hold`, `release`, `reset`, and `summary`.
Require a positive callback count before accepting a guarded run. A config
flag alone is not evidence of operation.

The geometric yielding rule is unchanged: horizontal FOV including body radius,
no occlusion test; approaching offscreen pedestrians pause. This is not a
collision exemption, and does not prevent the robot driving into a stationary
pedestrian. Old runs without verified binding must not be pooled with new runs.

Run inside the established Isaac Sim 4.5 container using a new pinned checkout:

```
docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/mnt/data8t/PINNED_FLUX \
  -e FLUX_ISAAC_EXPERIENCE=/workspace/IsaacLab/apps/isaacsim_4_5/isaaclab.python.dyn.kit \
  -w /workspace/FLUX CONTAINER /isaac-sim/python.sh -u \
  /mnt/data8t/PINNED_FLUX/runtime_fix/eval_render_safe.py \
  --port 9010 --gpu_id 0 --scene_dir SCENE_VIEW --scene_index 0 \
  --num_envs 1 --num_episodes 1 --episode_start 0 --speed 0.5 \
  --output_dir NEW_OUTPUT
```

Use separate result roots for each code/protocol version. Validate checkpoint,
dataset and evaluator hashes, native policy loading, camera profile, actual guard
callbacks, episode IDs, nonempty videos, metrics, and cleanup.

Tests: `PYTHONPATH=. python3 -m pytest -q tests/test_offscreen_guard.py
tests/test_offscreen_runtime.py tests/test_collision_termination.py`.
