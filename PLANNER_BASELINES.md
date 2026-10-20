# Native iPlanner / ViPlanner integration

Both servers provide `/health`, `/navigator_reset`, `/navigator_reset_env`, and
`/pointgoal_step`. Inputs use the existing RGB JPEG/PNG and metric depth uint16
PNG protocol (`depth_m * 10000`), with camera-frame planar point goals.
Responses contain `(B,N,3)` trajectory, `(B,1,N,3)` candidates and native fear
scores. The current agent preprocessing is validated for one environment.

## Checkpoints and devices

iPlanner accepts state dictionaries or the first element of a checkpoint tuple.
For known trusted legacy pickled model objects, explicitly pass
`--trusted-legacy-checkpoint`; the loader extracts the object's state dictionary
and strictly loads it into the current architecture. There is no automatic
unsafe fallback. Record the input checkpoint hash.

Both servers accept `--device cpu` or `--device cuda:0`, and bind to loopback
unless a different `--host` is explicitly passed. ViPlanner propagates the
device to both Mask2Former and its trajectory network.

The inspected Bench deployments use host Conda `navdp` for iPlanner and
`viplanner` for ViPlanner. The latter contains the compatible MMDetection/MMCV
stack; the Isaac container's MMCV installation is not used by this host service.
Supply absolute paths for every checkpoint and configuration.

## Validation

CPU HTTP tests on Bench loaded the existing iPlanner `plannernet.pt`, ViPlanner
`model.pt`, and Mask2Former COCO checkpoint. Three point-goal requests per model
returned finite `(1,51,3)` trajectories and `(1,1,51,3)` candidates. Reset and
health endpoints passed. Inputs used recorded RGB with synthetic 3 m depth:
this is a transport/inference smoke, not a navigation performance measurement.

GPU and closed-loop acceptance is separate. Use
`tools/run_planner_smokes_when_idle.py --config DEPLOYMENT.json`.
The configuration specifies output, dataset, code_root, container, wait_for_run,
and models. Each model specifies gpu, port, and its complete server_command
argument list. The launcher waits for both current model runs' COMPLETE markers,
at least 18 GB free per assigned GPU, and no unrelated compute process. It runs
Hospital episode 0 and Office episode 0 for each model using the shared evaluator,
without changing the currently running benchmark.

Each scene must yield metrics, an actual video, guard callbacks, and a cleanup
marker. Failures produce closedloop_status.json with stage=failed; no full
baseline suite is started automatically.
