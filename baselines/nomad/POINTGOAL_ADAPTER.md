# NoMaD PointGoal adapter

The checkpoint is an unmodified goal-masked NoMaD visual policy. It does not
natively consume metric PointGoal coordinates. For each RGB-history
observation, it samples `--sample_num` goal-masked NoMaD trajectories and
selects the candidate whose final waypoint is closest in Euclidean distance
to the camera-frame PointGoal. The PointGoal is used only by this deterministic
selector; no goal image, depth-derived target, map, pedestrian state, or
privileged observation is synthesized.

The service reports algorithm `nomad_pointgoal_adapter`, selection
`minimum_endpoint_goal_distance`, every sampled trajectory, its negative
endpoint-distance value, and the selected index. This name and adaptation must
be disclosed in tables; do not report it as native PointGoal NoMaD.

The checkpoint is loaded on CPU with `weights_only=True`, strictly matched,
then transferred to the requested device. Candidate layout is explicitly
sample-major to batch-major, including batches larger than one. Input RGB is
kept in RGB order. Input-video recording is opt-in and is not an evaluation
video.

Validated deployment dependencies (kept outside source/environment):

- Python: `/home/yizhong/miniconda3/envs/navdp/bin/python`
- `real-stanford/diffusion_policy` commit
  `5ba07ac6661db573af695b419a7947ecb704690f`
- `lukemelas/EfficientNet-PyTorch` commit
  `7e8b0d312162f335785fb5dcfa1df29a75a1783a`
- isolated `einops==0.4.1`

These are supplied through `PYTHONPATH`; no shared Conda environment was
modified. A formal run manifest must record checkpoint/dependency hashes,
sample count, selector, controller, evaluator commit, and episode set.
