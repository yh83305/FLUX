"""Streaming world-state logs; no RGB/depth or policy inputs are retained."""
import json
import math
import os
from pathlib import Path


def finite_vector(value, length):
    values = [float(x) for x in value]
    if len(values) != length or not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite or malformed trajectory state')
    return values


def trajectory_complete(output, episode, duration):
    """Constant-memory artifact check; legacy runs without config stay readable."""
    output = Path(output)
    config = output/'trajectory_config.json'
    if not config.exists():
        return True
    try:
        if not json.loads(config.read_text())['enabled']:
            return True
        path = output/'trajectories'/f'episode_{int(episode):03d}.jsonl'
        with path.open('rb') as stream:
            header = json.loads(stream.readline())
            stream.seek(max(0, path.stat().st_size-8192))
            footer = json.loads(stream.read().splitlines()[-1])
        return (header['schema'] == 'dynbench_world_trajectories_v1'
                and header['episode'] == int(episode)
                and footer['type'] == 'end' and footer['complete'] is True
                and footer['episode'] == int(episode)
                and footer['frames'] == footer['steps']+1
                and math.isclose(footer['duration_s'], float(duration), abs_tol=1e-6)
                and math.isclose(footer['steps']*header['step_dt_s'], float(duration), abs_tol=1e-6))
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


class EpisodeTrajectoryWriter:
    def __init__(self, root, step_dt, metadata=None):
        self.root = Path(root) / 'trajectories'
        self.root.mkdir(parents=True, exist_ok=True)
        self.dt = float(step_dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('step_dt must be positive')
        self.metadata = metadata or {}
        self.stream = None
        self.episode = None

    def _write(self, row):
        self.stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')

    def begin(self, episode):
        if self.stream is not None:
            raise RuntimeError('Previous episode has not been closed')
        self.episode = int(episode)
        self.final_path = self.root / f'episode_{self.episode:03d}.jsonl'
        self.partial_path = self.root / f'episode_{self.episode:03d}.jsonl.partial'
        if self.final_path.exists():
            raise FileExistsError(self.final_path)
        self.stream = self.partial_path.open('x', buffering=1)
        self.previous = {}
        self.last_time = None
        self.frames = 0
        self.last_terminal = False
        self._write(dict(
            type='metadata', schema='dynbench_world_trajectories_v1', episode=self.episode,
            coordinate_frame='simulator_world', distance_unit='m', time_unit='s',
            quaternion_order='wxyz', angular_velocity_unit='rad/s', step_dt_s=self.dt,
            sampling='initial state then every environment step, after termination compute and before auto-reset',
            robot_state_source='IsaacLab articulation root_state_w: position, quaternion, linear velocity, angular velocity',
            pedestrian_position_source='GlobalCharacterPositionManager.get_character_current_pos',
            pedestrian_velocity_source='backward finite difference of consecutive world positions over sim_time_s',
            pedestrian_identity='(episode, character USD prim path); not shared across episodes',
            missing_velocity='null on first observation, gaps, or nonpositive simulation-time delta',
            metadata=self.metadata,
        ))

    def sample(self, sim_time, robot_state, people, terminal=False, termination_terms=None, people_valid=True):
        if self.stream is None:
            raise RuntimeError('No open episode')
        sim_time = float(sim_time)
        if not math.isfinite(sim_time) or (self.last_time is not None and sim_time < self.last_time):
            raise ValueError('Simulation time must be finite and nondecreasing')
        if self.last_terminal:
            raise RuntimeError('Cannot append reset state after terminal sample')
        robot = finite_vector(robot_state, 13)
        qw, qx, qy, qz = robot[3:7]
        yaw = math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
        pedestrians = []
        current = {}
        for identity, position in sorted(people.items()):
            identity = str(identity)
            xyz = finite_vector(position, 3)
            previous = self.previous.get(identity)
            velocity = None
            if previous is not None and sim_time > previous[0]:
                velocity = [(xyz[i]-previous[1][i])/(sim_time-previous[0]) for i in range(3)]
                velocity = finite_vector(velocity, 3)
            pedestrians.append(dict(id=identity, position_world_m=xyz, velocity_world_mps=velocity))
            current[identity] = (sim_time, xyz)
        self._write(dict(
            type='frame', episode=self.episode, step=self.frames,
            episode_time_s=self.frames*self.dt, sim_time_s=sim_time,
            phase='initial' if self.frames == 0 else 'post_step_pre_reset',
            robot=dict(position_world_m=robot[:3], quaternion_wxyz=robot[3:7],
                       yaw_rad=yaw, linear_velocity_world_mps=robot[7:10],
                       angular_velocity_world_radps=robot[10:13]),
            pedestrians=pedestrians, people_valid=bool(people_valid),
            terminal=bool(terminal), termination_terms=list(termination_terms or []),
        ))
        # Drop identities absent from this sample so reappearance is not a
        # false high-speed trajectory bridging an unobserved interval.
        self.previous = current
        self.last_time = sim_time
        self.last_terminal = bool(terminal)
        self.frames += 1

    def finish(self, metrics):
        if self.stream is None or not self.last_terminal:
            raise RuntimeError('Cannot finalize without a terminal sample')
        if int(metrics['episode']) != self.episode:
            raise ValueError('Metric episode does not match trajectory')
        if not math.isclose(float(metrics['time_to_goal']), (self.frames-1)*self.dt, abs_tol=1e-6):
            raise ValueError('Metric duration does not match captured step count')
        self._write(dict(type='end', episode=self.episode, complete=True,
                         frames=self.frames, steps=self.frames-1,
                         success=float(metrics['success']), duration_s=float(metrics['time_to_goal'])))
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.stream = None
        self.partial_path.replace(self.final_path)
        self.previous = {}

    def close(self):
        if self.stream is not None:
            self._write(dict(type='end', episode=self.episode, complete=False,
                             frames=self.frames, reason='interrupted_before_metrics_commit'))
            self.stream.flush()
            self.stream.close()
            self.stream = None


class IsaacTrajectoryRecorder:
    """Wrap only one env.step; restore compute even when simulation raises.

    On the supported IsaacLab version, termination_manager.compute runs after
    physics/scene updates and before _reset_idx. No termination values change.
    """
    def __init__(self, output, env, people_reader, metadata=None):
        self.env = env.unwrapped
        if self.env.num_envs != 1:
            raise ValueError('Trajectory capture currently requires num_envs=1')
        self.people_reader = people_reader
        self.writer = EpisodeTrajectoryWriter(output, self.env.step_dt, metadata)

    def _snapshot(self, terminal=False, terms=None):
        state = self.env.scene.articulations['robot'].data.root_state_w[0].detach().cpu().tolist()
        positions, paths, valid = self.people_reader(self.env)
        if len(positions) != len(paths) or len(set(map(str, paths))) != len(paths):
            raise ValueError('Pedestrian position/identity mismatch')
        self.writer.sample(self.env.sim.current_time, state,
                           {str(k): v for k, v in zip(paths, positions)},
                           terminal, terms, valid)

    def step(self, wrapper, action, episode):
        if self.writer.stream is None:
            self.writer.begin(episode)
            self._snapshot()
        if self.writer.episode != int(episode):
            raise RuntimeError('Episode changed without trajectory finalization')
        manager = self.env.termination_manager
        original = manager.compute
        had_override = 'compute' in vars(manager)
        saved_override = vars(manager).get('compute')
        calls = 0

        def capture(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls != 1:
                raise RuntimeError('Unexpected multiple termination computes per environment step')
            terms = [name for name in manager.active_terms if bool(manager.get_term(name)[0])]
            self._snapshot(bool(result[0]), terms)
            return result

        manager.compute = capture
        try:
            result = wrapper.step(action)
            if calls != 1:
                raise RuntimeError('Missing pre-reset trajectory capture hook')
            return result
        finally:
            if had_override:
                manager.compute = saved_override
            else:
                del manager.compute

    def finish_episode(self, metrics):
        self.writer.finish(metrics)

    def close(self):
        self.writer.close()
