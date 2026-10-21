import json
from types import SimpleNamespace
import pytest
from trajectory_recorder import EpisodeTrajectoryWriter, IsaacTrajectoryRecorder, trajectory_complete


STATE = [0, 0, .1, 1, 0, 0, 0, 1, 0, 0, 0, 0, .2]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_positions_identity_velocity_and_terminal(tmp_path):
    (tmp_path/'trajectory_config.json').write_text('{"enabled":true}')
    writer = EpisodeTrajectoryWriter(tmp_path, .15)
    writer.begin(7)
    writer.sample(10, STATE, {'/B': [2, 0, 0], '/A': [1, 0, 0]})
    writer.sample(10.15, STATE, {'/A': [1.15, 0, 0], '/B': [2, 0, 0]}, True, ['stuck'])
    assert not trajectory_complete(tmp_path, 7, .15)
    writer.finish(dict(episode=7, success=0, time_to_goal=.15))
    assert trajectory_complete(tmp_path, 7, .15)
    assert not trajectory_complete(tmp_path, 7, .30)
    rows = read(tmp_path/'trajectories/episode_007.jsonl')
    assert rows[1]['pedestrians'][0]['id'] == '/A'
    assert rows[1]['pedestrians'][0]['velocity_world_mps'] is None
    assert rows[2]['pedestrians'][0]['velocity_world_mps'] == pytest.approx([1, 0, 0])
    assert rows[2]['pedestrians'][1]['velocity_world_mps'] == [0, 0, 0]
    assert rows[2]['terminal'] and rows[2]['termination_terms'] == ['stuck']
    assert rows[2]['robot']['linear_velocity_world_mps'] == [1, 0, 0]
    assert rows[-1]['complete'] and rows[-1]['steps'] == 1


def test_gap_duplicate_time_reset_and_partial(tmp_path):
    writer = EpisodeTrajectoryWriter(tmp_path, .15)
    writer.begin(1)
    writer.sample(0, STATE, {'/A': [0, 0, 0]})
    writer.sample(.15, STATE, {})
    writer.sample(.30, STATE, {'/A': [100, 0, 0]})
    writer.sample(.30, STATE, {'/A': [101, 0, 0]}, True)
    writer.finish(dict(episode=1, success=0, time_to_goal=.45))
    rows = read(tmp_path/'trajectories/episode_001.jsonl')
    assert rows[3]['pedestrians'][0]['velocity_world_mps'] is None
    assert rows[4]['pedestrians'][0]['velocity_world_mps'] is None
    writer.begin(8)
    writer.sample(1, STATE, {'/A': [999, 0, 0]})
    writer.close()
    path = tmp_path/'trajectories/episode_008.jsonl.partial'
    assert read(path)[1]['pedestrians'][0]['velocity_world_mps'] is None
    assert not read(path)[-1]['complete']
    assert not path.with_suffix('').exists()
    with pytest.raises(FileExistsError):
        writer.begin(1)


class FakeTensor:
    def __init__(self, values): self.values = values
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.values[:]


class FakeTermination:
    active_terms = ['goal']
    def compute(self): return [True]
    def get_term(self, name): return [True]


def fake_env():
    state = FakeTensor(STATE[:])
    env = SimpleNamespace(num_envs=1, step_dt=.15, sim=SimpleNamespace(current_time=1.),
                          termination_manager=FakeTermination(),
                          scene=SimpleNamespace(articulations={'robot': SimpleNamespace(data=SimpleNamespace(root_state_w=[state]))}))
    env.unwrapped = env
    return env, state


def test_hook_captures_terminal_before_auto_reset_and_restores(tmp_path):
    env, state = fake_env()
    pedestrians = {'position': [1, 0, 0]}
    recorder = IsaacTrajectoryRecorder(tmp_path, env,
        lambda _: ([pedestrians['position']], ['/person'], True))
    def step(action):
        state.values[0] = 2
        pedestrians['position'] = [1.15, 0, 0]
        env.sim.current_time += .15
        result = env.termination_manager.compute()
        state.values[0] = 999  # Isaac automatic reset would otherwise pollute final sample.
        pedestrians['position'] = [999, 0, 0]
        return result
    env.step = step
    assert recorder.step(env, None, 42) == [True]
    assert 'compute' not in vars(env.termination_manager)
    recorder.finish_episode(dict(episode=42, success=1, time_to_goal=.15))
    rows = read(tmp_path/'trajectories/episode_042.jsonl')
    assert rows[2]['robot']['position_world_m'][0] == 2
    assert rows[2]['pedestrians'][0]['position_world_m'][0] == 1.15


def test_hook_restored_on_failure_and_rejects_missing_callback(tmp_path):
    env, state = fake_env()
    recorder = IsaacTrajectoryRecorder(tmp_path, env, lambda _: ([], [], False))
    env.step = lambda _: None
    with pytest.raises(RuntimeError, match='Missing pre-reset'):
        recorder.step(env, None, 0)
    assert 'compute' not in vars(env.termination_manager)
    recorder.close()


def test_bad_state_and_missing_terminal_rejected(tmp_path):
    writer = EpisodeTrajectoryWriter(tmp_path, .15)
    writer.begin(0)
    with pytest.raises(ValueError):
        writer.sample(0, STATE, {'/A': [float('nan'), 0, 0]})
    writer.sample(0, STATE, {})
    with pytest.raises(RuntimeError, match='terminal'):
        writer.finish(dict(episode=0, success=0, time_to_goal=0))
    writer.close()
