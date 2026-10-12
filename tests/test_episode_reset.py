"""Exercise evaluator functions without importing Isaac Sim's CLI entrypoint."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'eval_socialnav_wheeled.py'


def load_function(name, scope):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope[name]


def test_reset_renders_new_pose_before_reading_observations():
    calls = []
    camera = SimpleNamespace(
        synchronize_pose=lambda: calls.append('pose'),
        update=lambda *a, **kw: calls.append('buffer'),
        data=SimpleNamespace(output={'rgb': 1, 'distance_to_image_plane': 2}),
    )
    core = SimpleNamespace(
        scene=SimpleNamespace(sensors={'camera_sensor': camera}), step_dt=.15,
        sim=SimpleNamespace(render=lambda: calls.append('render')),
        observation_manager=SimpleNamespace(compute=lambda: calls.append('obs') or 123),
    )
    refresh = load_function('refresh_camera_after_reset', {})
    assert refresh(SimpleNamespace(unwrapped=core), None) == 123
    assert calls == ['pose', 'render', 'pose', 'render', 'buffer', 'obs']


def test_inflight_old_result_cannot_publish_controller_or_trajectory():
    stop = threading.Event()
    inp = SimpleNamespace(**{k: np.zeros((1, 3)) for k in (
        'current_goal', 'current_image', 'current_depth', 'camera_pos',
        'robot_state', 'applied_command')})
    inp.camera_rot = np.eye(3)[None]
    inp.people_positions = None
    inp.people_paths = {}
    inp.people_valid = False
    inp.observation_frame_id = 0
    inp.observation_time_s = 0
    inp.episode_generation = 18
    sentinel = object()
    out = SimpleNamespace(mpc=sentinel, trajectory_points_world=None)

    def rpc(*args, **kwargs):
        # Simulate reset invalidating generation while the old RPC is in flight.
        inp.episode_generation = 19
        stop.set()
        return np.zeros((1, 2, 3)), np.zeros((1, 1, 2, 3)), np.zeros((1, 1)), {}

    scope = dict(np=np, copy=__import__('copy'), stop_event=stop,
        input_lock=threading.Lock(), output_lock=threading.Lock(),
        request_lock=threading.Lock(), planning_input=inp, planning_output=out,
        pointgoal_step=rpc, args_cli=SimpleNamespace(port=1, speed=.5),
        MPC_Controller=lambda *a, **kw: object(), ModeMPCController=None,
        time=SimpleNamespace(sleep=lambda _: None), mpc=sentinel)
    load_function('planning_thread', scope)(None, None)
    assert out.mpc is sentinel
    assert out.trajectory_points_world is None
    assert scope['mpc'] is sentinel
