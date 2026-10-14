import numpy as np
from offscreen_guard import should_hold

def test_visible_is_unchanged():
    assert not should_hold([1,0],[-1,0],[0,0],[0,0],0)
def test_invisible_approach_pauses():
    assert should_hold([-1,0],[1,0],[0,0],[0,0],0)
def test_away_and_stationary_unchanged():
    assert not should_hold([-1,0],[-1,0],[0,0],[0,0],0)
    assert not should_hold([-1,0],[0,0],[0,0],[0,0],0)
def test_camera_rotation_changes_visibility():
    assert not should_hold([-1,0],[1,0],[0,0],[0,0],np.pi)
def test_side_crossing_and_far_clear():
    assert should_hold([0,1],[0,-1],[0,0],[0,0],0)
    assert not should_hold([0,4],[0,-1],[0,0],[0,0],0)

def test_visibility_boundary_keeps_body_intersecting_view():
    # Center just outside 69-degree frustum, but body still visible.
    a=np.deg2rad(40);p=np.array([np.cos(a),np.sin(a)])
    assert not should_hold(p,-p,[0,0],[0,0],0)

def test_pause_clock_does_not_jump_and_releases_on_turn():
    position=np.array([-2.,0.]);v=np.array([1.,0.]);dt=.01
    for _ in range(300):
        if not should_hold(position,v,[0,0],[0,0],0):position+=v*dt
    assert np.linalg.norm(position)>=.8-1e-8
    before=position.copy()
    assert not should_hold(position,v,[0,0],[0,0],np.pi)
    position+=v*dt
    assert np.linalg.norm(position-before)<=dt+1e-8
