import sys,types,json
import numpy as np
from offscreen_people_runtime import install_guard

def test_callback_hold_release_cleanup(monkeypatch,tmp_path):
    name='omni.anim.people.scripts.character_behavior'
    module=types.ModuleType(name)
    class Character:
        def set_variable(self,key,value):
            raise AssertionError("Guard must not directly change animation variables")
    class Behavior:
        prim_path='/person';navigation_manager=None;avoidanceOn=False
        def __init__(self):
            self.character=Character();self.p=np.array([-2.,0.,0.]);self.steps=0
            self.current_command=types.SimpleNamespace(actual_walk_speed=1.,desired_walk_speed=1.)
            self.last_dt=None
        def get_current_position(self):return self.p
        def on_update(self,t,dt):self.p[0]+=dt;self.steps+=1;self.last_dt=dt
    module.CharacterBehavior=Behavior;monkeypatch.setitem(sys.modules,name,module)
    Actual = type("CharacterBehavior", (Behavior,), {"__module__": "kit_path_loaded"})
    actor = Actual()
    instances = [actor]
    manager = types.SimpleNamespace(_foreach_script_instance=lambda fn: [fn(x) for x in instances])
    callbacks = []
    stream = types.SimpleNamespace(create_subscription_to_pop=lambda fn: callbacks.append(fn))
    app = types.ModuleType("omni.kit.app")
    app.get_app = lambda: types.SimpleNamespace(get_update_event_stream=lambda: stream)
    scripting = types.ModuleType("omni.kit.scripting")
    scripting.ScriptManager = types.SimpleNamespace(get_instance=lambda: manager)
    omni = types.ModuleType("omni");kit=types.ModuleType("omni.kit")
    omni.kit=kit;kit.app=app
    for key,value in {"omni":omni,"omni.kit":kit,"omni.kit.app":app,
                      "omni.kit.scripting":scripting}.items():
        monkeypatch.setitem(sys.modules,key,value)
    heading=[0.];original=Behavior.on_update
    reset,close=install_guard(lambda:[([0,0],[0,0],heading[0])],tmp_path/'events.jsonl')
    assert Behavior.on_update is original  # The unrelated normal-import class stays untouched.
    try:
        for k in range(200):actor.on_update(k*.01,.01)
        assert actor.p[0]<-.8
        assert actor.steps == 200  # Original bookkeeping still executes while held.
        assert actor.last_dt == 0
        assert actor.current_command.actual_walk_speed == 0
        position=actor.p.copy()
        steps=actor.steps;heading[0]=np.pi;actor.on_update(2.01,.01)
        assert actor.steps == steps+1
        assert np.allclose(actor.p,position)
        actor.p[0] = -3.0
        actor.on_update(2.02,.01)
        assert actor.steps==steps+2
        assert actor.last_dt == .01
        events=[json.loads(x)['event'] for x in (tmp_path/'events.jsonl').read_text().splitlines()]
        assert 'hold' in events and 'release' in events and 'heartbeat' in events
        Reloaded = type("CharacterBehavior", (Behavior,), {"__module__": "kit_reloaded"})
        instances.append(Reloaded())
        callbacks[0](None)
        assert Reloaded.on_update is not original
        reset()
    finally:close()
    assert Behavior.on_update is original
    assert Actual.on_update is original
    assert Reloaded.on_update is original
