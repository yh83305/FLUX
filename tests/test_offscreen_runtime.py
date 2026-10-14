import sys,types,json
import numpy as np
from offscreen_people_runtime import install_guard

def test_callback_hold_release_cleanup(monkeypatch,tmp_path):
    name='omni.anim.people.scripts.character_behavior'
    module=types.ModuleType(name)
    class Character:
        def set_variable(self,key,value):self.action=value
    class Behavior:
        prim_path='/person';navigation_manager=None;avoidanceOn=False
        def __init__(self):self.character=Character();self.p=np.array([-2.,0.,0.]);self.steps=0
        def get_current_position(self):return self.p
        def on_update(self,t,dt):self.p[0]+=dt;self.steps+=1
    module.CharacterBehavior=Behavior;monkeypatch.setitem(sys.modules,name,module)
    heading=[0.];original=Behavior.on_update
    reset,close=install_guard(lambda:[([0,0],[0,0],heading[0])],tmp_path/'events.jsonl')
    actor=Behavior()
    try:
        for k in range(200):actor.on_update(k*.01,.01)
        assert actor.p[0]<-.8
        steps=actor.steps;heading[0]=np.pi;actor.on_update(2.01,.01)
        assert actor.steps==steps+1
        events=[json.loads(x)['event'] for x in (tmp_path/'events.jsonl').read_text().splitlines()]
        assert events==['hold','release']
        reset()
    finally:close()
    assert Behavior.on_update is original
