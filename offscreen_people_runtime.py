"""Install a reversible guard around omni.anim.people's character updates."""
import json
import numpy as np
from offscreen_guard import should_hold,GuardConfig

def install_guard(robot_states, log_path, horizontal_fov_deg=69.):
    import omni.kit.app
    from omni.kit.scripting import ScriptManager
    manager = ScriptManager.get_instance()
    originals = {}
    counters = dict(callbacks=0, holds=0, releases=0)
    state={};config=GuardConfig(horizontal_fov_deg=horizontal_fov_deg)
    log=open(log_path,'a',buffering=1)
    def update(self,current_time,delta_time):
        original = originals[type(self)]
        counters["callbacks"] += 1
        if counters["callbacks"] == 1 or counters["callbacks"] % 1000 == 0:
            log.write(json.dumps(dict(event="heartbeat", **counters))+"\n")
        if self.character is None:return original(self,current_time,delta_time)
        key=str(self.prim_path);p=np.asarray(self.get_current_position(),dtype=float)[:2].copy()
        previous=state.get(key)
        v=np.zeros(2)
        if previous and current_time>previous[0]:
            v=previous[2] if previous[3] else (p-previous[1])/(current_time-previous[0])
        hold=any(should_hold(p,v,robot,cam,yaw,config) for robot,cam,yaw in robot_states())
        if hold:
            if previous is None or not previous[3]:
                counters["holds"] += 1
                log.write(json.dumps(dict(time=current_time,person=key,event='hold',xy=p.tolist(),velocity=v.tolist()))+'\n')
            self.character.set_variable('Action','Idle')
            state[key]=(current_time,p,v,True)
            if self.navigation_manager and self.avoidanceOn:self.navigation_manager.publish_character_positions(delta_time,.5)
            return
        if previous and previous[3]:
            counters["releases"] += 1
            self.character.set_variable('Action','Walk')
            log.write(json.dumps(dict(time=current_time,person=key,event='release',xy=p.tolist()))+'\n')
        state[key]=(current_time,p,v,False)
        return original(self,current_time,delta_time)
    def bind(instance):
        cls = type(instance)
        if cls.__name__ != "CharacterBehavior" or cls in originals:
            return
        originals[cls] = cls.on_update
        cls.on_update = update
        log.write(json.dumps(dict(event="bind", module=cls.__module__,
                                 original_module=originals[cls].__module__))+"\n")
    def scan(_event=None):
        manager._foreach_script_instance(bind)
    scan()
    subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(scan)
    log.write(json.dumps(dict(event="installed", bound_classes=len(originals)))+"\n")
    def reset():
        state.clear()
        scan()
        log.write(json.dumps(dict(event="reset", **counters))+"\n")
    def close():
        nonlocal subscription
        subscription = None
        for cls, original in originals.items():
            cls.on_update = original
        log.write(json.dumps(dict(event="summary", bound_classes=len(originals), **counters))+"\n")
        log.close()
    return reset,close
