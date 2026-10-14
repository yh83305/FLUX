"""Install a reversible guard around omni.anim.people's character updates."""
import json
import numpy as np
from offscreen_guard import should_hold,GuardConfig

def install_guard(robot_states, log_path, horizontal_fov_deg=69.):
    from omni.anim.people.scripts.character_behavior import CharacterBehavior
    original=CharacterBehavior.on_update
    state={};config=GuardConfig(horizontal_fov_deg=horizontal_fov_deg)
    log=open(log_path,'a',buffering=1)
    def update(self,current_time,delta_time):
        if self.character is None:return original(self,current_time,delta_time)
        key=str(self.prim_path);p=np.asarray(self.get_current_position(),dtype=float)[:2]
        previous=state.get(key)
        v=np.zeros(2)
        if previous and current_time>previous[0]:
            v=previous[2] if previous[3] else (p-previous[1])/(current_time-previous[0])
        hold=any(should_hold(p,v,robot,cam,yaw,config) for robot,cam,yaw in robot_states())
        if hold:
            if previous is None or not previous[3]:log.write(json.dumps(dict(time=current_time,person=key,event='hold',xy=p.tolist()))+'\n')
            self.character.set_variable('Action','Idle')
            state[key]=(current_time,p,v,True)
            if self.navigation_manager and self.avoidanceOn:self.navigation_manager.publish_character_positions(delta_time,.5)
            return
        if previous and previous[3]:
            self.character.set_variable('Action','Walk')
            log.write(json.dumps(dict(time=current_time,person=key,event='release',xy=p.tolist()))+'\n')
        state[key]=(current_time,p,v,False)
        return original(self,current_time,delta_time)
    CharacterBehavior.on_update=update
    def reset():state.clear()
    def close():
        CharacterBehavior.on_update=original;log.close()
    return reset,close
