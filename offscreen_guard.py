"""Environment-side pedestrian yielding, not a robot safety controller.
Visibility means horizontal camera frustum (not occlusion). Pausing cannot prevent
collisions caused by the robot driving into an already stationary pedestrian.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class GuardConfig:
    horizontal_fov_deg: float = 69.
    clearance_m: float = .8
    horizon_s: float = 1.
    pedestrian_radius_m: float = .3

def should_hold(person, velocity, robot, camera_xy, camera_yaw, config=GuardConfig()):
    p=np.asarray(person)[:2];v=np.asarray(velocity)[:2];r=np.asarray(robot)[:2]
    if not np.isfinite(np.r_[p,v,r,camera_xy,camera_yaw]).all():raise ValueError('Nonfinite guard state')
    delta=p-np.asarray(camera_xy)[:2];distance=np.linalg.norm(delta)
    bearing=np.arctan2(delta[1],delta[0])-camera_yaw
    bearing=abs(np.arctan2(np.sin(bearing),np.cos(bearing)))
    # A pedestrian whose bounding disc intersects the FOV is treated as visible.
    padding=np.arcsin(min(1.,config.pedestrian_radius_m/max(distance,1e-9)))
    if distance<=config.pedestrian_radius_m or bearing<=np.deg2rad(config.horizontal_fov_deg)/2+padding:return False
    speed2=float(v@v)
    if speed2<1e-10:return False
    rel=p-r
    if float(rel@v)>=0:return False
    t=np.clip(-float(rel@v)/speed2,0,config.horizon_s)
    return bool(np.linalg.norm(rel+t*v)<config.clearance_m)
