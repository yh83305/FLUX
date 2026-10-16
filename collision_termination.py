"""Pedestrian collision termination evaluated inside env.step, before auto-reset."""
import numpy as np

def collision_mask(robot_xy,people_xy,threshold=.45):
 robots=np.asarray(robot_xy,dtype=float).reshape(-1,2);people=np.asarray(people_xy,dtype=float)
 if people.size==0:return np.zeros(len(robots),dtype=bool)
 people=people.reshape(-1,2)
 if not np.isfinite(robots).all() or not np.isfinite(people).all():raise ValueError('Nonfinite collision geometry')
 return (np.linalg.norm(robots[:,None]-people[None],axis=-1)<threshold).any(axis=1)

def pedestrian_collision(env):
 import torch
 from socialnav_metrics import get_people_positions
 people,_,_=get_people_positions(env)
 robot=env.scene.articulations['robot'].data.root_pos_w[:,:2]
 mask=collision_mask(robot.detach().cpu().numpy(),np.asarray(people)[:,:2] if len(people) else [])
 result=torch.as_tensor(mask,device=robot.device,dtype=torch.bool)
 # env.step may reset done environments before the evaluator receives infos.
 env._bench_pedestrian_collision=result.clone()
 env._bench_terminal_person_distance=np.linalg.norm(robot.detach().cpu().numpy()[:,None]-np.asarray(people)[None,:,:2],axis=-1).min(axis=1) if len(people) else np.full(len(robot),np.inf)
 return result

def episode_success(reached,collision):
 return float(bool(reached) and not bool(collision))
