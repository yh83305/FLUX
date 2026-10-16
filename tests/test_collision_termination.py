import numpy as np
from collision_termination import collision_mask,episode_success

def test_collision_threshold_and_empty():
 assert collision_mask([[0,0]],[]).tolist()==[False]
 assert collision_mask([[0,0],[2,0]],[[.449,0]]).tolist()==[True,False]
 assert not collision_mask([[0,0]],[[.45,0]])[0]

def test_collision_wins_over_goal():
 assert episode_success(True,True)==0
 assert episode_success(True,False)==1
 assert episode_success(False,False)==0
