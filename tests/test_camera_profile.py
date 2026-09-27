import math
import torch
from camera_profile import level_follow_pose, HORIZONTAL_APERTURE, FOCAL_LENGTH


def test_follow_pose_ignores_chassis_height_roll_pitch():
    from scipy.spatial.transform import Rotation
    quats=Rotation.from_euler('xyz',[[.2,-.1,.7],[-.3,.25,-1.1]]).as_quat()
    roots=torch.cat((torch.tensor([[2.,3.,.04],[-4.,5.,.9]]),torch.tensor(quats,dtype=torch.float32)),dim=1)
    positions,orientations=level_follow_pose(roots,torch.tensor([0.,2.]))
    torch.testing.assert_close(positions,torch.tensor([[2.,3.,.6],[-4.,5.,2.6]]))
    torch.testing.assert_close(orientations[:,1:3],torch.zeros(2,2))
    torch.testing.assert_close(2*torch.atan2(orientations[:,3],orientations[:,0]),torch.tensor([.7,-1.1]))


def test_aperture_matches_exact_69_degree_fov():
    assert abs(math.degrees(2*math.atan(HORIZONTAL_APERTURE/(2*FOCAL_LENGTH)))-69.)<1e-10
