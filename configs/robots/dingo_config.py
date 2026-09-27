import isaaclab.sim as sim_utils
import json
import math
import weakref
from pathlib import Path
import torch
import omni.physics.tensors.impl.api as physx
import omni.usd
from pxr import UsdGeom, UsdPhysics
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import Camera, ContactSensorCfg, patterns, CameraCfg, RayCasterCfg, OffsetCfg
from isaaclab.utils import configclass

FLUX_ROOT = Path(__file__).resolve().parents[2]

DINGO_CFG = ArticulationCfg(
    prim_path = "{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(FLUX_ROOT / "assets" / "robots" / "dingo.usd"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint","right_wheel_joint"],
            velocity_limit=100.0,
            effort_limit=20.0,
            stiffness=0.0,
            damping=1.0,
        ),
    },
)
DINGO_BASE_LINK = 'base_link'
DINGO_WHEEL_JOINTS = ["left_wheel_joint","right_wheel_joint"]
DINGO_WHEEL_RADIUS = 0.0591
DINGO_WHEEL_BASE = 0.22616
DINGO_THRESHOLD = 15.0
CAMERA_PROFILE_ID = "world060_pitch0_hfov69_v1"
CAMERA_HEIGHT_M = 0.60
CAMERA_HORIZONTAL_FOV_DEGREES = 69.0
CAMERA_FOCAL_LENGTH = 1.4
CAMERA_HORIZONTAL_APERTURE = 2.0 * CAMERA_FOCAL_LENGTH * math.tan(
    math.radians(CAMERA_HORIZONTAL_FOV_DEGREES) / 2.0
)
DINGO_CAMERA_TRANS = [0.0,0.0,CAMERA_HEIGHT_M]
DINGO_CAMERA_ROTS = [0.5, -0.5, 0.5, -0.5]
DINGO_IMAGEGOAL_TRANS = [5.0,0.0,0.3]
DINGO_IMAGEGOAL_ROTS = [0.5, -0.5, 0.5, -0.5]

DINGO_ContactCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/%s"%DINGO_BASE_LINK, 
                                    history_length=10, 
                                    track_air_time=True,
                                    update_period=0.02)


def level_follow_camera_pose(root_transforms_xyzw, ground_z, height=CAMERA_HEIGHT_M):
    """Follow robot XY/yaw while discarding chassis height, roll, pitch and USD scale."""
    roots = torch.as_tensor(root_transforms_xyzw)
    if roots.ndim != 2 or roots.shape[1] != 7 or not bool(torch.isfinite(roots).all()):
        raise ValueError("Robot transforms must be finite [N,7] XYZW poses")
    x, y, z, w = roots[:, 3:].unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    positions = roots[:, :3].clone()
    positions[:, 2] = torch.as_tensor(
        ground_z, device=roots.device, dtype=roots.dtype
    ) + float(height)
    orientations = torch.zeros((len(roots), 4), device=roots.device, dtype=roots.dtype)
    orientations[:, 0] = torch.cos(yaw / 2)
    orientations[:, 3] = torch.sin(yaw / 2)
    return positions, orientations


class DingoLevelFollowCamera(Camera):
    """The one Dingo policy camera, fixed at world height 0.60 m and pitch 0°."""

    def _initialize_impl(self):
        super()._initialize_impl()
        self._follow_sim = physx.create_simulation_view(self._backend)
        self._follow_sim.set_subspace_roots("/")
        robot_expr = self.cfg.prim_path.rsplit("/", 1)[0] + "/Robot"
        template = sim_utils.find_first_matching_prim(robot_expr)
        if template is None:
            raise RuntimeError(f"No robot for Dingo policy camera: {robot_expr}")
        template_path = template.GetPath().pathString
        roots = sim_utils.get_all_matching_child_prims(
            template_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
        )
        if len(roots) != 1:
            raise RuntimeError(
                "Dingo policy camera needs exactly one robot articulation per environment"
            )
        expression = robot_expr + roots[0].GetPath().pathString[len(template_path):]
        self._follow_root = self._follow_sim.create_articulation_view(
            expression.replace(".*", "*")
        )
        if self._follow_root.count != self._view.count:
            raise RuntimeError("Dingo policy camera/robot view counts differ")
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        self._ground_z = torch.tensor([
            float(cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath(path.rsplit("/", 1)[0])
            ).ExtractTranslation()[2])
            for path in self._view.prim_paths
        ], device=self._device)
        simulation = sim_utils.SimulationContext.instance()
        if not hasattr(simulation, "_dingo_policy_cameras"):
            simulation._dingo_policy_cameras = weakref.WeakSet()
            original_render = simulation.render

            def render_with_policy_camera(*args, **kwargs):
                for sensor in list(simulation._dingo_policy_cameras):
                    if sensor.is_initialized:
                        sensor.synchronize_pose()
                return original_render(*args, **kwargs)

            simulation.render = render_with_policy_camera
        simulation._dingo_policy_cameras.add(self)
        self.synchronize_pose()
        self._profile_logged = False

    def synchronize_pose(self):
        # ManagerBasedRLEnv renders before refreshing asset-data caches, so use
        # live PhysX root transforms rather than robot.data here.
        position, orientation = level_follow_camera_pose(
            self._follow_root.get_root_transforms(),
            self._ground_z,
            self.cfg.world_height_m,
        )
        self.set_world_poses(position, orientation, convention="world")

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        position = self._data.pos_w[env_ids]
        rotation = self._data.quat_w_world[env_ids]
        hfov = torch.rad2deg(2 * torch.atan(
            self.cfg.width / (2 * self._data.intrinsic_matrices[env_ids, 0, 0])
        ))
        if not bool(torch.all(torch.abs(
            position[:, 2] - self._ground_z[env_ids] - self.cfg.world_height_m
        ) < 1e-4)):
            raise RuntimeError("Dingo policy camera world height drifted")
        if not bool(torch.all(torch.abs(rotation[:, 1:3]) < 1e-4)):
            raise RuntimeError("Dingo policy camera inherited chassis pitch/roll")
        if not bool(torch.all(torch.abs(
            hfov - CAMERA_HORIZONTAL_FOV_DEGREES
        ) < 1e-3)):
            raise RuntimeError("Dingo policy camera horizontal FOV drifted")
        if not getattr(self, "_profile_logged", True):
            print("[CameraProfile] " + json.dumps(self.profile_report()), flush=True)
            self._profile_logged = True

    def profile_report(self):
        return {
            "id": self.cfg.profile_id,
            "world_height_m": self.cfg.world_height_m,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "horizontal_fov_deg": CAMERA_HORIZONTAL_FOV_DEGREES,
            "resolution": [self.cfg.width, self.cfg.height],
            "mount": "unscaled environment prim; live PhysX XY/yaw before render",
            "ground_reference": "environment origin Z (flat DynBench floor)",
            "intrinsics": self._data.intrinsic_matrices.detach().cpu().tolist(),
        }


@configclass
class DingoLevelFollowCameraCfg(CameraCfg):
    class_type: type = DingoLevelFollowCamera
    world_height_m: float = CAMERA_HEIGHT_M
    profile_id: str = CAMERA_PROFILE_ID

DINGO_CameraCfg = DingoLevelFollowCameraCfg(
    prim_path="{ENV_REGEX_NS}/policy_camera",
    update_period=0.05,
    height=360,
    width=640,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=CAMERA_FOCAL_LENGTH, focus_distance=0.205,
        horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
        clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=DINGO_CAMERA_TRANS, rot=(1.,0.,0.,0.), convention="world"),
)

DINGO_ImageGoal_CameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/goal_cam",
    update_period=0.05,
    height=360,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.4, focus_distance=0.205, horizontal_aperture=1.88, clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=DINGO_IMAGEGOAL_TRANS, rot=DINGO_IMAGEGOAL_ROTS, convention="ros"),
)

DINGO_MetricCameraCfg = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/%s/narritor_cam"%DINGO_BASE_LINK,
    update_period=0.05,
    height=90,
    width=160,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=1.4, focus_distance=0.205, horizontal_aperture=1.88, clipping_range=(0.01, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(pos=[0.0,0.0,1.0], rot=DINGO_CAMERA_ROTS, convention="ros"),
)
