"""Unscaled, level policy camera synchronized before each simulator render."""
from __future__ import annotations
import json
import weakref
import torch
import omni.physics.tensors.impl.api as physx
import omni.usd
from pxr import UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from camera_profile import CAMERA_PROFILE_ID, CAMERA_HEIGHT_M, HORIZONTAL_FOV_DEGREES, level_follow_pose


class LevelFollowCamera(Camera):
    def _initialize_impl(self):
        super()._initialize_impl()
        self._follow_sim = physx.create_simulation_view(self._backend)
        self._follow_sim.set_subspace_roots("/")
        robot_expr = self.cfg.prim_path.rsplit("/",1)[0]+"/Robot"
        template = sim_utils.find_first_matching_prim(robot_expr)
        if template is None:
            raise RuntimeError(f"No robot for level camera: {robot_expr}")
        template_path = template.GetPath().pathString
        roots = sim_utils.get_all_matching_child_prims(
            template_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI))
        if len(roots) != 1:
            raise RuntimeError("Level camera needs exactly one robot articulation per environment")
        expression = robot_expr+roots[0].GetPath().pathString[len(template_path):]
        self._follow_root = self._follow_sim.create_articulation_view(expression.replace(".*","*"))
        if self._follow_root.count != self._view.count:
            raise RuntimeError("Camera/robot view counts differ")
        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.XformCache()
        self._ground_z = torch.tensor([
            float(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path.rsplit('/',1)[0])).ExtractTranslation()[2])
            for path in self._view.prim_paths
        ],device=self._device)
        sim = sim_utils.SimulationContext.instance()
        if not hasattr(sim,"_flux_level_cameras"):
            sim._flux_level_cameras = weakref.WeakSet()
            original_render = sim.render
            def render_with_level_cameras(*args,**kwargs):
                for sensor in list(sim._flux_level_cameras):
                    if sensor.is_initialized:
                        sensor.synchronize_pose()
                return original_render(*args,**kwargs)
            sim.render = render_with_level_cameras
        sim._flux_level_cameras.add(self)
        self.synchronize_pose()
        self._profile_logged = False

    def synchronize_pose(self):
        # Read live PhysX transforms: scene/asset data caches are updated AFTER
        # render in ManagerBasedRLEnv, so reading robot.data here can lag a step.
        position, orientation = level_follow_pose(
            self._follow_root.get_root_transforms(), self._ground_z, self.cfg.world_height_m)
        self.set_world_poses(position,orientation,convention="world")

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        position = self._data.pos_w[env_ids]
        rotation = self._data.quat_w_world[env_ids]
        hfov = torch.rad2deg(2*torch.atan(self.cfg.width/(2*self._data.intrinsic_matrices[env_ids,0,0])))
        if not bool(torch.all(torch.abs(position[:,2]-self._ground_z[env_ids]-self.cfg.world_height_m)<1e-4)):
            raise RuntimeError("Level camera world height drifted")
        if not bool(torch.all(torch.abs(rotation[:,1:3])<1e-4)):
            raise RuntimeError("Level camera inherited chassis pitch/roll")
        if not bool(torch.all(torch.abs(hfov-HORIZONTAL_FOV_DEGREES)<1e-3)):
            raise RuntimeError("Level camera horizontal FOV drifted")
        if not getattr(self,"_profile_logged",True):
            print("[CameraProfile] "+json.dumps(self.profile_report()),flush=True)
            self._profile_logged=True

    def profile_report(self):
        return {"id":self.cfg.profile_id,"world_height_m":self.cfg.world_height_m,
                "pitch_deg":0.0,"roll_deg":0.0,"horizontal_fov_deg":HORIZONTAL_FOV_DEGREES,
                "resolution":[self.cfg.width,self.cfg.height],
                "mount":"unscaled environment prim, live PhysX XY/yaw before render",
                "ground_reference":"environment origin Z (flat DynBench floor)",
                "intrinsics":self._data.intrinsic_matrices.detach().cpu().tolist()}


@configclass
class LevelFollowCameraCfg(CameraCfg):
    class_type: type = LevelFollowCamera
    world_height_m: float = CAMERA_HEIGHT_M
    profile_id: str = CAMERA_PROFILE_ID
