#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import time
import h5py
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy

ROBOTIS_LAB_DIR = Path("/home/rokey/dev_ws/robotis_lab/scripts/sim2real/bringup")
if str(ROBOTIS_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOTIS_LAB_DIR))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay state-based HDF5 and capture vision data.")
parser.add_argument("--input_dir", type=str, default="/home/rokey/dev_ws/datasets", help="Directory with original HDF5 files")
parser.add_argument("--output_dir", type=str, default="/home/rokey/dev_ws/datasets/vision", help="Directory to save vision HDF5 files")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

print("[TRACE] Creating AppLauncher"); sys.stdout.flush()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[TRACE] AppLauncher created"); sys.stdout.flush()

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from robotis_lab.assets.robots import FFW_SH5_CFG
from common import robotis_config as cfg

from pxr import Gf, UsdGeom, Sdf

@configclass
class StandaloneSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.88, 0.85), intensity=4500.0),
    )
    rack = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/custom_rack2.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -1.5, 0.0), rot=(0.0, 0.0, 0.0, 1.0))
    )
    pedestal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/belt.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    )
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True,
                linear_damping=0.1, angular_damping=5.0,
                max_depenetration_velocity=0.3, enable_gyroscopic_forces=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.38, 0.08)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", static_friction=2.0, dynamic_friction=1.8, restitution=0.0,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0))
    )
    robot: ArticulationCfg = None

def _setup_cameras():
    import omni.replicator.core as rep
    from isaacsim.core.utils.stage import get_current_stage
    stage = get_current_stage()
    cam_paths = {}
    
    # Setup Top View
    top_cam_path = "/World/TopViewCamera"
    existing = stage.GetPrimAtPath(top_cam_path)
    if existing.IsValid():
        stage.RemovePrim(Sdf.Path(top_cam_path))
    top_cam = UsdGeom.Camera.Define(stage, Sdf.Path(top_cam_path))
    top_cam.GetFocalLengthAttr().Set(12.0)
    top_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))
    
    eye = Gf.Vec3d(0.0, 0.0, 3.0)
    target = Gf.Vec3d(0.4, 0.0, 0.0)
    up = Gf.Vec3d(1.0, 0.0, 0.0)
    
    forward = (target - eye).GetNormalized()
    right = Gf.Cross(forward, up).GetNormalized()
    up_real = Gf.Cross(right, forward).GetNormalized()
    
    m = Gf.Matrix4d(
         right[0],    right[1],    right[2],   0,
         up_real[0],  up_real[1],  up_real[2], 0,
        -forward[0], -forward[1], -forward[2], 0,
         eye[0],      eye[1],      eye[2],     1,
    )
    xform = UsdGeom.Xformable(top_cam.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(m)
    cam_paths["TopView"] = top_cam_path

    # Left and Right cameras are deep in the articulation tree
    cam_paths["Left Camera"] = "/World/envs/env_0/Robot/base_link/arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/Left_Camera"
    cam_paths["Right Camera"] = "/World/envs/env_0/Robot/base_link/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/Right_Camera"
    
    return cam_paths

def replay_and_capture():
    import omni.replicator.core as rep
    print("[TRACE] Entering replay_and_capture"); sys.stdout.flush()
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu", # Force CPU physics to avoid RTX 5080 PhysX crash
        dt=1.0 / cfg.STEP_HZ,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=16,
            min_velocity_iteration_count=2,
            enable_stabilization=True,
        )
    )
    print("[TRACE] Created SimulationCfg"); sys.stdout.flush()
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 0.0])
    
    scene_cfg = StandaloneSceneCfg(num_envs=1, env_spacing=2.0)
    robot_cfg = deepcopy(FFW_SH5_CFG)
    robot_cfg.spawn.usd_path = FFW_SH5_CFG.spawn.usd_path
    robot_cfg.spawn.rigid_props.disable_gravity = True
    robot_cfg.init_state.pos = cfg.ROBOT_POS
    scene_cfg.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    print("[TRACE] Creating InteractiveScene"); sys.stdout.flush()
    scene = InteractiveScene(scene_cfg)
    print("[TRACE] Calling sim.reset()"); sys.stdout.flush()
    sim.reset()
    print("[TRACE] sim.reset() done"); sys.stdout.flush()
    
    camera_paths = _setup_cameras()
    annotators = {}
    print("[INFO] Replicator 카메라 어노테이터 설정 중...")
    for cam_name, cam_path in camera_paths.items():
        rp = rep.create.render_product(cam_path, (160, 120))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([rp])
        annotators[cam_name] = rgb_annot
    print(f"[INFO] 준비된 비전 카메라: {list(annotators.keys())}")
    
    files = glob.glob(os.path.join(args_cli.input_dir, "*.hdf5"))
    files = [f for f in files if "vision" not in f]
    
    for input_file in files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(args_cli.output_dir, f"vision_{filename}")
        if os.path.exists(output_file):
            print(f"[INFO] Skipping {filename}, already processed.")
            continue
            
        print(f"\n[INFO] Processing {filename} -> {output_file}")
        
        # WARM UP THE CAMERAS FOR THIS FILE
        print(f"[INFO] {filename} 카메라 셰이더 워밍업 중... (약 5초 대기)")
        for _ in range(10):
            sim.step(render=True)
        print(f"[INFO] 워밍업 완료!")
        
        with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "w") as f_out:
            if "data" not in f_in:
                continue
                
            data_in = f_in["data"]
            data_out = f_out.create_group("data")
            
            demos = list(f_in["data"].keys())
            num_demos = len(demos)
            for ep_idx, demo_key in enumerate(demos):
                demo_in = data_in[demo_key]
                num_samples = demo_in.attrs["num_samples"]
                
                demo_out = data_out.create_group(demo_key)
                for attr_name, attr_val in demo_in.attrs.items():
                    demo_out.attrs[attr_name] = attr_val
                
                demo_out.copy(demo_in["actions"], "actions")
                demo_out.copy(demo_in["cmd_vel"], "cmd_vel")
                demo_out.copy(demo_in["rewards"], "rewards")
                demo_out.copy(demo_in["dones"], "dones")
                
                obs_out = demo_out.create_group("obs")
                for key in demo_in["obs"].keys():
                    if key != "images":
                        obs_out.copy(demo_in["obs"][key], key)
                
                img_grp = obs_out.create_group("images")
                images_buffers = {cam: [] for cam in annotators.keys()}
                
                robot = scene["robot"]
                for i in range(num_samples):
                    joint_pos = demo_in["obs"]["joint_positions"][i]
                    joint_vel = demo_in["obs"]["joint_velocities"][i]
                    robot_pose = demo_in["obs"]["robot_pose"][i]
                    
                    pos_tensor = torch.tensor(joint_pos, dtype=torch.float32, device=sim.device).unsqueeze(0)
                    vel_tensor = torch.tensor(joint_vel, dtype=torch.float32, device=sim.device).unsqueeze(0)
                    robot.write_joint_state_to_sim(pos_tensor, vel_tensor)
                    
                    root_pose_tensor = torch.zeros((1, 7), dtype=torch.float32, device=sim.device)
                    root_pose_tensor[0, :3] = torch.tensor(robot_pose[:3], device=sim.device)
                    root_pose_tensor[0, 3:7] = torch.tensor([robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6]], device=sim.device) 
                    robot.write_root_pose_to_sim(root_pose_tensor)
                    
                    if "box_pose" in demo_in["obs"]:
                        box_pose = demo_in["obs"]["box_pose"][i]
                        box_pose_tensor = torch.zeros((1, 7), dtype=torch.float32, device=sim.device)
                        box_pose_tensor[0, :3] = torch.tensor(box_pose[:3], device=sim.device)
                        box_pose_tensor[0, 3:7] = torch.tensor([box_pose[3], box_pose[4], box_pose[5], box_pose[6]], device=sim.device)
                        scene["box"].write_root_pose_to_sim(box_pose_tensor)
                        
                    if "rack_pose" in demo_in["obs"]:
                        rack_pose = demo_in["obs"]["rack_pose"][i]
                        rack_pose_tensor = torch.zeros((1, 7), dtype=torch.float32, device=sim.device)
                        rack_pose_tensor[0, :3] = torch.tensor(rack_pose[:3], device=sim.device)
                        rack_pose_tensor[0, 3:7] = torch.tensor([rack_pose[3], rack_pose[4], rack_pose[5], rack_pose[6]], device=sim.device)
                        scene["rack"].write_root_pose_to_sim(rack_pose_tensor)
                        
                    scene.write_data_to_sim()
                    sim.step(render=True)
                    scene.update(sim.get_physics_dt())
                    
                    for cam_name, annotator in annotators.items():
                        img_data = annotator.get_data()
                        if img_data is not None and getattr(img_data, 'size', 0) > 0 and len(img_data.shape) >= 3:
                            images_buffers[cam_name].append(img_data[..., :3])
                        else:
                            images_buffers[cam_name].append(np.zeros((120, 160, 3), dtype=np.uint8))
                            
                    if i % 10 == 0 or i == num_samples - 1:
                        progress = (i + 1) / num_samples
                        bar_len = 30
                        filled = int(bar_len * progress)
                        bar = '█' * filled + '░' * (bar_len - filled)
                        print(f"\r  [에피소드 {ep_idx+1}/{num_demos}] {demo_key} | {bar} {(progress*100):.1f}% ({i+1}/{num_samples} 프레임)", end="", flush=True)
                        
                print()
                for cam_name, frames in images_buffers.items():
                    img_array = np.stack(frames)
                    img_grp.create_dataset(cam_name, data=img_array, compression="gzip")
                    
                # Flush the file to disk so progress is saved and visible in file size
                f_out.flush()
                    
        print(f"[INFO] Finished saving to {output_file}")

if __name__ == "__main__":
    try:
        replay_and_capture()
    finally:
        simulation_app.close()
