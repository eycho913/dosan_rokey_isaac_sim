# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Howon Kim

import argparse
import os
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

from isaaclab.app import AppLauncher


ROBOTIS_LAB_DIR = Path("/home/rokey/dev_ws/robotis_lab/scripts/sim2real/bringup")
if str(ROBOTIS_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOTIS_LAB_DIR))

from common import robotis_config as cfg

# CLI and app launch
parser = argparse.ArgumentParser(description="FFW SH5 DDS bringup for Isaac Sim.")
parser.add_argument("--disable_head", action="store_true", help="Do not subscribe to the head topic.")
parser.add_argument("--disable_lift", action="store_true", help="Do not subscribe to the lift topic.")
parser.add_argument("--disable_cmd_vel", action="store_true", help="Do not subscribe to cmd_vel for the swerve base.")
parser.add_argument("--domain_id", type=int, default=None, help="DDS domain id. Defaults to ROS_DOMAIN_ID or 0.")
parser.add_argument("--enable_gravity", action="store_true", help="Enable gravity on the SH5 rigid bodies.")
parser.add_argument("--enable_environment", action="store_true", help="Spawn the environment USD.")
parser.add_argument(
    "--enable_camera_views",
    action="store_true",
    help="Open Isaac Sim viewport windows for overview, Head_Camera, Left_Camera, and Right_Camera.",
)
parser.add_argument(
    "--enable_ros2_cameras",
    action="store_true",
    help="Enable ROS 2 Camera publishing for the VR views (Left and Right Camera).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# [XR Extension 초기화] OpenXRDevice가 임포트되기 전에 반드시 XR 확장이 켜져 있어야 합니다.
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.kit.xr.profile.vr")
enable_extension("omni.kit.xr.core")

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from cyclonedds.core import Qos, Policy
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from robotis_dds_python.idl.builtin_interfaces.msg import Time_
from robotis_dds_python.idl.geometry_msgs.msg import (
    Point_,
    Pose_,
    PoseWithCovariance_,
    Quaternion_,
    Transform_,
    TransformStamped_,
    Twist_,
    TwistWithCovariance_,
    Vector3_,
)
from robotis_dds_python.idl.nav_msgs.msg import Odometry_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.tf2_msgs.msg import TFMessage_
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.tools.topic_manager import TopicManager

from robotis_lab.assets.robots import (
    FFW_SH5_CFG,
    SH5_SWERVE_MODULE_ANGLE_OFFSETS,
    SH5_SWERVE_MODULE_X_OFFSETS,
    SH5_SWERVE_MODULE_Y_OFFSETS,
    SH5_SWERVE_STEERING_JOINTS,
    SH5_SWERVE_WHEEL_RADIUS,
    SH5_SWERVE_WHEEL_JOINTS,
)
from common.environment import (
    make_card_boxes_graspable,
    make_simple_warehouse_environment_cfg,
)
from common.odometry import SwerveOdometry, yaw_to_quaternion
from common.swerve_drive import SwerveDriveController, SwerveModule


# ========== Scene Setup ==========

from isaaclab.assets import RigidObjectCfg
import h5py
import numpy as np

class VRDemonstrationLogger:
    def __init__(self, output_dir="/home/rokey/dev_ws/datasets", filename_prefix="coupang_demo"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.hdf5")
        self.reset_episode_buffer()
        self.episode_counter = 0
        self.is_recording = False
        print(f"[VR Logger] 데이터 로거 초기화. 저장 경로: {self.filepath}")

    def reset_episode_buffer(self):
        self.buffer = {
            "obs/joint_positions": [],
            "obs/joint_velocities": [],
            "actions/joint_targets": [],
            "rewards": [],
            "dones": []
        }
        
    def start_recording(self):
        if not self.is_recording:
            self.reset_episode_buffer()
            self.is_recording = True
            print(f"[VR Logger] 🔴 에피소드 {self.episode_counter} 녹화 시작!")

    def stop_recording_and_save(self):
        if self.is_recording and len(self.buffer["actions/joint_targets"]) > 0:
            self._save_episode_to_hdf5()
            print(f"[VR Logger] ⬛ 에피소드 {self.episode_counter} 저장 완료 (스텝 수: {len(self.buffer['actions/joint_targets'])})")
            self.episode_counter += 1
            self.is_recording = False

    def cancel_recording(self):
        if self.is_recording:
            print(f"[VR Logger] 🗑️ 에피소드 {self.episode_counter} 녹화 취소 (저장하지 않고 버림)")
            self.is_recording = False
            self.reset_episode_buffer()

    def log_step(self, joint_pos, joint_vel, action_target, reward=0.0, done=False):
        if not self.is_recording:
            return
        self.buffer["obs/joint_positions"].append(np.array(joint_pos, dtype=np.float32))
        self.buffer["obs/joint_velocities"].append(np.array(joint_vel, dtype=np.float32))
        self.buffer["actions/joint_targets"].append(np.array(action_target, dtype=np.float32))
        self.buffer["rewards"].append(np.float32(reward))
        self.buffer["dones"].append(bool(done))

    def _save_episode_to_hdf5(self):
        with h5py.File(self.filepath, "a") as f:
            if "data" not in f:
                data_grp = f.create_group("data")
            else:
                data_grp = f["data"]
            ep_grp = data_grp.create_group(f"demo_{self.episode_counter}")
            ep_grp.attrs["num_samples"] = len(self.buffer["actions/joint_targets"])
            obs_grp = ep_grp.create_group("obs")
            obs_grp.create_dataset("joint_positions", data=np.array(self.buffer["obs/joint_positions"]))
            obs_grp.create_dataset("joint_velocities", data=np.array(self.buffer["obs/joint_velocities"]))
            ep_grp.create_dataset("actions", data=np.array(self.buffer["actions/joint_targets"]))
            ep_grp.create_dataset("rewards", data=np.array(self.buffer["rewards"]))
            ep_grp.create_dataset("dones", data=np.array(self.buffer["dones"]))

# =========================================================================================
# [데이터 수집 환경(Scene) 설정 클래스]
# 이 클래스(CoupangSceneCfg) 안에서 시뮬레이션 환경의 모든 물체(작업대, 상자, 바닥, 빛 등)를 세팅합니다.
# 
# 1. 물체 추가/수정: AssetBaseCfg 또는 RigidObjectCfg를 사용해 물체를 추가합니다.
# 2. 위치/회전 수정: init_state의 pos(X, Y, Z 미터 단위)와 rot(사원수 W, X, Y, Z)를 변경합니다.
# 3. 새로운 상자 추가: box2 = RigidObjectCfg(...) 형태로 변수를 새로 만들어주면 씬에 자동 추가됩니다.
# =========================================================================================
@configclass
class CoupangSceneCfg(InteractiveSceneCfg):
    # 1. 바닥(Ground) 및 조명(Light) 설정
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/custom_rack2.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -1.2, 0.0),  # 로봇 정면 우측 (Y=-0.8m)
            rot=(0.0, 0.0, 0.0, 1.0)  # Z축 기준 90도 회전
        )
    )
    
    # 3. 상자 받침대(Pedestal) 설정 (상자를 로봇 가까이 올려두기 위한 투명/회색 테이블 역할)
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/belt.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # 👇 벨트 위치 조정 (X, Y, Z) / Y값을 바꾸면 좌우 간격이 조절됩니다.
            pos=(0.5, 0.0, 0.0),  # 로봇 정면 좌측 (Y=2.0m)
            rot=(1.0, 0.0, 0.0, 0.0)  # 원래 방향으로 90도 원복
        )
    )
    
    # 4. 목표물 상자(Box) 설정 (로봇이 실제로 집어야 하는 대상 물체)
    # 크기(size), 질량(mass), 마찰력(friction) 등을 수정하여 다양한 훈련 환경을 구축할 수 있습니다.
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.10), # 손 크기에 맞춰 10cm -> 6cm로 축소
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                static_friction=1000.0,
                dynamic_friction=1000.0,
                restitution=0.0,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # 👇 상자 위치 조정 (X, Y, Z) / 벨트를 옮기면 이 Y값도 똑같이 맞춰주세요.
            pos=(0.7, 0.0, 1.0), # 로봇에 더 가깝게 (X=0.6)
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )
    robot: ArticulationCfg = None


def _make_robot_cfg(usd_path: str) -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_SH5_CFG)
    robot_cfg.spawn.usd_path = usd_path
    robot_cfg.spawn.rigid_props.disable_gravity = not args_cli.enable_gravity
    robot_cfg.init_state.pos = cfg.ROBOT_POS
    return robot_cfg


# ========== OpenXR & IK Control Bridge ==========

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.device_base import DeviceBase
from isaaclab.utils.math import subtract_frame_transforms
import numpy as np
from omni.kit.xr.core import XRCore
from isaacsim.core.prims import SingleXFormPrim

class SH5OpenXRBridge:
    def __init__(self, robot, scene):
        self.robot = robot
        self.scene = scene
        
        print("[INFO] Initializing Direct XRCore Connection...")
        # 1. XRAnchor 이동 (VR 플레이어의 영점을 로봇 어깨 뒤로 이동)
        self.xr_anchor = SingleXFormPrim("/World/XRAnchor", position=np.array([-0.2, 0.0, 1.2]))
        
        # 2. 로봇의 양팔을 위한 Inverse Kinematics(IK) 컨트롤러 세팅
        print("[INFO] Setting up Differential IK Controllers for Left/Right arms...")
        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        
        self.left_ik = DifferentialIKController(ik_cfg, num_envs=1, device=robot.device)
        self.right_ik = DifferentialIKController(ik_cfg, num_envs=1, device=robot.device)
        
        # 3. 로봇 팔 엔드이펙터(손목/베이스) 씬 엔티티 구성
        self.left_arm_entity = SceneEntityCfg("robot", joint_names=["arm_l_joint.*"], body_names=["hx5_l_base"])
        self.right_arm_entity = SceneEntityCfg("robot", joint_names=["arm_r_joint.*"], body_names=["hx5_r_base"])
        
        self.left_arm_entity.resolve(scene)
        self.right_arm_entity.resolve(scene)
        
        # 고정 베이스 로봇이므로 body_ids에서 1을 빼서 자코비안 인덱스 매칭
        self.left_ee_idx = self.left_arm_entity.body_ids[0] - 1 if robot.is_fixed_base else self.left_arm_entity.body_ids[0]
        self.right_ee_idx = self.right_arm_entity.body_ids[0] - 1 if robot.is_fixed_base else self.right_arm_entity.body_ids[0]

        # 손가락 관절 이름 추출 (접착제 모드 제어용)
        self.left_finger_indices = [i for i, name in enumerate(robot.data.joint_names) if "finger_l" in name]
        self.right_finger_indices = [i for i, name in enumerate(robot.data.joint_names) if "finger_r" in name]
        self._debug_print_time = 0

    def apply_latest_targets(self):
        xr_core = XRCore.get_singleton()
        if not xr_core:
            return
            
        # VR 카메라(XRAnchor)가 로봇을 실시간으로 따라다니도록 설정
        root_pos = self.robot.data.root_pos_w[0].cpu().numpy()
        anchor_pos = np.array([root_pos[0] - 0.2, root_pos[1], root_pos[2] + 1.2])
        self.xr_anchor.set_world_pose(position=anchor_pos)
        
        position_target = self.robot.data.joint_pos_target.clone()
        
        # --- LEFT ARM IK ---
        left_dev = xr_core.get_input_device("/user/hand/left")
        if left_dev:
            pose = left_dev.get_virtual_world_pose()
            position = pose.ExtractTranslation()
            quat = pose.ExtractRotationQuat()
            
            if time.time() - self._debug_print_time > 2.0:
                print(f"[DEBUG XR] Left Controller World Position: {position}")
                
            target_pos_w = torch.tensor([[position[0], position[1], position[2]]], device=self.robot.device)
            target_quat_w = torch.tensor([[quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]]], device=self.robot.device)
            
            jacobian = self.robot.root_physx_view.get_jacobians()[:, self.left_ee_idx, :, self.left_arm_entity.joint_ids]
            ee_pose_w = self.robot.data.body_pose_w[:, self.left_arm_entity.body_ids[0]]
            root_pose_w = self.robot.data.root_pose_w
            joint_pos = self.robot.data.joint_pos[:, self.left_arm_entity.joint_ids]
            
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos_w, target_quat_w
            )
            # 왼쪽 팔 오프셋 적용
            target_pos_b[:, 0] += 0.5
            target_pos_b[:, 1] += 0.1
            target_pos_b[:, 2] -= 0.5
            
            self.left_ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
            joint_pos_des = self.left_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            position_target[:, self.left_arm_entity.joint_ids] = joint_pos_des
            
            trigger_val = 0.0
            if left_dev.has_input_gesture("trigger", "value"):
                trigger_val = float(left_dev.get_input_gesture_value("trigger", "value"))
            finger_target = 1.5 if trigger_val > 0.1 else 0.0
            for idx in self.left_finger_indices:
                position_target[:, idx] = finger_target

        # --- RIGHT ARM IK ---
        right_dev = xr_core.get_input_device("/user/hand/right")
        if right_dev:
            pose = right_dev.get_virtual_world_pose()
            position = pose.ExtractTranslation()
            quat = pose.ExtractRotationQuat()
            
            target_pos_w = torch.tensor([[position[0], position[1], position[2]]], device=self.robot.device)
            target_quat_w = torch.tensor([[quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2]]], device=self.robot.device)
            
            jacobian = self.robot.root_physx_view.get_jacobians()[:, self.right_ee_idx, :, self.right_arm_entity.joint_ids]
            ee_pose_w = self.robot.data.body_pose_w[:, self.right_arm_entity.body_ids[0]]
            root_pose_w = self.robot.data.root_pose_w
            joint_pos = self.robot.data.joint_pos[:, self.right_arm_entity.joint_ids]
            
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos_w, target_quat_w
            )
            # 오른쪽 팔 오프셋 적용
            target_pos_b[:, 0] += 0.5
            target_pos_b[:, 1] -= 0.1
            target_pos_b[:, 2] -= 0.5
            
            self.right_ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
            joint_pos_des = self.right_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            position_target[:, self.right_arm_entity.joint_ids] = joint_pos_des
            
            trigger_val = 0.0
            if right_dev.has_input_gesture("trigger", "value"):
                trigger_val = float(right_dev.get_input_gesture_value("trigger", "value"))
            finger_target = 1.5 if trigger_val > 0.1 else 0.0
            for idx in self.right_finger_indices:
                position_target[:, idx] = finger_target

        self.robot.set_joint_position_target(position_target)
        if time.time() - self._debug_print_time > 2.0:
            self._debug_print_time = time.time()

    def update_odometry(self, dt: float):
        pass # OpenXR 테스트 모드에서는 odometry 로직 생략

    def publish_joint_states(self):
        pass

    def publish_odometry(self):
        pass

    def publish_tf(self):
        pass

    def shutdown(self):
        print("[INFO] Shutting down OpenXR Bridge...")


# ========== Robot State ==========

def _swerve_modules() -> list[SwerveModule]:
    return [
        SwerveModule(
            steering_joint=steering_joint,
            wheel_joint=wheel_joint,
            x_offset=SH5_SWERVE_MODULE_X_OFFSETS[index],
            y_offset=SH5_SWERVE_MODULE_Y_OFFSETS[index],
            angle_offset=SH5_SWERVE_MODULE_ANGLE_OFFSETS[index],
            steering_limit_lower=cfg.AI_WORKER_SWERVE_STEERING_LIMIT_LOWER,
            steering_limit_upper=cfg.AI_WORKER_SWERVE_STEERING_LIMIT_UPPER,
            wheel_speed_limit_lower=cfg.AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
            wheel_speed_limit_upper=cfg.AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
        )
        for index, (steering_joint, wheel_joint) in enumerate(
            zip(SH5_SWERVE_STEERING_JOINTS, SH5_SWERVE_WHEEL_JOINTS)
        )
    ]


def _write_default_joint_state(robot):
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)


# ========== Camera View ==========

def _find_camera_prim_by_name(stage, prim_name: str):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name and prim.GetTypeName() == "Camera":
            return prim
    return None


def _ensure_camera_viewport_attrs(camera_prim):
    from pxr import Gf, Sdf

    coi_attr = camera_prim.GetProperty("omni:kit:centerOfInterest")
    if not coi_attr or not coi_attr.IsValid():
        coi_attr = camera_prim.CreateAttribute(
            "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
        )
    if coi_attr.Get() is None:
        coi_attr.Set(Gf.Vec3d(0.0, 0.0, -10.0))


def _position_window(window, width: int, height: int, x: int | None = None, y: int | None = None):
    for attr_name, value in (("width", width), ("height", height), ("position_x", x), ("position_y", y)):
        if value is None:
            continue
        try:
            setattr(window, attr_name, value)
        except Exception:
            pass
        try:
            frame = getattr(window, "frame", None)
            if frame is not None:
                setattr(frame, attr_name, value)
        except Exception:
            pass


def _set_viewport_camera(
    window_name: str,
    camera_path: str,
    width: int = 640,
    height: int = 480,
    x: int | None = None,
    y: int | None = None,
):
    try:
        from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name
        from pxr import Sdf

        viewport = get_viewport_from_window_name(window_name)
        if viewport is None:
            window = create_viewport_window(
                window_name,
                width=width,
                height=height,
                position_x=0 if x is None else x,
                position_y=0 if y is None else y,
                camera_path=Sdf.Path(camera_path),
            )
            cfg.AI_WORKER_CAMERA_VIEW_WINDOWS.append(window)
            _position_window(window, width, height, x, y)
            viewport = get_viewport_from_window_name(window_name)
        if viewport is not None:
            viewport.set_active_camera(camera_path)
            return True
    except Exception as exc:
        print(f"[WARN] Could not create viewport '{window_name}': {exc}")
    return False


def _setup_camera_views():
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()

    camera_specs = (
        ("Center Camera", cfg.AI_WORKER_CAMERA_CENTER_NAME, 780, 490, 50, 22),
        ("Left Camera", cfg.AI_WORKER_CAMERA_LEFT_NAME, 387, 280, 50, 517),
        ("Right Camera", cfg.AI_WORKER_CAMERA_RIGHT_NAME, 387, 280, 441, 517),
    )
    camera_paths: dict[str, str] = {}
    missing_camera_names: list[str] = []

    for window_name, camera_name, width, height, x, y in camera_specs:
        camera_prim = _find_camera_prim_by_name(stage, camera_name)
        if camera_prim is None:
            missing_camera_names.append(camera_name)
            continue
        _ensure_camera_viewport_attrs(camera_prim)
        camera_path = str(camera_prim.GetPath())
        camera_paths[camera_name] = camera_path
        _set_viewport_camera(window_name, camera_path, width=width, height=height, x=x, y=y)

    print("[INFO] Main Isaac Sim viewport left unchanged for overview/manual view.")
    for camera_name, camera_path in camera_paths.items():
        print(f"[INFO] {camera_name}: {camera_path}")
    if missing_camera_names:
        available_cameras = [
            str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "Camera"
        ]
        print(f"[WARN] Missing requested camera prims: {missing_camera_names}")
        print(f"[WARN] Available cameras: {available_cameras}")

    return camera_paths


# ROS 2 Camera Publisher (Vuer 비디오 스트리밍용) 코드는 하이브리드 모드에서 불필요하므로 제거됨.
# 이제 화면은 SteamVR/ALVR을 통해 다이렉트로 출력됩니다.
def _setup_ros2_camera_publishers(camera_paths: dict[str, str]):
    print("[INFO] 하이브리드 VR 모드: ROS 2 기반의 카메라 이미지 전송 기능이 비활성화 되었습니다. SteamVR을 사용하세요.")
    pass


# ========== Simulation Loop ==========

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, bridge: SH5OpenXRBridge):
    logger = VRDemonstrationLogger(output_dir="/home/rokey/dev_ws/datasets")
    import carb
    import carb.input
    import omni.appwindow

    def is_key_pressed(key_code):
        app_window = omni.appwindow.get_default_app_window()
        if not app_window: return False
        keyboard_device = app_window.get_keyboard()
        if not keyboard_device: return False
        input_interface = carb.input.acquire_input_interface()
        return input_interface.get_keyboard_value(keyboard_device, key_code) > 0.5

    sim_dt = sim.get_physics_dt()
    step_period = 1.0 / cfg.STEP_HZ if cfg.STEP_HZ > 0 else 0.0
    publish_period = 1.0 / cfg.PUBLISH_HZ if cfg.PUBLISH_HZ > 0 else 0.0
    last_publish = 0.0
    last_step = time.time()

    print("\n" + "="*70)
    print("🎥 [데이터 녹화 안내] 🎥")
    print("데이터 수집용 녹화(R키)가 안 된다면 현재 커서가 '터미널'에 있기 때문입니다!")
    print("반드시 마우스로 [Isaac Sim 3D 렌더링 화면]을 한번 클릭하여 활성화한 뒤에")
    print("R 키: 녹화 시작")
    print("T 키: 녹화 저장 및 종료 (성공 시)")
    print("C 키: 녹화 취소 (실패 시, 저장 안 함)")
    print("B 키: 상자 수동 리스폰 (버그로 끼었을 때 긴급 탈출)")
    print("V 키: 로봇 수동 리스폰 (로봇이 넘어졌을 때 제자리 복구)")
    print("="*70 + "\n")

    while simulation_app.is_running():
        if is_key_pressed(carb.input.KeyboardInput.R) and not logger.is_recording:
            logger.start_recording()
            time.sleep(0.5)
        elif is_key_pressed(carb.input.KeyboardInput.T) and logger.is_recording:
            logger.stop_recording_and_save()
            time.sleep(0.5)
        elif is_key_pressed(carb.input.KeyboardInput.C) and logger.is_recording:
            logger.cancel_recording()
            time.sleep(0.5)
        elif is_key_pressed(carb.input.KeyboardInput.B):
            if "box" in scene.keys():
                print("[INFO] 🔄 수동 리스폰(B키) 실행. 상자를 초기 위치로 되돌립니다.")
                default_state = scene["box"].data.default_root_state.clone()
                scene["box"].write_root_state_to_sim(default_state)
            time.sleep(0.5)
        elif is_key_pressed(carb.input.KeyboardInput.V):
            if "robot" in scene.keys():
                print("[INFO] 🔄 로봇 수동 리스폰(V키) 실행. 로봇을 초기 상태로 되돌립니다.")
                default_root_state = scene["robot"].data.default_root_state.clone()
                scene["robot"].write_root_state_to_sim(default_root_state)
                default_joint_pos = scene["robot"].data.default_joint_pos.clone()
                default_joint_vel = scene["robot"].data.default_joint_vel.clone()
                scene["robot"].write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            time.sleep(0.5)

        bridge.apply_latest_targets()
        
        if logger.is_recording:
            current_pos = scene["robot"].data.joint_pos.squeeze(0).detach().cpu().tolist()
            current_vel = scene["robot"].data.joint_vel.squeeze(0).detach().cpu().tolist()
            action_target = scene["robot"].data.joint_pos_target.squeeze(0).detach().cpu().tolist()
            logger.log_step(current_pos, current_vel, action_target)

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        bridge.update_odometry(sim_dt)

        # 박스 리스폰 로직 (바닥으로 떨어졌을 때 & 작업대에 안착되었을 때)
        if "box" in scene.keys():
            box_pos = scene["box"].data.root_pos_w
            box_vel = scene["box"].data.root_lin_vel_w
            if box_pos is not None:
                # 1. 바닥에 떨어졌을 때 (Z < 0.5)
                if box_pos[0, 2] < 0.5:
                    print("[INFO] 상자가 떨어졌습니다. 받침대 위로 리스폰합니다.")
                    default_state = scene["box"].data.default_root_state.clone()
                    scene["box"].write_root_state_to_sim(default_state)
                # 2. 작업대 쪽에 도달했고(X > 0.85, Y < -0.4) 로봇 손을 떠나 정지했을 때 (속도 거의 0)
                elif box_pos[0, 0] > 0.85 and box_pos[0, 1] < -0.4 and box_vel is not None and box_vel[0].norm() < 0.02:
                    print("[INFO] 🎉 상자가 작업대에 안착되었습니다! 새 상자를 리스폰합니다.")
                    default_state = scene["box"].data.default_root_state.clone()
                    scene["box"].write_root_state_to_sim(default_state)

        now = time.time()
        if publish_period == 0.0 or now - last_publish >= publish_period:
            bridge.publish_joint_states()
            bridge.publish_odometry()
            bridge.publish_tf()
            last_publish = now

        if step_period > 0.0:
            next_step = last_step + step_period
            sleep_time = next_step - time.time()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            last_step = next_step if sleep_time > 0.0 else time.time()


def main():
    usd_path = FFW_SH5_CFG.spawn.usd_path
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SH5 USD not found: {usd_path}")

    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",
        dt=1.0 / cfg.STEP_HZ,
        render_interval=cfg.RENDER_INTERVAL,
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = CoupangSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = _make_robot_cfg(usd_path).replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    _write_default_joint_state(robot)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    
    camera_paths = {}
    if args_cli.enable_camera_views or args_cli.enable_ros2_cameras:
        camera_paths = _setup_camera_views()
        
    if args_cli.enable_ros2_cameras:
        _setup_ros2_camera_publishers(camera_paths)

    # 이제 DDS 토픽 매니저 등을 생성할 필요 없이 OpenXR 브릿지만 생성합니다.
    bridge = SH5OpenXRBridge(robot=robot, scene=scene)

    print("[INFO] FFW SH5 Native OpenXR / SteamVR bringup ready.")
    if args_cli.enable_environment:
        print("[INFO] Environment: Simple Warehouse")
    print("[INFO] Controlling with SteamVR Controller (IK + Trigger)")

    try:
        run_simulator(sim, scene, bridge)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
