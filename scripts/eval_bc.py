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
import tty
import termios
import select
from copy import deepcopy
from pathlib import Path

from isaaclab.app import AppLauncher


ROBOTIS_LAB_DIR = Path("/home/rokey/dev_ws/robotis_lab/scripts/sim2real/bringup")
if str(ROBOTIS_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOTIS_LAB_DIR))

DEV_WS_DIR = Path("/home/rokey/dev_ws/robotis_lab/source/robotis_lab")
if str(DEV_WS_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_WS_DIR))

THIRD_PARTY_DIR = Path("/home/rokey/dev_ws/robotis_lab/third_party/robotis_dds_python")
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from common import robotis_config as cfg

# CLI and app launch
parser = argparse.ArgumentParser(description="FFW SH5 DDS bringup for Isaac Sim.")
parser.add_argument("--disable_head", action="store_true", help="Do not subscribe to the head topic.")
parser.add_argument("--disable_lift", action="store_true", help="Do not subscribe to the lift topic.")
parser.add_argument("--disable_cmd_vel", action="store_true", help="Do not subscribe to cmd_vel for the swerve base.")
parser.add_argument("--domain_id", type=int, default=None, help="DDS domain id. Defaults to ROS_DOMAIN_ID or 0.")
parser.add_argument("--disable_gravity", action="store_true", help="Disable gravity on the SH5 rigid bodies.")
parser.add_argument("--enable_environment", action="store_true", help="Spawn the environment USD.")
parser.add_argument("--robot_ip", type=str, default="127.0.0.1", help="Robot IP address")
parser.add_argument("--model", type=str, default=None, help="Path to trained policy .pth file for Inference mode")
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

import torch
import numpy as np
import collections
sys.path.append("/home/rokey/dev_ws/coupang_ws/scripts")
from train_bc import BehaviorCloningPolicy
from train_act import ACTPolicy, STATE_DIM as ACT_STATE_DIM, ACTION_DIM as ACT_ACTION_DIM

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
            "obs/robot_pose": [], # 로봇 본체의 위치+회전 (월드 기준)
            "obs/joint_positions": [],
            "obs/joint_velocities": [],
            "obs/box_pose": [],  # 상자 위치+회전 정보 추가
            "obs/rack_pose": [], # 작업대 위치+회전 정보 추가
            "actions/joint_targets": [],
            "actions/cmd_vel": [], # 이동 대차(모바일 베이스) 조작 명령 추가
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

    def log_step(self, robot_pose, joint_pos, joint_vel, action_target, cmd_vel, box_pose, rack_pose, reward=0.0, done=False):
        if not self.is_recording:
            return
        self.buffer["obs/robot_pose"].append(np.array(robot_pose, dtype=np.float32))
        self.buffer["obs/joint_positions"].append(np.array(joint_pos, dtype=np.float32))
        self.buffer["obs/joint_velocities"].append(np.array(joint_vel, dtype=np.float32))
        self.buffer["obs/box_pose"].append(np.array(box_pose, dtype=np.float32))
        self.buffer["obs/rack_pose"].append(np.array(rack_pose, dtype=np.float32))
        self.buffer["actions/joint_targets"].append(np.array(action_target, dtype=np.float32))
        self.buffer["actions/cmd_vel"].append(np.array(cmd_vel, dtype=np.float32))
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
            obs_grp.create_dataset("robot_pose", data=np.array(self.buffer["obs/robot_pose"]))
            obs_grp.create_dataset("joint_positions", data=np.array(self.buffer["obs/joint_positions"]))
            obs_grp.create_dataset("joint_velocities", data=np.array(self.buffer["obs/joint_velocities"]))
            obs_grp.create_dataset("box_pose", data=np.array(self.buffer["obs/box_pose"]))
            obs_grp.create_dataset("rack_pose", data=np.array(self.buffer["obs/rack_pose"]))
            ep_grp.create_dataset("actions", data=np.array(self.buffer["actions/joint_targets"]))
            ep_grp.create_dataset("cmd_vel", data=np.array(self.buffer["actions/cmd_vel"]))
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
            pos=(0.0, -1.5, 0.0),  # 로봇 정면 우측 (Y=-0.8m)
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=2.0,          # 공중에서 움직일 때 안정성 증가
                angular_damping=20.0,        # ★핵심: 회전(비틀림) 저항을 극대화하여 손가락 사이에서 헛도는 현상 방지
                max_depenetration_velocity=1.0, # 튕겨나가는 현상 방지
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05), # 핸드 트래킹 진동에 튕겨나가지 않도록 깃털 수준(1g)으로 질량 축소
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.05, rest_offset=0.0), # 밀착 접촉을 위해 offset 최소화
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
    robot_cfg.spawn.rigid_props.disable_gravity = args_cli.disable_gravity
    robot_cfg.init_state.pos = cfg.ROBOT_POS
    return robot_cfg


# ========== DDS Topic Parsing and Matching ==========

def _trajectory_qos() -> Qos:
    return Qos(
        Policy.Reliability.BestEffort,
        Policy.Durability.Volatile,
        Policy.History.KeepLast(10),
    )


def _now_stamp() -> Time_:
    now_ns = time.time_ns()
    return Time_(sec=now_ns // 1_000_000_000, nanosec=now_ns % 1_000_000_000)


def _enabled_topics() -> dict[str, str]:
    topics = {
        "right_arm": cfg.AI_WORKER_RIGHT_ARM_TOPIC,
        "right_hand": cfg.SH5_RIGHT_HAND_TOPIC,
        "left_arm": cfg.AI_WORKER_LEFT_ARM_TOPIC,
        "left_hand": cfg.SH5_LEFT_HAND_TOPIC,
    }
    if not args_cli.disable_head:
        topics["head"] = cfg.HEAD_TOPIC
    if not args_cli.disable_lift:
        topics["lift"] = cfg.LIFT_TOPIC
    return topics


class SH5DdsBridge:
    def __init__(
        self,
        robot,
        topic_manager: TopicManager,
        topic_names: dict[str, str],
        joint_states_topic: str,
        odom_topic: str,
        tf_topic: str,
        base_frame: str,
        odom_frame: str,
        trajectory_qos: Qos,
        cmd_vel_topic: str | None,
        swerve_modules: list[SwerveModule],
        wheel_radius: float,
        cmd_vel_timeout: float,
        inference_mode: bool = False,
    ):
        self.inference_mode = inference_mode
        self.robot = robot
        self.base_frame = base_frame
        self.odom_frame = odom_frame
        self.swerve_modules = swerve_modules
        self.wheel_radius = wheel_radius
        self.cmd_vel_timeout = cmd_vel_timeout
        self.swerve_controller = (
            SwerveDriveController(swerve_modules, wheel_radius) if swerve_modules else None
        )
        self.odometry = (
            SwerveOdometry(
                [module.x_offset for module in swerve_modules],
                [module.y_offset for module in swerve_modules],
                wheel_radius,
            )
            if swerve_modules
            else None
        )
        self._last_swerve_update_time = time.monotonic()
        self.running = True
        self.lock = threading.Lock()
        self.pending_positions: dict[str, float] = {}
        self.latest_cmd_vel = (0.0, 0.0, 0.0)
        self.last_cmd_vel_time = 0.0
        self.unknown_joints: set[str] = set()
        self._warned_missing_base_frame = False
        self._warned_missing_swerve_joints: set[str] = set()
        self._body_names = list(self.robot.data.body_names)
        self._base_id = (
            self._body_names.index(self.base_frame) if self.base_frame in self._body_names else None
        )
        self._joint_name_to_index = {
            name: index for index, name in enumerate(self.robot.data.joint_names)
        }
        self._missing_swerve_joints = [
            joint_name
            for module in self.swerve_modules
            for joint_name in (module.steering_joint, module.wheel_joint)
            if joint_name not in self._joint_name_to_index
        ]
        self._swerve_steering_joint_ids = [
            self._joint_name_to_index[module.steering_joint]
            for module in self.swerve_modules
            if module.steering_joint in self._joint_name_to_index
        ]
        self._swerve_wheel_joint_ids = [
            self._joint_name_to_index[module.wheel_joint]
            for module in self.swerve_modules
            if module.wheel_joint in self._joint_name_to_index
        ]
        self.readers = []
        self.threads = []
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name=joint_states_topic,
            topic_type=JointState_,
        )
        self.odom_writer = topic_manager.topic_writer(
            topic_name=odom_topic,
            topic_type=Odometry_,
        )
        self.tf_writer = topic_manager.topic_writer(
            topic_name=tf_topic,
            topic_type=TFMessage_,
        )

        for label, topic_name in topic_names.items():
            if not topic_name:
                continue
            reader = topic_manager.topic_reader(topic_name=topic_name, topic_type=JointTrajectory_, qos=trajectory_qos)
            thread = threading.Thread(
                target=self._trajectory_loop,
                args=(label, reader),
                daemon=True,
            )
            self.readers.append(reader)
            self.threads.append(thread)
            thread.start()
            print(f"[DDS] Subscribing {label}: {topic_name}")

        if cmd_vel_topic:
            cmd_vel_reader = topic_manager.topic_reader(
                topic_name=cmd_vel_topic,
                topic_type=Twist_,
                qos=trajectory_qos,
            )
            cmd_vel_thread = threading.Thread(target=self._cmd_vel_loop, args=(cmd_vel_reader,), daemon=True)
            self.readers.append(cmd_vel_reader)
            self.threads.append(cmd_vel_thread)
            cmd_vel_thread.start()
            print(f"[DDS] Subscribing cmd_vel: {cmd_vel_topic}")

    # Run DDS reader loops
    def _trajectory_loop(self, label: str, reader):
        try:
            while self.running:
                if self.inference_mode:
                    time.sleep(0.1)
                    continue
                for msg in reader.take_iter():
                    self._store_trajectory(label, msg)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] {label} subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    def _cmd_vel_loop(self, reader):
        try:
            while self.running:
                if self.inference_mode:
                    time.sleep(0.1)
                    continue
                for msg in reader.take_iter():
                    self._store_cmd_vel(msg)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] cmd_vel subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    # Parse trajectory topics and match joints
    def _store_trajectory(self, label: str, msg):
        if msg is None or not msg.points:
            return

        point = msg.points[-1]
        joint_names = list(msg.joint_names)
        positions = list(point.positions)

        if label == "lift":
            lift_position = None
            if cfg.LIFT_JOINT_NAME in joint_names:
                lift_position = (
                    cfg.LIFT_POSITION_SCALE
                    * positions[joint_names.index(cfg.LIFT_JOINT_NAME)]
                )
            elif len(positions) == 1:
                lift_position = cfg.LIFT_POSITION_SCALE * positions[0]
            if lift_position is None:
                print(
                    f"[DDS] Ignoring lift message: '{cfg.LIFT_JOINT_NAME}' "
                    f"not found in joint_names={joint_names}"
                )
                return
            joint_names = [cfg.LIFT_JOINT_NAME]
            positions = [lift_position]

        if len(joint_names) != len(positions):
            print(
                f"[DDS] Ignoring {label} message: joint_names={len(joint_names)} "
                f"positions={len(positions)}"
            )
            return

        with self.lock:
            self.pending_positions.update(dict(zip(joint_names, positions)))

    def clear_pending_targets(self):
        with self.lock:
            self.pending_positions.clear()
            self.latest_cmd_vel = (0.0, 0.0, 0.0)

    # Apply swerve drive mobile base command
    def _store_cmd_vel(self, msg):
        if msg is None:
            return
        with self.lock:
            self.latest_cmd_vel = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self.last_cmd_vel_time = time.monotonic()

    def _current_cmd_vel(self) -> tuple[float, float, float]:
        with self.lock:
            command = self.latest_cmd_vel
            last_msg_time = self.last_cmd_vel_time

        if last_msg_time == 0.0:
            return 0.0, 0.0, 0.0
        if self.cmd_vel_timeout > 0.0 and time.monotonic() - last_msg_time > self.cmd_vel_timeout:
            return 0.0, 0.0, 0.0
        return command

    def apply_latest_targets(self):
        with self.lock:
            commands = dict(self.pending_positions)

        position_target = self.robot.data.joint_pos_target.clone()
        velocity_target = self.robot.data.joint_vel_target.clone()

        for name, position in commands.items():
            joint_id = self._joint_name_to_index.get(name)
            if joint_id is None:
                if name not in self.unknown_joints:
                    self.unknown_joints.add(name)
                    print(f"[DDS] Joint '{name}' is not in the SH5 USD articulation; ignoring it.")
                continue
            position_target[:, joint_id] = float(position)

        self._apply_swerve_targets(position_target, velocity_target)

        self.robot.set_joint_position_target(position_target)
        self.robot.set_joint_velocity_target(velocity_target)

    def _apply_swerve_targets(self, position_target, velocity_target):
        if not self.swerve_modules:
            return

        for joint_name in self._missing_swerve_joints:
            if joint_name not in self._warned_missing_swerve_joints:
                self._warned_missing_swerve_joints.add(joint_name)
                print(f"[DDS] Swerve joint '{joint_name}' is not in the SH5 USD articulation; ignoring cmd_vel.")
        if self._missing_swerve_joints:
            return

        current_steering = [
            float(value)
            for value in self.robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist()
        ]
        current_wheel_velocities = [
            float(value)
            for value in self.robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]
        linear_x, linear_y, angular_z = self._current_cmd_vel()
        now = time.monotonic()
        dt = now - self._last_swerve_update_time
        self._last_swerve_update_time = now

        if self.swerve_controller is None:
            return
        module_commands = self.swerve_controller.compute_commands(
            linear_x,
            linear_y,
            angular_z,
            current_steering_positions=current_steering,
            current_wheel_velocities=current_wheel_velocities,
            dt=dt,
        )
        for module_command, steering_id, wheel_id in zip(
            module_commands,
            self._swerve_steering_joint_ids,
            self._swerve_wheel_joint_ids,
        ):
            position_target[:, steering_id] = module_command.steering_position
            velocity_target[:, wheel_id] = module_command.wheel_velocity

    def update_odometry(self, dt: float):
        if self.odometry is None or not self.swerve_modules or self._missing_swerve_joints:
            return

        steering_positions = [
            float(value) + module.angle_offset
            for value, module in zip(
                self.robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist(),
                self.swerve_modules,
            )
        ]
        wheel_velocities = [
            float(value)
            for value in self.robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]
        self.odometry.update(steering_positions, wheel_velocities, dt)

    # Publish robot state and close DDS resources
    def publish_joint_states(self):
        stamp = _now_stamp()
        header = Header_(stamp=stamp, frame_id="base_link")

        joint_names = list(self.robot.data.joint_names)
        positions = self.robot.data.joint_pos.squeeze(0).detach().cpu().tolist()
        velocities = self.robot.data.joint_vel.squeeze(0).detach().cpu().tolist()
        efforts = [0.0] * len(joint_names)

        msg = JointState_(
            header=header,
            name=joint_names,
            position=positions,
            velocity=velocities,
            effort=efforts,
        )
        try:
            self.joint_state_writer.write(msg)
        except Exception as exc:
            print(f"[DDS] joint_states write error: {exc}")

    def publish_odometry(self):
        if self.odometry is None:
            return

        state = self.odometry.state()
        quat_x, quat_y, quat_z, quat_w = yaw_to_quaternion(state.yaw)
        covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            covariance[index] = 0.001

        stamp = _now_stamp()
        msg = Odometry_(
            header=Header_(stamp=stamp, frame_id=self.odom_frame),
            child_frame_id=self.base_frame,
            pose=PoseWithCovariance_(
                pose=Pose_(
                    position=Point_(x=state.x, y=state.y, z=0.0),
                    orientation=Quaternion_(x=quat_x, y=quat_y, z=quat_z, w=quat_w),
                ),
                covariance=covariance,
            ),
            twist=TwistWithCovariance_(
                twist=Twist_(
                    linear=Vector3_(x=state.vx, y=state.vy, z=0.0),
                    angular=Vector3_(x=0.0, y=0.0, z=state.wz),
                ),
                covariance=covariance,
            ),
        )
        try:
            self.odom_writer.write(msg)
        except Exception as exc:
            print(f"[DDS] odom write error: {exc}")

    def publish_tf(self):
        if self._base_id is None:
            if not self._warned_missing_base_frame:
                self._warned_missing_base_frame = True
                print(
                    f"[DDS] Cannot publish TF: base frame '{self.base_frame}' is not in SH5 body names. "
                    f"Available bodies: {self._body_names}"
                )
            return

        stamp = _now_stamp()
        body_pose_w = self.robot.data.body_link_state_w[0, :, :7]
        base_pose_w = body_pose_w[self._base_id]
        base_pos_w = base_pose_w[:3].unsqueeze(0)
        base_quat_w = base_pose_w[3:7].unsqueeze(0)

        transforms = []
        for body_id, child_frame in enumerate(self._body_names):
            if child_frame == self.base_frame:
                continue

            child_pose_w = body_pose_w[body_id]
            child_pos_b, child_quat_b = math_utils.subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                child_pose_w[:3].unsqueeze(0),
                child_pose_w[3:7].unsqueeze(0),
            )
            pos = child_pos_b.squeeze(0).detach().cpu().tolist()
            quat_wxyz = child_quat_b.squeeze(0).detach().cpu().tolist()

            transforms.append(
                TransformStamped_(
                    header=Header_(stamp=stamp, frame_id=self.base_frame),
                    child_frame_id=child_frame,
                    transform=Transform_(
                        translation=Vector3_(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                        rotation=Quaternion_(
                            x=float(quat_wxyz[1]),
                            y=float(quat_wxyz[2]),
                            z=float(quat_wxyz[3]),
                            w=float(quat_wxyz[0]),
                        ),
                    ),
                )
            )

        self.tf_writer.write(TFMessage_(transforms=transforms))

    def shutdown(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
        for reader in self.readers:
            try:
                reader.Close()
            except Exception:
                pass
        try:
            self.joint_state_writer.Close()
        except Exception:
            pass
        try:
            self.odom_writer.Close()
        except Exception:
            pass
        try:
            self.tf_writer.Close()
        except Exception:
            pass


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


def _setup_ros2_camera_publishers(camera_paths: dict[str, str]):
    import omni.graph.core as og
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.isaac.ros2_bridge")

    topic_mapping = {
        "Left Camera": "/leader/left_camera/image_raw",
        "Right Camera": "/leader/right_camera/image_raw",
    }

    for camera_name, topic_name in topic_mapping.items():
        if camera_name not in camera_paths:
            continue
        
        camera_path = camera_paths[camera_name]
        graph_path = "/World/ActionGraph_" + topic_name.replace("/", "_")
        
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnPlaybackTick"),
                    ("CreateRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                    ("ROS2CameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                    ("CreateRenderProduct.outputs:execOut", "ROS2CameraHelper.inputs:execIn"),
                    ("CreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelper.inputs:renderProductPath"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("ROS2CameraHelper.inputs:topicName", topic_name),
                    ("ROS2CameraHelper.inputs:type", "rgb"),
                ],
            },
        )
        # Target syntax is tricky, let's use standard property setting
        og.Controller.attribute(graph_path + "/CreateRenderProduct.inputs:cameraPrim").set([og.SubGraph.Target(camera_path)])
        print(f"[INFO] ROS 2 Camera Publisher created for {camera_name} on {topic_name}")


class TerminalKeyboard:
    def __init__(self):
        self.key_pressed = None
        self.running = True
        self.old_settings = termios.tcgetattr(sys.stdin)
        import atexit
        atexit.register(self.restore_terminal)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def restore_terminal(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def _read_loop(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char = sys.stdin.read(1)
                    if char:
                        self.key_pressed = char.lower()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def get_key_and_clear(self):
        k = self.key_pressed
        self.key_pressed = None
        return k

# ========== Simulation Loop ==========

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, bridge: SH5DdsBridge, model=None, device=None, act_config=None):
    logger = VRDemonstrationLogger(output_dir="/home/rokey/dev_ws/datasets")
    term_kbd = TerminalKeyboard()
    
    sim_dt = sim.get_physics_dt()
    step_period = 1.0 / cfg.STEP_HZ if cfg.STEP_HZ > 0 else 0.0
    publish_period = 1.0 / cfg.PUBLISH_HZ if cfg.PUBLISH_HZ > 0 else 0.0
    last_publish = 0.0
    last_step = time.time()
    
    # AI 행동 부드럽게 만들기 위한 필터 변수
    smoothed_joint_targets = None
    smoothed_cmd_vel = None
    _monitor_step = 0   # 모니터링 출력 카운터
    
    # 관절 인덱스 (모니터링용)
    _ARM_R = [9, 12, 14, 16, 18, 20, 22]
    _ARM_L = [8, 11, 13, 15, 17, 19, 21]
    _LIFT  = 3
    _PHASE_NAMES = ["이동전", "파지", "리프트", "이동", "삽입/해제"]

    print("\n" + "="*70)
    print("🎥 [데이터 녹화 안내 (터미널 전용)] 🎥")
    print("이제 터미널 창을 클릭한 상태에서 키보드를 누르면 즉시 작동합니다!")
    print("r 키: 녹화 시작")
    print("t 키: 녹화 저장 및 종료 (성공 시)")
    print("c 키: 녹화 취소 (실패 시, 저장 안 함)")
    print("b 키: 상자 수동 리스폰 (버그로 끼었을 때 긴급 탈출)")
    print("v 키: 로봇 수동 리스폰 (로봇이 넘어졌을 때 제자리 복구)")
    print("="*70 + "\n")

    while simulation_app.is_running():
        key = term_kbd.get_key_and_clear()
        
        if key == 'r' and not logger.is_recording:
            logger.start_recording()
        elif key == 't' and logger.is_recording:
            logger.stop_recording_and_save()
        elif key == 'c' and logger.is_recording:
            logger.cancel_recording()
        elif key == 'b':
            if "box" in scene.keys():
                print("[INFO] 🔄 수동 리스폰(B키) 실행. 상자를 초기 위치로 되돌립니다.")
                default_state = scene["box"].data.default_root_state.clone()
                scene["box"].write_root_state_to_sim(default_state)
        elif key == 'v':
            if "robot" in scene.keys():
                print("[INFO] 🔄 로봇 수동 리스폰(V키) 실행. 로봇을 초기 상태로 되돌립니다.")
                default_root_state = scene["robot"].data.default_root_state.clone()
                scene["robot"].write_root_state_to_sim(default_root_state)
                default_joint_pos = scene["robot"].data.default_joint_pos.clone()
                default_joint_vel = scene["robot"].data.default_joint_vel.clone()
                scene["robot"].write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                scene["robot"].set_joint_position_target(default_joint_pos) # PD 제어기 타겟도 초기화
                bridge.clear_pending_targets() # 브릿지에 남아있는 이전 통신 데이터(높이 등) 삭제
                time.sleep(0.5)

        # 현재 상태를 항상 추출 (AI 제어 또는 데이터 로깅에 모두 사용)
        current_pos = scene["robot"].data.joint_pos.squeeze(0).detach().cpu().tolist()
        current_vel = scene["robot"].data.joint_vel.squeeze(0).detach().cpu().tolist()
        
        box_pose = [0.0] * 7
        if "box" in scene.keys() and scene["box"].data.root_state_w is not None:
            box_pose = scene["box"].data.root_state_w[0, :7].detach().cpu().tolist()
            
        rack_pose = [0.0] * 7
        if "rack" in scene.keys():
            rack_p, rack_q = scene["rack"].get_world_poses()
            rack_pose = rack_p[0].detach().cpu().tolist() + rack_q[0].detach().cpu().tolist()
            
        robot_pose = [0.0] * 7
        if "robot" in scene.keys() and scene["robot"].data.root_state_w is not None:
            robot_pose = scene["robot"].data.root_state_w[0, :7].detach().cpu().tolist()

        # 진행률 (Progress) 계산을 위한 스텝 카운터
        if not hasattr(bridge, "step_count"):
            bridge.step_count = 0
        
        # 평균 에피소드 길이(대략 850스텝)를 기준으로 0~100 사이의 진행률 생성 (스케일 증폭)
        progress_val = min(1.0, bridge.step_count / 850.0)
        progress = np.array([progress_val * 100.0], dtype=np.float32)
        bridge.step_count += 1
        
        # ==========================================
        # 실시간 단계 감지 (train_bc.py의 5단계와 동일)
        # 0=approach, 1=grasp, 2=lift_up, 3=transport, 4=place
        # ==========================================
        NUM_PHASES = 5
        if not hasattr(bridge, "current_phase"):
            bridge.current_phase = 0
            bridge.prev_lift = 0.0
            bridge.phase_history = []
        
        # 핵심 신호 추출
        finger_r_avg = np.mean(current_pos[40:60])
        lift_val = current_pos[62] if len(current_pos) > 62 else 0.0
        
        # 단계 전이 로직 (순방향만 전이, 뒤로 안 돌아감)
        if bridge.current_phase == 0:  # approach
            if finger_r_avg > 0.25:   # 손가락이 닫히기 시작하면
                bridge.current_phase = 1
        elif bridge.current_phase == 1:  # grasp
            if lift_val > bridge.prev_lift + 0.02:  # lift가 올라가기 시작하면
                bridge.current_phase = 2
        elif bridge.current_phase == 2:  # lift_up
            if hasattr(bridge, 'latest_cmd_vel') and bridge.latest_cmd_vel is not None:
                vx, vy, wz = bridge.latest_cmd_vel if isinstance(bridge.latest_cmd_vel, tuple) else (0,0,0)
                if abs(vx) > 0.01 or abs(wz) > 0.01:  # 이동이 시작되면
                    bridge.current_phase = 3
        elif bridge.current_phase == 3:  # transport
            if finger_r_avg < 0.15:   # 손가락이 펴지면
                bridge.current_phase = 4
        # phase 4 (place)는 에피소드 끝까지 유지
        
        bridge.prev_lift = lift_val
        
        # one-hot 벡터 생성
        phase_onehot = np.zeros(NUM_PHASES, dtype=np.float32)
        phase_onehot[bridge.current_phase] = 1.0
        
        # 인공지능 추론 모드
        if model is not None:
            state = np.concatenate([
                robot_pose, box_pose, rack_pose, current_pos, current_vel, progress, phase_onehot
            ])
            
            # ---- ACT 모델: 시퀀스 입력 + Temporal Ensembling ----
            if hasattr(model, 'chunk_size'):
                # 상태 히스토리 버퍼 초기화
                if not hasattr(bridge, 'state_history'):
                    _ctx = act_config['context_len'] if act_config else 10
                    _chk = act_config['chunk_size'] if act_config else model.chunk_size
                    bridge.state_history = collections.deque(maxlen=_ctx)
                    bridge.action_queue = []  # 남은 액션 청크
                    bridge.temporal_weights = np.exp(-0.01 * np.arange(_chk))  # 감쇠 가중치
                    bridge._act_context_len = _ctx
                
                bridge.state_history.append(state)
                
                # 충분한 히스토리가 쌍이면 새 청크 예측
                if len(bridge.state_history) >= bridge._act_context_len:
                    # 남은 액션이 없거나 적으면 새로 예측
                    if len(bridge.action_queue) == 0:
                        state_seq = np.array(list(bridge.state_history), dtype=np.float32)
                        state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            predicted = model(state_tensor)  # (1, chunk_size, action_dim)
                        
                        actions = predicted.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
                        bridge.action_queue = list(actions)
                    
                    # 큐에서 다음 액션 가져오기
                    action_tensor = bridge.action_queue.pop(0)
                else:
                    # 히스토리 부족: 현재 위치 유지
                    action_tensor = np.concatenate([current_pos, [0.0, 0.0, 0.0]])
            else:
                # ---- BC 모델: 단일 프레임 입력 ----
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_tensor = model(state_tensor).squeeze(0).cpu().numpy()
                
            raw_joint_targets = action_tensor[:63]
            raw_cmd_vel = action_tensor[63:]
            raw_cmd_vel = np.clip(raw_cmd_vel, -0.5, 0.5)

            alpha_joint = 0.5
            alpha_vel   = 0.3

            if smoothed_joint_targets is None:
                smoothed_joint_targets   = np.array(current_pos)
                smoothed_cmd_vel         = np.array([0.0, 0.0, 0.0])
                _phase3_start_time       = None
                _phase3_rack_dist_min    = 9999.0

            smoothed_joint_targets = (1-alpha_joint)*smoothed_joint_targets + alpha_joint*raw_joint_targets
            smoothed_cmd_vel       = (1-alpha_vel)*smoothed_cmd_vel + alpha_vel*raw_cmd_vel

            max_delta = 0.05
            smoothed_joint_targets = np.clip(
                smoothed_joint_targets,
                np.array(current_pos) - max_delta,
                np.array(current_pos) + max_delta
            )

            current_phase_val = getattr(bridge, 'current_phase', 0)

            # ── 오버라이드 1: 리프트 강제 하강 ─────────────────────────────
            # phase 0~1(접근·파지)에서 상자보다 손이 높으면 lift를 내림
            _LIFT_IDX = 3
            box_z        = box_pose[2]
            rob_z        = robot_pose[2]
            lift_current = current_pos[_LIFT_IDX]
            if current_phase_val in (0, 1):
                target_lift  = np.clip(box_z - rob_z - 1.1, -0.48, 0.0)
                if lift_current > target_lift + 0.05:
                    smoothed_joint_targets[_LIFT_IDX] = max(lift_current - 0.008, target_lift)

            # ── 오버라이드 1b: 파지 무한루프 탈출 ──────────────────────────
            # Phase1(파지)에서 4초 이상 경과하고 상자가 올라오지 않으면 → 강제 탈출
            if not hasattr(bridge, '_phase1_start_time'):
                bridge._phase1_start_time = None
                bridge._phase1_box_z_init = None
                bridge._grasp_forced      = False

            if current_phase_val == 1:
                if bridge._phase1_start_time is None:
                    bridge._phase1_start_time = time.time()
                    bridge._phase1_box_z_init = box_z
                    bridge._grasp_forced      = False
                phase1_elapsed = time.time() - bridge._phase1_start_time
                box_lifted = box_z - (bridge._phase1_box_z_init or box_z)

                # [수정C] 파지 성공 판단: 상자 상승 OR 손가락 실제로 닫혔는가
                _FINGER_R_IDX = [28,29,30,31,32]  # finger_r 첫 5개
                _FINGER_L_IDX = [23,24,25,26,27]  # finger_l 첫 5개
                _slot_now_c = (act_config.get('slot_id', 1) if act_config else 1)
                _active_finger_idx = _FINGER_R_IDX if _slot_now_c in (1,3) else _FINGER_L_IDX
                finger_avg = np.mean([current_pos[i] for i in _active_finger_idx])
                finger_closed = finger_avg > 0.3  # 손가락 평균 0.3rad 이상이면 닫힘 판정

                grasp_success = (box_lifted > 0.05) or finger_closed

                if phase1_elapsed > 4.0 and not grasp_success and not bridge._grasp_forced:
                    bridge._grasp_forced = True
                    print(f"\n[OVERRIDE-1b] 파지 {phase1_elapsed:.1f}s 초과 | "
                          f"상자상승={box_lifted:.3f}m, 손가락={finger_avg:.3f}rad -> 강제 리프트 상승")
                if bridge._grasp_forced:
                    smoothed_joint_targets[_LIFT_IDX] = min(lift_current + 0.01, 0.0)
            else:
                bridge._phase1_start_time = None
                bridge._grasp_forced      = False

            # ── 오버라이드 2: 비활성 팔 대기 자세 (벨트 충돌 방지) ────────
            _slot_now = (act_config.get('slot_id', 1) if act_config else 1)
            _IDX_INACTIVE = _ARM_L if _slot_now in (1, 3) else _ARM_R
            _REST_VALS    = [0.0, -1.2, -1.5, 0.0, 0.5, 0.0, 0.0]
            _REST_ALPHA   = 0.02
            for j_i, j_idx in enumerate(_IDX_INACTIVE):
                smoothed_joint_targets[j_idx] = (
                    (1-_REST_ALPHA)*smoothed_joint_targets[j_idx] + _REST_ALPHA*_REST_VALS[j_i]
                )

            # ── 오버라이드 3: Phase3 회전 루프 탈출 ─────────────────────────
            # 3초 이상 Phase3이고 랙에 가까워지지 않으면 강제 전진
            if current_phase_val == 3:
                if _phase3_start_time is None:
                    _phase3_start_time    = time.time()
                    _phase3_rack_dist_min = 9999.0
                rack_xy   = np.array(rack_pose[:2])
                rob_xy    = np.array(robot_pose[:2])
                rack_dist = float(np.linalg.norm(rack_xy - rob_xy))
                _phase3_rack_dist_min = min(_phase3_rack_dist_min, rack_dist)
                if time.time() - _phase3_start_time > 3.0 and rack_dist > _phase3_rack_dist_min + 0.1:
                    rack_dir = rack_xy - rob_xy
                    if np.linalg.norm(rack_dir) > 0.01:
                        rack_dir = rack_dir / np.linalg.norm(rack_dir)
                    smoothed_cmd_vel[0] = float(np.clip(rack_dir[0]*0.15, -0.2, 0.2))
                    smoothed_cmd_vel[1] = float(np.clip(rack_dir[1]*0.15, -0.2, 0.2))
                    smoothed_cmd_vel[2] = 0.0
                    print(f"\n[OVERRIDE-3] Phase3 루프 탈출 → 랙 강제 전진 dist={rack_dist:.2f}m")
            else:
                _phase3_start_time    = None
                _phase3_rack_dist_min = 9999.0

            # ── 명령 적용 ────────────────────────────────────────────────────
            joint_names = scene["robot"].data.joint_names
            commands    = {name: float(smoothed_joint_targets[i]) for i, name in enumerate(joint_names)}

            with bridge.lock:
                bridge.pending_positions.update(commands)
                bridge.latest_cmd_vel = (
                    float(smoothed_cmd_vel[0]),
                    float(smoothed_cmd_vel[1]),
                    float(smoothed_cmd_vel[2])
                )
                bridge.last_cmd_vel_time = time.monotonic()
            
            # ================================================================
            # 실시간 모니터링 출력 (10스텝마다)
            # ================================================================
            _monitor_step += 1
            if _monitor_step % 10 == 0:
                phase_name = _PHASE_NAMES[bridge.current_phase] if hasattr(bridge, 'current_phase') else "?"
                arm_r_vals = [smoothed_joint_targets[i] for i in _ARM_R]
                arm_l_vals = [smoothed_joint_targets[i] for i in _ARM_L]
                lift_val_disp = smoothed_joint_targets[_LIFT]
                
                # 상자와 로봇 거리 계산
                box_xy  = np.array(box_pose[:2])
                rob_xy  = np.array(robot_pose[:2])
                dist    = float(np.linalg.norm(box_xy - rob_xy))
                
                # 행동 큐 상태
                q_len = len(bridge.action_queue) if hasattr(bridge, 'action_queue') else 0
                h_len = len(bridge.state_history) if hasattr(bridge, 'state_history') else 0
                
                print(f"\r[MONITOR] "
                      f"Phase: {bridge.current_phase}({phase_name}) | "
                      f"Lift: {lift_val_disp:+.3f} | "
                      f"ArmR: [{', '.join(f'{v:+.2f}' for v in arm_r_vals[:3])}...] | "
                      f"ArmL: [{', '.join(f'{v:+.2f}' for v in arm_l_vals[:3])}...] | "
                      f"Vel: vx={smoothed_cmd_vel[0]:+.3f} vy={smoothed_cmd_vel[1]:+.3f} wz={smoothed_cmd_vel[2]:+.3f} | "
                      f"BoxDist: {dist:.2f}m | "
                      f"Queue: {q_len}/{h_len}",
                      end="", flush=True)

        bridge.apply_latest_targets()
        
        if logger.is_recording:
            action_target = scene["robot"].data.joint_pos_target.squeeze(0).detach().cpu().tolist()
            # 작업대(Rack) 위치 정보 로깅
            if "rack" in scene.keys():
                rack_p, rack_q = scene["rack"].get_world_poses()
                rack_pose = rack_p[0].detach().cpu().tolist() + rack_q[0].detach().cpu().tolist()
            else:
                rack_pose = [0.0] * 7
                
            # 로봇 본체(Base)의 월드 좌표 로깅 (이동 대차가 있을 경우 필수)
            if "robot" in scene.keys() and scene["robot"].data.root_state_w is not None:
                robot_pose = scene["robot"].data.root_state_w[0, :7].detach().cpu().tolist()
            else:
                robot_pose = [0.0] * 7
                
            # 현재 조이스틱(cmd_vel) 명령 값
            current_cmd_vel = list(bridge._current_cmd_vel())
                
            logger.log_step(robot_pose, current_pos, current_vel, action_target, current_cmd_vel, box_pose, rack_pose)

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        bridge.update_odometry(sim_dt)

        # 박스 리스폰 로직 (바닥으로 떨어졌을 때 & 작업대에 안착되었을 때)
        if "box" in scene.keys():
            box_pos = scene["box"].data.root_pos_w
            box_vel = scene["box"].data.root_lin_vel_w

            # ================================================================
            # [자석 부착 (Magic Snapping) 로직]
            # 거리가 12cm 이내이면서 손가락 타겟이 굽혀져 있으면(0.15 이상) 강제로 고정
            # ================================================================
            if "robot" in scene.keys() and box_pos is not None:
                if not hasattr(scene, "finger_indices"):
                    scene.finger_indices = [i for i, n in enumerate(scene["robot"].data.joint_names) if "finger" in n]
                
                if len(scene.finger_indices) > 0:
                    finger_target_avg = scene["robot"].data.joint_pos_target[0, scene.finger_indices].mean().item()
                    robot_body_pos = scene["robot"].data.body_pos_w[0] # (num_bodies, 3)
                    
                    dist_sq = torch.sum((robot_body_pos - box_pos[0])**2, dim=-1)
                    min_dist = torch.sqrt(torch.min(dist_sq)).item()
                    
                    # 꽉 쥐고(>0.15) 가까이 있을 때(<0.12m)
                    if min_dist < 0.12 and finger_target_avg > 0.15:
                        if not hasattr(scene, "grasped_body_idx"):
                            scene.grasped_body_idx = torch.argmin(dist_sq).item()
                            scene.grasp_offset = box_pos[0] - robot_body_pos[scene.grasped_body_idx]
                            scene.grasp_quat = scene["box"].data.root_quat_w[0].clone() # 잡은 순간의 회전 상태 저장
                        
                        target_state = scene["box"].data.root_state_w.clone()
                        target_state[0, :3] = robot_body_pos[scene.grasped_body_idx] + scene.grasp_offset
                        target_state[0, 3:7] = scene.grasp_quat # 회전(Orientation)도 고정하여 팽이처럼 도는 현상 원천 차단!
                        target_state[0, 7:13] = 0.0 # 속도를 0으로 만들어 튕김 방지
                        scene["box"].write_root_state_to_sim(target_state)
                    else:
                        # 손을 펴면(finger_target_avg < 0.15) 자석 효과 해제 (떨어짐)
                        if hasattr(scene, "grasped_body_idx"):
                            del scene.grasped_body_idx
            # ================================================================

            if box_pos is not None:
                # 1. 바닥에 떨어졌을 때 (Z < 0.5)
                if box_pos[0, 2] < 0.5:
                    print("[INFO] 상자가 떨어졌습니다. 받침대 위로 리스폰합니다.")
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
    sim.set_camera_view([0.5, 0.0, 4.0], [0.5, 0.0, 0.0]) # 위에서 아래를 내려다보는 Top-Down 카메라 뷰 설정

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

    domain_id = args_cli.domain_id if args_cli.domain_id is not None else int(os.getenv("ROS_DOMAIN_ID", 0))
    topic_manager = TopicManager(domain_id=domain_id)
    bridge = SH5DdsBridge(
        robot=robot,
        topic_manager=topic_manager,
        topic_names=_enabled_topics(),
        joint_states_topic=cfg.JOINT_STATES_TOPIC,
        odom_topic=cfg.ODOM_TOPIC,
        tf_topic=cfg.TF_TOPIC,
        base_frame=cfg.BASE_FRAME,
        odom_frame=cfg.ODOM_FRAME,
        trajectory_qos=_trajectory_qos(),
        cmd_vel_topic=None if args_cli.disable_cmd_vel else cfg.CMD_VEL_TOPIC,
        swerve_modules=[] if args_cli.disable_cmd_vel else _swerve_modules(),
        wheel_radius=SH5_SWERVE_WHEEL_RADIUS,
        cmd_vel_timeout=cfg.CMD_VEL_TIMEOUT,
        inference_mode=(args_cli.model is not None),
    )

    print(f"[INFO] FFW SH5 DDS bringup ready. ROS_DOMAIN_ID={domain_id}")
    if args_cli.enable_environment:
        print("[INFO] Environment: Simple Warehouse")
    print("[DDS] JointTrajectory subscriber reliability: best_effort")
    print(f"[DDS] Publishing joint states: {cfg.JOINT_STATES_TOPIC}")
    print(f"[DDS] Publishing odometry: {cfg.ODOM_TOPIC} ({cfg.ODOM_FRAME} -> {cfg.BASE_FRAME})")
    print(f"[DDS] Publishing TF: {cfg.TF_TOPIC} ({cfg.BASE_FRAME} -> robot links)")
    if not args_cli.disable_cmd_vel:
        print(f"[DDS] Applying swerve cmd_vel: {cfg.CMD_VEL_TOPIC}")

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = None
    act_config = None
    if args_cli.model is not None:
        import sys
        scripts_dir = str(Path(__file__).parent.absolute())
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        
        # 체크포인트 로드 후 모델 타입 자동 구분
        checkpoint = torch.load(args_cli.model, map_location=device)
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            # ACT 모델 (설정이 포함된 checkpoint)
            from train_act import ACTPolicy
            act_config = checkpoint['config']
            model = ACTPolicy(
                state_dim=act_config['state_dim'],
                action_dim=act_config['action_dim'],
                hidden_dim=act_config['hidden_dim'],
                chunk_size=act_config['chunk_size'],
                latent_dim=act_config['latent_dim'],
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"[INFO] ACT Model loaded! (chunk={act_config['chunk_size']}, "
                  f"context={act_config['context_len']}, hidden={act_config['hidden_dim']})")
        else:
            # BC 모델 (단순 state_dict)
            from train_bc import BehaviorCloningPolicy
            model = BehaviorCloningPolicy(state_dim=153, action_dim=66).to(device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            print(f"[INFO] BC (MLP) Model loaded from {args_cli.model}")

    try:
        run_simulator(sim, scene, bridge, model=model, device=device, act_config=act_config)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
