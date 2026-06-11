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
parser.add_argument("--slot", type=int, default=1, choices=[1,2,3,4],
    help="[수정A] 현재 수집 중인 슬롯 번호 (1~4). 비활성 팔 대기 자세 결정에 사용.")
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
import torch

from sh5_logger import VRDemonstrationLogger
from sh5_keyboard import TerminalKeyboard
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
        # 데이터 수집 품질을 위해 밝고 고르게 퍼지는 돔 조명 사용
        # intensity 4500: 카메라 영상이 너무 어둡거나 밝지 않은 균형점
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.88, 0.85), intensity=4500.0),
    )
    rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/custom_rack2.usd",
            # 랙 충돌 감도 최적화: 상자가 랙에 정확히 안착하도록
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -1.5, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0)
        )
    )
    
    # 3. 상자 받침대(Pedestal) 설정 (상자를 로봇 가까이 올려두기 위한 투명/회색 테이블 역할)
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/belt.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            # 벨트 충돌 감도 설정 - 상자가 관통하지 않도록 contact_offset 작게
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,   # 2mm - 빨리 접촉 감지 방지
                rest_offset=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )
    
    # 4. 목표물 상자(Box) 설정 (로봇이 실제로 집어야 하는 대상 물체)
    # 크기(size), 질량(mass), 마찰력(friction) 등을 수정하여 다양한 훈련 환경을 구축할 수 있습니다.
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 또마닙 감소: 손가락 마찰력이 자연스럽게 작용하도록
                # (damping이 너무 크면 손으로 밀어도 저항하여 실제로는 손가락에 달라붙지 않는 느낌)
                linear_damping=0.1,
                angular_damping=5.0,
                max_depenetration_velocity=0.3,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),  # 0.5 → 1.5kg: 더 묵직하고 안정적인 집기 동작
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.38, 0.08)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                # 손가락의 friction_combine_mode="max"와 연동: 최종 마찰력 = max(1000, 2.0) = 1000 적용
                friction_combine_mode="max",
                static_friction=2.0,     # 정지 마찰력 증가 (1.5 → 2.0)
                dynamic_friction=1.8,    # 동마찰력 증가 (1.2 → 1.8)
                restitution=0.0,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.7, 0.0, 1.0),
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


from sh5_dds_bridge import SH5DdsBridge

def _setup_camera_views():
    from isaacsim.core.utils.stage import get_current_stage

    stage = get_current_stage()

    camera_specs = (
        ("Center Camera", cfg.AI_WORKER_CAMERA_CENTER_NAME, 780, 490, 50, 22),
        ("Left Camera",   cfg.AI_WORKER_CAMERA_LEFT_NAME,   387, 280, 50, 517),
        ("Right Camera",  cfg.AI_WORKER_CAMERA_RIGHT_NAME,  387, 280, 441, 517),
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

    # ── [수정2] Top View: 로봇 작업 공간 위 고정 카메라 ─────────────────
    try:
        from pxr import UsdGeom, Gf, Sdf
        top_cam_path = "/World/TopViewCamera"

        # 기존 prim 삭제 후 재생성 (설정 반영 보장)
        existing = stage.GetPrimAtPath(top_cam_path)
        if existing.IsValid():
            stage.RemovePrim(Sdf.Path(top_cam_path))

        top_cam = UsdGeom.Camera.Define(stage, Sdf.Path(top_cam_path))
        top_cam.GetFocalLengthAttr().Set(12.0)   # 넓은 화각 (작을수록 넓음)
        top_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

        # eye/target 방식으로 정확히 아래 방향 설정
        # 로봇(0,0)과 상자(0.7,0)의 중앙인 0.4 지점 위 3.5m
        eye    = Gf.Vec3d(0.0, 0.0, 3.0)   # 카메라 위치 (높이 3.5m)
        target = Gf.Vec3d(0.4, 0.0, 0.0)   # 바닥 바라보는 지점
        up     = Gf.Vec3d(1.0, 0.0, 0.0)   # 뷰포트 위쪽 방향을 X축으로 변경하여 90도 회전

        # look-at 행렬로 카메라 Transform 설정
        forward = (target - eye).GetNormalized()
        right   = Gf.Cross(forward, up).GetNormalized()
        up_real = Gf.Cross(right, forward).GetNormalized()

        # USD Camera는 -Z 방향이 forward이므로 flip
        m = Gf.Matrix4d(
             right[0],    right[1],    right[2],   0,
             up_real[0],  up_real[1],  up_real[2], 0,
            -forward[0], -forward[1], -forward[2], 0,
             eye[0],      eye[1],      eye[2],     1,
        )
        xform = UsdGeom.Xformable(top_cam.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(m)

        _set_viewport_camera("Top View", top_cam_path, width=500, height=500, x=835, y=22)
        camera_paths["TopView"] = top_cam_path
        print("[INFO] Top View 카메라 설정 완료 (로봇 위 3.5m, 바닥 방향, 90도 회전)")
    except Exception as e:
        print(f"[WARN] Top View 카메라 생성 실패: {e}")

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




# ========== Simulation Loop ==========

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, bridge: SH5DdsBridge, camera_paths: dict):
    logger = VRDemonstrationLogger(output_dir="/home/rokey/dev_ws/datasets")
    logger._slot_id = getattr(args_cli, 'slot', 1)  # [수정A] CLI --slot 인자 반영
    print(f"[INFO] 현재 수집 슬롯: {logger._slot_id} | 비활성 팔: {'왼팔' if logger._slot_id in (1,3) else '오른팔'} 대기 자세 적용")
    term_kbd = TerminalKeyboard()

    # 리프트 위치 상태 (키보드로 누적 제어)
    lift_pos = 0.0
    # 헤드 상태 (pan=좌우, tilt=상하)
    head_pan = TerminalKeyboard.HEAD_PAN_DEFAULT
    head_tilt = TerminalKeyboard.HEAD_TILT_DEFAULT

    sim_dt = sim.get_physics_dt()
    step_period = 1.0 / cfg.STEP_HZ if cfg.STEP_HZ > 0 else 0.0
    publish_period = 1.0 / cfg.PUBLISH_HZ if cfg.PUBLISH_HZ > 0 else 0.0
    last_publish = 0.0
    last_step = time.time()

    # Replicator Annotators setup for vision data
    import omni.replicator.core as rep
    annotators = {}
    if camera_paths:
        print("[INFO] Replicator 카메라 어노테이터 설정 중...")
        for cam_name, cam_path in camera_paths.items():
            if cam_name in ["Left Camera", "Right Camera", "TopView"]:
                rp = rep.create.render_product(cam_path, (160, 120)) # Width 160, Height 120
                rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb_annot.attach([rp])
                annotators[cam_name] = rgb_annot
        print(f"[INFO] 준비된 비전 카메라: {list(annotators.keys())}")

    print("\n" + "="*70)
    print("🎮 [통합 조작 안내 - 터미널 1개로 이동 + 녹화 동시 제어] 🎮")
    print("")
    print("[이동 제어 (WASD/QE)]")
    print("  W/S : 전진/후진    A/D : 좌/우 회전    Q/E : 좌/우 횡이동")
    print("  U/O : 리프트 올림/내림    Space : 정지")
    print("[시야(헤드) 제어]")
    print("  I/K : 고개 위/아래 (Tilt)    J/L : 고개 좌/우 (Pan)")
    print("")
    print("[녹화 제어 (R/T/C/B 또는 숫자 1/2/3/4)]")
    print("  R 또는 1 : 🔴 녹화 시작")
    print("  T 또는 2 : ⬛ 녹화 저장 (성공 에피소드)")
    print("  C 또는 3 : 🗑️ 녹화 취소 (실패 에피소드, 버림)")
    print("  B 또는 4 : 📦 상자 랜덤 리스폰 + 로봇 초기화")
    print("  V       : 🔄 로봇만 초기 위치로")
    print("="*70 + "\n")

    while simulation_app.is_running():
        key = term_kbd.get_key_and_clear()
        
        # ---- 이동 제어 (mobile_teleop 통합) ----
        twist_x, twist_y, twist_w = 0.0, 0.0, 0.0
        if key == 'w':
            twist_x = TerminalKeyboard.LINEAR_SPEED
        elif key == 's':
            twist_x = -TerminalKeyboard.LINEAR_SPEED
        elif key == 'a':
            twist_w = TerminalKeyboard.ANGULAR_SPEED
        elif key == 'd':
            twist_w = -TerminalKeyboard.ANGULAR_SPEED
        elif key == 'q':
            twist_y = TerminalKeyboard.LINEAR_SPEED
        elif key == 'e':
            twist_y = -TerminalKeyboard.LINEAR_SPEED
        elif key == ' ':
            pass  # 정지 (0으로 유지)
        elif key == 'u':
            lift_pos = min(TerminalKeyboard.LIFT_MAX, lift_pos + TerminalKeyboard.LIFT_STEP)
        elif key == 'o':
            lift_pos = max(TerminalKeyboard.LIFT_MIN, lift_pos - TerminalKeyboard.LIFT_STEP)
        elif key == 'i':
            head_tilt = max(TerminalKeyboard.HEAD_TILT_MIN, head_tilt - TerminalKeyboard.HEAD_STEP)
            print(f"[HEAD] 위 tilt={head_tilt:.2f}rad")
        elif key == 'k':
            head_tilt = min(TerminalKeyboard.HEAD_TILT_MAX, head_tilt + TerminalKeyboard.HEAD_STEP)
            print(f"[HEAD] 아래 tilt={head_tilt:.2f}rad")
        elif key == 'j':
            head_pan = min(TerminalKeyboard.HEAD_PAN_MAX, head_pan + TerminalKeyboard.HEAD_STEP)
            print(f"[HEAD] 좌 pan={head_pan:.2f}rad")
        elif key == 'l':
            head_pan = max(TerminalKeyboard.HEAD_PAN_MIN, head_pan - TerminalKeyboard.HEAD_STEP)
            print(f"[HEAD] 우 pan={head_pan:.2f}rad")

        # 이동 명령을 DDS bridge에 직접 주입 (키가 없으면 0으로 정지)
        if key in ('w', 's', 'a', 'd', 'q', 'e', ' '):
            with bridge.lock:
                bridge.latest_cmd_vel = (twist_x, twist_y, twist_w)
                bridge.last_cmd_vel_time = time.monotonic()

        # 리프트 명령을 DDS bridge에 직접 주입
        if key in ('u', 'o'):
            with bridge.lock:
                bridge.pending_positions['lift_joint'] = lift_pos

        # 헤드 명령을 DDS bridge에 직접 주입
        if key in ('i', 'k', 'j', 'l'):
            with bridge.lock:
                bridge.pending_positions['head_joint1'] = head_pan
                bridge.pending_positions['head_joint2'] = head_tilt

        # ---- 녹화 제어 ----
        if (key == 'r' or key == '1') and not logger.is_recording:
            logger.start_recording()
        elif (key == 't' or key == '2') and logger.is_recording:
            logger.stop_recording_and_save()
        elif (key == 'c' or key == '3') and logger.is_recording:
            logger.cancel_recording()
        elif key == 'b' or key == '4':
            if "box" in scene.keys():
                rand_offset_x = np.random.uniform(-0.10, 0.20)
                rand_offset_y = np.random.uniform(-0.20, 0.20)
                default_state = scene["box"].data.default_root_state.clone()
                default_state[0, 0] += rand_offset_x
                default_state[0, 1] += rand_offset_y
                scene["box"].write_root_state_to_sim(default_state)
                print(f"[INFO] 📦 상자 랜덤 리스폰! 오프셋 X={rand_offset_x:+.3f}m, Y={rand_offset_y:+.3f}m")
            if "robot" in scene.keys():
                default_root_state = scene["robot"].data.default_root_state.clone()
                scene["robot"].write_root_state_to_sim(default_root_state)
                default_joint_pos = scene["robot"].data.default_joint_pos.clone()
                default_joint_vel = scene["robot"].data.default_joint_vel.clone()
                scene["robot"].write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                scene["robot"].set_joint_position_target(default_joint_pos)
                bridge.clear_pending_targets()
                lift_pos = 0.0  # 리프트 위치도 초기화
                print("[INFO] 🔄 로봇 초기화 완료")

        # ── [수정A+4] 녹화 중 비활성 팔 대기 자세 강제 적용 (벨트 충돌 방지) ──
        # 팔을 몸쪽으로 접어서 장애물에 걸리지 않도록 함
        # joint1(어깨): 0.0, joint2(어깨pitch): -1.8(뒤로 접기), joint3(elbow): 1.5(접기)
        # joint4~7: 손목 중립 자세
        if logger.is_recording:
            _INACTIVE_REST = [0.0, -1.8, 1.5, 0.0, 0.0, 0.0, 0.0]  # 몸쪽 접힌 안전 자세
            _ARM_L_NAMES = ['arm_l_joint1','arm_l_joint2','arm_l_joint3',
                            'arm_l_joint4','arm_l_joint5','arm_l_joint6','arm_l_joint7']
            _ARM_R_NAMES = ['arm_r_joint1','arm_r_joint2','arm_r_joint3',
                            'arm_r_joint4','arm_r_joint5','arm_r_joint6','arm_r_joint7']
            _SLOT_ID = getattr(logger, '_slot_id', 1)
            _INACTIVE_NAMES = _ARM_L_NAMES if _SLOT_ID in (1, 3) else _ARM_R_NAMES
            _REST_ALPHA = 0.015  # 매우 부드럽게 (급격한 움직임 방지)
            with bridge.lock:
                for j_name, rest_val in zip(_INACTIVE_NAMES, _INACTIVE_REST):
                    current_val = bridge.pending_positions.get(j_name, 0.0)
                    bridge.pending_positions[j_name] = (1 - _REST_ALPHA) * current_val + _REST_ALPHA * rest_val

        bridge.apply_latest_targets()
        
        if logger.is_recording:
            current_pos = scene["robot"].data.joint_pos.squeeze(0).detach().cpu().tolist()
            current_vel = scene["robot"].data.joint_vel.squeeze(0).detach().cpu().tolist()
            action_target = scene["robot"].data.joint_pos_target.squeeze(0).detach().cpu().tolist()
            
            # 상자 위치와 회전 정보를 합쳐서(7차원: x,y,z,qw,qx,qy,qz) 함께 로깅
            if "box" in scene.keys() and scene["box"].data.root_state_w is not None:
                box_pose = scene["box"].data.root_state_w[0, :7].detach().cpu().tolist()
            else:
                box_pose = [0.0] * 7 # 예외 처리
                
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
                
            images = {}
            if annotators:
                for cam_name, ann in annotators.items():
                    data = ann.get_data()
                    if data is not None and isinstance(data, np.ndarray) and data.size > 0:
                        images[cam_name] = data[..., :3].copy()  # RGBA to RGB
                    else:
                        images[cam_name] = np.zeros((120, 160, 3), dtype=np.uint8)

            logger.log_step(robot_pose, current_pos, current_vel, action_target, current_cmd_vel, box_pose, rack_pose, images=images)

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        bridge.update_odometry(sim_dt)

        # 박스 리스폰 로직 (바닥으로 떨어졌을 때 & 작업대에 안착되었을 때)
        if "box" in scene.keys():
            box_pos = scene["box"].data.root_pos_w
            box_vel = scene["box"].data.root_lin_vel_w

            # ================================================================
            # [Magic Snapping 로직 - 개선판]
            # 1. 거리 임계값 안에서 손가락을 궁히면 상자 고정
            # 2. offset을 로봇 실제 body의 로컈(local) 코디네이트로 저장 → 회전 시 이탈 방지
            # ================================================================
            if "robot" in scene.keys() and box_pos is not None:
                if not hasattr(scene, "finger_indices"):
                    scene.finger_indices = [i for i, n in enumerate(scene["robot"].data.joint_names) if "finger" in n]
                
                if len(scene.finger_indices) > 0:
                    finger_target_avg = scene["robot"].data.joint_pos_target[0, scene.finger_indices].mean().item()
                    robot_body_pos = scene["robot"].data.body_pos_w[0]      # (num_bodies, 3)
                    robot_body_quat = scene["robot"].data.body_quat_w[0]    # (num_bodies, 4) wxyz
                    
                    dist_sq = torch.sum((robot_body_pos - box_pos[0])**2, dim=-1)
                    min_dist = torch.sqrt(torch.min(dist_sq)).item()
                    
                    if min_dist < 0.15 and finger_target_avg > 0.20:
                        if not hasattr(scene, "grasped_body_idx"):
                            scene.grasped_body_idx = torch.argmin(dist_sq).item()
                            idx = scene.grasped_body_idx
                            
                            # 개선: 오프셋을 해당 body의 로컈 코디네이트로 저장 (world offset 아님!)
                            # q_inv(body_quat) * (box_pos - body_pos)
                            body_q = robot_body_quat[idx]  # wxyz
                            world_offset = box_pos[0] - robot_body_pos[idx]
                            # 쿠염턴으로 local 코디네이트 변환 (q_inv * v)
                            w, x, y, z = body_q[0], body_q[1], body_q[2], body_q[3]
                            # q_inv = (w, -x, -y, -z)
                            inv_q = torch.tensor([w, -x, -y, -z], device=body_q.device)
                            # v' = q_inv * (0, v) * q (쿠염턴 회전)
                            def quat_rotate(q, v):
                                """q(wxyz)로 v를 회전"""
                                wq, xq, yq, zq = q[0], q[1], q[2], q[3]
                                vx, vy, vz = v[0], v[1], v[2]
                                tx = 2*(yq*vz - zq*vy)
                                ty = 2*(zq*vx - xq*vz)
                                tz = 2*(xq*vy - yq*vx)
                                return torch.stack([
                                    vx + wq*tx + yq*tz - zq*ty,
                                    vy + wq*ty + zq*tx - xq*tz,
                                    vz + wq*tz + xq*ty - yq*tx
                                ])
                            scene.grasp_local_offset = quat_rotate(inv_q, world_offset)
                            scene.grasp_quat = scene["box"].data.root_quat_w[0].clone()
                        
                        # 개선: body 현재 회전으로 local offset을 다시 world로 변환
                        idx = scene.grasped_body_idx
                        body_q = robot_body_quat[idx]
                        def quat_rotate(q, v):
                            wq, xq, yq, zq = q[0], q[1], q[2], q[3]
                            vx, vy, vz = v[0], v[1], v[2]
                            tx = 2*(yq*vz - zq*vy)
                            ty = 2*(zq*vx - xq*vz)
                            tz = 2*(xq*vy - yq*vx)
                            return torch.stack([
                                vx + wq*tx + yq*tz - zq*ty,
                                vy + wq*ty + zq*tx - xq*tz,
                                vz + wq*tz + xq*ty - yq*tx
                            ])
                        world_offset_now = quat_rotate(body_q, scene.grasp_local_offset)
                        
                        target_state = scene["box"].data.root_state_w.clone()
                        target_state[0, :3] = robot_body_pos[idx] + world_offset_now
                        target_state[0, 3:7] = scene.grasp_quat
                        target_state[0, 7:13] = 0.0
                        scene["box"].write_root_state_to_sim(target_state)
                    else:
                        if hasattr(scene, "grasped_body_idx"):
                            del scene.grasped_body_idx
                        if hasattr(scene, "grasp_local_offset"):
                            del scene.grasp_local_offset
            # ================================================================

            # 밑로 떨어진 상자 리스폰
            if box_pos is not None:
                # 밡로 떨어졌을 때 (Z 높이 < 0.5m)
                if box_pos[0, 2] < 0.5:
                    print("[INFO] 상자가 떨어졌습니다. 받침대 위로 리스폰합니다.")
                    default_state = scene["box"].data.default_root_state.clone()
                    scene["box"].write_root_state_to_sim(default_state)
                    if hasattr(scene, "grasped_body_idx"): del scene.grasped_body_idx
                    if hasattr(scene, "grasp_local_offset"): del scene.grasp_local_offset

                # 빨리 움직이는 상자 속도 클램핑 (터널링 = 빨리 이동 시 충돌 감지 누락)
                if box_vel is not None and not hasattr(scene, "grasped_body_idx"):
                    speed = torch.norm(box_vel[0]).item()
                    MAX_BOX_SPEED = 1.5  # m/s 이상이면 속도 제한
                    if speed > MAX_BOX_SPEED:
                        clamped_vel = box_vel[0] * (MAX_BOX_SPEED / speed)
                        box_state = scene["box"].data.root_state_w.clone()
                        box_state[0, 7:10] = clamped_vel
                        scene["box"].write_root_state_to_sim(box_state)


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
        # RTX 5080 (Blackwell)에서 GPU PhysX 파이프라인이 불안정하여 CPU 사용
        # (GPU 파이프라인은 소프트웨어로 폴백되어 오히려 느려지고 경고를 발생시킴)
        device="cpu",
        dt=1.0 / cfg.STEP_HZ,
        render_interval=cfg.RENDER_INTERVAL,
        physx=sim_utils.PhysxCfg(
            solver_type=1,                    # TGS: PGS보다 안정적 (CPU에서도 유효)
            min_position_iteration_count=8,   # 기본 4 → 8: 충돌 해상도 향상
            max_position_iteration_count=16,
            min_velocity_iteration_count=2,
            # enable_ccd=True 는 GPU 파이프라인 전용 → CPU에서는 미지원, 제거
            enable_stabilization=True,        # dt=1/30s 경계값 경고 해소
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    # 데이터 수집에 최적화된 카메라 뷰: 로봇 왼쪽 45도 위에서 내려다보는 사선 뷰
    sim.set_camera_view([1.5, 1.5, 2.0], [0.3, 0.0, 0.8])

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

    try:
        run_simulator(sim, scene, bridge, camera_paths)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
