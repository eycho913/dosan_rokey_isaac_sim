#!/usr/bin/env python3
"""
=============================================================================
 SH5 HDF5 데이터 리플레이 스크립트
=============================================================================
 녹화된 HDF5 데이터를 Isaac Sim에서 재생하여 시각적으로 검증합니다.
 증강 데이터(미러링, Z오프셋 등)가 올바르게 생성되었는지 확인하는 데 사용합니다.

 사용법:
   isaac-python scripts/replay_data.py \
     --hdf5 /home/rokey/dev_ws/datasets/augmented_all_slots.hdf5 \
     --episode 0 \
     --enable_gravity

 키보드 조작:
   N / →  : 다음 에피소드
   P / ←  : 이전 에피소드
   R      : 현재 에피소드 처음부터 다시 재생
   Space  : 일시정지 / 재개
   Q      : 종료
   1~4    : 슬롯 1~4번 첫 에피소드로 점프 (증강 데이터용)
=============================================================================
"""

import argparse
import os
import sys
import time
import tty
import termios
import select
import threading
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

# CLI
parser = argparse.ArgumentParser(description="SH5 HDF5 데이터 리플레이")
parser.add_argument("--hdf5", type=str, required=True, help="재생할 HDF5 파일 경로")
parser.add_argument("--episode", type=int, default=0, help="시작 에피소드 번호")
parser.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율 (0.5=느리게, 2.0=빠르게)")
parser.add_argument("--enable_gravity", action="store_true", help="중력 활성화")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from robotis_lab.assets.robots import FFW_SH5_CFG


# ============================================================================
# Scene 설정 (bringup과 동일)
# ============================================================================
@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.88, 0.85), intensity=4500.0),
    )
    rack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/custom_rack2.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -1.5, 0.0), rot=(0.0, 0.0, 0.0, 1.0)
        )
    )
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/belt.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
    )
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.1, angular_damping=5.0,
                max_depenetration_velocity=0.3, enable_gyroscopic_forces=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.38, 0.08)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", static_friction=2.0,
                dynamic_friction=1.8, restitution=0.0,
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.7, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0))
    )
    robot: ArticulationCfg = None


# ============================================================================
# 키보드 입력 (논블로킹)
# ============================================================================
class ReplayKeyboard:
    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        import atexit
        atexit.register(self._restore)

    def _restore(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get_key(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            # 화살표키 처리
            if ch == '\x1b':
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'C': return 'right'  # →
                        if ch3 == 'D': return 'left'   # ←
            return ch
        return None


# ============================================================================
# HDF5 데이터 로더
# ============================================================================
class HDF5Player:
    def __init__(self, hdf5_path):
        self.f = h5py.File(hdf5_path, 'r')
        self.demo_names = sorted(self.f['data'].keys(), key=lambda x: int(x.split('_')[1]))
        self.num_episodes = len(self.demo_names)
        print(f"[Replay] 파일: {hdf5_path}")
        print(f"[Replay] 에피소드 수: {self.num_episodes}")
        
        # 슬롯 정보 확인
        slot_info = {}
        for d in self.demo_names:
            sid = self.f['data'][d].attrs.get('slot_id', 0)
            slot_info[sid] = slot_info.get(sid, 0) + 1
        if slot_info:
            print(f"[Replay] 슬롯별 에피소드: {slot_info}")
    
    def get_episode(self, ep_idx):
        """에피소드 데이터를 로드합니다."""
        ep_idx = max(0, min(ep_idx, self.num_episodes - 1))
        name = self.demo_names[ep_idx]
        demo = self.f['data'][name]
        
        data = {
            'actions': demo['actions'][:],        # (N, 63)
            'cmd_vel': demo['cmd_vel'][:],         # (N, 3)
            'joint_positions': demo['obs/joint_positions'][:],  # (N, 63)
            'box_pose': demo['obs/box_pose'][:],   # (N, 7)
            'robot_pose': demo['obs/robot_pose'][:],  # (N, 7)
        }
        
        slot_id = demo.attrs.get('slot_id', 0)
        num_samples = demo.attrs['num_samples']
        
        return data, num_samples, slot_id, name
    
    def find_slot_start(self, slot_id):
        """특정 슬롯의 첫 번째 에피소드 인덱스를 반환합니다."""
        for i, name in enumerate(self.demo_names):
            sid = self.f['data'][name].attrs.get('slot_id', 0)
            if sid == slot_id:
                return i
        return 0
    
    def close(self):
        self.f.close()


# ============================================================================
# 쿼터니언 회전 유틸리티 (Magic Snapping용)
# ============================================================================
def quat_rotate(q, v):
    """쿼터니언 q(wxyz)로 벡터 v를 회전"""
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


def clear_grasp_state(scene):
    """Magic Snap 상태 초기화"""
    for attr in ("grasped_body_idx", "grasp_local_offset", "grasp_quat", "finger_indices"):
        if hasattr(scene, attr):
            delattr(scene, attr)


# ============================================================================
# 리플레이 루프 (이동 + Magic Snapping 포함)
# ============================================================================
def run_replay(sim, scene, player, start_episode):
    kbd = ReplayKeyboard()
    sim_dt = sim.get_physics_dt()
    
    current_ep = start_episode
    frame_idx = 0
    paused = False
    
    # 에피소드 로드
    ep_data, num_samples, slot_id, ep_name = player.get_episode(current_ep)
    
    def print_status():
        slot_str = f" [슬롯 {slot_id}]" if slot_id > 0 else ""
        print(f"\n{'='*60}")
        print(f"  ▶ 재생 중: {ep_name} ({current_ep+1}/{player.num_episodes}){slot_str}")
        print(f"  프레임: {num_samples}개, 속도: x{args_cli.speed}")
        print(f"  조작: N=다음, P=이전, R=다시, Space=일시정지, 1~4=슬롯 점프")
        print(f"{'='*60}")
    
    def reset_to_frame0():
        """에피소드 첫 프레임으로 로봇+상자 초기화"""
        clear_grasp_state(scene)
        
        # 관절
        init_jp = torch.tensor(ep_data['joint_positions'][0], dtype=torch.float32).unsqueeze(0)
        robot.write_joint_state_to_sim(init_jp, torch.zeros_like(init_jp))
        robot.set_joint_position_target(init_jp)
        
        # 로봇 루트 위치
        rp = ep_data['robot_pose'][0]
        rstate = scene["robot"].data.default_root_state.clone()
        rstate[0, 0:3] = torch.tensor(rp[:3])
        rstate[0, 3:7] = torch.tensor(rp[3:7])
        rstate[0, 7:13] = 0.0  # 속도 0
        scene["robot"].write_root_state_to_sim(rstate)
        
        # 상자 위치
        if "box" in scene.keys():
            bp = ep_data['box_pose'][0]
            bstate = scene["box"].data.default_root_state.clone()
            bstate[0, 0:3] = torch.tensor(bp[:3])
            bstate[0, 3:7] = torch.tensor(bp[3:7])
            bstate[0, 7:13] = 0.0
            scene["box"].write_root_state_to_sim(bstate)
    
    print_status()
    robot = scene["robot"]
    reset_to_frame0()
    
    frame_delay = (1.0 / cfg.STEP_HZ) / args_cli.speed
    last_frame_time = time.time()
    
    while simulation_app.is_running():
        # 키보드 입력 처리
        key = kbd.get_key()
        reload_episode = False
        
        if key == 'q':
            break
        elif key == ' ':
            paused = not paused
            print(f"  {'⏸ 일시정지' if paused else '▶ 재생 재개'}")
        elif key in ('n', 'right'):
            current_ep = min(current_ep + 1, player.num_episodes - 1)
            reload_episode = True
        elif key in ('p', 'left'):
            current_ep = max(current_ep - 1, 0)
            reload_episode = True
        elif key == 'r':
            reload_episode = True
        elif key in ('1', '2', '3', '4'):
            target_slot = int(key)
            current_ep = player.find_slot_start(target_slot)
            reload_episode = True
            print(f"  → 슬롯 {target_slot}번으로 점프!")
        
        if reload_episode:
            ep_data, num_samples, slot_id, ep_name = player.get_episode(current_ep)
            frame_idx = 0
            print_status()
            reset_to_frame0()
            last_frame_time = time.time()
            continue
        
        # 일시정지 중이면 시뮬레이션만 돌리고 프레임 진행 안 함
        if paused:
            sim.step(render=True)
            scene.update(sim_dt)
            continue
        
        # 프레임 속도 제어
        now = time.time()
        if now - last_frame_time < frame_delay:
            sim.step(render=True)
            scene.update(sim_dt)
            continue
        last_frame_time = now
        
        # ================================================================
        # 프레임 재생
        # ================================================================
        if frame_idx < num_samples:
            # 1) 관절 타겟 적용
            joint_targets = torch.tensor(
                ep_data['actions'][frame_idx], dtype=torch.float32
            ).unsqueeze(0)
            robot.set_joint_position_target(joint_targets)
            
            # 2) 로봇 루트 위치를 녹화 데이터로 강제 이동 (이동 재현!)
            rp = ep_data['robot_pose'][frame_idx]
            root_state = scene["robot"].data.root_state_w.clone()
            root_state[0, 0:3] = torch.tensor(rp[:3])
            root_state[0, 3:7] = torch.tensor(rp[3:7])
            # 속도는 다음 프레임과의 차이로 추정 (부드러운 움직임)
            if frame_idx + 1 < num_samples:
                rp_next = ep_data['robot_pose'][frame_idx + 1]
                vel = (np.array(rp_next[:3]) - np.array(rp[:3])) / sim_dt
                root_state[0, 7:10] = torch.tensor(vel, dtype=torch.float32)
            else:
                root_state[0, 7:10] = 0.0
            root_state[0, 10:13] = 0.0
            scene["robot"].write_root_state_to_sim(root_state)
            
            # 3) 진행률 표시 (50프레임마다)
            if frame_idx % 50 == 0:
                progress = (frame_idx / num_samples) * 100
                bar_len = 30
                filled = int(bar_len * frame_idx / num_samples)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"\r  [{bar}] {progress:5.1f}% ({frame_idx}/{num_samples})", end='', flush=True)
        else:
            # 에피소드 끝 → 자동으로 다음 에피소드
            print(f"\n  ✅ 에피소드 완료!")
            if current_ep < player.num_episodes - 1:
                current_ep += 1
                ep_data, num_samples, slot_id, ep_name = player.get_episode(current_ep)
                frame_idx = 0
                print_status()
                reset_to_frame0()
                last_frame_time = time.time()
                continue
            else:
                print("  🏁 모든 에피소드 재생 완료!")
                while simulation_app.is_running():
                    key = kbd.get_key()
                    if key == 'q':
                        return
                    if key == 'r':
                        current_ep = 0
                        break
                    sim.step(render=True)
                    scene.update(sim_dt)
                continue
        
        frame_idx += 1
        
        # 시뮬레이션 스텝
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        
        # ================================================================
        # Magic Snapping 로직 (bringup.py와 동일)
        # ================================================================
        if "box" in scene.keys():
            box_pos = scene["box"].data.root_pos_w
            
            if "robot" in scene.keys() and box_pos is not None:
                if not hasattr(scene, "finger_indices"):
                    scene.finger_indices = [
                        i for i, n in enumerate(scene["robot"].data.joint_names)
                        if "finger" in n
                    ]
                
                if len(scene.finger_indices) > 0:
                    finger_target_avg = scene["robot"].data.joint_pos_target[
                        0, scene.finger_indices
                    ].mean().item()
                    robot_body_pos = scene["robot"].data.body_pos_w[0]
                    robot_body_quat = scene["robot"].data.body_quat_w[0]
                    
                    dist_sq = torch.sum((robot_body_pos - box_pos[0])**2, dim=-1)
                    min_dist = torch.sqrt(torch.min(dist_sq)).item()
                    
                    if min_dist < 0.15 and finger_target_avg > 0.20:
                        if not hasattr(scene, "grasped_body_idx"):
                            scene.grasped_body_idx = torch.argmin(dist_sq).item()
                            idx = scene.grasped_body_idx
                            body_q = robot_body_quat[idx]
                            world_offset = box_pos[0] - robot_body_pos[idx]
                            w, x, y, z = body_q[0], body_q[1], body_q[2], body_q[3]
                            inv_q = torch.tensor([w, -x, -y, -z], device=body_q.device)
                            scene.grasp_local_offset = quat_rotate(inv_q, world_offset)
                            scene.grasp_quat = scene["box"].data.root_quat_w[0].clone()
                        
                        idx = scene.grasped_body_idx
                        body_q = robot_body_quat[idx]
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


# ============================================================================
# 메인
# ============================================================================
def main():
    usd_path = FFW_SH5_CFG.spawn.usd_path
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SH5 USD not found: {usd_path}")
    
    if not os.path.exists(args_cli.hdf5):
        raise FileNotFoundError(f"HDF5 file not found: {args_cli.hdf5}")
    
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",
        dt=1.0 / cfg.STEP_HZ,
        render_interval=cfg.RENDER_INTERVAL,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=16,
            min_velocity_iteration_count=2,
            enable_stabilization=True,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 2.0], [0.3, 0.0, 0.8])
    
    scene_cfg = ReplaySceneCfg(num_envs=1, env_spacing=2.0)
    
    robot_cfg = deepcopy(FFW_SH5_CFG)
    robot_cfg.spawn.usd_path = usd_path
    robot_cfg.spawn.rigid_props.disable_gravity = not args_cli.enable_gravity
    robot_cfg.init_state.pos = cfg.ROBOT_POS
    scene_cfg.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())
    
    # HDF5 로드
    player = HDF5Player(args_cli.hdf5)
    
    print("\n" + "="*60)
    print("  🎬 SH5 데이터 리플레이 시작!")
    print("  N/→: 다음 | P/←: 이전 | R: 다시 | Space: 일시정지")
    print("  1~4: 슬롯 점프 | Q: 종료")
    print("="*60)
    
    try:
        run_replay(sim, scene, player, args_cli.episode)
    finally:
        player.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
