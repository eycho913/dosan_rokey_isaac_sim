#!/usr/bin/env python3
"""
=============================================================================
 replay_data2.py  —  sh5_bringup_ros2 환경과 동일한 HDF5 리플레이
=============================================================================
 sh5_bringup_ros2.py에서 사용하는 씬(rack, pedestal, TopView camera, box_assets)
 및 물리 설정과 동일한 환경을 그대로 구성하여 HDF5 에피소드를 재생합니다.

 사용법:
   isaac-python scripts/replay_data2.py \
     --hdf5 /home/rokey/dev_ws/datasets/train_data/frozen_set/slot1_1_f.hdf5 \
     --episode 0 \
     --speed 1.0 \
     --enable_gravity

 키보드 조작:
   N / →  : 다음 에피소드
   P / ←  : 이전 에피소드
   R      : 현재 에피소드 처음부터 다시
   Space  : 일시정지 / 재개
   1~4    : 슬롯 점프
   Q      : 종료
=============================================================================
"""

import argparse
import os
import sys
import time
import tty
import termios
import select
from copy import deepcopy
from pathlib import Path

from isaaclab.app import AppLauncher

# ── 경로 설정 ────────────────────────────────────────────────────────────────
ROBOTIS_LAB_DIR = Path("/home/rokey/dev_ws/robotis_lab/scripts/sim2real/bringup")
if str(ROBOTIS_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOTIS_LAB_DIR))

DEV_WS_DIR = Path("/home/rokey/dev_ws/robotis_lab/source/robotis_lab")
if str(DEV_WS_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_WS_DIR))

from common import robotis_config as cfg

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="sh5_bringup_ros2 환경 HDF5 리플레이")
parser.add_argument("--hdf5",          type=str,   required=True,  help="재생할 HDF5 파일 경로")
parser.add_argument("--episode",       type=int,   default=0,      help="시작 에피소드 번호")
parser.add_argument("--speed",         type=float, default=1.0,    help="재생 속도 배율")
parser.add_argument("--enable_gravity",action="store_true",        help="로봇 중력 활성화")
parser.add_argument("--no_camera",     action="store_true",        help="TopView 카메라 비활성화 (경량화)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# TopView 카메라 활성화 시 enable_cameras 필요
if not args_cli.no_camera:
    args_cli.enable_cameras = True

app_launcher  = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab 임포트 (AppLauncher 이후) ──────────────────────────────────────
import h5py
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from robotis_lab.assets.robots import FFW_SH5_CFG

# TopView 카메라 (no_camera 옵션 시 제외)
if not args_cli.no_camera:
    from isaaclab.sensors import CameraCfg

# ── 상수 ─────────────────────────────────────────────────────────────────────
BOX_ASSETS_DIR  = Path("/home/rokey/dev_ws/box_assets")
PLAYBACK_SPEED  = max(1, int(args_cli.speed))   # 정수 배속 (프레임 건너뜀)
BOX_SPAWN_QUAT  = [0.7071, -0.7071, 0.0, 0.0]  # QR면이 위를 향하는 X-90도 쿼터니언


# ============================================================================
# 씬 설정 (sh5_bringup_ros2.BringupSceneCfg와 동일)
# ============================================================================
@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground   = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )
    light    = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.88, 0.85), intensity=4500.0)
    )
    rack     = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Rack",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/custom_rack2.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -1.5, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/rokey/dev_ws/assets/belt.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    box  : RigidObjectCfg  = None
    robot: ArticulationCfg = None


# ============================================================================
# 키보드 입력 (논블로킹)
# ============================================================================
class ReplayKeyboard:
    def __init__(self):
        self._fd  = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        import atexit
        atexit.register(self._restore)

    def _restore(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get_key(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        ch3 = sys.stdin.read(1)
                        if ch3 == 'C': return 'right'
                        if ch3 == 'D': return 'left'
            return ch
        return None


# ============================================================================
# HDF5 플레이어
# ============================================================================
class HDF5Player:
    def __init__(self, hdf5_path: str):
        self.f = h5py.File(hdf5_path, 'r')
        self.demo_names = sorted(
            self.f['data'].keys(),
            key=lambda x: int(x.split('_')[1])
        )
        self.num_episodes = len(self.demo_names)
        print(f"[Replay] 파일: {hdf5_path}")
        print(f"[Replay] 에피소드 수: {self.num_episodes}")

        # 슬롯 정보 요약
        slot_info = {}
        for d in self.demo_names:
            sid = self.f['data'][d].attrs.get('slot_id', 0)
            slot_info[sid] = slot_info.get(sid, 0) + 1
        if slot_info:
            print(f"[Replay] 슬롯별 에피소드: {slot_info}")

    def get_episode(self, ep_idx: int):
        ep_idx = max(0, min(ep_idx, self.num_episodes - 1))
        name   = self.demo_names[ep_idx]
        demo   = self.f['data'][name]

        data = {
            'actions':         demo['actions'][:],               # (N, 63)
            'joint_positions': demo['obs/joint_positions'][:],   # (N, 63)
            'box_pose':        demo['obs/box_pose'][:],          # (N, 7)
            'robot_pose':      demo['obs/robot_pose'][:],        # (N, 7)
        }
        # cmd_vel은 있을 수도 없을 수도 있음
        if 'cmd_vel' in demo:
            data['cmd_vel'] = demo['cmd_vel'][:]
        else:
            data['cmd_vel'] = np.zeros((len(data['actions']), 3))

        slot_id    = demo.attrs.get('slot_id', 0)
        num_samples = demo.attrs['num_samples']
        return data, num_samples, slot_id, name

    def find_slot_start(self, slot_id: int) -> int:
        for i, name in enumerate(self.demo_names):
            sid = self.f['data'][name].attrs.get('slot_id', 0)
            if sid == slot_id:
                return i
        return 0

    def close(self):
        self.f.close()


# ============================================================================
# 상자 USD 선택 (sh5_bringup_ros2._get_box_usd와 동일)
# ============================================================================
def _get_box_usd(package_id: str = "INITIAL"):
    exact = BOX_ASSETS_DIR / f"{package_id}.usd"
    if exact.exists():
        return str(exact)
    parts = package_id.split("_")
    if parts:
        matches = list(BOX_ASSETS_DIR.glob(f"*_{parts[-1]}.usd"))
        if matches:
            return str(matches[0])
    usd_files = list(BOX_ASSETS_DIR.glob("*.usd"))
    if usd_files:
        import random
        return str(random.choice(usd_files))
    return None


# ============================================================================
# 리플레이 루프
# ============================================================================
def run_replay(sim, scene, player: HDF5Player, start_episode: int):
    kbd    = ReplayKeyboard()
    sim_dt = sim.get_physics_dt()
    robot  = scene["robot"]

    current_ep = start_episode
    frame_idx  = 0
    paused     = False

    ep_data, num_samples, slot_id, ep_name = player.get_episode(current_ep)

    # ── 헬퍼: 상자 위치를 HDF5 pose로 강제 주입 (물리 엔진 완전 무시) ─────
    def _force_box_pose(bp):
        """HDF5 box_pose(7d) → Isaac Sim 상자에 직접 강제 주입.
        상자를 kinematic처럼 다루어 물리 간섭 없이 수집 당시 궤적 완벽 재현."""
        if "box" not in scene.keys():
            return
        bstate = scene["box"].data.root_state_w.clone()
        bstate[0, 0] = float(bp[0])
        bstate[0, 1] = float(bp[1])
        bstate[0, 2] = float(bp[2])
        bstate[0, 3] = float(bp[3])  # qw
        bstate[0, 4] = float(bp[4])  # qx
        bstate[0, 5] = float(bp[5])  # qy
        bstate[0, 6] = float(bp[6])  # qz
        bstate[0, 7:]  = 0.0          # 선/각속도 = 0 (물리 누적 방지)
        scene["box"].write_root_state_to_sim(bstate)
        # ★ USD 단계 kinematic 강제 → PhysX가 덮어쓰지 못하게
        try:
            from pxr import UsdPhysics
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            box_prim = stage.GetPrimAtPath("/World/envs/env_0/Box")
            if box_prim and box_prim.IsValid():
                rb_api = UsdPhysics.RigidBodyAPI(box_prim)
                if rb_api:
                    rb_api.GetKinematicEnabledAttr().Set(True)
        except Exception:
            pass

    # ── 헬퍼: 에피소드 첫 프레임으로 리셋 ──────────────────────────────────
    def reset_to_frame0():
        # 관절 상태 초기화
        init_jp = torch.tensor(
            ep_data['joint_positions'][0], dtype=torch.float32
        ).unsqueeze(0)
        robot.write_joint_state_to_sim(init_jp, torch.zeros_like(init_jp))
        robot.set_joint_position_target(init_jp)
        # 로봇 베이스
        rp    = ep_data['robot_pose'][0]
        rpose = scene["robot"].data.default_root_state[:, :7].clone()
        rpose[0, 0:3] = torch.tensor(rp[:3])
        rpose[0, 3:7] = torch.tensor(rp[3:7])
        scene["robot"].write_root_pose_to_sim(rpose)
        # 상자: HDF5 첫 프레임 위치 강제 배치
        if "box" in scene.keys():
            _force_box_pose(ep_data['box_pose'][0])

    # ── 헬퍼: 상태 출력 ─────────────────────────────────────────────────────
    def print_status():
        slot_str = f" [슬롯 {slot_id}]" if slot_id > 0 else ""
        spd_str  = f" x{args_cli.speed}" if args_cli.speed != 1.0 else ""
        print(f"\n{'='*60}")
        print(f"  ▶ {ep_name} ({current_ep+1}/{player.num_episodes}){slot_str}{spd_str}")
        print(f"  프레임: {num_samples}개")
        print(f"  조작: N=다음 P=이전 R=다시 Space=일시정지 1~4=슬롯 Q=종료")
        print(f"{'='*60}")

    print_status()
    reset_to_frame0()

    frame_delay  = (1.0 / cfg.STEP_HZ) / args_cli.speed
    last_frame_t = time.time()

    while simulation_app.is_running():
        # ── 키보드 처리 ─────────────────────────────────────────────────────
        key            = kbd.get_key()
        reload_episode = False

        if key == 'q':
            break
        elif key == ' ':
            paused = not paused
            print(f"  {'⏸ 일시정지' if paused else '▶ 재개'}")
        elif key in ('n', 'right'):
            current_ep = min(current_ep + 1, player.num_episodes - 1)
            reload_episode = True
        elif key in ('p', 'left'):
            current_ep = max(current_ep - 1, 0)
            reload_episode = True
        elif key == 'r':
            reload_episode = True
        elif key in ('1', '2', '3', '4'):
            current_ep = player.find_slot_start(int(key))
            reload_episode = True
            print(f"  → 슬롯 {key} 점프!")

        if reload_episode:
            ep_data, num_samples, slot_id, ep_name = player.get_episode(current_ep)
            frame_idx = 0
            print_status()
            reset_to_frame0()
            last_frame_t = time.time()
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(sim_dt)
            continue

        if paused:
            sim.step(render=True)
            scene.update(sim_dt)
            continue

        # 프레임 속도 제어
        now = time.time()
        if now - last_frame_t < frame_delay:
            sim.step(render=True)
            scene.update(sim_dt)
            continue
        last_frame_t = now

        # ── 프레임 재생 ─────────────────────────────────────────────────────
        if frame_idx < num_samples:
            # 1) 관절 텔레포트
            jt = torch.tensor(
                ep_data['actions'][frame_idx], dtype=torch.float32
            ).unsqueeze(0)
            robot.write_joint_state_to_sim(jt, torch.zeros_like(jt))
            robot.set_joint_position_target(jt)

            # 2) 로봇 베이스 이동
            rp = ep_data['robot_pose'][frame_idx]
            rs = scene["robot"].data.default_root_state[:, :7].clone()
            rs[0, 0:3] = torch.tensor(rp[:3])
            rs[0, 3:7] = torch.tensor(rp[3:7] if len(rp) >= 7 else [1, 0, 0, 0])
            scene["robot"].write_root_pose_to_sim(rs)

            # 3) 상자 위치 — HDF5 기록값 직접 주입 (★ 핵심: 물리 무시 강제 배치)
            _force_box_pose(ep_data['box_pose'][frame_idx])

            # 4) 진행률 표시
            if frame_idx % 50 == 0:
                progress = (frame_idx / num_samples) * 100
                filled   = int(30 * frame_idx / num_samples)
                bar      = '█' * filled + '░' * (30 - filled)
                print(f"\r  [{bar}] {progress:5.1f}% ({frame_idx}/{num_samples})", end='', flush=True)

            frame_idx += PLAYBACK_SPEED

        else:
            # 에피소드 완료 → 다음 에피소드 자동 전환
            print(f"\n  ✅ 에피소드 완료!")
            if current_ep < player.num_episodes - 1:
                current_ep += 1
                ep_data, num_samples, slot_id, ep_name = player.get_episode(current_ep)
                frame_idx = 0
                print_status()
                reset_to_frame0()
                last_frame_t = time.time()
            else:
                print("  🏁 모든 에피소드 재생 완료! (R=처음부터, Q=종료)")
                while simulation_app.is_running():
                    k = kbd.get_key()
                    if k == 'q':
                        return
                    if k == 'r':
                        current_ep = 0
                        ep_data, num_samples, slot_id, ep_name = player.get_episode(0)
                        frame_idx = 0
                        print_status()
                        reset_to_frame0()
                        last_frame_t = time.time()
                        break
                    sim.step(render=True)
                    scene.update(sim_dt)
                continue

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


# ============================================================================
# 메인
# ============================================================================
def main():
    # HDF5 파일 존재 확인
    if not os.path.exists(args_cli.hdf5):
        raise FileNotFoundError(f"HDF5 파일 없음: {args_cli.hdf5}")

    usd_path = FFW_SH5_CFG.spawn.usd_path
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SH5 USD 없음: {usd_path}")

    # ── 시뮬레이션 설정 (sh5_bringup_ros2와 동일) ──────────────────────────
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

    # ── 상자 설정 (box_assets 우선, 없으면 기본 큐브) ───────────────────────
    box_usd = _get_box_usd("INITIAL")
    if box_usd:
        print(f"[Scene] 상자 USD: {Path(box_usd).name}")
        box_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.UsdFileCfg(
                usd_path=box_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    linear_damping=0.1, angular_damping=5.0,
                    max_depenetration_velocity=0.3,
                    enable_gyroscopic_forces=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0), rot=(1, 0, 0, 0)),
        )
    else:
        print("[Scene] box_assets 없음 → 기본 큐브 사용")
        box_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.10, 0.10, 0.10),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    linear_damping=0.1, angular_damping=5.0,
                    max_depenetration_velocity=0.3, enable_gyroscopic_forces=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.38, 0.08)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max", static_friction=2.0,
                    dynamic_friction=1.8, restitution=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -10.0), rot=(1, 0, 0, 0)),
        )

    # ── 로봇 설정 ────────────────────────────────────────────────────────────
    robot_cfg = deepcopy(FFW_SH5_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = not args_cli.enable_gravity
    robot_cfg.init_state.pos = cfg.ROBOT_POS

    # ── 씬 조립 ──────────────────────────────────────────────────────────────
    scene_cfg       = ReplaySceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.box   = box_cfg
    scene_cfg.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # TopView 카메라 추가 (sh5_bringup_ros2와 동일한 위치/설정)
    if not args_cli.no_camera:
        scene_cfg.topview_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/TopViewCamera",
            update_period=0.1,
            height=320, width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.5, 0.0, 2.0),
                rot=(0.0, 1.0, 0.0, 0.0),
                convention="world",
            ),
        )

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())

    # 초기 관절 상태 기록
    robot = scene["robot"]
    default_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos))
    robot.set_joint_position_target(default_pos)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    # 로봇 자체 카메라 비활성화 (sh5_bringup_ros2와 동일)
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for cam_path in [
            "/World/envs/env_0/Robot/head_camera",
            "/World/envs/env_0/Robot/left_camera",
            "/World/envs/env_0/Robot/right_camera",
            "/World/envs/env_0/Robot/wrist_camera",
        ]:
            prim = stage.GetPrimAtPath(cam_path)
            if prim and prim.IsValid():
                prim.GetAttribute("visibility").Set("invisible")
        print("[Scene] 📵 로봇 카메라 비활성화 완료")
    except Exception as e:
        print(f"[Scene] 카메라 비활성화 실패 (무시 가능): {e}")

    print("\n" + "=" * 60)
    print("  🎬 replay_data2  —  sh5_bringup_ros2 동일 환경 리플레이")
    print(f"  HDF5: {args_cli.hdf5}")
    print(f"  속도: x{args_cli.speed}  |  중력: {'ON' if args_cli.enable_gravity else 'OFF'}")
    print(f"  N/→: 다음 | P/←: 이전 | R: 다시 | Space: 정지 | 1~4: 슬롯 | Q: 종료")
    print("=" * 60 + "\n")

    player = HDF5Player(args_cli.hdf5)
    try:
        run_replay(sim, scene, player, args_cli.episode)
    finally:
        player.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
