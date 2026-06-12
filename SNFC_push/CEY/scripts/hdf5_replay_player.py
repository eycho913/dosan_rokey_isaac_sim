"""
SH5 HDF5 궤적 재생 서브 플랜 (Trajectory Replay Sub-Plan)
===========================================================
ACT 모델이 불안정할 때의 확실한 Fallback.

[핵심 아이디어]
  1. HDF5의 demo_N/obs/box_pose[0] → 상자 초기 위치를 그 좌표에 정확히 스폰
  2. demo_N/actions[0~N]           → 프레임마다 로봇 관절에 그대로 주입
  → 수집 당시의 완벽한 Expert 궤적을 100% 재현

[데이터 구조 (확인 완료)]
  /data/demo_N/
    obs/
      box_pose       (T, 7)  - 상자 위치/회전 [x,y,z, qw,qx,qy,qz]
      joint_positions (T, 63) - 관절 위치
      rack_pose       (T, 7)  - 작업대 위치/회전
      robot_pose      (T, 7)  - 로봇 베이스 위치
    actions          (T, 63)  - 관절 제어 명령

[슬롯별 HDF5 매핑]
  슬롯 1 (상층 우측) → slot1_1.hdf5 / slot1_2.hdf5
  슬롯 2 (상층 좌측) → slot2_1.hdf5 / slot2_2.hdf5
  슬롯 3 (하층 우측) → slot3_1.hdf5 / slot3_2.hdf5
  슬롯 4 (하층 좌측) → slot4_1.hdf5 / slot4_2.hdf5

실행:
  sh5_integrated.py 에서 import 후 pick_and_place_replay() 호출
  또는 단독 테스트: isaac-python hdf5_replay_player.py
"""

import h5py
import numpy as np
import random
import time
import os
from pathlib import Path

# Isaac Sim 사용 가능 여부 감지
ISAAC_AVAILABLE = False
try:
    import omni.usd
    from pxr import UsdGeom, Sdf, Gf
    ISAAC_AVAILABLE = True
except ImportError:
    pass


# ============================================================
# 슬롯별 HDF5 파일 매핑
# ============================================================
HDF5_BASE_DIR = Path("/home/rokey/dev_ws/datasets/train_data")

SLOT_HDF5_MAP = {
    1: [
        HDF5_BASE_DIR / "slot1_1.hdf5",
        HDF5_BASE_DIR / "slot1_2.hdf5",
    ],
    2: [
        HDF5_BASE_DIR / "slot2_1.hdf5",
        HDF5_BASE_DIR / "slot2_2.hdf5",
    ],
    3: [
        HDF5_BASE_DIR / "slot3_1.hdf5",
        HDF5_BASE_DIR / "slot3_2.hdf5",
    ],
    4: [
        HDF5_BASE_DIR / "slot4_1.hdf5",
        HDF5_BASE_DIR / "slot4_2.hdf5",
    ],
}

# 재생 속도 (실제 수집은 약 20Hz → 0.05s/frame)
REPLAY_FRAME_DT = 0.05   # 초/프레임
REPLAY_SPEED_MULTIPLIER = 1.0  # 1.0=원속, 2.0=2배속

# ============================================================
# SH5 홈(Standby) 포지션 - 컨베이어 앞 대기 자세
# ============================================================
# HDF5 데이터의 첫 프레임(frame[0]) = 로봇 홈 자세
# 실제 컨베이어 앞 대기 자세를 VR 데이터의 첫 프레임으로 정의
# 추중 취합 후 동일한 지점에서 다음 상자를 기다리는 자세
HOME_RETURN_STEPS = 50     # 홈 복귀 보간 스텝 수 (도달 시간: ~2.5초)
HOME_RETURN_DT   = 0.05   # 홈 복귀 프레임 간격


# ============================================================
# HDF5 에피소드 로더
# ============================================================
class HDF5EpisodeLoader:
    """HDF5 파일에서 랜덤 에피소드를 로드합니다."""

    def __init__(self, slot_num: int):
        self.slot_num = slot_num
        self.hdf5_paths = [
            p for p in SLOT_HDF5_MAP.get(slot_num, []) if p.exists()
        ]
        if not self.hdf5_paths:
            raise FileNotFoundError(
                f"슬롯 {slot_num}에 해당하는 HDF5 파일을 찾을 수 없습니다.\n"
                f"탐색 경로: {SLOT_HDF5_MAP.get(slot_num, [])}"
            )
        print(f"[HDF5 Loader] 슬롯 {slot_num}: {len(self.hdf5_paths)}개 파일 발견")

    def load_specific_episode(self, filename_substr: str, demo_key: str) -> dict:
        """특정 파일명(부분 문자열)과 demo_key에 해당하는 에피소드를 로드합니다."""
        target_path = None
        for p in self.hdf5_paths:
            if filename_substr in p.name:
                target_path = p
                break
                
        if not target_path:
            raise FileNotFoundError(f"'{filename_substr}'을 포함하는 HDF5 파일을 찾을 수 없습니다.")

        with h5py.File(target_path, 'r') as f:
            if demo_key not in f['data']:
                raise KeyError(f"파일 {target_path.name} 내에 {demo_key}가 없습니다.")
            
            demo = f['data'][demo_key]
            episode = {
                'source_file': str(target_path),
                'demo_key': demo_key,
                'box_initial_pose': np.array(demo['obs']['box_pose'][0]),
                'rack_initial_pose': np.array(demo['obs']['rack_pose'][0]),
                'robot_initial_pose': np.array(demo['obs']['robot_pose'][0]),
                'joint_trajectory': np.array(demo['actions']),
                'box_trajectory': np.array(demo['obs']['box_pose']),
                'robot_trajectory': np.array(demo['obs']['robot_pose']),
                'total_frames': len(demo['actions']),
            }

        print(f"[HDF5 Loader] 🎯 지정 로드 완료: {target_path.name} / {demo_key}")
        print(f"  총 프레임: {episode['total_frames']}")
        return episode

    def load_random_episode(self) -> dict:
        """랜덤으로 파일과 에피소드를 선택하여 데이터를 반환합니다."""
        hdf5_path = random.choice(self.hdf5_paths)
        with h5py.File(hdf5_path, 'r') as f:
            demo_keys = list(f['data'].keys())
            demo_key = random.choice(demo_keys)
            demo = f['data'][demo_key]

            episode = {
                'source_file': str(hdf5_path),
                'demo_key': demo_key,
                # 상자 초기 위치 (첫 프레임): [x, y, z, qw, qx, qy, qz]
                'box_initial_pose': np.array(demo['obs']['box_pose'][0]),
                # 작업대 위치 (첫 프레임): [x, y, z, qw, qx, qy, qz]
                'rack_initial_pose': np.array(demo['obs']['rack_pose'][0]),
                # 로봇 베이스 위치 (첫 프레임)
                'robot_initial_pose': np.array(demo['obs']['robot_pose'][0]),
                # 전체 관절 궤적: (T, 63)
                'joint_trajectory': np.array(demo['actions']),
                # 전체 상자 궤적 (시각화용): (T, 7)
                'box_trajectory': np.array(demo['obs']['box_pose']),
                # 전체 로봇 베이스 궤적 (모바일 이동 재현): (T, 7)
                'robot_trajectory': np.array(demo['obs']['robot_pose']),
                'total_frames': len(demo['actions']),
            }

        print(f"[HDF5 Loader] ✅ 로드 완료: {hdf5_path.name} / {demo_key}")
        print(f"  상자 초기 위치: {episode['box_initial_pose'][:3]}")
        print(f"  총 프레임: {episode['total_frames']}")
        return episode


# ============================================================
# 궤적 재생 플레이어 (Isaac Sim Script Editor 환경 전용)
# ============================================================
class TrajectoryReplayPlayer:
    """
    HDF5 에피소드를 Isaac Sim에서 재생합니다.
    Script Editor exec() 환경에서 동작합니다.
    """

    def __init__(self, robot_articulation=None, box_prim_path: str = "",
                 robot_world_pos: tuple = None):
        """
        Args:
            robot_articulation: omni.isaac.core.robots.Robot 인스턴스 (SH5)
            box_prim_path: Stage 상의 상자 Prim 경로
            robot_world_pos: 현재 로봇의 월드 좌표 (x, y, z)
                             HDF5 녹화 당시 로봇 위치와의 offset 계산에 사용.
                             None이면 HDF5 절대 좌표 그대로 사용.
        """
        self.robot = robot_articulation
        self.box_prim_path = box_prim_path
        self.robot_world_pos = np.array(robot_world_pos[:3]) if robot_world_pos else None

    def _spawn_box_at_pose(self, box_pose_7d: np.ndarray,
                           recorded_robot_pos: np.ndarray):
        """
        상자를 HDF5 기록 좌표 + 로봇 위치 offset으로 보정하여 배치합니다.

        핵심 원리:
          관절값(actions)은 로봇 내부 기준 → 어느 라인이든 동일
          상자는 항상 '로봇 기준 같은 상대 위치'에 있어야 궤적이 맞음

          offset = 현재_로봇_위치 - 녹화당시_로봇_위치
          보정_상자위치 = HDF5_상자위치 + offset

        box_pose_7d: [x, y, z, qw, qx, qy, qz]  HDF5 녹화 당시 절대 좌표
        recorded_robot_pos: 녹화 당시 로봇 베이스 위치 (x, y, z)
        """
        try:
            from pxr import UsdGeom, Gf
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.box_prim_path)
            if not prim.IsValid():
                print(f"  ⚠️ 상자 Prim 없음: {self.box_prim_path}")
                return False

            hdf5_box = np.array([box_pose_7d[0], box_pose_7d[1], box_pose_7d[2]])

            if self.robot_world_pos is not None:
                # ★ 핵심: 로봇 위치 offset 적용
                offset = self.robot_world_pos - recorded_robot_pos
                corrected = hdf5_box + offset
                x, y, z = corrected[0], corrected[1], corrected[2]
                print(f"  [Replay] 📦 상자 스폰 (offset 보정)")
                print(f"           HDF5원본: ({hdf5_box[0]:.3f}, {hdf5_box[1]:.3f}, {hdf5_box[2]:.3f})")
                print(f"           로봇offset: ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")
                print(f"           보정후:    ({x:.3f}, {y:.3f}, {z:.3f})")
            else:
                # offset 없음 → HDF5 절대 좌표 그대로 (단독 테스트 시)
                x, y, z = hdf5_box[0], hdf5_box[1], hdf5_box[2]
                print(f"  [Replay] 📦 상자 스폰 @ ({x:.3f}, {y:.3f}, {z:.3f}) [offset 없음]")

            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            return True
        except Exception as e:
            print(f"  [Replay] 상자 배치 오류: {e}")
            return False

    def _apply_joint_positions(self, joint_positions: np.ndarray):
        """
        관절 위치를 로봇에 직접 적용합니다.
        
        joint_positions: (63,) 배열 - SH5 전체 관절
        """
        if self.robot is None:
            # 로봇 없는 환경 (테스트 모드)
            return

        try:
            # omni.isaac.core Articulation 방식 (물리 폭발 방지를 위해 target 제어 사용)
            self.robot.set_joint_position_targets(joint_positions)
        except Exception as e:
            print(f"  [Replay] 관절 적용 오류: {e}")

    def _apply_box_pose(self, box_pose_7d: np.ndarray, offset: np.ndarray):
        """
        매 프레임 상자 위치를 HDF5 궤적 + offset으로 강제 지정.

        ★ Magic Snapping 재현 원리:
          VR 녹화 당시 Magic Snapping으로 그리퍼에 붙어 이동한 상자 궤적이
          box_trajectory (T,7)에 그대로 저장되어 있음.
          → 이 궤적을 매 프레임 강제 주입하면 물리 없이 완벽 재현.

        box_pose_7d: HDF5 box_trajectory[frame_idx]  [x,y,z, qw,qx,qy,qz]
        offset:      현재_로봇_위치 - 녹화_로봇_위치  (3D 벡터)
        """
        try:
            from pxr import UsdGeom, Gf
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.box_prim_path)
            if not prim.IsValid():
                return
            # offset 적용된 상자 위치
            x = box_pose_7d[0] + offset[0]
            y = box_pose_7d[1] + offset[1]
            z = box_pose_7d[2] + offset[2]
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        except Exception:
            pass   # 프레임마다 호출되므로 오류 출력 생략

    def _return_to_home(self, home_joints: np.ndarray):
        """
        Pick & Place 완료 후 로봇을 컨베이어 앞 홈(Standby) 자세로 부드럽게 복귀.
        현재 관절 위치 → home_joints 를 HOME_RETURN_STEPS 단계로 보간(Lerp).

        핵심 설계:
          - HDF5 궤적의 첫 프레임(frame[0]) = VR 수집 당시 로봇이
            컨베이어 앞에서 대기하던 정확한 자세
          - 따라서 별도 홈 자세 파라미터 없이 데이터 자체가 기준점
        """
        if self.robot is None:
            return
        try:
            current_joints = self.robot.get_joint_positions()
            if current_joints is None:
                return
            print(f"[Replay] 🏠 홈 복귀 시작 ({HOME_RETURN_STEPS}스텝 / ~{HOME_RETURN_STEPS * HOME_RETURN_DT:.1f}초)")
            for step in range(HOME_RETURN_STEPS + 1):
                alpha = step / HOME_RETURN_STEPS          # 0.0 → 1.0
                interp = (1 - alpha) * current_joints + alpha * home_joints
                self._apply_joint_positions(interp)
                time.sleep(HOME_RETURN_DT)
            print("[Replay] ✅ 홈 복귀 완료 - 컨베이어 대기 자세")
        except Exception as e:
            print(f"[Replay] 홈 복귀 오류: {e}")

    def play_episode(self, episode: dict, realtime: bool = True) -> bool:
        """
        에피소드를 처음부터 끝까지 재생한 후 홈 자세로 복귀.

        로봇 위치 offset 자동 적용:
          녹화 당시 로봇 위치(episode['robot_initial_pose'][:3])와
          현재 로봇 위치(self.robot_world_pos)의 차이를 상자 위치에 더함.
          → 3개 라인 모두 동일한 HDF5 데이터로 동작 가능.

        Args:
            episode: HDF5EpisodeLoader.load_random_episode()의 반환값
            realtime: True면 실제 속도로 재생, False면 최대한 빠르게

        Returns:
            bool: 재생 성공 여부
        """
        print(f"\n[Replay] 🎬 궤적 재생 시작: {episode['demo_key']}")
        print(f"  총 {episode['total_frames']} 프레임 재생")

        # 1. 녹화 당시 로봇 위치 추출
        recorded_robot_pos = episode['robot_initial_pose'][:3]
        if self.robot_world_pos is not None:
            offset = self.robot_world_pos - recorded_robot_pos
            print(f"  [Replay] 🤖 로봇 위치 offset: ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")
        else:
            offset = np.zeros(3)
            print(f"  [Replay] ⚠️ robot_world_pos 없음 → offset=0 (절대좌표 사용)")

        # 2. 상자를 offset 보정된 위치에 배치
        self._spawn_box_at_pose(episode['box_initial_pose'], recorded_robot_pos)
        time.sleep(0.3)  # 물리 엔진 안정화 대기

        # 3. 홈 자세 = 궤적의 첫 프레임 (VR 수집 당시 컨베이어 앞 대기 자세)
        home_joints = episode['joint_trajectory'][0].copy()

        # 4. 관절 궤적 + 상자 궤적 동시 재생 (★ Magic Snapping 재현)
        #    관절값: 로봇 내부 기준 → offset 불필요
        #    상자 위치: HDF5 궤적 + robot offset → 물리 없이 그리퍼 추종
        trajectory   = episode['joint_trajectory']   # (T, 63)
        box_traj     = episode['box_trajectory']      # (T, 7)
        dt = REPLAY_FRAME_DT / REPLAY_SPEED_MULTIPLIER

        for frame_idx, (joint_cmd, box_pose) in enumerate(
                zip(trajectory, box_traj)):

            # ① 관절값 주입
            self._apply_joint_positions(joint_cmd)

            # ② 상자 위치 강제 주입 (Magic Snapping 효과)
            self._apply_box_pose(box_pose, offset)

            # 진행 상황 출력 (10% 단위)
            if frame_idx % max(1, episode['total_frames'] // 10) == 0:
                pct = (frame_idx / episode['total_frames']) * 100
                print(f"  [Replay] {pct:5.1f}% ({frame_idx}/{episode['total_frames']})")

            if realtime:
                time.sleep(dt)

        print(f"[Replay] ✅ 궤적 재생 완료!")

        # 5. 픽앤플레이스 완료 → 컨베이어 앞 홈 자세로 복귀
        self._return_to_home(home_joints)
        return True


# ============================================================
# 메인 인터페이스 함수 (sh5_integrated.py에서 호출)
# ============================================================
def pick_and_place_replay(
    slot_num: int,
    robot_articulation=None,
    box_prim_path: str = "",
    realtime: bool = True,
    robot_world_pos: tuple = None,
) -> bool:
    """
    HDF5 데이터 기반 Pick & Place 재생.

    Args:
        slot_num: 목표 슬롯 번호 (1~4)
        robot_articulation: SH5 Robot Articulation 객체
        box_prim_path: Stage 상 상자 Prim 경로
        realtime: 실시간 재생 여부
        robot_world_pos: 현재 로봇의 월드 좌표 (x, y, z)
            ★ 핵심: 이 값을 넘겨야 3개 라인 모두에서 올바른 위치에 상자 배치됨
            None이면 HDF5 절대 좌표 그대로 사용 (sg2_in_01 전용 단독 테스트)

    Returns:
        bool: 성공 여부
    """
    try:
        loader = HDF5EpisodeLoader(slot_num=slot_num)
        episode = loader.load_random_episode()
        player = TrajectoryReplayPlayer(
            robot_articulation=robot_articulation,
            box_prim_path=box_prim_path,
            robot_world_pos=robot_world_pos,   # ★ offset 전달
        )
        return player.play_episode(episode, realtime=realtime)
    except FileNotFoundError as e:
        print(f"[Replay] ❌ HDF5 파일 없음: {e}")
        return False
    except Exception as e:
        print(f"[Replay] ❌ 재생 오류: {e}")
        import traceback; traceback.print_exc()
        return False


def get_box_spawn_position(slot_num: int) -> tuple:
    """
    HDF5에서 상자의 초기 좌표만 읽어서 반환합니다.
    sh5_integrated.py의 상자 스폰 위치 결정에 사용합니다.
    
    Returns:
        (x, y, z) 상자 초기 위치
    """
    try:
        loader = HDF5EpisodeLoader(slot_num=slot_num)
        episode = loader.load_random_episode()
        pos = episode['box_initial_pose'][:3]
        return float(pos[0]), float(pos[1]), float(pos[2])
    except Exception as e:
        print(f"[Replay] get_box_spawn_position 오류: {e}")
        # Fallback: 기본 컨베이어 위치
        return (0.7, 0.0, 1.0)


# ============================================================
# 단독 실행 테스트 (isaac-python hdf5_replay_player.py)
# ============================================================
if __name__ == "__main__":
    print("=== HDF5 궤적 재생 테스트 (단독 실행) ===")
    print()

    for slot in [1, 2, 3, 4]:
        print(f"--- 슬롯 {slot} 테스트 ---")
        pos = get_box_spawn_position(slot)
        print(f"  상자 스폰 위치: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    print()
    print("슬롯 1 에피소드 로드 테스트:")
    success = pick_and_place_replay(
        slot_num=1,
        robot_articulation=None,   # 단독 테스트 시 None
        box_prim_path="",          # 단독 테스트 시 빈 문자열
        realtime=False,            # 빠르게 로드만 테스트
    )
    print(f"결과: {'✅ 성공' if success else '❌ 실패'}")
