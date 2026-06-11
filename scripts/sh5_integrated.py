"""
SH5 통합 물류 컨트롤러 - Script Editor 버전
==============================================
실행 방법 (AMR 팀과 동일한 방식):
  Isaac Sim 실행 후 → Window → Script Editor
  exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_integrated.py', encoding='utf-8').read())

구조:
  - 이미 열려있는 final_coupan.usd Stage 위에서 SH5 로봇 3대 제어
  - AMR 팀과 동일한 Script Editor exec() 아키텍처 사용
  - ACT 모델 완성 전: Dummy Teleport로 Pick & Place 수행
  - ACT 모델 완성 후: --use_act 플래그로 전환 가능
  
참조 레포:
  - AMR 팀: /home/rokey/dev_ws/coupang_ws/scripts/amr_live_existing_stage_true8_qr_camera_controller_gpu.py
  - DB 인터페이스: /home/rokey/dev_ws/cobot3_ws_ref/src/cobot3_interfaces/

핵심 설계:
  - 로봇 베이스 기준 상대 좌표계(Local Coordinate)로 AI 모델 입력 → 어느 위치든 동일한 모델 재사용 가능
  - 각 로봇의 컨베이어 벨트와 작업대의 상대 거리를 학습 환경과 동일하게 유지
"""

import json
import math
import os
import random
import threading
import time
from pathlib import Path
from enum import Enum, auto

# ============================================================
# Isaac Sim 연결 (Script Editor 환경에서는 이미 초기화되어 있음)
# ============================================================
try:
    import omni.usd
    import omni.kit.app
    from pxr import Usd, UsdGeom, Gf, Sdf
    ISAAC_AVAILABLE = True
    print("[SH5] ✅ Isaac Sim 연결 성공")
except ImportError:
    ISAAC_AVAILABLE = False
    print("[SH5] ⚠️ Isaac Sim 환경 외부에서 실행 중 (디버그 모드)")

# ============================================================
# ROS 2 연결
# ============================================================
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, String
    from cobot3_interfaces.srv import (
        CheckWarehouseStatus,
        ReportInboundProgress,
        GetDailyPackageList,
    )
    from cobot3_interfaces.action import ManageWorkstation, MovePackage
    from rclpy.action import ActionClient
    ROS2_AVAILABLE = True
    print("[SH5] ✅ ROS 2 연결 성공")
except ImportError:
    ROS2_AVAILABLE = False
    print("[SH5] ⚠️ ROS 2 없음 - 로컬 시뮬레이션 모드로 동작합니다.")

# ============================================================
# Pick & Place 모드 선택 (3가지 중 하나 선택)
# ============================================================
#  "DUMMY_TELEPORT" : 상자 순간이동 (가장 빠른 테스트용)
#  "HDF5_REPLAY"    : VR 수집 궤적 재생 (서브 플랜 ★ 내일 데모 보험)
#  "ACT_MODEL"      : AI 추론 (모델 완성 후)
PICK_AND_PLACE_MODE = "HDF5_REPLAY"   # ← 여기서 모드 선택!

ACT_MODEL_PATH = "/home/rokey/dev_ws/models/augmented_sh5_vision_act_20ep.pth"

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# HDF5 재생 모듈 임포트
try:
    import sys as _sys
    if '/home/rokey/dev_ws/coupang_ws/scripts' not in _sys.path:
        _sys.path.insert(0, '/home/rokey/dev_ws/coupang_ws/scripts')
    from hdf5_replay_player import pick_and_place_replay, get_box_spawn_position
    HDF5_REPLAY_AVAILABLE = True
    print("[SH5] ✅ HDF5 재생 모듈 로드 완료")
except ImportError as e:
    HDF5_REPLAY_AVAILABLE = False
    print(f"[SH5] ⚠️ HDF5 재생 모듈 로드 실패: {e}")


# ============================================================
# 통합 환경 설정 (final_coupan.usd 기준 - PHYSICAL_LAYOUT.md 공식 좌표 적용)
# ============================================================
# 각 라인의 로봇(SH5) 배치 절대 좌표 (World)
ROBOT_POSITIONS = [
    (7.5,  3.0),   # 1번 라인 (좌측, sg2_in_01)
    (7.5, -1.5),   # 2번 라인 (중앙, sg2_in_02)
    (7.5, -6.0),   # 3번 라인 (우측, sg2_in_03)
]
# ============================================================
# HDF5 분석으로 확인된 실제 상자 스폰 범위 (학습 환경 기준)
# ============================================================
# python3 분석 결과 (2026-06-10)
#   slot1: mean=(0.723, -0.008, 0.817)  std=(0.026, 0.107, 0.006)
#   slot2: mean=(0.784, -0.047, 0.815)  std=(0.094, 0.088, 0.010)
#   slot3: mean=(0.770, -0.036, 0.813)  std=(0.130, 0.093, 0.013)
#   slot4: mean=(0.740, -0.053, 0.815)  std=(0.076, 0.083, 0.007)
# 상자 스폰 X 범위: 0.63 ~ 0.97m (로봇 기준 상대 좌표)
# 상자 스폰 Z 범위: 0.81 ~ 0.83m (컨베이어 높이 고정)
HDF5_BOX_SPAWN_MEAN = (0.756, -0.036, 0.815)   # 4슬롯 평균값 (로봇 기준)
HDF5_BOX_SPAWN_X_RANGE = (0.62, 0.97)          # 로봇 기준 X 범위
HDF5_BOX_SPAWN_Y_RANGE = (-0.20, 0.15)         # 로봇 기준 Y 범위
HDF5_BOX_Z_HEIGHT = 0.815                       # 컨베이어 높이 (고정)
# 학습 환경에서의 슬롯 목표 좌표 (로봇 기준 상대 좌표, 단위: m)
SLOT_TARGETS_LOCAL = {
    1: (0.0, -1.5, 1.2),   # 상층 우측
    2: (0.0, -1.5, 1.2),   # 상층 좌측 (좌우 오프셋 추가 필요)
    3: (0.0, -1.5, 0.5),   # 하층 우측
    4: (0.0, -1.5, 0.5),   # 하층 좌측
}

# SH5 로봇 USD 경로
SH5_USD_PATH = "/home/rokey/dev_ws/assets/sh5.usd"

# ============================================================
# 상태 머신
# ============================================================
class WorkstationState(Enum):
    IDLE       = auto()
    SCANNING   = auto()
    AMR_CALL   = auto()
    ALLOCATE   = auto()
    PLACING    = auto()
    WAIT_REFRESH = auto()


# ============================================================
# Stage Prim 유틸리티 (Script Editor 전용)
# ============================================================
def get_stage():
    if not ISAAC_AVAILABLE:
        return None
    return omni.usd.get_context().get_stage()


def get_prim(path: str):
    stage = get_stage()
    if stage is None:
        return None
    return stage.GetPrimAtPath(path)


def set_prim_world_pose(prim_path: str, pos: tuple, rot_quat_wxyz: tuple = (1, 0, 0, 0)):
    """지정된 Prim의 월드 위치/회전을 설정합니다."""
    stage = get_stage()
    if stage is None:
        print(f"  [Teleport] Stage 없음 - 좌표 {pos} 적용 스킵")
        return False
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"  [Teleport] ⚠️ Prim 없음: {prim_path}")
        return False
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
    return True


def spawn_box_prim(unit_idx: int, package_id: str, world_pos: tuple) -> str:
    """
    컨베이어 벨트 위에 상자 Prim을 동적으로 생성합니다.
    DB 팀의 generate_sh5_boxes.py가 만든 USD 에셋을 참조합니다.
    """
    stage = get_stage()
    if stage is None:
        return ""

    # DB 팀이 생성한 QR USD 에셋 경로
    qr_box_usd = os.path.expanduser(
        f"~/cobot3_ws/scratch/box_assets/PKG_{package_id.replace('QR_', '')}.usd"
    )
    fallback_box_usd = "/home/rokey/dev_ws/assets/belt.usd"  # 없으면 기본 상자 사용
    usd_to_use = qr_box_usd if os.path.exists(qr_box_usd) else fallback_box_usd

    prim_path = f"/World/SH5_Box_{unit_idx:02d}_{package_id.replace('-', '_')}"

    # 기존 Prim 제거 후 재생성
    existing = stage.GetPrimAtPath(prim_path)
    if existing.IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))

    box_prim = stage.DefinePrim(prim_path, "Xform")
    box_prim.GetReferences().AddReference(usd_to_use)

    # 위치 설정
    xform = UsdGeom.Xformable(box_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*world_pos))

    print(f"  [Spawn] 📦 상자 스폰: {prim_path} @ {world_pos}")
    return prim_path


def teleport_box_to_slot(box_prim_path: str, robot_world_pos: tuple, slot_num: int) -> bool:
    """
    Dummy Teleport: 상자를 목표 슬롯의 월드 좌표로 즉시 이동.
    로봇 기준 상대 좌표(SLOT_TARGETS_LOCAL)를 월드 좌표로 변환합니다.
    """
    local_target = SLOT_TARGETS_LOCAL.get(slot_num, SLOT_TARGETS_LOCAL[1])
    world_target = (
        robot_world_pos[0] + local_target[0],
        robot_world_pos[1] + local_target[1],
        local_target[2],  # Z는 절대값 사용
    )
    print(f"  [Dummy Teleport] 🚀 슬롯 {slot_num} 로컬{local_target} → 월드{world_target}")
    return set_prim_world_pose(box_prim_path, world_target)

# ============================================================
# SH5 로봇 Articulation 로더
# ============================================================
def load_sh5_robot(unit_idx: int, world_pos: tuple):
    """
    Isaac Sim Stage에서 SH5 로봇 Articulation 객체를 로드합니다.

    두 가지 데이스:
      A) final_coupan.usd에 이미 SH5 Prim이 있는 경우 → 경로만 지정하면 됨
      B) 없는 경우 → SH5_USD_PATH에서 신규 스폰

    Returns:
      omni.isaac.core.robots.Robot 또는 None
    """
    if not ISAAC_AVAILABLE:
        print(f"  [Articulation] Isaac Sim 없음 - 로봇 로드 스킵")
        return None

    try:
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from pxr import UsdGeom, Gf

        stage = get_stage()

        # 예상 Prim 경로 (final_coupan.usd 실측 후 수정 필요)
        # Case A: 이미 있는 Prim 경로 (AMR팀 확인 후 입력)
        existing_paths = [
            f"/World/SH5_{unit_idx:02d}",        # 추정 1
            f"/World/sh5_{unit_idx:02d}",        # 추정 2
            f"/World/SH5_Robot_{unit_idx:02d}",  # 추정 3
        ]
        prim_path = None
        for ep in existing_paths:
            if stage.GetPrimAtPath(ep).IsValid():
                prim_path = ep
                print(f"  [Articulation] 기존 SH5 Prim 발견: {prim_path}")
                break

        # Case B: Prim 없으면 실제 SH5 USD로 시내 스폰
        if prim_path is None:
            prim_path = f"/World/SH5_{unit_idx:02d}"
            print(f"  [Articulation] SH5 Prim 없음 → 신규 스폰: {prim_path}")
            if not os.path.exists(SH5_USD_PATH):
                print(f"  [Articulation] ⚠️ SH5 USD 없음: {SH5_USD_PATH}")
                return None
            add_reference_to_stage(usd_path=SH5_USD_PATH, prim_path=prim_path)

            # 위치 설정
            prim = stage.GetPrimAtPath(prim_path)
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*world_pos))

        # Articulation 래핑
        robot = Robot(prim_path=prim_path, name=f"SH5_{unit_idx:02d}")
        robot.initialize()
        print(f"  [Articulation] ✅ SH5_{unit_idx:02d} Articulation 완료! DOF={robot.num_dof}")
        return robot

    except Exception as e:
        print(f"  [Articulation] ❌ 로드 실패: {e}")
        import traceback; traceback.print_exc()
        return None


class SH5WorkUnit:
    def __init__(self, unit_idx: int, conveyor_xy: tuple, db_node=None):
        self.unit_idx = unit_idx
        # ControlTowerNode가 sg2_in_* 네임스페이스로 Pause 발행 → 반드시 일치시켜야 함
        self.robot_id = f"sg2_in_{unit_idx:02d}"
        self.conveyor_xy = conveyor_xy  # 컨베이어 끝 월드 좌표

        # 로봇 월드 좌표 = 컨베이어 끝 + 오프셋
        self.robot_world_pos = (
            conveyor_xy[0] + ROBOT_OFFSET_X,
            conveyor_xy[1] + ROBOT_OFFSET_Y,
            0.0,
        )
        # 상자 스폰 위치 = 컨베이어 끝 위
        self.box_spawn_pos = (
            conveyor_xy[0] + BOX_SPAWN_OFFSET[0],
            conveyor_xy[1] + BOX_SPAWN_OFFSET[1],
            BOX_SPAWN_OFFSET[2],
        )

        self.db = db_node
        self.state = WorkstationState.SCANNING
        self.slots = {1: None, 2: None, 3: None, 4: None}  # None=빈슬롯
        self.current_box_prim = ""
        self.current_customer = ""
        self.current_package_id = ""
        self.current_qr_id = ""
        self.robot = None  # load_sh5_robot()으로 나중에 채움 (Isaac Sim 실행 후)

        # /sim/sg2_spawn_trigger 토픽에서 수신한 패키지 대기큐 (thread-safe)
        import queue as _queue
        self.pending_packages: _queue.Queue = _queue.Queue()

        print(f"[Unit {unit_idx}] 🤖 {self.robot_id} 초기화 완료")
        print(f"  로봇 기준 위치: {self.robot_world_pos}")
        print(f"  안전 스폰 보정: {self.box_spawn_pos}")

    def _find_free_slot(self):
        for slot_num, owner in self.slots.items():
            if owner is None:
                return slot_num
        return None

    def _mock_scan_qr(self):
        """QR 스캔 시뮬레이션 (실제 카메라 연결 전 가상 데이터)"""
        # TODO: Top-View 카메라 Prim에서 실제 이미지를 받아 OpenCV QR 검출로 교체
        qr_ids = [
            "QR_20260612_001", "QR_20260612_002",
            "QR_20260612_003", "QR_20260612_004",
        ]
        qr_id = random.choice(qr_ids)
        # 실제 카메라 기반 좌표 대신 스폰 위치 반환
        box_x = self.box_spawn_pos[0] + random.uniform(-0.05, 0.05)
        box_y = self.box_spawn_pos[1] + random.uniform(-0.05, 0.05)
        return qr_id, (box_x, box_y)

    def step(self):
        """상태 머신 1 스텝 실행"""
        # Pause 인터락
        if self.db and hasattr(self.db, 'is_paused') and self.db.is_paused:
            return

        if self.state == WorkstationState.SCANNING:
            self._step_scanning()
        elif self.state == WorkstationState.AMR_CALL:
            self._step_amr_call()
        elif self.state == WorkstationState.ALLOCATE:
            self._step_allocate()
        elif self.state == WorkstationState.WAIT_REFRESH:
            self._step_wait_refresh()

    def _step_scanning(self):
        """
        /sim/sg2_spawn_trigger 토픽에서 도착한 패키지를 우선 처리.
        토픽 데이터 없으면 Mock QR 스캔으로 폴백.
        """
        print(f"\n[Unit {self.unit_idx}] --- 상자 도착 대기 ---")

        # 1. SimSyncNode에서 실제 패키지 도착 여부 확인 (0.5초 대기)
        try:
            payload = self.pending_packages.get(timeout=0.5)
            package_id = payload.get('package_id', '')
            qr_id = f"QR_{package_id.replace('PKG_', '')}"
            print(f"[Unit {self.unit_idx}] 📦 SimSync 도착: {package_id} (QR: {qr_id})")
        except Exception:
            # 2. 토픽 데이터 없으면 Mock QR 스캔으로 폴백 (테스트용)
            qr_id, _ = self._mock_scan_qr()
            package_id = qr_id.replace("QR_", "PKG_")
            print(f"[Unit {self.unit_idx}] 📷 Mock QR 폴백: {qr_id}")

        box_xy = (
            self.box_spawn_pos[0] + random.uniform(-0.03, 0.03),
            self.box_spawn_pos[1] + random.uniform(-0.03, 0.03),
        )
        print(f"[Unit {self.unit_idx}] 상자 스폰 위치: {box_xy}")

        # 상자 스폰
        self.current_qr_id = qr_id
        self.current_package_id = package_id
        self.current_customer = f"고객_{qr_id[-3:]}"  # 실제로는 DB 에서 연동

        self.current_box_prim = spawn_box_prim(
            self.unit_idx, qr_id, self.box_spawn_pos
        )

        # DB - 중복 검사
        other_ws = None
        if self.db:
            other_ws = self._db_check_warehouse()

        if other_ws:
            print(f"[Unit {self.unit_idx}] ⚠️ 중복 보관 감지 → AMR 회수 요청")
            self.state = WorkstationState.AMR_CALL
        else:
            self.state = WorkstationState.ALLOCATE

    def _step_amr_call(self):
        """
        [AMR_CALL 상태] 중복 입고 감지 시:
        AMR에게 MovePackage.action으로 택배 직송 명령을 보냄.
        화면에 AMR의 실시간 위치/진행률이 피드백으로 표시됨.
        """
        print(f"[Unit {self.unit_idx}] 🚛 AMR 회수 요청: {self.current_package_id} ({self.current_customer})")

        if self.db:
            ok = self.db.send_move_package_action(
                package_id      = self.current_package_id,
                customer_name   = self.current_customer,
                package_qr_id   = self.current_qr_id,
                destination_zone= "MAIN_storage",
            )
            if ok:
                print(f"[Unit {self.unit_idx}] ✅ AMR가 {self.current_package_id} 회수 완료")
            else:
                print(f"[Unit {self.unit_idx}] ⚠️ AMR 회수 실패 - 상자 삭제 후 재스캔")
        else:
            print(f"[Unit {self.unit_idx}] ⚠️ DB 없음 - 로컈 Mock 회수 처리")
            time.sleep(1.0)

        self.current_box_prim = ""
        self.state = WorkstationState.SCANNING

    def _step_allocate(self):
        target_slot = self._find_free_slot()

        if target_slot is None:
            print(f"[Unit {self.unit_idx}] ⚠️ 슬롯 만석! 작업대 갱신 요청")
            self.state = WorkstationState.WAIT_REFRESH
            return

        print(f"[Unit {self.unit_idx}] ✅ 슬롯 {target_slot} 할당 → Pick & Place 시작")
        self.slots[target_slot] = self.current_customer

        # ── Pick & Place (모드별 분기) ────────────────────────
        if PICK_AND_PLACE_MODE == "ACT_MODEL" and TORCH_AVAILABLE:
            self._run_act_inference(target_slot)
        elif PICK_AND_PLACE_MODE == "HDF5_REPLAY" and HDF5_REPLAY_AVAILABLE:
            print(f"[Unit {self.unit_idx}] 🎬 HDF5 궤적 재샛 모드 (슬롯 {target_slot})")
            success = pick_and_place_replay(
                slot_num=target_slot,
                robot_articulation=self.robot,   # ← Articulation 연결! (None이면 관절 스킵)
                box_prim_path=self.current_box_prim,
                realtime=True,
            )
            if success:
                print(f"[Unit {self.unit_idx}] ✅ HDF5 재생 완료!")
            else:
                print(f"[Unit {self.unit_idx}] ❌ HDF5 재생 실패 → Dummy Teleport로 폴백")
                teleport_box_to_slot(self.current_box_prim, self.robot_world_pos, target_slot)
        else:
            # DUMMY_TELEPORT (기본 폴백)
            success = teleport_box_to_slot(
                self.current_box_prim, self.robot_world_pos, target_slot
            )
            if success:
                print(f"[Unit {self.unit_idx}] ✅ Dummy Teleport 완료!")
        # ─────────────────────────────────────────────────────

        # DB - 적재 완료 보고
        if self.db:
            filled = sum(1 for v in self.slots.values() if v is not None)
            self._db_report_progress(filled, target_slot)

        self.current_box_prim = ""
        self.state = WorkstationState.SCANNING

    def _step_wait_refresh(self):
        """
        [슬롯 만석 상태] 4칸이 모두 참았을 때:
        ManageWorkstation.action으로 AMR에게 작업대 이동을 요청.
        화면에 남은 거리/상태가 피드백으로 표시됨.
        (ControlTower가 자동 처리하는 경우는 호출 불필요)
        """
        ws_id = f"WS_{self.unit_idx:02d}"
        print(f"[Unit {self.unit_idx}] 🏭 작업대 교체 요청: {ws_id}")

        if self.db:
            ok = self.db.send_manage_workstation_action(
                workstation_id = ws_id,
                unit_idx       = self.unit_idx,
                target_location= "warehouse",
            )
            if ok:
                print(f"[Unit {self.unit_idx}] ✅ 작업대 교체 완료 - 새 작업대 대기")
            else:
                print(f"[Unit {self.unit_idx}] ⚠️ AMR 응답 없음 - 3초 후 로켈 리셋")
                time.sleep(3.0)
        else:
            print(f"[Unit {self.unit_idx}] ⚠️ DB 없음 - 3초 후 로켈 슬롯 리셋")
            time.sleep(3.0)

        self.slots = {1: None, 2: None, 3: None, 4: None}
        print(f"[Unit {self.unit_idx}] ✅ 슬롯 초기화 - 다음 상자 대기")
        self.state = WorkstationState.SCANNING

    def _db_check_warehouse(self):
        try:
            if not self.db.check_status_client.wait_for_service(timeout_sec=0.5):
                return None
            req = CheckWarehouseStatus.Request()
            req.customer_name = self.current_customer
            req.package_id = self.current_package_id
            req.qr_id = self.current_qr_id
            future = self.db.check_status_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, future, timeout_sec=1.0)
            if future.result() and future.result().is_already_in_warehouse:
                return "other_workstation"
        except Exception as e:
            print(f"[DB] CheckWarehouseStatus 오류: {e}")
        return None

    def _db_report_progress(self, filled_count: int, slot_num: int):
        try:
            if not self.db.report_client.wait_for_service(timeout_sec=0.5):
                print(f"[DB] ReportInboundProgress 서비스 없음, 로컬만 반영")
                return
            req = ReportInboundProgress.Request()
            req.workstation_id = f"WS_{self.unit_idx:02d}"
            req.robot_id = self.robot_id
            req.filled_slots_count = filled_count
            req.package_id = self.current_package_id
            req.workstation_qr_id = f"WORKSTATION_WS_{self.unit_idx:02d}"
            req.package_qr_id = self.current_qr_id
            future = self.db.report_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, future, timeout_sec=1.0)
            if future.result() and future.result().success:
                print(f"[DB] ✅ 슬롯 {slot_num} 적재 보고 완료")
        except Exception as e:
            print(f"[DB] ReportInboundProgress 오류: {e}")

    def _run_act_inference(self, target_slot: int):
        """
        ACT 모델 추론 (모델 완성 후 USE_ACT_MODEL = True 로 활성화)
        evaluate_test_vision.py의 run_simulator() 로직 참조
        """
        print(f"[Unit {self.unit_idx}] 🤖 ACT 모델 추론 시작 (슬롯 {target_slot})")
        # TODO: evaluate_test_vision.py의 추론 루프를 여기에 통합
        # 1. 로봇 관절 상태 읽기 (Articulation)
        # 2. 카메라 이미지 읽기 (Top/Left/Right)
        # 3. 상자 좌표 계산 (로봇 기준 상대 좌표로 변환!)
        # 4. ACT 모델 forward pass
        # 5. 관절 명령 적용
        pass


# ============================================================
# ROS 2 DB 노드 (cobot3_ws_ref 인터페이스 기반)
# ============================================================
class SH5DBNode(Node):
    def __init__(self, units_ref: list = None):
        super().__init__('sh5_integrated_db_client')
        self.is_paused = False
        self.units_ref = units_ref or []  # 토픽 라우팅을 위한 작업단위 상호 참조

        # 라인 이름 → Unit 인덱스 매핑
        # sg2_in_01 → unit_idx=1, sg2_in_02 → unit_idx=2, ...
        self.line_to_unit_idx = {
            f"sg2_in_{i:02d}": i for i in range(1, len(self.units_ref) + 1)
        }

        self.check_status_client = self.create_client(
            CheckWarehouseStatus, '/check_warehouse_status'
        )
        self.report_client = self.create_client(
            ReportInboundProgress, '/report_inbound_progress'
        )

        # ―――――――――――――――――――――――――――――――――――――――――――――――――――――
        # Action 클라이언트 (중복 감지 AMR 회수 + 만석 작업대 교체)
        # ―――――――――――――――――――――――――――――――――――――――――――――――――――――
        self.move_pkg_client = ActionClient(
            self, MovePackage, '/move_package'
        )
        self.manage_ws_client = ActionClient(
            self, ManageWorkstation, '/manage_workstation'
        )
        # Pause 구독 — ControlTowerNode을 /sg2_in_*/pause_status 토픽으로 발행
        # 단일 통합 토픽으로 3대 모두 커버
        self.pause_subs = []
        for i in range(1, 4):
            sub = self.create_subscription(
                Bool,
                f'/sg2_in_{i:02d}/pause_status',
                lambda msg, ui=i: self._pause_cb(msg, ui),
                10
            )
            self.pause_subs.append(sub)

        # /sim/sg2_spawn_trigger — SimSyncNode가 발행하는 상자 소환 명령 수신
        self.spawn_trigger_sub = self.create_subscription(
            String,
            '/sim/sg2_spawn_trigger',
            self._spawn_trigger_cb,
            10
        )
        print("[DB Node] ✅ SH5 DB 클라이언트 노드 초기화 완료")
        print(f"[DB Node] 📬 Pause 컵 대상: /sg2_in_01~0{len(self.units_ref)}/pause_status")
        print("[DB Node] 📦 /sim/sg2_spawn_trigger 구독 시작")
        print("[DB Node] 🚛 MovePackage.action  서버 대기 중: /move_package")
        print("[DB Node] 🏭 ManageWorkstation.action 서버 대기 중: /manage_workstation")

    # ――――――――――――――――――――――――――――――――――――――――――――――――――
    # Action 클라이언트: MovePackage (중복 감지 시 AMR에 택배 회수 요청)
    # ――――――――――――――――――――――――――――――――――――――――――――――――――
    def send_move_package_action(
        self,
        package_id: str,
        customer_name: str,
        package_qr_id: str,
        destination_zone: str = "MAIN_storage",
    ) -> bool:
        """
        MovePackage.action: AMR에게 택배 직송 명령.
        주로 중복 감지 시('AMR_CALL' 상태) 호출.

        Goal:
          package_id, customer_name, destination_zone, package_qr_id
        Feedback (AMR 실시간 위치 / 진행률):
          current_position, progress (0.0~100.0%)
        Result:
          success, error_msg
        """
        if not self.move_pkg_client.wait_for_server(timeout_sec=2.0):
            print("[Action] MovePackage 서버 없음 - AMR 회수 스킵")
            return False

        goal = MovePackage.Goal()
        goal.package_id       = package_id
        goal.customer_name    = customer_name
        goal.destination_zone = destination_zone
        goal.package_qr_id    = package_qr_id

        print(f"[Action] 🚛 MovePackage 요청: {package_id} → {destination_zone}")

        def _feedback_cb(fb):
            print(f"  [AMR] 현재위치: {fb.feedback.current_position} | 진행: {fb.feedback.progress:.1f}%")

        future = self.move_pkg_client.send_goal_async(
            goal, feedback_callback=_feedback_cb
        )
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            print("[Action] ❌ MovePackage Goal 거절")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        result = result_future.result()

        if result and result.result.success:
            print(f"[Action] ✅ MovePackage 완료: {package_id}")
            return True
        else:
            err = result.result.error_msg if result else "timeout"
            print(f"[Action] ❌ MovePackage 실패: {err}")
            return False

    # ――――――――――――――――――――――――――――――――――――――――――――――――――
    # Action 클라이언트: ManageWorkstation (만석 시 AMR에 작업대 교체 요청)
    # ――――――――――――――――――――――――――――――――――――――――――――――――――
    def send_manage_workstation_action(
        self,
        workstation_id: str,
        unit_idx: int,
        target_location: str = "warehouse",
    ) -> bool:
        """
        ManageWorkstation.action: AMR에게 작업대 통잸 이동 명령.
        주로 슬롯 4칸 만석('전체 완료이지만 AMR 직접 요청') 시 사용.
        (ControlTower가 자동 처리하는 경우는 호출 불필요)

        Goal: workstation_id, start_location, target_location,
              workstation_qr_id, target_x, target_y, target_yaw
        Feedback: distance_remaining, status ("PICKING"/"NAVIGATING"/"PLACING")
        Result: success
        """
        if not self.manage_ws_client.wait_for_server(timeout_sec=2.0):
            print("[Action] ManageWorkstation 서버 없음 - 로켈 슬롯 리셋으로 대체")
            return False

        # 한 라인당 작업대 1대 전제: 찌당 입고 위치(sg2_in_0X_A)로 출발
        goal = ManageWorkstation.Goal()
        goal.workstation_id     = workstation_id
        goal.start_location     = f"sg2_in_{unit_idx:02d}_A"
        goal.target_location    = target_location
        goal.workstation_qr_id  = f"WORKSTATION_WS_{unit_idx:02d}"
        goal.target_qr_id       = ""
        # 한진 접소지 좌표: PHYSICAL_LAYOUT.md 기준 스포트 1번
        goal.target_x   = 1.5
        goal.target_y   = 3.0
        goal.target_yaw = 0.0

        print(f"[Action] 🏭 ManageWorkstation 요청: {workstation_id} → {target_location}")

        def _feedback_cb(fb):
            print(f"  [AMR] 남은 거리: {fb.feedback.distance_remaining:.2f}m | 상태: {fb.feedback.status}")

        future = self.manage_ws_client.send_goal_async(
            goal, feedback_callback=_feedback_cb
        )
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            print("[Action] ❌ ManageWorkstation Goal 거절")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=120.0)
        result = result_future.result()

        if result and result.result.success:
            print(f"[Action] ✅ ManageWorkstation 완료: {workstation_id}")
            return True
        else:
            print(f"[Action] ❌ ManageWorkstation 실패")
            return False

    def _pause_cb(self, msg, unit_idx: int):
        """ControlTowerNode가 대상 라인에 Pause 를 발행하면 해당 Unit만 일시정지"""
        # unit_idx 기반으로 해당 Unit 찾기
        for unit in self.units_ref:
            if unit.unit_idx == unit_idx:
                prev = getattr(unit, '_is_paused', False)
                unit._is_paused = msg.data
                if msg.data and not prev:
                    print(f"[DB Node] ⏸️ sg2_in_{unit_idx:02d} Pause 수신")
                elif not msg.data and prev:
                    print(f"[DB Node] ▶️ sg2_in_{unit_idx:02d} Resume 수신")
                break
        # 전체 시스템 일시정지 플래그도 업데이트
        self.is_paused = any(
            getattr(u, '_is_paused', False) for u in self.units_ref
        )

    def _spawn_trigger_cb(self, msg):
        """
        SimSyncNode가 /sim/sg2_spawn_trigger로 발행한 상자 소환 명령 처리.
        JSON 페이로드: { "package_id": "PKG_20260612_001",
                             "target_line": "sg2_in_01",
                             "timestamp": 1234567890.0 }
        """
        try:
            import json
            payload = json.loads(msg.data)
            package_id = payload.get('package_id', '')
            target_line = payload.get('target_line', '')   # 예: "sg2_in_01"

            print(f"[DB Node] 📦 sg2_spawn_trigger 수신: {package_id} → {target_line}")

            # target_line 기반으로 해당 Unit을 찾아 대기 큐에 적제
            unit_idx = self.line_to_unit_idx.get(target_line)
            if unit_idx is None:
                print(f"[DB Node] ⚠️ 알 수 없는 target_line: {target_line}")
                return

            for unit in self.units_ref:
                if unit.unit_idx == unit_idx:
                    unit.pending_packages.put(payload)
                    print(f"[DB Node] ✅ Unit {unit_idx} 대기큐에 패키지 적제: {package_id}")
                    break
        except Exception as e:
            print(f"[DB Node] ❌ sg2_spawn_trigger 파싱 실패: {e}")


# ============================================================
# 메인 컨트롤러
# ============================================================
class SH5IntegratedController:
    def __init__(self):
        self.db_node = None
        self.units: list[SH5WorkUnit] = []
        self._running = False
        self._thread = None

    def init(self):
        """초기화: DB 노드 + 작업 단위 3대 생성"""
        # ROS 2 초기화
        if ROS2_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            # units를 먼저 생성하고 db_node에 참조를 넘길 필요 있음
            for idx, r_pos in enumerate(ROBOT_POSITIONS, start=1):
                unit = SH5WorkUnit(
                    unit_idx=idx,
                    robot_pos=r_pos,
                    db_node=None,  # 일단 None 설정
                )
                self.units.append(unit)

            # DB 노드 생성 시 units 참조 전달
            self.db_node = SH5DBNode(units_ref=self.units)

            # db_node 연결 후 반영
            for unit in self.units:
                unit.db = self.db_node

            print("[Controller] ROS 2 DB 노드 활성화")
        else:
            print("[Controller] ⚠️ ROS 2 없음 - 시뮬레이션 모드")
            for idx, r_pos in enumerate(ROBOT_POSITIONS, start=1):
                unit = SH5WorkUnit(
                    unit_idx=idx,
                    robot_pos=r_pos,
                    db_node=None,
                )
                self.units.append(unit)

        # SH5 Articulation 로드 (Isaac Sim 열린 상태에서만 동작)
        if ISAAC_AVAILABLE:
            print("[Controller] 🤖 SH5 Articulation 로드 시작...")
            for unit in self.units:
                unit.robot = load_sh5_robot(
                    unit_idx=unit.unit_idx,
                    world_pos=unit.robot_world_pos,
                )
            loaded = sum(1 for u in self.units if u.robot is not None)
            print(f"[Controller] Articulation 로드 결과: {loaded}/{len(self.units)}대")
        else:
            print("[Controller] ⚠️ Isaac Sim 없음 - Articulation 스킵 (Mock 모드)")

        print(f"\n[Controller] ✅ SH5 {len(self.units)}대 초기화 완료!")
        print(f"[Controller] PICK_AND_PLACE_MODE = {PICK_AND_PLACE_MODE}")
        mode_emoji = {"ACT_MODEL": "🤖", "HDF5_REPLAY": "🎬", "DUMMY_TELEPORT": "🚀"}
        print(f"[Controller] {mode_emoji.get(PICK_AND_PLACE_MODE, '❓')} {PICK_AND_PLACE_MODE} 모드로 실행합니다")

    def run_once(self):
        """각 작업 단위를 1 스텝 실행 (Script Editor 타이머에 바인딩 가능)"""
        if self.db_node and ROS2_AVAILABLE:
            rclpy.spin_once(self.db_node, timeout_sec=0.05)
        for unit in self.units:
            unit.step()

    def run_loop(self, interval_sec: float = 0.1):
        """별도 스레드에서 루프 실행"""
        self._running = True
        print(f"[Controller] 🔄 메인 루프 시작 (interval={interval_sec}s)")
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                print(f"[Controller] ❌ 루프 오류: {e}")
                import traceback; traceback.print_exc()
            time.sleep(interval_sec)
        print("[Controller] 루프 종료")

    def start(self):
        self.init()
        self._thread = threading.Thread(
            target=self.run_loop, kwargs={"interval_sec": 0.5}, daemon=True
        )
        self._thread.start()
        print("[Controller] 🚀 SH5 통합 컨트롤러 백그라운드 스레드 시작!")

    def stop(self):
        self._running = False
        if self.db_node and ROS2_AVAILABLE:
            self.db_node.destroy_node()
            rclpy.shutdown()
        print("[Controller] 🛑 SH5 통합 컨트롤러 종료")


# ============================================================
# Script Editor exec() 진입점
# ============================================================
# 이미 실행 중인 컨트롤러가 있으면 중지
if 'sh5_controller' in dir() and sh5_controller is not None:
    try:
        sh5_controller.stop()
        print("[SH5] 이전 컨트롤러 종료 완료")
    except Exception:
        pass

sh5_controller = SH5IntegratedController()
sh5_controller.start()

print("""
╔══════════════════════════════════════════════════════════╗
║       SH5 통합 컨트롤러 실행 중                          ║
║                                                          ║
║  • 상태 확인:  sh5_controller.units[0].slots             ║
║  • 일시정지:   sh5_controller.db_node.is_paused          ║
║  • 종료:       sh5_controller.stop()                     ║
║                                                          ║
║  모드 전환 (스크립트 상단 PICK_AND_PLACE_MODE 변경):     ║
║    "DUMMY_TELEPORT" → 순간이동 (빠른 테스트)             ║
║    "HDF5_REPLAY"    → VR 궤적 재생 (★ 데모 보험)         ║
║    "ACT_MODEL"      → AI 추론 (모델 완성 후)             ║
╚══════════════════════════════════════════════════════════╝
""")
