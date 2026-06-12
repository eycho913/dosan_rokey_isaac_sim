#!/usr/bin/env python3
"""
sh5_bringup_ros2_3robot.py — SH5 HDF5 Replay + ROS2 통합 v3 (3대 로봇)
=======================================================
시나리오:
  1. 초기 상자 없음 (Z=-10 숨김)
  2. 트리거 수신 → box_assets에서 해당 상자 스폰
  3. DB check_warehouse_status → 중복=디스폰, 신규=pick&place
  4. 고객별 슬롯 유지 할당 (같은 고객 → 같은 슬롯)
  5. 완료 후 report_inbound_progress 보고

실행:
  isaac-python sh5_bringup_ros2.py --slot 1

  # 별도 터미널 (ROS2):
  python3 ros2_sh5_bridge.py

  # 테스트 (ROS2 없이):
  echo '{"package_id":"PKG_001","qr_id":"QR_001","customer_id":"CUST_A","target_line":"sg2_in_01"}' >> /tmp/sh5_queue.jsonl
"""

import argparse, json, os, queue, random, sys, time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from isaaclab.app import AppLauncher

ROBOTIS_LAB_DIR = Path("/home/rokey/dev_ws/robotis_lab/scripts/sim2real/bringup")
if str(ROBOTIS_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOTIS_LAB_DIR))
from common import robotis_config as cfg

parser = argparse.ArgumentParser()
parser.add_argument("--slot", type=int, default=1, choices=[1,2,3,4])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True   # QR 스캐너용 TopView 카메라 활성화
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py, numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass
from robotis_lab.assets.robots import FFW_SH5_CFG

sys.path.insert(0, "/home/rokey/dev_ws/coupang_ws/scripts")
from hdf5_replay_player import HDF5EpisodeLoader

# ── 설정 ──────────────────────────────────────────────────────────────────
BOX_ASSETS_DIR  = Path("/home/rokey/dev_ws/box_assets")
QUEUE_FILE      = "/tmp/sh5_queue.jsonl"       # bridge → Isaac Sim (트리거)
QR_REQ_FILE     = "/tmp/sh5_qr_req.jsonl"      # Isaac Sim → bridge (QR 확인 요청)
QR_RESULT_FILE  = "/tmp/sh5_qr_result.jsonl"   # bridge → Isaac Sim (DB 체크 결과)
REPORT_REQ_FILE = "/tmp/sh5_report_req.jsonl"  # Isaac Sim → bridge (입고 보고)
PAUSE_FILE      = "/tmp/sh5_pause.json"         # bridge → Isaac Sim (일시정지 신호)
BOX_DESPAWN_POS = (0.0, 0.0, -10.0)
SKIP_FRAMES     = 1
MAX_SLOTS       = 8    # 앞뒷면 총 8칸
QR_SCAN_TIMEOUT = 5.0
QR_SCAN_INTERVAL = 0.2
DB_WAIT_TIMEOUT = 5.0
PLACEMENT_FREEZE_FRAMES = 30
CAMERA_SKIP_FRAMES = 3
HOMING_FRAMES      = 120
PLAYBACK_SPEED     = 2    # 재생 배속 (1=원속, 2=2배속, 4=4배속)
                          # 너무 크면 움직임이 끊겨보임 → 권장: 2~4

# [Fix 1] 재생 시작 시 현재 자세 → 첫 프레임까지 부드럽게 보간하는 워밍업 프레임 수
# 0으로 설정하면 즉시 텔레포트 (기존 동작)
WARMUP_FRAMES = 30   # 약 1초 (30Hz 기준) 동안 서서히 첫 프레임 자세로 이동

# [Fix 2] 안전 에피소드 폴더: 방해 안되는 팔 frozen 전처리 데이터 사용
FROZEN_SET_DIR = Path("/home/rokey/dev_ws/datasets/train_data/frozen_set")

# [Fix 3] 복귀 시 팔 안전 자세: stay.hdf5 첫 프레임 관절값 사용
STAY_HDF5_PATH = "/home/rokey/dev_ws/datasets/stay.hdf5"

# ── 작업대 Spawn/Despawn 설정 ────────────────────────────────────────────────────
WS_TRIGGER_FILE = "/tmp/sh5_ws_trigger.jsonl"  # 관제탐 → Isaac Sim (작업대 트리거)

# 작업대 USD 에셋 경로 (실제 파일로 수정 필요)
WORKSTATION_USD = "/home/rokey/dev_ws/Collected_finalfac/workstation.usd"

# 작업대 스데이션 스폰 좌표
# finalfac.usd 안 실제 좌표로 수정 필요 (x, y, z)
WORKSTATION_SPAWN_POS = {
    "sg2_in_01_A": (9.5,  4.0, 0.0),
    "sg2_in_01_B": (9.5,  2.0, 0.0),
    "sg2_in_02_A": (9.5, -0.5, 0.0),
    "sg2_in_02_B": (9.5, -2.5, 0.0),
    "sg2_in_03_A": (9.5, -5.0, 0.0),
    "sg2_in_03_B": (9.5, -7.0, 0.0),
}
# location → RACK prim 매핑 (Stage 패널로 확인 완료)
# RACK_01 = sg2_out (출고 컨베이어)
# RACK_02 = sg2_in_01_A (1번 라인 작업대)
# RACK_03 = sg2_in_02_A (2번 라인 작업대)
# RACK_04 = sg2_in_03_A (3번 라인 작업대)
# RACK_05~10 = 창고 대기 (씬에 없음)
RACK_PREFIX = "/World/FinalFac/"
WS_LOCATION_TO_RACK = {
    "sg2_out":    "RACK_01",   # 출고 컨베이어
    "sg2_in_01_A": "RACK_02",  # 1번 라인 작업대
    "sg2_in_02_A": "RACK_03",  # 2번 라인 작업대
    "sg2_in_03_A": "RACK_04",  # 3번 라인 작업대
    # B 슬롯은 창고 대기 중 → SPAWN 시 신규 생성 필요
}
WORKSTATION_PRIM_PATTERN = [
    RACK_PREFIX + "{loc}",
    "/World/envs/env_0/FinalFac/{loc}",
]
# X축 -90도 회전 quaternion (QR이 위를 보도록) - (w, x, y, z)
BOX_SPAWN_QUAT  = [0.7071, -0.7071, 0.0, 0.0]

LINE_TO_SLOT = {"sg2_in_01": 1, "sg2_in_02": 2, "sg2_in_03": 3}
WORKSTATION_ID = {"sg2_in_01": "WS01", "sg2_in_02": "WS02", "sg2_in_03": "WS03"}
WORKSTATION_QR = {"sg2_in_01": "WS_QR_01", "sg2_in_02": "WS_QR_02", "sg2_in_03": "WS_QR_03"}

# ★ 라인별 로봇 스폰 위치 (씬 좌표에 맞게 조정)
LINE_ROBOT_POS = {
    "sg2_in_01": (7.5,  3.0, -0.18),
    "sg2_in_02": (7.5, -1.5, -0.18),
    "sg2_in_03": (7.5, -6.0, -0.18),
}
LINE_IDS = ["sg2_in_01", "sg2_in_02", "sg2_in_03"]

# ── ★ 환경 설정 (여기서만 수정) ─────────────────────────────────────────
# finalfac.usd 환경 파일 경로
FINALFAC_USD   = "/home/rokey/dev_ws/Collected_finalfac/finalfac.usd"

# 로봇 초기 스폰 위치 — finalfac 씬 좌표계 보고 필요 시 수정
# cfg.ROBOT_POS 기본값: (0.0, 0.0, -0.18)
# ROBOT_INIT_POS = cfg.ROBOT_POS   # 예: (2.5, -1.0, -0.18)
ROBOT_INIT_POS = (7.5, 3.0, -0.18)  # 예: (2.5, -1.0, -0.18)

# ── 씬 설정 ───────────────────────────────────────────────────────────────
@configclass
class BringupSceneCfg(InteractiveSceneCfg):
    # ★ finalfac.usd 를 월드 배경으로 로드
    #   USD 안에 ground/light/rack/belt 가 이미 포함되어 있으므로
    #   GroundPlane, DomeLight, rack, pedestal 은 별도 생성하지 않음
    world = AssetBaseCfg(
        prim_path="/World/FinalFac",
        spawn=sim_utils.UsdFileCfg(
            usd_path=FINALFAC_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # TopView 카메라: 컨베이어 위 내려다보는 시점 (QR 인식용)
    topview_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopViewCamera",
        update_period=0.1,   # 경량화: 10fps만 업데이트
        height=320, width=320,   # 해상도 낮춤: QR 인식에 320x320으로 충분
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
    box01: RigidObjectCfg  = None
    box02: RigidObjectCfg  = None
    box03: RigidObjectCfg  = None
    robot01: ArticulationCfg = None
    robot02: ArticulationCfg = None
    robot03: ArticulationCfg = None

# ── 고객 슬롯 레지스트리 ──────────────────────────────────────────────────
class SlotRegistry:
    """고객별 슬롯 유지 할당 (같은 고객 → 같은 슬롯, 신규 → 다음 슬롯)"""
    def __init__(self, max_slots=4):
        self._cust_to_slot: dict[str, int] = {}
        self._slot_counts:  dict[int, int] = {s: 0 for s in range(1, max_slots+1)}
        self._next = 1
        self._max  = max_slots

    def assign(self, customer_id: str) -> int:
        if customer_id in self._cust_to_slot:
            slot = self._cust_to_slot[customer_id]
            print(f"  [Slot] 기존 고객 '{customer_id}' → 슬롯 {slot}")
            return slot
        slot = self._next
        self._cust_to_slot[customer_id] = slot
        self._next = (self._next % self._max) + 1
        print(f"  [Slot] 신규 고객 '{customer_id}' → 슬롯 {slot} 배정")
        return slot

    def increment(self, slot: int) -> int:
        self._slot_counts[slot] = self._slot_counts.get(slot, 0) + 1
        return self._slot_counts[slot]

# ── 파일 큐 리더 ──────────────────────────────────────────────────────────
# ── 작업대 Spawn/Despawn 관리자 ──────────────────────────────────────────────
class WorkstationManager:
    """
    /tmp/sh5_ws_trigger.jsonl 파일큐를 폴링해서
    작업대 Prim을 보이거나 숨깁니다.

    메시지 형식:
      {"workstation_id": "WS01", "location": "sg2_in_01_A", "action": "DESPAWN"}
      {"workstation_id": "WS02", "location": "sg2_in_01_A", "action": "SPAWN"}

    DESPAWN: 해당 위치 Prim을 Z=-200으로 이동 (보이지 않게)
    SPAWN  : 해당 위치 Prim을 WORKSTATION_SPAWN_POS 좌표로 복원
             (USD 에셋이 있으면 신규 생성, 없으면 기존 Prim 복원)
    """

    def __init__(self, stage):
        self._stage = stage
        self._file_pos = 0
        if not os.path.exists(WS_TRIGGER_FILE):
            open(WS_TRIGGER_FILE, "w").close()
        self._file_pos = os.path.getsize(WS_TRIGGER_FILE)
        # 동적 생성된 prim 추적: location → prim_path
        self._dynamic_prims: dict[str, str] = {}
        # 원래 위치 캐시: location → (x, y, z) — DESPAWN 전 저장
        self._original_pos: dict[str, tuple] = {}
        print(f"[WS] ✅ 작업대 관리자 초기화 | 트리거: {WS_TRIGGER_FILE}")

    # ── 폴링 ──────────────────────────────────────────────────────────────────
    def poll(self):
        try:
            size = os.path.getsize(WS_TRIGGER_FILE)
        except OSError:
            return
        if size <= self._file_pos:
            return
        with open(WS_TRIGGER_FILE, "r") as f:
            f.seek(self._file_pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    self._handle(msg)
                except Exception as e:
                    print(f"[WS] ⚠️ 파싱 오류: {e} | '{line}'")
            self._file_pos = f.tell()

    # ── 메시지 처리 ───────────────────────────────────────────────────────────
    def _handle(self, msg: dict):
        action   = str(msg.get("action", "")).upper()
        location = str(msg.get("location", ""))
        ws_id    = str(msg.get("workstation_id", "?"))
        print(f"[WS] 📨 {action} | {ws_id} | {location}")
        if action == "DESPAWN":
            self._despawn(location, ws_id)
        elif action == "SPAWN":
            self._spawn(location, ws_id)
        else:
            print(f"[WS] ⚠️ 알 수 없는 action: {action}")

    # ── DESPAWN: Prim을 Z=-200으로 내려서 숨김 ──────────────────────────────
    def _despawn(self, location: str, ws_id: str):
        prim_path = self._find_prim(location)
        if not prim_path:
            print(f"[WS] DESPAWN 실패: '{location}' prim 없음")
            print(f"       → WORKSTATION_PRIM_PATTERN을 실제 prim 경로로 수정하세요")
            return
        try:
            from pxr import UsdGeom, Gf
            prim = self._stage.GetPrimAtPath(prim_path)
            if not (prim and prim.IsValid()):
                print(f"[WS] DESPAWN 실패: prim 유효하지 않음 ({prim_path})")
                return
            xf = UsdGeom.Xformable(prim)
            # 현재 위치 저장 (나중에 SPAWN 시 복원용)
            ops = xf.GetOrderedXformOps()
            if ops:
                cur = ops[0].Get()
                if cur:
                    self._original_pos[location] = (cur[0], cur[1], cur[2])
            # Z=-200으로 이동 (화면 밖)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -200.0))
            print(f"[WS] ✅ DESPAWN 완료: {location} ({ws_id}) → 숨김")
        except Exception as e:
            print(f"[WS] DESPAWN 오류: {e}")

    # ── SPAWN: Prim을 정상 위치로 복원 또는 신규 생성 ───────────────────────
    def _spawn(self, location: str, ws_id: str):
        # 목표 좌표 결정: 저장된 원래 위치 → 설정 좌표 순서로 폴백
        target_pos = (self._original_pos.get(location)
                      or WORKSTATION_SPAWN_POS.get(location))
        if target_pos is None:
            print(f"[WS] SPAWN 실패: '{location}' 좌표 없음 (WORKSTATION_SPAWN_POS 설정 필요)")
            return

        # 기존 Prim 복원 시도
        prim_path = self._find_prim(location)
        if prim_path:
            try:
                from pxr import UsdGeom, Gf
                prim = self._stage.GetPrimAtPath(prim_path)
                if prim and prim.IsValid():
                    xf = UsdGeom.Xformable(prim)
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(*target_pos))
                    print(f"[WS] ✅ SPAWN 복원: {location} ({ws_id}) @ {target_pos}")
                    return
            except Exception as e:
                print(f"[WS] SPAWN 복원 오류: {e}")

        # 신규 USD Prim 생성 (기존 Prim 없을 때)
        if os.path.exists(WORKSTATION_USD):
            new_path = f"/World/envs/env_0/WS_{location.replace('-','_')}"
            try:
                from pxr import UsdGeom, Gf
                prim = self._stage.DefinePrim(new_path, "Xform")
                prim.GetReferences().AddReference(WORKSTATION_USD)
                xf = UsdGeom.Xformable(prim)
                xf.AddTranslateOp().Set(Gf.Vec3d(*target_pos))
                self._dynamic_prims[location] = new_path
                print(f"[WS] ✅ SPAWN 신규: {location} ({ws_id}) @ {target_pos}")
            except Exception as e:
                print(f"[WS] SPAWN 신규 생성 오류: {e}")
        else:
            print(f"[WS] SPAWN 실패: USD 에셋 없음 ({WORKSTATION_USD})")
            print(f"       → WORKSTATION_USD를 실제 작업대 USD 경로로 수정하세요")

    # ── Prim 경로 탐색 ─────────────────────────────────────────────────────────
    def _find_prim(self, location: str) -> str | None:
        """location 명으로 Stage 내 작업대 prim 경로 반환"""
        candidates = []

        # 1) 동적 생성된 prim 먼저 확인
        if location in self._dynamic_prims:
            candidates.append(self._dynamic_prims[location])

        # 2) ★ 명시 매핑(WS_LOCATION_TO_RACK) 우선 사용
        rack_name = WS_LOCATION_TO_RACK.get(location)
        if rack_name:
            candidates.append(RACK_PREFIX + rack_name)                     # /World/FinalFac/RACK_02
            candidates.append("/World/envs/env_0/FinalFac/" + rack_name)   # 대체 경로
            candidates.append("/World/" + rack_name)                        # 최상위 경로

        # 3) 패턴 기반 폴백
        for pat in WORKSTATION_PRIM_PATTERN:
            candidates.append(pat.format(loc=location))

        # 탐색 + 디버그 출력
        print(f"[WS] 🔍 prim 탐색: {location}")
        for path in candidates:
            prim = self._stage.GetPrimAtPath(path)
            valid = prim and prim.IsValid()
            print(f"[WS]   {'✅' if valid else '✗ '} {path}")
            if valid:
                return path
        return None


# ── 파일 큐 리더 ──────────────────────────────────────────────────────────
class FileQueueReader:
    def __init__(self, pkg_queue: queue.Queue):
        self._q   = pkg_queue
        self._pos = 0
        if not os.path.exists(QUEUE_FILE):
            open(QUEUE_FILE, "w").close()
        self._pos = os.path.getsize(QUEUE_FILE)
        print(f"[FileQueue] ✅ {QUEUE_FILE} 모니터링 시작")

    def poll(self):
        try:
            size = os.path.getsize(QUEUE_FILE)
        except OSError:
            return
        if size < self._pos:
            self._pos = 0
        elif size == self._pos:
            return
        with open(QUEUE_FILE, "r") as f:
            f.seek(self._pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    self._q.put(payload)
                    print(f"[FileQueue] 📨 {payload.get('package_id','?')} → {payload.get('target_line','?')}")
                except Exception:
                    pass
            self._pos = f.tell()

# ── QR 스캐너 (TopView 카메라 기반) ──────────────────────────────────────
class QRScanner:
    """
    Isaac Lab Camera 센서에서 RGB 프레임을 읽어 WeChatQR로 QR 코드 인식.
    - 인식 성공 시: qr_id 문자열 반환
    - 타임아웃 시: package_id에서 자동 유도 (PKG_YYYYMMDD_NNN → QR_YYYYMMDD_NNN)
    """
    def __init__(self, camera):
        self._camera   = camera
        self._detector = None
        self._init_detector()

    def _init_detector(self):
        try:
            self._detector = cv2.wechat_qrcode_WeChatQRCode()
            print("[QR] ✅ WeChatQRCode 초기화 완료")
        except Exception as e:
            print(f"[QR] ⚠️  WeChatQRCode 초기화 실패: {e} → fallback 모드")

    def _fallback_qr_id(self, package_id: str) -> str:
        """package_id에서 qr_id 자동 유도: PKG_YYYYMMDD_NNN → QR_YYYYMMDD_NNN"""
        return package_id.replace("PKG_", "QR_", 1)

    def scan(self, package_id: str, timeout: float = QR_SCAN_TIMEOUT) -> str:
        """
        최대 timeout초 동안 카메라에서 QR 인식 시도.
        성공 시 qr_id 반환, 실패 시 fallback.
        """
        if self._detector is None:
            fb = self._fallback_qr_id(package_id)
            print(f"[QR] detector 없음 → fallback: {fb}")
            return fb

        deadline   = time.time() + timeout
        last_scan  = 0.0
        attempt    = 0

        print(f"[QR] 📷 스캔 시작 (최대 {timeout}초)...")
        while time.time() < deadline:
            now = time.time()
            if now - last_scan < QR_SCAN_INTERVAL:
                time.sleep(0.05)
                continue
            last_scan = now
            attempt  += 1

            try:
                # Isaac Lab Camera: output["rgb"] → (H, W, 4) RGBA uint8
                rgb_data = self._camera.data.output.get("rgb")
                if rgb_data is None:
                    continue
                # Tensor → numpy, RGBA → RGB → BGR (cv2 입력 포맷)
                if hasattr(rgb_data, "cpu"):
                    frame = rgb_data[0].cpu().numpy()  # (H,W,4)
                else:
                    frame = np.array(rgb_data[0])
                bgr = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)

                texts, _ = self._detector.detectAndDecode(bgr)
                if texts:
                    qr_id = texts[0]
                    print(f"[QR] ✅ 인식 성공 (시도 {attempt}): {qr_id}")
                    return qr_id
            except Exception as e:
                if attempt == 1:
                    print(f"[QR] 프레임 읽기 오류: {e}")

        fb = self._fallback_qr_id(package_id)
        print(f"[QR] ⏱️  타임아웃 → fallback: {fb}")
        return fb

# ── 보고 함수 (브릿지가 읽는 파일에 기록) ───────────────────────────────
def _write_report_request(payload: dict):
    """report_inbound_progress 요청을 파일에 기록 → ros2_sh5_bridge.py가 서비스 호출"""
    with open(REPORT_REQ_FILE, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
    print(f"  [보고] 완료 기록 → {REPORT_REQ_FILE}")

# ── box_assets USD 선택 ───────────────────────────────────────────────────
def _get_box_usd(package_id: str) -> str | None:
    # 1. 정확 일치
    exact = BOX_ASSETS_DIR / f"{package_id}.usd"
    if exact.exists():
        print(f"  [BoxUSD] 정확 일치: {exact.name}")
        return str(exact)
    # 2. 번호 매핑 (PKG_YYYYMMDD_NNN → *_NNN.usd)
    parts = package_id.split("_")
    if parts:
        suffix = parts[-1]
        matches = list(BOX_ASSETS_DIR.glob(f"*_{suffix}.usd"))
        if matches:
            print(f"  [BoxUSD] 번호 매핑: {matches[0].name}")
            return str(matches[0])
    # 3. 랜덤
    usd_files = list(BOX_ASSETS_DIR.glob("*.usd"))
    if usd_files:
        chosen = random.choice(usd_files)
        print(f"  [BoxUSD] 랜덤 선택: {chosen.name}")
        return str(chosen)
    return None

# ── IsaacLab 로봇 어댑터 ──────────────────────────────────────────────────
class RobotAdapter:
    def __init__(self, robot):
        self._robot  = robot
        self.num_dof = robot.data.joint_pos.shape[1]

    def set_joint_position_targets(self, positions: np.ndarray):
        """PD 제어 목표설정 (호밍용)"""
        t = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
        self._robot.set_joint_position_target(t)

    def teleport_joints(self, positions: np.ndarray):
        """PD 라그 없이 직접 텔레포트 (replay 전용 → 상자 동기 정확)"""
        t = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
        self._robot.write_joint_state_to_sim(t, torch.zeros_like(t))
        self._robot.set_joint_position_target(t)   # 다음 스텝 안정화

# ── 상태머신 재생 컨트롤러 ────────────────────────────────────────────────
class ReplayController:
    IDLE, SCANNING, WAITING_DB, REPLAYING, HOMING, DONE = \
        "IDLE", "SCANNING", "WAITING_DB", "REPLAYING", "HOMING", "DONE"

    def __init__(self, robot: RobotAdapter, scene, slot_registry: SlotRegistry,
                 robot_key: str = "robot01", box_key: str = "box01"):
        self.robot         = robot
        self.scene         = scene
        self.slot_reg      = slot_registry
        self._robot_key    = robot_key
        self._box_key      = box_key
        self._box_prim_path = f"/World/envs/env_0/{box_key[0].upper()}{box_key[1:]}"
        self.state         = self.IDLE
        self.episode       = None
        self.frame_idx     = 0
        self._offset       = np.zeros(3)
        self._pkg_id       = ""
        self._qr_id        = ""
        self._line_id      = ""
        self._ws_id        = ""
        self._ws_qr        = ""
        self._customer_id  = ""
        self._slot         = 1
        self._slot_count   = 0
        # QR 스캔 관련
        self._camera             = None
        self._detector           = None
        self._db_deadline        = 0.0
        self._qr_result_pos      = 0
        # 호밍 관련
        self._home_joint_pos         = None
        self._home_base_pos          = None   # 베이스 홈 위치 (XYZ)
        self._homing_start_pos       = None   # 호밍 시작 시 관절 위치
        self._homing_start_base_pos  = None   # 호밍 시작 시 및 베이스 위치 (XYZ)
        self._homing_start_base_quat = None   # 호밍 시작 시 베이스 회전
        self._homing_frame           = 0
        self._is_dup_from_bridge = None
        self._pending_episode    = None
        self._pending_offset     = None
        self._scan_deadline      = 0.0
        self._scan_attempts      = 0

    def set_camera(self, camera):
        self._camera = camera
        try:
            self._detector = cv2.wechat_qrcode_WeChatQRCode()
            print("[QR] ✅ WeChatQRCode 디텍터 초기화 완료")
        except Exception as e:
            print(f"[QR] ⚠️  WeChatQRCode 실패: {e} → fallback 모드")

    def set_home_pos(self, home_joint_pos: torch.Tensor, home_base_pos=None):
        """로봇 HOME 관절 위치 등록 + 베이스(mobile base) 홈 위치"""
        self._home_joint_pos = home_joint_pos.clone()
        # 베이스 홈 위치: (x, y, z), 회전은 identity (1,0,0,0)
        if home_base_pos is not None:
            self._home_base_pos = torch.tensor(home_base_pos, dtype=torch.float32)
        else:
            self._home_base_pos = None

    def _fallback_qr(self, pkg_id: str) -> str:
        return pkg_id.replace("PKG_", "QR_", 1)

    def is_busy(self) -> bool:
        return self.state != self.IDLE

    def _is_paused(self) -> bool:
        """pause 파일 폴링 (브릿지가 /{robot_id}/pause_status 수신 시 업데이트)"""
        try:
            with open(PAUSE_FILE, "r") as f:
                paused = json.load(f).get("paused", False)
        except Exception:
            paused = False

        # 상태가 변경됐을 때만 터미널 출력
        prev = getattr(self, "_prev_paused", None)
        if prev != paused:
            self._prev_paused = paused
            now = time.strftime("%H:%M:%S")
            if paused:
                print(f"\n[{now}] ⏸  pause_status = TRUE  → 작업 일시정지 (작업대 만석/회전 대기)")
            else:
                print(f"\n[{now}] ▶  pause_status = FALSE → 작업 재개")
        return paused

    # ── Kinematic 박스 포즈 전용 쓰기 ─────────────────────────────────────────
    def _write_box_pose(self, pos: torch.Tensor, quat: torch.Tensor):
        """
        Kinematic 박스에 위치+회전만 씁니다 (velocity 미설정).
        
        write_root_state_to_sim은 내부에서 PhysX setLinearVelocity/setAngularVelocity를
        호출하는데, kinematic 바디에는 이 API가 금지되어 있어 에러가 1000개 누적되면
        PhysX가 시뮬레이션 전체를 강제 종료합니다.
        
        이 헬퍼는:
          1. write_root_pose_to_sim  (Isaac Lab ≥ 0.3 — pose만 쓰는 전용 API)
          2. 없으면 USD XFormable   (USD 레이어 직접 쓰기 → PhysX 우회)
        순서로 시도합니다.
        """
        box = self.scene[self._box_key]

        # 방법 1: Isaac Lab write_root_pose_to_sim (pose only, velocity 없음)
        if hasattr(box, "write_root_pose_to_sim"):
            root_pose = box.data.default_root_state[:, :7].clone()
            root_pose[0, :3]  = pos
            root_pose[0, 3:7] = quat
            box.write_root_pose_to_sim(root_pose)
            return

        # 방법 2: USD XFormable 직접 쓰기 (PhysX kinematic target 우회)
        try:
            from pxr import UsdGeom, Gf
            import omni.usd
            _stage = omni.usd.get_context().get_stage()
            _prim  = _stage.GetPrimAtPath(self._box_prim_path)
            if _prim and _prim.IsValid():
                xf = UsdGeom.Xformable(_prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                xf.AddOrientOp().Set(Gf.Quatd(w, x, y, z))
                return
        except Exception:
            pass

        # 방법 3: 최후 폴백 — 에러가 발생하지만 기능은 유지
        bs = box.data.default_root_state.clone()
        bs[0, :3]  = pos
        bs[0, 3:7] = quat
        bs[0, 7:]  = 0.0
        box.write_root_state_to_sim(bs)

    def start_scan(self, pkg_id, customer_id, line_id, episode, offset, is_dup_from_bridge):
        """상자 스폰 완료 후 QR 스캔 단계 진입 (non-blocking)"""
        self._pkg_id             = pkg_id
        self._customer_id        = customer_id
        self._line_id            = line_id
        self._ws_id              = WORKSTATION_ID.get(line_id, "WS01")
        self._ws_qr              = WORKSTATION_QR.get(line_id, "")
        self._slot               = self.slot_reg.assign(customer_id)
        self._pending_episode    = episode
        self._pending_offset     = offset
        self._is_dup_from_bridge = is_dup_from_bridge
        self._scan_deadline      = time.time() + QR_SCAN_TIMEOUT
        self._scan_attempts      = 0
        self._qr_id              = ""
        self.state               = self.SCANNING
        print(f"[QR] 📷 스캔 시작 (최대 {QR_SCAN_TIMEOUT}초)...")

    def start(self, pkg_id, qr_id, customer_id, line_id, episode, offset_xy):
        self._pkg_id      = pkg_id
        self._qr_id       = qr_id
        self._customer_id = customer_id
        self._line_id     = line_id
        self._ws_id       = WORKSTATION_ID.get(line_id, "WS01")
        self._ws_qr       = WORKSTATION_QR.get(line_id, "")
        self._slot        = self.slot_reg.assign(customer_id)
        self._offset      = offset_xy
        self.episode      = episode
        self.frame_idx    = 0
        self.state        = self.REPLAYING
        # [Fix 1] 워밍업: 현재 관절 자세 스냅샷 저장 (첫 프레임까지 보간용)
        self._warmup_frame = 0
        self._warmup_start_joints = self.robot._robot.data.joint_pos[0].cpu().numpy().copy()
        if hasattr(self, "_prev_box_pos_w"): del self._prev_box_pos_w  # 속도 클램프 리셋
        total = len(episode.get("joint_trajectory", []))
        print(f"\n[Replay] 🎬 {pkg_id} | 슬롯{self._slot} | {total}프레임 (워밍업 {WARMUP_FRAMES}f)")

    def step(self):
        if self.state == self.IDLE:
            return

        # ── QR 스캔 단계 (매 sim 스텝마다 카메라 1프레임 체크) ─────────────
        if self.state == self.SCANNING:
            self._scan_attempts += 1
            qr_found = ""

            if self._detector is not None and self._camera is not None:
                try:
                    rgb = self._camera.data.output.get("rgb")
                    if rgb is not None:
                        frame = rgb[0].cpu().numpy() if hasattr(rgb, "cpu") else np.array(rgb[0])
                        bgr   = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
                        texts, _ = self._detector.detectAndDecode(bgr)
                        if texts:
                            qr_found = texts[0]
                except Exception:
                    pass

            timed_out = time.time() > self._scan_deadline

            if qr_found:
                self._qr_id = qr_found
                print(f"[QR] ✅ 인식 성공 ({self._scan_attempts}번째 스텝): {self._qr_id}")
                self._after_qr_known()
            elif timed_out:
                # 방어 전략: QR 인식 실패 시 package_id 기반 qr_id 생성
                self._qr_id = self._fallback_qr(self._pkg_id)
                print(f"[QR] ⚠️  인식 실패 → package_id 기반 fallback: {self._qr_id}")
                print(f"       (DB는 package_id로 조회 가능하므로 서비스 호출은 정상 진행)")  
                self._after_qr_known()
            return

        # ── DB 응답 대기 (WAITING_DB) ──────────────────────────────
        if self.state == self.WAITING_DB:
            is_dup = self._poll_db_result()
            if is_dup is not None:
                self._proceed_after_scan(is_dup)
            elif time.time() > self._db_deadline:
                print("[DB] ⏱️  응답 타임아웃 → 신규 처리")
                self._proceed_after_scan(False)
            return

        if self.state == self.REPLAYING:
            # 폰즈 체크 (/{robot_id}/pause_status 수신 시 일시정지)
            if self._is_paused():
                if not getattr(self, "_pause_logged", False):
                    print("\n  [⏸] 일시정지 중... (pause_status=true 수신)")
                    self._pause_logged = True
                return   # 프레임 진행 안 함
            else:
                if getattr(self, "_pause_logged", False):
                    print("  [▶] 재개 (pause_status=false 수신)")
                self._pause_logged = False

            ep    = self.episode
            jt    = ep.get("joint_trajectory")
            bt    = ep.get("box_trajectory")
            rt    = ep.get("robot_trajectory")
            total = len(jt) if jt is not None else 0

            if self.frame_idx < total:
                # ★ 순서: 상자 먼저 → 관절 나중 (타이밍 lag 제거)

                # 1) 상자: HDF5 가이드 Magic Snapping
                # ─────────────────────────────────────────────────────────────
                # [동작 원리]
                #   Phase A (컨베이어): HDF5 궤적 그대로 → 상자 컨베이어에 고정
                #   Phase B (파지 중): HDF5 박스 위치로 '올바른 손' 특정
                #                     → 그 링크에 ATTACH_FACTOR 비율 오프셋으로 부착
                #                     → 손이 빠르게 움직여도 상자가 즉시 따라옴
                #   Phase C (해제): 실제 손가락 관절값(joint_pos)이 열리면 자연스럽게 해제
                #                   → 안착 후 손 뗄 때 상자가 선반에 남음
                #
                # ATTACH_FACTOR: 0=HDF5 원본 위치, 1=로봇 링크 완전 중심
                #   값이 작을수록 손에서 멀어지고, 클수록 링크 중심에 가까워짐
                # ─────────────────────────────────────────────────────────────
                ATTACH_FACTOR     = 1.0   # 오프셋 0 = 링크 콘터에 완전 부착
                GRASP_DIST        = 0.30  # 실제 손 접근 시 스냅 조기 활성화 (0.15 → 0.30으로 늘려 딩레이 감소)
                FINGER_OPEN_THRESH = 0.80  # joint_pos 기준 손가락 열림 판정 (rad)
                MAX_BOX_STEP      = 3.0   # 속도 클램프 (사실상 해제 → 즉시 반응)

                if bt is not None and self.frame_idx < len(bt):
                    bp = bt[self.frame_idx]
                    hdf5_pos = torch.tensor(
                        [bp[0]+self._offset[0], bp[1]+self._offset[1], bp[2]],
                        dtype=torch.float32)
                    target_quat = torch.tensor(
                        bp[3:7] if len(bp) >= 7 else [1, 0, 0, 0], dtype=torch.float32)

                    robot = self.scene[self._robot_key]
                    robot_body_pos  = robot.data.body_pos_w[0]
                    robot_body_quat = robot.data.body_quat_w[0]

                    # 손가락 실제 관절 위치 (target이 아닌 실제값 → 해제 감지 정확)
                    if not hasattr(self, "finger_indices"):
                        self.finger_indices = [
                            i for i, n in enumerate(robot.data.joint_names) if "finger" in n]
                    if self.finger_indices:
                        finger_pos_avg = robot.data.joint_pos[0, self.finger_indices].mean().item()
                    else:
                        finger_pos_avg = 0.0

                    fingers_open   = finger_pos_avg < FINGER_OPEN_THRESH
                    is_hdf5_grasped = hasattr(self, "_hdf5_grasp_body_idx")

                    # HDF5 박스 위치 기준으로 가장 가까운 로봇 링크 찾기 (올바른 손 식별)
                    dist_sq_hdf5 = torch.sum((robot_body_pos - hdf5_pos)**2, dim=-1)
                    closest_idx  = torch.argmin(dist_sq_hdf5).item()
                    closest_dist = torch.sqrt(dist_sq_hdf5[closest_idx]).item()

                    # ── Helper 함수: 쿼터니언 회전 ──
                    def _qr(q, v):
                        wq,xq,yq,zq = q[0],q[1],q[2],q[3]
                        vx,vy,vz    = v[0],v[1],v[2]
                        tx,ty,tz = 2*(yq*vz-zq*vy), 2*(zq*vx-xq*vz), 2*(xq*vy-yq*vx)
                        return torch.stack([vx+wq*tx+yq*tz-zq*ty,
                                            vy+wq*ty+zq*tx-xq*tz,
                                            vz+wq*tz+xq*ty-yq*tx])

                    if not fingers_open and (is_hdf5_grasped or closest_dist < GRASP_DIST):
                        # ── Phase B: 파지 중 ── HDF5 가이드 Snapping
                        if not is_hdf5_grasped:
                            # 파지 시작: 오프셋 저장 (HDF5 위치 기준)
                            self._hdf5_grasp_body_idx = closest_idx
                            body_q = robot_body_quat[closest_idx]
                            inv_q  = torch.stack([body_q[0], -body_q[1], -body_q[2], -body_q[3]])
                            world_off = hdf5_pos - robot_body_pos[closest_idx]
                            # ATTACH_FACTOR로 상자를 손 안쪽으로 당기기
                            self._hdf5_grasp_local_off = _qr(inv_q, world_off) * (1.0 - ATTACH_FACTOR)

                        idx    = self._hdf5_grasp_body_idx
                        body_q = robot_body_quat[idx]
                        target_pos = robot_body_pos[idx] + _qr(body_q, self._hdf5_grasp_local_off)
                    else:
                        # ── Phase A/C: 컨베이어 위 또는 해제 후 ──
                        if is_hdf5_grasped:
                            del self._hdf5_grasp_body_idx, self._hdf5_grasp_local_off
                        target_pos = hdf5_pos

                    # 속도 클램프: 파지 순간 급격한 점프 완화
                    prev_pos = getattr(self, "_prev_box_pos_w", None)
                    if prev_pos is not None:
                        delta = target_pos - prev_pos
                        d = delta.norm().item()
                        if d > MAX_BOX_STEP:
                            target_pos = prev_pos + delta * (MAX_BOX_STEP / d)
                    self._prev_box_pos_w = target_pos.clone()

                    self._write_box_pose(target_pos, target_quat)

                # 2) 관절 주입 — [Fix 1] 워밍업 중: 현재→첫프레임 선형 보간, 이후: 텔레포트
                if self.frame_idx % SKIP_FRAMES == 0:
                    if self._warmup_frame < WARMUP_FRAMES and WARMUP_FRAMES > 0:
                        t_w = self._warmup_frame / WARMUP_FRAMES
                        interp = ((1.0 - t_w) * self._warmup_start_joints
                                  + t_w * np.array(jt[0], dtype=np.float32))
                        self.robot.teleport_joints(interp)
                        self._warmup_frame += 1
                    else:
                        self.robot.teleport_joints(jt[self.frame_idx])

                # 3) 로봇 베이스 (XY offset, Z 유지)
                if rt is not None and self.frame_idx < len(rt):
                    rp   = rt[self.frame_idx]
                    rpos = torch.tensor(
                        [rp[0]+self._offset[0], rp[1]+self._offset[1], rp[2]],
                        dtype=torch.float32)
                    rq   = torch.tensor(rp[3:7] if len(rp)>=7 else [1,0,0,0], dtype=torch.float32)
                    rs   = self.scene[self._robot_key].data.default_root_state.clone()
                    rs[0,:3]=rpos; rs[0,3:7]=rq; rs[0,7:]=0.0
                    self.scene[self._robot_key].write_root_state_to_sim(rs)

                if self.frame_idx % max(100, 100 * PLAYBACK_SPEED) == 0:
                    pct = self.frame_idx / total * 100
                    bar = "█"*int(pct/5) + "░"*(20-int(pct/5))
                    spd = f" x{PLAYBACK_SPEED}" if PLAYBACK_SPEED > 1 else ""
                    print(f"\r  [{bar}] {pct:4.0f}%{spd} ({self.frame_idx}/{total})", end="", flush=True)

                # PLAYBACK_SPEED 만큼 프레임 건너뜀 (배속 재생)
                self.frame_idx = min(self.frame_idx + PLAYBACK_SPEED, total)
            else:
                # ── 재생 완료: 상자 스냅 + kinematic 전환 ──────────────
                if bt is not None and len(bt) > 0:
                    bp_last = bt[-1]
                    bpos_f = torch.tensor(
                        [bp_last[0]+self._offset[0], bp_last[1]+self._offset[1], bp_last[2]],
                        dtype=torch.float32)
                    bq_f = torch.tensor(
                        bp_last[3:7] if len(bp_last) >= 7 else [1,0,0,0],
                        dtype=torch.float32)
                    self._write_box_pose(bpos_f, bq_f)

                try:
                    from pxr import UsdPhysics
                    import omni.usd
                    stage = omni.usd.get_context().get_stage()
                    box_prim = stage.GetPrimAtPath(self._box_prim_path)
                    if box_prim and box_prim.IsValid():
                        rb_api = UsdPhysics.RigidBodyAPI(box_prim)
                        if rb_api:
                            rb_api.GetKinematicEnabledAttr().Set(True)
                            print("  [Physics] ✅ 상자 kinematic 전환 완료")
                except Exception as e:
                    print(f"  [Physics] kinematic 전환 실패: {e}")

                print(f"\n  ✅ 재생 완료! ({total}프레임)")
                # HOMING 진입: 현재 관절 + 베이스 위치 기록
                self._homing_start_pos       = self.robot._robot.data.joint_pos.clone()
                self._homing_start_base_pos  = self.scene[self._robot_key].data.root_pos_w.clone()
                self._homing_start_base_quat = self.scene[self._robot_key].data.root_quat_w.clone()
                self._homing_frame           = 0
                if hasattr(self, "_stay_joints"): del self._stay_joints  # 매 호밍마다 재로드
                self.state = self.HOMING

        # ── HOMING: 바퀴 복귀 + 팔을 stay.hdf5 안전자세로 서서히 이동 ──────
        elif self.state == self.HOMING:
            t = min(1.0, self._homing_frame / HOMING_FRAMES)

            # [Fix 3] 팔: stay.hdf5 안전자세로 보간 (마지막 자세 → stay 자세)
            # 이렇게 하면 마지막 손 동작으로 넘어지는 현상 방지
            if self._homing_start_pos is not None:
                stay_joints = getattr(self, "_stay_joints", None)
                if stay_joints is None:
                    stay_joints = _load_stay_joints()
                    if stay_joints is None:
                        stay_joints = self._homing_start_pos[0].cpu().numpy()
                    self._stay_joints = stay_joints
                interp = ((1.0 - t) * self._homing_start_pos[0].cpu().numpy()
                          + t * stay_joints)
                self.robot.teleport_joints(interp)

            # 베이스(바퀴)만 선형 보간
            if (self._home_base_pos is not None and
                    self._homing_start_base_pos is not None):
                bpos = ((1.0 - t) * self._homing_start_base_pos[0]
                        + t * self._home_base_pos)
                bq_start = self._homing_start_base_quat[0]
                bq_home  = torch.tensor([1.0, 0.0, 0.0, 0.0])
                bq = (1.0 - t) * bq_start + t * bq_home
                bq = bq / (bq.norm() + 1e-8)
                rs = self.scene[self._robot_key].data.default_root_state.clone()
                rs[0, :3]  = bpos
                rs[0, 3:7] = bq
                rs[0, 7:]  = 0.0
                self.scene[self._robot_key].write_root_state_to_sim(rs)

            self._homing_frame += 1
            if self._homing_frame % 30 == 0:
                pct = t * 100
                print(f"\r  [Homing] 🏠{pct:.0f}% 베이스 이동 중... ({self._homing_frame}/{HOMING_FRAMES})", end="", flush=True)

            if self._homing_frame >= HOMING_FRAMES:
                print(f"\n  [Homing] ✅ 베이스 복귀 완료 (팔은 안전위치 유지)")
                self.state = self.DONE

        elif self.state == self.DONE:
            # increment()는 호출하되, 서버로는 현재 배정된 슬롯 번호 자체(1~8)를 보냄
            self.slot_reg.increment(self._slot)
            _write_report_request({
                "workstation_id"   : self._ws_id,
                "workstation_qr_id": self._ws_qr,
                "robot_id"         : self._line_id,
                "package_id"       : self._pkg_id,
                "package_qr_id"    : self._qr_id,
                "customer_id"      : self._customer_id,
                "slot"             : self._slot,
                "filled_slots_count": self._slot,  # 누적 카운트가 아닌 슬롯 고유 번호(1~8) 전송
            })
            print(f"  [Ctrl] 보고 기록 완료 → {self._pkg_id} 슬롯{self._slot}")
            # 다음 상자 스폰을 위해 kinematic 리셋은 새 스폰 시점에 수행
            self.state = self.IDLE

    def _after_qr_known(self):
        """QR 스캔 완료 후: bridge에 check 요청 또는 바로 진행"""
        if self._is_dup_from_bridge is not None:
            # bridge가 이미 is_dup 판단 완료 → 바로 진행
            self._proceed_after_scan(self._is_dup_from_bridge)
        else:
            # bridge가 is_dup 돌려주지 않았음
            # → qr_id로 check_warehouse_status 요청 파일 작성
            req = {"pkg_id": self._pkg_id, "qr_id": self._qr_id,
                   "customer_name": self._customer_id}
            with open(QR_REQ_FILE, "a") as f:
                f.write(json.dumps(req) + "\n")
            self._db_deadline   = time.time() + DB_WAIT_TIMEOUT
            self._qr_result_pos = 0
            print(f"[DB] 📬 check_warehouse_status 요청 → bridge ({self._qr_id})")
            self.state = self.WAITING_DB

    def _poll_db_result(self):
        """qr_result 파일에서 bridge의 check_warehouse_status 결과 콜렉"""
        try:
            size = os.path.getsize(QR_RESULT_FILE)
        except OSError:
            return None
        if size < self._qr_result_pos:
            self._qr_result_pos = 0
        elif size == self._qr_result_pos:
            return None
        with open(QR_RESULT_FILE, "r") as f:
            f.seek(self._qr_result_pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("pkg_id") == self._pkg_id:
                        self._qr_result_pos = f.tell()
                        return data.get("is_duplicate", False)
                except Exception:
                    pass
            self._qr_result_pos = f.tell()
        return None

    def _proceed_after_scan(self, is_dup: bool):
        """QR + DB 체크 완료 후 선택"""
        print(f"\n{'='*55}")
        print(f"  패키지 : {self._pkg_id}")
        print(f"  QR ID : {self._qr_id}")
        print(f"  DB 응답: {'🔴 중복 (디스폰)' if is_dup else '🟢 신규 (pick & place)'}")
        print(f"{'='*55}\n")

        if is_dup:
            bs_despawn_pos = torch.tensor(BOX_DESPAWN_POS, dtype=torch.float32)
            bs_despawn_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            self._write_box_pose(bs_despawn_pos, bs_despawn_quat)
            print("  [Ctrl] 상자 디스폰 완료")
            self.state = self.IDLE
        else:
            ep     = self._pending_episode
            offset = self._pending_offset
            self.episode   = ep
            self._offset   = offset
            self.frame_idx = 0
            self.state     = self.REPLAYING
            # [Fix 1] 워밍업 초기화 (_proceed_after_scan 경로도 동일하게 초기화)
            self._warmup_frame = 0
            self._warmup_start_joints = self.robot._robot.data.joint_pos[0].cpu().numpy().copy()
            if hasattr(self, "_prev_box_pos_w"): del self._prev_box_pos_w  # 속도 클램프 리셋
            total = len(ep.get("joint_trajectory", []))
            print(f"\n[Replay] 🎬 {self._pkg_id} | 슬롯{self._slot} | {total}프레임 (워밍업 {WARMUP_FRAMES}f)")

# ── HDF5 로드 ─────────────────────────────────────────────────────────────
def _load_episode(slot: int) -> dict | None:
    """[Fix 2] frozen_set 폴더에서 우선 h5py 직접 로드. 없으면 기존 폴더로 폴백."""
    frozen_files = sorted(FROZEN_SET_DIR.glob(f"slot{slot}_*.hdf5")) if FROZEN_SET_DIR.exists() else []
    if frozen_files:
        chosen = random.choice(frozen_files)
        print(f"  [HDF5] frozen_set 로드: {chosen.name}")
        try:
            with h5py.File(chosen, "r") as f:
                demo_keys = list(f["data"].keys())
                demo_key  = random.choice(demo_keys)
                demo      = f["data"][demo_key]
                # rack_pose 없는 파일 대비 안전 로드
                rack_key  = "rack_pose" if "rack_pose" in demo["obs"] else "box_pose"
                episode   = {
                    "source_file"       : str(chosen),
                    "demo_key"          : demo_key,
                    "box_initial_pose"  : np.array(demo["obs"]["box_pose"][0]),
                    "rack_initial_pose" : np.array(demo["obs"][rack_key][0]),
                    "robot_initial_pose": np.array(demo["obs"]["robot_pose"][0]),
                    "joint_trajectory"  : np.array(demo["actions"]),
                    "box_trajectory"    : np.array(demo["obs"]["box_pose"]),
                    "robot_trajectory"  : np.array(demo["obs"]["robot_pose"]),
                    "total_frames"      : len(demo["actions"]),
                }
            print(f"  [HDF5] ✅ frozen_set: {chosen.name}/{demo_key} ({episode['total_frames']}프레임)")
            return episode
        except Exception as e:
            print(f"  [HDF5] frozen_set 로드 실패 ({e}) → 기본 폴더로 폴백")
    # 폴백: 기존 HDF5EpisodeLoader 기본 경로
    try:
        return HDF5EpisodeLoader(slot_num=slot).load_random_episode()
    except Exception as e:
        print(f"  [HDF5] 로드 실패: {e}")
        return None

def _load_stay_joints() -> np.ndarray | None:
    """[Fix 3] stay.hdf5에서 안전 자세 관절값(첫 프레임) 로드"""
    try:
        import h5py
        with h5py.File(STAY_HDF5_PATH, "r") as f:
            jp = f["data/demo_0/obs/joint_positions"][0]  # 첫 프레임
        print(f"  [Stay] stay.hdf5 관절값 로드 완료 (shape: {jp.shape})")
        return jp.astype(np.float32)
    except Exception as e:
        print(f"  [Stay] stay.hdf5 로드 실패: {e} → 마지막 재생 자세 유지")
        return None

def _write_default_joint_state(robot):
    default_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos))
    robot.set_joint_position_target(default_pos)

# ── 메인 ──────────────────────────────────────────────────────────────────
def main():
    usd_path = FFW_SH5_CFG.spawn.usd_path
    if not os.path.exists(usd_path):
        raise FileNotFoundError(f"SH5 USD not found: {usd_path}")

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

    # 상자: 초기 Z=-10 (숨김), box_assets에서 랜덤 모델 선택
    box_usd = _get_box_usd("INITIAL")
    # ★ 상자 물리 설정: 시연 재생(Replay) 시 물리 간섭 방지를 위해 kinematic을 True로 설정
    # kinematic=True 이면 손가락과의 충돌 계산으로 인해 밀리거나 공중에 뜨는 현상(lag)이 사라짐
    _BOX_RIGID = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
        linear_damping=10.0,
        angular_damping=10.0,
        max_depenetration_velocity=0.1,
        enable_gyroscopic_forces=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=1,
    )
    _BOX_COLLISION = sim_utils.CollisionPropertiesCfg(contact_offset=0.0001, rest_offset=0.0)
    _BOX_MASS = sim_utils.MassPropertiesCfg(mass=0.001)

    if box_usd:
        box_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.UsdFileCfg(
                usd_path=box_usd,
                rigid_props=_BOX_RIGID,
                mass_props=_BOX_MASS,
                collision_props=_BOX_COLLISION,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, -10.0),
                rot=(1,0,0,0)
            ),
        )
    else:
        box_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.10,0.10,0.10),
                rigid_props=_BOX_RIGID,
                mass_props=_BOX_MASS,
                collision_props=_BOX_COLLISION,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85,0.38,0.08)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max", static_friction=2.0,
                    dynamic_friction=1.8, restitution=0.0),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0,0.0,-10.0), rot=(1,0,0,0)),
        )

    # ── 3대 로봇 + 3개 상자 씬 구성 ─────────────────────────────────────────
    scene_cfg = BringupSceneCfg(num_envs=1, env_spacing=2.0)

    for i, line_id in enumerate(LINE_IDS, 1):
        pos   = LINE_ROBOT_POS[line_id]
        r_cfg = deepcopy(FFW_SH5_CFG)
        r_cfg.spawn.rigid_props.disable_gravity = False
        r_cfg.init_state.pos = pos
        b_cfg = deepcopy(box_cfg)
        b_cfg = b_cfg.replace(prim_path=f"{{ENV_REGEX_NS}}/Box0{i}")
        setattr(scene_cfg, f"box0{i}",   b_cfg)
        setattr(scene_cfg, f"robot0{i}", r_cfg.replace(prim_path=f"{{ENV_REGEX_NS}}/Robot0{i}"))

    scene = InteractiveScene(scene_cfg)
    sim.reset(); scene.reset(); scene.update(sim.get_physics_dt())

    for i in range(1, 4):
        _write_default_joint_state(scene[f"robot0{i}"])
    scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())

    # 로봇 카메라 비활성화
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for i in range(1, 4):
            for cam in ["head_camera", "left_camera", "right_camera", "wrist_camera"]:
                p = stage.GetPrimAtPath(f"/World/envs/env_0/Robot0{i}/{cam}")
                if p and p.IsValid():
                    p.GetAttribute("visibility").Set("invisible")
        print("[Scene] 📵 3대 로봇 카메라 비활성화 완료")
    except Exception as e:
        print(f"[Scene] 카메라 비활성화 실패 (무시 가능): {e}")

    open(REPORT_REQ_FILE, "w").close()

    # ── 라인별 컨트롤러 생성 ─────────────────────────────────────────────────
    pkg_queue   = queue.Queue()
    file_reader = FileQueueReader(pkg_queue)
    line_queues: dict = {}
    controllers: dict = {}

    for i, line_id in enumerate(LINE_IDS, 1):
        rk      = f"robot0{i}"
        bk      = f"box0{i}"
        pos     = LINE_ROBOT_POS[line_id]
        robot_i = scene[rk]
        adp_i   = RobotAdapter(robot_i)
        slr_i   = SlotRegistry(max_slots=MAX_SLOTS)
        ctrl_i  = ReplayController(adp_i, scene, slr_i, robot_key=rk, box_key=bk)
        ctrl_i.set_camera(scene["topview_camera"])
        ctrl_i.set_home_pos(robot_i.data.default_joint_pos, home_base_pos=list(pos))
        line_queues[line_id] = queue.Queue()
        controllers[line_id] = ctrl_i

    sim_dt = sim.get_physics_dt()

    print("\n" + "="*60)
    print("  SH5 HDF5 Replay + ROS2 v3  ─  3-Robot Mode")
    for lid in LINE_IDS:
        print(f"  {lid} @ {LINE_ROBOT_POS[lid]}")
    print(f"  트리거: {QUEUE_FILE}")
    print(f"  작업대: {WS_TRIGGER_FILE}")
    print("="*60 + "\n")

    # 작업대 관리자 초기화
    try:
        import omni.usd as _omni_usd
        _stage = _omni_usd.get_context().get_stage()
        ws_manager = WorkstationManager(_stage)

        # ── Stage 스캔: 작업대 관련 prim 경로 자동 탐색 ──────────────────────
        # WORKSTATION_PRIM_PATTERN 설정을 위해 실제 경로를 출력합니다.
        # 올바른 경로를 확인 후 아래 WORKSTATION_PRIM_PATTERN 상수를 수정하세요.
        print("\n" + "─"*60)
        print("  [WS 디버그] Stage 내 작업대 관련 Prim 경로 탐색")
        print("─"*60)
        _KEYWORDS = ["rack", "shelf", "workstation", "ws", "cart",
                     "cabinet", "sg2_in", "locker", "bin", "unit"]
        _found_prims = []
        for _prim in _stage.Traverse():
            _path = str(_prim.GetPath()).lower()
            if any(kw in _path for kw in _KEYWORDS):
                _found_prims.append(str(_prim.GetPath()))
        if _found_prims:
            print(f"  총 {len(_found_prims)}개 발견:")
            for _p in sorted(_found_prims)[:60]:  # 최대 60개만 출력
                print(f"    {_p}")
            if len(_found_prims) > 60:
                print(f"    ... 외 {len(_found_prims)-60}개")
        else:
            print("  ⚠️ 작업대 관련 prim을 찾지 못했습니다.")
            print("  키워드를 직접 검색하려면:")
            print("  → Isaac Sim의 Stage 패널에서 작업대 클릭 후 경로 확인")
        print("─"*60 + "\n")

    except Exception as _e:
        print(f"[WS] 작업대 관리자 초기화 실패 (Isaac Sim 환경 필요): {_e}")
        ws_manager = None

    while simulation_app.is_running():
        # 작업대 Spawn/Despawn 트리거 폴링
        if ws_manager:
            ws_manager.poll()

        # 공유 큐 → 라인별 큐로 분배
        file_reader.poll()
        while True:
            try:
                payload = pkg_queue.get_nowait()
                lid = payload.get("target_line", "sg2_in_01")
                if lid in line_queues:
                    line_queues[lid].put(payload)
                else:
                    print(f"[Main] ⚠️ 알 수 없는 라인: {lid}")
            except queue.Empty:
                break

        for line_id, ctrl in controllers.items():
            lq = line_queues[line_id]
            if not ctrl.is_busy() and not lq.empty():
                payload     = lq.get_nowait()
                pkg_id      = payload.get("package_id", f"PKG_{int(time.time())}")
                customer_id = payload.get("customer_id") or payload.get("customer_name", "UNKNOWN")
                is_dup_from_bridge = payload.get("is_duplicate", None)

                print(f"\n{'='*50}")
                print(f"[{line_id}] 📩 {pkg_id} | {customer_id}")

                slot      = ctrl.slot_reg.assign(customer_id)
                hdf5_slot = slot if slot <= 4 else slot - 4
                episode   = _load_episode(hdf5_slot)
                if episode is None:
                    print(f"  [{line_id}] ❌ HDF5 없음 → 스킵")
                    continue

                home_pos = np.array(LINE_ROBOT_POS[line_id])
                rec_pos  = np.array(episode.get("robot_initial_pose", [0,0,0])[:3])
                offset   = np.array([home_pos[0]-rec_pos[0], home_pos[1]-rec_pos[1], 0.0])
                bi       = np.array(episode.get("box_initial_pose", [0.7, 0.0, 1.0])[:3])
                sp       = np.array([bi[0]+offset[0], bi[1]+offset[1], bi[2]])

                # ★ kinematic=True 유지 — _write_box_pose로 velocity 설정 없이 위치만 씀
                sp_pos  = torch.tensor(sp, dtype=torch.float32)
                sp_quat = torch.tensor(BOX_SPAWN_QUAT, dtype=torch.float32)
                ctrl._write_box_pose(sp_pos, sp_quat)
                print(f"  [{line_id}] 상자 스폰 @ ({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})")
                ctrl.start_scan(pkg_id, customer_id, line_id, episode, offset, is_dup_from_bridge)

        for ctrl in controllers.values():
            ctrl.step()

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
