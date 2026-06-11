"""
sh5_final.py  —  SH5 최종 통합 컨트롤러
=========================================

DB 담당자 확정 인터페이스 (2가지만):
  [수신]  /sim/sg2_spawn_trigger  (String/JSON)
          → {"package_id": "PKG_xxx", "target_line": "sg2_in_01", "timestamp": ...}
  [송신]  /report_inbound_progress  (Service)
          → workstation_id, robot_id, filled_slots_count, package_id,
             workstation_qr_id, package_qr_id

SH5 기능 3가지:
  1. QR 인식  → 상자 위치 & 슬롯 결정
  2. Pick & Place  (ACT_MODEL / HDF5_REPLAY / DUMMY_TELEPORT)
  3. DB 적재 보고

슬롯: 4칸 (1~4), 4칸 완충 시 자동 리셋 (DB가 회전/교체 처리)

실행:
  exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_final.py', encoding='utf-8').read())
"""

import os, sys, time, json, threading

# ============================================================
# Isaac Sim
# ============================================================
ISAAC_AVAILABLE = False
try:
    import omni.usd
    from pxr import UsdGeom, Sdf, Gf
    ISAAC_AVAILABLE = True
    print("[SH5] ✅ Isaac Sim")
except ImportError:
    print("[SH5] ⚠️ 외부 실행 (Mock 모드)")

# ============================================================
# ROS 2  —  ReportInboundProgress 서비스만 사용
# ============================================================
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, String
    from cobot3_interfaces.srv import ReportInboundProgress
    ROS2_AVAILABLE = True
    print("[SH5] ✅ ROS 2")
except ImportError:
    print("[SH5] ⚠️ ROS 2 없음 → Mock 모드")

# ============================================================
# HDF5 재생
# ============================================================
HDF5_AVAILABLE = False
try:
    sys.path.insert(0, '/home/rokey/dev_ws/coupang_ws/scripts')
    from hdf5_replay_player import pick_and_place_replay
    HDF5_AVAILABLE = True
    print("[SH5] ✅ HDF5 모듈")
except ImportError:
    print("[SH5] ⚠️ HDF5 없음")

# ============================================================
# ACT 모델
# ============================================================
TORCH_AVAILABLE = False
try:
    import torch
    from evaluate_test_vision import run_act_inference_step
    TORCH_AVAILABLE = True
    print("[SH5] ✅ ACT 모델")
except ImportError:
    print("[SH5] ⚠️ ACT 없음")

# ============================================================
# QR 스캐너  (sh5_qr_scanner.py 필요)
# ============================================================
QR_AVAILABLE = False
try:
    from sh5_qr_scanner import get_camera_image_rgb, decode_qr_from_image
    QR_AVAILABLE = True
    print("[SH5] ✅ QR 스캐너")
except ImportError:
    print("[SH5] ⚠️ QR 스캐너 없음 → package_id 폴백")

# ============================================================
# ★ 설정  —  이 블록만 수정
# ============================================================
PICK_AND_PLACE_MODE = "HDF5_REPLAY"
# "HDF5_REPLAY"    ← 내일 기본값 (VR 궤적 재생)
# "DUMMY_TELEPORT" ← 통신 테스트용 (순간이동)
# "ACT_MODEL"      ← AI 추론 (모델 완성 후)

ACT_MODEL_PATH = "/home/rokey/dev_ws/models/augmented_sh5_vision_act_20ep.pth"

USE_QR_CAMERA = False   # True: 카메라 QR 스캔 / False: package_id 그대로 사용

# ★ 순차 실행 모드 (FPS 보호)
# True  → 한 대가 완전히 끝난 뒤 다음 대 시작 (권장, FPS 안정)
# False → 3대 동시 실행 (빠르지만 FPS 하락 위험)
SEQUENTIAL_MODE = True

# Top-View 카메라 Prim 경로 (find_camera_prims()로 확인)
CAMERA_PRIMS = {
    "sg2_in_01": "/World/TopCamera_Line01",
    "sg2_in_02": "/World/TopCamera_Line02",
    "sg2_in_03": "/World/TopCamera_Line03",
}

# ============================================================
# 물리 좌표  (PHYSICAL_LAYOUT.md 기준)
# ============================================================
CONVEYOR_SPAWN = {
    "sg2_in_01": (9.0,  1.5, 0.83),
    "sg2_in_02": (9.0, -3.0, 0.83),
    "sg2_in_03": (9.0, -7.5, 0.83),
}
ROBOT_POS = {
    "sg2_in_01": (7.5,  3.0, 0.0),
    "sg2_in_02": (7.5, -1.5, 0.0),
    "sg2_in_03": (7.5, -6.0, 0.0),
}
# 작업대 슬롯 위치 (로봇 기준 상대 좌표, 4칸)
SLOT_LOCAL = {
    1: ( 0.0, -1.5, 1.2),
    2: ( 0.0, -1.5, 0.9),
    3: ( 0.0, -1.5, 0.6),
    4: ( 0.0, -1.5, 0.3),
}
# 작업대 ID 매핑 (라인 → 워크스테이션)
# ★ DB init.sql 기준: WS01 (언더스코어 없음)
WORKSTATION_ID = {
    "sg2_in_01": "WS01",
    "sg2_in_02": "WS02",
    "sg2_in_03": "WS03",
}
WORKSTATION_QR = {
    "sg2_in_01": "WORKSTATION_WS01",
    "sg2_in_02": "WORKSTATION_WS02",
    "sg2_in_03": "WORKSTATION_WS03",
}

BOX_USD = "/home/rokey/dev_ws/assets/sh5_box.usd"

# ============================================================
# Isaac Sim 유틸
# ============================================================
def _stage():
    if not ISAAC_AVAILABLE:
        return None
    try:
        return omni.usd.get_context().get_stage()
    except:
        return None


def spawn_box(line_id: str, pkg_id: str) -> str:
    pos   = CONVEYOR_SPAWN[line_id]
    safe  = pkg_id.replace("-","_").replace(" ","_")
    path  = f"/World/SH5Box_{line_id[-2:]}_{safe}"
    stage = _stage()

    if stage is None:
        print(f"  [Spawn] Mock: {path} @ {pos}")
        return path

    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(Sdf.Path(path))

    if os.path.exists(BOX_USD):
        p = stage.DefinePrim(path, "Xform")
        p.GetReferences().AddReference(BOX_USD)
    else:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(0.12)

    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    print(f"  [Spawn] 📦 {path}")
    return path


def teleport_to_slot(box_path: str, robot_pos: tuple, slot: int) -> bool:
    local = SLOT_LOCAL[slot]
    world = (robot_pos[0]+local[0], robot_pos[1]+local[1], local[2])
    stage = _stage()
    if stage is None:
        print(f"  [TP] Mock 슬롯{slot} → {world}")
        return True
    prim = stage.GetPrimAtPath(box_path)
    if not prim.IsValid():
        return False
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*world))
    print(f"  [TP] 🚀 슬롯{slot} → {world}")
    return True


# ============================================================
# QR 인식  —  상자 위치 & package_id 확정
# ============================================================
def scan_qr(line_id: str, fallback_id: str) -> tuple[str, str]:
    """
    카메라로 QR 스캔 → (package_id, qr_id) 반환.

    DB 규칙 (packages_2026-06-xx.csv 기준):
      qr_id      = "QR_20260612_001"   ← 상자에 붙은 실제 QR
      package_id = "PKG_20260612_001"  ← DB 내부 PK (QR_ → PKG_ 치환)

    USE_QR_CAMERA=False 시:
      fallback_id = BG2 토픽의 package_id ("PKG_20260612_001")
      → qr_id는 QR_ 접두사로 역변환하여 반환
    """
    if not USE_QR_CAMERA or not QR_AVAILABLE:
        # package_id(PKG_...) → qr_id(QR_...) 역변환
        if fallback_id.startswith("PKG_"):
            qr_id = fallback_id.replace("PKG_", "QR_")
        else:
            qr_id = fallback_id
        return fallback_id, qr_id

    cam = CAMERA_PRIMS.get(line_id)
    if not cam:
        qr_id = fallback_id.replace("PKG_", "QR_") if fallback_id.startswith("PKG_") else fallback_id
        return fallback_id, qr_id

    print(f"  [QR] 카메라 스캔: {cam}")
    img = get_camera_image_rgb(cam)
    if img is None:
        print(f"  [QR] 이미지 없음 → 폴백")
        qr_id = fallback_id.replace("PKG_", "QR_") if fallback_id.startswith("PKG_") else fallback_id
        return fallback_id, qr_id

    results = decode_qr_from_image(img)
    if not results:
        print(f"  [QR] 미검출 → 폴백")
        qr_id = fallback_id.replace("PKG_", "QR_") if fallback_id.startswith("PKG_") else fallback_id
        return fallback_id, qr_id

    qr_text, (px, py) = results[0]   # qr_text = "QR_20260612_001"
    print(f"  [QR] ✅ 인식: '{qr_text}' @ 픽셀({px},{py})")

    # ★ QR_ → PKG_ 변환 (DB packages.package_id 형식)
    if qr_text.startswith("QR_"):
        pkg_id = qr_text.replace("QR_", "PKG_")
    else:
        pkg_id = qr_text

    return pkg_id, qr_text  # (PKG_20260612_001, QR_20260612_001)


# ============================================================
# Pick & Place  —  3가지 모드
# ============================================================
def do_pick_and_place(box_path: str, robot_pos: tuple, slot: int,
                      robot_art=None) -> bool:
    """
    픽앤플레이스 실행.
    ACT → HDF5 → Teleport 순으로 자동 폴백.
    """
    mode = PICK_AND_PLACE_MODE

    # --- ACT_MODEL ---
    if mode == "ACT_MODEL":
        if TORCH_AVAILABLE:
            print(f"  [P&P] 🤖 ACT 추론 슬롯{slot}")
            try:
                ok = run_act_inference_step(
                    model_path=ACT_MODEL_PATH,
                    slot_num=slot,
                    robot_articulation=robot_art,
                    box_prim_path=box_path,
                    robot_world_pos=robot_pos,
                )
                if ok:
                    return True
                print(f"  [P&P] ACT 실패 → HDF5 폴백")
            except Exception as e:
                print(f"  [P&P] ACT 오류: {e} → HDF5 폴백")
        else:
            print(f"  [P&P] ACT 모듈 없음 → HDF5 폴백")

        if HDF5_AVAILABLE:
            ok = pick_and_place_replay(slot_num=slot,
                                       robot_articulation=robot_art,
                                       box_prim_path=box_path,
                                       realtime=True,
                                       robot_world_pos=robot_pos)  # ★ offset 전달
            if ok:
                return True
        print(f"  [P&P] 최종 폴백 → Teleport")
        return teleport_to_slot(box_path, robot_pos, slot)

    # --- HDF5_REPLAY ---
    elif mode == "HDF5_REPLAY" and HDF5_AVAILABLE:
        print(f"  [P&P] 🎬 HDF5 재생 슬롯{slot}")
        ok = pick_and_place_replay(slot_num=slot,
                                   robot_articulation=robot_art,
                                   box_prim_path=box_path,
                                   realtime=True,
                                   robot_world_pos=robot_pos)  # ★ offset 전달
        if ok:
            return True
        print(f"  [P&P] HDF5 실패 → Teleport 폴백")
        return teleport_to_slot(box_path, robot_pos, slot)

    # --- DUMMY_TELEPORT ---
    else:
        print(f"  [P&P] 🚀 Dummy Teleport 슬롯{slot}")
        return teleport_to_slot(box_path, robot_pos, slot)


# ============================================================
# SH5 라인 작업 단위
# ============================================================
class SH5Line:
    """
    라인 1개(sg2_in_0X)를 담당하는 단위.
    수신 → QR 스캔 → Pick&Place → 보고 → 반복
    """
    def __init__(self, line_id: str, db_node=None):
        self.line_id    = line_id
        self.robot_pos  = ROBOT_POS[line_id]
        self.ws_id      = WORKSTATION_ID[line_id]
        self.ws_qr      = WORKSTATION_QR[line_id]
        self.db         = db_node
        self.robot_art  = None         # Isaac Sim Articulation (나중에 연결)
        self.filled     = 0            # 현재 채워진 슬롯 수 (0~4)
        self._paused    = False
        self._busy      = False        # ★ 순차 모드용 작업 중 플래그

        import queue
        self.queue = queue.Queue()
        print(f"[{line_id}] 초기화 | WS={self.ws_id} | 모드={PICK_AND_PLACE_MODE}")

    def next_slot(self) -> int:
        return self.filled + 1   # 1~4

    def step(self):
        if self._paused or self._busy:
            return

        try:
            payload = self.queue.get_nowait()
        except:
            return

        self._busy = True
        fallback_id = payload.get("package_id", f"PKG_MOCK_{int(time.time())}")
        print(f"\n[{self.line_id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[{self.line_id}] 📩 수신: {fallback_id}")

        try:
            # ─── STEP 1: 상자 리스폰 ──────────────────────────
            box_path = spawn_box(self.line_id, fallback_id)
            time.sleep(0.4)

            # ─── STEP 2: QR 인식 → package_id, qr_id 확정 ────
            pkg_id, qr_id = scan_qr(self.line_id, fallback_id)
            print(f"[{self.line_id}] 🏷️  pkg={pkg_id} | qr={qr_id}")

            # 슬롯 결정 (순차: 1→2→3→4)
            slot = self.next_slot()

            # ─── STEP 3: Pick & Place ─────────────────────────
            ok = do_pick_and_place(box_path, self.robot_pos, slot, self.robot_art)
            if not ok:
                print(f"[{self.line_id}] ❌ P&P 실패 — 이번 패키지 스킵")
                return

            # ─── STEP 4: DB 보고 ──────────────────────────────
            self.filled += 1
            self._report(pkg_id, qr_id, slot)

            # 4칸 완충 → 슬롯 카운터 리셋 (DB가 회전/교체 처리)
            if self.filled >= 4:
                print(f"[{self.line_id}] 🔄 4칸 완충 → 슬롯 리셋 (DB가 회전 처리)")
                self.filled = 0
        finally:
            self._busy = False   # ★ 작업 완료 → 다음 상자 수신 가능

    def _report(self, pkg_id: str, qr_id: str, slot: int):
        """ReportInboundProgress 서비스 호출"""
        if not self.db or not ROS2_AVAILABLE:
            print(f"  [보고] Mock — pkg={pkg_id} slot={slot}")
            return
        try:
            if not self.db.report_client.wait_for_service(timeout_sec=1.0):
                print(f"  [보고] 서비스 없음 — 스킵")
                return

            req = ReportInboundProgress.Request()
            req.workstation_id     = self.ws_id          # "WS_01"
            req.robot_id           = self.line_id         # "sg2_in_01"
            req.filled_slots_count = slot                 # 1~4 (슬롯 번호)
            req.package_id         = pkg_id               # "PKG_xxx"
            req.workstation_qr_id  = self.ws_qr           # "WORKSTATION_WS01"
            req.package_qr_id      = qr_id                # QR 스캔값 or pkg_id

            fut = self.db.report_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, fut, timeout_sec=2.0)

            if fut.result() and fut.result().success:
                print(f"  [보고] ✅ DB 갱신 완료 | {self.ws_id} 슬롯{slot} ← {pkg_id}")
            else:
                print(f"  [보고] ⚠️ DB 응답 실패")
        except Exception as e:
            print(f"  [보고] 오류: {e}")


# ============================================================
# ROS 2 노드  —  토픽 수신 + 보고 서비스
# ============================================================
class SH5Node(Node):
    def __init__(self, lines: list):
        super().__init__("sh5_final_node")
        self.line_map = {l.line_id: l for l in lines}

        # ① ReportInboundProgress 서비스 클라이언트
        self.report_client = self.create_client(
            ReportInboundProgress, "/report_inbound_progress"
        )

        # ② /sim/sg2_spawn_trigger 구독
        self.create_subscription(
            String, "/sim/sg2_spawn_trigger",
            self._on_spawn_trigger, 10
        )

        # ③ /{robot_id}/pause_status 구독 (DB가 회전 시 발행)
        for lid in self.line_map:
            self.create_subscription(
                Bool, f"/{lid}/pause_status",
                lambda msg, l=lid: self._on_pause(msg, l), 10
            )

        print("[Node] ✅ sh5_final_node 가동")
        print("[Node] 📡 /sim/sg2_spawn_trigger 대기 중")

    def _on_spawn_trigger(self, msg: String):
        try:
            payload = json.loads(msg.data)
            line_id = payload.get("target_line", "")
            pkg_id  = payload.get("package_id", "")
            print(f"\n[Node] 🚨 트리거 수신: {pkg_id} → {line_id}")
            unit = self.line_map.get(line_id)
            if unit:
                unit.queue.put(payload)
            else:
                print(f"[Node] ⚠️ 알 수 없는 라인: {line_id}")
        except Exception as e:
            print(f"[Node] 파싱 오류: {e}")

    def _on_pause(self, msg: Bool, line_id: str):
        unit = self.line_map.get(line_id)
        if unit:
            unit._paused = msg.data
            state = "⏸️ 일시정지 (DB 회전 중)" if msg.data else "▶️ 재개"
            print(f"[Node] {state}: {line_id}")


# ============================================================
# Mock 자동 투입 (ROS2 없을 때 테스트용)
# ============================================================
def _mock_loop(lines: list):
    from itertools import cycle
    import random
    pool = cycle(lines)
    while True:
        time.sleep(4.0)
        unit = next(pool)
        pkg  = f"PKG_MOCK_{int(time.time())}"
        unit.queue.put({"package_id": pkg, "target_line": unit.line_id})
        print(f"\n[Mock] 🎲 {pkg} → {unit.line_id}")


# ============================================================
# 메인 컨트롤러
# ============================================================
class SH5Controller:
    def __init__(self):
        print("\n" + "="*60)
        print("  SH5 최종 통합 컨트롤러  (sh5_final.py)")
        print(f"  모드: {PICK_AND_PLACE_MODE}")
        print(f"  QR 카메라: {'ON' if USE_QR_CAMERA else 'OFF (package_id 폴백)'}")
        print("="*60)

        self.lines = [
            SH5Line("sg2_in_01"),
            SH5Line("sg2_in_02"),
            SH5Line("sg2_in_03"),
        ]

        self.ros_node = None
        if ROS2_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            self.ros_node = SH5Node(self.lines)
            for l in self.lines:
                l.db = self.ros_node
            t = threading.Thread(
                target=lambda: rclpy.spin(self.ros_node), daemon=True)
            t.start()
            print("[Controller] ✅ ROS 2 스핀 가동")
        else:
            print("[Controller] ⚠️ ROS 2 없음 → Mock 자동 투입")
            threading.Thread(target=_mock_loop, args=(self.lines,), daemon=True).start()

        print("[Controller] 🚀 시연 준비 완료!")
        print("[Controller]    /sim/sg2_spawn_trigger 수신 대기 중...")

    def tick(self):
        """Isaac Sim Update 콜백에서 호출"""
        if SEQUENTIAL_MODE:
            # ★ 순차 모드: 현재 작업 중인 라인이 있으면 다른 라인 시작 안 함
            any_busy = any(l._busy for l in self.lines)
            if any_busy:
                return   # 작업 중인 로봇이 있으면 다른 로봇 대기
            # 작업 중인 로봇 없음 → 큐에 대기 중인 첫 번째 라인만 실행
            for l in self.lines:
                if not l.queue.empty() and not l._paused:
                    l.step()
                    break   # 한 대만 시작 후 이번 tick 종료
        else:
            # 동시 모드 (기존 동작)
            for l in self.lines:
                l.step()

    def loop(self, interval=0.1, cycles=99999):
        for _ in range(cycles):
            self.tick()
            time.sleep(interval)


# ============================================================
# 진입점
# ============================================================
print("\n[SH5 Final] 초기화 중...")
controller = SH5Controller()

if ISAAC_AVAILABLE:
    try:
        import omni.kit.app
        _sub = omni.kit.app.get_app().get_update_event_stream() \
                   .create_subscription_to_pop(
                       lambda e: controller.tick(),
                       name="sh5_final_update"
                   )
        print("[SH5 Final] ✅ Isaac Sim 콜백 등록 완료")
        print()
        print("  수동 투입 (테스트):  controller.lines[0].queue.put({'package_id':'PKG_TEST','target_line':'sg2_in_01'})")
        print("  상태 확인:           [(l.line_id, l.filled) for l in controller.lines]")
    except Exception as e:
        print(f"[SH5 Final] 콜백 실패: {e} → 루프 모드")
        controller.loop()
else:
    controller.loop()
