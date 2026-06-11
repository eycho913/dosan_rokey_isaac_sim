"""
sh5_spawn_controller_qr.py
==========================
[QR 카메라 인식 버전]

기존 sh5_spawn_controller.py와 동일하지만,
상자 리스폰 후 Top-View 카메라로 실제 QR을 스캔하여
package_id/qr_id를 확정한 뒤 DB 서비스를 호출합니다.

변경된 흐름:
  기존: BG2 신호의 package_id → 그대로 DB 호출
  QR버전: BG2 신호 수신 → 리스폰 → 카메라 QR 스캔 → DB 호출

실행 (Isaac Sim Script Editor):
  exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_spawn_controller_qr.py', encoding='utf-8').read())
"""

import os, sys, time, random, json, threading

# ============================================================
# Isaac Sim
# ============================================================
ISAAC_AVAILABLE = False
try:
    import omni.usd
    from pxr import UsdGeom, Sdf, Gf
    ISAAC_AVAILABLE = True
    print("[SH5-QR] ✅ Isaac Sim 연결")
except ImportError:
    print("[SH5-QR] ⚠️ Isaac Sim 외부 (디버그 모드)")

# ============================================================
# ROS 2
# ============================================================
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from std_msgs.msg import Bool, String
    from cobot3_interfaces.srv import CheckWarehouseStatus, ReportInboundProgress
    from cobot3_interfaces.action import MovePackage, ManageWorkstation
    ROS2_AVAILABLE = True
    print("[SH5-QR] ✅ ROS 2 연결")
except ImportError:
    print("[SH5-QR] ⚠️ ROS 2 없음")

# ============================================================
# HDF5 재생
# ============================================================
HDF5_REPLAY_AVAILABLE = False
try:
    sys.path.insert(0, '/home/rokey/dev_ws/coupang_ws/scripts')
    from hdf5_replay_player import pick_and_place_replay
    HDF5_REPLAY_AVAILABLE = True
    print("[SH5-QR] ✅ HDF5 모듈 로드")
except ImportError:
    print("[SH5-QR] ⚠️ HDF5 없음")

# ============================================================
# QR 스캐너 (Plan B)
# ============================================================
QR_SCANNER_AVAILABLE = False
try:
    from sh5_qr_scanner import SH5QRScanner, get_camera_image_rgb, decode_qr_from_image
    QR_SCANNER_AVAILABLE = True
    print("[SH5-QR] ✅ QR 스캐너 모듈 로드")
except ImportError:
    print("[SH5-QR] ⚠️ QR 스캐너 없음 → package_id 폴백 사용")

# ============================================================
# ACT 모델
# ============================================================
TORCH_AVAILABLE = False
try:
    import torch
    from evaluate_test_vision import run_act_inference_step
    TORCH_AVAILABLE = True
    print("[SH5-QR] ✅ ACT 모듈 로드")
except ImportError:
    print("[SH5-QR] ⚠️ ACT 모듈 없음")

# ============================================================
# ★ 시연 모드 설정
# ============================================================
PICK_AND_PLACE_MODE = "HDF5_REPLAY"   # HDF5_REPLAY | DUMMY_TELEPORT | ACT_MODEL
ACT_MODEL_PATH = "/home/rokey/dev_ws/models/augmented_sh5_vision_act_20ep.pth"

# QR 스캔 활성화 여부 (False 시 기존 sh5_spawn_controller.py와 동일 동작)
USE_QR_CAMERA = True

# ============================================================
# 좌표 설정
# ============================================================
CONVEYOR_SPAWN_POSITIONS = {
    "sg2_in_01": (9.0,  1.5, 0.83),
    "sg2_in_02": (9.0, -3.0, 0.83),
    "sg2_in_03": (9.0, -7.5, 0.83),
}
ROBOT_POSITIONS = {
    "sg2_in_01": (7.5,  3.0),
    "sg2_in_02": (7.5, -1.5),
    "sg2_in_03": (7.5, -6.0),
}
SLOT_TARGETS_LOCAL = {
    1: (0.0, -1.5, 1.2),
    2: (0.0, -1.5, 1.2),
    3: (0.0, -1.5, 0.5),
    4: (0.0, -1.5, 0.5),
}

# Top-View 카메라 Prim 경로 (find_camera_prims()로 확인 후 수정)
CAMERA_PRIMS = {
    "sg2_in_01": "/World/TopCamera_Line01",
    "sg2_in_02": "/World/TopCamera_Line02",
    "sg2_in_03": "/World/TopCamera_Line03",
}

BOX_USD_PATH = "/home/rokey/dev_ws/assets/sh5_box.usd"

# ============================================================
# Isaac Sim 유틸
# ============================================================
def get_stage():
    if not ISAAC_AVAILABLE:
        return None
    try:
        return omni.usd.get_context().get_stage()
    except:
        return None


def respawn_box(line_id: str, package_id: str) -> str:
    stage = get_stage()
    spawn_pos = CONVEYOR_SPAWN_POSITIONS.get(line_id, (9.0, 0.0, 0.83))
    safe_id = package_id.replace('-', '_').replace(' ', '_')
    prim_path = f"/World/QRBox_{line_id.split('_')[-1]}_{safe_id}"

    if stage is None:
        print(f"  [Spawn] Mock: {prim_path} @ {spawn_pos}")
        return prim_path

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))

    if os.path.exists(BOX_USD_PATH):
        p = stage.DefinePrim(prim_path, "Xform")
        p.GetReferences().AddReference(BOX_USD_PATH)
    else:
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.GetSizeAttr().Set(0.12)

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*spawn_pos))
    print(f"  [Spawn] 📦 리스폰: {prim_path} @ {spawn_pos}")
    return prim_path


def teleport_box_to_slot(box_prim_path: str, robot_world_pos: tuple, slot_num: int) -> bool:
    local = SLOT_TARGETS_LOCAL.get(slot_num, SLOT_TARGETS_LOCAL[1])
    world = (robot_world_pos[0] + local[0], robot_world_pos[1] + local[1], local[2])
    stage = get_stage()
    if stage is None:
        print(f"  [Teleport] Mock 슬롯{slot_num} → {world}")
        return True
    prim = stage.GetPrimAtPath(box_prim_path)
    if not prim.IsValid():
        return False
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*world))
    print(f"  [Teleport] 🚀 슬롯{slot_num} → {world}")
    return True


# ============================================================
# ★ 핵심 추가: QR 카메라 스캔
# ============================================================
def scan_qr_from_camera(line_id: str, fallback_pkg_id: str) -> tuple[str, str]:
    """
    Top-View 카메라로 상자 QR을 스캔하여 (package_id, qr_id) 반환.
    스캔 실패 시 BG2 신호에서 받은 fallback_pkg_id 사용.

    Returns:
        (package_id, qr_id)
    """
    if not USE_QR_CAMERA or not QR_SCANNER_AVAILABLE:
        return fallback_pkg_id, fallback_pkg_id

    cam_path = CAMERA_PRIMS.get(line_id)
    if not cam_path:
        return fallback_pkg_id, fallback_pkg_id

    print(f"  [QR Scan] 📷 카메라 스캔 시작: {cam_path}")

    # 카메라 이미지 획득
    image = get_camera_image_rgb(cam_path)
    if image is None:
        print(f"  [QR Scan] ⚠️ 이미지 없음 → 폴백: {fallback_pkg_id}")
        return fallback_pkg_id, fallback_pkg_id

    # QR 디코딩
    results = decode_qr_from_image(image)
    if not results:
        print(f"  [QR Scan] ⚠️ QR 미검출 → 폴백: {fallback_pkg_id}")
        return fallback_pkg_id, fallback_pkg_id

    # 첫 번째 QR 텍스트 사용
    qr_text, (px, py) = results[0]
    print(f"  [QR Scan] ✅ QR 인식 성공: '{qr_text}' @ 픽셀({px}, {py})")

    # QR 텍스트에서 package_id 추출
    # 포맷: "PKG_20260612_001" 또는 "QR_20260612_001"
    if qr_text.startswith("QR_"):
        package_id = qr_text.replace("QR_", "PKG_")
        qr_id = qr_text
    elif qr_text.startswith("PKG_"):
        package_id = qr_text
        qr_id = qr_text
    else:
        # 알 수 없는 포맷 → 그대로 사용
        package_id = qr_text
        qr_id = qr_text

    return package_id, qr_id


# ============================================================
# SH5 라인 작업 단위 (QR 버전)
# ============================================================
class SH5LineUnitQR:
    """
    QR 카메라 인식 버전의 SH5 라인 작업 단위.

    기존 SH5LineUnit과의 차이점:
      step() 내에서 상자 리스폰 후 카메라 QR 스캔을 수행하고,
      스캔된 실제 qr_id를 DB 서비스 호출에 사용.
    """
    def __init__(self, line_id: str, db_node=None):
        self.line_id = line_id
        self.robot_world_pos = ROBOT_POSITIONS[line_id] + (0.0,)
        self.db = db_node
        self.slots = {1: None, 2: None, 3: None, 4: None}
        self.current_box_prim = ""
        self.current_package_id = ""
        self.current_qr_id = ""          # ★ QR 스캔으로 획득한 실제 QR ID
        self._is_paused = False

        import queue
        self.pending_packages = queue.Queue()
        print(f"[{line_id}] 🤖 초기화 (QR카메라 버전) | USE_QR_CAMERA={USE_QR_CAMERA}")
        print(f"[{line_id}]    카메라: {CAMERA_PRIMS.get(line_id, 'N/A')}")

    def _find_free_slot(self):
        for k, v in self.slots.items():
            if v is None:
                return k
        return None

    def step(self):
        if self._is_paused:
            return

        try:
            payload = self.pending_packages.get_nowait()
        except:
            return

        # BG2 신호에서 받은 package_id (폴백용)
        fallback_pkg_id = payload.get('package_id', f"PKG_MOCK_{int(time.time())}")

        print(f"\n[{self.line_id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[{self.line_id}] 📦 BG2 수신: {fallback_pkg_id}")

        # ① 리스폰 (BG2 신호의 package_id로 임시 스폰)
        self.current_box_prim = respawn_box(self.line_id, fallback_pkg_id)
        time.sleep(0.5)   # 물리 안정화

        # ② ★ QR 카메라 스캔 (실제 package_id, qr_id 확정)
        self.current_package_id, self.current_qr_id = scan_qr_from_camera(
            self.line_id, fallback_pkg_id
        )
        print(f"[{self.line_id}] 🏷️ 확정 ID: package={self.current_package_id} | qr={self.current_qr_id}")

        # ③ DB 중복 검사 (스캔된 qr_id 사용)
        if self.db and ROS2_AVAILABLE:
            if self._db_check_duplicate():
                print(f"[{self.line_id}] ⚠️ 중복 감지 → AMR 직송")
                self._send_move_package()
                self.current_box_prim = ""
                return

        # ④ 빈 슬롯 배정
        slot = self._find_free_slot()
        if slot is None:
            print(f"[{self.line_id}] 🏭 만석 → 작업대 교체 요청")
            self._send_manage_workstation()
            self.slots = {1: None, 2: None, 3: None, 4: None}
            slot = 1

        self.slots[slot] = self.current_qr_id  # ★ QR ID로 슬롯 기록
        print(f"[{self.line_id}] ✅ 슬롯 {slot} 배정 (QR: {self.current_qr_id})")

        # ⑤ 픽앤플레이스
        self._do_pick_and_place(slot)

        # ⑥ 적재 보고 (스캔된 qr_id 사용)
        filled = sum(1 for v in self.slots.values() if v is not None)
        self._db_report(filled, slot)
        self.current_box_prim = ""

    def _do_pick_and_place(self, slot: int):
        mode = PICK_AND_PLACE_MODE

        if mode == "ACT_MODEL":
            if TORCH_AVAILABLE:
                print(f"[{self.line_id}] 🤖 ACT 추론 (슬롯 {slot})")
                try:
                    ok = run_act_inference_step(
                        model_path=ACT_MODEL_PATH,
                        slot_num=slot,
                        robot_articulation=getattr(self, 'robot', None),
                        box_prim_path=self.current_box_prim,
                        robot_world_pos=self.robot_world_pos,
                    )
                    if ok:
                        return
                    print(f"[{self.line_id}] ❌ ACT 실패 → HDF5 폴백")
                except Exception as e:
                    print(f"[{self.line_id}] ❌ ACT 오류: {e} → HDF5 폴백")
            if HDF5_REPLAY_AVAILABLE:
                ok = pick_and_place_replay(
                    slot_num=slot,
                    robot_articulation=getattr(self, 'robot', None),
                    box_prim_path=self.current_box_prim,
                    realtime=True,
                )
                if ok:
                    return
            teleport_box_to_slot(self.current_box_prim, self.robot_world_pos, slot)

        elif mode == "HDF5_REPLAY" and HDF5_REPLAY_AVAILABLE:
            print(f"[{self.line_id}] 🎬 HDF5 재생 (슬롯 {slot})")
            ok = pick_and_place_replay(
                slot_num=slot,
                robot_articulation=getattr(self, 'robot', None),
                box_prim_path=self.current_box_prim,
                realtime=True,
            )
            if not ok:
                teleport_box_to_slot(self.current_box_prim, self.robot_world_pos, slot)
        else:
            print(f"[{self.line_id}] 🚀 Dummy Teleport (슬롯 {slot})")
            teleport_box_to_slot(self.current_box_prim, self.robot_world_pos, slot)

    # ── DB 통신 (QR ID 사용) ─────────────────────────
    def _db_check_duplicate(self) -> bool:
        try:
            if not self.db.check_client.wait_for_service(timeout_sec=0.5):
                return False
            req = CheckWarehouseStatus.Request()
            req.package_id    = self.current_package_id   # ★ 스캔된 package_id
            req.customer_name = ""
            req.qr_id         = self.current_qr_id        # ★ 스캔된 qr_id
            fut = self.db.check_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, fut, timeout_sec=1.0)
            result = fut.result()
            if result and result.is_already_in_warehouse:
                print(f"  [DB] 중복 감지! QR={self.current_qr_id}")
                return True
            return False
        except Exception as e:
            print(f"  [DB] 중복검사 오류: {e}")
            return False

    def _db_report(self, filled: int, slot: int):
        if not self.db or not ROS2_AVAILABLE:
            return
        try:
            if not self.db.report_client.wait_for_service(timeout_sec=0.5):
                return
            req = ReportInboundProgress.Request()
            req.workstation_id    = f"WS_{self.line_id[-2:]}"
            req.robot_id          = self.line_id
            req.filled_slots_count = filled
            req.package_id        = self.current_package_id   # ★ 스캔된 ID
            req.workstation_qr_id = f"WORKSTATION_WS_{self.line_id[-2:]}"
            req.package_qr_id     = self.current_qr_id        # ★ 스캔된 QR ID
            fut = self.db.report_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, fut, timeout_sec=1.0)
            if fut.result() and fut.result().success:
                print(f"  [DB] ✅ 슬롯{slot} 보고 완료 | QR={self.current_qr_id}")
        except Exception as e:
            print(f"  [DB] 보고 오류: {e}")

    def _send_move_package(self):
        if not self.db or not ROS2_AVAILABLE:
            return
        self.db.send_move_package(
            package_id=self.current_package_id,
            qr_id=self.current_qr_id,       # ★ QR ID 추가 전달
            line_id=self.line_id,
        )

    def _send_manage_workstation(self):
        if not self.db or not ROS2_AVAILABLE:
            return
        self.db.send_manage_workstation(
            f"WS_{self.line_id[-2:]}",
            int(self.line_id[-2:])
        )


# ============================================================
# ROS 2 DB 노드 (QR 버전)
# ============================================================
class SH5DBNodeQR(Node):
    def __init__(self, units: list):
        super().__init__('sh5_qr_db_node')
        self.units = units
        self.line_map = {u.line_id: u for u in units}

        self.check_client  = self.create_client(CheckWarehouseStatus, '/check_warehouse_status')
        self.report_client = self.create_client(ReportInboundProgress, '/report_inbound_progress')
        self.move_pkg_client  = ActionClient(self, MovePackage, '/move_package')
        self.manage_ws_client = ActionClient(self, ManageWorkstation, '/manage_workstation')

        for line_id in CONVEYOR_SPAWN_POSITIONS:
            self.create_subscription(
                Bool, f'/{line_id}/pause_status',
                lambda msg, lid=line_id: self._pause_cb(msg, lid), 10
            )

        self.create_subscription(
            String, '/sim/sg2_spawn_trigger',
            self._spawn_trigger_cb, 10
        )
        print("[DB Node QR] ✅ 초기화 완료")

    def _pause_cb(self, msg, line_id: str):
        unit = self.line_map.get(line_id)
        if unit:
            unit._is_paused = msg.data
            print(f"[DB Node QR] {'⏸️' if msg.data else '▶️'} {line_id}")

    def _spawn_trigger_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            target_line = payload.get('target_line', '')
            unit = self.line_map.get(target_line)
            if unit:
                unit.pending_packages.put(payload)
                print(f"[DB Node QR] 📩 큐 등록: {payload.get('package_id')} → {target_line}")
        except Exception as e:
            print(f"[DB Node QR] 파싱 오류: {e}")

    def send_move_package(self, package_id: str, qr_id: str, line_id: str):
        """★ QR ID 포함하여 AMR에 직송 명령"""
        if not self.move_pkg_client.wait_for_server(timeout_sec=2.0):
            print("[Action] MovePackage 서버 없음")
            return
        goal = MovePackage.Goal()
        goal.package_id       = package_id
        goal.customer_name    = ""
        goal.destination_zone = "MAIN_storage"
        goal.package_qr_id   = qr_id   # ★ 실제 스캔된 QR ID

        def _fb(fb):
            print(f"  [AMR] {fb.feedback.current_position} | {fb.feedback.progress:.1f}%")

        print(f"[Action] 🚛 MovePackage: {package_id} (QR:{qr_id}) → MAIN_storage")
        self.move_pkg_client.send_goal_async(goal, feedback_callback=_fb)

    def send_manage_workstation(self, ws_id: str, unit_idx: int):
        if not self.manage_ws_client.wait_for_server(timeout_sec=2.0):
            print("[Action] ManageWorkstation 서버 없음")
            return
        goal = ManageWorkstation.Goal()
        goal.workstation_id    = ws_id
        goal.start_location    = f"sg2_in_{unit_idx:02d}_A"
        goal.target_location   = "warehouse"
        goal.workstation_qr_id = f"WORKSTATION_WS_{unit_idx:02d}"
        goal.target_qr_id      = ""
        goal.target_x, goal.target_y, goal.target_yaw = 1.5, 3.0, 0.0

        def _fb(fb):
            print(f"  [AMR] 남은거리: {fb.feedback.distance_remaining:.2f}m | {fb.feedback.status}")

        print(f"[Action] 🏭 ManageWorkstation: {ws_id}")
        self.manage_ws_client.send_goal_async(goal, feedback_callback=_fb)


# ============================================================
# Mock 자동 투입 (ROS2 없을 때)
# ============================================================
def _mock_trigger_loop(units: list):
    line_ids = list(CONVEYOR_SPAWN_POSITIONS.keys())
    idx = 0
    while True:
        time.sleep(5.0)
        line_id = line_ids[idx % len(line_ids)]
        pkg_id = f"PKG_MOCK_{int(time.time())}"
        for u in units:
            if u.line_id == line_id:
                u.pending_packages.put({'package_id': pkg_id, 'target_line': line_id})
                print(f"\n[Mock] 🎲 가상 투입: {pkg_id} → {line_id}")
                break
        idx += 1


# ============================================================
# 메인 컨트롤러 (QR 버전)
# ============================================================
class SH5SpawnControllerQR:
    def __init__(self):
        print("\n" + "="*60)
        print("  SH5 QR 카메라 컨트롤러")
        print(f"  픽앤플레이스 모드: {PICK_AND_PLACE_MODE}")
        print(f"  QR 카메라 스캔: {'ON' if USE_QR_CAMERA else 'OFF'}")
        print("="*60)

        self.units = [
            SH5LineUnitQR("sg2_in_01"),
            SH5LineUnitQR("sg2_in_02"),
            SH5LineUnitQR("sg2_in_03"),
        ]

        self.db_node = None
        if ROS2_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            self.db_node = SH5DBNodeQR(self.units)
            for u in self.units:
                u.db = self.db_node
            self._spin_thread = threading.Thread(
                target=lambda: rclpy.spin(self.db_node), daemon=True)
            self._spin_thread.start()
            print("[Controller QR] ✅ ROS 2 가동")
        else:
            self._mock_thread = threading.Thread(
                target=_mock_trigger_loop, args=(self.units,), daemon=True)
            self._mock_thread.start()
            print("[Controller QR] ⚠️ Mock 모드")

        print("[Controller QR] 🚀 시연 준비 완료")

    def run_once(self):
        for unit in self.units:
            unit.step()

    def run_loop(self, max_cycles: int = 9999, interval: float = 0.1):
        for _ in range(max_cycles):
            self.run_once()
            time.sleep(interval)


# ============================================================
# 진입점
# ============================================================
print("\n[SH5 QR Controller] 초기화 중...")
controller_qr = SH5SpawnControllerQR()

if ISAAC_AVAILABLE:
    try:
        import omni.kit.app
        def _on_update(e):
            controller_qr.run_once()
        _update_sub_qr = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            _on_update, name="sh5_qr_controller_update"
        )
        print("[Controller QR] ✅ Isaac Sim Update 콜백 등록")
        print("[Controller QR] 📡 BG2 신호 대기 중... (QR 카메라 스캔 활성)")
    except Exception as e:
        print(f"[Controller QR] Update 콜백 실패: {e}")
        controller_qr.run_loop()
else:
    controller_qr.run_loop()
