"""
sh5_spawn_controller.py
=======================
[실제 시연 아키텍처 기반 SH5 컨트롤러]

아키텍처:
  [다른 PC - BG2 Isaac Sim]
    SG2 분류 로봇이 상자를 컨베이어 끝까지 밀어냄
    → /sim/transit_package 서비스 호출 → SimSyncNode
                                           ↓
                               /sim/sg2_spawn_trigger 토픽 발행
                                           ↓
  [우리 PC - AMR+SH5 Isaac Sim]
    SH5가 토픽 수신 → 해당 라인 컨베이어 끝 지정 위치에 상자 리스폰(Spawn)
    → HDF5 궤적 재생으로 픽앤플레이스 → 작업대 슬롯 배치
    → 홈 포지션 복귀 → 다음 상자 대기

실행 방법 (Isaac Sim Script Editor):
  exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_spawn_controller.py', encoding='utf-8').read())
"""

import os, sys, time, random, json, threading

# ============================================================
# Isaac Sim 연결
# ============================================================
ISAAC_AVAILABLE = False
try:
    import omni.usd
    from pxr import UsdGeom, Sdf, Gf
    ISAAC_AVAILABLE = True
    print("[SH5] ✅ Isaac Sim 연결 성공")
except ImportError:
    print("[SH5] ⚠️ Isaac Sim 외부 (디버그 모드)")

# ============================================================
# ROS 2 연결
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
    print("[SH5] ✅ ROS 2 연결 성공")
except ImportError:
    print("[SH5] ⚠️ ROS 2 없음 - Mock 모드")

# HDF5 Replay
HDF5_REPLAY_AVAILABLE = False
try:
    sys.path.insert(0, '/home/rokey/dev_ws/coupang_ws/scripts')
    from hdf5_replay_player import pick_and_place_replay
    HDF5_REPLAY_AVAILABLE = True
    print("[SH5] ✅ HDF5 재생 모듈 로드 완료")
except ImportError:
    print("[SH5] ⚠️ HDF5 모듈 없음")

# ACT 모델 (AI 추론)
TORCH_AVAILABLE = False
try:
    import torch
    sys.path.insert(0, '/home/rokey/dev_ws/coupang_ws/scripts')
    from evaluate_test_vision import run_act_inference_step
    TORCH_AVAILABLE = True
    print("[SH5] ✅ ACT 추론 모듈 로드 완료")
except ImportError:
    print("[SH5] ⚠️ ACT 모듈 없음 (torch 또는 evaluate_test_vision 없음)")

# ============================================================
# ★ 시연 모드 선택 (이 한 줄만 변경)
# ============================================================
#  "HDF5_REPLAY"    - VR 궤적 재생 (기본/보험)  ← 내일 기본값
#  "DUMMY_TELEPORT" - 상자 순간이동 (통신 테스트)
#  "ACT_MODEL"      - AI 비전 추론 (모델 완성 시)
PICK_AND_PLACE_MODE = "HDF5_REPLAY"

# ACT 모델 파일 경로 (ACT_MODEL 모드 선택 시 확인)
ACT_MODEL_PATH = "/home/rokey/dev_ws/models/augmented_sh5_vision_act_20ep.pth"

# ============================================================
# 좌표 설정 (PHYSICAL_LAYOUT.md 공식 좌표)
# ============================================================
# 각 라인별 컨베이어 끝단 스폰 위치 (X=9.0 고정, Z=컨베이어 높이)
# BG2가 상자를 디스폰하면 이 위치에 리스폰됨
CONVEYOR_SPAWN_POSITIONS = {
    "sg2_in_01": (9.0,  1.5, 0.83),   # 1번 라인 컨베이어 끝
    "sg2_in_02": (9.0, -3.0, 0.83),   # 2번 라인 컨베이어 끝
    "sg2_in_03": (9.0, -7.5, 0.83),   # 3번 라인 컨베이어 끝
}

# SH5 로봇 배치 위치 (PHYSICAL_LAYOUT.md 기준)
ROBOT_POSITIONS = {
    "sg2_in_01": (7.5,  3.0),
    "sg2_in_02": (7.5, -1.5),
    "sg2_in_03": (7.5, -6.0),
}

# 작업대 슬롯 목표 (로봇 기준 상대 좌표)
SLOT_TARGETS_LOCAL = {
    1: (0.0, -1.5, 1.2),
    2: (0.0, -1.5, 1.2),
    3: (0.0, -1.5, 0.5),
    4: (0.0, -1.5, 0.5),
}

# 상자 USD (없으면 간단한 큐브로 대체)
BOX_USD_PATH = "/home/rokey/dev_ws/assets/sh5_box.usd"
BOX_FALLBACK_SIZE = (0.12, 0.12, 0.12)   # 120mm 큐브 폴백

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
    """
    BG2가 디스폰한 상자를 우리 환경의 컨베이어 끝에 리스폰.
    returns: prim_path (스폰된 상자 경로)
    """
    stage = get_stage()
    spawn_pos = CONVEYOR_SPAWN_POSITIONS.get(line_id, (9.0, 0.0, 0.83))
    safe_id = package_id.replace('-', '_').replace(' ', '_')
    prim_path = f"/World/SH5_SpawnBox_{line_id.split('_')[-1]}_{safe_id}"

    if stage is None:
        print(f"  [Spawn] ⚠️ Stage 없음 - 상자 스폰 스킵 ({prim_path})")
        return prim_path   # Mock: 경로만 반환

    # 기존 Prim 제거
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))

    # 상자 Prim 생성
    if os.path.exists(BOX_USD_PATH):
        box_prim = stage.DefinePrim(prim_path, "Xform")
        box_prim.GetReferences().AddReference(BOX_USD_PATH)
    else:
        # 폴백: 간단한 큐브 생성
        box_prim = UsdGeom.Cube.Define(stage, prim_path)
        box_prim.GetSizeAttr().Set(BOX_FALLBACK_SIZE[0])

    # 위치 설정
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*spawn_pos))

    print(f"  [Spawn] 📦 상자 리스폰: {prim_path}")
    print(f"  [Spawn]    위치: {spawn_pos}  (라인: {line_id})")
    return prim_path


def teleport_box_to_slot(box_prim_path: str, robot_world_pos: tuple, slot_num: int) -> bool:
    stage = get_stage()
    local = SLOT_TARGETS_LOCAL.get(slot_num, SLOT_TARGETS_LOCAL[1])
    world = (
        robot_world_pos[0] + local[0],
        robot_world_pos[1] + local[1],
        local[2],
    )
    if stage is None:
        print(f"  [Teleport] 상자 → 슬롯{slot_num} {world} (Mock)")
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
# SH5 작업 단위 (라인 1대 담당)
# ============================================================
class SH5LineUnit:
    """
    한 입고 라인(sg2_in_0X)을 담당하는 SH5 작업 단위.
    - /sim/sg2_spawn_trigger 수신 시 큐에 적재
    - 큐에서 꺼내 상자 리스폰 → 픽앤플레이스 → 보고
    """
    def __init__(self, line_id: str, db_node=None):
        self.line_id = line_id                      # "sg2_in_01"
        self.robot_world_pos = ROBOT_POSITIONS[line_id] + (0.0,)
        self.db = db_node
        self.slots = {1: None, 2: None, 3: None, 4: None}
        self.current_box_prim = ""
        self.current_package_id = ""
        self._is_paused = False

        import queue
        self.pending_packages = queue.Queue()
        print(f"[{line_id}] 🤖 초기화 완료 | 로봇 위치: {ROBOT_POSITIONS[line_id]}")
        print(f"[{line_id}]    컨베이어 스폰: {CONVEYOR_SPAWN_POSITIONS[line_id]}")

    def _find_free_slot(self):
        for k, v in self.slots.items():
            if v is None:
                return k
        return None

    def step(self):
        """1 스텝 처리 - 메인 루프에서 반복 호출"""
        if self._is_paused:
            return

        # 큐에 패키지가 있는지 확인
        try:
            payload = self.pending_packages.get_nowait()
        except:
            return   # 큐 비어있음 → 대기

        package_id = payload.get('package_id', f"PKG_MOCK_{int(time.time())}")
        self.current_package_id = package_id

        print(f"\n[{self.line_id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[{self.line_id}] 📦 패키지 수신: {package_id}")

        # ① BG2가 디스폰한 상자를 우리 환경에 리스폰
        self.current_box_prim = respawn_box(self.line_id, package_id)
        time.sleep(0.5)   # 물리 엔진 안정화

        # ② DB 중복 검사
        if self.db and ROS2_AVAILABLE:
            if self._db_check_duplicate():
                print(f"[{self.line_id}] ⚠️ 중복 감지 → AMR 직송")
                self._send_move_package()
                self.current_box_prim = ""
                return

        # ③ 빈 슬롯 배정
        slot = self._find_free_slot()
        if slot is None:
            print(f"[{self.line_id}] 🏭 슬롯 만석 → 작업대 교체 요청")
            self._send_manage_workstation()
            self.slots = {1: None, 2: None, 3: None, 4: None}
            slot = 1

        self.slots[slot] = package_id
        print(f"[{self.line_id}] ✅ 슬롯 {slot} 배정")

        # ④ 픽앤플레이스
        self._do_pick_and_place(slot)

        # ⑤ 적재 보고
        filled = sum(1 for v in self.slots.values() if v is not None)
        self._db_report(filled, slot)
        self.current_box_prim = ""

    def _do_pick_and_place(self, slot: int):
        """
        픽앤플레이스 실행 — PICK_AND_PLACE_MODE에 따라 3가지 방식으로 동작.

        ACT_MODEL    : AI 비전 추론 → 실패 시 HDF5 → 최종 폴백 Teleport
        HDF5_REPLAY  : VR 수집 궤적 재생 → 실패 시 Teleport 폴백
        DUMMY_TELEPORT: 상자 순간이동 (통신 검증용)
        """
        mode = PICK_AND_PLACE_MODE

        # ── Mode 1: ACT_MODEL ────────────────────────────────
        if mode == "ACT_MODEL":
            if TORCH_AVAILABLE:
                print(f"[{self.line_id}] 🤖 ACT 모델 추론 시작 (슬롯 {slot})")
                try:
                    ok = run_act_inference_step(
                        model_path      = ACT_MODEL_PATH,
                        slot_num        = slot,
                        robot_articulation = getattr(self, 'robot', None),
                        box_prim_path   = self.current_box_prim,
                        robot_world_pos = self.robot_world_pos,
                    )
                    if ok:
                        print(f"[{self.line_id}] ✅ ACT 추론 픽앤플레이스 완료")
                        return
                    else:
                        print(f"[{self.line_id}] ❌ ACT 추론 실패 → HDF5 폴백")
                except Exception as e:
                    print(f"[{self.line_id}] ❌ ACT 오류: {e} → HDF5 폴백")
            else:
                print(f"[{self.line_id}] ⚠️ ACT 모듈 없음 → HDF5 폴백")

            # ACT 실패 시 HDF5로 폴백
            if HDF5_REPLAY_AVAILABLE:
                ok = pick_and_place_replay(
                    slot_num=slot,
                    robot_articulation=getattr(self, 'robot', None),
                    box_prim_path=self.current_box_prim,
                    realtime=True,
                )
                if ok:
                    return
            # 최종 폴백
            print(f"[{self.line_id}] ⚠️ 최종 폴백: Dummy Teleport")
            teleport_box_to_slot(self.current_box_prim, self.robot_world_pos, slot)

        # ── Mode 2: HDF5_REPLAY ──────────────────────────────
        elif mode == "HDF5_REPLAY" and HDF5_REPLAY_AVAILABLE:
            print(f"[{self.line_id}] 🎬 HDF5 재생 (슬롯 {slot})")
            ok = pick_and_place_replay(
                slot_num=slot,
                robot_articulation=getattr(self, 'robot', None),
                box_prim_path=self.current_box_prim,
                realtime=True,
            )
            if not ok:
                print(f"[{self.line_id}] ❌ HDF5 실패 → Dummy Teleport 폴백")
                teleport_box_to_slot(self.current_box_prim, self.robot_world_pos, slot)

        # ── Mode 3: DUMMY_TELEPORT (기본 폴백) ───────────────
        else:
            print(f"[{self.line_id}] 🚀 Dummy Teleport (슬롯 {slot})")
            teleport_box_to_slot(self.current_box_prim, self.robot_world_pos, slot)

    # ── DB 통신 ──────────────────────────────────────
    def _db_check_duplicate(self) -> bool:
        try:
            if not self.db.check_client.wait_for_service(timeout_sec=0.5):
                return False
            req = CheckWarehouseStatus.Request()
            req.package_id = self.current_package_id
            req.customer_name = ""
            req.qr_id = self.current_package_id
            fut = self.db.check_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, fut, timeout_sec=1.0)
            return fut.result() and fut.result().is_already_in_warehouse
        except Exception as e:
            print(f"[{self.line_id}] [DB] 중복검사 오류: {e}")
            return False

    def _db_report(self, filled: int, slot: int):
        if not self.db or not ROS2_AVAILABLE:
            return
        try:
            if not self.db.report_client.wait_for_service(timeout_sec=0.5):
                return
            req = ReportInboundProgress.Request()
            req.workstation_id = f"WS_{self.line_id[-2:]}"
            req.robot_id = self.line_id
            req.filled_slots_count = filled
            req.package_id = self.current_package_id
            req.workstation_qr_id = f"WORKSTATION_WS_{self.line_id[-2:]}"
            req.package_qr_id = self.current_package_id
            fut = self.db.report_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, fut, timeout_sec=1.0)
            if fut.result() and fut.result().success:
                print(f"[{self.line_id}] [DB] ✅ 슬롯{slot} 보고 완료 (총{filled}/4)")
        except Exception as e:
            print(f"[{self.line_id}] [DB] 보고 오류: {e}")

    def _send_move_package(self):
        if not self.db or not ROS2_AVAILABLE:
            return
        self.db.send_move_package(self.current_package_id, self.line_id)

    def _send_manage_workstation(self):
        if not self.db or not ROS2_AVAILABLE:
            return
        self.db.send_manage_workstation(
            f"WS_{self.line_id[-2:]}",
            int(self.line_id[-2:])
        )


# ============================================================
# ROS 2 DB 노드
# ============================================================
class SH5DBNode(Node):
    def __init__(self, units: list):
        super().__init__('sh5_spawn_db_node')
        self.units = units
        self.line_map = {u.line_id: u for u in units}

        # 서비스 클라이언트
        self.check_client = self.create_client(
            CheckWarehouseStatus, '/check_warehouse_status')
        self.report_client = self.create_client(
            ReportInboundProgress, '/report_inbound_progress')

        # 액션 클라이언트
        self.move_pkg_client = ActionClient(self, MovePackage, '/move_package')
        self.manage_ws_client = ActionClient(self, ManageWorkstation, '/manage_workstation')

        # Pause 구독
        for line_id in CONVEYOR_SPAWN_POSITIONS:
            self.create_subscription(
                Bool, f'/{line_id}/pause_status',
                lambda msg, lid=line_id: self._pause_cb(msg, lid), 10
            )

        # ★ 핵심: /sim/sg2_spawn_trigger 구독
        # BG2(다른 PC)가 상자를 디스폰하면 SimSyncNode가 이 토픽 발행
        self.create_subscription(
            String, '/sim/sg2_spawn_trigger',
            self._spawn_trigger_cb, 10
        )

        print("[DB Node] ✅ 초기화 완료")
        print("[DB Node] 📡 /sim/sg2_spawn_trigger 구독 - BG2 상자 소환 대기 중")
        for lid in CONVEYOR_SPAWN_POSITIONS:
            print(f"[DB Node]    /{lid}/pause_status 구독")

    def _pause_cb(self, msg, line_id: str):
        unit = self.line_map.get(line_id)
        if unit:
            unit._is_paused = msg.data
            state = "⏸️ Pause" if msg.data else "▶️ Resume"
            print(f"[DB Node] {state}: {line_id}")

    def _spawn_trigger_cb(self, msg):
        """
        BG2 PC에서 상자를 디스폰 → SimSyncNode가 이 토픽 발행
        → 해당 라인 Unit의 큐에 넣어 리스폰 처리
        """
        try:
            payload = json.loads(msg.data)
            package_id = payload.get('package_id', '')
            target_line = payload.get('target_line', '')   # "sg2_in_01"
            print(f"\n[DB Node] 🚨 BG2 상자 디스폰 감지!")
            print(f"[DB Node]    패키지: {package_id}")
            print(f"[DB Node]    목표 라인: {target_line}")
            print(f"[DB Node]    → 우리 환경({target_line} 컨베이어 끝)에 리스폰 예약")

            unit = self.line_map.get(target_line)
            if unit is None:
                print(f"[DB Node] ⚠️ 알 수 없는 라인: {target_line}")
                return
            unit.pending_packages.put(payload)
        except Exception as e:
            print(f"[DB Node] spawn_trigger 파싱 오류: {e}")

    def send_move_package(self, package_id: str, line_id: str):
        if not self.move_pkg_client.wait_for_server(timeout_sec=2.0):
            print("[Action] MovePackage 서버 없음 - 스킵")
            return
        goal = MovePackage.Goal()
        goal.package_id = package_id
        goal.customer_name = ""
        goal.destination_zone = "MAIN_storage"
        goal.package_qr_id = package_id

        def _fb(fb):
            print(f"  [AMR] 위치: {fb.feedback.current_position} | 진행: {fb.feedback.progress:.1f}%")

        print(f"[Action] 🚛 MovePackage: {package_id} → MAIN_storage")
        self.move_pkg_client.send_goal_async(goal, feedback_callback=_fb)

    def send_manage_workstation(self, ws_id: str, unit_idx: int):
        if not self.manage_ws_client.wait_for_server(timeout_sec=2.0):
            print("[Action] ManageWorkstation 서버 없음 - 스킵")
            return
        goal = ManageWorkstation.Goal()
        goal.workstation_id = ws_id
        goal.start_location = f"sg2_in_{unit_idx:02d}_A"
        goal.target_location = "warehouse"
        goal.workstation_qr_id = f"WORKSTATION_WS_{unit_idx:02d}"
        goal.target_qr_id = ""
        goal.target_x = 1.5
        goal.target_y = 3.0
        goal.target_yaw = 0.0

        def _fb(fb):
            print(f"  [AMR] 남은거리: {fb.feedback.distance_remaining:.2f}m | {fb.feedback.status}")

        print(f"[Action] 🏭 ManageWorkstation: {ws_id} → warehouse")
        self.manage_ws_client.send_goal_async(goal, feedback_callback=_fb)


# ============================================================
# Mock 큐 자동 투입 (ROS2 없을 때 테스트용)
# ============================================================
def _mock_trigger_loop(units: list):
    """DB팀 연결 없을 때 5초마다 가상 상자를 순환 투입"""
    line_ids = list(CONVEYOR_SPAWN_POSITIONS.keys())
    idx = 0
    while True:
        time.sleep(5.0)
        line_id = line_ids[idx % len(line_ids)]
        pkg_id = f"PKG_MOCK_{int(time.time())}"
        for u in units:
            if u.line_id == line_id:
                u.pending_packages.put({
                    'package_id': pkg_id,
                    'target_line': line_id,
                })
                print(f"\n[Mock] 🎲 가상 상자 투입: {pkg_id} → {line_id}")
                break
        idx += 1


# ============================================================
# 메인 컨트롤러
# ============================================================
class SH5SpawnController:
    def __init__(self):
        print("\n" + "="*60)
        print("  SH5 물류 컨트롤러 (BG2 리스폰 방식)")
        print(f"  모드: {PICK_AND_PLACE_MODE}")
        print("="*60)

        # 라인 Unit 생성
        self.units = [
            SH5LineUnit("sg2_in_01"),
            SH5LineUnit("sg2_in_02"),
            SH5LineUnit("sg2_in_03"),
        ]

        # DB 노드
        self.db_node = None
        if ROS2_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            self.db_node = SH5DBNode(self.units)
            for u in self.units:
                u.db = self.db_node

            # ROS2 spin 스레드
            self._spin_thread = threading.Thread(
                target=lambda: rclpy.spin(self.db_node), daemon=True)
            self._spin_thread.start()
            print("[Controller] ✅ ROS 2 DB 노드 가동")
        else:
            print("[Controller] ⚠️ ROS 2 없음 - Mock 모드")
            self._mock_thread = threading.Thread(
                target=_mock_trigger_loop, args=(self.units,), daemon=True)
            self._mock_thread.start()

        print(f"[Controller] ✅ SH5 {len(self.units)}대 준비 완료")
        print(f"[Controller] 📡 BG2 상자 디스폰 신호 대기 중...")
        print(f"[Controller]    → 수신 시 컨베이어 끝에 자동 리스폰")

    def run_once(self):
        """Isaac Sim update 루프에서 1프레임마다 호출"""
        for unit in self.units:
            unit.step()

    def run_loop(self, max_cycles: int = 9999, step_interval: float = 0.1):
        """단독 실행 루프 (Script Editor 직접 실행 시)"""
        print(f"\n[Controller] 🔄 메인 루프 시작 (간격: {step_interval}s)")
        for _ in range(max_cycles):
            self.run_once()
            time.sleep(step_interval)


# ============================================================
# 진입점 (Script Editor exec() 방식)
# ============================================================
print("\n[SH5 Spawn Controller] 초기화 중...")
controller = SH5SpawnController()

# Isaac Sim update callback 등록
if ISAAC_AVAILABLE:
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()

        def _on_update(e):
            controller.run_once()

        _update_sub = app.get_update_event_stream().create_subscription_to_pop(
            _on_update, name="sh5_spawn_controller_update"
        )
        print("[Controller] ✅ Isaac Sim Update 콜백 등록 완료")
        print("[Controller] 🚀 시연 준비 완료! BG2 신호를 기다립니다...")
    except Exception as e:
        print(f"[Controller] Update 콜백 실패: {e}")
        print("[Controller] → run_loop() 수동 실행 필요")
        controller.run_loop()
else:
    # Isaac Sim 외부: 단독 루프 실행
    controller.run_loop()
