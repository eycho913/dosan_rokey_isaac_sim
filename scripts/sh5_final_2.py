"""
sh5_final_2.py  —  SH5 최종 통합 컨트롤러 v2
================================================

확정 시나리오 (AMR/SH5 환경 분리):
  1. /sim/sg2_spawn_trigger 수신 (BG2 → SH5 상자 디스폰 신호)
  2. 해당 슬롯의 HDF5 에피소드에서 상자 스폰 위치 추출 → 스폰
  3. check_warehouse_status 중복 검사
     - 중복(is_already_in_warehouse=True) → 스폰 상자 제거, 스킵
     - 중복 없음 → HDF5 Replay pick&place → report_inbound_progress 보고

변경사항 (vs sh5_final.py):
  - 상자 스폰: CONVEYOR_SPAWN 고정값 → HDF5 box_initial_pose + robot offset
  - 중복 검사: check_warehouse_status 서비스 추가
  - QR 좌표 인식 제거 (package_id는 토픽에서 직접 받음)
  - customer_name: packages CSV에서 조회 (없으면 "UNKNOWN")

실행:
  exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_final_2.py', encoding='utf-8').read())
"""

import os, sys, time, json, csv, threading
from pathlib import Path

# ============================================================
# Isaac Sim
# ============================================================
ISAAC_AVAILABLE = False
try:
    import omni.usd
    from pxr import UsdGeom, Sdf, Gf
    ISAAC_AVAILABLE = True
    print("[SH5v2] ✅ Isaac Sim")
except ImportError:
    print("[SH5v2] ⚠️ 외부 실행 (Mock 모드)")

# ============================================================
# ROS 2
# ============================================================
ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node as _Node
    from std_msgs.msg import String as _String
    from cobot3_interfaces.srv import CheckWarehouseStatus, ReportInboundProgress
    ROS2_AVAILABLE = True
    print("[SH5v2] ✅ ROS 2")
except ImportError:
    _Node   = object   # Isaac Sim 환경에서 ROS2 없을 때 더미
    _String = None
    CheckWarehouseStatus  = None
    ReportInboundProgress = None
    print("[SH5v2] ⚠️ ROS 2 없음 → Mock 모드")

# ============================================================
# HDF5 재생
# ============================================================
HDF5_AVAILABLE = False
try:
    sys.path.insert(0, '/home/rokey/dev_ws/coupang_ws/scripts')
    from hdf5_replay_player import HDF5EpisodeLoader, TrajectoryReplayPlayer
    HDF5_AVAILABLE = True
    print("[SH5v2] ✅ HDF5 모듈")
except ImportError:
    print("[SH5v2] ⚠️ HDF5 없음 → Teleport 폴백")

# ============================================================
# ★ 설정  —  이 블록만 수정
# ============================================================
PICK_AND_PLACE_MODE = "HDF5_REPLAY"
# "HDF5_REPLAY"    ← D-Day 기본 (VR 궤적 재생 + robot offset)
# "DUMMY_TELEPORT" ← 통신 테스트용 (순간이동)

# ── 로봇 월드 좌표 (final_coupan.usd 실측값으로 수정) ──────
ROBOT_POS = {
    "sg2_in_01": (7.5,  3.0, 0.0),
    "sg2_in_02": (7.5, -1.5, 0.0),
    "sg2_in_03": (7.5, -6.0, 0.0),
}

# ── 작업대 슬롯 위치 (로봇 기준 상대 좌표, 4칸) ────────────
SLOT_LOCAL = {
    1: ( 0.0, -1.5, 1.2),
    2: ( 0.0, -1.5, 0.9),
    3: ( 0.0, -1.5, 0.6),
    4: ( 0.0, -1.5, 0.3),
}

# ── 작업대 ID 매핑 (DB init.sql 기준) ──────────────────────
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

# ── 패키지 CSV 경로 (customer_name 조회용) ─────────────────
QR_DATA_DIR  = Path("/home/rokey/dev_ws/qr_data")
BOX_USD      = "/home/rokey/dev_ws/assets/sh5_box.usd"

# ── 순차 실행 모드 (FPS 보호) ──────────────────────────────
SEQUENTIAL_MODE = True

# ============================================================
# 패키지 CSV 로더 (customer_name 조회)
# ============================================================
_pkg_cache: dict[str, dict] = {}   # package_id → {customer_name, qr_id, ...}

def _load_pkg_csv():
    """QR_DATA_DIR의 모든 CSV를 메모리에 로드."""
    if _pkg_cache:
        return
    for csv_path in QR_DATA_DIR.glob("packages_*.csv"):
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    pid = row.get("package_id", "").strip()
                    if pid:
                        _pkg_cache[pid] = {
                            "customer_name": row.get("customer_name", "UNKNOWN").strip(),
                            "qr_id":         row.get("qr_id", pid.replace("PKG_","QR_")).strip(),
                            "route_zone":    row.get("route_zone", "").strip(),
                        }
        except Exception as e:
            print(f"[CSV] 로드 오류 {csv_path}: {e}")
    print(f"[CSV] ✅ {len(_pkg_cache)}개 패키지 로드 완료")

def get_pkg_info(package_id: str) -> dict:
    """package_id → {customer_name, qr_id} 반환."""
    _load_pkg_csv()
    if package_id in _pkg_cache:
        return _pkg_cache[package_id]
    # CSV에 없으면 ID 규칙으로 역추정
    qr_id = package_id.replace("PKG_", "QR_") if package_id.startswith("PKG_") else package_id
    return {"customer_name": "UNKNOWN", "qr_id": qr_id, "route_zone": ""}

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


def _remove_prim(path: str):
    """Stage에서 Prim 삭제 (중복 처리 시 상자 제거)."""
    stage = _stage()
    if stage is None:
        return
    try:
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(Sdf.Path(path))
    except:
        pass


def spawn_box_from_hdf5(line_id: str, pkg_id: str, slot: int,
                        robot_art=None) -> tuple[str, object]:
    """
    HDF5 에피소드에서 상자 초기 위치를 읽어 + robot offset 적용 후 스폰.

    Returns:
        (box_prim_path, episode) — episode는 이후 pick&place에 재사용
    """
    safe   = pkg_id.replace("-","_").replace(" ","_")
    path   = f"/World/SH5v2Box_{line_id[-2:]}_{safe}"
    stage  = _stage()
    robot_world = ROBOT_POS[line_id]

    episode = None

    if HDF5_AVAILABLE:
        try:
            loader  = HDF5EpisodeLoader(slot_num=slot)
            episode = loader.load_random_episode()

            # ── 상자 스폰 위치: HDF5 box_pose + robot offset ──
            import numpy as np
            rec_robot = np.array(episode['robot_initial_pose'][:3])
            cur_robot = np.array(robot_world)
            offset    = cur_robot - rec_robot
            hdf5_box  = np.array(episode['box_initial_pose'][:3])
            spawn_pos = hdf5_box + offset          # 보정된 스폰 위치

            print(f"  [Spawn] HDF5 에피소드: {episode['demo_key']}")
            print(f"  [Spawn] robot offset: ({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})")
            print(f"  [Spawn] 상자 위치: ({spawn_pos[0]:.3f}, {spawn_pos[1]:.3f}, {spawn_pos[2]:.3f})")
        except Exception as e:
            print(f"  [Spawn] HDF5 오류({e}) → ROBOT_POS 기준 기본 위치 사용")
            spawn_pos = None
    else:
        spawn_pos = None

    # HDF5 없으면 로봇 앞 고정 offset 사용
    if spawn_pos is None:
        import numpy as np
        spawn_pos = np.array(robot_world) + np.array([1.5, -1.5, 0.83])

    if stage is None:
        print(f"  [Spawn] Mock: {path} @ {tuple(spawn_pos)}")
        return path, episode

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
    xf.AddTranslateOp().Set(Gf.Vec3d(float(spawn_pos[0]),
                                     float(spawn_pos[1]),
                                     float(spawn_pos[2])))
    print(f"  [Spawn] 📦 {path}")
    return path, episode


def teleport_to_slot(box_path: str, robot_pos: tuple, slot: int) -> bool:
    """상자를 슬롯 좌표로 즉시 이동 (Dummy Teleport / 폴백)."""
    local = SLOT_LOCAL[slot]
    world = (robot_pos[0] + local[0],
             robot_pos[1] + local[1],
             robot_pos[2] + local[2])
    stage = _stage()
    if stage is None:
        print(f"  [Teleport] Mock: {box_path} → 슬롯{slot} {world}")
        return True
    try:
        prim = stage.GetPrimAtPath(box_path)
        if not prim.IsValid():
            return False
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*world))
        print(f"  [Teleport] ✅ 슬롯{slot} @ {world}")
        return True
    except Exception as e:
        print(f"  [Teleport] 오류: {e}")
        return False


# ============================================================
# Pick & Place
# ============================================================
def do_pick_and_place(box_path: str, robot_pos: tuple, slot: int,
                      robot_art, episode: object) -> bool:
    """
    HDF5 Replay (episode 재사용) 또는 Dummy Teleport.
    robot offset은 TrajectoryReplayPlayer에서 처리.
    """
    mode = PICK_AND_PLACE_MODE

    if mode == "HDF5_REPLAY" and HDF5_AVAILABLE and episode is not None:
        print(f"  [P&P] 🎬 HDF5 재생 슬롯{slot}")
        try:
            player = TrajectoryReplayPlayer(
                robot_articulation=robot_art,
                box_prim_path=box_path,
                robot_world_pos=robot_pos,   # ★ offset 전달
            )
            ok = player.play_episode(episode, realtime=True)
            if ok:
                return True
            print(f"  [P&P] HDF5 실패 → Teleport 폴백")
        except Exception as e:
            print(f"  [P&P] HDF5 오류: {e} → Teleport 폴백")

    print(f"  [P&P] 🚀 Dummy Teleport 슬롯{slot}")
    return teleport_to_slot(box_path, robot_pos, slot)


# ============================================================
# ROS 2 노드
# ============================================================
class SH5NodeV2(_Node):
    """CheckWarehouseStatus + ReportInboundProgress 클라이언트."""

    def __init__(self, lines: list):
        if not ROS2_AVAILABLE:
            return
        super().__init__("sh5_controller_v2")

        # ① check_warehouse_status 클라이언트
        self.check_client = self.create_client(
            CheckWarehouseStatus, "check_warehouse_status"
        )

        # ② report_inbound_progress 클라이언트
        self.report_client = self.create_client(
            ReportInboundProgress, "report_inbound_progress"
        )

        # ③ /sim/sg2_spawn_trigger 구독
        self.create_subscription(
            _String, "/sim/sg2_spawn_trigger",
            self._on_spawn_trigger, 10
        )
        self.lines = {l.line_id: l for l in lines}
        self.get_logger().info("[SH5v2] 노드 초기화 완료")

    def _on_spawn_trigger(self, msg):   # 타입 어노테이션 제거 (오류 방지)
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        line_id = payload.get("target_line", "")
        if line_id in self.lines:
            self.lines[line_id].queue.put(payload)
            self.get_logger().info(
                f"[SH5v2] 📨 {line_id} ← {payload.get('package_id','?')}"
            )


# ============================================================
# SH5 라인 작업 단위
# ============================================================
class SH5LineV2:
    """
    라인 1개를 담당.
    수신 → HDF5 스폰 → 중복검사 → pick&place → 보고
    """

    def __init__(self, line_id: str, db_node=None):
        self.line_id   = line_id
        self.robot_pos = ROBOT_POS[line_id]
        self.ws_id     = WORKSTATION_ID[line_id]
        self.ws_qr     = WORKSTATION_QR[line_id]
        self.db        = db_node
        self.robot_art = None
        self.filled    = 0        # 현재 슬롯 수 (0~4)
        self._paused   = False
        self._busy     = False

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
        pkg_id = payload.get("package_id", f"PKG_MOCK_{int(time.time())}")
        print(f"\n[{self.line_id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[{self.line_id}] 📩 수신: {pkg_id}")

        try:
            # ── 패키지 메타 조회 (customer_name, qr_id) ──────
            meta         = get_pkg_info(pkg_id)
            customer     = meta["customer_name"]
            qr_id        = meta["qr_id"]
            print(f"[{self.line_id}] 🏷️  pkg={pkg_id} | qr={qr_id} | 수령인={customer}")

            # ── STEP 1: 슬롯 결정 & HDF5에서 상자 스폰 ──────
            slot = self.next_slot()
            box_path, episode = spawn_box_from_hdf5(
                self.line_id, pkg_id, slot, self.robot_art
            )
            time.sleep(0.4)

            # ── STEP 2: 중복 검사 ─────────────────────────────
            if self._check_duplicate(customer, pkg_id, qr_id):
                print(f"[{self.line_id}] ⚠️ 중복 감지 → 상자 제거 & 스킵")
                _remove_prim(box_path)
                return

            # ── STEP 3: Pick & Place ──────────────────────────
            ok = do_pick_and_place(
                box_path, self.robot_pos, slot, self.robot_art, episode
            )
            if not ok:
                print(f"[{self.line_id}] ❌ P&P 실패 → 스킵")
                _remove_prim(box_path)
                return

            # ── STEP 4: DB 적재 보고 ──────────────────────────
            self.filled += 1
            self._report(pkg_id, qr_id, slot)

            if self.filled >= 4:
                print(f"[{self.line_id}] 🔄 4칸 완충 → 슬롯 리셋 (DB가 회전 처리)")
                self.filled = 0

        finally:
            self._busy = False

    # ── 중복 검사 ──────────────────────────────────────────────
    def _check_duplicate(self, customer: str, pkg_id: str, qr_id: str) -> bool:
        """
        check_warehouse_status 서비스 호출.
        Returns True if 중복(창고에 이미 있음).
        """
        if not self.db or not ROS2_AVAILABLE:
            print(f"  [중복검사] Mock — 중복 없음으로 처리")
            return False
        try:
            if not self.db.check_client.wait_for_service(timeout_sec=1.0):
                print(f"  [중복검사] 서비스 없음 → 중복 없음으로 처리")
                return False

            req = CheckWarehouseStatus.Request()
            req.customer_name = customer
            req.package_id    = pkg_id
            req.qr_id         = qr_id

            fut = self.db.check_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, fut, timeout_sec=2.0)

            if fut.result():
                result = fut.result().is_already_in_warehouse
                status = "중복" if result else "신규"
                print(f"  [중복검사] ✅ 응답: {status}")
                return result
            else:
                print(f"  [중복검사] 응답 없음 → 신규 처리")
                return False
        except Exception as e:
            print(f"  [중복검사] 오류: {e} → 신규 처리")
            return False

    # ── DB 적재 보고 ───────────────────────────────────────────
    def _report(self, pkg_id: str, qr_id: str, slot: int):
        """ReportInboundProgress 서비스 호출."""
        if not self.db or not ROS2_AVAILABLE:
            print(f"  [보고] Mock — pkg={pkg_id} slot={slot}")
            return
        try:
            if not self.db.report_client.wait_for_service(timeout_sec=1.0):
                print(f"  [보고] 서비스 없음 → 스킵")
                return

            req = ReportInboundProgress.Request()
            req.workstation_id     = self.ws_id
            req.robot_id           = self.line_id
            req.filled_slots_count = slot
            req.package_id         = pkg_id
            req.workstation_qr_id  = self.ws_qr
            req.package_qr_id      = qr_id

            fut = self.db.report_client.call_async(req)
            rclpy.spin_until_future_complete(self.db, fut, timeout_sec=2.0)

            if fut.result() and fut.result().success:
                print(f"  [보고] ✅ DB 갱신 | {self.ws_id} 슬롯{slot} ← {pkg_id}")
            else:
                print(f"  [보고] ⚠️ DB 응답 실패")
        except Exception as e:
            print(f"  [보고] 오류: {e}")


# ============================================================
# Mock 자동 투입 (ROS2 없을 때)
# ============================================================
def _mock_loop(lines: list):
    from itertools import cycle
    pool = cycle(lines)
    while True:
        time.sleep(5.0)
        unit = next(pool)
        import random
        today = "20260612"
        num   = random.randint(1, 20)
        pkg   = f"PKG_{today}_{num:03d}"
        unit.queue.put({"package_id": pkg, "target_line": unit.line_id})
        # print(f"\n[Mock] 🎲 {pkg} → {unit.line_id}")


# ============================================================
# 메인 컨트롤러
# ============================================================
class SH5ControllerV2:
    def __init__(self):
        print("\n" + "="*60)
        print("  SH5 최종 통합 컨트롤러 v2  (sh5_final_2.py)")
        print(f"  모드: {PICK_AND_PLACE_MODE}")
        print(f"  순차 실행: {'ON' if SEQUENTIAL_MODE else 'OFF'}")
        print("="*60)

        self.lines = [
            SH5LineV2("sg2_in_01"),
            SH5LineV2("sg2_in_02"),
            SH5LineV2("sg2_in_03"),
        ]

        self.ros_node = None
        if ROS2_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            self.ros_node = SH5NodeV2(self.lines)
            for l in self.lines:
                l.db = self.ros_node
            t = threading.Thread(
                target=lambda: rclpy.spin(self.ros_node), daemon=True)
            t.start()
            print("[Controller] ✅ ROS 2 스핀 가동")
        else:
            print("[Controller] ⚠️ ROS 2 없음 → Mock 자동 투입")
            threading.Thread(
                target=_mock_loop, args=(self.lines,), daemon=True).start()

        print("[Controller] 🚀 시연 준비 완료!")
        print("[Controller]    /sim/sg2_spawn_trigger 수신 대기 중...")

    def tick(self):
        if SEQUENTIAL_MODE:
            if any(l._busy for l in self.lines):
                return
            for l in self.lines:
                if not l.queue.empty() and not l._paused:
                    l.step()
                    break
        else:
            for l in self.lines:
                l.step()

    def loop(self, interval=0.1, cycles=99999):
        for _ in range(cycles):
            self.tick()
            time.sleep(interval)


# ============================================================
# 진입점
# ============================================================
print("\n[SH5v2] 초기화 중...")
controller = SH5ControllerV2()

if ISAAC_AVAILABLE:
    try:
        import omni.kit.app
        _sub = omni.kit.app.get_app().get_update_event_stream() \
                   .create_subscription_to_pop(
                       lambda e: controller.tick(),
                       name="sh5_final_v2_update"
                   )
        print("[SH5v2] ✅ Isaac Sim 콜백 등록 완료")
        print()
        print("  수동 투입:   controller.lines[0].queue.put({'package_id':'PKG_20260612_001','target_line':'sg2_in_01'})")
        print("  상태 확인:   [(l.line_id, l.filled, l._busy) for l in controller.lines]")
    except Exception as e:
        print(f"[SH5v2] 콜백 실패: {e} → 루프 모드")
        controller.loop()
else:
    controller.loop()
