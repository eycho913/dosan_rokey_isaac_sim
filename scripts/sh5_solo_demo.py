"""
sh5_solo_demo.py
================
DB / ROS2 / 다른 PC 없이 혼자서 Isaac Sim 에서 SH5 데모를 테스트하는 스크립트.

기능:
  - 키보드 입력으로 원하는 라인에 상자를 수동 스폰
  - HDF5 궤적 재생으로 픽앤플레이스 동작 검증
  - 4슬롯 채우면 자동 리셋 (AMR 없이 로컬 처리)

실행 (Isaac Sim Script Editor):
  exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_solo_demo.py', encoding='utf-8').read())

키 바인딩:
  1 → 1번 라인(sg2_in_01)에 상자 수동 투입
  2 → 2번 라인(sg2_in_02)에 상자 수동 투입
  3 → 3번 라인(sg2_in_03)에 상자 수동 투입
  A → 3개 라인 순서대로 자동 연속 투입 (5초 간격)
  S → 자동 투입 중지
  R → 모든 슬롯 리셋
  Q → 종료
"""

import os, sys, time, threading, random
from pathlib import Path

# ============================================================
# Isaac Sim 연결
# ============================================================
ISAAC_AVAILABLE = False
try:
    import omni.usd
    import omni.kit.app
    from pxr import UsdGeom, Sdf, Gf
    ISAAC_AVAILABLE = True
    print("[Solo Demo] ✅ Isaac Sim 연결 성공")
except ImportError:
    print("[Solo Demo] ⚠️ Isaac Sim 외부 실행 (터미널 테스트 모드)")

# HDF5 재생
HDF5_AVAILABLE = False
try:
    sys.path.insert(0, '/home/rokey/dev_ws/coupang_ws/scripts')
    from hdf5_replay_player import pick_and_place_replay
    HDF5_AVAILABLE = True
    print("[Solo Demo] ✅ HDF5 재생 모듈 로드")
except ImportError:
    print("[Solo Demo] ⚠️ HDF5 모듈 없음 → Dummy Teleport만 사용")

# ============================================================
# ★ 테스트 모드 설정
# ============================================================
DEMO_MODE = "HDF5_REPLAY"     # "HDF5_REPLAY" | "DUMMY_TELEPORT"
AUTO_INTERVAL_SEC = 5.0       # 자동 투입 간격 (초)

# ============================================================
# 좌표 (PHYSICAL_LAYOUT.md 기준)
# ============================================================
LINES = {
    "sg2_in_01": {
        "spawn_pos": (9.0,  1.5, 0.83),    # 컨베이어 끝 스폰 위치
        "robot_pos": (7.5,  3.0, 0.0),     # SH5 로봇 위치
    },
    "sg2_in_02": {
        "spawn_pos": (9.0, -3.0, 0.83),
        "robot_pos": (7.5, -1.5, 0.0),
    },
    "sg2_in_03": {
        "spawn_pos": (9.0, -7.5, 0.83),
        "robot_pos": (7.5, -6.0, 0.0),
    },
}

SLOT_TARGETS_LOCAL = {
    1: (0.0, -1.5, 1.2),
    2: (0.0, -1.5, 1.2),
    3: (0.0, -1.5, 0.5),
    4: (0.0, -1.5, 0.5),
}

BOX_USD = "/home/rokey/dev_ws/assets/sh5_box.usd"  # 없으면 큐브로 대체

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


def spawn_box(line_id: str, pkg_id: str) -> str:
    """컨베이어 끝 지정 위치에 상자 생성"""
    cfg = LINES[line_id]
    pos = cfg["spawn_pos"]
    safe = pkg_id.replace("-", "_")
    path = f"/World/SoloBox_{line_id.split('_')[-1]}_{safe}"

    stage = get_stage()
    if stage is None:
        print(f"  [Spawn] Mock 스폰: {path} @ {pos}")
        return path

    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(Sdf.Path(path))

    if os.path.exists(BOX_USD):
        p = stage.DefinePrim(path, "Xform")
        p.GetReferences().AddReference(BOX_USD)
    else:
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(0.12)

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    print(f"  [Spawn] 📦 상자 스폰: {path}")
    print(f"  [Spawn]    위치: {pos}")
    return path


def teleport_to_slot(box_path: str, robot_pos: tuple, slot: int) -> bool:
    local = SLOT_TARGETS_LOCAL[slot]
    world = (robot_pos[0] + local[0], robot_pos[1] + local[1], local[2])

    stage = get_stage()
    if stage is None:
        print(f"  [Teleport] Mock: 슬롯{slot} {world}")
        return True

    prim = stage.GetPrimAtPath(box_path)
    if not prim.IsValid():
        return False
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*world))
    print(f"  [Teleport] 🚀 슬롯{slot} → {world}")
    return True


def remove_box(box_path: str):
    stage = get_stage()
    if stage and stage.GetPrimAtPath(box_path).IsValid():
        stage.RemovePrim(Sdf.Path(box_path))

# ============================================================
# 라인 상태 관리
# ============================================================
class LineState:
    def __init__(self, line_id: str):
        self.line_id = line_id
        self.slots = {1: None, 2: None, 3: None, 4: None}
        self.busy = False

    def free_slot(self):
        for k, v in self.slots.items():
            if v is None:
                return k
        return None

    def fill_slot(self, slot: int, pkg_id: str):
        self.slots[slot] = pkg_id

    def reset(self):
        self.slots = {1: None, 2: None, 3: None, 4: None}
        print(f"  [{self.line_id}] 🔄 슬롯 리셋")

    def status(self) -> str:
        filled = sum(1 for v in self.slots.values() if v)
        return f"{self.line_id}: [{filled}/4] {'■'*filled}{'□'*(4-filled)}"

# ============================================================
# 메인 데모 로직
# ============================================================
class SH5SoloDemo:
    def __init__(self):
        self.lines = {lid: LineState(lid) for lid in LINES}
        self._auto_running = False
        self._auto_thread = None
        self._pkg_counter = 1
        self._lock = threading.Lock()
        print("\n" + "="*55)
        print("  SH5 Solo Demo (DB 없이 단독 테스트)")
        print(f"  모드: {DEMO_MODE}")
        print("="*55)
        self._print_help()

    def _print_help(self):
        print("""
키 바인딩:
  1 → sg2_in_01 상자 투입
  2 → sg2_in_02 상자 투입
  3 → sg2_in_03 상자 투입
  A → 자동 순환 투입 시작
  S → 자동 투입 정지
  R → 전체 슬롯 리셋
  Q → 종료
상태 조회는 콘솔 출력 확인
""")

    def _next_pkg_id(self) -> str:
        pid = f"PKG_SOLO_{self._pkg_counter:04d}"
        self._pkg_counter += 1
        return pid

    def trigger_line(self, line_id: str):
        """지정 라인에 상자 투입 및 픽앤플레이스 실행"""
        with self._lock:
            state = self.lines[line_id]
            if state.busy:
                print(f"\n[{line_id}] ⏳ 작업 중 - 잠시 후 다시 시도")
                return
            state.busy = True

        def _work():
            try:
                pkg_id = self._next_pkg_id()
                cfg = LINES[line_id]
                robot_pos = cfg["robot_pos"]

                print(f"\n[{line_id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                print(f"[{line_id}] 📦 상자 투입: {pkg_id}")

                # 슬롯 확인
                slot = state.free_slot()
                if slot is None:
                    print(f"[{line_id}] 🏭 슬롯 만석 → 로컬 리셋 후 계속")
                    # 슬롯의 기존 박스들 제거
                    for s, pid in state.slots.items():
                        if pid:
                            remove_box(f"/World/SoloBox_{line_id.split('_')[-1]}_{pid.replace('-','_')}")
                    state.reset()
                    slot = 1

                # 상자 스폰 (BG2 디스폰 → 우리 환경 리스폰)
                box_path = spawn_box(line_id, pkg_id)
                time.sleep(0.3)

                # 픽앤플레이스
                print(f"[{line_id}] 🎬 픽앤플레이스 시작 (슬롯 {slot})")
                if DEMO_MODE == "HDF5_REPLAY" and HDF5_AVAILABLE:
                    ok = pick_and_place_replay(
                        slot_num=slot,
                        robot_articulation=None,  # 내일 Articulation 연결 시 교체
                        box_prim_path=box_path,
                        realtime=True,
                    )
                    if not ok:
                        print(f"[{line_id}] ❌ HDF5 실패 → Dummy Teleport")
                        teleport_to_slot(box_path, robot_pos, slot)
                else:
                    time.sleep(1.0)   # 짧은 딜레이로 모션 시뮬레이션
                    teleport_to_slot(box_path, robot_pos, slot)

                state.fill_slot(slot, pkg_id)
                filled = sum(1 for v in state.slots.values() if v)
                print(f"[{line_id}] ✅ 슬롯 {slot} 배치 완료")
                print(f"[{line_id}] 📊 {state.status()}")

                if filled == 4:
                    print(f"[{line_id}] 🏭 4슬롯 만석! (실제 시연: AMR 자동 출동)")
                    print(f"[{line_id}]    Solo 모드: 5초 후 자동 리셋")
                    time.sleep(5.0)
                    state.reset()

            finally:
                state.busy = False

        t = threading.Thread(target=_work, daemon=True)
        t.start()

    def auto_start(self):
        """3개 라인 순환 자동 투입"""
        if self._auto_running:
            print("[Auto] 이미 실행 중")
            return
        self._auto_running = True
        line_ids = list(LINES.keys())
        idx = [0]

        def _loop():
            print(f"\n[Auto] 🔄 자동 투입 시작 (간격: {AUTO_INTERVAL_SEC}초)")
            while self._auto_running:
                lid = line_ids[idx[0] % len(line_ids)]
                self.trigger_line(lid)
                idx[0] += 1
                time.sleep(AUTO_INTERVAL_SEC)
            print("[Auto] ⏹️ 자동 투입 정지")

        self._auto_thread = threading.Thread(target=_loop, daemon=True)
        self._auto_thread.start()

    def auto_stop(self):
        self._auto_running = False

    def reset_all(self):
        for state in self.lines.values():
            state.reset()
        print("[Demo] 🔄 전체 슬롯 리셋 완료")

    def print_status(self):
        print("\n[Status] ──────────────────────────")
        for lid, state in self.lines.items():
            print(f"  {state.status()}")
        print("────────────────────────────────────")

    def run_keyboard_loop(self):
        """터미널 키보드 루프 (Isaac Sim 외부 실행 시)"""
        import tty, termios, select

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            print("\n[Demo] 🎮 키보드 입력 대기 중...")
            while True:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1).upper()
                    if ch == '1':
                        self.trigger_line("sg2_in_01")
                    elif ch == '2':
                        self.trigger_line("sg2_in_02")
                    elif ch == '3':
                        self.trigger_line("sg2_in_03")
                    elif ch == 'A':
                        self.auto_start()
                    elif ch == 'S':
                        self.auto_stop()
                    elif ch == 'R':
                        self.reset_all()
                    elif ch == 'P':
                        self.print_status()
                    elif ch in ('Q', '\x03'):
                        print("\n[Demo] 종료")
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ============================================================
# 진입점 (Isaac Sim Script Editor exec() 방식)
# ============================================================
demo = SH5SoloDemo()

if ISAAC_AVAILABLE:
    # Isaac Sim 환경: 키보드 폴링 스레드 + Update 콜백
    import carb.input

    _keyboard = carb.input.acquire_input_interface()
    _input = omni.kit.app.get_app().get_input_interface() if hasattr(
        omni.kit.app.get_app(), 'get_input_interface') else None

    _key_map = {
        carb.input.KeyboardInput.KEY_1: "sg2_in_01",
        carb.input.KeyboardInput.KEY_2: "sg2_in_02",
        carb.input.KeyboardInput.KEY_3: "sg2_in_03",
    }
    _pressed = set()

    def _on_update(e):
        global _pressed
        for key, line_id in _key_map.items():
            try:
                if _keyboard.get_keyboard_value(None, key) > 0.5:
                    if key not in _pressed:
                        _pressed.add(key)
                        demo.trigger_line(line_id)
                else:
                    _pressed.discard(key)
            except:
                pass

    _sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        _on_update, name="sh5_solo_demo_key"
    )

    print("\n[Solo Demo] ✅ Isaac Sim 키 바인딩 등록 완료")
    print("[Solo Demo] Isaac Sim 창에서 키보드 1/2/3 눌러서 테스트")
    print("[Solo Demo] 터미널에서 직접 호출:")
    print("           demo.trigger_line('sg2_in_01')")
    print("           demo.trigger_line('sg2_in_02')")
    print("           demo.trigger_line('sg2_in_03')")
    print("           demo.auto_start()   # 자동 순환")
    print("           demo.auto_stop()")
    print("           demo.reset_all()")

else:
    # 터미널 단독 실행
    demo.run_keyboard_loop()
