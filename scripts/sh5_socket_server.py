"""
sh5_socket_server.py
====================
[SH5/AMR PC - 수신측 서버]

BG2 PC가 상자를 디스폰하면 소켓으로 JSON 메시지를 보냄.
이 서버가 수신해서 Isaac Sim에 상자를 리스폰시킴.

실행 (Isaac Sim Script Editor):
  exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_socket_server.py', encoding='utf-8').read())

또는 터미널에서 단독 실행 (Isaac Sim 없이 테스트):
  python3 sh5_socket_server.py

통신 프로토콜:
  - TCP, 포트 9000 (변경 가능)
  - 메시지: JSON 한 줄 (개행문자 종료)
  - 형식: {"package_id": "PKG_001", "target_line": "sg2_in_01", "timestamp": 1234567890.0}
  - 응답: {"status": "ok"} 또는 {"status": "error", "msg": "..."}
"""

import socket
import threading
import json
import time
import os
import sys
import queue

# ============================================================
# 서버 설정
# ============================================================
SERVER_HOST = "0.0.0.0"    # 모든 인터페이스에서 수신
SERVER_PORT = 9000          # BG2 PC와 맞춰야 함
BUFFER_SIZE = 4096

# ============================================================
# Isaac Sim / HDF5 연결
# ============================================================
ISAAC_AVAILABLE = False
try:
    import omni.usd
    import omni.kit.app
    from pxr import UsdGeom, Sdf, Gf
    ISAAC_AVAILABLE = True
    print("[SocketServer] ✅ Isaac Sim 연결")
except ImportError:
    print("[SocketServer] ⚠️ Isaac Sim 외부 (테스트 모드)")

HDF5_AVAILABLE = False
try:
    sys.path.insert(0, '/home/rokey/dev_ws/coupang_ws/scripts')
    from hdf5_replay_player import pick_and_place_replay
    HDF5_AVAILABLE = True
    print("[SocketServer] ✅ HDF5 모듈 로드")
except ImportError:
    print("[SocketServer] ⚠️ HDF5 모듈 없음")

# ============================================================
# 좌표 설정
# ============================================================
DEMO_MODE = "HDF5_REPLAY"   # "HDF5_REPLAY" | "DUMMY_TELEPORT"

LINES = {
    "sg2_in_01": {"spawn_pos": (9.0,  1.5, 0.83), "robot_pos": (7.5,  3.0, 0.0)},
    "sg2_in_02": {"spawn_pos": (9.0, -3.0, 0.83), "robot_pos": (7.5, -1.5, 0.0)},
    "sg2_in_03": {"spawn_pos": (9.0, -7.5, 0.83), "robot_pos": (7.5, -6.0, 0.0)},
}

SLOT_TARGETS_LOCAL = {
    1: (0.0, -1.5, 1.2),
    2: (0.0, -1.5, 1.2),
    3: (0.0, -1.5, 0.5),
    4: (0.0, -1.5, 0.5),
}

BOX_USD = "/home/rokey/dev_ws/assets/ffw_description/usd/ffw_sh5_follower_custom.usd"

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
    cfg = LINES.get(line_id, list(LINES.values())[0])
    pos = cfg["spawn_pos"]
    safe = pkg_id.replace("-", "_").replace(" ", "_")
    path = f"/World/SockBox_{line_id.split('_')[-1]}_{safe}"

    stage = get_stage()
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

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    print(f"  [Spawn] 📦 리스폰: {path} @ {pos}")
    return path


def teleport_to_slot(box_path: str, robot_pos: tuple, slot: int):
    local = SLOT_TARGETS_LOCAL[slot]
    world = (robot_pos[0] + local[0], robot_pos[1] + local[1], local[2])
    stage = get_stage()
    if stage is None:
        print(f"  [Teleport] Mock → 슬롯{slot} {world}")
        return
    prim = stage.GetPrimAtPath(box_path)
    if prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*world))
        print(f"  [Teleport] 🚀 슬롯{slot} → {world}")

# ============================================================
# 슬롯 상태
# ============================================================
class SlotState:
    def __init__(self):
        self.slots = {1: None, 2: None, 3: None, 4: None}
        self.lock = threading.Lock()

    def free_slot(self):
        with self.lock:
            for k, v in self.slots.items():
                if v is None:
                    return k
        return None

    def fill(self, slot, pkg_id):
        with self.lock:
            self.slots[slot] = pkg_id

    def reset(self):
        with self.lock:
            self.slots = {1: None, 2: None, 3: None, 4: None}
        print("  [Slot] 🔄 슬롯 리셋")

    def status(self):
        with self.lock:
            filled = sum(1 for v in self.slots.values() if v)
            return f"[{filled}/4] {'■'*filled}{'□'*(4-filled)}"

_states = {lid: SlotState() for lid in LINES}

# ============================================================
# 메시지 처리 (수신된 JSON → 리스폰 → 픽앤플레이스)
# ============================================================
_task_queue = queue.Queue()   # worker 스레드에서 Isaac Sim 조작

def process_payload(payload: dict) -> str:
    """
    BG2로부터 받은 payload 처리.
    실제 Isaac Sim 조작은 worker 스레드에서 수행.
    """
    package_id = payload.get("package_id", "PKG_UNKNOWN")
    target_line = payload.get("target_line", "sg2_in_01")

    if target_line not in LINES:
        return f"error: 알 수 없는 라인 {target_line}"

    _task_queue.put({"package_id": package_id, "target_line": target_line})
    return "ok"


def worker_loop():
    """Isaac Sim 조작 전용 스레드 (소켓 스레드와 분리)"""
    while True:
        try:
            task = _task_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        pkg_id = task["package_id"]
        line_id = task["target_line"]
        cfg = LINES[line_id]
        state = _states[line_id]

        print(f"\n[Worker] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[Worker] 📦 BG2 수신: {pkg_id} → {line_id}")

        # 슬롯 확인
        slot = state.free_slot()
        if slot is None:
            print(f"[Worker] 🏭 만석 → 리셋 후 슬롯 1 사용")
            state.reset()
            slot = 1

        # ① 리스폰
        box_path = spawn_box(line_id, pkg_id)
        time.sleep(0.5)

        # ② 픽앤플레이스
        if DEMO_MODE == "HDF5_REPLAY" and HDF5_AVAILABLE:
            ok = pick_and_place_replay(
                slot_num=slot,
                robot_articulation=None,
                box_prim_path=box_path,
                realtime=True,
            )
            if not ok:
                teleport_to_slot(box_path, cfg["robot_pos"], slot)
        else:
            time.sleep(1.0)
            teleport_to_slot(box_path, cfg["robot_pos"], slot)

        state.fill(slot, pkg_id)
        print(f"[Worker] ✅ {line_id} 슬롯{slot} 완료 | {state.status()}")

# ============================================================
# TCP 소켓 서버
# ============================================================
class BoxSpawnServer:
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT):
        self.host = host
        self.port = port
        self._running = False
        self._server_sock = None

    def start(self):
        self._running = True

        # worker 스레드 시작
        t_worker = threading.Thread(target=worker_loop, daemon=True)
        t_worker.start()

        # 서버 스레드 시작
        t_server = threading.Thread(target=self._serve, daemon=True)
        t_server.start()

        print(f"\n[SocketServer] 🚀 서버 시작")
        print(f"[SocketServer]    수신 주소: {self.host}:{self.port}")
        print(f"[SocketServer]    BG2 PC에서 이 PC의 IP:{self.port} 로 연결")
        print(f"[SocketServer]    내 IP 확인: hostname -I")

    def stop(self):
        self._running = False
        if self._server_sock:
            self._server_sock.close()

    def _serve(self):
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(5)
        print(f"[SocketServer] 🔌 대기 중... (Ctrl+C 또는 server.stop()으로 종료)")

        while self._running:
            try:
                self._server_sock.settimeout(1.0)
                conn, addr = self._server_sock.accept()
                print(f"\n[SocketServer] 🔗 연결: {addr[0]}:{addr[1]}")
                t = threading.Thread(
                    target=self._handle_client, args=(conn, addr), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_client(self, conn: socket.socket, addr):
        try:
            data = b""
            while True:
                chunk = conn.recv(BUFFER_SIZE)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:   # 개행문자가 메시지 종료 신호
                    break

            raw = data.decode("utf-8").strip()
            print(f"[SocketServer] 📩 수신: {raw}")

            payload = json.loads(raw)
            result = process_payload(payload)

            # 응답 전송
            resp = json.dumps({"status": result}) + "\n"
            conn.sendall(resp.encode("utf-8"))
            print(f"[SocketServer] 📤 응답: {resp.strip()}")

        except json.JSONDecodeError as e:
            err = json.dumps({"status": "error", "msg": f"JSON 파싱 실패: {e}"}) + "\n"
            conn.sendall(err.encode("utf-8"))
        except Exception as e:
            print(f"[SocketServer] ⚠️ 클라이언트 처리 오류: {e}")
        finally:
            conn.close()


# ============================================================
# 진입점
# ============================================================
server = BoxSpawnServer(host=SERVER_HOST, port=SERVER_PORT)
server.start()

if ISAAC_AVAILABLE:
    print("\n[SocketServer] Isaac Sim 모드: 백그라운드 수신 대기")
    print("  수동 테스트: server._task_queue.put({'package_id':'PKG_TEST','target_line':'sg2_in_01'})")
else:
    # 터미널 단독 실행 시 블로킹 대기
    print("\n[SocketServer] 터미널 모드: Ctrl+C로 종료")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("[SocketServer] 종료")
