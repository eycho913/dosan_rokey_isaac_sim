#!/usr/bin/env python3
"""
=============================================================================
  sh5_replay_ros2.py  — SH5 HDF5 Replay + ROS2 통합 컨트롤러
=============================================================================

실행 방법:
  isaac-python /home/rokey/dev_ws/coupang_ws/scripts/sh5_replay_ros2.py \
    --usd /home/rokey/dev_ws/final_coupan.usd

시나리오:
  1. /sim/sg2_spawn_trigger 토픽 수신 (BG2→SH5 신호)
  2. HDF5 에피소드에서 상자 스폰 (robot offset 자동 적용)
  3. check_warehouse_status 중복 검사
     - 중복 → 상자 제거 + 스킵 (AMR 회수는 DB가 자동 처리)
     - 신규 → HDF5 Replay pick&place
  4. report_inbound_progress 적재 보고

핵심 설계:
  - isaac-python AppLauncher로 기존 USD 씬(final_coupan.usd)을 직접 열음
  - omni.isaac.core.articulations.Articulation으로 로봇 3대 연결
  - ROS2는 백그라운드 스레드에서 spin
  - 메인 루프는 Isaac Sim simulation step 유지
=============================================================================
"""

import argparse
import os
import sys
import threading
import time
import json
import csv
import queue
from pathlib import Path

# ── AppLauncher (Isaac Sim 런치) ──────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SH5 HDF5 Replay + ROS2")
parser.add_argument(
    "--usd",
    type=str,
    default="/home/rokey/dev_ws/Collected_finalfac/finalfac.usd",
    help="Isaac Sim에서 열 씬 USD 파일 경로",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Sim API (AppLauncher 이후에 임포트) ─────────────────────────────
import omni.usd
import omni.timeline

from omni.isaac.dynamic_control import _dynamic_control as dc_lib
import numpy as np
from pxr import UsdGeom, Sdf, Gf

# =============================================================================
# DCRobotAdapter: dynamic_control → hdf5_replay_player 인터페이스 어댑터
# (Isaac Sim 5.x에서 omni.isaac.core 제거 → dynamic_control로 대체)
# =============================================================================
class DCRobotAdapter:
    """omni.isaac.dynamic_control을 hdf5_replay_player가 기대하는 API로 래핑"""
    def __init__(self, prim_path: str, world_pos: tuple):
        self.prim_path  = prim_path
        self._world_pos = np.array(world_pos)
        self._dc        = None
        self._art       = None
        self.num_dof    = 0

    def initialize(self):
        self._dc  = dc_lib.acquire_dynamic_control_interface()
        self._art = self._dc.get_articulation(self.prim_path)
        if not self._art:   # 0 or None = invalid handle
            raise RuntimeError(f"Articulation not found: {self.prim_path}")
        self.num_dof = self._dc.get_articulation_dof_count(self._art)
        print(f"  [DC] {self.prim_path} → {self.num_dof} DOF")

    def set_joint_position_targets(self, positions: np.ndarray):
        if self._art is not None:
            self._dc.set_articulation_dof_position_targets(self._art, positions.tolist())

    def get_joint_positions(self) -> np.ndarray:
        if self._art is None:
            return np.zeros(self.num_dof)
        states = self._dc.get_articulation_dof_states(self._art, dc_lib.STATE_POS)
        return np.array([s[0] for s in states])

    def get_world_pose(self):
        return (self._world_pos, np.array([1.0, 0.0, 0.0, 0.0]))

# ── HDF5 Replay 모듈 ─────────────────────────────────────────────────────
sys.path.insert(0, "/home/rokey/dev_ws/coupang_ws/scripts")
from hdf5_replay_player import HDF5EpisodeLoader, TrajectoryReplayPlayer

# ROS2는 별도 브리지로 처리 (python 3.11 환경에서 rclpy 문제)
# ros2_sh5_bridge.py 를 별도 터미널에서 실행할 것
QUEUE_FILE = "/tmp/sh5_queue.jsonl"

# =============================================================================
# ★ 설정  —  이 블록만 수정
# =============================================================================

# ── 로봇 Prim 경로 (Isaac Sim Stage 실측값) ───────────────────────────────
ROBOT_PRIMS = {
    "sg2_in_01": "/World/sh5_line1/Robot/base_link/base_link",
    "sg2_in_02": "/World/sh5_line2/Robot/base_link/base_link",
    "sg2_in_03": "/World/sh5_line3/Robot/base_link/base_link",
}

# ── 로봇 월드 좌표 (Stage Property에서 측정한 실측값) ─────────────────────
ROBOT_POS = {
    "sg2_in_01": (7.5,  3.0, 0.0),
    "sg2_in_02": (7.5, -1.5, 0.0),
    "sg2_in_03": (7.5, -6.0, 0.0),
}

# ── 작업대 ID/QR 매핑 ────────────────────────────────────────────────────
WORKSTATION_ID = {"sg2_in_01": "WS01", "sg2_in_02": "WS02", "sg2_in_03": "WS03"}
WORKSTATION_QR = {
    "sg2_in_01": "WORKSTATION_WS01",
    "sg2_in_02": "WORKSTATION_WS02",
    "sg2_in_03": "WORKSTATION_WS03",
}

# ── 패키지 CSV 경로 ───────────────────────────────────────────────────────
QR_DATA_DIR = Path("/home/rokey/dev_ws/qr_data")

# ── 순차 실행 모드 (FPS 보호: True=한 번에 1대만 동작) ────────────────────
SEQUENTIAL_MODE = True

# ── 상자 USD (없으면 Cube 자동 생성) ─────────────────────────────────────
BOX_USD = "/home/rokey/dev_ws/assets/sh5_box.usd"

# =============================================================================
# 패키지 메타 로더 (CSV → customer_name, qr_id)
# =============================================================================
_pkg_cache: dict[str, dict] = {}

def _load_pkg_csv():
    if _pkg_cache:
        return
    for csv_path in QR_DATA_DIR.glob("packages_*.csv"):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = row.get("package_id", "").strip()
                    if pid:
                        _pkg_cache[pid] = {
                            "customer_name": row.get("customer_name", "UNKNOWN").strip(),
                            "qr_id": row.get("qr_id", pid.replace("PKG_", "QR_")).strip(),
                        }
        except Exception as e:
            print(f"[CSV] 로드 오류 {csv_path}: {e}")
    print(f"[CSV] ✅ {len(_pkg_cache)}개 패키지 로드")

def get_pkg_info(package_id: str) -> dict:
    _load_pkg_csv()
    if package_id in _pkg_cache:
        return _pkg_cache[package_id]
    qr_id = package_id.replace("PKG_", "QR_") if package_id.startswith("PKG_") else package_id
    return {"customer_name": "UNKNOWN", "qr_id": qr_id}

# =============================================================================
# =============================================================================
# USD 유틸 (스레드 안전을 위한 전역 락)
# =============================================================================
_USD_LOCK = threading.Lock()   # 주 스레드 + 워커 스레드 간 USD 조작 직렬화

def _stage():
    return omni.usd.get_context().get_stage()

def _spawn_box(path: str, pos: tuple) -> bool:
    """USD Stage에 상자 프림을 생성 (외부 USD 참조 없이 단순 Cube 사용 → OOM 방지)"""
    with _USD_LOCK:
        try:
            stage = _stage()
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(Sdf.Path(path))
            # 외부 USD 로드 금지 (메모리 부족 방지) → 상시 단순 Cube
            cube = UsdGeom.Cube.Define(stage, path)
            cube.GetSizeAttr().Set(0.12)
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            return True
        except Exception as e:
            print(f"  [Spawn] USD 오류: {e}")
            return False

def _remove_prim(path: str):
    with _USD_LOCK:
        try:
            stage = _stage()
            if stage.GetPrimAtPath(path).IsValid():
                stage.RemovePrim(Sdf.Path(path))
        except Exception:
            pass

# =============================================================================
# 파일 큐 수신자 (별도 브리지 돈시에 ros2_sh5_bridge.py 필요)
# =============================================================================
class FileQueueReader:
    """∕tmp∕sh5_queue.jsonl에서 새 엔트리를 읽어 line_queues에 분배"""
    def __init__(self, line_queues: dict, queue_file: str = QUEUE_FILE):
        self.line_queues = line_queues
        self.queue_file  = queue_file
        self._pos        = 0
        # 파일 필요 시 생성
        if not os.path.exists(queue_file):
            open(queue_file, "w").close()
        self._pos = os.path.getsize(queue_file)   # 이전 데이터 무시
        print(f"[FileQueue] 파일 큐: {queue_file}")
        print(f"[FileQueue] ✅ 신호 대기 중 (ros2_sh5_bridge.py 도 실행하세요)")

    def poll(self):
        """매 프레임 호임하여 새 엔트리를 큐에 저장"""
        try:
            size = os.path.getsize(self.queue_file)
        except OSError:
            return
        if size <= self._pos:
            return
        with open(self.queue_file, "r") as f:
            f.seek(self._pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                line_id = payload.get("target_line", "")
                if line_id in self.line_queues:
                    self.line_queues[line_id].put(payload)
                    print(f"[FileQueue] 📨 {line_id} ← {payload.get('package_id', '?')}")
            self._pos = f.tell()

    def check_duplicate(self, *args) -> bool:
        """브리지 모드에서는 중복 검사 서비스 호출 불가 → 항상 신규 처리"""
        print("  [중복검사] FileQueue 모드 → 신규 처리 (Bridge를 통해 DB 연동 미지원)")
        return False

    def report(self, ws_id, ws_qr, line_id, pkg_id, qr_id, slot):
        """브리지 모드에서는 보고 로그만 출력"""
        print(f"  [보고] {ws_id} 슬롯{slot} ← {pkg_id} (ROS2 연동 없음 → 로그만)")

# =============================================================================
# SH5 라인 작업 단위
# =============================================================================
class SH5LineWorker:
    def __init__(self, line_id: str, robot_art: DCRobotAdapter, ros: FileQueueReader):
        self.line_id    = line_id
        self.robot_art  = robot_art
        self.robot_pos  = ROBOT_POS[line_id]
        self.ws_id      = WORKSTATION_ID[line_id]
        self.ws_qr      = WORKSTATION_QR[line_id]
        self.ros        = ros
        self.filled     = 0
        self._busy      = False

        self.queue: queue.Queue = queue.Queue()
        print(f"[{line_id}] ✅ 초기화 완료 | WS={self.ws_id}")

    def step(self):
        """Isaac Sim update 루프에서 매 프레임 호출."""
        if self._busy:
            return
        try:
            payload = self.queue.get_nowait()
        except queue.Empty:
            return

        threading.Thread(target=self._process, args=(payload,), daemon=True).start()

    def _process(self, payload: dict):
        # 백그라운드 스레드에서 asyncio 이벤트 루프 생성 (Isaac Sim USD ops 필요)
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._busy = True
        pkg_id = payload.get("package_id", f"PKG_MOCK_{int(time.time())}")
        print(f"\n[{self.line_id}] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[{self.line_id}] 📩 수신: {pkg_id}")

        try:
            meta     = get_pkg_info(pkg_id)
            customer = meta["customer_name"]
            qr_id    = meta["qr_id"]
            slot     = (self.filled % 4) + 1

            # STEP 1: HDF5에서 상자 스폰 위치 추출 + 스폰
            box_path, episode = self._spawn_box_hdf5(pkg_id, slot)
            time.sleep(0.5)   # 물리 안정화

            # STEP 2: 중복 검사
            if self.ros.check_duplicate(customer, pkg_id, qr_id):
                print(f"[{self.line_id}] ⚠️ 중복 감지 → 상자 제거")
                _remove_prim(box_path)
                return

            # STEP 3: HDF5 Replay pick & place
            ok = self._do_replay(box_path, episode)
            if not ok:
                print(f"[{self.line_id}] ❌ Replay 실패 → 스킵")
                _remove_prim(box_path)
                return

            # STEP 4: DB 보고
            self.filled += 1
            self.ros.report(self.ws_id, self.ws_qr, self.line_id, pkg_id, qr_id, slot)

        except Exception as e:
            print(f"[{self.line_id}] ❌ 처리 오류: {e}")
        finally:
            self._busy = False

    def _spawn_box_hdf5(self, pkg_id: str, slot: int):
        import numpy as np
        safe     = pkg_id.replace("-", "_").replace(" ", "_")
        box_path = f"/World/SH5ros2Box_{self.line_id[-2:]}_{safe}"

        episode  = None
        try:
            loader   = HDF5EpisodeLoader(slot_num=slot)
            episode  = loader.load_random_episode()
            rec_robot = np.array(episode["robot_initial_pose"][:3])
            cur_robot = np.array(self.robot_pos)
            offset    = cur_robot - rec_robot
            hdf5_box  = np.array(episode["box_initial_pose"][:3])
            spawn_pos = tuple(hdf5_box + offset)
            print(f"  [Spawn] {episode['demo_key']} | 상자: ({spawn_pos[0]:.3f}, {spawn_pos[1]:.3f}, {spawn_pos[2]:.3f})")
        except Exception as e:
            print(f"  [Spawn] HDF5 오류({e}) → 기본 offset 사용")
            import numpy as np
            spawn_pos = tuple(np.array(self.robot_pos) + np.array([1.5, -1.5, 0.83]))

        _spawn_box(box_path, spawn_pos)
        return box_path, episode

    def _do_replay(self, box_path: str, episode) -> bool:
        if episode is None:
            print("  [Replay] episode 없음 → 스킵")
            return False
        try:
            player = TrajectoryReplayPlayer(
                robot_articulation=self.robot_art,
                box_prim_path=box_path,
                robot_world_pos=self.robot_pos,
            )
            return player.play_episode(episode, realtime=True)
        except Exception as e:
            print(f"  [Replay] 오류: {e}")
            return False

# =============================================================================
# 메인 진입점
# =============================================================================
def main():
    # ── 1. USD 씬 열기 ─────────────────────────────────────────────────────
    print(f"\n[SH5 ROS2] USD 씬 로드 중: {args_cli.usd}")
    omni.usd.get_context().open_stage(args_cli.usd)

    # 씬 로드 완료 대기 (대형 USD는 비동기 로드)
    import time as _time
    _time.sleep(3.0)

    # 물리 시뮬레이션 시작 (SimulationContext 없이 → OOM 방지)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()   # 첫 프레임 렌더
    print("[SH5 ROS2] ✅ 씬 로드 + 물리 시작")

    # ── 2. 로봇 연결 (dynamic_control 어댑터) ───────────────────────────────
    print("[SH5 ROS2] 로봇 연결 중 (dynamic_control)...")
    robots: dict[str, DCRobotAdapter] = {}
    for line_id, prim_path in ROBOT_PRIMS.items():
        try:
            r = DCRobotAdapter(prim_path, ROBOT_POS[line_id])
            r.initialize()
            print(f"  ✅ {line_id}: {prim_path} ({r.num_dof} DOF)")
            robots[line_id] = r
        except Exception as e:
            print(f"  ❌ {line_id} 연결 실패: {e}")

    # ── 3. 파일 큐 리더 (별도 ros2_sh5_bridge.py 필요) ──────────────────────
    line_queues = {lid: queue.Queue() for lid in ROBOT_PRIMS}
    file_reader = FileQueueReader(line_queues)

    # ── 4. 라인 워커 생성 ───────────────────────────────────────────────────
    workers = {}
    for line_id in ROBOT_PRIMS:
        if line_id in robots:
            w = SH5LineWorker(line_id, robots[line_id], file_reader)
            w.queue = line_queues[line_id]
            workers[line_id] = w

    print("\n" + "="*60)
    print("  SH5 HDF5 Replay + 파일큐 컨트롤러 가동!")
    print(f"  연결된 로봇: {list(workers.keys())}")
    print(f"  파일 큐: {QUEUE_FILE}")
    print("  추가 터미널에서 실행:  python3 ros2_sh5_bridge.py")
    print("="*60 + "\n")

    # ── 5. 메인 시뮬레이션 루프 ─────────────────────────────────────────────
    while simulation_app.is_running():
        simulation_app.update()   # sim.step() 대신 (OOM 방지)
        file_reader.poll()        # 새 ROS2 메시지 확인

        if SEQUENTIAL_MODE:
            # 한 번에 1대만 동작
            if not any(w._busy for w in workers.values()):
                for w in workers.values():
                    if not w.queue.empty():
                        w.step()
                        break
        else:
            for w in workers.values():
                w.step()

    # ── 종료 ────────────────────────────────────────────────────────────────
    simulation_app.close()

if __name__ == "__main__":
    main()
