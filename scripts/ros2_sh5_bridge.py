#!/usr/bin/env python3
"""
ros2_sh5_bridge.py — ROS2 ↔ Isaac Sim 양방향 브릿지 v2
=========================================================
역할:
  1. /sim/sg2_spawn_trigger 구독
     → check_warehouse_status 서비스 호출
     → 결과(is_duplicate) 포함해 /tmp/sh5_queue.jsonl 기록
  2. /tmp/sh5_report_req.jsonl 모니터링
     → report_inbound_progress 서비스 호출

실행:
  python3 ros2_sh5_bridge.py
"""

import json, os, sys, time, threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

QUEUE_FILE      = "/tmp/sh5_queue.jsonl"      # bridge → Isaac (트리거)
QR_REQ_FILE     = "/tmp/sh5_qr_req.jsonl"     # Isaac → bridge (QR 확인 요청)
QR_RESULT_FILE  = "/tmp/sh5_qr_result.jsonl"  # bridge → Isaac (DB 체크 결과)
REPORT_REQ_FILE = "/tmp/sh5_report_req.jsonl" # Isaac → bridge (입고 보고)
PAUSE_FILE_TEMPLATE = "/tmp/sh5_pause_{robot_id}.json" # 라인별 개별 pause 파일
PAUSE_FILE          = "/tmp/sh5_pause.json"             # 전체 공통 폴백(하위호환)

# 폴즈 대상 로봇 ID 리스트 (관제탑이 /{robot_id}/pause_status 주제)
PAUSE_ROBOT_IDS  = ["sg2_in_01", "sg2_in_02", "sg2_in_03"]

try:
    from cobot3_interfaces.srv import CheckWarehouseStatus, ReportInboundProgress
    HAS_SRV = True
except ImportError:
    HAS_SRV = False
    print("[Bridge] ⚠️  cobot3_interfaces 없음 → 서비스 미사용")


class BridgeNode(Node):
    def __init__(self):
        super().__init__("sh5_ros2_bridge")

        with open(QUEUE_FILE,      "w") as f: pass
        with open(QR_REQ_FILE,     "w") as f: pass
        with open(QR_RESULT_FILE,  "w") as f: pass
        with open(REPORT_REQ_FILE, "w") as f: pass
        # 라인별 pause 파일 전체 초기화 (false = 재개 상태)
        for robot_id in PAUSE_ROBOT_IDS:
            pf = PAUSE_FILE_TEMPLATE.format(robot_id=robot_id)
            with open(pf, "w") as f:
                json.dump({"paused": False}, f)

        self._check_client  = None
        self._report_client = None
        if HAS_SRV:
            self._check_client  = self.create_client(CheckWarehouseStatus,  "check_warehouse_status")
            self._report_client = self.create_client(ReportInboundProgress, "report_inbound_progress")

        self.create_subscription(String, "/sim/sg2_spawn_trigger", self._on_trigger, 10)

        # pause_status 구독 (/{robot_id}/pause_status) — 라인별 개별 파일에 쓰기
        for robot_id in PAUSE_ROBOT_IDS:
            pause_topic = f"/{robot_id}/pause_status"
            callback = self._make_pause_callback(robot_id)  # ← 라인 ID 바인딩
            self.create_subscription(Bool, pause_topic, callback, 10)
            self.get_logger().info(f"[Bridge] 폰즈 리스너: {pause_topic}")

        self.get_logger().info("[Bridge] ✅ 시작")
        self.get_logger().info(f"  트리거 : {QUEUE_FILE}")
        self.get_logger().info(f"  QR요청 : {QR_REQ_FILE}  →  {QR_RESULT_FILE}")
        self.get_logger().info(f"  보고   : {REPORT_REQ_FILE}")
        self.get_logger().info(f"  폰즈   : {PAUSE_FILE_TEMPLATE} (\ub77c\uc778\ubcc4 \uac1c\ubcc4, \ud1a0\ud53d: {PAUSE_ROBOT_IDS})")

        self._report_pos = 0
        self._qr_req_pos = 0
        threading.Thread(target=self._poll_report_requests, daemon=True).start()
        threading.Thread(target=self._poll_qr_requests,    daemon=True).start()

    def _make_pause_callback(self, robot_id: str):
        """라인별 pause 콜백 팩토리 — robot_id를 클로저로 바인딩"""
        pause_file = PAUSE_FILE_TEMPLATE.format(robot_id=robot_id)

        def _on_pause(msg: Bool):
            paused = bool(msg.data)
            # 1) 라인별 전용 파일에 쓰기
            with open(pause_file, "w") as f:
                json.dump({"paused": paused}, f)
            status = "🔴 일시정지" if paused else "🟢 재개"
            print(f"\n[Bridge] ━━━━ ⏸️  pause_status ━━━━")
            print(f"         라인  : {robot_id}")
            print(f"         상태  : {status}")
            print(f"         파일  : {pause_file}")
            print(f"[Bridge] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        return _on_pause



    # ── check_warehouse_status ────────────────────────────────────────────
    def _check_warehouse(self, pkg_id: str, qr_id: str, customer_name: str = "") -> bool:
        """check_warehouse_status 서비스 호출. 중복이면 True 반환."""
        if self._check_client is None:
            print(f"[Bridge] ⚠️  check_warehouse_status 클라이언트 없음 → 신규 처리")
            return False
        if not self._check_client.wait_for_service(timeout_sec=1.5):
            self.get_logger().warn("[Bridge] check_warehouse_status 서비스 없음 → 신규 처리")
            return False
        req = CheckWarehouseStatus.Request()
        req.package_id    = pkg_id
        req.qr_id         = qr_id
        req.customer_name = customer_name
        print(f"\n[Bridge] ━━━━ 📡 check_warehouse_status 요청 ━━━━")
        print(f"         package_id   : {pkg_id}")
        print(f"         qr_id        : {qr_id}")
        print(f"         customer_name: {customer_name}")
        future = self._check_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result() is not None:
            dup = future.result().is_already_in_warehouse
            mark = '🔴 중복' if dup else '🟢 신규'
            print(f"         응답: {mark} (is_already_in_warehouse={dup})")
            print(f"[Bridge] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            return dup
        print(f"         응답: ❌ 없음 → 신규 처리")
        print(f"[Bridge] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return False

    # ── /sim/sg2_spawn_trigger 콜백 ──────────────────────────────────
    def _on_trigger(self, msg: String):
        """BG2 트리거 수신 → 패키지 정보를 큐에 기록 (is_duplicate는 Isaac이 QR 스캔 후 요청)"""
        try:
            payload = json.loads(msg.data)
        except Exception:
            payload = {"raw": msg.data}

        pkg_id  = payload.get("package_id", "")
        line_id = payload.get("target_line", "")
        self.get_logger().info(f"[Bridge] 📨 트리거: {pkg_id} → {line_id}")

        # is_duplicate는 Isaac Sim이 QR 스캔 후 직접 요청
        # (QR_REQ_FILE 기여 구조) → 여기서는 패키지 정보만 전달
        with open(QUEUE_FILE, "a") as f:
            f.write(json.dumps(payload) + "\n")
            f.flush()
        self.get_logger().info(f"[Bridge] → Isaac Sim 전달 완료")

    # ── QR 확인 요청 파일 polling (Isaac Sim이 QR 스캔 후 요청) ─────
    def _poll_qr_requests(self):
        while rclpy.ok():
            try:
                size = os.path.getsize(QR_REQ_FILE)
            except OSError:
                time.sleep(0.3); continue

            if size < self._qr_req_pos:
                # 파일이 초기화됨
                self._qr_req_pos = 0
            elif size == self._qr_req_pos:
                time.sleep(0.1); continue

            with open(QR_REQ_FILE, "r") as f:
                f.seek(self._qr_req_pos)
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data   = json.loads(line)
                        pkg_id = data.get("pkg_id", "")
                        qr_id  = data.get("qr_id", pkg_id)
                        cust   = data.get("customer_name", "")
                        self.get_logger().info(f"[Bridge] 🔍 check_warehouse_status({qr_id})")
                        is_dup = self._check_warehouse(pkg_id, qr_id, cust)
                        # 결과를 Isaac Sim에 반환
                        result = {"pkg_id": pkg_id, "is_duplicate": is_dup}
                        with open(QR_RESULT_FILE, "a") as rf:
                            rf.write(json.dumps(result) + "\n")
                            rf.flush()
                        self.get_logger().info(f"[Bridge] {'🔴' if is_dup else '🟢'} 결과 기록: {pkg_id}")
                    except Exception as e:
                        print(f"[Bridge] QR 요청 오류: {e}")
                self._qr_req_pos = f.tell()
            time.sleep(0.05)

    # ── 보고 요청 파일 polling ────────────────────────────────────────────
    def _poll_report_requests(self):
        while rclpy.ok():
            try:
                size = os.path.getsize(REPORT_REQ_FILE)
            except OSError:
                time.sleep(0.5); continue

            if size < self._report_pos:
                # 파일이 초기화됨 (Isaac Sim 재시작 등)
                self._report_pos = 0
            elif size == self._report_pos:
                time.sleep(0.2); continue

            with open(REPORT_REQ_FILE, "r") as f:
                f.seek(self._report_pos)
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        data = json.loads(line)
                        self._call_report(data)
                    except Exception as e:
                        print(f"[Bridge] 보고 파싱 오류: {e}")
                self._report_pos = f.tell()
            time.sleep(0.1)

    def _call_report(self, data: dict):
        if not HAS_SRV or self._report_client is None:
            print(f"\n[Bridge] ━━━━ 📋 report_inbound_progress [DRY-RUN] ━━━━")
            print(f"         package_id : {data.get('package_id')}")
            print(f"         qr_id      : {data.get('package_qr_id')}")
            print(f"         slot       : {data.get('filled_slots_count')}번 슬롯 배정")
            print(f"[Bridge] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            return
        if not self._report_client.wait_for_service(timeout_sec=1.5):
            self.get_logger().warn("[Bridge] report_inbound_progress 없음")
            return
        req = ReportInboundProgress.Request()
        req.workstation_id      = data.get("workstation_id", "")
        req.workstation_qr_id   = data.get("workstation_qr_id", "")
        req.robot_id            = data.get("robot_id", "")
        req.package_id          = data.get("package_id", "")
        req.package_qr_id       = data.get("package_qr_id", "")
        req.filled_slots_count  = data.get("filled_slots_count", 0)
        print(f"\n[Bridge] ━━━━ 📦 report_inbound_progress 요청 ━━━━")
        print(f"         workstation_id    : {req.workstation_id}")
        print(f"         package_id        : {req.package_id}")
        print(f"         package_qr_id     : {req.package_qr_id}")
        print(f"         filled_slots_count: {req.filled_slots_count}")
        fut = self._report_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        if fut.result() and fut.result().success:
            print(f"         응답: ✅ 보고 성공")
        else:
            print(f"         응답: ❌ 보고 실패")
        print(f"[Bridge] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


def main():
    print(f"[Bridge] Python {sys.version.split()[0]}")
    rclpy.init()
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
