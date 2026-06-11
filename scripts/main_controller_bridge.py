#!/usr/bin/env python3
"""
main_controller_bridge.py
─────────────────────────────────────────────────────────────────────────────
SH5 메인 컨트롤러 통신 브릿지 노드

역할:
  1. /sh5/task_assignment (qr_scanner_node.py 에서 발행) 를 구독한다.
  2. 태스크를 받으면 해당 슬롯의 ACT 모델(.pth)을 로드하고 추론을 실행한다.
     → eval_bc.py 의 ACT 추론 로직과 동일한 방식으로 Isaac Sim 없이도 동작.
  3. 적재 완료 후 ReportInboundProgress 서비스로 관제탑에 진척도를 보고한다.
  4. /fleet/workstation_states, /fleet/amr_states, /fleet/task_events 를
     1Hz 또는 이벤트 발생 시 발행하여 메인 컨트롤러가 상태를 모니터링할 수 있게 한다.

인터페이스 근거: /home/rokey/dev_ws/인터페이스 (1).ipynb  v2.2 규격
─────────────────────────────────────────────────────────────────────────────
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import json
import time
import uuid
import threading
import sys
from pathlib import Path
from typing import Optional

from std_msgs.msg import String

from cobot3_interfaces.srv import ReportInboundProgress

# ── 경로 설정 (eval_bc.py 와 동일한 sys.path) ─────────────────────────────
SCRIPTS_DIR = Path("/home/rokey/dev_ws/coupang_ws/scripts")
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# 슬롯 상수
# ─────────────────────────────────────────────────────────────────────────────
SLOT_COUNT = 4  # 현재 운영 슬롯 수 (인터페이스 명세에는 8칸이나 SH5는 4칸 운영)


class MainControllerBridge(Node):
    """
    메인 컨트롤러(다른 팀원)와 ROS2 서비스/토픽으로 통신하는 브릿지 노드.

    ┌─ 수신 ─────────────────────────────────────────────────────────────────┐
    │  /sh5/task_assignment   (std_msgs/String, JSON)  ← qr_scanner_node    │
    └────────────────────────────────────────────────────────────────────────┘
    ┌─ 발행 ─────────────────────────────────────────────────────────────────┐
    │  /fleet/workstation_states  (std_msgs/String, JSON, 1Hz)              │
    │  /fleet/amr_states          (std_msgs/String, JSON, 1Hz)              │
    │  /fleet/task_events         (std_msgs/String, JSON, event-driven)     │
    │  /sh5/status               (std_msgs/String, JSON, 1Hz)              │
    └────────────────────────────────────────────────────────────────────────┘
    ┌─ 서비스 클라이언트 ─────────────────────────────────────────────────────┐
    │  /report_inbound_progress   (ReportInboundProgress.srv)               │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__("main_controller_bridge")

        # ── 파라미터 ──────────────────────────────────────────
        self.declare_parameter("robot_id",          "sh5_in_01")
        self.declare_parameter("workstation_id",    "WS01")
        self.declare_parameter("workstation_qr_id", "WORKSTATION_WS01")
        self.declare_parameter("current_qr_id",     "QR_0030")   # 로봇 현재 위치 바닥 QR
        self.declare_parameter("report_service",    "/report_inbound_progress")
        self.declare_parameter("use_isaac_sim",     False)       # Isaac Sim 없이 순수 ROS2 모드

        self.robot_id          = self.get_parameter("robot_id").value
        self.workstation_id    = self.get_parameter("workstation_id").value
        self.workstation_qr_id = self.get_parameter("workstation_qr_id").value
        self.current_qr_id     = self.get_parameter("current_qr_id").value
        self.use_isaac_sim     = self.get_parameter("use_isaac_sim").value

        self.cb_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # ── 상태 변수 ─────────────────────────────────────────
        self._filled_slots: list[int] = []         # 채워진 슬롯 번호 목록
        self._current_task: Optional[dict] = None  # 현재 수행 중인 태스크
        self._robot_state: str = "IDLE"            # IDLE / PICKING / NAVIGATING / PLACING / ERROR
        self._task_history: list[dict] = []        # 완료된 태스크 이력

        # ACT 모델 캐시 {slot_id: ACTPolicy 인스턴스}
        self._policy_cache: dict[int, object] = {}

        # ── 서비스 클라이언트 ──────────────────────────────────
        self.report_client = self.create_client(
            ReportInboundProgress,
            self.get_parameter("report_service").value,
            callback_group=self.cb_group,
        )

        # ── 구독 ──────────────────────────────────────────────
        self.create_subscription(
            String,
            "/sh5/task_assignment",
            self._on_task_assignment,
            10,
            callback_group=self.cb_group,
        )

        # ── 발행 ──────────────────────────────────────────────
        self.ws_states_pub   = self.create_publisher(String, "/fleet/workstation_states", 10)
        self.amr_states_pub  = self.create_publisher(String, "/fleet/amr_states", 10)
        self.task_events_pub = self.create_publisher(String, "/fleet/task_events", 10)
        self.sh5_status_pub  = self.create_publisher(String, "/sh5/status", 10)

        # 1Hz 주기 상태 발행
        self.create_timer(1.0, self._publish_fleet_states, callback_group=self.cb_group)

        self.get_logger().info(
            f"[Bridge] 메인 컨트롤러 브릿지 시작 | robot_id={self.robot_id} "
            f"| isaac_sim={'활성' if self.use_isaac_sim else '비활성'}"
        )

    # ─────────────────────────────────────────────────────────
    # 1. 태스크 수신 및 ACT 추론 트리거
    # ─────────────────────────────────────────────────────────

    def _on_task_assignment(self, msg: String):
        """
        qr_scanner_node 에서 발행한 태스크 JSON을 수신한다.
        이미 수행 중인 태스크가 있으면 큐에 보관 (현재는 단순 로깅).
        """
        try:
            task = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"[Bridge] 태스크 JSON 파싱 실패: {e}")
            return

        qr_id   = task.get("qr_id", "UNKNOWN")
        slot_id = task.get("slot_id", 1)
        self.get_logger().info(
            f"[Bridge] 태스크 수신 | QR={qr_id} → Slot {slot_id}"
        )

        with self._lock:
            if self._robot_state != "IDLE":
                self.get_logger().warn(
                    f"[Bridge] 로봇 현재 {self._robot_state} 상태. 태스크 대기열 추가: {qr_id}"
                )
                # TODO: 우선순위 큐 연동 (Redis Sorted Set 규격 §4.②)
                return
            self._current_task = task
            self._robot_state  = "PICKING"

        # 이벤트 발행: ASSIGNED
        self._emit_task_event(task, "ASSIGNED")

        # 백그라운드 스레드로 실행 (ROS2 콜백 블로킹 방지)
        t = threading.Thread(target=self._execute_task, args=(task,), daemon=True)
        t.start()

    # ─────────────────────────────────────────────────────────
    # 2. 태스크 실행 (PICKING → NAVIGATING → PLACING → REPORTING)
    # ─────────────────────────────────────────────────────────

    def _execute_task(self, task: dict):
        """
        태스크를 순서대로 실행한다.
        Isaac Sim 연동 여부에 따라 실제 ACT 추론 또는 시뮬레이션 모드로 동작.
        """
        qr_id       = task["qr_id"]
        slot_id     = task["slot_id"]
        model_path  = task["model_path"]
        use_left    = task.get("use_left_hand", False)

        try:
            # ── Phase 1: PICKING ─────────────────────────────
            self.get_logger().info(f"[Bridge] Phase 1 PICKING | QR={qr_id}")
            self._emit_task_event(task, "PICKING")
            self._set_state("PICKING")

            if self.use_isaac_sim:
                policy = self._load_policy(slot_id, model_path)
                self._run_act_inference(policy, task, phase="pick")
            else:
                # 시뮬레이션/더미 모드: 실제 로봇 없이 슬립으로 대체
                self.get_logger().info(f"[Bridge] [더미] 파지 동작 실행 중 (slot={slot_id})")
                time.sleep(2.0)

            # ── Phase 2: NAVIGATING ───────────────────────────
            self.get_logger().info(f"[Bridge] Phase 2 NAVIGATING → Slot {slot_id}")
            self._emit_task_event(task, "NAVIGATING")
            self._set_state("NAVIGATING")
            time.sleep(1.0)  # 이동 시간 (실제 주행 명령은 eval_bc.py 가 처리)

            # ── Phase 3: PLACING ──────────────────────────────
            self.get_logger().info(f"[Bridge] Phase 3 PLACING | Slot {slot_id}")
            self._emit_task_event(task, "PLACING")
            self._set_state("PLACING")

            if self.use_isaac_sim:
                policy = self._load_policy(slot_id, model_path)
                self._run_act_inference(policy, task, phase="place")
            else:
                self.get_logger().info(f"[Bridge] [더미] 배치 동작 실행 중 (slot={slot_id})")
                time.sleep(2.0)

            # ── Phase 4: 완료 보고 ────────────────────────────
            with self._lock:
                if slot_id not in self._filled_slots:
                    self._filled_slots.append(slot_id)

            success = self._report_inbound_progress(task)
            status  = "COMPLETED" if success else "FAILED"
            self._emit_task_event(task, status)

            self.get_logger().info(
                f"[Bridge] ✅ 태스크 완료 | QR={qr_id} → Slot {slot_id} | 보고: {status}"
            )

        except Exception as e:
            self.get_logger().error(f"[Bridge] 태스크 실행 오류: {e}")
            self._emit_task_event(task, "FAILED")

        finally:
            with self._lock:
                self._task_history.append(task)
                self._current_task = None
                self._robot_state  = "IDLE"

    # ─────────────────────────────────────────────────────────
    # 3. ACT 정책 로드 및 추론
    # ─────────────────────────────────────────────────────────

    def _load_policy(self, slot_id: int, model_path: str):
        """
        슬롯별 ACTPolicy .pth 를 로드한다. 캐시가 있으면 재사용.
        """
        if slot_id in self._policy_cache:
            return self._policy_cache[slot_id]

        try:
            import torch
            from train_act import ACTPolicy, STATE_DIM, ACTION_DIM

            policy = ACTPolicy(state_dim=STATE_DIM, action_dim=ACTION_DIM)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            policy.load_state_dict(state_dict)
            policy.eval()

            self._policy_cache[slot_id] = policy
            self.get_logger().info(f"[Bridge] Slot {slot_id} 모델 로드 완료: {model_path}")
            return policy

        except FileNotFoundError:
            self.get_logger().warn(
                f"[Bridge] 모델 파일 미존재: {model_path} (학습 완료 후 재시도)"
            )
            return None
        except Exception as e:
            self.get_logger().error(f"[Bridge] 모델 로드 실패: {e}")
            return None

    def _run_act_inference(self, policy, task: dict, phase: str):
        """
        ACT 정책으로 추론을 실행한다.
        Isaac Sim 루프와 통합 시 eval_bc.py 내부에서 직접 호출되므로,
        여기서는 독립 실행 테스트용으로만 사용.
        """
        if policy is None:
            self.get_logger().warn(f"[Bridge] 정책 없음 — {phase} 스킵")
            return

        import torch
        import numpy as np

        # 더미 상태 벡터 (실제 Isaac Sim 연동 시 eval_bc.py 가 주입)
        dummy_state = torch.zeros(1, 153, dtype=torch.float32)

        with torch.no_grad():
            action_chunk = policy(dummy_state)  # (1, chunk_size, action_dim)

        actions = action_chunk.squeeze(0).numpy()
        self.get_logger().info(
            f"[Bridge] ACT 추론 완료 | phase={phase} | "
            f"action_chunk shape={actions.shape} | "
            f"첫 번째 액션 norm={float(np.linalg.norm(actions[0])):.4f}"
        )

    # ─────────────────────────────────────────────────────────
    # 4. ReportInboundProgress 서비스 호출
    # ─────────────────────────────────────────────────────────

    def _report_inbound_progress(self, task: dict) -> bool:
        """
        인터페이스 명세 §2.③ ReportInboundProgress.srv 에 따라
        메인 컨트롤러에 적재 진척도를 동기 보고한다.
        """
        if not self.report_client.service_is_ready():
            self.get_logger().warn(
                "[Bridge] ReportInboundProgress 서비스 미준비 — 건너뜀"
            )
            return False

        req = ReportInboundProgress.Request()
        req.workstation_id    = task.get("workstation_id", self.workstation_id)
        req.robot_id          = task.get("robot_id", self.robot_id)
        req.filled_slots_count = task.get("slot_id", 1)   # 채워진 슬롯 번호
        req.package_id        = task.get("package_id", "")
        req.workstation_qr_id = task.get("workstation_qr_id", self.workstation_qr_id)
        req.package_qr_id     = task.get("qr_id", "")

        future = self.report_client.call_async(req)
        # 최대 5초 대기 (동기적으로 결과 확인)
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)

        if future.done():
            try:
                res = future.result()
                if res.success:
                    self.get_logger().info(
                        f"[Bridge] ✅ ReportInboundProgress 성공 | "
                        f"QR={req.package_qr_id} Slot={req.filled_slots_count}"
                    )
                    return True
                else:
                    self.get_logger().warn("[Bridge] ⚠️ ReportInboundProgress 실패 응답")
                    return False
            except Exception as e:
                self.get_logger().error(f"[Bridge] 서비스 응답 오류: {e}")
                return False
        else:
            self.get_logger().warn("[Bridge] ReportInboundProgress 타임아웃 (5초)")
            return False

    # ─────────────────────────────────────────────────────────
    # 5. Fleet 상태 토픽 발행 (1Hz)
    # ─────────────────────────────────────────────────────────

    def _publish_fleet_states(self):
        self._publish_workstation_states()
        self._publish_amr_states()
        self._publish_sh5_status()

    def _publish_workstation_states(self):
        """
        인터페이스 명세 §4.② /fleet/workstation_states
        """
        with self._lock:
            filled = list(self._filled_slots)
            ws_status = "FULL" if len(filled) >= SLOT_COUNT else "WAITING"

        payload = {
            "workstations": [
                {
                    "workstation_id":    self.workstation_id,
                    "workstation_qr_id": self.workstation_qr_id,
                    "current_location":  self.current_qr_id,
                    "status":            ws_status,
                    "slot_count":        SLOT_COUNT,
                    "filled_slots":      filled,
                    "reserved_by":       self.robot_id,
                }
            ]
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.ws_states_pub.publish(msg)

    def _publish_amr_states(self):
        """
        인터페이스 명세 §4.① /fleet/amr_states
        """
        with self._lock:
            state     = self._robot_state
            cur_task  = self._current_task

        target_qr = ""
        if cur_task and state != "IDLE":
            target_qr = cur_task.get("workstation_qr_id", "")

        payload = {
            self.robot_id: {
                "state":                 state,
                "current_qr_id":         self.current_qr_id,
                "target_qr_id":          target_qr,
                "carrying_workstation_id": (
                    self.workstation_id if state in ("NAVIGATING", "PLACING") else None
                ),
                "battery":  100.0,    # TODO: 실제 배터리 토픽 연동
                "available": state == "IDLE",
            }
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.amr_states_pub.publish(msg)

    def _publish_sh5_status(self):
        """
        /sh5/status — SH5 전용 상태 요약 (다른 팀원이 쉽게 파싱할 수 있도록)
        """
        with self._lock:
            state    = self._robot_state
            cur_task = self._current_task
            filled   = list(self._filled_slots)

        payload = {
            "robot_id":        self.robot_id,
            "workstation_id":  self.workstation_id,
            "state":           state,
            "filled_slots":    filled,
            "current_slot":    cur_task.get("slot_id") if cur_task else None,
            "current_qr":      cur_task.get("qr_id") if cur_task else None,
            "model_loaded":    list(self._policy_cache.keys()),
            "timestamp":       time.time(),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.sh5_status_pub.publish(msg)

    # ─────────────────────────────────────────────────────────
    # 6. 태스크 이벤트 발행 (/fleet/task_events)
    # ─────────────────────────────────────────────────────────

    def _emit_task_event(self, task: dict, status: str):
        """
        인터페이스 명세 §4.④ /fleet/task_events
        이벤트 발생 시 즉시 발행 (Event-driven).
        """
        payload = {
            "schema_version":    "1.0",
            "timestamp":         time.time(),
            "task_id":           str(uuid.uuid4()),
            "type":              "INBOUND_PLACE",
            "priority":          80,
            "workstation_id":    task.get("workstation_id", self.workstation_id),
            "workstation_qr_id": task.get("workstation_qr_id", self.workstation_qr_id),
            "start_location":    self.current_qr_id,
            "target_location":   f"slot_{task.get('slot_id', 1)}",
            "status":            status,
            "assigned_amr":      self.robot_id,
            "package_qr_id":     task.get("qr_id", ""),
            "slot_id":           task.get("slot_id", 1),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.task_events_pub.publish(msg)

    # ─────────────────────────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────────────────────────

    def _set_state(self, state: str):
        with self._lock:
            self._robot_state = state
        self.get_logger().debug(f"[Bridge] 상태 변경: {state}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = MainControllerBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
