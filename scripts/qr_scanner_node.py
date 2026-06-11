#!/usr/bin/env python3
"""
qr_scanner_node.py
─────────────────────────────────────────────────────────────────────────────
SH5 입고 라인 QR 코드 스캐너 노드

역할:
  1. 카메라 이미지에서 QR 코드를 인식하고 상자의 3D 위치를 계산한다.
  2. GetPackageRoute 서비스를 통해 관제탑(메인 컨트롤러)에서 배송 예정일(슬롯 ID)을 조회한다.
  3. 슬롯 결정 결과를 /sh5/task_assignment 토픽으로 발행하여
     main_controller_bridge.py 가 ACT 추론을 트리거하도록 한다.

토픽/서비스 정의 근거: /home/rokey/dev_ws/인터페이스 (1).ipynb  v2.2 규격
─────────────────────────────────────────────────────────────────────────────
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

import cv2
import numpy as np
import json
import time
from pyzbar.pyzbar import decode as pyzbar_decode   # pip install pyzbar

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
from cv_bridge import CvBridge

from cobot3_interfaces.srv import GetPackageRoute

# ─────────────────────────────────────────────────────────
# 슬롯 → 배송 예정일 매핑 (DB에서 받아온 route_destination 기준)
# 작업대 좌표 파일 (/home/rokey/dev_ws/작업대 공간 좌표) 참고
# ─────────────────────────────────────────────────────────
SLOT_ROUTE_MAP = {
    # route_destination(날짜 문자열) → slot_id
    # 라인 고정 규칙:
    #   sg2_in_01  →  Slot 1/2 (오늘, 위층)
    #   sg2_in_02  →  Slot 3/4 (내일, 아래층)
    # 실제 날짜는 런타임에 동적으로 결정되므로 아래는 예시 기본값
    # main_controller_bridge.py 가 GET /api/today_date 로 갱신
    "TODAY":    1,   # 오늘 물량 → 위층 우측 (Slot 1, 오른손)
    "TOMORROW": 3,   # 내일 물량 → 아래층 우측 (Slot 3, 오른손)
}

# 왼손을 사용하는 슬롯 (짝수 슬롯)
LEFT_HAND_SLOTS = {2, 4}

# 슬롯 물리 중심 좌표 (Isaac Sim 기준, 작업대 공간 좌표 파일에서 참고)
SLOT_CENTERS = {
    1: {"x": -0.240, "y": -1.255, "z": 1.560},   # 위층 좌측
    2: {"x":  0.240, "y": -1.255, "z": 1.560},   # 위층 우측
    3: {"x": -0.240, "y": -1.255, "z": 0.860},   # 아래층 좌측
    4: {"x":  0.240, "y": -1.255, "z": 0.860},   # 아래층 우측
}


class QRScannerNode(Node):
    """
    카메라 이미지에서 QR 코드를 감지하고,
    GetPackageRoute 서비스를 호출하여 슬롯을 결정한 뒤
    /sh5/task_assignment 에 발행하는 노드.
    """

    def __init__(self):
        super().__init__("qr_scanner_node")

        # ── 파라미터 ──────────────────────────────────────────
        self.declare_parameter("robot_id",        "sh5_in_01")
        self.declare_parameter("workstation_id",  "WS01")
        self.declare_parameter("workstation_qr_id", "WORKSTATION_WS01")
        self.declare_parameter("camera_topic",    "/sh5/head_camera/image_raw")
        self.declare_parameter("depth_topic",     "/sh5/head_camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/sh5/head_camera/camera_info")
        self.declare_parameter("route_service",   "/get_package_route")
        self.declare_parameter("scan_cooldown_sec", 2.0)  # 동일 QR 재처리 방지
        self.declare_parameter("today_date",      "")     # 빈 문자열이면 동적 추론

        self.robot_id          = self.get_parameter("robot_id").value
        self.workstation_id    = self.get_parameter("workstation_id").value
        self.workstation_qr_id = self.get_parameter("workstation_qr_id").value
        self.scan_cooldown     = self.get_parameter("scan_cooldown_sec").value
        self.today_date        = self.get_parameter("today_date").value

        self.bridge = CvBridge()
        self.cb_group = ReentrantCallbackGroup()

        # 카메라 내부 파라미터 (K matrix)
        self.camera_K: np.ndarray | None = None
        self.latest_depth: np.ndarray | None = None

        # 쿨다운 추적 {qr_id: last_processed_time}
        self._recently_processed: dict[str, float] = {}

        # ── 구독 ──────────────────────────────────────────────
        self.create_subscription(
            Image,
            self.get_parameter("camera_topic").value,
            self._image_callback,
            10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Image,
            self.get_parameter("depth_topic").value,
            self._depth_callback,
            10,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self._camera_info_callback,
            10,
        )

        # ── 서비스 클라이언트 ──────────────────────────────────
        self.route_client = self.create_client(
            GetPackageRoute,
            self.get_parameter("route_service").value,
            callback_group=self.cb_group,
        )

        # ── 발행 ──────────────────────────────────────────────
        # /sh5/task_assignment  — JSON std_msgs/String
        self.task_pub = self.create_publisher(String, "/sh5/task_assignment", 10)

        # /sh5/qr_detection_viz — 시각화용 이미지
        self.viz_pub = self.create_publisher(Image, "/sh5/qr_detection_viz", 10)

        # /fleet/package_states — 패키지 상태 모니터링 (1Hz)
        self.pkg_states_pub = self.create_publisher(String, "/fleet/package_states", 10)
        self._active_packages: list[dict] = []
        self.create_timer(1.0, self._publish_package_states)

        self.get_logger().info(
            f"[QRScanner] 노드 시작 | robot_id={self.robot_id} "
            f"workstation={self.workstation_id}"
        )

    # ── 콜백: CameraInfo ──────────────────────────────────────
    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_K is None:
            self.camera_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info("[QRScanner] CameraInfo 수신 완료")

    # ── 콜백: 깊이 이미지 ─────────────────────────────────────
    def _depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"[QRScanner] 깊이 이미지 변환 실패: {e}")

    # ── 콜백: RGB 이미지 (QR 스캔 메인 루프) ─────────────────
    def _image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"[QRScanner] RGB 이미지 변환 실패: {e}")
            return

        # ── QR 코드 감지 ────────────────────────────────────
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        qr_codes = pyzbar_decode(gray)

        for qr in qr_codes:
            qr_id = qr.data.decode("utf-8").strip()

            # 쿨다운 체크 (동일 QR 중복 처리 방지)
            now = time.monotonic()
            if now - self._recently_processed.get(qr_id, 0.0) < self.scan_cooldown:
                continue
            self._recently_processed[qr_id] = now

            # QR 바운딩 박스 중심 픽셀 좌표
            poly = np.array([[p.x, p.y] for p in qr.polygon], dtype=np.float32)
            cx_px = float(np.mean(poly[:, 0]))
            cy_px = float(np.mean(poly[:, 1]))

            # 3D 위치 추정 (깊이 카메라 이용)
            box_3d = self._estimate_3d_position(cx_px, cy_px)

            # 시각화 오버레이
            cv2.polylines(cv_image, [poly.astype(np.int32)], True, (0, 255, 0), 2)
            cv2.putText(
                cv_image, qr_id,
                (int(cx_px), int(cy_px) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            self.get_logger().info(
                f"[QRScanner] QR 감지: {qr_id}  |  3D 위치: {box_3d}"
            )

            # 비동기로 서비스 호출 + 태스크 발행
            self._trigger_route_query(qr_id, box_3d)

        # 시각화 이미지 발행
        viz_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.viz_pub.publish(viz_msg)

    # ── 3D 위치 추정 ──────────────────────────────────────────
    def _estimate_3d_position(self, cx_px: float, cy_px: float) -> dict:
        """
        깊이 카메라 + 핀홀 모델로 QR 코드 중심의 3D 좌표를 계산한다.
        깊이 카메라가 없으면 기본 더미 좌표를 반환한다.
        """
        depth_m = 0.7   # 기본 거리 (깊이 카메라 미사용 시)

        if self.latest_depth is not None:
            ix, iy = int(cx_px), int(cy_px)
            h, w = self.latest_depth.shape[:2]
            if 0 <= iy < h and 0 <= ix < w:
                raw_depth = float(self.latest_depth[iy, ix])
                if raw_depth > 0.01:
                    depth_m = raw_depth * 0.001  # mm → m (RealSense 기준)

        if self.camera_K is not None:
            fx = float(self.camera_K[0, 0])
            fy = float(self.camera_K[1, 1])
            ppx = float(self.camera_K[0, 2])
            ppy = float(self.camera_K[1, 2])
            x3d = (cx_px - ppx) / fx * depth_m
            y3d = (cy_px - ppy) / fy * depth_m
            z3d = depth_m
        else:
            # CameraInfo 미수신 시 더미
            x3d, y3d, z3d = 0.7, 0.0, 1.0

        return {"x": round(x3d, 4), "y": round(y3d, 4), "z": round(z3d, 4)}

    # ── GetPackageRoute 서비스 호출 ────────────────────────────
    def _trigger_route_query(self, qr_id: str, box_3d: dict):
        """
        관제탑(메인 컨트롤러)의 GetPackageRoute 서비스를 비동기 호출하여
        배송 예정일(route_destination)과 슬롯 번호를 결정한다.
        """
        if not self.route_client.service_is_ready():
            self.get_logger().warn(
                "[QRScanner] GetPackageRoute 서비스 미준비. 로컬 매핑으로 대체."
            )
            self._fallback_slot_assignment(qr_id, box_3d)
            return

        req = GetPackageRoute.Request()
        req.package_id   = ""          # QR ID로 조회하므로 공란
        req.customer_name = ""
        req.qr_id        = qr_id

        future = self.route_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_route_response(f, qr_id, box_3d)
        )

    def _on_route_response(self, future, qr_id: str, box_3d: dict):
        """GetPackageRoute 응답 처리 → 슬롯 결정 → 태스크 발행"""
        try:
            res = future.result()
            route_dest = res.route_destination   # 예: "2026-06-09"
        except Exception as e:
            self.get_logger().error(f"[QRScanner] GetPackageRoute 호출 실패: {e}")
            self._fallback_slot_assignment(qr_id, box_3d)
            return

        slot_id = self._route_to_slot(route_dest)
        self._publish_task(qr_id, box_3d, slot_id, route_dest)

    # ── 슬롯 결정 로직 ────────────────────────────────────────
    def _route_to_slot(self, route_destination: str) -> int:
        """
        배송 예정일 문자열을 슬롯 ID(1~4)로 변환한다.
        라인 고정 규칙 (인터페이스 명세 §4.① 참고):
          오늘 날짜 → Slot 1(오른손) 또는 Slot 2(왼손) — 위층
          내일 날짜 → Slot 3(오른손) 또는 Slot 4(왼손) — 아래층
        현재는 짝/홀 라운드로빈으로 좌/우를 교대 배분한다.
        """
        today = self._get_today_str()
        tomorrow = self._get_tomorrow_str()

        # 위층(오늘): Slot 1 → 2 → 1 → 2 ...
        # 아래층(내일): Slot 3 → 4 → 3 → 4 ...
        if route_destination == today:
            base_slots = [1, 2]
        elif route_destination == tomorrow:
            base_slots = [3, 4]
        else:
            # 모레 이후나 미매핑 날짜 → 기본 Slot 1
            self.get_logger().warn(
                f"[QRScanner] route_destination='{route_destination}' 매핑 불가 → Slot 1 기본값"
            )
            return 1

        # 라운드로빈: active_packages 수로 좌/우 교대
        idx = len(self._active_packages) % 2
        return base_slots[idx]

    def _fallback_slot_assignment(self, qr_id: str, box_3d: dict):
        """서비스 미연결 시 로컬 매핑으로 슬롯 결정 (데모/테스트용)"""
        self.get_logger().warn("[QRScanner] 폴백: 로컬 슬롯 매핑 사용 (Slot 1)")
        self._publish_task(qr_id, box_3d, slot_id=1, route_dest="FALLBACK")

    # ── 태스크 발행 ───────────────────────────────────────────
    def _publish_task(self, qr_id: str, box_3d: dict, slot_id: int, route_dest: str):
        """
        /sh5/task_assignment 에 JSON 형식으로 태스크를 발행한다.
        main_controller_bridge.py 가 이를 구독하여 ACT 추론을 트리거한다.
        """
        slot_center = SLOT_CENTERS.get(slot_id, SLOT_CENTERS[1])
        use_left_hand = slot_id in LEFT_HAND_SLOTS
        model_path = (
            f"/home/rokey/dev_ws/models/slot{slot_id}_act_policy.pth"
        )

        payload = {
            "qr_id":            qr_id,
            "package_id":       qr_id,          # QR ID = Package ID 규칙
            "workstation_id":   self.workstation_id,
            "workstation_qr_id": self.workstation_qr_id,
            "robot_id":         self.robot_id,
            "slot_id":          slot_id,
            "route_destination": route_dest,
            "box_pose_camera": box_3d,           # 카메라 좌표계 3D 위치
            "slot_center_world": slot_center,    # 목표 슬롯 월드 좌표
            "use_left_hand":    use_left_hand,
            "model_path":       model_path,
            "timestamp":        time.time(),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.task_pub.publish(msg)

        # 활성 패키지 목록 갱신
        self._active_packages.append({
            "package_id":  qr_id,
            "qr_id":       qr_id,
            "customer_name": "",
            "route_zone":  route_dest,
            "status":      "ASSIGNED",
            "outbound_id": None,
            "workstation_id": self.workstation_id,
            "slot_number": slot_id,
        })

        self.get_logger().info(
            f"[QRScanner] ✅ 태스크 발행 | QR={qr_id} → Slot {slot_id} "
            f"({'왼손' if use_left_hand else '오른손'}) | model={model_path}"
        )

    # ── 1Hz 패키지 상태 발행 (/fleet/package_states) ─────────
    def _publish_package_states(self):
        payload = {"packages": self._active_packages}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pkg_states_pub.publish(msg)

    # ── 날짜 유틸 ─────────────────────────────────────────────
    def _get_today_str(self) -> str:
        if self.today_date:
            return self.today_date
        from datetime import date
        return date.today().isoformat()   # "YYYY-MM-DD"

    def _get_tomorrow_str(self) -> str:
        from datetime import date, timedelta
        if self.today_date:
            from datetime import datetime
            d = datetime.strptime(self.today_date, "%Y-%m-%d").date()
            return (d + timedelta(days=1)).isoformat()
        return (date.today() + timedelta(days=1)).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = QRScannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
