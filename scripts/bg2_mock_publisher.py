"""
bg2_mock_publisher.py
=====================
BG2가 아직 구현되지 않았을 때, /sim/sg2_spawn_trigger 토픽을
임의로 발행하여 sh5_final.py를 테스트하는 Mock 발행기.

사용법 (터미널):
  source /home/rokey/dev_ws/cobot3_ws_ref/install/setup.bash
  export ROS_DOMAIN_ID=119
  python3 /home/rokey/dev_ws/coupang_ws/scripts/bg2_mock_publisher.py

옵션:
  --mode auto    : 자동 순환 발행 (기본, 5초 간격)
  --mode manual  : 수동 입력 발행 (키보드 입력)
  --mode once    : 1회만 발행 후 종료
  --interval 3   : 발행 간격 (초, 기본 5)
  --line sg2_in_01  : 특정 라인만 고정 발행
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json, time, argparse, sys, itertools, random

# ── 시나리오 설정 ──────────────────────────────────
# 발행할 패키지 ID 목록 (실제처럼 보이도록 날짜 포함)
PACKAGE_IDS = [
    "PKG_20260612_001", "PKG_20260612_002", "PKG_20260612_003",
    "PKG_20260612_004", "PKG_20260612_005", "PKG_20260612_006",
    "PKG_20260613_001", "PKG_20260613_002", "PKG_20260613_003",
    "PKG_20260614_001", "PKG_20260614_002", "PKG_20260614_003",
]

# 라인 순환 순서 (날짜 → 라인 매핑)
LINE_CYCLE = [
    "sg2_in_01",   # 오늘 날짜 물량
    "sg2_in_02",   # 내일 날짜 물량
    "sg2_in_03",   # 모레 날짜 물량
]


class BG2MockPublisher(Node):
    def __init__(self, mode: str, interval: float, fixed_line: str | None):
        super().__init__("bg2_mock_publisher")

        self.pub = self.create_publisher(
            String, "/sim/sg2_spawn_trigger", 10
        )
        self.mode       = mode
        self.interval   = interval
        self.fixed_line = fixed_line

        self._pkg_pool  = itertools.cycle(PACKAGE_IDS)
        self._line_pool = itertools.cycle(LINE_CYCLE)
        self._count     = 0

        self.get_logger().info("=" * 50)
        self.get_logger().info("  BG2 Mock Publisher 가동")
        self.get_logger().info(f"  모드: {mode} | 간격: {interval}s")
        self.get_logger().info(f"  토픽: /sim/sg2_spawn_trigger")
        self.get_logger().info("=" * 50)

    def publish_one(self, package_id: str | None = None,
                    target_line: str | None = None):
        pkg  = package_id  or next(self._pkg_pool)
        line = target_line or self.fixed_line or next(self._line_pool)

        payload = {
            "package_id": pkg,
            "target_line": line,
            "timestamp": time.time(),
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)
        self._count += 1

        self.get_logger().info(
            f"[발행 #{self._count}] {pkg} → {line}"
        )
        return payload

    def run_auto(self):
        """자동 순환 발행 (interval 초 간격)"""
        self.get_logger().info(
            f"[Auto] {self.interval}초 간격으로 자동 발행 시작. Ctrl+C로 종료."
        )
        while True:
            self.publish_one()
            time.sleep(self.interval)

    def run_manual(self):
        """수동 입력 발행"""
        self.get_logger().info("[Manual] 엔터를 누를 때마다 발행. q+엔터=종료.")
        print("\n  명령어:")
        print("    [엔터]          → 다음 패키지 자동 발행")
        print("    1 / 2 / 3       → sg2_in_01 / 02 / 03 지정 발행")
        print("    PKG_xxx         → 특정 package_id 지정 발행")
        print("    q               → 종료\n")

        line_map = {"1": "sg2_in_01", "2": "sg2_in_02", "3": "sg2_in_03"}

        while True:
            try:
                inp = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if inp.lower() == "q":
                print("종료.")
                break
            elif inp in line_map:
                self.publish_one(target_line=line_map[inp])
            elif inp.startswith("PKG_"):
                self.publish_one(package_id=inp)
            else:
                # 엔터 or 기타 → 자동 발행
                self.publish_one()

    def run_once(self):
        """1회 발행 후 종료"""
        payload = self.publish_one()
        self.get_logger().info(f"[Once] 1회 발행 완료: {payload}")


def main():
    parser = argparse.ArgumentParser(description="BG2 Mock Publisher")
    parser.add_argument("--mode",     default="auto",
                        choices=["auto", "manual", "once"],
                        help="발행 모드 (기본: auto)")
    parser.add_argument("--interval", default=5.0, type=float,
                        help="자동 발행 간격 초 (기본: 5)")
    parser.add_argument("--line",     default=None,
                        choices=["sg2_in_01", "sg2_in_02", "sg2_in_03"],
                        help="특정 라인 고정 발행 (기본: 순환)")
    args = parser.parse_args()

    rclpy.init()
    node = BG2MockPublisher(
        mode=args.mode,
        interval=args.interval,
        fixed_line=args.line,
    )

    import threading
    spin_t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_t.start()

    try:
        if args.mode == "auto":
            node.run_auto()
        elif args.mode == "manual":
            node.run_manual()
        elif args.mode == "once":
            node.run_once()
    except KeyboardInterrupt:
        print("\n[종료] Ctrl+C 감지")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
