"""
bg2_socket_sender.py
====================
[BG2 PC - 송신측 클라이언트]

BG2 Isaac Sim에서 상자를 디스폰할 때 이 함수를 호출하면
SH5/AMR PC의 서버로 JSON 메시지를 전송함.

사용법 (BG2 PC의 Isaac Sim Script Editor):
  exec(open('/path/to/bg2_socket_sender.py', encoding='utf-8').read())
  
  # 이후 상자 디스폰 시점에:
  sender.send("PKG_20260612_001", "sg2_in_01")

또는 BG2 기존 스크립트에 함수만 임포트:
  from bg2_socket_sender import send_box_event
  send_box_event("PKG_001", "sg2_in_02")
"""

import socket
import json
import time

# ============================================================
# 설정 (SH5/AMR PC의 IP와 포트)
# ============================================================
SH5_PC_HOST = "192.168.10.XX"   # ← SH5/AMR PC의 실제 IP로 교체
SH5_PC_PORT = 9000               # sh5_socket_server.py의 SERVER_PORT와 동일
TIMEOUT_SEC  = 3.0               # 연결/응답 타임아웃


# ============================================================
# 단발성 전송 함수 (가장 간단한 사용법)
# ============================================================
def send_box_event(package_id: str, target_line: str) -> bool:
    """
    BG2에서 상자 디스폰 시 호출.
    SH5 PC에 TCP로 JSON 메시지를 전송하고 응답을 확인.

    Args:
        package_id:  패키지 고유 ID (예: "PKG_20260612_001")
        target_line: 목표 입고 라인 (예: "sg2_in_01", "sg2_in_02", "sg2_in_03")

    Returns:
        bool: 전송 성공 여부
    """
    payload = {
        "package_id":  package_id,
        "target_line": target_line,
        "timestamp":   time.time(),
    }
    message = json.dumps(payload) + "\n"   # 개행문자가 메시지 종료 신호

    try:
        with socket.create_connection((SH5_PC_HOST, SH5_PC_PORT), timeout=TIMEOUT_SEC) as sock:
            sock.sendall(message.encode("utf-8"))
            print(f"[BG2 Sender] 📤 전송: {payload}")

            # 응답 수신
            resp_data = sock.recv(1024).decode("utf-8").strip()
            resp = json.loads(resp_data)

            if resp.get("status") == "ok":
                print(f"[BG2 Sender] ✅ SH5 PC 수신 확인: {package_id} → {target_line}")
                return True
            else:
                print(f"[BG2 Sender] ⚠️ SH5 PC 오류 응답: {resp}")
                return False

    except ConnectionRefusedError:
        print(f"[BG2 Sender] ❌ 연결 거부 - SH5 PC({SH5_PC_HOST}:{SH5_PC_PORT}) 서버가 실행 중인지 확인")
        return False
    except socket.timeout:
        print(f"[BG2 Sender] ❌ 타임아웃 - SH5 PC 응답 없음")
        return False
    except Exception as e:
        print(f"[BG2 Sender] ❌ 전송 오류: {e}")
        return False


# ============================================================
# 연결 유지 클라이언트 (고빈도 전송 시 권장)
# ============================================================
class BoxEventSender:
    """
    TCP 연결을 유지하면서 반복 전송하는 클라이언트.
    상자 디스폰이 빈번할 때 매번 연결하는 오버헤드를 줄임.
    """
    def __init__(self, host=SH5_PC_HOST, port=SH5_PC_PORT):
        self.host = host
        self.port = port
        self._sock = None
        print(f"[BG2 Sender] 초기화 | 목표: {host}:{port}")

    def connect(self) -> bool:
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=TIMEOUT_SEC)
            print(f"[BG2 Sender] 🔗 SH5 PC 연결 성공: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[BG2 Sender] ❌ 연결 실패: {e}")
            self._sock = None
            return False

    def send(self, package_id: str, target_line: str) -> bool:
        """
        BG2에서 상자 디스폰 시 이 메서드를 호출.

        사용 예:
            sender.send("PKG_001", "sg2_in_01")   # 1번 라인으로 전송
            sender.send("PKG_002", "sg2_in_02")   # 2번 라인으로 전송
        """
        if self._sock is None:
            print("[BG2 Sender] 연결 없음 → 재연결 시도")
            if not self.connect():
                return False

        payload = {
            "package_id":  package_id,
            "target_line": target_line,
            "timestamp":   time.time(),
        }
        message = json.dumps(payload) + "\n"

        try:
            self._sock.sendall(message.encode("utf-8"))
            print(f"[BG2 Sender] 📤 {package_id} → {target_line}")

            resp_data = self._sock.recv(1024).decode("utf-8").strip()
            resp = json.loads(resp_data)

            if resp.get("status") == "ok":
                print(f"[BG2 Sender] ✅ 확인됨")
                return True
            else:
                print(f"[BG2 Sender] ⚠️ 응답 오류: {resp}")
                return False

        except (BrokenPipeError, ConnectionResetError):
            print("[BG2 Sender] 연결 끊김 → 재연결")
            self._sock = None
            return self.send(package_id, target_line)   # 1회 재시도
        except Exception as e:
            print(f"[BG2 Sender] ❌ 전송 오류: {e}")
            return False

    def disconnect(self):
        if self._sock:
            self._sock.close()
            self._sock = None
            print("[BG2 Sender] 🔌 연결 종료")


# ============================================================
# 테스트 (이 파일을 직접 실행 시)
# ============================================================
if __name__ == "__main__":
    import sys

    print("=== BG2 소켓 송신 테스트 ===")
    print(f"대상: {SH5_PC_HOST}:{SH5_PC_PORT}")
    print()

    # 연결 유지 클라이언트 테스트
    sender = BoxEventSender()
    if not sender.connect():
        print("연결 실패 - SH5_PC_HOST를 실제 IP로 수정하세요")
        sys.exit(1)

    # 테스트 시나리오: 3개 라인에 순서대로 상자 전송
    test_cases = [
        ("PKG_TEST_001", "sg2_in_01"),
        ("PKG_TEST_002", "sg2_in_02"),
        ("PKG_TEST_003", "sg2_in_03"),
        ("PKG_TEST_004", "sg2_in_01"),
    ]

    for pkg_id, line_id in test_cases:
        print(f"\n--- 전송: {pkg_id} → {line_id} ---")
        sender.send(pkg_id, line_id)
        time.sleep(2.0)

    sender.disconnect()
    print("\n테스트 완료")

# ============================================================
# Isaac Sim Script Editor에서 실행 시
# ============================================================
# exec() 방식으로 실행하면 아래 객체가 자동 생성됨
# sender.send("PKG_001", "sg2_in_01") 으로 바로 사용 가능
else:
    print("\n[BG2 Sender] ✅ 로드 완료")
    print(f"[BG2 Sender] 대상 SH5 PC: {SH5_PC_HOST}:{SH5_PC_PORT}")
    print("[BG2 Sender] 사용법:")
    print("   send_box_event('PKG_001', 'sg2_in_01')   # 단발 전송")
    print("   sender = BoxEventSender()")
    print("   sender.connect()")
    print("   sender.send('PKG_001', 'sg2_in_01')      # 연결 유지 전송")

    # BG2 Script Editor에서 바로 쓸 수 있도록 기본 sender 생성
    sender = BoxEventSender()
