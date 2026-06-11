import select
import sys
import termios
import tty
import threading

class TerminalKeyboard:
    """
    터미널 키보드 입력기 (통합 버전)
    
    [이동 제어] - 모바일 텔레옵 통합
      W/S : 전진 / 후진
      A/D : 좌회전 / 우회전
      Q/E : 좌측켜르기 / 우측켜르기
      U/O : 리프트 올림 / 내림
    
    [녹화 제어] - 두 가지 방식 모두 지원
      R 또는 1 : 녹화 시작
      T 또는 2 : 녹화 저장 (성공)
      C 또는 3 : 녹화 취소
      B 또는 4 : 상자 랜덤 리스폰 + 로봇 초기화
    """
    LINEAR_SPEED = 0.4   # m/s
    ANGULAR_SPEED = 0.8  # rad/s
    LIFT_STEP = 0.05     # m per keypress
    LIFT_MIN = -0.5
    LIFT_MAX = 0.0
    HEAD_STEP = 0.05     # rad per keypress
    HEAD_PAN_MIN = -1.57
    HEAD_PAN_MAX = 1.57
    HEAD_TILT_MIN = -1.57
    HEAD_TILT_MAX = 1.57
    HEAD_PAN_DEFAULT = 0.44    # head_joint1 기본값: 우측 벨트 방향 (0.44rad)
    HEAD_TILT_DEFAULT = 0.0    # head_joint2 기본값

    def __init__(self):
        self.key_pressed = None
        self.running = True
        self.old_settings = termios.tcgetattr(sys.stdin)
        import atexit
        atexit.register(self.restore_terminal)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def restore_terminal(self):
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
        except Exception:
            pass

    def _read_loop(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    char = sys.stdin.read(1)
                    if char:
                        self.key_pressed = char.lower()
        finally:
            self.restore_terminal()

    def get_key_and_clear(self):
        k = self.key_pressed
        self.key_pressed = None
        return k
