"""
sh5_qr_scanner.py
=================
[Plan B] Top-View 카메라로 QR 코드를 실제 인식하고 상자의 월드 XY 좌표를 반환.

Plan A (현재):
  BG2 소켓/ROS2 신호에서 package_id + 고정 위치 → QR 스캔 불필요
  
Plan B (이 파일):
  Isaac Sim Top-View 카메라 이미지 → OpenCV QR 디코딩
  → 픽셀 좌표 → 월드 XY 좌표 변환 → 로봇이 정확히 그 위치에서 집음

통합 방법 (sh5_integrated.py 또는 sh5_spawn_controller.py):
  from sh5_qr_scanner import SH5QRScanner
  
  scanner = SH5QRScanner(camera_prim_path="/World/TopCamera_01")
  result = scanner.scan()
  if result:
      qr_id, world_x, world_y = result
      print(f"QR: {qr_id} | 상자 위치: ({world_x:.3f}, {world_y:.3f})")
"""

import time
import numpy as np

# ============================================================
# Isaac Sim 연결
# ============================================================
ISAAC_AVAILABLE = False
try:
    import omni.usd
    from pxr import UsdGeom, Gf
    ISAAC_AVAILABLE = True
except ImportError:
    pass

# OpenCV
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
    print("[QRScanner] ✅ OpenCV 로드 완료:", cv2.__version__)
except ImportError:
    print("[QRScanner] ⚠️ OpenCV 없음 - pip install opencv-python")

# Isaac Sim 카메라 센서
SENSOR_AVAILABLE = False
try:
    import omni.replicator.core as rep
    import omni.syntheticdata as syn
    SENSOR_AVAILABLE = True
    print("[QRScanner] ✅ Isaac Sim Replicator 로드 완료")
except ImportError:
    try:
        from omni.isaac.sensor import Camera
        SENSOR_AVAILABLE = True
        print("[QRScanner] ✅ Isaac Sim Camera 센서 로드 완료")
    except ImportError:
        print("[QRScanner] ⚠️ Isaac Sim 카메라 모듈 없음")


# ============================================================
# 카메라 설정 (라인별 Top-View 카메라)
# ============================================================
# Isaac Sim Stage에서 카메라 Prim 경로 확인 후 수정
# final_coupan.usd 기준으로 추정 경로 작성
CAMERA_PRIMS = {
    "sg2_in_01": "/World/TopCamera_Line01",   # ← 실제 Prim 경로로 교체
    "sg2_in_02": "/World/TopCamera_Line02",
    "sg2_in_03": "/World/TopCamera_Line03",
}

# 카메라 해상도 (Isaac Sim 카메라 설정과 일치시켜야 함)
CAM_WIDTH  = 640
CAM_HEIGHT = 480

# Top-View 카메라 높이 (월드 좌표 → 픽셀 역투영에 사용)
# 컨베이어 벨트 위에서 카메라까지 수직 거리 (m)
CAMERA_HEIGHT_M = 2.5   # ← 실제 카메라 설치 높이로 교체

# 카메라 FOV (도) → 실제 카메라 설정에서 확인
CAMERA_FOV_DEG = 60.0


# ============================================================
# 카메라 이미지 획득
# ============================================================
def get_camera_image_rgb(camera_prim_path: str) -> np.ndarray | None:
    """
    Isaac Sim Stage의 카메라 Prim에서 RGB 이미지를 numpy 배열로 획득.
    
    Returns:
        np.ndarray shape (H, W, 3) uint8, 또는 None (실패 시)
    """
    if not ISAAC_AVAILABLE:
        print("[QRScanner] Isaac Sim 없음 - 이미지 획득 불가")
        return None

    try:
        # 방법 1: omni.isaac.sensor.Camera 사용 (Isaac Sim 4.x)
        from omni.isaac.sensor import Camera
        cam = Camera(
            prim_path=camera_prim_path,
            resolution=(CAM_WIDTH, CAM_HEIGHT),
        )
        cam.initialize()

        # 1프레임 렌더링 대기
        import omni.kit.app
        app = omni.kit.app.get_app()
        for _ in range(3):
            app.update()

        rgba = cam.get_rgba()   # shape: (H, W, 4)
        if rgba is None:
            print(f"[QRScanner] ⚠️ 카메라 이미지 없음: {camera_prim_path}")
            return None

        rgb = rgba[:, :, :3].astype(np.uint8)
        print(f"[QRScanner] 📷 이미지 획득: {camera_prim_path} | shape={rgb.shape}")
        return rgb

    except Exception as e:
        print(f"[QRScanner] 카메라 이미지 획득 오류: {e}")
        return None


# ============================================================
# QR 코드 디코딩
# ============================================================
def decode_qr_from_image(image_rgb: np.ndarray) -> list[tuple[str, tuple]]:
    """
    RGB 이미지에서 QR 코드를 모두 검출하고 디코딩.
    
    Returns:
        list of (qr_data_str, bbox_center_pixel (cx, cy))
        예: [("QR_20260612_001", (320, 240)), ...]
    """
    if not CV2_AVAILABLE:
        print("[QRScanner] OpenCV 없음")
        return []

    # BGR 변환 (OpenCV 기본 포맷)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    results = []

    # 방법 1: cv2.QRCodeDetector (기본, 빠름)
    detector = cv2.QRCodeDetector()
    
    # 다중 QR 코드 검출 시도
    try:
        retval, decoded_list, points_list, _ = detector.detectAndDecodeMulti(bgr)
        if retval and decoded_list:
            for qr_text, pts in zip(decoded_list, points_list):
                if qr_text:
                    pts_arr = pts.reshape(-1, 2)
                    cx = int(np.mean(pts_arr[:, 0]))
                    cy = int(np.mean(pts_arr[:, 1]))
                    results.append((qr_text, (cx, cy)))
                    print(f"  [QRScanner] QR 검출: '{qr_text}' @ 픽셀({cx}, {cy})")
    except Exception:
        # 구버전 OpenCV: 단일 QR 검출
        try:
            qr_text, pts, _ = detector.detectAndDecode(bgr)
            if qr_text and pts is not None:
                pts_arr = pts.reshape(-1, 2)
                cx = int(np.mean(pts_arr[:, 0]))
                cy = int(np.mean(pts_arr[:, 1]))
                results.append((qr_text, (cx, cy)))
                print(f"  [QRScanner] QR 검출 (단일): '{qr_text}' @ 픽셀({cx}, {cy})")
        except Exception as e2:
            print(f"  [QRScanner] QR 검출 오류: {e2}")

    if not results:
        # 방법 2: 이미지 전처리 후 재시도 (대비 향상)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        try:
            qr_text, pts, _ = detector.detectAndDecode(enhanced)
            if qr_text and pts is not None:
                pts_arr = pts.reshape(-1, 2)
                cx = int(np.mean(pts_arr[:, 0]))
                cy = int(np.mean(pts_arr[:, 1]))
                results.append((qr_text, (cx, cy)))
                print(f"  [QRScanner] QR 검출 (전처리 후): '{qr_text}' @ 픽셀({cx}, {cy})")
        except Exception:
            pass

    if not results:
        print("  [QRScanner] ⚠️ QR 코드 미검출")

    return results


# ============================================================
# 픽셀 좌표 → 월드 XY 좌표 변환
# ============================================================
def pixel_to_world_xy(
    pixel_x: int,
    pixel_y: int,
    camera_prim_path: str,
    box_z: float = 0.83,
) -> tuple[float, float]:
    """
    Top-View 카메라의 픽셀 좌표를 월드 XY 좌표로 변환.
    
    카메라가 Z축 아래를 바라보는 Top-View 설정 기준.
    
    Args:
        pixel_x, pixel_y: 이미지 상의 픽셀 좌표
        camera_prim_path: 카메라 Prim 경로 (월드 위치 읽기용)
        box_z: 상자 높이 (컨베이어 표면, m)
    
    Returns:
        (world_x, world_y) in meters
    """
    if not ISAAC_AVAILABLE:
        # 폴백: 픽셀 중심 기준 단순 비율 계산 (테스트용)
        cam_world_x = 9.0   # 컨베이어 끝 X
        cam_world_y = 0.0
        scale = CAMERA_HEIGHT_M * np.tan(np.radians(CAMERA_FOV_DEG / 2)) * 2
        world_x = cam_world_x + (pixel_x / CAM_WIDTH  - 0.5) * scale
        world_y = cam_world_y + (0.5 - pixel_y / CAM_HEIGHT) * scale
        return world_x, world_y

    try:
        # 카메라 월드 위치 읽기
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(camera_prim_path)

        if not cam_prim.IsValid():
            print(f"[QRScanner] ⚠️ 카메라 Prim 없음: {camera_prim_path}")
            return _fallback_pixel_to_world(pixel_x, pixel_y)

        # 카메라 월드 변환 행렬에서 위치 추출
        xform_cache = UsdGeom.XformCache()
        world_transform = xform_cache.GetLocalToWorldTransform(cam_prim)
        cam_pos = world_transform.ExtractTranslation()
        cam_x, cam_y, cam_z = cam_pos[0], cam_pos[1], cam_pos[2]

        # Top-View 핀홀 역투영 (카메라 정면이 -Z 방향)
        fov_rad = np.radians(CAMERA_FOV_DEG)
        f_x = (CAM_WIDTH  / 2) / np.tan(fov_rad / 2)
        f_y = (CAM_HEIGHT / 2) / np.tan(fov_rad / 2)

        # 픽셀 → 정규화 좌표
        norm_x = (pixel_x - CAM_WIDTH  / 2) / f_x
        norm_y = (pixel_y - CAM_HEIGHT / 2) / f_y

        # 카메라에서 상자까지 거리 (Z 방향)
        depth = cam_z - box_z   # 카메라 높이 - 상자 높이

        # 월드 좌표 (Top-View 기준: 픽셀X → 월드X, 픽셀Y → 월드Y 반전)
        world_x = cam_x + norm_x * depth
        world_y = cam_y - norm_y * depth   # Y축 반전 (이미지 좌표계)

        print(f"  [QRScanner] 픽셀({pixel_x},{pixel_y}) → 월드({world_x:.3f},{world_y:.3f})")
        return world_x, world_y

    except Exception as e:
        print(f"[QRScanner] 좌표 변환 오류: {e}")
        return _fallback_pixel_to_world(pixel_x, pixel_y)


def _fallback_pixel_to_world(px: int, py: int) -> tuple[float, float]:
    """카메라 정보 없을 때 단순 비율 계산 폴백"""
    scale = CAMERA_HEIGHT_M * np.tan(np.radians(CAMERA_FOV_DEG / 2)) * 2
    world_x = 9.0 + (px / CAM_WIDTH  - 0.5) * scale
    world_y = 0.0 + (0.5 - py / CAM_HEIGHT) * scale
    return world_x, world_y


# ============================================================
# 메인 스캐너 클래스
# ============================================================
class SH5QRScanner:
    """
    Top-View 카메라 기반 QR 스캐너.
    sh5_integrated.py의 _mock_scan_qr() 대체용.
    
    사용법:
        scanner = SH5QRScanner(line_id="sg2_in_01")
        result = scanner.scan()
        if result:
            qr_id, world_x, world_y = result
    """

    def __init__(self, line_id: str, max_retries: int = 3, retry_interval: float = 0.5):
        self.line_id = line_id
        self.camera_prim = CAMERA_PRIMS.get(line_id, "/World/TopCamera_Line01")
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        print(f"[QRScanner] 초기화: {line_id} | 카메라: {self.camera_prim}")

    def scan(self) -> tuple[str, float, float] | None:
        """
        카메라로 QR 스캔 → (qr_id, world_x, world_y) 반환.
        실패 시 None 반환 (호출자가 Mock 폴백 처리).
        
        Returns:
            (qr_id, world_x, world_y) 또는 None
        """
        for attempt in range(self.max_retries):
            print(f"\n[QRScanner] 스캔 시도 {attempt+1}/{self.max_retries} | {self.line_id}")

            # 1. 이미지 획득
            image = get_camera_image_rgb(self.camera_prim)
            if image is None:
                print(f"  [QRScanner] 이미지 없음 - {self.retry_interval}초 후 재시도")
                time.sleep(self.retry_interval)
                continue

            # 2. QR 디코딩
            qr_results = decode_qr_from_image(image)
            if not qr_results:
                time.sleep(self.retry_interval)
                continue

            # 3. 첫 번째 QR 사용 (여러 개 감지 시 가장 선명한 것)
            qr_text, (px, py) = qr_results[0]

            # 4. 픽셀 → 월드 좌표
            world_x, world_y = pixel_to_world_xy(
                pixel_x=px,
                pixel_y=py,
                camera_prim_path=self.camera_prim,
            )

            print(f"[QRScanner] ✅ 스캔 완료!")
            print(f"  QR ID: {qr_text}")
            print(f"  상자 월드 위치: ({world_x:.3f}, {world_y:.3f})")
            return qr_text, world_x, world_y

        print(f"[QRScanner] ❌ {self.max_retries}회 시도 후 실패 → Mock 폴백 권장")
        return None

    def debug_capture(self, save_path: str = "/tmp/qr_debug.png") -> bool:
        """
        디버그용: 카메라 이미지를 파일로 저장.
        QR이 카메라에 잡히는지 확인할 때 사용.
        
        Isaac Sim Script Editor에서:
            scanner = SH5QRScanner("sg2_in_01")
            scanner.debug_capture("/tmp/cam_check.png")
        """
        image = get_camera_image_rgb(self.camera_prim)
        if image is None:
            print("[QRScanner] 이미지 없음")
            return False

        if CV2_AVAILABLE:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr)
            print(f"[QRScanner] 📸 이미지 저장: {save_path}")

            # QR 위치 시각화
            qr_results = decode_qr_from_image(image)
            if qr_results:
                for qr_text, (cx, cy) in qr_results:
                    cv2.circle(bgr, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.putText(bgr, qr_text[:12], (cx-50, cy-15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                vis_path = save_path.replace(".png", "_annotated.png")
                cv2.imwrite(vis_path, bgr)
                print(f"[QRScanner] 📸 어노테이션 저장: {vis_path}")
            return True
        return False


# ============================================================
# sh5_integrated.py 교체용 래퍼 함수
# ============================================================
def scan_qr_real(line_id: str, box_spawn_pos: tuple) -> tuple[str, tuple]:
    """
    sh5_integrated.py의 _mock_scan_qr() 교체 드롭인 함수.
    
    실패 시 Mock 데이터로 자동 폴백.
    
    Returns:
        (qr_id, (box_x, box_y))  ← _mock_scan_qr()과 동일한 반환 형식
    """
    scanner = SH5QRScanner(line_id=line_id)
    result = scanner.scan()

    if result:
        qr_id, world_x, world_y = result
        return qr_id, (world_x, world_y)

    # 폴백: Mock 데이터
    import random
    print(f"[QRScanner] ⚠️ 실제 스캔 실패 → Mock 데이터로 폴백")
    qr_id = f"QR_MOCK_{random.randint(1000, 9999)}"
    box_x = box_spawn_pos[0] + random.uniform(-0.03, 0.03)
    box_y = box_spawn_pos[1] + random.uniform(-0.03, 0.03)
    return qr_id, (box_x, box_y)


# ============================================================
# sh5_integrated.py 통합 패치 함수
# ============================================================
def patch_sh5_integrated_qr(controller):
    """
    이미 실행 중인 sh5_integrated.py 컨트롤러의 QR 스캔을
    실제 카메라 스캔으로 교체하는 패치 함수.
    
    사용법 (Script Editor):
        # 먼저 sh5_integrated.py exec() 실행 후
        exec(open('.../sh5_qr_scanner.py', encoding='utf-8').read())
        patch_sh5_integrated_qr(controller)
    """
    import types

    for unit in controller.units if hasattr(controller, 'units') else []:
        line_id = getattr(unit, 'robot_id', getattr(unit, 'line_id', 'sg2_in_01'))

        def _real_scan_qr(self, _line_id=line_id):
            return scan_qr_real(_line_id, self.box_spawn_pos)

        unit._mock_scan_qr = types.MethodType(_real_scan_qr, unit)
        print(f"[QRScanner] ✅ {line_id} QR 스캔 패치 완료")

    print("[QRScanner] 🔄 sh5_integrated.py QR 스캔 → 실제 카메라로 교체됨")


# ============================================================
# 카메라 Prim 경로 자동 탐색 유틸
# ============================================================
def find_camera_prims() -> list[str]:
    """
    Stage에서 카메라 Prim을 모두 탐색하여 경로 목록 반환.
    실행 후 콘솔 출력으로 실제 카메라 경로를 확인할 수 있음.
    
    사용법 (Script Editor):
        exec(open('.../sh5_qr_scanner.py', encoding='utf-8').read())
        find_camera_prims()
    """
    if not ISAAC_AVAILABLE:
        print("[QRScanner] Isaac Sim 없음")
        return []

    try:
        stage = omni.usd.get_context().get_stage()
        cam_paths = []

        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            if type_name in ("Camera", "camera"):
                path = str(prim.GetPath())
                cam_paths.append(path)
                print(f"  📷 카메라 발견: {path}")

        if not cam_paths:
            print("[QRScanner] ⚠️ Stage에서 카메라를 찾을 수 없음")
        else:
            print(f"\n[QRScanner] 총 {len(cam_paths)}개 카메라 발견")
            print("CAMERA_PRIMS 딕셔너리에 위 경로 중 Top-View 카메라를 할당하세요.")

        return cam_paths
    except Exception as e:
        print(f"[QRScanner] 탐색 오류: {e}")
        return []


# ============================================================
# 실행 안내
# ============================================================
print("\n[QRScanner] ✅ Plan B QR 스캐너 로드 완료")
print("""
사용법:
  1. 카메라 경로 확인:
       find_camera_prims()

  2. CAMERA_PRIMS 딕셔너리 수정 (이 파일 상단):
       CAMERA_PRIMS = {
           "sg2_in_01": "/World/실제카메라경로",
           ...
       }

  3. 단일 스캔 테스트:
       scanner = SH5QRScanner("sg2_in_01")
       result = scanner.scan()

  4. 카메라 이미지 디버그:
       scanner.debug_capture("/tmp/cam_check.png")

  5. sh5_integrated.py에 패치 (이미 실행 중일 때):
       patch_sh5_integrated_qr(controller)
""")
