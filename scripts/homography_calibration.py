import cv2
import numpy as np
import json
import os

def calibrate_homography():
    """
    4개의 기준점(작업대 모서리 등)을 이용하여
    Top-view 카메라 이미지 픽셀(2D) -> 로봇 베이스 기준 물리 좌표(2D) 변환 행렬을 계산합니다.
    """
    print("=== 평면 호모그래피(Planar Homography) 캘리브레이션 ===")

    # 1. 카메라 이미지 상의 픽셀 좌표 (사용자가 이미지에서 직접 클릭하거나 찍은 좌표)
    # 예시: 해상도가 1024x1024일 때, 모서리 4개의 픽셀 좌표 [u, v]
    # TODO: 실제 환경에 맞게 좌표를 업데이트해야 합니다.
    pixel_points = np.array([
        [100, 100],  # 좌측 상단 모서리
        [924, 100],  # 우측 상단 모서리
        [924, 924],  # 우측 하단 모서리
        [100, 924]   # 좌측 하단 모서리
    ], dtype=np.float32)

    # 2. 로봇 베이스 기준의 실제 물리 좌표 (단위: 미터)
    # 예시: 로봇 베이스(0,0) 기준 작업대 모서리의 3D 공간상 (X, Y) 좌표
    # 높이(Z)는 평면이므로 일정하다고 가정합니다.
    # TODO: 로봇으로 실제 측정하여 일치하는 4개 지점의 [X, Y]를 입력해야 합니다.
    real_points = np.array([
        [0.8,  0.5], # 좌측 상단에 대응하는 로봇 물리 좌표
        [0.8, -0.5], # 우측 상단 대응
        [0.2, -0.5], # 우측 하단 대응
        [0.2,  0.5]  # 좌측 하단 대응
    ], dtype=np.float32)

    # 3. 호모그래피 변환 행렬 계산
    # H 행렬은 3x3 크기이며, pixel_points를 real_points로 변환해줍니다.
    H, status = cv2.findHomography(pixel_points, real_points)
    
    print("\n계산된 Homography Matrix (H):")
    print(H)

    # 4. 검증 (테스트)
    # 계산된 H 행렬을 통해 픽셀 좌표를 다시 넣어보고 실제 좌표와 맞게 나오는지 확인
    pixel_points_homogeneous = np.array([[[p[0], p[1]]] for p in pixel_points], dtype=np.float32)
    transformed_points = cv2.perspectiveTransform(pixel_points_homogeneous, H)
    
    print("\n검증: 이미지 픽셀 -> 물리 좌표 변환 결과")
    for i in range(4):
        print(f"입력 픽셀 {pixel_points[i]} -> 변환 좌표 {transformed_points[i][0]} (실제 정답: {real_points[i]})")

    # 5. 행렬 저장 (JSON 형태)
    output_path = os.path.join(os.path.dirname(__file__), 'homography_matrix.json')
    with open(output_path, 'w') as f:
        json.dump(H.tolist(), f, indent=4)
    print(f"\n변환 행렬이 저장되었습니다: {output_path}")

def pixel_to_world(u, v, H):
    """
    저장된 호모그래피 행렬(H)을 사용하여 특정 픽셀(u, v)을 로봇의 물리 좌표(X, Y)로 변환합니다.
    """
    pt = np.array([[[u, v]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H)
    return res[0][0][0], res[0][0][1]

if __name__ == "__main__":
    calibrate_homography()
