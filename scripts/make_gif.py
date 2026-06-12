#!/usr/bin/env python3
"""webm → GIF 변환 스크립트 (OpenCV + Pillow)"""
import cv2
from PIL import Image
import os

VIDEO_DIR = "/home/rokey/dev_ws/ee_video/video"
OUT_DIR   = "/home/rokey/dev_ws/ee_video/gif"
os.makedirs(OUT_DIR, exist_ok=True)

# (파일명, 출력 GIF명, 시작초, 길이초, fps, 가로px)
JOBS = [
    ("SG2로_pick&place_강화학습.webm",      "01_sg2_rl_pickplace.gif",        0,  8, 6, 480),
    ("모방학습_물리적잡기.webm",              "02_il_physical_grasp.gif",       0, 10, 6, 480),
    ("모방학습_물리적잡기_한계.webm",         "03_il_grasp_limit.gif",          0,  8, 6, 480),
    ("모방학습_환경_변환_한계.webm",          "04_il_env_limit.gif",            0,  8, 6, 480),
    ("모방학습결과_상자잡기실패.webm",        "05_il_fail.gif",                 0, 10, 6, 480),
    ("magic_snapping_방식그리퍼.webm",       "06_magic_snapping.gif",          0, 10, 6, 480),
    ("최종_시나리오.webm",                    "07_final_scenario.gif",          0, 15, 8, 480),
]

def webm_to_gif(src, dst, start_sec, duration_sec, fps, width):
    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start_frame = int(start_sec * src_fps)
    total_frames = int(duration_sec * src_fps)
    step = max(1, int(src_fps / fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    count = 0
    while count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            h, w = frame.shape[:2]
            new_h = int(h * width / w)
            frame = cv2.resize(frame, (width, new_h))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1
    cap.release()

    if not frames:
        print(f"  ⚠️ 프레임 없음: {src}")
        return
    frames[0].save(
        dst, save_all=True, append_images=frames[1:],
        loop=0, duration=int(1000/fps), optimize=True
    )
    size_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"  ✅ {os.path.basename(dst)} ({len(frames)} frames, {size_mb:.1f}MB)")

for fname, gname, t0, dur, fps, w in JOBS:
    src = os.path.join(VIDEO_DIR, fname)
    dst = os.path.join(OUT_DIR, gname)
    if not os.path.exists(src):
        print(f"  ⚠️ 파일 없음: {fname}")
        continue
    print(f"🎬 변환 중: {fname}")
    try:
        webm_to_gif(src, dst, t0, dur, fps, w)
    except Exception as e:
        print(f"  ❌ 오류: {e}")

print("\n✅ 전체 변환 완료 →", OUT_DIR)
