#!/usr/bin/env python3
"""
=============================================================================
 SH5 로봇 데이터 증강 스크립트 v2 (좌표 기반 정밀 증강)
=============================================================================
 슬롯 1 데이터에서 슬롯 3 데이터를 자동 생성합니다.
 (같은 손, 같은 X축, Z축만 다름)

 작업대 슬롯 배치도 (정면에서 봤을 때):
   ┌──────────┬──────────┐
   │ 2 (x>0)  │ 1 (x<0)  │  ← 위층 Z_center: 1.3887
   ├──────────┼──────────┤
   │ 4 (x>0)  │ 3 (x<0)  │  ← 아래층 Z_center: 0.6505
   └──────────┴──────────┘

 증강 전략:
   - 접근/잡기 단계 (box가 벨트 위에 있음): 변경 없음 (벨트 위치는 동일)
   - 운반/배치 단계 (box를 들고 이동): lift_joint와 box_z에 오프셋 적용
   - 자동 단계 감지로 전환점을 찾음

 사용법:
   python3 augment_data.py \
     --input /home/rokey/dev_ws/datasets/coupang_demo_20260608_150012.hdf5 \
     --output /home/rokey/dev_ws/datasets/augmented_slot1_slot3.hdf5
=============================================================================
"""

import argparse
import h5py
import numpy as np
import os


# ============================================================================
# 슬롯 좌표 정의
# ============================================================================
SLOTS = {
    1: {"x": (-0.44569, -0.03577), "y": (-1.45619, -1.05474), "z": (1.24297, 1.3887, 1.885)},
    2: {"x": (0.03577, 0.44569),   "y": (-1.45619, -1.05474), "z": (1.24297, 1.3887, 1.885)},
    3: {"x": (-0.44569, -0.03577), "y": (-1.45619, -1.05474), "z": (0.54241, 0.65046, 1.175)},
    4: {"x": (0.03577, 0.44569),   "y": (-1.45619, -1.05474), "z": (0.54241, 0.65046, 1.175)},
}

# Z 중심값 차이 (슬롯3 - 슬롯1)
Z_OFFSET_1_TO_3 = SLOTS[3]["z"][1] - SLOTS[1]["z"][1]  # 0.65046 - 1.3887 = -0.73824

IDX_LIFT = 62


def detect_grasp_frame(joint_pos_seq):
    """
    오른손 손가락이 닫히기 시작하는 프레임 (잡기 시작점)을 반환합니다.
    이 프레임 이후부터 Z 오프셋을 적용합니다.
    """
    finger_r = joint_pos_seq[:, 40:60].mean(axis=1)
    finger_median = np.median(finger_r)
    
    # 처음으로 median을 넘는 프레임 = 잡기 시작
    for i in range(len(finger_r)):
        if finger_r[i] > finger_median:
            return max(0, i - 5)  # 5프레임 여유
    
    return len(finger_r) // 3  # 감지 실패 시 1/3 지점


def detect_release_frame(joint_pos_seq, grasp_frame):
    """
    상자를 놓는 프레임 (잡기 이후 손가락이 다시 열리는 지점)을 반환합니다.
    """
    finger_r = joint_pos_seq[:, 40:60].mean(axis=1)
    finger_median = np.median(finger_r)
    
    # grasp_frame 이후, 처음으로 median 아래로 떨어지는 프레임
    for i in range(grasp_frame + 10, len(finger_r)):
        if finger_r[i] < finger_median:
            return i
    
    return len(finger_r)  # 감지 실패 시 끝까지


def smooth_blend(n_frames, blend_len=20):
    """
    0에서 1로 부드럽게 전환되는 블렌딩 커브 (blend_len 프레임에 걸쳐)
    급격한 전환 방지를 위한 코사인 보간
    """
    curve = np.zeros(n_frames)
    for i in range(n_frames):
        if i < blend_len:
            # 코사인 보간: 0 → 1
            curve[i] = 0.5 * (1 - np.cos(np.pi * i / blend_len))
        else:
            curve[i] = 1.0
    return curve


def augment_episode_slot1_to_slot3(demo_grp):
    """
    슬롯 1 에피소드 → 슬롯 3 에피소드로 변환
    
    전략:
      1. 접근/잡기 (frame 0 ~ grasp_frame): 변경 없음 (벨트에서 잡는 건 동일)
      2. 잡기→운반 (grasp_frame ~ grasp_frame+blend): 점진적으로 Z 오프셋 적용
      3. 운반/배치 (grasp+blend ~ release): 전체 Z 오프셋 적용
      4. 놓기 이후 (release ~ 끝): Z 오프셋 유지
    """
    obs = demo_grp['obs']
    
    # 원본 데이터 로드
    robot_pose = obs['robot_pose'][:].copy()
    box_pose = obs['box_pose'][:].copy()
    rack_pose = obs['rack_pose'][:].copy()
    joint_pos = obs['joint_positions'][:].copy()
    joint_vel = obs['joint_velocities'][:].copy()
    actions = demo_grp['actions'][:].copy()
    cmd_vel = demo_grp['cmd_vel'][:].copy()
    rewards = demo_grp['rewards'][:].copy()
    dones = demo_grp['dones'][:].copy()
    
    N = len(joint_pos)
    
    # 잡기/놓기 프레임 감지
    grasp_frame = detect_grasp_frame(joint_pos)
    release_frame = detect_release_frame(joint_pos, grasp_frame)
    
    # 블렌딩 커브 생성 (잡기 후 부드럽게 전환)
    blend_len = min(30, (release_frame - grasp_frame) // 4)
    
    # 프레임별 Z 오프셋 가중치 생성
    z_weight = np.zeros(N)
    for i in range(N):
        if i < grasp_frame:
            z_weight[i] = 0.0  # 접근 단계: 변경 없음
        elif i < grasp_frame + blend_len:
            # 부드러운 전환
            t = (i - grasp_frame) / blend_len
            z_weight[i] = 0.5 * (1 - np.cos(np.pi * t))
        else:
            z_weight[i] = 1.0  # 운반/배치: 전체 오프셋
    
    # Z 오프셋 적용
    z_offset = Z_OFFSET_1_TO_3  # -0.73824m
    
    for i in range(N):
        w = z_weight[i]
        if w > 0:
            # lift_joint 오프셋 (관절 위치 & 관절 타겟)
            joint_pos[i, IDX_LIFT] += z_offset * w
            actions[i, IDX_LIFT] += z_offset * w
            
            # box_pose Z 오프셋 (상자가 들려있을 때만)
            box_pose[i, 2] += z_offset * w
    
    return {
        'obs/robot_pose': robot_pose,
        'obs/box_pose': box_pose,
        'obs/rack_pose': rack_pose,
        'obs/joint_positions': joint_pos,
        'obs/joint_velocities': joint_vel,
        'actions': actions,
        'cmd_vel': cmd_vel,
        'rewards': rewards,
        'dones': dones,
        'num_samples': demo_grp.attrs['num_samples'],
        'grasp_frame': grasp_frame,
        'release_frame': release_frame,
    }


def main(args):
    with h5py.File(args.input, 'r') as f_in:
        demo_names = sorted(f_in['data'].keys(), key=lambda x: int(x.split('_')[1]))
        n_demos = len(demo_names)
        print(f"[INFO] 입력 파일: {args.input}")
        print(f"[INFO] 원본 에피소드 수: {n_demos}")
        print(f"[INFO] Z 오프셋: {Z_OFFSET_1_TO_3:.5f}m (슬롯1 → 슬롯3)")
        
        with h5py.File(args.output, 'w') as f_out:
            data_grp = f_out.create_group('data')
            total_count = 0
            
            # ---- 슬롯 1: 원본 그대로 복사 ----
            print(f"\n{'='*50}")
            print(f"  슬롯 1 (원본) 복사 중...")
            print(f"{'='*50}")
            for demo_name in demo_names:
                demo = f_in['data'][demo_name]
                ep_name = f"demo_{total_count}"
                ep_grp = data_grp.create_group(ep_name)
                ep_grp.attrs['num_samples'] = demo.attrs['num_samples']
                ep_grp.attrs['slot_id'] = 1
                
                obs_grp = ep_grp.create_group('obs')
                obs_grp.create_dataset('robot_pose', data=demo['obs/robot_pose'][:])
                obs_grp.create_dataset('box_pose', data=demo['obs/box_pose'][:])
                obs_grp.create_dataset('rack_pose', data=demo['obs/rack_pose'][:])
                obs_grp.create_dataset('joint_positions', data=demo['obs/joint_positions'][:])
                obs_grp.create_dataset('joint_velocities', data=demo['obs/joint_velocities'][:])
                ep_grp.create_dataset('actions', data=demo['actions'][:])
                ep_grp.create_dataset('cmd_vel', data=demo['cmd_vel'][:])
                ep_grp.create_dataset('rewards', data=demo['rewards'][:])
                ep_grp.create_dataset('dones', data=demo['dones'][:])
                total_count += 1
            print(f"  → {n_demos}개 에피소드 복사 완료")
            
            # ---- 슬롯 3: Z 오프셋 적용하여 증강 ----
            print(f"\n{'='*50}")
            print(f"  슬롯 3 (Z 오프셋 증강) 생성 중...")
            print(f"{'='*50}")
            for i, demo_name in enumerate(demo_names):
                demo = f_in['data'][demo_name]
                aug = augment_episode_slot1_to_slot3(demo)
                
                ep_name = f"demo_{total_count}"
                ep_grp = data_grp.create_group(ep_name)
                ep_grp.attrs['num_samples'] = aug['num_samples']
                ep_grp.attrs['slot_id'] = 3
                
                obs_grp = ep_grp.create_group('obs')
                obs_grp.create_dataset('robot_pose', data=aug['obs/robot_pose'])
                obs_grp.create_dataset('box_pose', data=aug['obs/box_pose'])
                obs_grp.create_dataset('rack_pose', data=aug['obs/rack_pose'])
                obs_grp.create_dataset('joint_positions', data=aug['obs/joint_positions'])
                obs_grp.create_dataset('joint_velocities', data=aug['obs/joint_velocities'])
                ep_grp.create_dataset('actions', data=aug['actions'])
                ep_grp.create_dataset('cmd_vel', data=aug['cmd_vel'])
                ep_grp.create_dataset('rewards', data=aug['rewards'])
                ep_grp.create_dataset('dones', data=aug['dones'])
                
                if i < 3:
                    print(f"    {demo_name}: grasp@frame{aug['grasp_frame']}, "
                          f"release@frame{aug['release_frame']}")
                
                total_count += 1
            print(f"  → {n_demos}개 에피소드 증강 완료")
            
            print(f"\n{'='*50}")
            print(f"  증강 완료!")
            print(f"  슬롯1: {n_demos}개 + 슬롯3: {n_demos}개 = 총 {total_count}개")
            print(f"  저장: {args.output}")
            print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SH5 슬롯1→슬롯3 정밀 증강")
    parser.add_argument("--input", type=str, required=True,
                        help="원본 HDF5 파일 경로 (슬롯 1번 데이터)")
    parser.add_argument("--output", type=str, required=True,
                        help="증강된 데이터 저장 경로")
    args = parser.parse_args()
    main(args)
