#!/usr/bin/env python3
"""
=============================================================================
 Slot3 → Slot4 데이터 증강 (slot1↔slot2 실측 비교 기반)
=============================================================================
 slot1 vs slot2 실측 분석 결과:
 
   robot_pose X : 같은부호 → 반전 안함
   robot_pose qz: 같은부호 → 반전 안함
   box_pose X   : 같은부호 → 반전 안함  ← 기존 코드가 틀림!
   box_pose qz  : 반전      → qz만 반전
   cmd_vel wz   : 같은부호 → 반전 안함  ← 기존 코드가 틀림!
   cmd_vel vy   : 반전      → vy만 반전

 arm 관절 부호 규칙 (slot1 arm_R → slot4 arm_L):
   joint1 [9→8]  : +swap  (부호 유지)
   joint2 [12→11]: -swap  (부호 반전)
   joint3 [14→13]: -swap  (부호 반전)
   joint4 [16→15]: +swap  (부호 유지)
   joint5 [18→17]: -swap  (부호 반전)
   joint6 [20→19]: +swap  (부호 유지, 복잡하지만 +로 처리)
   joint7 [22→21]: +swap  (부호 유지)
 
 finger: +swap (부호 유지)
=============================================================================
"""

import argparse
import os
import h5py
import numpy as np


# ============================================================================
# 실제 관절 인덱스 (Isaac Sim 디버그 출력 기준)
# ============================================================================
IDX_ARM_L    = [8, 11, 13, 15, 17, 19, 21]
IDX_ARM_R    = [9, 12, 14, 16, 18, 20, 22]
IDX_FINGER_L = [23,24,25,26,27,33,34,35,36,37,43,44,45,46,47,53,54,55,56,57]
IDX_FINGER_R = [28,29,30,31,32,38,39,40,41,42,48,49,50,51,52,58,59,60,61,62]

# slot1_R → slot4_L 부호 규칙 (실측 분석 기반)
# True = 부호 유지, False = 부호 반전
ARM_SIGN_KEEP = [True, False, False, True, False, True, True]


# ============================================================================
# 변환 함수
# ============================================================================

def mirror_pose_qz_only(pose_seq):
    """
    pose_seq: (N, 7) — [x, y, z, qw, qx, qy, qz]
    box/rack 방향만 반전 (qz 부호 반전). XYZ 위치는 유지.
    """
    m = pose_seq.copy()
    m[:, 6] = -pose_seq[:, 6]   # qz 반전
    return m


def mirror_cmd_vel(cmd_vel_seq):
    """
    cmd_vel_seq: (N, 3) — [vx, vy, wz]
    vy만 반전 (좌우 이동 방향). wz 유지.
    """
    m = cmd_vel_seq.copy()
    m[:, 1] = -cmd_vel_seq[:, 1]  # vy 반전
    return m


def mirror_joints(joint_seq):
    """
    joint_seq: (N, 63)
    arm_R → arm_L 스왑 (관절별 부호 규칙 적용)
    finger_R → finger_L 스왑 (부호 유지)
    """
    m = joint_seq.copy()

    # arm 스왑 (관절별 부호 규칙 적용)
    for i, (r_idx, l_idx, keep_sign) in enumerate(zip(IDX_ARM_R, IDX_ARM_L, ARM_SIGN_KEEP)):
        sign = 1.0 if keep_sign else -1.0
        m[:, l_idx] = sign * joint_seq[:, r_idx]   # arm_R → arm_L
        m[:, r_idx] = sign * joint_seq[:, l_idx]   # arm_L → arm_R (대칭 스왑)

    # finger 스왑 (부호 유지)
    for l_idx, r_idx in zip(IDX_FINGER_L, IDX_FINGER_R):
        m[:, l_idx] = joint_seq[:, r_idx]
        m[:, r_idx] = joint_seq[:, l_idx]

    return m


# ============================================================================
# 메인
# ============================================================================

def augment_slot3_to_slot4(input_path, output_path):
    with h5py.File(input_path, 'r') as f_in:
        demos = sorted(f_in['data'].keys())
        total = len(demos)
        print(f"[INFO] 입력: {input_path}  ({total}개 에피소드)")
        print(f"[INFO] 변환 규칙 (slot1↔slot2 실측 기반):")
        print(f"       - robot_pose: 변환 없음")
        print(f"       - box/rack pose: qz만 반전 (위치 유지)")
        print(f"       - cmd_vel: vy만 반전 (wz 유지)")
        print(f"       - arm: R↔L 스왑 + 관절별 부호: {ARM_SIGN_KEEP}")
        print(f"       - finger: R↔L 스왑 (부호 유지)")

        with h5py.File(output_path, 'w') as f_out:
            grp_data = f_out.create_group('data')
            if 'metadata' in f_in:
                f_in.copy('metadata', f_out)

            for i, demo_name in enumerate(demos):
                demo = f_in['data'][demo_name]
                obs  = demo['obs']

                robot_pose = obs['robot_pose'][:]
                box_pose   = obs['box_pose'][:]
                rack_pose  = obs['rack_pose'][:]
                joint_pos  = obs['joint_positions'][:]
                joint_vel  = obs['joint_velocities'][:]
                joint_tgt  = demo['actions'][:]
                cmd_vel    = demo['cmd_vel'][:]

                # 변환
                new_robot_pose = robot_pose.copy()               # 변환 없음
                new_box_pose   = mirror_pose_qz_only(box_pose)   # qz만 반전
                new_rack_pose  = mirror_pose_qz_only(rack_pose)  # qz만 반전
                new_joint_pos  = mirror_joints(joint_pos)
                new_joint_vel  = mirror_joints(joint_vel)
                new_joint_tgt  = mirror_joints(joint_tgt)
                new_cmd_vel    = mirror_cmd_vel(cmd_vel)         # vy만 반전

                ep = grp_data.create_group(demo_name)
                ep_obs = ep.create_group('obs')
                ep_obs.create_dataset('robot_pose',       data=new_robot_pose.astype(np.float32))
                ep_obs.create_dataset('box_pose',         data=new_box_pose.astype(np.float32))
                ep_obs.create_dataset('rack_pose',        data=new_rack_pose.astype(np.float32))
                ep_obs.create_dataset('joint_positions',  data=new_joint_pos.astype(np.float32))
                ep_obs.create_dataset('joint_velocities', data=new_joint_vel.astype(np.float32))
                ep.create_dataset('actions',  data=new_joint_tgt.astype(np.float32))
                ep.create_dataset('cmd_vel',  data=new_cmd_vel.astype(np.float32))

                for attr_key, attr_val in demo.attrs.items():
                    ep.attrs[attr_key] = attr_val
                ep.attrs['slot_id'] = 4
                if 'num_samples' not in ep.attrs:
                    ep.attrs['num_samples'] = len(robot_pose)

                for key in obs.keys():
                    if key not in ep_obs:
                        obs.copy(key, ep_obs)

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  [{i+1}/{total}] {demo_name} (N={len(robot_pose)})")

    print(f"\n[완료] {output_path}")


def verify(slot3_path, slot4_path):
    print("\n[검증]")
    with h5py.File(slot3_path,'r') as f3, h5py.File(slot4_path,'r') as f4:
        d3 = f3['data'][sorted(f3['data'].keys())[0]]
        d4 = f4['data'][sorted(f4['data'].keys())[0]]

        bp3, bp4 = d3['obs']['box_pose'][:], d4['obs']['box_pose'][:]
        cv3, cv4 = d3['cmd_vel'][:], d4['cmd_vel'][:]
        jp3, jp4 = d3['obs']['joint_positions'][:], d4['obs']['joint_positions'][:]
        rp3, rp4 = d3['obs']['robot_pose'][:], d4['obs']['robot_pose'][:]

        print(f"  robot_pose X: {rp3[:,0].mean():.4f} → {rp4[:,0].mean():.4f}  (기대: 동일)")
        print(f"  robot_pose qz: {rp3[:,6].mean():.4f} → {rp4[:,6].mean():.4f}  (기대: 동일)")
        print(f"  box_pose X: {bp3[:,0].mean():.4f} → {bp4[:,0].mean():.4f}  (기대: 동일)")
        print(f"  box_pose qz: {bp3[:,6].mean():.4f} → {bp4[:,6].mean():.4f}  (기대: 반전)")
        print(f"  cmd_vel vy: {cv3[:,1].mean():.4f} → {cv4[:,1].mean():.4f}  (기대: 반전)")
        print(f"  cmd_vel wz: {cv3[:,2].mean():.4f} → {cv4[:,2].mean():.4f}  (기대: 동일)")
        # arm 스왑 확인
        print(f"  arm_R[9]→arm_L[8]: slot3={jp3[:,9].mean():.4f} slot4_L={jp4[:,8].mean():.4f}  기대={jp3[:,9].mean():.4f}")
        print(f"  arm_R[12]→arm_L[11] (-): slot3={jp3[:,12].mean():.4f} slot4_L={jp4[:,11].mean():.4f}  기대={-jp3[:,12].mean():.4f}")
        print(f"  slot_id={d4.attrs.get('slot_id')}, num_samples={d4.attrs.get('num_samples')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/home/rokey/dev_ws/datasets/slot3_coupang_demo_20260608_210842.hdf5")
    parser.add_argument("--output", default="/home/rokey/dev_ws/datasets/slot4_augmented.hdf5")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    augment_slot3_to_slot4(args.input, args.output)
    if args.verify:
        verify(args.input, args.output)
