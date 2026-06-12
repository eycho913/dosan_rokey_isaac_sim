#!/usr/bin/env python3
"""
슬롯당 N개 에피소드만 추출하여 fine-tune용 서브셋 HDF5 생성
Usage (로컬):
  python3 create_subset.py --n_per_slot 20 --output /home/rokey/dev_ws/datasets/subset_80ep.hdf5

Usage (Colab):
  python3 create_subset.py \
    --data_dir /content/drive/MyDrive/dousan_3/datasets/vision_data \
    --n_per_slot 100 \
    --output /content/drive/MyDrive/dousan_3/datasets/subset_400ep.hdf5
"""
import argparse, glob, os
import h5py
import numpy as np

DATA_DIR = '/home/rokey/dev_ws/datasets/train_data/vision_data'  # 기본값 (로컬)

SLOT_FILES = {
    1: ['vision_slot1_1_f.hdf5', 'vision_slot1_2.hdf5'],
    2: ['vision_slot2_1_f.hdf5', 'vision_slot2_2.hdf5'],
    3: ['vision_slot3_1.hdf5',   'vision_slot3_2_f.hdf5'],
    4: ['vision_slot4_1_f.hdf5', 'vision_slot4_2.hdf5'],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str, default=DATA_DIR,
                        help='원본 HDF5 파일들이 있는 폴더 경로')
    parser.add_argument('--n_per_slot', type=int, default=20, help='슬롯당 에피소드 수')
    parser.add_argument('--output',     type=str,
                        default='/home/rokey/dev_ws/datasets/subset_80ep.hdf5',
                        help='출력 HDF5 경로')
    args = parser.parse_args()

    src_dir = args.data_dir
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    total_copied = 0


    with h5py.File(args.output, 'w') as f_out:
        out_data = f_out.create_group('data')
        demo_idx = 0

        for slot_id, filenames in SLOT_FILES.items():
            slot_collected = 0
            print(f"\n[Slot {slot_id}] 목표: {args.n_per_slot}개 에피소드")

            for fname in filenames:
                if slot_collected >= args.n_per_slot:
                    break
                fpath = os.path.join(src_dir, fname)
                if not os.path.exists(fpath):
                    print(f"  파일 없음: {fpath}")
                    continue

                with h5py.File(fpath, 'r') as f_in:
                    # slot_id 속성 확인
                    file_slot = f_in.attrs.get('slot_id', slot_id)
                    demos = list(f_in['data'].keys())
                    np.random.shuffle(demos)  # 랜덤 선택

                    for demo_name in demos:
                        if slot_collected >= args.n_per_slot:
                            break
                        src = f_in['data'][demo_name]
                        # 이미지 있는 것만
                        if 'obs' not in src or 'images' not in src['obs']:
                            continue

                        dst_name = f'demo_{demo_idx:04d}'
                        dst = out_data.create_group(dst_name)

                        # 속성 복사
                        dst.attrs['slot_id'] = int(slot_id)
                        for k, v in src.attrs.items():
                            dst.attrs[k] = v

                        # obs 복사
                        obs_out = dst.create_group('obs')
                        obs_in = src['obs']
                        for key in ['joint_positions', 'joint_velocities',
                                    'robot_pose', 'box_pose', 'rack_pose']:
                            if key in obs_in:
                                obs_out.create_dataset(key, data=obs_in[key][:], compression='gzip')

                        # 이미지 복사
                        img_out = obs_out.create_group('images')
                        for cam in obs_in['images'].keys():
                            img_out.create_dataset(cam, data=obs_in['images'][cam][:],
                                                   compression='gzip')

                        # actions, cmd_vel 복사
                        for key in ['actions', 'cmd_vel', 'phases', 'rewards', 'dones']:
                            if key in src:
                                dst.create_dataset(key, data=src[key][:], compression='gzip')

                        frames = len(obs_in['joint_positions'])
                        print(f"  [{slot_collected+1}/{args.n_per_slot}] {fname}/{demo_name} "
                              f"→ {dst_name} ({frames} frames)")

                        demo_idx += 1
                        slot_collected += 1
                        total_copied += 1

            print(f"  Slot {slot_id} 완료: {slot_collected}개 수집")

        # 전체 메타데이터
        out_data.attrs['total_demos'] = total_copied
        out_data.attrs['n_per_slot'] = args.n_per_slot

    print(f"\n✅ 서브셋 완료: {args.output}")
    print(f"   총 에피소드: {total_copied}개 ({args.n_per_slot}/슬롯 × 4슬롯)")

if __name__ == '__main__':
    main()
