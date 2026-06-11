import h5py
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, required=True, help="분석할 HDF5 파일 경로")
    # 슬롯 2번(위층 우측) 기준 목표 좌표: x=0.24, y=-1.255, z=1.56
    parser.add_argument("--target_z", type=float, default=1.56, help="목표 높이 (슬롯2/1=1.56, 슬롯4/3=0.86)")
    parser.add_argument("--target_y", type=float, default=-1.10, help="목표 깊이 (선반 안쪽으로 들어갔는지 판단 기준, 예: -1.1 이하)")
    args = parser.parse_args()

    print(f"[{args.hdf5}] 성공/실패 분석 시작...")
    print(f"성공 기준: 마지막 스텝에서 상자의 Z가 {args.target_z - 0.2} ~ {args.target_z + 0.2} 사이고, Y가 {args.target_y} 보다 작아야 함(음수 방향)")
    
    failed_episodes = []
    
    with h5py.File(args.hdf5, 'r') as f:
        data = f['data']
        episodes = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
        
        for ep in episodes:
            # 마지막 스텝의 상자 좌표 가져오기 (x, y, z, qx, qy, qz, qw)
            box_poses = data[ep]['obs']['box_pose'][()]
            final_box_pose = box_poses[-1]
            box_x, box_y, box_z = final_box_pose[0], final_box_pose[1], final_box_pose[2]
            
            # 실패 조건 1: 상자가 바닥으로 떨어짐 (높이가 목표보다 한참 낮음)
            dropped = box_z < (args.target_z - 0.2)
            # 실패 조건 2: 상자가 선반 안쪽까지 덜 들어감
            not_inserted = box_y > args.target_y
            
            if dropped or not_inserted:
                reason = "바닥에 떨어짐" if dropped else "끝까지 안 들어감"
                print(f"❌ 실패 의심: {ep} (마지막 위치: X={box_x:.2f}, Y={box_y:.2f}, Z={box_z:.2f}) -> 사유: {reason}")
                failed_episodes.append(ep)
                
    print("-" * 50)
    print(f"총 {len(episodes)}개 중 실패 의심 에피소드: {len(failed_episodes)}개")
    if failed_episodes:
        print("삭제용 명령어 복사(예시):")
        print(f"isaac-python /home/rokey/dev_ws/coupang_ws/scripts/filter_dataset.py --input {args.hdf5} --output 수정본.hdf5 --remove_episodes {' '.join(failed_episodes)}")

if __name__ == "__main__":
    main()
