import os
import glob
import h5py
import numpy as np
import argparse
import sys
import shutil

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="입력 HDF5 폴더")
parser.add_argument("--output_dir", type=str, required=True, help="저장할 새 HDF5 폴더")
parser.add_argument("--stay_hdf5", type=str, default="/home/rokey/dev_ws/datasets/stay.hdf5", help="동결 자세의 기준이 되는 stay.hdf5 경로")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
import isaaclab.sim as sim_utils
from robotis_lab.assets.robots import FFW_SH5_CFG
from isaaclab.assets.articulation import Articulation

def main():
    print("[INFO] 시뮬레이션 환경 초기화 중 (관절 인덱스 탐색용)...")
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    robot_cfg = FFW_SH5_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    sim.reset()
    
    # 오른팔 인덱스
    right_arm_idx, _ = robot.find_joints("arm_r_.*")
    right_finger_idx, _ = robot.find_joints("finger_r_.*")
    right_freeze_indices = right_arm_idx + right_finger_idx
    
    # 왼팔 인덱스
    left_arm_idx, _ = robot.find_joints("arm_l_.*")
    left_finger_idx, _ = robot.find_joints("finger_l_.*")
    left_freeze_indices = left_arm_idx + left_finger_idx
    
    # Stay.hdf5에서 목표(안전한 오므린 자세) 관절값 추출
    if not os.path.exists(args_cli.stay_hdf5):
        print(f"[ERROR] {args_cli.stay_hdf5} 파일이 존재하지 않습니다!")
        app_launcher.app.close()
        return
        
    print(f"[INFO] Stay 데이터({args_cli.stay_hdf5})에서 기준 관절값을 가져옵니다...")
    with h5py.File(args_cli.stay_hdf5, "r") as f:
        demo_key = list(f["data"].keys())[0]
        # 마지막 프레임(-1)이 가장 완벽하게 정지된 자세입니다.
        stay_joint_pos = f["data"][demo_key]["obs"]["joint_positions"][-1].copy()
        stay_joint_vel = np.zeros_like(f["data"][demo_key]["obs"]["joint_velocities"][-1])
        stay_action = f["data"][demo_key]["actions"][-1].copy()

    files = glob.glob(os.path.join(args_cli.input_dir, "slot*.hdf5"))
    if not files:
        print(f"[INFO] {args_cli.input_dir} 경로에 slot 데이터가 없습니다.")
        app_launcher.app.close()
        return

    # 새 폴더 생성
    os.makedirs(args_cli.output_dir, exist_ok=True)
    print(f"[INFO] 출력 폴더를 준비했습니다: {args_cli.output_dir}")

    for filepath in files:
        if "vision" in filepath:
            continue
            
        filename = os.path.basename(filepath)
        
        # 슬롯 번호 파악
        slot_num_str = filename.split('_')[0].replace('slot', '')
        if not slot_num_str.isdigit():
            continue
        slot_num = int(slot_num_str)
        
        # 슬롯 1, 3: 오른손 작업 -> 왼팔 얼리기
        if slot_num in [1, 3]:
            freeze_indices = left_freeze_indices
            target_arm = "왼팔(Left Arm)"
        # 슬롯 2, 4: 왼손 작업 -> 오른팔 얼리기
        elif slot_num in [2, 4]:
            freeze_indices = right_freeze_indices
            target_arm = "오른팔(Right Arm)"
        else:
            continue
            
        out_filepath = os.path.join(args_cli.output_dir, filename)
        
        # 파일 원본을 먼저 복사
        print(f"\n[INFO] 파일 복사 중: {filename} -> {args_cli.output_dir}")
        shutil.copy2(filepath, out_filepath)
        
        print(f"[INFO] 데이터 수술 시작: {filename} (작업: {target_arm}을 Stay 자세로 동결 🧊)")
        
        # 복사된 새 파일을 열어서 수술 진행 (r+ 모드)
        with h5py.File(out_filepath, "r+") as f:
            if "data" not in f:
                continue
            demos = list(f["data"].keys())
            for demo_key in demos:
                demo = f["data"][demo_key]
                
                joint_positions = demo["obs"]["joint_positions"][:]
                joint_velocities = demo["obs"]["joint_velocities"][:]
                actions = demo["actions"][:]
                
                # Stay 자세의 관절값으로 덮어씌웁니다.
                for idx in freeze_indices:
                    joint_positions[:, idx] = stay_joint_pos[idx]
                    joint_velocities[:, idx] = stay_joint_vel[idx]
                    actions[:, idx] = stay_action[idx]
                
                demo["obs"]["joint_positions"][...] = joint_positions
                demo["obs"]["joint_velocities"][...] = joint_velocities
                demo["actions"][...] = actions
                
        print(f"[INFO] {filename} {target_arm} Stay 자세 수술 완료! 🏥")

    print(f"\n[INFO] 모든 Slot 데이터가 성공적으로 처리되어 {args_cli.output_dir} 에 저장되었습니다!")
    app_launcher.app.close()

if __name__ == "__main__":
    main()
