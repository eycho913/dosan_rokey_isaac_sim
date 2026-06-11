import os
import glob
import h5py
import numpy as np
import argparse
import sys

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default="/home/rokey/dev_ws/datasets/clean_data")
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
    
    # 오른팔 및 오른손 관절 인덱스 찾기
    right_arm_indices, right_arm_names = robot.find_joints("arm_r_.*")
    right_finger_indices, right_finger_names = robot.find_joints("finger_r_.*")
    
    freeze_indices = right_arm_indices + right_finger_indices
    print(f"[INFO] 고정할 오른팔/오른손 관절 인덱스 (총 {len(freeze_indices)}개): {freeze_indices}")
    
    files = glob.glob(os.path.join(args_cli.input_dir, "slot4_*.hdf5"))
    if not files:
        print("[INFO] slot4 데이터가 없습니다.")
        app_launcher.app.close()
        return

    for filepath in files:
        if "vision" in filepath:
            continue
        print(f"\n[INFO] 데이터 수술 시작: {filepath}")
        
        with h5py.File(filepath, "r+") as f:
            demos = list(f["data"].keys())
            for demo_key in demos:
                demo = f["data"][demo_key]
                num_samples = demo.attrs["num_samples"]
                
                # 0번 프레임의 관절값을 가져옵니다. (Stay 시작 자세)
                start_joint_pos = demo["obs"]["joint_positions"][0].copy()
                start_joint_vel = np.zeros_like(demo["obs"]["joint_velocities"][0])
                start_action = demo["actions"][0].copy()
                
                # 원본 데이터를 메모리에 로드
                joint_positions = demo["obs"]["joint_positions"][:]
                joint_velocities = demo["obs"]["joint_velocities"][:]
                actions = demo["actions"][:]
                
                # 오른팔 인덱스에 해당하는 열(Column)을 모두 0번 프레임 값으로 덮어씌웁니다.
                for idx in freeze_indices:
                    joint_positions[:, idx] = start_joint_pos[idx]
                    joint_velocities[:, idx] = start_joint_vel[idx]
                    actions[:, idx] = start_action[idx]
                
                # 수정된 데이터를 HDF5에 덮어쓰기
                demo["obs"]["joint_positions"][...] = joint_positions
                demo["obs"]["joint_velocities"][...] = joint_velocities
                demo["actions"][...] = actions
                
        print(f"[INFO] {os.path.basename(filepath)} 오른팔 고정(Freeze) 수술 완료! 🏥")

    print("\n[INFO] 모든 Slot 4 데이터 수술이 성공적으로 완료되었습니다!")
    app_launcher.app.close()

if __name__ == "__main__":
    main()
