#!/usr/bin/env python3
import argparse
import os
import glob
import h5py
import numpy as np
import time
import cv2

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay HDF5 datasets to extract vision data.")
parser.add_argument("--data_dir", type=str, default="/home/rokey/dev_ws/datasets", help="디렉토리 경로")
parser.add_argument("--output_dir", type=str, default="/home/rokey/dev_ws/datasets_vision", help="저장 경로")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import omni.replicator.core as rep
from isaaclab.scene import InteractiveScene
from robotis_lab.scripts.sim2real.bringup.common import robotis_config as cfg
import coupang_sh5_bringup_v as bringup

def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = bringup.CoupangSceneCfg(num_envs=1, env_spacing=2.0)
    # Replay에서는 로봇과 환경이 중력에 의해 떨어지지 않게 Kinematic/Disable gravity 설정이 필요할 수 있으나, 
    # 매 프레임 강제로 set_root_state 및 set_joint_pos를 하므로 문제 없습니다.
    scene_cfg.robot = bringup._make_robot_cfg(bringup.FFW_SH5_CFG.spawn.usd_path)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # 카메라 셋업
    camera_specs = [
        ("Overview", "/World/OverviewCamera", 1280, 720, 10, 22),
        ("Head Camera", f"{{ENV_REGEX_NS}}/Robot/{cfg.HEAD_CAMERA_FRAME}", 500, 500, 1345, 22),
        ("Left Camera", f"{{ENV_REGEX_NS}}/Robot/{cfg.LEFT_CAMERA_FRAME}", 500, 500, 1345, 545),
        ("Right Camera", f"{{ENV_REGEX_NS}}/Robot/{cfg.RIGHT_CAMERA_FRAME}", 500, 500, 835, 545),
    ]
    camera_paths = bringup._setup_camera_views(sim.stage, camera_specs)
    
    # Replicator Annotator 설정 (이미지 획득용)
    annotators = {}
    TARGET_CAMERAS = ["Head Camera", "Left Camera", "Right Camera", "TopView"]
    
    for cam_name, cam_path in camera_paths.items():
        if cam_name in TARGET_CAMERAS:
            rp = rep.create.render_product(cam_path, (224, 224)) # ResNet 기본 해상도 224x224
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(rp)
            annotators[cam_name] = annotator

    print("[INFO] Replicator Annotators initialized.")

    hdf5_files = glob.glob(os.path.join(args_cli.data_dir, "*.hdf5"))
    print(f"[INFO] 발견된 데이터셋 파일: {len(hdf5_files)}개")

    for file_idx, filepath in enumerate(hdf5_files):
        filename = os.path.basename(filepath)
        out_filepath = os.path.join(args_cli.output_dir, filename)
        
        if os.path.exists(out_filepath):
            print(f"[{file_idx+1}/{len(hdf5_files)}] 스킵 (이미 존재): {filename}")
            continue
            
        print(f"[{file_idx+1}/{len(hdf5_files)}] 변환 시작: {filename}")
        
        with h5py.File(filepath, "r") as f_in, h5py.File(out_filepath, "w") as f_out:
            # 복사할 때 slot_id 같은 최상위 속성 유지
            for k, v in f_in.attrs.items():
                f_out.attrs[k] = v
                
            out_data_grp = f_out.create_group("data")
            in_data_grp = f_in["data"]
            for k, v in in_data_grp.attrs.items():
                out_data_grp.attrs[k] = v

            demos = list(in_data_grp.keys())
            for demo_idx, demo_name in enumerate(demos):
                demo_in = in_data_grp[demo_name]
                demo_out = out_data_grp.create_group(demo_name)
                
                # 기존 데이터 복사
                for k, v in demo_in.attrs.items():
                    demo_out.attrs[k] = v
                    
                obs_in = demo_in["obs"]
                obs_out = demo_out.create_group("obs")
                
                # 기존 obs 데이터 복사 (NumPy array로 가져와서 새로 씀)
                robot_pose_arr = np.array(obs_in["robot_pose"])
                joint_pos_arr = np.array(obs_in["joint_positions"])
                joint_vel_arr = np.array(obs_in["joint_velocities"])
                box_pose_arr = np.array(obs_in["box_pose"])
                rack_pose_arr = np.array(obs_in["rack_pose"])
                
                obs_out.create_dataset("robot_pose", data=robot_pose_arr, compression="gzip")
                obs_out.create_dataset("joint_positions", data=joint_pos_arr, compression="gzip")
                obs_out.create_dataset("joint_velocities", data=joint_vel_arr, compression="gzip")
                obs_out.create_dataset("box_pose", data=box_pose_arr, compression="gzip")
                obs_out.create_dataset("rack_pose", data=rack_pose_arr, compression="gzip")
                
                # actions 복사
                demo_out.create_dataset("actions", data=np.array(demo_in["actions"]), compression="gzip")
                demo_out.create_dataset("cmd_vel", data=np.array(demo_in["cmd_vel"]), compression="gzip")
                if "rewards" in demo_in:
                    demo_out.create_dataset("rewards", data=np.array(demo_in["rewards"]), compression="gzip")
                if "dones" in demo_in:
                    demo_out.create_dataset("dones", data=np.array(demo_in["dones"]), compression="gzip")

                num_samples = len(robot_pose_arr)
                
                # 이미지 저장용 배열 (JPEG byte 리스트 또는 uint8 원시 배열)
                # 여기서는 224x224x3 uint8 원시 배열을 gzip 압축하여 저장 (안전성)
                images_grp = obs_out.create_group("images")
                img_datasets = {}
                for cam_name in TARGET_CAMERAS:
                    img_datasets[cam_name] = images_grp.create_dataset(
                        cam_name, 
                        shape=(num_samples, 224, 224, 3), 
                        dtype=np.uint8, 
                        compression="gzip"
                    )

                print(f"  - {demo_name}: {num_samples} 프레임 렌더링 중...", end="", flush=True)
                
                for t in range(num_samples):
                    # 1. 로봇 및 환경 상태 설정
                    # 로봇 위치 (0 = base)
                    # wait, sim2real bridge typically handles robot. robot_pose is base link pose in world.
                    # but root state is enough.
                    robot_state = scene["robot"].data.default_root_state.clone()
                    robot_state[0, :7] = torch.tensor(robot_pose_arr[t], device=sim.device)
                    scene["robot"].write_root_state_to_sim(robot_state)
                    
                    # 로봇 관절
                    j_pos = torch.tensor(joint_pos_arr[t], device=sim.device).unsqueeze(0)
                    j_vel = torch.tensor(joint_vel_arr[t], device=sim.device).unsqueeze(0)
                    scene["robot"].write_joint_state_to_sim(j_pos, j_vel)
                    
                    # 박스 위치
                    if "box" in scene.keys():
                        box_state = scene["box"].data.default_root_state.clone()
                        box_state[0, :7] = torch.tensor(box_pose_arr[t], device=sim.device)
                        scene["box"].write_root_state_to_sim(box_state)
                        
                    # 랙 위치 (필요한 경우)
                    if "rack" in scene.keys():
                        rack_state = scene["rack"].data.default_root_state.clone()
                        rack_state[0, :7] = torch.tensor(rack_pose_arr[t], device=sim.device)
                        scene["rack"].write_root_state_to_sim(rack_state)

                    # 2. 시뮬레이션 스텝 진행 (물리엔진 끄거나 1 step 강제 렌더링)
                    # rep_step() generates the annotator data
                    rep.orchestrator.step(rt_subframes=1)
                    sim.step(render=True)
                    
                    # 3. 이미지 획득 및 저장
                    for cam_name, ann in annotators.items():
                        img_data = ann.get_data()
                        if img_data is not None and len(img_data) > 0:
                            rgb = img_data[..., :3]  # RGBA to RGB
                            img_datasets[cam_name][t] = rgb
                        else:
                            # 렌더링 실패 시 이전 프레임이나 빈 이미지
                            pass
                            
                print(" 완료!")

    print("\n[SUCCESS] 모든 비전 이미지 추출 작업 완료!")
    simulation_app.close()

if __name__ == "__main__":
    main()
