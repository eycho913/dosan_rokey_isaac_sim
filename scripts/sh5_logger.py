import os
import time
import h5py
import numpy as np

class VRDemonstrationLogger:
    def __init__(self, output_dir="/home/rokey/dev_ws/datasets", filename_prefix="coupang_demo"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.hdf5")
        self.reset_episode_buffer()
        self.episode_counter = 0
        self.is_recording = False
        print(f"[VR Logger] 데이터 로거 초기화. 저장 경로: {self.filepath}")

    def reset_episode_buffer(self):
        self.buffer = {
            "obs/robot_pose": [], # 로봇 본체의 위치+회전 (월드 기준)
            "obs/joint_positions": [],
            "obs/joint_velocities": [],
            "obs/box_pose": [],  # 상자 위치+회전 정보 추가
            "obs/rack_pose": [], # 작업대 위치+회전 정보 추가
            "obs/images/Left Camera": [],
            "obs/images/Right Camera": [],
            "obs/images/TopView": [],
            "actions/joint_targets": [],
            "actions/cmd_vel": [], # 이동 대차(모바일 베이스) 조작 명령 추가
            "rewards": [],
            "dones": []
        }
        
    def start_recording(self):
        if not self.is_recording:
            self.reset_episode_buffer()
            self.is_recording = True
            print(f"[VR Logger] 🔴 에피소드 {self.episode_counter} 녹화 시작!")

    def stop_recording_and_save(self):
        if self.is_recording and len(self.buffer["actions/joint_targets"]) > 0:
            self._save_episode_to_hdf5()
            print(f"[VR Logger] ⬛ 에피소드 {self.episode_counter} 저장 완료 (스텝 수: {len(self.buffer['actions/joint_targets'])})")
            self.episode_counter += 1
            self.is_recording = False

    def cancel_recording(self):
        if self.is_recording:
            print(f"[VR Logger] 🗑️ 에피소드 {self.episode_counter} 녹화 취소 (저장하지 않고 버림)")
            self.is_recording = False
            self.reset_episode_buffer()

    def log_step(self, robot_pose, joint_pos, joint_vel, action_target, cmd_vel, box_pose, rack_pose, images=None, reward=0.0, done=False):
        if not self.is_recording:
            return
        self.buffer["obs/robot_pose"].append(np.array(robot_pose, dtype=np.float32))
        self.buffer["obs/joint_positions"].append(np.array(joint_pos, dtype=np.float32))
        self.buffer["obs/joint_velocities"].append(np.array(joint_vel, dtype=np.float32))
        self.buffer["obs/box_pose"].append(np.array(box_pose, dtype=np.float32))
        self.buffer["obs/rack_pose"].append(np.array(rack_pose, dtype=np.float32))
        self.buffer["actions/joint_targets"].append(np.array(action_target, dtype=np.float32))
        self.buffer["actions/cmd_vel"].append(np.array(cmd_vel, dtype=np.float32))
        self.buffer["rewards"].append(np.float32(reward))
        self.buffer["dones"].append(bool(done))
        
        if images:
            for cam_name in ["Left Camera", "Right Camera", "TopView"]:
                if cam_name in images:
                    self.buffer[f"obs/images/{cam_name}"].append(images[cam_name])
                else:
                    self.buffer[f"obs/images/{cam_name}"].append(np.zeros((120, 160, 3), dtype=np.uint8))


    def _save_episode_to_hdf5(self):
        with h5py.File(self.filepath, "a") as f:
            if "data" not in f:
                data_grp = f.create_group("data")
            else:
                data_grp = f["data"]
            ep_grp = data_grp.create_group(f"demo_{self.episode_counter}")
            ep_grp.attrs["num_samples"] = len(self.buffer["actions/joint_targets"])
            obs_grp = ep_grp.create_group("obs")
            obs_grp.create_dataset("robot_pose", data=np.array(self.buffer["obs/robot_pose"]))
            obs_grp.create_dataset("joint_positions", data=np.array(self.buffer["obs/joint_positions"]))
            obs_grp.create_dataset("joint_velocities", data=np.array(self.buffer["obs/joint_velocities"]))
            obs_grp.create_dataset("box_pose", data=np.array(self.buffer["obs/box_pose"]))
            obs_grp.create_dataset("rack_pose", data=np.array(self.buffer["obs/rack_pose"]))
            
            img_grp = obs_grp.create_group("images")
            for cam_name in ["Left Camera", "Right Camera", "TopView"]:
                if len(self.buffer[f"obs/images/{cam_name}"]) > 0:
                    img_data = np.stack(self.buffer[f"obs/images/{cam_name}"])
                    # Use gzip compression for image datasets to save space
                    img_grp.create_dataset(cam_name, data=img_data, compression="gzip")
                    
            ep_grp.create_dataset("actions", data=np.array(self.buffer["actions/joint_targets"]))
            ep_grp.create_dataset("cmd_vel", data=np.array(self.buffer["actions/cmd_vel"]))
            ep_grp.create_dataset("rewards", data=np.array(self.buffer["rewards"]))
            ep_grp.create_dataset("dones", data=np.array(self.buffer["dones"]))

