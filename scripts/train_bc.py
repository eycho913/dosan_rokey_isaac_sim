import os
import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 1-a. 자동 단계 감지 함수 (Manual labeling 불필요!)
# ============================================================================
# 데이터의 물리적 신호(lift, 손가락, cmd_vel)로 현재 에피소드의 단계를 자동 구분
# 5개 단계: approach(0), grasp(1), lift_up(2), transport(3), place(4)
def auto_detect_phase(joint_pos_seq, cmd_vel_seq):
    """
    에피소드 전체 시퀀스를 분석해 각 프레임의 단계(0~4) 반환
    
    단계 판별 기준:
      0 (approach) : 초기 상태 ~ 손가락 닫힘 전
      1 (grasp)    : 손가락이 닫힌 순간 ~ 들어올리기 전
      2 (lift_up)  : lift가 상승하기 시작 ~ 이동 전
      3 (transport): cmd_vel이 활성화(이동/회전)
      4 (place)    : 손가락 열림 ~ 에피소드 끝
    """
    N = len(joint_pos_seq)
    phases = np.zeros(N, dtype=np.int64)
    
    lift = joint_pos_seq[:, 62]                    # lift_joint
    finger_r = joint_pos_seq[:, 40:60].mean(axis=1) # 오른손 평균
    
    # 손가락 닫힘 감지: 평균값이 에피소드 전체 중앙값 이상이면 "쥐고 있음"
    finger_median = np.median(finger_r)
    is_grasping = finger_r > finger_median
    
    # 이동 감지: cmd_vel이 0이 아니면 "이동 중"
    is_moving = (np.abs(cmd_vel_seq[:, 0]) > 0.01) | (np.abs(cmd_vel_seq[:, 2]) > 0.01)
    
    # lift 상승 감지: 이전 프레임 대비 증가
    lift_diff = np.diff(lift, prepend=lift[0])
    is_lifting = lift_diff > 0.001
    
    # 단계 할당 (순차적 상태 머신)
    first_grasp = -1
    first_lift = -1
    first_move = -1
    first_release = -1
    
    for i in range(N):
        if first_grasp < 0 and is_grasping[i]:
            first_grasp = i
        if first_grasp >= 0 and first_lift < 0 and is_lifting[i]:
            first_lift = i
        if first_lift >= 0 and first_move < 0 and is_moving[i]:
            first_move = i
        if first_move >= 0 and first_release < 0 and not is_grasping[i]:
            first_release = i
    
    # 각 구간에 단계 번호 할당
    for i in range(N):
        if first_release >= 0 and i >= first_release:
            phases[i] = 4  # place
        elif first_move >= 0 and i >= first_move:
            phases[i] = 3  # transport
        elif first_lift >= 0 and i >= first_lift:
            phases[i] = 2  # lift_up
        elif first_grasp >= 0 and i >= first_grasp:
            phases[i] = 1  # grasp
        else:
            phases[i] = 0  # approach
    
    return phases

NUM_PHASES = 5  # one-hot 백터 크기

# ============================================================================
# 1-b. Dataset Class: HDF5에서 데이터 추출
# ============================================================================
class SH5DemonstrationDataset(Dataset):
    def __init__(self, dataset_dir):
        self.states = []
        self.actions = []
        
        # 지정된 폴더 안의 모든 hdf5 파일 검색
        hdf5_files = glob.glob(os.path.join(dataset_dir, "*.hdf5"))
        if not hdf5_files:
            print(f"[경고] {dataset_dir}에 데이터가 없습니다!")
            return
            
        print(f"[INFO] 총 {len(hdf5_files)}개의 데이터 파일을 찾았습니다.")
        
        # 파일 순회하며 데이터 추출
        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' not in f:
                        continue
                    
                    # 각 에피소드(demo_0, demo_1 ...) 순회
                    for demo_name in f['data'].keys():
                        demo = f['data'][demo_name]
                        
                        # 관측(Observation) 데이터 로드
                        obs = demo['obs']
                        robot_pose = obs['robot_pose'][:]      # (N, 7)
                        box_pose = obs['box_pose'][:]          # (N, 7)
                        rack_pose = obs['rack_pose'][:]        # (N, 7)
                        joint_pos = obs['joint_positions'][:]  # (N, 63)
                        joint_vel = obs['joint_velocities'][:] # (N, 63)
                        
                        # 정답(Action) 데이터 로드
                        joint_targets = demo['actions'][:]     # (N, 63)
                        cmd_vel = demo['cmd_vel'][:]           # (N, 3)
                        
                        num_samples = robot_pose.shape[0]
                        
                        # 자동 단계 감지 (에피소드 전체를 분석하여 각 프레임의 단계 할당)
                        phases = auto_detect_phase(joint_pos, cmd_vel)
                        
                        for i in range(num_samples):
                            # 진행률에 100을 곱해서(0~100) 스케일을 키움
                            progress = np.array([(i / num_samples) * 100.0], dtype=np.float32)
                            
                            # 단계 one-hot 벡터 (5차원)
                            phase_onehot = np.zeros(NUM_PHASES, dtype=np.float32)
                            phase_onehot[phases[i]] = 1.0
                            
                            # State 이어붙이기 (총 153 차원: 기존 147 + 진행률 1 + 단계 5)
                            state = np.concatenate([
                                robot_pose[i], box_pose[i], rack_pose[i], 
                                joint_pos[i], joint_vel[i], progress, phase_onehot
                            ])
                            
                            # Action 이어붙이기 (총 66 차원)
                            action = np.concatenate([
                                joint_targets[i], cmd_vel[i]
                            ])
                            
                            self.states.append(state)
                            self.actions.append(action)
            except Exception as e:
                print(f"[ERROR] 파일 {file_path} 읽기 실패: {e}")
                
        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        print(f"[INFO] 총 {len(self.states)}개의 학습 샘플(프레임)이 준비되었습니다.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return torch.tensor(self.states[idx]), torch.tensor(self.actions[idx])


# ============================================================================
# 2. Neural Network Model (Behavior Cloning Policy)
# ============================================================================
class BehaviorCloningPolicy(nn.Module):
    def __init__(self, state_dim=153, action_dim=66):  # 148 + 5(phase one-hot) = 153
        super().__init__()
        # 3계층 MLP (다층 퍼셉트론) 구조
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 3. Training Loop
# ============================================================================
def train(data_dir, output_name):
    dataset = SH5DemonstrationDataset(dataset_dir=data_dir)
    if len(dataset) == 0:
        print(f"[ERROR] {data_dir} 에 학습할 데이터가 없습니다.")
        return

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 모델, 손실함수, 최적화 기법 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloningPolicy().to(device)
    criterion = nn.MSELoss() # 평균 제곱 오차 (모방학습 기본)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 500
    print(f"[INFO] 학습을 시작합니다! (장치: {device}, 에포크: {epochs})")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_states, batch_actions in dataloader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            
            # 입력 데이터에 미세한 노이즈 추가 (Data Augmentation - OOD 표류 방지)
            noise = torch.randn_like(batch_states) * 0.005
            noisy_states = batch_states + noise
            
            # 예측 (Forward)
            predictions = model(noisy_states)
            
            # 오차 계산 (Loss) - 관절별 가중치 적용
            pred_joints = predictions[:, :63]
            pred_cmd = predictions[:, 63:]
            target_joints = batch_actions[:, :63]
            target_cmd = batch_actions[:, 63:]
            
            # ============================================================================
            # 관절별 Loss 가중치 (실제 Isaac Sim joint_names 순서 기준)
            # ============================================================================
            # swerve steering : [0, 1, 2]   → 가중치 0.5
            # lift_joint      : [3]         → 가중치 50.0 (핵심!)
            # swerve drive    : [4, 5, 6]   → 가중치 0.5
            # head_joint1,2   : [7, 10]     → 가중치 0.3
            # arm_L (1~7)     : [8,11,13,15,17,19,21]  ← arm_L/R 인터리브!
            # arm_R (1~7)     : [9,12,14,16,18,20,22]
            # finger_L (20)   : [23-27,33-37,43-47,53-57]
            # finger_R (20)   : [28-32,38-42,48-52,58-62]
            # ============================================================================
            joint_weights = torch.ones(63, device=device)
            
            # 공통
            for i in [0,1,2,4,5,6]: joint_weights[i] = 0.5   # swerve
            joint_weights[3] = 50.0                           # lift_joint (idx=3)
            for i in [7, 10]:       joint_weights[i] = 0.3   # head
            
            arm_l    = [8,11,13,15,17,19,21]
            arm_r    = [9,12,14,16,18,20,22]
            finger_l = [23,24,25,26,27,33,34,35,36,37,43,44,45,46,47,53,54,55,56,57]
            finger_r = [28,29,30,31,32,38,39,40,41,42,48,49,50,51,52,58,59,60,61,62]
            
            # 오른손 피킹 (slot1, 3) 기준 - 왼팔/왼손 마스킹
            for i in arm_l:    joint_weights[i] = 0.0
            for i in arm_r:    joint_weights[i] = 1.0
            for i in finger_l: joint_weights[i] = 0.0
            for i in finger_r: joint_weights[i] = 1.0
            
            # 가중치 적용 MSE: (pred - target)^2 * weight, 관절별 평균
            joint_diff_sq = (pred_joints - target_joints) ** 2  # (batch, 63)
            loss_joints = (joint_diff_sq * joint_weights.unsqueeze(0)).mean()
            loss_cmd = criterion(pred_cmd, target_cmd)
            
            # 바퀴 이동(cmd_vel) 100배 중요
            loss = loss_joints + (100.0 * loss_cmd)
            
            # 역전파 (Backward) 및 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
            
    # 학습 완료 후 모델 저장
    os.makedirs("/home/rokey/dev_ws/models", exist_ok=True)
    save_path = f"/home/rokey/dev_ws/models/{output_name}"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 학습 완료! 인공지능 모델이 저장되었습니다: {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Behavior Cloning Policy")
    parser.add_argument("--data_dir", type=str, required=True, help="학습할 HDF5 파일들이 있는 폴더 경로 (예: datasets/slot_1)")
    parser.add_argument("--output", type=str, required=True, help="저장할 모델 파일 이름 (예: slot_1_policy.pth)")
    args = parser.parse_args()
    
    train(args.data_dir, args.output)
