#!/usr/bin/env python3
"""
=============================================================================
 SH5 로봇 모방학습 - ACT (Action Chunking with Transformers) 학습 스크립트
=============================================================================
 사용법:
   # 슬롯 1 (오른손) 학습
   python3 train_act.py --data /home/rokey/dev_ws/datasets/slot1_coupang_demo_20260608_150012.hdf5 --output slot1_act_policy.pth --slot 1 --epochs 500
   
   # 슬롯 2 (왼손) 학습
   python3 train_act.py --data /home/rokey/dev_ws/datasets/slot2_coupang_demo_20260608_172613.hdf5 --output slot2_act_policy.pth --slot 2 --epochs 500
=============================================================================
"""

import argparse
import glob
import math
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 1. 자동 단계 감지
# ============================================================================
NUM_PHASES = 5

# 관절 인덱스 (auto_detect_phase에서 사용)
_LIFT_IDX      = 3
_FINGER_L_IDX  = [23, 24, 25, 26, 27, 33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 53, 54, 55, 56, 57]
_FINGER_R_IDX  = [28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 48, 49, 50, 51, 52, 58, 59, 60, 61, 62]

def auto_detect_phase(joint_pos_seq, cmd_vel_seq):
    """에피소드 시퀀스를 분석하여 각 프레임의 단계(0~4)를 반환"""
    N = len(joint_pos_seq)
    phases = np.zeros(N, dtype=np.int64)
    
    lift = joint_pos_seq[:, _LIFT_IDX]
    # 양손 중 더 활성화된 손으로 파지 감지
    finger_l = joint_pos_seq[:, _FINGER_L_IDX].mean(axis=1)
    finger_r = joint_pos_seq[:, _FINGER_R_IDX].mean(axis=1)
    finger_mean = np.maximum(finger_l, finger_r)
    
    finger_median = np.median(finger_mean)
    is_grasping = finger_mean > finger_median
    is_moving = (np.abs(cmd_vel_seq[:, 0]) > 0.01) | (np.abs(cmd_vel_seq[:, 2]) > 0.01)
    lift_diff = np.diff(lift, prepend=lift[0])
    is_lifting = lift_diff > 0.001
    
    first_grasp = first_lift = first_move = first_release = -1
    for i in range(N):
        if first_grasp < 0 and is_grasping[i]:
            first_grasp = i
        if first_grasp >= 0 and first_lift < 0 and is_lifting[i]:
            first_lift = i
        if first_lift >= 0 and first_move < 0 and is_moving[i]:
            first_move = i
        if first_move >= 0 and first_release < 0 and not is_grasping[i]:
            first_release = i
    
    for i in range(N):
        if first_release >= 0 and i >= first_release:
            phases[i] = 4
        elif first_move >= 0 and i >= first_move:
            phases[i] = 3
        elif first_lift >= 0 and i >= first_lift:
            phases[i] = 2
        elif first_grasp >= 0 and i >= first_grasp:
            phases[i] = 1
        else:
            phases[i] = 0
    return phases


# ============================================================================
# 2. 실제 관절 인덱스 맵 (Isaac Sim joint_names 순서 기준)
# ============================================================================
# swerve steering : [0, 1, 2]
# lift_joint      : [3]
# swerve drive    : [4, 5, 6]
# head_joint1,2   : [7, 10]
# arm_L (1~7)     : [8, 11, 13, 15, 17, 19, 21]  ← arm_L/R 인터리브!
# arm_R (1~7)     : [9, 12, 14, 16, 18, 20, 22]
# finger_L (20)   : [23-27, 33-37, 43-47, 53-57]
# finger_R (20)   : [28-32, 38-42, 48-52, 58-62]

IDX_SWERVE   = [0, 1, 2, 4, 5, 6]
IDX_LIFT     = [3]
IDX_HEAD     = [7, 10]
IDX_ARM_L    = [8, 11, 13, 15, 17, 19, 21]
IDX_ARM_R    = [9, 12, 14, 16, 18, 20, 22]
IDX_FINGER_L = [23, 24, 25, 26, 27, 33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 53, 54, 55, 56, 57]
IDX_FINGER_R = [28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 48, 49, 50, 51, 52, 58, 59, 60, 61, 62]

# 단계 감지: 손가락 인덱스 (파지 감지용)
FINGER_R_MEAN_IDX = IDX_FINGER_R  # 오른손 감지
FINGER_L_MEAN_IDX = IDX_FINGER_L  # 왼손 감지

# ============================================================================
# 3. 시퀀스 데이터셋 (ACT는 과거 N프레임을 입력으로 사용)
# ============================================================================
STATE_DIM = 153
ACTION_DIM = 66  # joint_targets(63) + cmd_vel(3)


class ACTSequenceDataset(Dataset):
    """
    ACT용 시퀀스 데이터셋.
    각 샘플은 (과거 context_len 프레임의 상태, 미래 chunk_size 프레임의 액션) 쌍.
    """
    def __init__(self, dataset_dir, context_len=10, chunk_size=20):
        self.context_len = context_len
        self.chunk_size = chunk_size
        self.episodes = []  # (states_array, actions_array) per episode
        
        # 파일 또는 디렉토리 분기 처리
        if os.path.isdir(dataset_dir):
            hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.hdf5"), recursive=True))
        else:
            hdf5_files = [dataset_dir] if dataset_dir.endswith('.hdf5') else []
            
        if not hdf5_files:
            print(f"[경고] {dataset_dir}에 해당하는 데이터 파일이 없습니다!")
            return
        
        print(f"[INFO] 총 {len(hdf5_files)}개의 데이터 파일을 찾았습니다.")
        
        total_samples = 0
        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' not in f:
                        continue
                    for demo_name in sorted(f['data'].keys()):
                        demo = f['data'][demo_name]
                        obs = demo['obs']
                        
                        robot_pose = obs['robot_pose'][:]
                        box_pose = obs['box_pose'][:]
                        rack_pose = obs['rack_pose'][:]
                        joint_pos = obs['joint_positions'][:]
                        joint_vel = obs['joint_velocities'][:]
                        joint_targets = demo['actions'][:]
                        cmd_vel = demo['cmd_vel'][:]
                        
                        N = robot_pose.shape[0]
                        if N < context_len + chunk_size:
                            print(f"  [스킵] {demo_name}: 길이 {N} < 필요 {context_len + chunk_size}")
                            continue
                        
                        phases = auto_detect_phase(joint_pos, cmd_vel)
                        
                        # 에피소드 전체를 state/action 배열로 변환
                        states = []
                        actions = []
                        for i in range(N):
                            progress = np.array([(i / N) * 100.0], dtype=np.float32)
                            phase_oh = np.zeros(NUM_PHASES, dtype=np.float32)
                            phase_oh[phases[i]] = 1.0
                            
                            state = np.concatenate([
                                robot_pose[i], box_pose[i], rack_pose[i],
                                joint_pos[i], joint_vel[i], progress, phase_oh
                            ])
                            action = np.concatenate([joint_targets[i], cmd_vel[i]])
                            states.append(state)
                            actions.append(action)
                        
                        states = np.array(states, dtype=np.float32)
                        actions = np.array(actions, dtype=np.float32)
                        
                        n_samples = N - context_len - chunk_size + 1
                        total_samples += max(0, n_samples)
                        self.episodes.append((states, actions, phases))  # phases 함께 저장
                        
            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")
        
        # 인덱스 매핑: 글로벌 인덱스 → (에피소드 번호, 프레임 번호, 단계)
        self.index_map = []
        for ep_idx, (states, actions, phases) in enumerate(self.episodes):
            N = len(states)
            for t in range(self.context_len, N - self.chunk_size + 1):
                self.index_map.append((ep_idx, t, int(phases[t])))
        
        print(f"[INFO] 총 {len(self.index_map)}개의 학습 샘플이 준비되었습니다. (에피소드 {len(self.episodes)}개)")
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        ep_idx, t, phase = self.index_map[idx]
        states, actions, phases = self.episodes[ep_idx]
        
        state_seq = states[t - self.context_len : t]   # (context_len, state_dim)
        action_chunk = actions[t : t + self.chunk_size] # (chunk_size, action_dim)
        
        return torch.tensor(state_seq), torch.tensor(action_chunk), torch.tensor(phase, dtype=torch.long)


# ============================================================================
# 3. ACT 모델 아키텍처
# ============================================================================

class SinusoidalPositionEncoding(nn.Module):
    """Transformer용 사인파 위치 인코딩"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ACTPolicy(nn.Module):
    """
    Action Chunking with Transformers (ACT) 정책 네트워크.
    """
    def __init__(
        self,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=256,
        n_heads=8,
        n_enc_layers=4,
        n_dec_layers=4,
        chunk_size=20,
        latent_dim=32,
        dropout=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        
        # ---- State Encoder ----
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.pos_enc = SinusoidalPositionEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)
        
        # ---- CVAE Encoder (학습 시에만 사용) ----
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        cvae_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_enc_layer, num_layers=2)
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # CLS 토큰 (CVAE 인코더의 집약 토큰)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # ---- Action Decoder ----
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_query = nn.Embedding(chunk_size, hidden_dim)  # 학습 가능한 쿼리
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)
    
    def encode_state(self, state_seq):
        """과거 N프레임 상태를 인코딩 → (B, context_len, hidden_dim)"""
        x = self.state_proj(state_seq)
        x = self.pos_enc(x)
        return self.state_encoder(x)
    
    def encode_cvae(self, state_memory, action_chunk):
        """CVAE: 상태+액션에서 latent z 추출 (학습 시에만)"""
        B = state_memory.size(0)
        action_tokens = self.action_proj(action_chunk)  # (B, chunk_size, hidden_dim)
        cls = self.cls_token.expand(B, -1, -1)
        combined = torch.cat([cls, state_memory, action_tokens], dim=1)
        encoded = self.cvae_encoder(combined)
        cls_out = encoded[:, 0]
        mu = self.latent_mu(cls_out)
        logvar = self.latent_logvar(cls_out)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """재매개변수화 트릭: z = mu + std * eps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode_actions(self, state_memory, z):
        """인코딩된 상태 + latent z로부터 chunk_size개의 미래 액션을 디코딩"""
        B = state_memory.size(0)
        z_token = self.latent_proj(z).unsqueeze(1)  # (B, 1, hidden_dim)
        memory = torch.cat([z_token, state_memory], dim=1)  # (B, 1+context_len, hidden_dim)
        queries = self.action_query.weight.unsqueeze(0).expand(B, -1, -1)  # (B, chunk_size, hidden_dim)
        decoded = self.action_decoder(queries, memory)
        actions = self.action_head(decoded)  # (B, chunk_size, action_dim)
        return actions
    
    def forward(self, state_seq, action_chunk=None):
        state_memory = self.encode_state(state_seq)
        if action_chunk is not None:
            mu, logvar = self.encode_cvae(state_memory, action_chunk)
            z = self.reparameterize(mu, logvar)
            predicted = self.decode_actions(state_memory, z)
            return predicted, mu, logvar
        else:
            B = state_seq.size(0)
            z = torch.zeros(B, self.latent_dim, device=state_seq.device)
            predicted = self.decode_actions(state_memory, z)
            return predicted


# ============================================================================
# 4. 관절별 가중치 (실제 인덱스 기반 - 슬롯별 자동화)
# ============================================================================
def make_joint_weights(device, slot_id=1):
    """슬롯 번호(1~4)에 맞춰 양손 관절 마스킹 및 리프트 강조"""
    w = torch.ones(ACTION_DIM, device=device)
    
    # 공통 가중치
    for i in [0, 1, 2, 4, 5, 6]: w[i] = 0.5   # swerve (낮은 비중)
    for i in [7, 10]:             w[i] = 0.3   # head
    w[3]  = 50.0                               # lift_joint (실제 idx=3)
    # cmd_vel 세분화 가중치 (이동 정밀도 강조)
    w[63] = 20.0   # vx: 전진/후진
    w[64] = 30.0   # vy: 좌우 평행이동 (상자 세부 정렬에 핵심) ← 최고 가중치
    w[65] = 15.0   # wz: 회전
    
    arm_l    = [8, 11, 13, 15, 17, 19, 21]
    arm_r    = [9, 12, 14, 16, 18, 20, 22]
    finger_l = [23,24,25,26,27,33,34,35,36,37,43,44,45,46,47,53,54,55,56,57]
    finger_r = [28,29,30,31,32,38,39,40,41,42,48,49,50,51,52,58,59,60,61,62]
    
    if slot_id in (2, 4):
        # 왼손 피킹
        for i in arm_l:    w[i] = 1.0
        for i in arm_r:    w[i] = 0.1   # [수정B] 비활성 팔도 최소 학습 (벨트 충돌 방지)
        for i in finger_l: w[i] = 1.0
        for i in finger_r: w[i] = 0.1   # [수정B] 비활성 손가락도 최소 학습
        print("[INFO] 학습 설정: 왼손 피킹 (슬롯 2, 4) | 오른팔 대기 자세 학습 ON")
    else:
        # 오른손 피킹
        for i in arm_l:    w[i] = 0.1   # [수정B] 비활성 팔도 최소 학습 (벨트 충돌 방지)
        for i in arm_r:    w[i] = 1.0
        for i in finger_l: w[i] = 0.1   # [수정B] 비활성 손가락도 최소 학습
        for i in finger_r: w[i] = 1.0
        print("[INFO] 학습 설정: 오른손 피킹 (슬롯 1, 3) | 왼팔 대기 자세 학습 ON")
    return w


# ============================================================================
# 5. 학습 루프
# ============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ACTSequenceDataset(
        dataset_dir=args.data,
        context_len=args.context_len,
        chunk_size=args.chunk_size,
    )
    if len(dataset) == 0:
        print("[ERROR] 학습할 데이터가 없습니다.")
        return
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    model = ACTPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        chunk_size=args.chunk_size,
        latent_dim=args.latent_dim,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] ACT 모델 파라미터 수: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    joint_weights = make_joint_weights(device, slot_id=args.slot)
    kl_weight_max = args.kl_weight
    
    print(f"[INFO] ACT 학습 시작! (장치: {device}, 에포크: {args.epochs}, "
          f"context: {args.context_len}, chunk: {args.chunk_size})")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        # KL annealing: 첫 20%는 0에서 서서히 올라감
        kl_weight = min(1.0, epoch / (args.epochs * 0.2)) * kl_weight_max
        
        for batch_states, batch_actions, batch_phases in dataloader:
            batch_states   = batch_states.to(device)
            batch_actions  = batch_actions.to(device)
            batch_phases   = batch_phases.to(device)  # (B,)
            
            predicted, mu, logvar = model(batch_states, batch_actions)
            
            diff_sq = (predicted - batch_actions) ** 2
            weighted = diff_sq * joint_weights.unsqueeze(0).unsqueeze(0)  # (B, chunk, action)
            
            # ---- Phase 가중치: 파지(1)·삽입(4) 순간을 3배 강조 ----
            # phase 0=이동전, 1=파지, 2=리프트, 3=이동, 4=삽입/해제
            phase_w = torch.ones(len(batch_phases), device=device)
            phase_w[batch_phases == 1] = 3.0   # 파지 순간 3배
            phase_w[batch_phases == 4] = 3.0   # 슬롯 삽입/해제 순간 3배
            phase_w[batch_phases == 2] = 1.5   # 리프트 순간 1.5배
            
            recon_loss = (weighted.mean(dim=(1,2)) * phase_w).mean()
            
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_weight * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        scheduler.step()
        
        n_batches = len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Loss: {epoch_loss/n_batches:.6f} "
                  f"(Recon: {epoch_recon/n_batches:.6f}, "
                  f"KL: {epoch_kl/n_batches:.6f}, "
                  f"β={kl_weight:.4f})")
    
    output_path = os.path.join("/home/rokey/dev_ws/models", args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': {
            'state_dim': STATE_DIM,
            'action_dim': ACTION_DIM,
            'hidden_dim': args.hidden_dim,
            'chunk_size': args.chunk_size,
            'context_len': args.context_len,
            'latent_dim': args.latent_dim,
            'slot_id': args.slot,
        }
    }
    torch.save(save_dict, output_path)
    print(f"\n🎉 ACT 학습 완료! 모델 저장: {output_path}")


# ============================================================================
# 6. CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SH5 ACT (Action Chunking Transformer) 학습")
    parser.add_argument("--data", type=str, required=True, help="HDF5 데이터 디렉토리 또는 파일 경로")
    parser.add_argument("--output", type=str, default="1_red_act_policy.pth", help="모델 저장 파일명")
    parser.add_argument("--epochs", type=int, default=500, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률 (Adam)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--chunk_size", type=int, default=20, help="한 번에 예측할 미래 액션 수")
    parser.add_argument("--context_len", type=int, default=10, help="참조할 과거 프레임 수")
    parser.add_argument("--latent_dim", type=int, default=32, help="CVAE latent 차원")
    parser.add_argument("--kl_weight", type=float, default=10.0, help="KL divergence 가중치 (β)")
    parser.add_argument("--slot", type=int, default=1, choices=[1, 2, 3, 4], help="대상 슬롯 번호 (1,3: 오른손 피킹 / 2,4: 왼손 피킹)")
    args = parser.parse_args()
    
    train(args)
