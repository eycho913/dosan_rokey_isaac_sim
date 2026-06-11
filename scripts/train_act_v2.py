#!/usr/bin/env python3
"""
=============================================================================
 SH5 로봇 모방학습 - Vision-Based ACT (Action Chunking with Transformers)
=============================================================================
 양손 카메라 및 탑뷰 카메라 이미지를 포함한 상태값을 입력으로 사용합니다.
=============================================================================
"""

import argparse
import glob
import math
import os
from tqdm import tqdm

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


NUM_PHASES = 5
_LIFT_IDX      = 3
_FINGER_L_IDX  = [23, 24, 25, 26, 27, 33, 34, 35, 36, 37, 43, 44, 45, 46, 47, 53, 54, 55, 56, 57]
_FINGER_R_IDX  = [28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 48, 49, 50, 51, 52, 58, 59, 60, 61, 62]

def auto_detect_phase(joint_pos_seq, cmd_vel_seq):
    N = len(joint_pos_seq)
    phases = np.zeros(N, dtype=np.int64)
    lift = joint_pos_seq[:, _LIFT_IDX]
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
        if first_grasp < 0 and is_grasping[i]: first_grasp = i
        if first_grasp >= 0 and first_lift < 0 and is_lifting[i]: first_lift = i
        if first_lift >= 0 and first_move < 0 and is_moving[i]: first_move = i
        if first_move >= 0 and first_release < 0 and not is_grasping[i]: first_release = i
    
    for i in range(N):
        if first_release >= 0 and i >= first_release: phases[i] = 4
        elif first_move >= 0 and i >= first_move: phases[i] = 3
        elif first_lift >= 0 and i >= first_lift: phases[i] = 2
        elif first_grasp >= 0 and i >= first_grasp: phases[i] = 1
        else: phases[i] = 0
    return phases

# ── [만능 모델] 슬롯별 타겟 좌표 정의 ──
SLOT_TARGETS = {
    1: np.array([-0.24, -1.25, 1.38], dtype=np.float32),
    2: np.array([ 0.24, -1.25, 1.38], dtype=np.float32),
    3: np.array([-0.24, -1.25, 0.65], dtype=np.float32),
    4: np.array([ 0.24, -1.25, 0.65], dtype=np.float32),
}

STATE_DIM = 156  # 기존 153 + 3 (목표 좌표)
ACTION_DIM = 66

class VisionACTSequenceDataset(Dataset):
    def __init__(self, dataset_dir, context_len=10, chunk_size=20):
        self.context_len = context_len
        self.chunk_size = chunk_size
        self.episodes = [] 
        self.hdf5_handles = {} # Keep files open for lazy image loading
        
        if os.path.isdir(dataset_dir):
            hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.hdf5"), recursive=True))
        else:
            hdf5_files = [dataset_dir] if dataset_dir.endswith('.hdf5') else []
            
        if not hdf5_files:
            print(f"[경고] {dataset_dir}에 해당하는 데이터 파일이 없습니다!")
            return
            
        total_samples = 0
        for file_path in hdf5_files:
            # 파일 이름에서 슬롯 번호 추출 (만능 모델용)
            filename = os.path.basename(file_path).lower()
            slot_id = 1
            if "slot2" in filename: slot_id = 2
            elif "slot3" in filename: slot_id = 3
            elif "slot4" in filename: slot_id = 4
            elif "slot1" in filename: slot_id = 1
            else:
                print(f"[경고] {filename} 에서 슬롯 번호를 찾을 수 없어 기본값(Slot 1) 적용")
                
            target_coord = SLOT_TARGETS[slot_id]
            
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' not in f:
                        continue
                        
                    import random
                    demo_keys = list(f['data'].keys())
                    # 모든 데이터를 로드하도록 제한 해제 (총 800개 데이터 사용 가능)
                    import random
                    random.shuffle(demo_keys)
                        
                    for demo_name in demo_keys:
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
                            continue
                            
                        phases = auto_detect_phase(joint_pos, cmd_vel)
                        
                        states = []
                        actions = []
                        for i in range(N):
                            progress = np.array([(i / N) * 100.0], dtype=np.float32)
                            phase_oh = np.zeros(NUM_PHASES, dtype=np.float32)
                            phase_oh[phases[i]] = 1.0
                            
                            state = np.concatenate([
                                robot_pose[i], box_pose[i], rack_pose[i],
                                joint_pos[i], joint_vel[i], progress, phase_oh,
                                target_coord
                            ])
                            action = np.concatenate([joint_targets[i], cmd_vel[i]])
                            states.append(state)
                            actions.append(action)
                            
                        states = np.array(states, dtype=np.float32)
                        actions = np.array(actions, dtype=np.float32)
                        
                        self.episodes.append((file_path, demo_name, states, actions, phases, slot_id))
            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")
                
        self.index_map = []
        for ep_idx, (_, _, states, _, phases, _) in enumerate(self.episodes):
            N = len(states)
            for t in range(self.context_len, N - self.chunk_size + 1):
                self.index_map.append((ep_idx, t, int(phases[t])))
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        ep_idx, t, phase = self.index_map[idx]
        file_path, demo_name, states, actions, phases, slot_id = self.episodes[ep_idx]
        
        state_seq = states[t - self.context_len : t]
        action_chunk = actions[t : t + self.chunk_size]
        
        # HDF5 Lazy Loading (worker-safe)
        if not hasattr(self, 'local_handles'):
            self.local_handles = {}
        if file_path not in self.local_handles:
            self.local_handles[file_path] = h5py.File(file_path, 'r')
            
        f = self.local_handles[file_path]
        img_grp = f['data'][demo_name]['obs']['images']
        
        # Extract images at t-1 (the most recent frame in context)
        img_left = img_grp['Left Camera'][t-1]
        img_right = img_grp['Right Camera'][t-1]
        img_top = img_grp['TopView'][t-1]
        
        img_left_t = self.transform(img_left)
        img_right_t = self.transform(img_right)
        img_top_t = self.transform(img_top)
        
        images = torch.stack([img_left_t, img_right_t, img_top_t], dim=0) # (3, 3, H, W)
        
        return torch.tensor(state_seq), torch.tensor(action_chunk), torch.tensor(phase, dtype=torch.long), images, torch.tensor(slot_id, dtype=torch.long)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VisionACTPolicy(nn.Module):
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
        
        # ---- Vision Encoder (ResNet18) ----
        resnet = models.resnet18(pretrained=True)
        # Remove fc layer
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        # ResNet18 output dim is 512. 3 cameras = 1536
        self.vision_proj = nn.Linear(512 * 3, hidden_dim)
        
        # ---- State Encoder ----
        # Now input combines state + vision features
        self.state_proj = nn.Linear(state_dim + hidden_dim, hidden_dim)
        self.pos_enc = SinusoidalPositionEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)
        
        # ---- CVAE Encoder ----
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        cvae_enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.cvae_encoder = nn.TransformerEncoder(cvae_enc_layer, num_layers=2)
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # ---- Action Decoder ----
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_query = nn.Embedding(chunk_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_dec_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Goal Conditioning (Slot ID)
        self.slot_emb = nn.Embedding(num_embeddings=10, embedding_dim=hidden_dim)
    
    def extract_vision_features(self, images):
        """
        images: (B, 3, 3, 120, 160)
        Returns: (B, hidden_dim)
        """
        B = images.shape[0]
        # Flatten batch and cameras: (B*3, 3, 120, 160)
        imgs_flat = images.view(B*3, 3, 120, 160)
        feats = self.vision_backbone(imgs_flat) # (B*3, 512, 1, 1)
        feats = feats.view(B, 3, 512).view(B, 3*512) # (B, 1536)
        return self.vision_proj(feats) # (B, hidden_dim)
        
    def encode_state(self, state_seq, img_feats):
        """
        state_seq: (B, context_len, state_dim)
        img_feats: (B, hidden_dim)
        We concatenate img_feats to every timestep in state_seq.
        """
        B, T, _ = state_seq.shape
        img_feats_exp = img_feats.unsqueeze(1).expand(B, T, -1) # (B, T, hidden_dim)
        combined = torch.cat([state_seq, img_feats_exp], dim=-1) # (B, T, state_dim + hidden_dim)
        
        x = self.state_proj(combined)
        x = self.pos_enc(x)
        return self.state_encoder(x)
    
    def encode_cvae(self, state_memory, action_chunk):
        B = state_memory.size(0)
        action_tokens = self.action_proj(action_chunk)
        cls = self.cls_token.expand(B, -1, -1)
        combined = torch.cat([cls, state_memory, action_tokens], dim=1)
        encoded = self.cvae_encoder(combined)
        cls_out = encoded[:, 0]
        mu = self.latent_mu(cls_out)
        logvar = self.latent_logvar(cls_out)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode_actions(self, state_memory, z):
        B = state_memory.size(0)
        z_token = self.latent_proj(z).unsqueeze(1)
        memory = torch.cat([z_token, state_memory], dim=1)
        queries = self.action_query.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.action_decoder(queries, memory)
        actions = self.action_head(decoded)
        return actions
    
    def forward(self, state_seq, images, slot_ids, action_chunk=None):
        img_feats = self.extract_vision_features(images)
        
        # Add Goal Embedding
        goal_feats = self.slot_emb(slot_ids) # (B, hidden_dim)
        img_feats = img_feats + goal_feats
        
        state_memory = self.encode_state(state_seq, img_feats)
        
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


def make_batch_joint_weights(device, slot_ids):
    """
    만능 모델용 동적 가중치 생성 함수.
    배치 안의 각 데이터가 속한 슬롯 번호에 따라 비활성 팔의 loss 가중치를 낮춥니다.
    slot_ids: (B,) tensor
    Returns: (B, ACTION_DIM) tensor
    """
    B = len(slot_ids)
    w = torch.ones((B, ACTION_DIM), device=device)
    
    # 공통 가중치
    w[:, [0, 1, 2, 4, 5, 6]] = 0.5
    w[:, [7, 10]] = 0.3
    w[:, 3] = 50.0
    w[:, 63] = 20.0
    w[:, 64] = 30.0
    w[:, 65] = 15.0
    
    arm_l    = [8, 11, 13, 15, 17, 19, 21]
    arm_r    = [9, 12, 14, 16, 18, 20, 22]
    finger_l = [23,24,25,26,27,33,34,35,36,37,43,44,45,46,47,53,54,55,56,57]
    finger_r = [28,29,30,31,32,38,39,40,41,42,48,49,50,51,52,58,59,60,61,62]
    
    for b in range(B):
        s = slot_ids[b].item()
        if s in (2, 4):  # 왼팔 활성화
            w[b, arm_l] = 1.0
            w[b, finger_l] = 1.0
            w[b, arm_r] = 0.1
            w[b, finger_r] = 0.1
        else:  # 오른팔 활성화 (1, 3)
            w[b, arm_l] = 0.1
            w[b, finger_l] = 0.1
            w[b, arm_r] = 1.0
            w[b, finger_r] = 1.0
    return w


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = VisionACTSequenceDataset(
        dataset_dir=args.data,
        context_len=args.context_len,
        chunk_size=args.chunk_size,
    )
    if len(dataset) == 0:
        print("[ERROR] 학습할 데이터가 없습니다.")
        return
        
    # [안전] HDF5 다중 프로세스 충돌 방지: num_workers=0 (단일 스레드로 순차 로드)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    
    model = VisionACTPolicy(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim,
        chunk_size=args.chunk_size,
        latent_dim=args.latent_dim,
    ).to(device)
    
    # [Fine-tune] 사전 학습 모델에서 가중치 로드
    if args.pretrain:
        pretrain_path = os.path.join('/home/rokey/dev_ws/models', args.pretrain)
        if os.path.exists(pretrain_path):
            ckpt = torch.load(pretrain_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"[Fine-tune] 사전 학습 모델 로드 완료: {pretrain_path}")
        else:
            print(f"[WARNING] pretrain 파일 없음: {pretrain_path}, 처음부터 학습")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Vision ACT 모델 파라미터 수: {param_count:,}")
    
    # Fine-tune 시 학습률 조정
    effective_lr = args.finetune_lr if args.pretrain else args.lr
    print(f"[INFO] 학습률: {effective_lr} ({'fine-tune' if args.pretrain else 'scratch'})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    kl_weight_max = args.kl_weight
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        kl_weight = min(1.0, epoch / (args.epochs * 0.2)) * kl_weight_max
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        for batch_states, batch_actions, batch_phases, batch_images, batch_slot_ids in pbar:
            batch_states   = batch_states.to(device)
            batch_actions  = batch_actions.to(device)
            batch_phases   = batch_phases.to(device)
            batch_images   = batch_images.to(device)
            batch_slot_ids = batch_slot_ids.to(device)
            
            # 동적 가중치 생성 (배치마다 파일 이름 기반 슬롯 ID로 계산)
            joint_weights = make_batch_joint_weights(device, batch_slot_ids)
            
            predicted, mu, logvar = model(batch_states, batch_images, batch_slot_ids, batch_actions)
            
            diff_sq = (predicted - batch_actions) ** 2
            weighted = diff_sq * joint_weights.unsqueeze(1)  # (B, chunk_size, ACTION_DIM)
            
            phase_w = torch.ones(len(batch_phases), device=device)
            phase_w[batch_phases == 1] = 3.0
            phase_w[batch_phases == 4] = 3.0
            phase_w[batch_phases == 2] = 1.5
            
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
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{recon_loss.item():.4f}"
            })
        
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
            'is_unified_model': True, # 만능 모델 마커
        }
    }
    torch.save(save_dict, output_path)
    print(f"\n🎉 Unified Vision ACT 학습 완료! 모델 저장: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SH5 Unified Vision ACT 학습 (만능 모델)")
    parser.add_argument("--data", type=str, required=True, help="HDF5 데이터 디렉토리 (vision_ 포함)")
    parser.add_argument("--output", type=str, default="unified_vision_act.pth", help="모델 저장 파일명")
    parser.add_argument('--pretrain', type=str, default='', help='Fine-tune 시 기존 모델 파일명 (models/ 내 상대 경로)')
    parser.add_argument('--finetune_lr', type=float, default=1e-5, help='Fine-tune 학습률 (일반적으로 lr의 1/10)')
    parser.add_argument('--epochs', type=int, default=500, help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기 (메모리 부족시 줄임)')
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률 (Adam)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--chunk_size", type=int, default=20, help="미래 액션 수")
    parser.add_argument("--context_len", type=int, default=10, help="과거 프레임 수")
    parser.add_argument("--latent_dim", type=int, default=32, help="CVAE latent 차원")
    parser.add_argument("--kl_weight", type=float, default=10.0, help="KL 가중치")
    # --slot 인자는 더 이상 필요하지 않습니다 (파일 이름에서 자동 추출)
    args = parser.parse_args()
    
    train(args)
