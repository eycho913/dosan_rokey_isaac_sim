#!/usr/bin/env python3
"""
=============================================================================
 SH5 Vision ACT - Google Colab 학습용 버전 (train_act_v2_col.py)
=============================================================================
 [Colab 전용 추가 기능]
  1. Google Drive 마운트 + 경로 자동 설정
  2. N 에포크마다 체크포인트 자동 저장 (세션 끊김 대비)
  3. 체크포인트에서 학습 재개 (--resume)
  4. 기존 모델 이어받기 (--pretrain)
  5. batch_size 기본값 64 (A100/H100 최적화)

 [Colab 실행 방법]
  ① Google Drive 마운트 후 HDF5 데이터와 기존 .pth 모델을 Drive에 업로드
  ② 아래 셀을 실행:

    !pip install h5py tqdm -q

    # 새로 학습 (처음부터):
    !python3 /content/train_act_v2_col.py \
        --data /content/drive/MyDrive/sh5_data/subset_80ep.hdf5 \
        --output /content/drive/MyDrive/sh5_models/ft100.pth \
        --epochs 100 --batch_size 64

    # 기존 모델 이어서 학습 (권장):
    !python3 /content/train_act_v2_col.py \
        --data /content/drive/MyDrive/sh5_data/subset_80ep.hdf5 \
        --output /content/drive/MyDrive/sh5_models/ft100.pth \
        --pretrain /content/drive/MyDrive/sh5_models/augmented_sh5_vision_act_20ep.pth \
        --epochs 100 --batch_size 64 --finetune_lr 2e-5

    # 중단된 학습 재개:
    !python3 /content/train_act_v2_col.py \
        --data /content/drive/MyDrive/sh5_data/subset_80ep.hdf5 \
        --output /content/drive/MyDrive/sh5_models/ft100.pth \
        --resume /content/drive/MyDrive/sh5_models/checkpoint_latest.pth \
        --epochs 100 --batch_size 64
=============================================================================
"""

import argparse
import glob
import math
import os
import time
from tqdm import tqdm

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────
NUM_PHASES = 5
_LIFT_IDX      = 3
_FINGER_L_IDX  = [23,24,25,26,27,33,34,35,36,37,43,44,45,46,47,53,54,55,56,57]
_FINGER_R_IDX  = [28,29,30,31,32,38,39,40,41,42,48,49,50,51,52,58,59,60,61,62]

SLOT_TARGETS = {
    1: np.array([-0.24, -1.25, 1.38], dtype=np.float32),
    2: np.array([ 0.24, -1.25, 1.38], dtype=np.float32),
    3: np.array([-0.24, -1.25, 0.65], dtype=np.float32),
    4: np.array([ 0.24, -1.25, 0.65], dtype=np.float32),
}

STATE_DIM  = 156
ACTION_DIM = 66


# ─────────────────────────────────────────────────────────────
# Phase 자동 감지
# ─────────────────────────────────────────────────────────────
def auto_detect_phase(joint_pos_seq, cmd_vel_seq):
    N = len(joint_pos_seq)
    phases = np.zeros(N, dtype=np.int64)
    lift       = joint_pos_seq[:, _LIFT_IDX]
    finger_l   = joint_pos_seq[:, _FINGER_L_IDX].mean(axis=1)
    finger_r   = joint_pos_seq[:, _FINGER_R_IDX].mean(axis=1)
    finger_mean = np.maximum(finger_l, finger_r)

    finger_median = np.median(finger_mean)
    is_grasping = finger_mean > finger_median
    is_moving   = (np.abs(cmd_vel_seq[:, 0]) > 0.01) | (np.abs(cmd_vel_seq[:, 2]) > 0.01)
    lift_diff   = np.diff(lift, prepend=lift[0])
    is_lifting  = lift_diff > 0.001

    fg = fl = fm = fr = -1
    for i in range(N):
        if fg < 0 and is_grasping[i]:            fg = i
        if fg >= 0 and fl < 0 and is_lifting[i]: fl = i
        if fl >= 0 and fm < 0 and is_moving[i]:  fm = i
        if fm >= 0 and fr < 0 and not is_grasping[i]: fr = i

    for i in range(N):
        if fr >= 0 and i >= fr: phases[i] = 4
        elif fm >= 0 and i >= fm: phases[i] = 3
        elif fl >= 0 and i >= fl: phases[i] = 2
        elif fg >= 0 and i >= fg: phases[i] = 1
    return phases


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class VisionACTSequenceDataset(Dataset):
    def __init__(self, dataset_dir, context_len=10, chunk_size=20):
        self.context_len = context_len
        self.chunk_size  = chunk_size
        self.episodes    = []

        if os.path.isdir(dataset_dir):
            hdf5_files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.hdf5"), recursive=True))
        else:
            hdf5_files = [dataset_dir] if dataset_dir.endswith('.hdf5') else []

        if not hdf5_files:
            print(f"[경고] {dataset_dir} 에 HDF5 파일이 없습니다!")
            return

        for file_path in hdf5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' not in f:
                        continue
                    for demo_name in f['data'].keys():
                        try:
                            demo = f['data'][demo_name]
                            obs  = demo['obs']
                            
                            # 슬롯 ID를 파일명이 아닌 속성에서 가져오기 (병합본 완벽 대응)
                            slot_id = int(demo.attrs.get('slot_id', 1))
                            target_coord = SLOT_TARGETS.get(slot_id, SLOT_TARGETS[1])
                            
                            joint_pos = obs['joint_positions'][:]
                            N = joint_pos.shape[0]
                            
                            if N < context_len + chunk_size:
                                continue
                                
                            # 일부 데이터 누락 시 프로그램이 죽지 않도록 0 배열로 대체
                            robot_pose    = obs['robot_pose'][:] if 'robot_pose' in obs else np.zeros((N, 7), dtype=np.float32)
                            box_pose      = obs['box_pose'][:] if 'box_pose' in obs else np.zeros((N, 7), dtype=np.float32)
                            rack_pose     = obs['rack_pose'][:] if 'rack_pose' in obs else np.zeros((N, 7), dtype=np.float32)
                            joint_vel     = obs['joint_velocities'][:] if 'joint_velocities' in obs else np.zeros_like(joint_pos)
                            joint_targets = demo['actions'][:] if 'actions' in demo else joint_pos.copy()
                            cmd_vel       = demo['cmd_vel'][:] if 'cmd_vel' in demo else np.zeros((N, 6), dtype=np.float32)
                            
                            phases = auto_detect_phase(joint_pos, cmd_vel)
                            states, actions = [], []
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
                            
                            self.episodes.append((
                                file_path, demo_name,
                                np.array(states,  dtype=np.float32),
                                np.array(actions, dtype=np.float32),
                                phases, slot_id
                            ))
                        except Exception as e:
                            print(f"[ERROR] {demo_name} 스킵 (데이터 누락): {e}")
            except Exception as e:
                print(f"[ERROR] {file_path} 열기 실패: {e}")

        self.index_map = []
        for ep_idx, (_, _, states, _, phases, _) in enumerate(self.episodes):
            N = len(states)
            for t in range(self.context_len, N - self.chunk_size + 1):
                self.index_map.append((ep_idx, t, int(phases[t])))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"[Dataset] 에피소드: {len(self.episodes)}, 학습샘플: {len(self.index_map):,}")

    def __len__(self):  return len(self.index_map)

    def __getitem__(self, idx):
        ep_idx, t, phase = self.index_map[idx]
        file_path, demo_name, states, actions, _, slot_id = self.episodes[ep_idx]
        state_seq    = states [t - self.context_len : t]
        action_chunk = actions[t : t + self.chunk_size]

        if not hasattr(self, '_handles'):
            self._handles = {}
        if file_path not in self._handles:
            self._handles[file_path] = h5py.File(file_path, 'r')
        img_grp   = self._handles[file_path]['data'][demo_name]['obs']['images']
        img_left  = img_grp['Left Camera'][t-1]
        img_right = img_grp['Right Camera'][t-1]
        img_top   = img_grp['TopView'][t-1]
        images = torch.stack([self.transform(img_left),
                               self.transform(img_right),
                               self.transform(img_top)], dim=0)  # (3,3,H,W)
        return (torch.tensor(state_seq),
                torch.tensor(action_chunk),
                torch.tensor(phase, dtype=torch.long),
                images,
                torch.tensor(slot_id, dtype=torch.long))


# ─────────────────────────────────────────────────────────────
# Model (기존 train_act_v2.py 와 완전히 동일)
# ─────────────────────────────────────────────────────────────
class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class VisionACTPolicy(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden_dim=256, n_heads=8, n_enc_layers=4, n_dec_layers=4,
                 chunk_size=20, latent_dim=32, dropout=0.1):
        super().__init__()
        self.chunk_size  = chunk_size
        self.latent_dim  = latent_dim
        resnet = models.resnet18(pretrained=True)
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_proj  = nn.Linear(512 * 3, hidden_dim)
        self.state_proj   = nn.Linear(state_dim + hidden_dim, hidden_dim)
        self.pos_enc      = SinusoidalPositionEncoding(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.state_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)
        self.action_proj   = nn.Linear(action_dim, hidden_dim)
        cvae_enc = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.cvae_encoder  = nn.TransformerEncoder(cvae_enc, num_layers=2)
        self.latent_mu     = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        self.cls_token     = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.latent_proj   = nn.Linear(latent_dim, hidden_dim)
        self.action_query  = nn.Embedding(chunk_size, hidden_dim)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, batch_first=True, activation='gelu')
        self.action_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec_layers)
        self.action_head    = nn.Linear(hidden_dim, action_dim)
        self.slot_emb       = nn.Embedding(10, hidden_dim)

    def extract_vision_features(self, images):
        B = images.shape[0]
        feats = self.vision_backbone(images.view(B*3, 3, 120, 160))
        return self.vision_proj(feats.view(B, 3*512))

    def encode_state(self, state_seq, img_feats):
        B, T, _ = state_seq.shape
        combined = torch.cat([state_seq, img_feats.unsqueeze(1).expand(B, T, -1)], dim=-1)
        return self.state_encoder(self.pos_enc(self.state_proj(combined)))

    def forward(self, state_seq, images, slot_ids, action_chunk=None):
        img_feats = self.extract_vision_features(images) + self.slot_emb(slot_ids)
        mem = self.encode_state(state_seq, img_feats)
        if action_chunk is not None:
            B = mem.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            enc = self.cvae_encoder(torch.cat([cls, mem, self.action_proj(action_chunk)], dim=1))
            mu, logvar = self.latent_mu(enc[:,0]), self.latent_logvar(enc[:,0])
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = torch.zeros(mem.size(0), self.latent_dim, device=mem.device)
            mu = logvar = z
        z_tok = self.latent_proj(z).unsqueeze(1)
        q = self.action_query.weight.unsqueeze(0).expand(mem.size(0), -1, -1)
        pred = self.action_head(self.action_decoder(q, torch.cat([z_tok, mem], dim=1)))
        return (pred, mu, logvar) if action_chunk is not None else pred


# ─────────────────────────────────────────────────────────────
# Joint Weights
# ─────────────────────────────────────────────────────────────
def make_batch_joint_weights(device, slot_ids):
    B = len(slot_ids)
    w = torch.ones((B, ACTION_DIM), device=device)
    w[:, [0,1,2,4,5,6]] = 0.5
    w[:, [7,10]] = 0.3
    w[:, 3]  = 50.0
    w[:, 63] = 20.0; w[:, 64] = 30.0; w[:, 65] = 15.0
    arm_l    = [8,11,13,15,17,19,21]
    arm_r    = [9,12,14,16,18,20,22]
    fl_idx   = _FINGER_L_IDX
    fr_idx   = _FINGER_R_IDX
    for b in range(B):
        s = slot_ids[b].item()
        if s in (2, 4):
            w[b, arm_l] = 1.0; w[b, fl_idx] = 1.0
            w[b, arm_r] = 0.1; w[b, fr_idx] = 0.1
        else:
            w[b, arm_r] = 1.0; w[b, fr_idx] = 1.0
            w[b, arm_l] = 0.1; w[b, fl_idx] = 0.1
    return w


# ─────────────────────────────────────────────────────────────
# 체크포인트 저장/로드 헬퍼
# ─────────────────────────────────────────────────────────────
def save_checkpoint(path, model, optimizer, scheduler, epoch, config):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }, path)
    print(f"  💾 체크포인트 저장: {path} (epoch {epoch+1})")


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  ✅ 체크포인트 재개: epoch {start_epoch} 부터 학습")
    return start_epoch


# ─────────────────────────────────────────────────────────────
# 학습 메인
# ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    dataset    = VisionACTSequenceDataset(args.data, args.context_len, args.chunk_size)
    if len(dataset) == 0:
        print("[ERROR] 학습 데이터 없음."); return

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, drop_last=True,
                            num_workers=args.num_workers, pin_memory=True)

    config = dict(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                  hidden_dim=args.hidden_dim, chunk_size=args.chunk_size,
                  context_len=args.context_len, latent_dim=args.latent_dim,
                  is_unified_model=True)

    model = VisionACTPolicy(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        hidden_dim=args.hidden_dim, chunk_size=args.chunk_size,
        latent_dim=args.latent_dim,
    ).to(device)

    effective_lr = args.finetune_lr if (args.pretrain and not args.resume) else args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0

    # 1순위: --resume (중단된 체크포인트)
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device)

    # 2순위: --pretrain (기존 모델 가중치만 이어받기)
    elif args.pretrain and os.path.exists(args.pretrain):
        ckpt = torch.load(args.pretrain, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"[Fine-tune] 기존 모델 로드: {args.pretrain}")
        print(f"[Fine-tune] 학습률: {effective_lr} (원래 lr의 1/5~1/10)")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 파라미터 수: {param_count:,}")

    # 출력 디렉토리 생성
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ckpt_path = args.output.replace('.pth', '_checkpoint_latest.pth')

    kl_weight_max = args.kl_weight
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0
        kl_weight = min(1.0, epoch / max(1, args.epochs * 0.2)) * kl_weight_max

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        for batch_states, batch_actions, batch_phases, batch_images, batch_slot_ids in pbar:
            batch_states   = batch_states.to(device)
            batch_actions  = batch_actions.to(device)
            batch_phases   = batch_phases.to(device)
            batch_images   = batch_images.to(device)
            batch_slot_ids = batch_slot_ids.to(device)

            joint_weights = make_batch_joint_weights(device, batch_slot_ids)
            predicted, mu, logvar = model(batch_states, batch_images, batch_slot_ids, batch_actions)

            diff_sq  = (predicted - batch_actions) ** 2
            weighted = diff_sq * joint_weights.unsqueeze(1)

            phase_w = torch.ones(len(batch_phases), device=device)
            phase_w[batch_phases == 1] = 3.0
            phase_w[batch_phases == 4] = 3.0
            phase_w[batch_phases == 2] = 1.5

            recon_loss = (weighted.mean(dim=(1,2)) * phase_w).mean()
            kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss       = recon_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl    += kl_loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}", Recon=f"{recon_loss.item():.4f}")

        scheduler.step()

        n_batches = len(dataloader)
        elapsed   = (time.time() - t0) / 60
        if (epoch + 1) % 5 == 0 or epoch == 0:
            remaining = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1)
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Loss: {epoch_loss/n_batches:.6f} "
                  f"(Recon: {epoch_recon/n_batches:.6f}, KL: {epoch_kl/n_batches:.6f}) "
                  f"| 경과: {elapsed:.1f}분, 남은시간: {remaining:.1f}분")

        # N 에포크마다 체크포인트 저장 (세션 끊김 대비)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, config)

    # 최종 모델 저장
    torch.save({'model_state_dict': model.state_dict(), 'config': config}, args.output)
    print(f"\n🎉 학습 완료! 최종 모델: {args.output}")


# ─────────────────────────────────────────────────────────────
# 인자 파싱
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SH5 Vision ACT - Colab 버전")
    parser.add_argument("--data",         type=str,   required=True,  help="HDF5 데이터 경로 (파일 or 디렉토리)")
    parser.add_argument("--output",       type=str,   required=True,  help="최종 모델 저장 경로 (.pth)")
    parser.add_argument("--pretrain",     type=str,   default='',     help="이어받을 기존 모델 경로 (가중치만 로드)")
    parser.add_argument("--resume",       type=str,   default='',     help="중단된 체크포인트 경로 (optimizer 포함 완전 재개)")
    parser.add_argument("--epochs",       type=int,   default=100,    help="학습 에포크 수")
    parser.add_argument("--batch_size",   type=int,   default=64,     help="배치 크기 (A100: 64, H100: 128)")
    parser.add_argument("--lr",           type=float, default=1e-4,   help="처음부터 학습할 때 학습률")
    parser.add_argument("--finetune_lr",  type=float, default=2e-5,   help="Fine-tune 학습률")
    parser.add_argument("--hidden_dim",   type=int,   default=256)
    parser.add_argument("--chunk_size",   type=int,   default=20)
    parser.add_argument("--context_len",  type=int,   default=10)
    parser.add_argument("--latent_dim",   type=int,   default=32)
    parser.add_argument("--kl_weight",    type=float, default=10.0)
    parser.add_argument("--save_every",   type=int,   default=5,      help="체크포인트 저장 주기 (에포크)")
    parser.add_argument("--num_workers",  type=int,   default=4,      help="DataLoader worker 수")
    args = parser.parse_args()
    train(args)
