# SG2 Logistics Sorting RL Environment

## 프로젝트 개요
본 프로젝트는 DOOSAN M0609 로봇 팔과 FFW SG2 모바일 로봇을 결합한 자율 작업 로봇이 택배 상자를 인식하고, 이를 들어 올려 지정된 색상/ID의 바구니(Bin)에 정확히 분류하는 **강화 학습(Reinforcement Learning) 환경**을 구축하는 것을 목표로 합니다.

학습 환경은 NVIDIA Omniverse 기반의 **Isaac Sim** 및 **IsaacLab** 프레임워크 위에서 구동되며, 로봇은 ArUco 마커 정보와 자신의 관절(Joint) 상태를 관측(Observation)하여 연속적인 제어(Continuous Control)를 통해 목표를 달성합니다.

## 시스템 요구사항 (Troubleshooting)
- **GPU 아키텍처 호환성 (중요):** 현재 환경은 RTX 5080 (Blackwell 아키텍처)에서 구동 중입니다. Isaac Sim 4.5/5.1의 PhysX GPU 파이프라인은 아직 Blackwell과 완벽하게 호환되지 않아 `Segmentation fault` 에러를 발생시킵니다.
- **해결책:** 따라서 물리 시뮬레이션(PhysX)은 CPU 모드로, PPO 신경망 연산은 GPU(cuda:0)로 분리하여 구동되도록 설계되었습니다.

## 실행 방법

### 1. 기본 학습 (터미널 모드 - 권장)
빠른 학습 속도를 원할 경우 헤드리스(Headless) 모드로 실행합니다. 64개의 환경을 동시에 시뮬레이션합니다.
```bash
cd /home/rokey/dev_ws/isaac_sim/IsaacLab
./isaaclab.sh -p /home/rokey/dev_ws/rl_ws/train.py \
  --task Isaac-Sort-FFW-SG2-v0 \
  --num_envs 64 \
  --headless
```

### 2. 시각적 확인 (GUI 모드)
로봇의 움직임을 시각적으로 확인하며 뽕맛(?)을 느끼고 싶을 때 사용합니다.
```bash
cd /home/rokey/dev_ws/isaac_sim/IsaacLab
./isaaclab.sh -p /home/rokey/dev_ws/rl_ws/train.py \
  --task Isaac-Sort-FFW-SG2-v0 \
  --num_envs 64
```

### 3. 원격 라이브 스트리밍 (WebRTC)
다른 기기의 웹 브라우저에서 학습 상황을 3D로 모니터링할 때 사용합니다.
```bash
./isaaclab.sh -p /home/rokey/dev_ws/rl_ws/train.py \
  --task Isaac-Sort-FFW-SG2-v0 \
  --num_envs 64 \
  --livestream 1
```

### 4. 실시간 학습 지표 모니터링 (Tensorboard)
터미널을 하나 더 열고 아래 명령어를 통해 Tensorboard를 실행합니다. 브라우저에서 보상과 패널티 곡선을 실시간으로 추적할 수 있습니다.
```bash
cd /home/rokey/dev_ws/isaac_sim/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir /home/rokey/dev_ws/rl_ws/logs/rsl_rl/ffw_sg2_sort
```

## 주요 기능 및 보상 체계 (Reward System)
- **`reaching_package`**: 그리퍼가 상자에 가까워지면 점수를 부여 (Dense Reward).
- **`lifting_package`**: 상자를 일정 높이 이상 들어 올렸을 때의 큰 보상.
- **`package_to_bin_fine`**: 상자를 목표 바구니 안으로 정밀하게 유도.
- **`sorting_success`**: 올바른 바구니에 성공적으로 떨어뜨렸을 때 지급되는 최종 목표 달성 보상.
- **`action_rate`, `joint_vel`**: 관절 속도와 제어 명령의 급격한 변화를 제한하는 패널티 (부드러운 움직임 유도).

## 의존성 및 오픈소스 수정 내역 (Dependencies & Open Source Modifications)
본 프로젝트는 아래의 오픈소스 레포지토리들을 클론하여 일부 코드를 프로젝트 목적에 맞게 수정하여 사용하고 있습니다. 

### 1. `robotis_lab`
- **원본 저장소:** [ROBOTIS-GIT/robotis_lab](https://github.com/ROBOTIS-GIT/robotis_lab.git)
- **수정 내역:**
  - `scripts/sim2real/bringup/sh5_dds_bringup.py` 수정
    - VR Teleoperation을 위한 **ROS 2 Camera Helper** 노드 추가 (RGB 카메라 스트림 퍼블리시).
    - `sim.step(render=True)`로 변경하여 시뮬레이션 내 카메라 렌더링 활성화 및 물리엔진 연산을 CPU 모드(`device="cpu"`)로 고정.
    - `_setup_camera_views` 및 `_setup_ros2_camera_publishers` 함수 추가.

### 2. `robotis_applications`
- **원본 저장소:** [ROBOTIS-GIT/robotis_applications](https://github.com/ROBOTIS-GIT/robotis_applications.git)
- **수정 내역:**
  - `docker/Dockerfile` 수정
    - 컨테이너 아키텍처 인자(`TARGETARCH=amd64`) 추가.
    - `ROS_DOMAIN_ID=119`로 네트워크 도메인 변경.
  - `docker/container.sh` 수정
    - `docker compose` 명령어를 구버전인 `docker-compose`로 호환되도록 수정.
  - `robotis_vuer/robotis_vuer/vr_publisher_sh5.py` 수정
    - 압축 이미지(`CompressedImage`) 뿐만 아니라 **Raw 이미지(`Image`)**도 수신할 수 있도록 OpenCV 브릿지 파이프라인 추가 (`enable_vr_image_raw` 파라미터).
    - VR 웹 인터페이스(Vuer)의 배경 이미지 전송 속도 및 안정성 최적화.

### 3. `robotis_dds_python` & `cyclonedds`
- **원본 저장소:** 
  - [ROBOTIS-GIT/robotis_dds_python](https://github.com/ROBOTIS-GIT/robotis_dds_python.git)
  - [eclipse-cyclonedds/cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- **수정 내역:** 별도의 코드 수정 없이 클론하여 그대로 사용.


## 🤖 SH5 VR Teleoperation & Imitation Learning (ACT)

프로젝트 고도화 단계에서 로봇 플랫폼을 **다관절 로봇 핸드를 가진 SH5**로 교체하고, VR 원격 조작(Teleoperation)을 통해 전문가 데이터 수집 및 ACT(Action Chunking with Transformers) 모방 학습 파이프라인을 추가하였습니다.

### 1. 전문가 데이터 수집 및 리플레이
- **데이터 수집:** `coupang_sh5_bringup.py` 스크립트를 사용하여 VR로 궤적을 제어하고, `obs`(로봇 Pose, 조인트 State, Box/Rack Pose) 및 `actions`(cmd_vel, target_joint_pos)를 HDF5 형식으로 기록합니다.
  - 물리 시뮬레이션의 한계(미끄러짐, 팽이 현상)를 극복하기 위해 **Magic Snapping(로컬 좌표계 기반 자석 부착)** 물리 안정화 기능을 도입하였습니다.
- **데이터 검증 (리플레이):** HDF5 데이터가 올바르게 녹화되었는지 확인하기 위한 독립 툴입니다.
  ```bash
  # HDF5 리플레이 실행
  isaac-python /home/rokey/dev_ws/coupang_ws/scripts/replay_data.py \
    --hdf5 /home/rokey/dev_ws/datasets/coupang_demo_20260608_150012.hdf5 \
    --episode 0 --enable_gravity
  ```
  - **조작 키:** `N`/`→` (다음 에피소드), `P`/`←` (이전 에피소드), `R` (재시작), `Space` (일시정지), `1`~`4` (슬롯 점프), `Q` (종료).

### 2. ACT 모델 학습 (Action Chunking with Transformers)
MLP 기반 Behavior Cloning의 한계인 시퀀스 흐름 무시 및 회전 튀기 현상을 개선하기 위해 **CVAE와 Transformer가 결합된 ACT 모델**을 도입했습니다.
- 과거 `context_len` 프레임의 상태 이력을 참조하여 미래 `chunk_size` 프레임의 행동들을 한 번에 예측(Action Chunking)합니다.
- **학습 스크립트 실행:**
  ```bash
  python3 /home/rokey/dev_ws/coupang_ws/scripts/train_act.py \
    --data /home/rokey/dev_ws/datasets/augmented_slot1_slot3.hdf5 \
    --output /home/rokey/dev_ws/models/sh5_act_slot1_3.pth \
    --epochs 500
  ```

### 3. 데이터 증강 및 물리 리밋 분석
- `augment_data.py`를 통해 슬롯 1(위층 좌측) 데이터를 Z오프셋($-0.738\text{m}$) 기반으로 가공하여 슬롯 3(아래층 좌측) 데이터를 자동으로 증강합니다.
- **물리 한계 발견:** SH5 로봇의 `lift_joint` 하향 리밋은 URDF 상 `-0.5m`로 제한되어 있어, $-0.738\text{m}$의 오프셋이 필요한 아래층 슬롯(3번, 4번)은 단순 증강이 아닌 **직접적인 데이터 수집**이 필수적으로 필요합니다.


