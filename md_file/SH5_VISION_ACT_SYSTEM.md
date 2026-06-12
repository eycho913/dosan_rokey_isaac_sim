# SH5 Vision ACT 기반 물류 로봇 모방 학습 시스템

## 1. 프로젝트 개요
본 프로젝트는 NVIDIA Isaac Sim 환경 내에서 다수의 AMR(SG2)과 로봇 암(SH5)이 연동되는 물류 자동화 시스템을 구축하고, 양팔 로봇의 정밀한 픽앤플레이스(Pick & Place) 작업을 수행하기 위한 모방 학습(Imitation Learning) 프레임워크입니다. 

## 2. ROBOTIS AI_WORKER 오픈소스 활용 및 커스터마이징
본 시스템은 **로보티즈(ROBOTIS)의 범용 AI 조작 오픈소스인 `AI_WORKER`**를 채택하여, 검증된 ALOHA/ACT 기반 모방 학습 파이프라인을 효율적으로 도입했습니다.
- **오픈소스 도입 배경**: 핵심 프레임워크를 활용해 초기 개발 시간을 단축하고 로봇 제어의 안정성을 확보했습니다.
- **프로젝트 특화 커스터마이징 포인트**:
  - 기존 체계를 SH5 양팔 로봇과 Isaac Sim 가상 환경에 맞춰 아키텍처 재설계.
  - 다중 모달 비전(TopView + Left/Right 카메라) 데이터 수집을 위해 HDF5 데이터 구조 및 I/O 파이프라인 전면 확장.
  - 4개의 서로 다른 적재 슬롯 목표 처리를 위한 **Goal Conditioning(목표 조건부 임베딩)** 알고리즘을 자체 추가하여 단일 모델로 모든 작업을 수행하는 만능 모델(Unified Model) 구현.

## 3. 개발 주요 내용
1. **데이터 수집 (VR Teleoperation)**:
   - VR 컨트롤러 및 키보드를 융합한 하이브리드 조작으로 400+ 에피소드의 양질의 데이터를 수집.
   - HDF5 포맷으로 저장: 3대의 카메라 이미지(120x160x3) 및 156차원의 방대한 상태 공간.
2. **데이터 전처리 (Preprocessing)**:
   - `freeze_idle_arms.py`: 사용하지 않는 대기 팔의 노이즈 데이터를 필터링/고정하여 학습 효율 증대.
   - `create_subset.py`, `augment_data.py`: 슬롯 간 데이터를 이동(Augmentation)시켜 데이터 불균형 해소.
3. **데이터 검증 (HDF5 Replay)**:
   - `replay_data.py`: 기존 물리(PD) 제어에서 발생하는 진동 문제를 원천 차단하기 위해 `teleport_joints` 기반 리플레이 기능을 구현하여 시각적 재생 정확도 100% 보장 및 고속 검증 달성.
4. **Vision ACT 모델 아키텍처**:
   - ResNet18 백본을 활용한 다중 카메라 특징 추출.
   - Transformer State Encoder (10프레임 Context) 및 CVAE 구조 기반의 20-Step Action Chunking 디코딩 (총 파라미터 약 2,070만 개 수준으로 경량화).

---

## 4. 환경 설정 및 실행 가이드 (Prerequisites & Setup)

### 4.1 시스템 요구 사항
- **OS**: Ubuntu 22.04 LTS
- **Simulator**: NVIDIA Isaac Sim 2023.1.1 (Python 환경: `~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh`)
- **Frameworks**: ROS2 Humble, PyTorch, HDF5, OpenCV

### 4.2 설치 및 연동 방법
다른 환경에서 본 코드를 실행하고 ROBOTIS 오픈소스를 불러오기 위한 세팅 가이드입니다.

```bash
# 1. 메인 작업 저장소 Clone
git clone https://github.com/eycho913/dosan_rokey_isaac_sim.git
cd dosan_rokey_isaac_sim

# 2. 필수 의존성 설치
pip3 install torch torchvision h5py numpy opencv-python tqdm

# 3. ROBOTIS AI_WORKER 호환 라이브러리 연동
# AI_WORKER 오픈소스에서 사용하는 제어 의존성(robotis_lab 등)을 PYTHONPATH에 추가하거나 설치해야 합니다.
# ROS2 환경일 경우 workspace를 빌드하고 source 합니다.
# (예시)
# git clone https://github.com/ROBOTIS-GIT/AI_WORKER.git
# export PYTHONPATH=$PYTHONPATH:$(pwd)/AI_WORKER/src
```

### 4.3 파이프라인 실행 스크립트

#### Step 1: 시뮬레이션 환경 구동 및 데이터 수집
Isaac Sim의 내장 Python 인터프리터를 사용하여 VR 조작 환경을 실행합니다.
```bash
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh scripts/coupang_sh5_bringup_v.py
```

#### Step 2: HDF5 리플레이 (데이터 검증)
수집된 데이터가 오차 없이 저장되었는지 재생 속도를 조절해가며 육안 검증합니다.
```bash
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh scripts/replay_data.py
```

#### Step 3: 데이터 전처리 (필터링 및 증강)
일반 Python 환경을 사용하여 빠른 전처리를 수행합니다.
```bash
python3 scripts/freeze_idle_arms.py
python3 scripts/augment_data.py
```

#### Step 4: Vision ACT 모델 학습
다중 GPU 또는 단일 GPU 환경에서 모방 학습을 시작합니다. HDF5 데이터 디렉토리를 지정합니다.
```bash
python3 scripts/train_act_v3.py --data /path/to/dataset/hdf5_dir --epochs 500 --batch_size 16
```

#### Step 5: 학습된 모델 추론 (Inference)
학습된 체크포인트를 불러와 Isaac Sim 환경 내에서 자율 픽앤플레이스를 평가합니다.
```bash
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh scripts/evaluate_test_vision.py
```
