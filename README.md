# SH5 Coupang Logistics Automation System

> Isaac Sim 기반 SH5 로봇 물류 자동화 시스템 — 데이터 수집, HDF5 재생, ROS 2 연동, 다중 로봇 시연

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [최종 산출물 목록](#최종-산출물-목록)
3. [디렉토리 구조](#디렉토리-구조)
4. [핵심 스크립트 설명](#핵심-스크립트-설명)
5. [실행 방법](#실행-방법)
6. [주요 개발 이력](#주요-개발-이력)
7. [상자 파지(Grasp) 로직 — HDF5-Guided Snapping](#상자-파지grasp-로직--hdf5-guided-snapping)
8. [작업대 Spawn/Despawn 기능](#작업대-spawndespawn-기능)
9. [파라미터 레퍼런스](#파라미터-레퍼런스)
10. [알려진 이슈 및 주의사항](#알려진-이슈-및-주의사항)

---

## 시스템 개요

```
[AMR / WMS 관제탑]
    │  ROS 2 Topic (sg2_spawn_trigger)
    │  ROS 2 Topic (sg2_workstation_trigger)  ← 작업대 Spawn/Despawn
    ▼
ros2_sh5_bridge.py          ← ROS 2 ↔ Isaac Sim 브릿지 (파일 큐 방식)
    │  /tmp/sh5_queue.jsonl
    │  /tmp/sh5_ws_trigger.jsonl              ← 작업대 트리거
    ▼
sh5_bringup_ros2_3robot.py  ← Isaac Sim 메인 (3대 로봇 동시 시연)
    ├── 상자 스폰 (box_assets/*.usd)
    ├── QR 스캔 (TopView 카메라 + WeChatQRCode)
    ├── HDF5 궤적 재생 (frozen_set 에피소드)
    │       └── HDF5-Guided Snapping 파지 로직
    ├── 워밍업 보간 (현재 자세 → 첫 프레임, 30프레임)
    ├── 호밍 (stay.hdf5 안전 자세)
    ├── WorkstationManager (작업대 Despawn/Spawn)
    └── report_inbound_progress 보고
```

---

## 최종 산출물 목록

### 🚀 시연 (Isaac Sim 실행)

| 파일 | 역할 |
|------|------|
| `scripts/sh5_bringup_ros2_3robot.py` | ★ **메인 시연** — 3대 로봇 동시 pick & place |
| `scripts/ros2_sh5_bridge.py` | ★ **ROS 2 브릿지** — 관제탑 ↔ Isaac Sim 연동 |
| `scripts/test_trigger.sh` | 테스트 트리거 (3라인 × 3배치 패키지 전송) |
| `scripts/send_packages.sh` | 단순 패키지 전송 스크립트 |

### 📦 데이터 수집

| 파일 | 역할 |
|------|------|
| `scripts/coupang_sh5_bringup_v.py` | ★ **VR 조작 데이터 수집** — HDF5 녹화 |
| `scripts/coupang_sh5_bringup.py` | 데이터 수집 구버전 |

### 🔧 데이터 전처리 / 증강

| 파일 | 역할 |
|------|------|
| `scripts/augment_data.py` | ★ **데이터 증강** — 슬롯 좌우 미러링, 노이즈 추가 |
| `scripts/augment_slot3_to_slot4.py` | 슬롯 3 → 슬롯 4 변환 증강 |
| `scripts/create_subset.py` | ★ **서브셋 추출** — frozen_set 전처리 (방해 팔 제거) |
| `scripts/filter_dataset.py` | HDF5 에피소드 필터링 (실패 에피소드 제거) |
| `scripts/freeze_idle_arms.py` | ★ **방해 팔 고정** — 비동작 팔을 stay 자세로 오버라이드 |
| `scripts/freeze_right_arm.py` | 오른팔만 고정하는 특수 버전 |
| `scripts/split_v2.py` | HDF5 에피소드 분할 |

### 🧠 학습 (ACT 모델)

| 파일 | 역할 |
|------|------|
| `scripts/train_act_v2.py` | ★ **ACT v2 학습** — Vision-ACT 파인튜닝 (최신) |
| `scripts/train_act_v3.py` | ACT v3 (실험적) |
| `scripts/train_act.py` | ACT 초기 버전 |
| `scripts/train_bc.py` | Behavior Cloning 학습 |
| `scripts/evaluate_act.py` | ACT 모델 추론 평가 |
| `scripts/evaluate_test_vision.py` | Vision-ACT 테스트 (Isaac Sim 인터랙티브) |

### 🔄 HDF5 재생 (디버그/검증)

| 파일 | 역할 |
|------|------|
| `scripts/hdf5_replay_player.py` | ★ **HDF5 에피소드 로더** |
| `scripts/replay_data.py` | 단일 에피소드 재생 |
| `scripts/replay_data2.py` | 박스 pose 강제 주입 버전 |
| `scripts/replay_and_capture.py` | 재생 + 카메라 캡처 동시 수행 |

---

## 디렉토리 구조

```
coupang_ws/
├── README.md
└── scripts/
    ├── 🚀 시연
    │   ├── sh5_bringup_ros2_3robot.py   ★ 3대 로봇 동시 시연 (최신)
    │   ├── sh5_bringup_ros2.py          1대 로봇 시연
    │   ├── ros2_sh5_bridge.py           ★ ROS 2 ↔ Isaac Sim 브릿지
    │   ├── test_trigger.sh              3-로봇 테스트 트리거
    │   └── send_packages.sh             패키지 전송
    │
    ├── 📦 데이터 수집
    │   ├── coupang_sh5_bringup_v.py     ★ VR 조작 데이터 수집
    │   └── coupang_sh5_bringup.py       수집 구버전
    │
    ├── 🔧 데이터 전처리
    │   ├── augment_data.py              ★ 데이터 증강
    │   ├── augment_slot3_to_slot4.py    슬롯 변환 증강
    │   ├── create_subset.py             ★ frozen_set 서브셋 추출
    │   ├── filter_dataset.py            에피소드 필터링
    │   ├── freeze_idle_arms.py          ★ 방해 팔 고정 전처리
    │   └── freeze_right_arm.py          오른팔 고정
    │
    ├── 🧠 학습
    │   ├── train_act_v2.py              ★ ACT v2 학습 (최신)
    │   ├── train_act_v3.py              ACT v3
    │   ├── train_act.py                 ACT 초기
    │   ├── train_bc.py                  BC 학습
    │   ├── evaluate_act.py              ACT 추론
    │   └── evaluate_test_vision.py      Vision-ACT 테스트
    │
    ├── 🔄 HDF5 재생
    │   ├── hdf5_replay_player.py        ★ 에피소드 로더
    │   ├── replay_data.py               단일 재생
    │   ├── replay_data2.py              pose 주입 버전
    │   └── replay_and_capture.py        재생+캡처
    │
    └── 📡 기타
        ├── qr_scanner_node.py           QR 스캐너 노드
        ├── sh5_logger.py                로깅 유틸
        ├── homography_calibration.py    카메라 캘리브레이션
        └── box_fsm_manager.py           상자 FSM 관리자
```

---

## 핵심 스크립트 설명

### `sh5_bringup_ros2_3robot.py` ★ (메인 시연)

3대의 SH5 로봇이 각각 독립적인 컨베이어 라인(`sg2_in_01~03`)에서 동시에 pick & place를 수행하는 메인 시연 스크립트.

**주요 컴포넌트:**
- `BringupSceneCfg` — finalfac.usd 환경 + 3대 로봇 + 3개 상자 씬 구성
- `SlotRegistry` — 고객별 슬롯 유지 할당 (같은 고객 → 같은 슬롯)
- `WorkstationManager` — 작업대 Prim Despawn/Spawn 실시간 관리
- `ReplayController` — 상태머신 기반 재생 컨트롤러
- `FileQueueReader` — `/tmp/sh5_queue.jsonl` 폴링

**상태머신 흐름:**
```
IDLE ──트리거──▶ SCANNING ──QR인식──▶ WAITING_DB ──DB응답──▶ REPLAYING ──완료──▶ HOMING ──▶ DONE ──▶ IDLE
```

---

### `coupang_sh5_bringup_v.py` ★ (데이터 수집)

VR 조작을 통해 로봇 시연 데이터를 HDF5 파일로 기록하는 스크립트.

**저장 데이터:**
| 키 | 내용 |
|---|---|
| `obs/joint_positions` | 관절 각도 (rad) |
| `obs/box_pose` | 상자 위치 + 쿼터니언 (7D) |
| `obs/robot_pose` | 로봇 베이스 위치 + 쿼터니언 (7D) |
| `obs/images/*` | 카메라 RGB 이미지 (160×120) |
| `actions/joint_targets` | PD 제어 목표 관절값 |

---

### `create_subset.py` + `freeze_idle_arms.py` ★ (전처리 파이프라인)

데이터 수집 시 키보드 조작에 의해 방해가 되는 반대 팔(비동작 팔)의 움직임을 제거하는 전처리 파이프라인.

```bash
# 1단계: 방해 팔 고정 처리
python3 freeze_idle_arms.py --input /datasets/raw/ --output /datasets/frozen/

# 2단계: 학습용 서브셋 추출
python3 create_subset.py --input /datasets/frozen/ --output /datasets/frozen_set/ --n 100
```

---

### `augment_data.py` ★ (데이터 증강)

좌우 미러링, 관절 노이즈 추가 등으로 에피소드 수를 늘리는 증강 스크립트.

---

### `train_act_v2.py` ★ (ACT 학습)

Vision-ACT 모델을 fine-tuning하는 학습 스크립트. Google Colab A100 기준 150 epoch 학습.

```bash
python3 train_act_v2.py \
  --data_dir /datasets/train_data/frozen_set \
  --output_dir /models/ \
  --epochs 150 \
  --batch_size 64
```

---

## 실행 방법

### 1. Isaac Sim 시연 (3대 로봇)

```bash
# 터미널 1 — Isaac Sim 메인
isaac-python /home/rokey/dev_ws/coupang_ws/scripts/sh5_bringup_ros2_3robot.py

# 터미널 2 — ROS 2 브릿지 (관제탑 연동 시)
python3 /home/rokey/dev_ws/coupang_ws/scripts/ros2_sh5_bridge.py

# 터미널 3 — 테스트 트리거
bash /home/rokey/dev_ws/coupang_ws/scripts/test_trigger.sh
```

### 2. 작업대 Despawn/Spawn 수동 테스트

```bash
# 작업대 숨기기 (AMR이 작업대 이동 시작)
echo '{"workstation_id":"WS02","location":"sg2_in_01_A","action":"DESPAWN"}' >> /tmp/sh5_ws_trigger.jsonl

# 작업대 복원 (AMR이 새 작업대 안착 완료)
echo '{"workstation_id":"WS02","location":"sg2_in_01_A","action":"SPAWN"}' >> /tmp/sh5_ws_trigger.jsonl
```

### 3. 데이터 수집 (VR 조작)

```bash
isaac-python /home/rokey/dev_ws/coupang_ws/scripts/coupang_sh5_bringup_v.py
```

**키보드 조작:**
| 키 | 동작 |
|---|---|
| W/S | 전진/후진 |
| A/D | 좌/우 회전 |
| R 또는 1 | 🔴 녹화 시작 |
| T 또는 2 | ⬛ 저장 (성공) |
| C 또는 3 | 🗑️ 취소 (실패) |

### 4. 전처리 파이프라인

```bash
# 방해 팔 고정
python3 scripts/freeze_idle_arms.py

# 서브셋 추출 (frozen_set)
python3 scripts/create_subset.py

# 데이터 증강
python3 scripts/augment_data.py
```

---

## 주요 개발 이력

### v1.0 — 단일 로봇 기반 구축
- Isaac Sim + SH5 USD 로봇 스폰
- VR 조작 데이터 수집 파이프라인 구현
- HDF5 에피소드 저장 (`VRDemonstrationLogger`)

### v2.0 — HDF5 재생 + ROS 2 연동
- `sh5_bringup_ros2.py` 구현
- ROS 2 브릿지 파일 큐 방식 도입
- QR 스캐너 (TopView 카메라 + WeChatQRCode) 통합
- `report_inbound_progress` 입고 보고 구현

### v3.0 — 3대 로봇 동시 시연
- `sh5_bringup_ros2_3robot.py` 신규 구현
- 3대 로봇 독립 상태머신(`ReplayController`) 병렬 운영
- pause/resume 기능 (`/tmp/sh5_pause.json` 폴링)

### v3.1 — 재생 안정화
- **워밍업 보간**: 현재 자세 → 첫 프레임 30프레임 선형 보간 (`WARMUP_FRAMES=30`)
  - 기존: 첫 프레임으로 텔레포트 → 부자연스러운 순간이동
  - 개선: 약 1초에 걸쳐 서서히 이동
- **frozen_set 에피소드 사용**: 방해 팔이 제거된 전처리 데이터로 교체
- **stay.hdf5 호밍**: 복귀 시 안전 자세로 이동 (기존 복귀 모션의 쓰러짐 방지)

### v3.2 — HDF5-Guided Snapping (파지 안정화)
- **문제**: 손가락 파지 후 상자가 딜레이되어 손에 안착되지 않거나 공중에 떠 있는 현상
- **해결**: `ATTACH_FACTOR=1.0` + `MAX_BOX_STEP=3.0` — 박스를 링크 중심에 즉시 부착
- **개선**: HDF5 `box_trajectory` 기반 파지 링크 자동 선택 (왼/오른손 오류 방지)
- **릴리즈**: 손가락 열림(`finger_pos_avg < 0.80`) 감지 시 자동 릴리즈 → 슬롯에 자연스러운 안착

### v3.3 — Kinematic 물리 충돌 수정 (yo-yo 현상 제거)
- **문제**: 상자가 yo-yo처럼 Z축 방향으로 진동하며 손에서 일정 offset을 유지
- **원인 1**: 패키지 스폰 시 `kinematic_enabled=False`로 물리 활성화 → 중력 vs `write_root_state_to_sim` 싸움
- **원인 2**: `write_root_state_to_sim`이 내부적으로 `setLinearVelocity/setAngularVelocity` 호출 → kinematic 바디에서 PhysX 에러 1000개 → 시뮬 강제 종료
- **해결 1**: 스폰 시 `kinematic_enabled=False` 코드 완전 제거 (항상 kinematic=True 유지)
- **해결 2**: `_write_box_pose()` 헬퍼 도입 — `write_root_pose_to_sim` 또는 USD XFormable 직접 쓰기로 velocity 설정 우회

### v3.4 — 작업대 Spawn/Despawn 기능
- **기능**: AMR이 작업대 이동 시 Isaac Sim에서 3D 모델(RACK prim) 실시간 숨김/복원
- **구현**: `WorkstationManager` 클래스 추가
  - DESPAWN: 해당 RACK prim을 Z=-200으로 이동 (화면 밖)
  - SPAWN: 원래 위치 또는 `WORKSTATION_SPAWN_POS` 좌표로 복원
- **트리거**: `/tmp/sh5_ws_trigger.jsonl` 파일큐 폴링 방식
- **RACK 매핑** (finalfac.usd 기준):
  - RACK_01 = sg2_out (출고 컨베이어)
  - RACK_02 = sg2_in_01_A (1번 라인 작업대)
  - RACK_03 = sg2_in_02_A (2번 라인 작업대)
  - RACK_04 = sg2_in_03_A (3번 라인 작업대)

---

## 상자 파지(Grasp) 로직 — HDF5-Guided Snapping

> `sh5_bringup_ros2_3robot.py`의 `ReplayController.step()` 내부

### 동작 원리

```
매 프레임 (REPLAYING 상태):
  ├── [1단계] HDF5 box_trajectory로 상자 기준 위치 계산
  ├── [2단계] 손가락 상태 확인 (finger_pos_avg)
  │     ├── >= 0.80 (닫힘): HDF5 기반 파지 링크로 상자 즉시 부착
  │     │     ├── HDF5 박스 위치로 가장 가까운 로봇 링크 선택
  │     │     ├── target_pos = robot_body_pos[idx]  (ATTACH_FACTOR=1.0)
  │     │     └── MAX_BOX_STEP=3.0으로 속도 클램프 (사실상 즉시 반응)
  │     └── < 0.80 (열림): HDF5 원본 위치 사용 (자연스러운 릴리즈)
  └── [3단계] _write_box_pose() 로 위치 적용 (velocity 설정 없음)
```

### 핵심 파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `ATTACH_FACTOR` | `1.0` | 0=HDF5 원본, 1=링크 완전 중심 부착 |
| `GRASP_DIST` | `0.30` m | 스냅 활성화 거리 (30cm 이내 시 파지 시작) |
| `FINGER_OPEN_THRESH` | `0.80` rad | 손가락 열림 판정 임계값 |
| `MAX_BOX_STEP` | `3.0` m/frame | 속도 클램프 (사실상 즉시 반응) |

---

## 작업대 Spawn/Despawn 기능

### 메시지 형식 (`/tmp/sh5_ws_trigger.jsonl`)

```json
{"workstation_id": "WS02", "location": "sg2_in_01_A", "action": "DESPAWN"}
{"workstation_id": "WS02", "location": "sg2_in_01_A", "action": "SPAWN"}
```

### RACK 매핑 (`WS_LOCATION_TO_RACK`)

```python
WS_LOCATION_TO_RACK = {
    "sg2_out":     "RACK_01",   # 출고 컨베이어
    "sg2_in_01_A": "RACK_02",   # 1번 라인 작업대
    "sg2_in_02_A": "RACK_03",   # 2번 라인 작업대
    "sg2_in_03_A": "RACK_04",   # 3번 라인 작업대
}
```

---

## 파라미터 레퍼런스

### `sh5_bringup_ros2_3robot.py` 전역 상수

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `PLAYBACK_SPEED` | `2` | 재생 배속 (1=원속, 2=2배속) |
| `WARMUP_FRAMES` | `30` | 첫 프레임 보간 프레임 수 (~1초) |
| `HOMING_FRAMES` | `120` | 호밍 보간 프레임 수 (~4초) |
| `QR_SCAN_TIMEOUT` | `5.0` | QR 스캔 최대 대기 시간(초) |
| `DB_WAIT_TIMEOUT` | `5.0` | DB 응답 최대 대기 시간(초) |
| `ATTACH_FACTOR` | `1.0` | 파지 링크 부착 강도 |
| `FINGER_OPEN_THRESH` | `0.80` | 손가락 열림 임계값 (rad) |
| `FROZEN_SET_DIR` | `datasets/train_data/frozen_set` | 재생 에피소드 경로 |
| `STAY_HDF5_PATH` | `datasets/stay.hdf5` | 호밍 자세 데이터 |

### 파일 큐 인터페이스

| 파일 | 방향 | 내용 |
|------|------|------|
| `/tmp/sh5_queue.jsonl` | Bridge → Isaac | 패키지 투입 트리거 |
| `/tmp/sh5_qr_req.jsonl` | Isaac → Bridge | QR DB 확인 요청 |
| `/tmp/sh5_qr_result.jsonl` | Bridge → Isaac | DB 중복 체크 결과 |
| `/tmp/sh5_report_req.jsonl` | Isaac → Bridge | 입고 완료 보고 |
| `/tmp/sh5_pause.json` | Bridge → Isaac | 일시정지 신호 |
| `/tmp/sh5_ws_trigger.jsonl` | Bridge → Isaac | 작업대 Spawn/Despawn |

### 라인별 로봇 스폰 위치

```python
LINE_ROBOT_POS = {
    "sg2_in_01": (7.5,  3.0, -0.18),
    "sg2_in_02": (7.5, -1.5, -0.18),
    "sg2_in_03": (7.5, -6.0, -0.18),
}
```

---

## 알려진 이슈 및 주의사항

### ⚠️ Kinematic 박스 — velocity 설정 금지
`write_root_state_to_sim` 대신 반드시 `_write_box_pose()` 헬퍼를 사용해야 합니다.
kinematic 바디에 velocity를 설정하면 PhysX 에러가 1000개 누적되어 시뮬레이션이 강제 종료됩니다.

### ⚠️ GPU vs CPU PhysX
RTX 5080 (Blackwell)에서 GPU PhysX 파이프라인이 불안정하여 **CPU 모드** 사용 중:
```python
sim_cfg = sim_utils.SimulationCfg(device="cpu", ...)
```

### ⚠️ RACK prim 경로 확인
`WORKSTATION_PRIM_PATTERN` 및 `WS_LOCATION_TO_RACK`은 `finalfac.usd` 파일의 실제 prim 경로와 일치해야 합니다. Isaac Sim Stage 패널에서 RACK을 클릭하여 경로를 확인하세요.

### ⚠️ finger_indices가 비어있을 경우
SH5 USD의 관절 이름에 `"finger"`가 포함된 조인트가 없으면 파지 로직이 동작하지 않습니다.

---

## 데이터셋 경로

```
/home/rokey/dev_ws/datasets/
├── train_data/
│   ├── frozen_set/     ★ 방해 팔 제거된 전처리 데이터 (학습/재생 사용)
│   ├── slot1_*.hdf5    슬롯 1 원본
│   ├── slot2_*.hdf5    슬롯 2 원본
│   ├── slot3_*.hdf5    슬롯 3 원본
│   └── slot4_*.hdf5    슬롯 4 원본
└── stay.hdf5           ★ 안전 자세 (호밍용)

/home/rokey/dev_ws/box_assets/
└── *.usd               상자 USD 모델
```

---

*Last updated: 2026-06-12 (v3.4 — 작업대 Spawn/Despawn + Kinematic 물리 수정)*
