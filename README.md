# SH5 Coupang Logistics Automation System

> Isaac Sim 기반 SH5 로봇 물류 자동화 시스템 — 데이터 수집, HDF5 재생, ROS 2 연동, 다중 로봇 시연

---

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [디렉토리 구조](#디렉토리-구조)
3. [핵심 스크립트 설명](#핵심-스크립트-설명)
4. [실행 방법](#실행-방법)
5. [주요 개발 이력](#주요-개발-이력)
6. [상자 파지(Grasp) 로직 — Magic Snapping](#상자-파지grasp-로직--magic-snapping)
7. [파라미터 레퍼런스](#파라미터-레퍼런스)
8. [알려진 이슈 및 주의사항](#알려진-이슈-및-주의사항)

---

## 시스템 개요

```
[AMR / WMS]
    │  ROS 2 Topic (sg2_spawn_trigger)
    ▼
ros2_sh5_bridge.py          ← ROS 2 ↔ Isaac Sim 브릿지 (파일 큐 방식)
    │  /tmp/sh5_queue.jsonl
    ▼
sh5_bringup_ros2_3robot.py  ← Isaac Sim 메인 (3대 로봇 동시 시연)
    ├── 상자 스폰 (box_assets/*.usd)
    ├── QR 스캔 (TopView 카메라 + WeChatQRCode)
    ├── HDF5 궤적 재생 (hdf5_replay_player.py)
    │       └── Magic Snapping 파지 로직
    ├── 호밍 (팔 안전 복귀)
    └── report_inbound_progress 보고 (/tmp/sh5_report_req.jsonl)
```

---

## 디렉토리 구조

```
coupang_ws/
└── scripts/
    ├── 📦 메인 시뮬레이션
    │   ├── sh5_bringup_ros2_3robot.py   ★ 3대 로봇 동시 시연 (최신)
    │   ├── sh5_bringup_ros2.py          ★ 1대 로봇 시연
    │   └── coupang_sh5_bringup_v.py     ★ 데이터 수집 전용 (VR 조작)
    │
    ├── 🌉 브릿지
    │   ├── ros2_sh5_bridge.py           ROS 2 ↔ Isaac Sim 파일 큐 브릿지
    │   └── sh5_dds_bridge.py            DDS 기반 브릿지
    │
    ├── 🤖 HDF5 재생
    │   ├── hdf5_replay_player.py        HDF5 에피소드 로더 (slot별 랜덤 선택)
    │   ├── replay_data.py               단일 HDF5 재생 (디버그용)
    │   └── replay_data2.py              박스 pose 강제 주입 버전
    │
    ├── 🧠 학습
    │   ├── train_act.py                 ACT 모델 학습
    │   ├── train_act_v3.py              ACT v3 (최신)
    │   ├── evaluate_act.py              ACT 추론 평가
    │   └── augment_data.py              데이터 증강
    │
    ├── 🔧 유틸리티
    │   ├── test_trigger.sh              ★ 3-로봇 테스트 트리거 (9패키지 3배치)
    │   ├── send_packages.sh             패키지 전송 스크립트
    │   ├── create_subset.py             HDF5 서브셋 추출
    │   └── filter_dataset.py            데이터셋 필터링
    │
    └── 📡 기타
        ├── qr_scanner_node.py           QR 스캐너 노드
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
- `ReplayController` — 상태머신 기반 재생 컨트롤러 (IDLE → SCANNING → WAITING_DB → REPLAYING → HOMING → DONE)
- `FileQueueReader` — `/tmp/sh5_queue.jsonl` 폴링

**상태머신 흐름:**
```
IDLE ──트리거 수신──▶ SCANNING ──QR인식──▶ WAITING_DB ──DB응답──▶ REPLAYING ──완료──▶ HOMING ──▶ DONE ──▶ IDLE
```

---

### `coupang_sh5_bringup_v.py` ★ (데이터 수집)

VR 조작을 통해 로봇 시연 데이터를 HDF5 파일로 기록하는 스크립트.

**주요 기능:**
- `VRDemonstrationLogger` — HDF5 에피소드 녹화/저장/취소
- `TerminalKeyboard` — WASD 이동 + R/T/C/B 녹화 제어
- Magic Snapping (데이터 수집 중 파지 보조)
- 카메라 어노테이터 (Left/Right/TopView 이미지 동시 저장)

**저장 데이터:**
| 키 | 내용 |
|---|---|
| `obs/joint_positions` | 관절 각도 (rad) |
| `obs/box_pose` | 상자 위치 + 쿼터니언 (7D) |
| `obs/robot_pose` | 로봇 베이스 위치 + 쿼터니언 (7D) |
| `obs/images/*` | 카메라 RGB 이미지 (160×120) |
| `actions/joint_targets` | PD 제어 목표 관절값 |

---

### `hdf5_replay_player.py`

HDF5 파일에서 에피소드를 로드하여 Isaac Sim 시뮬레이터에 주입하는 로더.

```python
loader = HDF5EpisodeLoader(slot_num=1)
episode = loader.load_random_episode()
# episode["joint_trajectory"]  — 관절 궤적
# episode["box_trajectory"]    — 상자 궤적 (7D: xyz + quat)
# episode["robot_trajectory"]  — 로봇 베이스 궤적 (7D)
```

---

### `test_trigger.sh` ★ (테스트 도구)

3대 로봇에 패키지 투입 트리거를 3배치(9개)로 전송하는 테스트 쉘 스크립트.

```bash
bash /home/rokey/dev_ws/coupang_ws/scripts/test_trigger.sh
```

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BATCH_WAIT` | `10` | 배치 사이 대기 시간(초) |

---

## 실행 방법

### 1. Isaac Sim 시연 (3대 로봇)

```bash
# 터미널 1 — Isaac Sim 메인
isaac-python /home/rokey/dev_ws/coupang_ws/scripts/sh5_bringup_ros2_3robot.py

# 터미널 2 — ROS 2 브릿지 (선택)
python3 /home/rokey/dev_ws/coupang_ws/scripts/ros2_sh5_bridge.py

# 터미널 3 — 테스트 트리거
bash /home/rokey/dev_ws/coupang_ws/scripts/test_trigger.sh
```

### 2. 데이터 수집 (VR 조작)

```bash
isaac-python /home/rokey/dev_ws/coupang_ws/scripts/coupang_sh5_bringup_v.py \
  --slot 1 \
  --enable_camera_views
```

**키보드 조작:**
| 키 | 동작 |
|---|---|
| W/S | 전진/후진 |
| A/D | 좌/우 회전 |
| Q/E | 좌/우 횡이동 |
| U/O | 리프트 올림/내림 |
| R 또는 1 | 🔴 녹화 시작 |
| T 또는 2 | ⬛ 저장 (성공) |
| C 또는 3 | 🗑️ 취소 (실패) |
| B 또는 4 | 📦 상자 랜덤 리스폰 |

### 3. 단일 HDF5 재생 (디버그)

```bash
isaac-python /home/rokey/dev_ws/coupang_ws/scripts/replay_data.py \
  --hdf5 /home/rokey/dev_ws/datasets/train_data/slot1_1.hdf5 \
  --episode 0
```

---

## 주요 개발 이력

### v1.0 — 단일 로봇 기반 구축

- Isaac Sim + SH5 USD 로봇 스폰
- DDS 브릿지 (`SH5DdsBridge`) 구축
- VR 조작 데이터 수집 파이프라인 구현
- HDF5 에피소드 저장 (`VRDemonstrationLogger`)

### v2.0 — HDF5 재생 + ROS 2 연동

- `sh5_bringup_ros2.py` 구현 — HDF5 궤적 재생
- ROS 2 브릿지 파일 큐 방식 도입 (`/tmp/sh5_queue.jsonl`)
- QR 스캐너 (TopView 카메라 + WeChatQRCode) 통합
- `SlotRegistry` 고객별 슬롯 유지 할당
- `report_inbound_progress` 입고 보고 구현

### v3.0 — 3대 로봇 동시 시연

- `sh5_bringup_ros2_3robot.py` 신규 구현
- 3대 로봇 독립 상태머신(`ReplayController`) 병렬 운영
- `LINE_ROBOT_POS`로 라인별 로봇 스폰 위치 관리
- pause/resume 기능 (`/tmp/sh5_pause.json` 폴링)

### v3.1 — Magic Snapping 파지 로직 강화

- **문제**: HDF5 재생 시 상자가 물리 엔진 간섭으로 공중에 떠서 지연됨
- **해결 1**: 상자 `kinematic_enabled = True` — 물리 충돌 간섭 완전 차단
- **해결 2**: Magic Snapping 이식 — 손가락이 닫히는 순간 상자를 손 링크에 강체 결합
- **개선**: 쿼터니언 회전 동기화 추가 (`grasp_local_quat`) — 손목 회전 시 상자도 함께 기울어짐
- **개선**: Offset 비율 조정 (`* 0.1`) — 상자를 손 중심부로 90% 당겨오기
- **버그 수정**: 급격한 움직임 시 상자 이탈 방지 — `is_grasped` 플래그로 거리 조건 우회

---

## 상자 파지(Grasp) 로직 — Magic Snapping

> `sh5_bringup_ros2_3robot.py` 및 `sh5_bringup_ros2.py`의 `ReplayController.step()` 내부

### 동작 원리

```
매 프레임:
  ├── [1단계] HDF5 box_trajectory로 상자 위치 초기 설정 (텔레포트)
  └── [2단계] Magic Snapping 판단
        ├── 손가락 조인트 평균 > 0.20 (닫혀 있음)
        │     AND
        ├── (이미 잡은 상태) OR (가장 가까운 로봇 링크가 25cm 이내)
        │
        ├── 조건 충족 → 잡기 시작 (최초 1회)
        │     ├── 가장 가까운 로봇 링크 인덱스 저장 (grasped_body_idx)
        │     ├── 로컬 오프셋 계산: q_inv * (box_pos - body_pos) * 0.1
        │     └── 로컬 회전 계산: q_inv * box_quat
        │
        └── 조건 충족 → 매 프레임 강체 결합 유지
              ├── world_offset = body_quat * grasp_local_offset
              ├── world_quat = body_quat * grasp_local_quat
              └── box 위치/회전을 위 값으로 강제 덮어쓰기
```

### 핵심 파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `finger_target_avg > 0.20` | 0.20 rad | 손가락이 닫혔다고 판단하는 임계값 |
| `min_dist < 0.25` | 0.25 m | 최초 잡기를 시작하는 감지 범위 |
| `* 0.1` | 10% | 잡는 순간 오프셋 거리를 10%로 축소 (손 안으로 당기기) |

### 놓기 조건

```python
# 손가락이 펴지면 (finger_target_avg <= 0.20) 자동으로 Snapping 해제
if hasattr(self, "grasped_body_idx"): del self.grasped_body_idx
if hasattr(self, "grasp_local_offset"): del self.grasp_local_offset
if hasattr(self, "grasp_local_quat"): del self.grasp_local_quat
```

---

## 파라미터 레퍼런스

### `sh5_bringup_ros2_3robot.py` 전역 상수

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `SKIP_FRAMES` | `1` | HDF5 재생 프레임 스킵 (1=원속) |
| `PLAYBACK_SPEED` | `1` | 재생 배속 (1=원속, 2=2배속) |
| `HOMING_FRAMES` | `120` | 호밍 보간 프레임 수 (~4초 @ 30Hz) |
| `QR_SCAN_TIMEOUT` | `5.0` | QR 스캔 최대 대기 시간(초) |
| `DB_WAIT_TIMEOUT` | `5.0` | DB 응답 최대 대기 시간(초) |
| `PLACEMENT_FREEZE_FRAMES` | `30` | 안착 후 kinematic 고정 프레임 |
| `BOX_DESPAWN_POS` | `(0,0,-10)` | 상자 숨김 위치 (Z=-10m) |

### 라인별 로봇 스폰 위치

```python
LINE_ROBOT_POS = {
    "sg2_in_01": (7.5,  3.0, -0.18),
    "sg2_in_02": (7.5, -1.5, -0.18),
    "sg2_in_03": (7.5, -6.0, -0.18),
}
```

### 파일 큐 인터페이스

| 파일 | 방향 | 형식 |
|------|------|------|
| `/tmp/sh5_queue.jsonl` | Bridge → Isaac | 패키지 투입 트리거 |
| `/tmp/sh5_qr_req.jsonl` | Isaac → Bridge | QR DB 확인 요청 |
| `/tmp/sh5_qr_result.jsonl` | Bridge → Isaac | DB 중복 체크 결과 |
| `/tmp/sh5_report_req.jsonl` | Isaac → Bridge | 입고 완료 보고 |
| `/tmp/sh5_pause.json` | Bridge → Isaac | 일시정지 신호 |

---

## 알려진 이슈 및 주의사항

### ⚠️ Kinematic 상자 — 재생 완료 후 상자 고정

재생 완료 후 상자를 랙에 안착시키려면 USD Physics API로 `kinematic_enabled=True`를 설정해야 합니다. 이는 `ReplayController.step()` 내 재생 완료 블록에 이미 구현되어 있습니다.

```python
rb_api.GetKinematicEnabledAttr().Set(True)  # 재생 완료 후 상자 랙 위에 고정
```

### ⚠️ GPU vs CPU PhysX

RTX 5080 (Blackwell)에서 GPU PhysX 파이프라인이 불안정하여 **CPU 모드**를 사용 중입니다.

```python
sim_cfg = sim_utils.SimulationCfg(device="cpu", ...)
```

### ⚠️ finger_indices가 비어있을 경우

SH5 USD의 관절 이름에 `"finger"`가 포함된 조인트가 없으면 Magic Snapping이 동작하지 않습니다. 이 경우 조건을 `joint_pos_target` 기반 다른 조인트로 변경 필요.

### ⚠️ HDF5 box_trajectory 누락

일부 구형 데이터셋(`slot_*_f.hdf5`)에는 `box_pose` 키가 없을 수 있습니다. `HDF5EpisodeLoader`가 `None`을 반환하면 Magic Snapping만으로 동작합니다.

---

## 데이터셋 경로

```
/home/rokey/dev_ws/datasets/
├── train_data/
│   ├── slot1_*.hdf5    # 슬롯 1 (sg2_in_01 우측 상단)
│   ├── slot2_*.hdf5    # 슬롯 2 (sg2_in_01 우측 하단)
│   ├── slot3_*.hdf5    # 슬롯 3 (sg2_in_02 우측 상단)
│   └── slot4_*.hdf5    # 슬롯 4 (sg2_in_02 우측 하단)
└── stay.hdf5           # 안전 자세 (호밍용)

/home/rokey/dev_ws/box_assets/
└── *.usd               # 다양한 상자 USD 모델
```

---

*Last updated: 2026-06-12*
