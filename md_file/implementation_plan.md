# SH5 Robot ACT Manipulation Implementation Plan

본 문서는 SH5 로봇의 다관절 파지 및 분류 성능 극대화를 위한 **Action Chunking with Transformers (ACT)** 도입 및 데이터 증강/검증 파이프라인의 이행 계획서입니다.

---

## 📅 1. 구현 목표 (Objectives)
- **플랫폼 전환:** FFW SG2 → **FFW SH5** (26-DOF 다관절 핸드 로봇)으로 전환하여 다품종 물류 조작 가능성 확보.
- **모방 학습 고도화:** 단일 프레임 MLP BC의 행동 튐 현상 극복을 위해 **ACT (Action Chunking with Transformers)** 도입.
- **데이터 파이프라인 무결성 확보:** VR 시연 데이터 손실 최소화, 증강 기법 검증, 시뮬레이션 1:1 시각 검증 툴 마련.

---

## 🛠️ 2. 이행 단계 및 진행 상황 (Milestones)

### Phase 1: VR 원격 조작 및 데이터 수집 인프라 구축 [✅ 완료]
- Vuer WebXR & ROS2 통신, OpenXR & Differential IK 구현.
- HDF5 데이터 로거: 상태(153차원)와 행동(66차원) 기록, 실패 에피소드 제외(C키) 기능.
- Magic Snapping: 로봇 손바닥 기준 로컬 좌표계 오프셋 연산 기반 파지 보조 로직.

### Phase 2: ACT 모델 아키텍처 개발 [✅ 완료]
- `train_act.py`: PyTorch 기반 CVAE + Transformer Encoder-Decoder 구조.
- Temporal Ensembling: 추론 시 오버랩 청크 가중치 평균으로 제어 안정성 극대화.
- Phase 가중치 Loss: 파지(3.0x), 리프트(1.5x), 슬롯 삽입(3.0x) 순간 강조.

### Phase 3: 관절 인덱스 오류 수정 [✅ 완료 - 2026-06-09]
- **발견:** SH5 Isaac Sim joint_names 순서가 예상과 전혀 다름 (arm_L/R 인터리브, lift_joint=[3]).
- **수정:** `train_act.py`, `train_bc.py`, `augment_slot3_to_slot4.py` 전면 수정.
- **실제 인덱스:**
  - arm_L: [8, 11, 13, 15, 17, 19, 21]
  - arm_R: [9, 12, 14, 16, 18, 20, 22]
  - finger_L: [23-27, 33-37, 43-47, 53-57]
  - finger_R: [28-32, 38-42, 48-52, 58-62]
  - lift_joint: [3]

### Phase 4: 데이터 증강 완성 (`augment_slot3_to_slot4.py`) [✅ 완료 - 2026-06-09]
- Slot3(오른손, 아래층 우측) → Slot4(왼손, 아래층 좌측) 증강.
- slot1 vs slot2 실측 비교로 올바른 변환 규칙 도출:
  - pose X/Y/Z, robot quat: 그대로 유지
  - box/rack qz: 반전
  - cmd_vel vy: 반전 (wz 유지)
  - arm 관절: 인터리브 쌍별 스왑 + 관절별 부호 규칙 [유지,반전,반전,유지,반전,유지,유지]
  - finger 관절: 쌍별 스왑 (부호 유지)

### Phase 5: 시각적 재생 검증 도구 (`replay_data.py`) [✅ 완료]
- `robot_pose` 루트 강제 매핑 및 Magic Snapping 재현으로 HDF5 궤적 무결성 검증.

### Phase 6: ACT 학습 실행 [🔄 진행 중 - 2026-06-09]
- RTX 5080 Laptop GPU (16GB VRAM, CUDA 12.8) 활용.
- 3개 슬롯 동시 학습 (batch=256, epoch=500, lr=3e-4).

| 슬롯 | 데이터 | 사용 손 | 상태 |
|---|---|---|---|
| Slot1 | 100 에피소드 | 오른손 | 🔄 학습 중 |
| Slot2 | 100 에피소드 | 왼손 | 🔄 학습 중 |
| Slot3 | 75 에피소드 | 오른손 | 🔄 학습 중 |
| Slot4 | 75 에피소드 (증강) | 왼손 | ⏳ 학습 예정 |


### Phase 7: QR 인식 + 메인 컨트롤러 통신 구현 [✅ 완료 - 2026-06-09]
- **`qr_scanner_node.py`**: 카메라로 QR 코드 인식 → `GetPackageRoute` 서비스로 DB 조회 → 슬롯 결정 → `/sh5/task_assignment` 발행.
- **`main_controller_bridge.py`**: 태스크 수신 → ACT 모델 로드 → 추론 → `ReportInboundProgress` 서비스로 관제탑에 진척도 보고.
- **Fleet 상태 토픽 발행 (인터페이스 v2.2 규격 준수)**:
  - `/fleet/workstation_states` (1Hz): 작업대 슬롯 채움 현황
  - `/fleet/amr_states` (1Hz): 로봇 현재 상태/위치/가용 여부
  - `/fleet/task_events` (이벤트 발생 시): ASSIGNED / PICKING / PLACING / COMPLETED
  - `/sh5/status` (1Hz): SH5 전용 상태 요약

---

## 📋 3. 남은 작업 (D-2, D-1)

### D-2 (내일)
1. **학습 완료 확인** → `Epoch [500/500]` 출력 + `.pth` 파일 생성 확인
2. **eval 스크립트 실행** → `eval_bc.py`로 시뮬레이션 정책 동작 검증
3. **Slot4 학습** → 증강 데이터로 slot4 모델 추가 학습

### D-1 (모레)
4. **슬롯 선택 통합 로직** → 입력 슬롯 번호에 따라 해당 모델 자동 로드
5. **최종 데모 완성 및 검증**

---

## 🗂️ 핵심 파일 목록

| 파일 | 설명 |
|---|---|
| `scripts/train_act.py` | ACT 학습 스크립트 (Phase 가중치 포함) |
| `scripts/train_bc.py` | MLP BC 학습 스크립트 |
| `scripts/augment_slot3_to_slot4.py` | Slot3→4 좌우 미러링 증강 |
| `scripts/replay_data.py` | HDF5 궤적 시뮬레이션 재생 검증 |
| `scripts/eval_bc.py` | 학습된 정책 시뮬레이션 평가 |
| `scripts/coupang_sh5_bringup.py` | SH5 Isaac Sim 브링업 |
| `scripts/qr_scanner_node.py` | **[신규]** QR 인식 → GetPackageRoute → 슬롯 결정 → 태스크 발행 |
| `scripts/main_controller_bridge.py` | **[신규]** 태스크 수신 → ACT 추론 → ReportInboundProgress + Fleet 상태 발행 |
| `datasets/slot1_...hdf5` | Slot1 데모 데이터 (100 에피소드) |
| `datasets/slot2_...hdf5` | Slot2 데모 데이터 (100 에피소드) |
| `datasets/slot3_...hdf5` | Slot3 데모 데이터 (75 에피소드) |
| `datasets/slot4_augmented.hdf5` | Slot4 증강 데이터 (75 에피소드) |
| `models/slot{1,2,3}_act_policy.pth` | 학습된 ACT 정책 모델 |

---

## 🔌 8. 인터페이스 통신 규약 (v2.2 기준)

### 발행 토픽 (SH5 → 메인 컨트롤러)

| 토픽 | 타입 | 주기 | 내용 |
|---|---|---|---|
| `/fleet/workstation_states` | `std_msgs/String` (JSON) | 1Hz | 작업대 슬롯 채움 현황 |
| `/fleet/amr_states` | `std_msgs/String` (JSON) | 1Hz | 로봇 상태/위치/배터리 |
| `/fleet/task_events` | `std_msgs/String` (JSON) | 이벤트 | 태스크 생애주기 로그 |
| `/sh5/status` | `std_msgs/String` (JSON) | 1Hz | SH5 전용 상태 요약 |

### 서비스 클라이언트 (SH5 → 메인 컨트롤러)

| 서비스 | 타입 | 사용 시점 |
|---|---|---|
| `/get_package_route` | `GetPackageRoute.srv` | QR 스캔 시 배송일→슬롯 조회 |
| `/report_inbound_progress` | `ReportInboundProgress.srv` | 슬롯 적재 완료 시 보고 |
