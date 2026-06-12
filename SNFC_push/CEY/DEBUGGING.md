# 🛠️ DEV_LOG — SH5 물류 자동화 개발 일지

> **담당자**: 최은예  
> **기간**: 2026-05-26 ~ 2026-06-12  
> **프로젝트**: Isaac Sim 기반 SH5 쌍팔 로봇 물류 자동화 시스템

---

## 2026-05-26 ~ 05-30 | 환경 구축 및 데이터 수집

### 씬 세팅
- `finalfac.usd` 로드하여 Isaac Sim 물류 창고 씬 구성
- SH5 쌍팔 로봇 3대 스폰 (`sg2_in_01`, `sg2_in_02`, `sg2_in_03`)
- 상자 USD (`box_assets/*.usd`) 각 라인별 배치

### VR 조작 인터페이스
- `coupang_sh5_bringup_v.py` 구현 — 키보드/VR로 로봇 원격 조작
- WASD 이동, R/T/C로 녹화 시작/저장/취소

### HDF5 녹화 파이프라인
- `VRDemonstrationLogger` 구현
- 저장 데이터: `joint_positions`, `box_pose`, `robot_pose`, `images/*`
- Left/Right/TopView 카메라 RGB 160×120 동시 저장

### Magic Snapping (수집 보조)
- 데이터 수집 중 파지 정확도 향상을 위한 보조 스냅 로직
- 손가락 닫힘 + 상자 근거리 시 자동 부착

---

## 2026-06-01 ~ 06-02 | 에피소드 수집 완료 + QR 연동

### 에피소드 수집
- 슬롯 1~4 각 100+ 에피소드 수집 (총 400+ 에피소드)
- 슬롯별 저장: `/datasets/train_data/slot{N}_*.hdf5`

### QR 카메라 연동
- TopView 카메라 + WeChatQRCode 기반 상자 위치 인식
- 카메라 픽셀 → 월드 좌표 변환 (homography)

---

## 2026-06-03 ~ 06-05 | 데이터 전처리 파이프라인

### 방해 팔 고정 처리
- **문제**: 데이터 수집 시 키보드 조작으로 비동작 팔이 움직여 학습에 방해
- **해결**: `freeze_idle_arms.py` — 비동작 팔 관절 궤적을 `stay.hdf5` 자세로 오버라이드
- **결과**: 학습 시 반대 팔 간섭 완전 제거

### 서브셋 추출
- `create_subset.py` — 방해 팔 제거된 에피소드 중 품질 좋은 것만 추출
- 출력: `/datasets/train_data/frozen_set/` (학습 + 재생 공통 사용)

### 데이터 증강
- `augment_data.py` — 좌우 미러링, 관절 가우시안 노이즈 추가
- `augment_slot3_to_slot4.py` — 슬롯 3을 좌우 반전하여 슬롯 4 데이터 생성
- `filter_dataset.py` — 실패 에피소드 (trajectory 품질 기준 미달) 제거

---

## 2026-06-06 ~ 06-07 | ACT 모델 학습

### Vision-ACT 학습 (Google Colab A100)
- `train_act_v2.py` 사용
- 150 epoch, batch_size=64
- 체크포인트 주기 저장 및 resume 기능

### HDF5 에피소드 로더 구현
- `hdf5_replay_player.py` — 슬롯별 랜덤 에피소드 로드
- offset 보정: 녹화 위치 → 라인별 실제 로봇 위치로 자동 보정
- 키: `joint_trajectory`, `box_trajectory`, `robot_trajectory`

---

## 2026-06-08 ~ 06-09 | ROS 2 연동 및 3-로봇 시스템

### ROS 2 브릿지
- `ros2_sh5_bridge.py` 구현
- `/sim/sg2_spawn_trigger` 구독 → `/tmp/sh5_queue.jsonl` 기록
- `check_warehouse_status` 서비스로 DB 중복 확인
- `report_inbound_progress` 서비스로 입고 완료 보고

### 3대 로봇 병렬 시스템
- `sh5_bringup_ros2_3robot.py` 구현
- `ReplayController` 3개가 독립적으로 상태머신 운영
- 상태: `IDLE → SCANNING → WAITING_DB → REPLAYING → HOMING → DONE`
- `SlotRegistry`: 고객별 슬롯 유지 할당 (같은 고객 → 항상 같은 슬롯)

### 파일 큐 인터페이스
| 파일 | 방향 | 내용 |
|------|------|------|
| `/tmp/sh5_queue.jsonl` | Bridge → Isaac | 패키지 투입 트리거 |
| `/tmp/sh5_qr_req.jsonl` | Isaac → Bridge | QR DB 확인 요청 |
| `/tmp/sh5_qr_result.jsonl` | Bridge → Isaac | DB 중복 체크 결과 |
| `/tmp/sh5_report_req.jsonl` | Isaac → Bridge | 입고 완료 보고 |
| `/tmp/sh5_pause.json` | Bridge → Isaac | 일시정지 신호 |
| `/tmp/sh5_ws_trigger.jsonl` | Bridge → Isaac | 작업대 Spawn/Despawn |

---

## 2026-06-10 | 재생 안정화 3종 적용

### [Fix 1] 워밍업 보간 (텔레포트 제거)
- **문제**: HDF5 재생 시작 시 현재 자세 → 첫 프레임 자세로 순간이동
- **해결**: `WARMUP_FRAMES=30` — 30프레임(약 1초) 선형 보간
- **결과**: 로봇이 자연스럽게 시작 자세로 이동

### [Fix 2] frozen_set 에피소드 사용
- **문제**: 원본 에피소드에 비동작 팔의 움직임이 포함되어 재생 시 방해
- **해결**: `FROZEN_SET_DIR` 설정으로 freeze_idle_arms 전처리 에피소드 사용

### [Fix 3] stay.hdf5 호밍
- **문제**: 복귀 모션 마지막 손 동작에 의해 쓰러짐 발생
- **해결**: `STAY_HDF5_PATH` — 안전 자세 HDF5 첫 프레임으로 복귀

---

## 2026-06-11 | HDF5-Guided Snapping (파지 안정화)

### 문제
- 손가락 파지 후 상자가 딜레이되어 손에 안착되지 않거나 공중에 떠 있는 현상
- 슬롯 2번(왼손)에 오른손 텔레포트 문제

### 해결
- **HDF5 가이드 링크 선택**: `box_trajectory` 기준으로 가장 가까운 로봇 링크 자동 선택
  → 왼/오른손 구분 자동화
- **ATTACH_FACTOR = 1.0**: 박스를 링크 중심에 완전 부착 (오프셋 0)
- **GRASP_DIST = 0.30**: 30cm 이내 접근 시 즉시 스냅 활성화
- **MAX_BOX_STEP = 3.0**: 속도 클램프 사실상 해제 (즉시 반응)
- **FINGER_OPEN_THRESH = 0.80**: 손가락 80% 열림 시 자동 릴리즈

---

## 2026-06-12 | yo-yo 현상 수정 + Spawn/Despawn 구현

### yo-yo 현상 원인 분석 및 수정

**증상**: 상자가 Z축 방향으로 진동하며 손에서 일정 offset 유지  
**원인 1**: 패키지 스폰 시 `kinematic_enabled=False`로 물리 활성화
- 중력이 박스를 아래로 당기고 `write_root_state_to_sim`이 다시 올리는 싸움

**원인 2**: `write_root_state_to_sim`이 내부에서 `setLinearVelocity/setAngularVelocity` 호출
- kinematic 바디에는 이 API가 금지 → PhysX 에러 1000개 누적 → 시뮬 강제 종료

**해결**:
1. 스폰 시 `kinematic_enabled=False` 코드 완전 제거 (항상 `kinematic=True`)
2. `_write_box_pose()` 헬퍼 구현:
   - `write_root_pose_to_sim` (pose만, velocity 없음) 우선 사용
   - 없으면 `USD XFormable` 직접 쓰기 (PhysX 완전 우회)

```
수정 전 라이프사이클:
  박스 생성: kinematic=True ✓
  스폰 시:   kinematic=False ← ❌ 물리 활성화
  재생 중:   중력 ↓ vs write_root_state ↑ → yo-yo 현상
  재생 완료: kinematic=True ✓

수정 후:
  박스 항상: kinematic=True → 중력 없음 → yo-yo 사라짐
  위치 쓰기: _write_box_pose() → velocity 설정 없음 → PhysX 에러 없음
```

### WorkstationManager 구현

- `/tmp/sh5_ws_trigger.jsonl` 폴링으로 SPAWN/DESPAWN 메시지 수신
- **DESPAWN**: RACK prim을 Z=-200으로 이동 (화면 밖)
- **SPAWN**: 원래 위치 또는 `WORKSTATION_SPAWN_POS` 좌표로 복원

```json
{"workstation_id": "WS02", "location": "sg2_in_01_A", "action": "DESPAWN"}
{"workstation_id": "WS02", "location": "sg2_in_01_A", "action": "SPAWN"}
```

**RACK 매핑** (`/World/FinalFac/RACK_NN`):
- `RACK_01` = sg2_out (출고 컨베이어)
- `RACK_02` = sg2_in_01_A (1번 라인 작업대)
- `RACK_03` = sg2_in_02_A (2번 라인 작업대)
- `RACK_04` = sg2_in_03_A (3번 라인 작업대)

---

## 📦 최종 파라미터 설정

```python
PLAYBACK_SPEED       = 2       # 2배속 재생
WARMUP_FRAMES        = 30      # 첫 프레임 보간 (~1초)
HOMING_FRAMES        = 120     # 호밍 보간 (~4초)
ATTACH_FACTOR        = 1.0     # 링크 완전 중심 부착
GRASP_DIST           = 0.30    # 스냅 활성화 거리 (m)
FINGER_OPEN_THRESH   = 0.80    # 손가락 열림 임계값 (rad)
MAX_BOX_STEP         = 3.0     # 속도 클램프 (m/frame)
FROZEN_SET_DIR       = /datasets/train_data/frozen_set
STAY_HDF5_PATH       = /datasets/stay.hdf5
RACK_PREFIX          = /World/FinalFac/
```

---

## ⚠️ 알려진 이슈

| 이슈 | 상태 | 비고 |
|------|------|------|
| kinematic 박스 velocity 설정 금지 | ✅ 해결 | `_write_box_pose()` 사용 |
| 첫 프레임 텔레포트 | ✅ 해결 | `WARMUP_FRAMES=30` |
| 비동작 팔 방해 | ✅ 해결 | frozen_set 사용 |
| 복귀 모션 쓰러짐 | ✅ 해결 | stay.hdf5 호밍 |
| 왼/오른손 선택 오류 | ✅ 해결 | HDF5-Guided Snapping |
| GPU PhysX 불안정 (RTX 5080) | ⚠️ 우회 | CPU 모드 사용 중 |
| RACK prim 경로 확인 필요 | ⚠️ 진행중 | `_find_prim` 디버그 출력으로 탐색 |
