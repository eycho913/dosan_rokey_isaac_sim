# 🤖 SH5 물류 로봇 시연 실행 가이드 (담당자용)

> **작성일:** 2026-06-11 (최종 업데이트)  
> **담당자:** SH5 로봇 3대 통합 제어 / 모방 학습 AI 파이프라인  
> **핵심 원칙:** 어떤 상황에도 `HDF5_REPLAY` 모드로 시연은 반드시 성공한다.
> **최종 실행 파일:** `sh5_final.py`

---

## 📁 내가 담당한 파일 목록

| 파일 | 경로 | 역할 |
|:---|:---|:---|
| `sh5_final.py` | `coupang_ws/scripts/` | **D-Day 메인 컨트롤러** (DB 확정 인터페이스 반영) |
| `sh5_spawn_controller.py` | `coupang_ws/scripts/` | 이전 버전 컨트롤러 (레거시) |
| `sh5_qr_scanner.py` | `coupang_ws/scripts/` | **QR 카메라 인식 모듈** (Plan B) |
| `hdf5_replay_player.py` | `coupang_ws/scripts/` | **HDF5 재생 엔진** — VR 궤적 주입 + 홈 복귀 |
| `evaluate_test_vision.py` | `coupang_ws/scripts/` | **AI 추론 엔진** — ACT 비전 모델 |
| `sh5_solo_demo.py` | `coupang_ws/scripts/` | 단독 테스트용 (DB/ROS2 없이 동작) |

---

## ✅ D-Day 시작 전 사전 점검 체크리스트

### 📋 Isaac Sim 씬 설정

```
□ final_coupan.usd에 SH5 로봇 3대 배치 확인
  → Stage 패널에서 /World/SH5_01, SH5_02, SH5_03 경로 확인
  → 없으면 sh5_final.py가 자동 스폰 (BOX_USD 경로만 맞으면 됨)

□ 컨베이어 끝단 XY 좌표 → CONVEYOR_SPAWN 맞추기
  → sh5_final.py 상단 CONVEYOR_SPAWN 딕셔너리 확인
  → 실제 씬 좌표와 다르면 수정:
     "sg2_in_01": (9.0,  1.5, 0.83),
     "sg2_in_02": (9.0, -3.0, 0.83),
     "sg2_in_03": (9.0, -7.5, 0.83),

□ 작업대(랙) 슬롯 XY 좌표 → SLOT_LOCAL 맞추기
  → 실제 작업대 높이/위치 맞게 수정:
     1: (0.0, -1.5, 1.2),
     2: (0.0, -1.5, 0.9),
     3: (0.0, -1.5, 0.6),
     4: (0.0, -1.5, 0.3),

□ (QR 사용 시) Top-View 카메라 3대 배치
  → 카메라 해상도: 320×240 이상
  → 컨베이어 끝단 위 정면에서 내려다보는 각도
  → Play 전 카메라 Prim 경로 확인: find_camera_prims() 실행
```

### 📋 코드 설정 (sh5_final.py 상단)

```
□ PICK_AND_PLACE_MODE 설정
  → AI 모델 있고 안정적 → "ACT_MODEL"
  → 기본/보험     → "HDF5_REPLAY"  ← D-Day 기본값
  → 통신 테스트   → "DUMMY_TELEPORT"

□ USE_QR_CAMERA 설정
  → 카메라 없거나 테스트 생략 → False (기본)
  → Top-View 카메라 있고 QR 부착 상자 사용 시 → True

□ (USE_QR_CAMERA=True 시) CAMERA_PRIMS 경로 설정
  → find_camera_prims() 실행 후 출력된 경로로 교체:
     "sg2_in_01": "/World/실제TopCamera경로",
     "sg2_in_02": "/World/실제TopCamera경로2",
     "sg2_in_03": "/World/실제TopCamera경로3",

□ ACT_MODEL_PATH 확인 (ACT 모드 시만)
  → /home/rokey/dev_ws/models/augmented_sh5_vision_act_20ep.pth 존재 확인
```

### 📋 데이터/파일 확인

```bash
# HDF5 파일 (HDF5_REPLAY 모드 필수)
ls -lh /home/rokey/dev_ws/datasets/train_data/slot*.hdf5
# → slot1_1.hdf5 ~ slot4_2.hdf5 8개 있어야 함

# AI 모델 파일 (ACT_MODEL 모드 시)
ls -lh /home/rokey/dev_ws/models/

# 스크립트 파일
ls /home/rokey/dev_ws/coupang_ws/scripts/sh5_final.py
ls /home/rokey/dev_ws/coupang_ws/scripts/sh5_qr_scanner.py
ls /home/rokey/dev_ws/coupang_ws/scripts/hdf5_replay_player.py
```

---

## 🚀 시연 실행 순서

### STEP 1: DB팀 노드 연결 확인 (터미널)

```bash
source /home/rokey/dev_ws/cobot3_ws_ref/install/setup.bash
export ROS_DOMAIN_ID=119
ros2 topic list | grep sg2_spawn
```

출력에 `/sim/sg2_spawn_trigger` 보이면 ✅  
안 보이면 → DB팀에 문의. 없어도 Mock 모드로 진행 가능.

---

### STEP 2: Isaac Sim 열기

1. Isaac Sim 실행
2. `File → Open` → `/home/rokey/dev_ws/assets/final_coupan.usd`
3. **▶ Play 버튼** 클릭

> ⚠️ Play 버튼 누르기 전에 스크립트 실행 금지

---

### STEP 3: 카메라 경로 확인 (QR 사용 시만)

Script Editor에서:
```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_qr_scanner.py', encoding='utf-8').read())
find_camera_prims()   # 출력된 경로를 CAMERA_PRIMS에 입력
```

---

### STEP 4: AMR 팀 스크립트 먼저 실행

```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/amr_live_existing_stage_true8_qr_camera_controller_gpu.py', encoding='utf-8').read())
```

AMR팀 **"SG2 로봇 5대 로드 완료"** 확인 후 다음 단계.

---

### STEP 5: SH5 최종 컨트롤러 실행 ← **핵심**

```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_final.py', encoding='utf-8').read())
```

---

### STEP 6: ros2 토픽 수신 확인 (별도 터미널)

```bash
source /home/rokey/dev_ws/cobot3_ws_ref/install/setup.bash
export ROS_DOMAIN_ID=119
ros2 topic echo /sim/sg2_spawn_trigger
```

BG2가 상자를 보내면 아래처럼 출력돼야 함:
```json
{"package_id": "PKG_20260612_001", "target_line": "sg2_in_01", "timestamp": 1234567890}
```

---

### STEP 7: 정상 동작 확인 (콘솔 출력)

```
[SH5] ✅ ROS 2
[SH5] ✅ HDF5 모듈
[SH5 Final] 초기화 중...
[sg2_in_01] 초기화 | WS=WS_01 | 모드=HDF5_REPLAY
[sg2_in_02] 초기화 | WS=WS_02 | 모드=HDF5_REPLAY
[sg2_in_03] 초기화 | WS=WS_03 | 모드=HDF5_REPLAY
[Node] ✅ sh5_final_node 가동
[Node] 📡 /sim/sg2_spawn_trigger 대기 중
[Controller] 🚀 시연 준비 완료!
```

---

## 🎬 자동 동작 흐름

```
① BG2 PC에서 상자 디스폰
     ↓
② /sim/sg2_spawn_trigger 수신
   → [Node] 🚨 트리거 수신: PKG_001 → sg2_in_01
     ↓
③ 컨베이어 끝 지정 위치에 상자 리스폰
   → [Spawn] 📦 /World/SH5Box_01_PKG_001
     ↓
④ (USE_QR_CAMERA=True 시) 카메라로 QR 스캔
   → [QR] ✅ 인식: 'PKG_001' @ 픽셀(320,240)
   → (False 시) package_id 그대로 사용
     ↓
⑤ 슬롯 결정 (1→2→3→4 순차)
     ↓
⑥ Pick & Place 실행
   → [P&P] 🎬 HDF5 재생 슬롯1
   → [Replay] 🎬 궤적 재생 완료!
   → [Replay] 🏠 홈 복귀 완료
     ↓
⑦ DB 적재 보고 (ReportInboundProgress)
   → [보고] ✅ DB 갱신 완료 | WS_01 슬롯1 ← PKG_001
     ↓
⑧ 4칸 완충 시 → 슬롯 카운터 리셋 (DB가 회전/교체 처리)
   → [sg2_in_01] 🔄 4칸 완충 → 슬롯 리셋
     ↓
⑨ Pause 수신 시 대기 → Resume 수신 시 재개
   → [Node] ⏸️ 일시정지 (DB 회전 중): sg2_in_01
   → [Node] ▶️ 재개: sg2_in_01
     ↓
⑩ 다음 상자 대기 → 사이클 반복
```

---

## 🔧 발표 중 즉각 대응

### 모드 전환 (30초 이내)

```python
# sh5_final.py 상단 수정 후 재실행
PICK_AND_PLACE_MODE = "ACT_MODEL"      # AI 추론
PICK_AND_PLACE_MODE = "HDF5_REPLAY"   # HDF5 재생 ← 롤백
PICK_AND_PLACE_MODE = "DUMMY_TELEPORT" # 순간이동

exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_final.py', encoding='utf-8').read())
```

### 수동 상자 투입 (테스트)

```python
# Script Editor 콘솔에서 직접 입력
controller.lines[0].queue.put({'package_id': 'PKG_TEST', 'target_line': 'sg2_in_01'})
controller.lines[1].queue.put({'package_id': 'PKG_TEST', 'target_line': 'sg2_in_02'})
controller.lines[2].queue.put({'package_id': 'PKG_TEST', 'target_line': 'sg2_in_03'})
```

### 상태 확인

```python
[(l.line_id, l.filled) for l in controller.lines]
# → [('sg2_in_01', 2), ('sg2_in_02', 0), ('sg2_in_03', 1)]
```

### DB 없이 단독 테스트

```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_solo_demo.py', encoding='utf-8').read())
demo.auto_start()   # 자동 순환 투입
```

---

## 🚨 문제별 즉각 대응표

| 콘솔 출력 | 의미 | 조치 |
|:---|:---|:---|
| `✅ ROS 2` | 정상 | 계속 진행 |
| `⚠️ ROS 2 없음` | DB 미연결 | Mock 모드 자동 진행 — 그냥 두면 됨 |
| `⚠️ QR 스캐너 없음` | qr_scanner 없음 | package_id 폴백 자동 — 무시 |
| `[QR] 이미지 없음` | 카메라 Prim 오류 | CAMERA_PRIMS 경로 수정 또는 USE_QR_CAMERA=False |
| `[QR] 미검출` | QR 코드 안 보임 | 카메라 각도/조명 조정 또는 USE_QR_CAMERA=False |
| `[보고] 서비스 없음` | control_tower 미연결 | DB팀 노드 확인 |
| `HDF5 없음` | dataset 경로 오류 | hdf5_replay_player.py HDF5_BASE_DIR 수정 |
| `에러 쏟아짐` | 코드/경로 오류 | exec() 재실행 |

---

## 📌 발표 중 말할 포인트

1. **"BG2 분류 로봇이 상자를 보내면 /sim/sg2_spawn_trigger 토픽으로 SH5가 수신합니다."**

2. **"(USE_QR_CAMERA=True 시) Top-View 카메라로 상자 QR을 인식하여 package_id를 확정합니다."**

3. **"VR로 수집한 전문가 시연 데이터(HDF5)를 로봇 63개 관절에 직접 주입하여 픽앤플레이스를 재현합니다."**

4. **"4개 슬롯이 채워지면 ReportInboundProgress로 DB에 보고하고, DB가 자동으로 AMR에 작업대 회전/교체 명령을 내립니다."**

5. **"AI 모델(ACT) → HDF5 재생 → 순간이동 순으로 자동 폴백 체인이 구성되어 있어 어떤 상황에도 시연이 멈추지 않습니다."**

---

*저장 경로: `/home/rokey/dev_ws/md_file/SH5_DEMO_EXECUTION_GUIDE.md`*
