# 🤖 SH5 최종 코드 전체 리뷰

> **파일:** `sh5_final.py` + `bg2_mock_publisher.py`  
> **작성일:** 2026-06-11  
> **상태:** D-Day 배포 준비 완료

---

## 1. 전체 아키텍처

```
[BG2 PC]
   └─ /sim/sg2_spawn_trigger 토픽 발행
         {"package_id": "PKG_20260612_001", "target_line": "sg2_in_01"}
              │
              ▼
[SH5 PC  —  sh5_final.py]
   STEP 1: spawn_box()         → 컨베이어 끝 고정 좌표에 상자 생성
   STEP 2: scan_qr()           → QR 인식 (또는 package_id 폴백)
   STEP 3: do_pick_and_place() → 로봇 팔 동작
   STEP 4: _report()           → DB에 적재 완료 보고
              │
              ▼
[DB / Control Tower]
   UPDATE packages SET workstation_id, slot_number, status='IN_WORKSTATION'
   슬롯 4 도달 시 → 자동 회전 + Pause 발행
```

---

## 2. DB 확정 인터페이스 (2가지만)

| 방향 | 인터페이스 | 데이터 |
|:---|:---|:---|
| **DB → SH5** | `/sim/sg2_spawn_trigger` 토픽 | `package_id`, `target_line`, `timestamp` |
| **SH5 → DB** | `ReportInboundProgress` 서비스 | `workstation_id`, `robot_id`, `filled_slots_count`, `package_id`, `workstation_qr_id`, `package_qr_id` |

---

## 3. 전체 기능 목록

### ① 상자 스폰 (`spawn_box`)

```
/sim/sg2_spawn_trigger 수신 → CONVEYOR_SPAWN 고정 좌표에 상자 Prim 생성

스폰 좌표 (수정 가능):
  sg2_in_01: (9.0,  1.5, 0.83)
  sg2_in_02: (9.0, -3.0, 0.83)
  sg2_in_03: (9.0, -7.5, 0.83)

상자 USD 있으면 → sh5_box.usd 로드
상자 USD 없으면 → 12cm 큐브로 자동 대체 (폴백)
Isaac Sim 없으면 → Mock (경로만 출력)
```

### ② QR 인식 (`scan_qr`)

```
USE_QR_CAMERA = False (기본):
  BG2 토픽의 package_id 그대로 사용
  PKG_20260612_001 → QR_20260612_001 자동 역변환

USE_QR_CAMERA = True:
  Top-View 카메라로 이미지 캡처
  OpenCV WeChatQRCode로 QR 디코딩
  QR_20260612_001 → PKG_20260612_001 변환 후 DB 보고
  실패 시 → package_id 폴백 자동 적용
```

### ③ Pick & Place (`do_pick_and_place`) — 3모드 자동 폴백

```
ACT_MODEL
  └─ AI 추론 실패 → HDF5 재생 시도
         └─ HDF5 실패 → Dummy Teleport (최종 안전망)

HDF5_REPLAY  ← D-Day 기본값
  └─ HDF5 실패 → Dummy Teleport

DUMMY_TELEPORT
  └─ 상자를 슬롯 좌표로 즉시 이동 (항상 성공)
```

슬롯 좌표 (로봇 기준 상대, 수정 가능):
```python
SLOT_LOCAL = {
    1: (0.0, -1.5, 1.2),
    2: (0.0, -1.5, 0.9),
    3: (0.0, -1.5, 0.6),
    4: (0.0, -1.5, 0.3),
}
```

### ④ DB 적재 보고 (`_report`)

```
ReportInboundProgress 서비스 호출

보내는 값:
  workstation_id     = "WS01" / "WS02" / "WS03"
  robot_id           = "sg2_in_01" / "sg2_in_02" / "sg2_in_03"
  filled_slots_count = 슬롯 번호 (1~4)
  package_id         = "PKG_20260612_001"
  workstation_qr_id  = "WORKSTATION_WS01"
  package_qr_id      = "QR_20260612_001"

DB 자동 처리:
  → packages.status = 'IN_WORKSTATION'
  → packages.slot_number = filled_slots_count
  → 슬롯 4 도달 시 작업대 회전 AMR 태스크 등록
```

### ⑤ Pause 인터락

```
/{robot_id}/pause_status 구독
  → DB가 True 발행 → SH5 일시정지 (작업대 회전 중)
  → DB가 False 발행 → SH5 재개
  → 4칸 완충 시 슬롯 카운터 자동 리셋
```

### ⑥ Mock 자동 투입

```
ROS2 없을 때 자동 활성화
  → 4초마다 PKG_MOCK_xxx를 3개 라인 순환 투입
  → 단독 터미널 테스트 가능
```

---

## 4. 설정값 (sh5_final.py 상단 수정)

```python
# ── 반드시 확인 ─────────────────────────────────────
PICK_AND_PLACE_MODE = "HDF5_REPLAY"   # D-Day 기본
# "ACT_MODEL"      ← 학습 완료 후 전환
# "DUMMY_TELEPORT" ← 통신 테스트

ACT_MODEL_PATH = "/home/rokey/dev_ws/models/unified_vision_act.pth"
# 학습 완료 후 실제 파일명으로 수정

USE_QR_CAMERA = False   # QR 카메라 없으면 False 유지

# ── 좌표 현장에서 확인 후 수정 ──────────────────────
CONVEYOR_SPAWN = {
    "sg2_in_01": (9.0,  1.5, 0.83),
    "sg2_in_02": (9.0, -3.0, 0.83),
    "sg2_in_03": (9.0, -7.5, 0.83),
}
ROBOT_POS = {
    "sg2_in_01": (7.5,  3.0, 0.0),
    "sg2_in_02": (7.5, -1.5, 0.0),
    "sg2_in_03": (7.5, -6.0, 0.0),
}
```

---

## 5. 필요한 환경 구축

### Python 패키지

```bash
pip install h5py opencv-python numpy
# QR 카메라 사용 시 (WeChatQRCode 포함)
pip install opencv-contrib-python
```

### 파일 의존성

| 파일 | 경로 | 없으면? |
|:---|:---|:---|
| `hdf5_replay_player.py` | `coupang_ws/scripts/` | HDF5 재생 불가 → Teleport 폴백 |
| `evaluate_test_vision.py` | `coupang_ws/scripts/` | ACT 추론 불가 → HDF5 폴백 |
| `sh5_qr_scanner.py` | `coupang_ws/scripts/` | QR 스캔 불가 → package_id 폴백 |
| `slot1_1.hdf5` ~ `slot4_2.hdf5` | `datasets/train_data/` | HDF5 재생 불가 |
| `unified_vision_act.pth` | `models/` | ACT 추론 불가 |
| `sh5_box.usd` | `assets/` | 큐브로 대체 (없어도 동작) |

### ROS 2 환경

```bash
source /home/rokey/dev_ws/cobot3_ws_ref/install/setup.bash
export ROS_DOMAIN_ID=119
export ROS_LOCALHOST_ONLY=0   # 분산 환경 (다른 PC와 연결 시)
```

### DB 환경 (DB팀이 준비)

```
✅ robots 테이블: sg2_in_01~03 (init.sql에 이미 있음)
✅ workstations 테이블: WS01~WS10 (init.sql에 이미 있음)
❌ packages 테이블: CSV 업로드 필요
   → http://localhost:8009 대시보드 → CSV 업로드
   → 파일: /home/rokey/dev_ws/qr_data/packages_2026-06-12.csv
```

### Isaac Sim 씬 (final_coupan.usd)

```
□ SH5 로봇 3대 배치 확인 (/World/SH5_01 ~ SH5_03)
□ CONVEYOR_SPAWN 좌표 실제 씬과 일치 확인
□ SLOT_LOCAL 좌표 작업대 위치와 일치 확인
□ (QR 사용 시) Top-View 카메라 3대 배치 (320×240 이상)
```

---

## 6. 실행 방법

### A. Isaac Sim (D-Day)

```python
# Script Editor에서
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_final.py', encoding='utf-8').read())
```

### B. 터미널 단독 테스트 (학습 중에도 가능)

```bash
# 새 터미널 탭에서 (학습 터미널 영향 없음)
python3 /home/rokey/dev_ws/coupang_ws/scripts/sh5_final.py
# → Mock 자동 투입으로 파이프라인 전체 확인 가능
```

### C. BG2 없이 토픽 수동 발행

```bash
# 자동 순환 (5초마다)
python3 bg2_mock_publisher.py --mode auto --interval 5

# 키보드 수동 (1=라인1, 2=라인2, 3=라인3, 엔터=자동)
python3 bg2_mock_publisher.py --mode manual

# 1회만
python3 bg2_mock_publisher.py --mode once --line sg2_in_01
```

### D. 상태 확인 (Script Editor 콘솔)

```python
# 각 라인 슬롯 채움 현황
[(l.line_id, l.filled) for l in controller.lines]
# → [('sg2_in_01', 2), ('sg2_in_02', 0), ('sg2_in_03', 1)]

# 수동 투입
controller.lines[0].queue.put({'package_id': 'PKG_TEST', 'target_line': 'sg2_in_01'})
```

---

## 7. 즉각 대응표

| 콘솔 출력 | 원인 | 조치 |
|:---|:---|:---|
| `⚠️ ROS 2 없음` | ROS2 미연결 | Mock 자동 진행 — 무시 가능 |
| `⚠️ HDF5 없음` | hdf5_replay_player.py 없음 | 경로 확인 후 재실행 |
| `⚠️ ACT 없음` | torch/evaluate_test_vision 없음 | HDF5로 자동 폴백 |
| `[보고] 서비스 없음` | control_tower 미연결 | DB팀 노드 확인 |
| `[QR] 미검출` | 카메라 각도 문제 | USE_QR_CAMERA=False로 변경 |
| `HDF5 파일 없음` | datasets 경로 오류 | HDF5_BASE_DIR 수정 |

---

## 8. 모드 전환 (시연 중 즉각 변경)

```python
# sh5_final.py 89번 줄 수정 후 재실행
PICK_AND_PLACE_MODE = "HDF5_REPLAY"   # 안정 최우선
PICK_AND_PLACE_MODE = "ACT_MODEL"     # AI 추론 (학습 완료 후)
PICK_AND_PLACE_MODE = "DUMMY_TELEPORT" # 통신 검증

exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_final.py', encoding='utf-8').read())
```

---

*저장 경로: `/home/rokey/dev_ws/md_file/SH5_FINAL_REVIEW.md`*
