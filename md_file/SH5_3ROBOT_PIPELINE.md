# SH5 3대 로봇 물류 파이프라인 — 개발 상세 문서

## 1. 시스템 개요

Isaac Sim 환경에서 3개의 입고 라인(sg2_in_01/02/03)을 3대의 SH5 양팔 로봇이 독립적으로 운용하는 물류 자동화 시스템입니다.

```
관제탑 (cobot3 WMS)
    ↓ ROS2 토픽
ros2_sh5_bridge.py  ←→  /tmp/ 파일 큐
    ↓
sh5_bringup_ros2_3robot.py (Isaac Sim)
    ↓
3대 SH5 로봇 독립 작동
```

---

## 2. 핵심 파일 목록

| 파일 | 역할 |
|---|---|
| `scripts/sh5_bringup_ros2_3robot.py` | Isaac Sim 메인 시뮬레이터 |
| `scripts/ros2_sh5_bridge.py` | ROS2 ↔ Isaac Sim 파일 큐 브릿지 |
| `scripts/send_by_date.py` | CSV 날짜 기반 패키지 투입 스크립트 |

---

## 3. 작업대(Workstation) — DB 매핑 규칙

> **중요**: WS01은 출고 대기 창고(stage_01)이며 입고 라인이 **아닙니다**.

| Isaac Sim 라인 ID | USD 작업대 | DB Workstation ID | 배송일 |
|---|---|---|---|
| `sg2_in_01` | RACK_02 | **WS02** | 0608 당일 |
| `sg2_in_02` | RACK_03 | **WS03** | 0609 익일 |
| `sg2_in_03` | RACK_04 | **WS04** | 0610 모레 |

---

## 4. 6월 8일 이월 재고 초기 배치 기능

### 4.1 배경

`init_june_8th_state.py`에 정의된 발표 데모 초기 상태:
- WS02 (sg2_in_01): 이월 상자 5칸 차 있음 (슬롯 1~5)
- WS03 (sg2_in_02): 이월 상자 5칸 차 있음 (슬롯 1~5)
- WS04 (sg2_in_03): 비어 있음

### 4.2 구현 (`spawn_initial_stock()`)

시뮬레이터 시작 시 자동 실행. 이월 상자를 씬에 스폰하고, SlotRegistry를 사전 등록합니다.

```python
# 설정 상수
JUNE8_INITIAL_STOCK = {
    "sg2_in_01": [
        {"package_id": "PKG_20260607_008", "customer_name": "오주원", "slot": 1},
        ...  # 총 5개
    ],
    "sg2_in_02": [
        {"package_id": "PKG_20260607_001", "customer_name": "정서준", "slot": 1},
        ...  # 총 5개
    ],
    # sg2_in_03: 이월 없음
}
```

### 4.3 슬롯 좌표 자동 추출 (`_auto_detect_slot_positions()`)

HDF5 frozen_set 에피소드의 `box_trajectory` **마지막 프레임** = 최종 안착 위치를 읽어 자동 계산.

```python
offset = robot_pos - initial_robot  # 씬 오프셋 보정
final_world = box_traj[-1, :3] + offset
```

- `slot1~4_*.hdf5`: 4개의 팔 포지션 (LINE_TO_SLOT과 무관하게 항상 동일 파일 사용)
- 슬롯 5: 슬롯 4 위치 + 0.3m 위로 자동 추정

### 4.4 SlotRegistry 사전 등록

이월 5칸 등록 → `next_slot = 6` → **새 패키지는 자동으로 슬롯 6번부터** 배정

---

## 5. 날짜 기반 패키지 투입 (`send_by_date.py`)

CSV `route_zone` 날짜에 따라 올바른 라인으로 분배합니다.

```python
ROUTE_TO_LINE = {
    "2026-06-08": "sg2_in_01",  # 오늘 당일
    "2026-06-09": "sg2_in_02",  # 익일
    "2026-06-10": "sg2_in_03",  # 모레
}
```

### 실행 순서

```bash
# 터미널 1: 시뮬레이터 먼저 실행
~/.local/share/ov/pkg/isaac_sim-2023.1.1/python.sh ~/dev_ws/coupang_ws/scripts/sh5_bringup_ros2_3robot.py

# 터미널 2: 브릿지 실행
python3 ~/dev_ws/coupang_ws/scripts/ros2_sh5_bridge.py

# 터미널 3: 패키지 투입 (시뮬레이터 완전 로드 후)
rm -f /tmp/sh5_queue.jsonl
python3 ~/dev_ws/coupang_ws/scripts/send_by_date.py
```

---

## 6. 라인별 독립 Pause/Resume 기능

### 6.1 배경 및 문제

기존에는 `/tmp/sh5_pause.json` 단 하나를 모든 로봇이 공유 → 어느 라인에서 pause 신호가 와도 **전체 정지**

### 6.2 수정 내용

#### 브릿지 (`ros2_sh5_bridge.py`)

`_make_pause_callback(robot_id)` 팩토리 함수로 라인별 클로저 생성:

```python
def _make_pause_callback(self, robot_id: str):
    pause_file = f"/tmp/sh5_pause_{robot_id}.json"

    def _on_pause(msg: Bool):
        with open(pause_file, "w") as f:
            json.dump({"paused": bool(msg.data)}, f)
    return _on_pause
```

#### Isaac Sim (`sh5_bringup_ros2_3robot.py`)

각 `ReplayController`가 자신의 `line_id` 파일만 폴링:

```python
# 우선순위:
# 1. /tmp/sh5_pause_{line_id}.json  (라인별 개별)
# 2. /tmp/sh5_pause.json            (전체 공통 폴백)
line_pause_file = PAUSE_FILE_TEMPLATE.format(line_id=self._line_id_key)
```

### 6.3 라인별 정지/재개 명령어

```bash
# 1번 라인만 정지
ros2 topic pub --once /sg2_in_01/pause_status std_msgs/msg/Bool "{data: true}"

# 1번 라인 재개
ros2 topic pub --once /sg2_in_01/pause_status std_msgs/msg/Bool "{data: false}"

# 2번, 3번도 동일 패턴
ros2 topic pub --once /sg2_in_02/pause_status std_msgs/msg/Bool "{data: true}"
ros2 topic pub --once /sg2_in_03/pause_status std_msgs/msg/Bool "{data: true}"
```

---

## 7. 처리 완료 상자 자동 숨김

### 문제
재생 완료 후 상자 prim이 마지막 위치에 그대로 남아 바닥에 쌓이는 현상.

### 해결
`DONE` 상태 진입 시 다음 패키지 대기 전에 상자를 `Z=-10`으로 이동:

```python
elif self.state == self.DONE:
    # ... 보고 기록 ...
    # 처리 완료 상자 숨김
    hide_pos  = torch.tensor([0.0, 0.0, -10.0], dtype=torch.float32)
    hide_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    self._write_box_pose(hide_pos, hide_quat)
    self.state = self.IDLE
```

- 물리 설정(kinematic/mass) 변경 없음 — 위치만 이동
- 다음 패키지 스폰 시 해당 위치로 자동 복귀

---

## 8. 주요 버그 수정 이력

| 날짜 | 버그 | 수정 내용 |
|---|---|---|
| 2026-06-12 | `WORKSTATION_ID` 매핑 오류 | `sg2_in_01→WS01`을 `sg2_in_01→WS02`로 수정 (WS01은 출고창고) |
| 2026-06-12 | 모든 라인 동시 pause | 라인별 독립 pause 파일 구조로 분리 |
| 2026-06-12 | 슬롯 좌표 자동추출 오류 | `LINE_TO_SLOT` 기반 HDF5 선택 제거, 항상 `slot1~4` 사용 |

---

## 9. 파일 큐 구조 (`/tmp/`)

| 파일 | 방향 | 내용 |
|---|---|---|
| `sh5_queue.jsonl` | bridge → Isaac | 패키지 투입 트리거 |
| `sh5_qr_req.jsonl` | Isaac → bridge | QR check 요청 |
| `sh5_qr_result.jsonl` | bridge → Isaac | DB 중복 확인 결과 |
| `sh5_report_req.jsonl` | Isaac → bridge | 입고 보고 요청 |
| `sh5_pause_{line_id}.json` | bridge → Isaac | 라인별 pause 신호 |
| `sh5_ws_trigger.jsonl` | 관제탑 → Isaac | 작업대 Spawn/Despawn |
