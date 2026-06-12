# DEV_LOG.md — SH5 3대 로봇 개발 일지

---

## 2026-06-12 개발 세션

### 1. 날짜 기반 패키지 분배 스크립트 (`send_by_date.py`)
- CSV `route_zone` 날짜에 따라 패키지를 올바른 라인으로 자동 분배
- `0608 → sg2_in_01`, `0609 → sg2_in_02`, `0610 → sg2_in_03`
- 5초 간격으로 20개 순차 투입

### 2. 6월 8일 이월 재고 초기 배치 기능
- `init_june_8th_state.py` DB 상태와 시뮬레이터 씬을 동기화
- 시뮬레이터 시작 시 `spawn_initial_stock()` 자동 실행
- WS02/WS03 슬롯 1~5 이월 상자 스폰 + SlotRegistry 사전 등록
- 슬롯 좌표는 HDF5 frozen_set에서 자동 추출 (`_auto_detect_slot_positions()`)
- 결과: 신규 패키지는 자동으로 슬롯 6번부터 배정

### 3. Workstation ID 버그 수정
- **원인**: `WORKSTATION_ID = {"sg2_in_01": "WS01", ...}` 오매핑
- WS01은 출고 대기 창고(stage_01)라 DB 입고 업데이트 실패
- **수정**: `sg2_in_01→WS02`, `sg2_in_02→WS03`, `sg2_in_03→WS04`
- `WORKSTATION_QR`도 `WORKSTATION_WS02/03/04`로 수정

### 4. 라인별 독립 Pause 기능
- **원인**: 단일 `/tmp/sh5_pause.json` 공유 → 전체 동시 정지
- 브릿지: `_make_pause_callback(robot_id)` 팩토리로 라인별 파일 쓰기
- Isaac Sim: `ReplayController`에 `line_id` 주입, 자신의 파일만 폴링
- 결과: 각 라인 독립 정지/재개 가능

### 5. 처리 완료 상자 자동 숨김
- 재생 완료 후 상자가 바닥에 쌓이는 시각적 문제 해결
- `DONE` 상태에서 `_write_box_pose(Z=-10)` 호출
- 물리 설정 변경 없음, 안전한 위치 이동만 수행

---

## 관련 파일

- `scripts/sh5_bringup_ros2_3robot.py` — 메인 시뮬레이터 (다수 수정)
- `scripts/ros2_sh5_bridge.py` — 브릿지 (pause 라인별 분리)
- `scripts/send_by_date.py` — 신규: 날짜 기반 투입 스크립트
