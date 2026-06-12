# 🔍 SH5 시나리오 구현 가능성 전체 분석

> 작성일: 2026-06-11 | 기준 파일: `sh5_final_2.py` + `hdf5_replay_player.py`

---

## 1. HDF5 실측 데이터 분석 (핵심 발견)

```
실제 VR 녹화 환경 (coupang_sh5_bringup_v.py):

  robot_initial_pose ≈ (0.001, 0.000, 0.000)  ← 로봇이 거의 원점에서 녹화
  box_relative_pos   ≈ (+0.707, -0.001, +0.817) ← 로봇 앞 70cm, 높이 82cm
  rack_relative_pos  ≈ (0.000, -1.500, 0.000)   ← 로봇 좌측 1.5m

슬롯별 상대 좌표 (모두 동일 — 상자 시작 위치 고정):
  슬롯1: box_rel=(+0.705, -0.001, +0.817) | 877 프레임
  슬롯2: box_rel=(+0.709, -0.001, +0.817) | 1303 프레임
  슬롯3: box_rel=(+0.708, -0.001, +0.817) | 893 프레임
  슬롯4: box_rel=(+0.708, -0.001, +0.817) | 1194 프레임
```

> **✅ 핵심 결론**: 4개 슬롯 모두 상자의 상대 시작 위치가 동일합니다.
> Robot offset 계산이 완벽하게 적용 가능합니다.

---

## 2. 시나리오별 구현 가능성 판단

| 시나리오 단계 | 구현 상태 | 가능성 | 비고 |
|:---|:---:|:---:|:---|
| BG2 → spawn trigger 수신 | ✅ 구현됨 | 🟢 확실 | ROS2 연결만 하면 됨 |
| HDF5에서 상자 스폰 위치 계산 | ✅ 구현됨 | 🟢 확실 | offset 로직 검증 완료 |
| 상자 Prim 스폰 (Isaac Sim) | ✅ 구현됨 | 🟢 확실 | USD or Cube 자동 선택 |
| check_warehouse_status | ✅ 구현됨 | 🟡 조건부 | DB팀 서비스 가동 필요 |
| 중복 시 상자 제거 & 스킵 | ✅ 구현됨 | 🟢 확실 | |
| HDF5 Replay pick&place | ✅ 구현됨 | 🟡 조건부 | robot_art 연결 필요 |
| report_inbound_progress | ✅ 구현됨 | 🟡 조건부 | DB팀 서비스 가동 필요 |
| 순차 실행 (1대씩) | ✅ 구현됨 | 🟢 확실 | SEQUENTIAL_MODE=True |

---

## 3. 완전 구현 가능성 판단

### ✅ 완전히 가능한 것

```
① 토픽/서비스 통신 (ROS2 연결 후)
② 상자 스폰 + offset 보정
③ 순차 pick&place 실행 구조
④ DB 적재 보고
⑤ Mock 모드 (BG2/DB 없이 터미널 단독 테스트)
```

### ⚠️ 조건부 가능 (해결 필요)

```
① robot_art 연결 문제
   - 현재: robot_art = None (로봇 관절 제어 불가)
   - 해결: Isaac Sim Stage에서 SH5 Articulation 객체를 찾아서 연결해야 함
   - 방법: stage 탐색 후 로봇 Prim 경로로 Robot() 객체 생성

② HDF5 상자 위치 ≠ final_coupan.usd 씬 컨베이어 위치
   - 녹화 환경: 로봇이 (0,0,0)에 있음
   - 실제 씬: 로봇이 (7.5, 3.0, 0) 등에 배치됨
   - offset 계산: box_spawn = (0.707, -0.001, 0.817) + (7.5, 3.0, 0) = (8.207, 2.999, 0.817)
   - ✅ offset 로직이 이미 이걸 처리함

③ Isaac Sim 물리 안정화 시간
   - 상자 스폰 후 0.4초 대기 → 부족할 수 있음
   - 튜닝 필요: 0.4s → 0.8~1.0s로 늘릴 수 있음
```

### ❌ 현재 미구현 / 외부 의존

```
① robot_art = None → 관절값 주입 불가
   → HDF5 Replay 시 로봇 팔이 안 움직임
   → Dummy Teleport는 동작함 (상자만 이동)

② DB check_warehouse_status 서비스
   → DB팀이 control_tower 노드를 실행해야 함
   → 없으면 Mock(항상 신규)으로 처리됨

③ BG2 실제 토픽
   → bg2_mock_publisher.py로 대체 가능
```

---

## 4. 예상 문제 & 해결방법

### 🔴 HIGH - robot_art 미연결

**증상**: HDF5 재생 시 상자만 이동, 로봇 팔 안 움직임

**원인**: `robot_art = None`으로 `_apply_joint_positions()`가 즉시 리턴

**해결**:
```python
# Isaac Sim Script Editor에서 먼저 실행
from omni.isaac.core.robots import Robot
robot_prim_path = "/World/SH5_01"   # ← 실제 Prim 경로로 수정
robot_art = Robot(prim_path=robot_prim_path)
robot_art.initialize()

# 그 다음 sh5_final_2.py 실행 후
controller.lines[0].robot_art = robot_art  # 수동 연결
```

---

### 🔴 HIGH - HDF5 replay가 Isaac Sim 프레임을 블로킹

**증상**: HDF5 재생(763~1303 프레임) 동안 Isaac Sim 전체 시뮬레이션 멈춤

**원인**: `time.sleep(0.05)` × N 프레임이 update callback 안에서 동작

**해결**: HDF5 replay를 별도 스레드에서 실행
```python
threading.Thread(target=player.play_episode, args=(episode,), daemon=True).start()
```
단, 이 경우 `_busy` 플래그 관리가 더 복잡해짐

---

### 🟡 MEDIUM - 상자 스폰 위치 미세 오차

**증상**: 로봇이 허공을 잡거나 상자를 약간 빗나감

**원인**: 
- ROBOT_POS 딕셔너리 값이 실제 씬 좌표와 ±0.01m 오차
- offset = (8.207, 2.999, 0.817) → 실제는 (8.21, 3.00, 0.82)로 미세 차이

**튜닝 방법**:
```python
# Script Editor에서 실제 로봇 위치 확인
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/SH5_01")
xf = UsdGeom.Xformable(prim)
print(xf.GetLocalTransformation())

# ROBOT_POS를 실측값으로 업데이트
ROBOT_POS = {"sg2_in_01": (실측X, 실측Y, 실측Z)}
```

---

### 🟡 MEDIUM - 물리 안정화 시간 부족

**증상**: 상자 스폰 직후 픽앤플레이스 시작 → 상자가 아직 움직이는 상태

**원인**: `time.sleep(0.4)` 가 짧을 수 있음

**튜닝**:
```python
time.sleep(0.4)  → time.sleep(1.0)  # 보수적으로 1초
```

---

### 🟡 MEDIUM - check_warehouse_status 타임아웃

**증상**: DB가 느릴 때 매 상자마다 2초 지연 발생

**현재 코드**:
```python
rclpy.spin_until_future_complete(self.db, fut, timeout_sec=2.0)
```
**튜닝**: 타임아웃을 1.0s로 줄이거나, 비동기 처리로 전환

---

### 🟢 LOW - 슬롯 4 완충 후 리셋 타이밍

**증상**: 4칸 완충 직후 바로 다음 상자를 슬롯1에 넣으려 함

**현재 동작**: `self.filled = 0` 즉시 리셋
**실제로는**: DB가 AMR에게 회전 명령 → 완료 신호 오기 전에 리셋될 수 있음

**해결**: pause_status 토픽 구독 (이미 sh5_final.py에 있음, sh5_final_2.py에 추가 필요)

---

## 5. 디버깅 순서 (현장 D-Day 아침)

```
STEP 1: 좌표 확인
  → Script Editor에서 SH5 로봇 Prim 경로 확인
  → ROBOT_POS 딕셔너리 실측값 업데이트

STEP 2: 단독 테스트 (Isaac Sim 없이)
  → python3 sh5_final_2.py
  → Mock 투입으로 CSV 조회, HDF5 로드, offset 계산 확인

STEP 3: Dummy Teleport 모드로 E2E 테스트
  → PICK_AND_PLACE_MODE = "DUMMY_TELEPORT"
  → 상자 스폰 → 슬롯 좌표로 이동 → DB 보고 확인
  → FPS 체크

STEP 4: robot_art 연결
  → Robot() 객체 생성 후 controller.lines[i].robot_art 연결
  → HDF5_REPLAY 모드로 전환

STEP 5: 실 데이터 테스트
  → bg2_mock_publisher.py --mode manual 로 1개씩 수동 투입
  → 각 슬롯 1,2,3,4 순서로 확인
```

---

## 6. 최종 판단

```
시나리오 전체 구현 가능성: 🟡 조건부 가능 (80%)

확실히 동작:  통신 / 스폰 / 보고 / 순차 실행
조건부 동작:  HDF5 Replay (robot_art 연결 필요)
데모 폴백:    Dummy Teleport (항상 성공, 로봇 팔 미동작)

현실적 권장:
  D-Day 1순위 → DUMMY_TELEPORT (통신 시연 목적)
  D-Day 2순위 → HDF5_REPLAY (robot_art 연결 성공 시)
  학습 완료 후 → ACT_MODEL (선택)
```

---

*저장: `/home/rokey/dev_ws/md_file/SH5_SCENARIO_ANALYSIS.md`*
