# 📜 SH5 exec() 스크립트 총정리

> **작성일:** 2026-06-10  
> **경로 기준:** `/home/rokey/dev_ws/coupang_ws/scripts/`

---

## 🗺️ 전체 스크립트 지도

```
시연 상황
  ├─ [A] 완전 통합 시연 (DB+AMR+BG2 전부 연결)
  │     └─ sh5_spawn_controller.py   ← 메인 권장
  │
  ├─ [B] DB+AMR 있지만 BG2 PC 분리 (소켓 통신)
  │     ├─ sh5_socket_server.py       ← SH5 PC에서 실행
  │     └─ bg2_socket_sender.py       ← BG2 PC에 전달
  │
  ├─ [C] 혼자 단독 테스트 (아무것도 없음)
  │     └─ sh5_solo_demo.py
  │
  ├─ [D] 기존 통합 컨트롤러 (레거시/범용)
  │     └─ sh5_integrated.py
  │
  └─ [E] QR 카메라 인식 추가 (Plan B 옵션)
        └─ sh5_qr_scanner.py          ← 위 스크립트에 패치
```

---

## 🟢 [A] `sh5_spawn_controller.py` — 완전 통합 시연용 메인

### exec() 명령
```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_spawn_controller.py', encoding='utf-8').read())
```

### 기능
| 기능 | 설명 |
|:---|:---|
| BG2 리스폰 | `/sim/sg2_spawn_trigger` 수신 시 컨베이어 끝에 상자 자동 리스폰 |
| HDF5 재생 | 수신 즉시 슬롯별 VR 궤적으로 픽앤플레이스 실행 |
| DB 통신 | CheckWarehouseStatus, ReportInboundProgress 서비스 호출 |
| Action | MovePackage(중복 감지), ManageWorkstation(만석) Action 발행 |
| Pause 인터락 | `/sg2_in_0X/pause_status` 구독하여 AMR과 동기화 |
| Isaac Sim 콜백 | Update 이벤트 기반 프레임 동기 실행 |
| Mock 자동 전환 | ROS2 없으면 5초 간격 자동 가상 투입으로 전환 |

### 필요한 환경
```
✅ Isaac Sim (final_coupan.usd 열린 상태)
✅ AMR 팀 스크립트 먼저 실행
✅ DB팀 ROS2 노드 (없으면 Mock 자동 전환)
✅ BG2 PC SimSyncNode (/sim/sg2_spawn_trigger 발행)
```

### 언제 사용?
> **D-Day 실제 통합 시연** — DB+AMR+BG2 모두 연결된 최종 환경

---

## 🔵 [B-1] `sh5_socket_server.py` — C-to-C 소켓 수신 서버

### exec() 명령
```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_socket_server.py', encoding='utf-8').read())
```

### 기능
| 기능 | 설명 |
|:---|:---|
| TCP 서버 | 포트 9000에서 JSON 패킷 수신 대기 |
| 상자 리스폰 | BG2 PC로부터 수신 즉시 컨베이어 끝에 상자 생성 |
| HDF5 재생 | 수신 후 픽앤플레이스 실행 |
| Worker 분리 | 소켓 스레드 ↔ Isaac Sim 조작 스레드 완전 분리 |
| 수동 테스트 | `server._task_queue.put({...})` 으로 직접 주입 가능 |

### 필요한 환경
```
✅ Isaac Sim (SH5 PC)
✅ BG2 PC가 bg2_socket_sender.py 실행 중
✅ 두 PC가 같은 네트워크 (IP 통신 가능)
❌ ROS2 불필요
❌ DB 불필요
```

### 언제 사용?
> **ROS2 없이 BG2 PC와 직접 소켓 통신** — DB팀 환경 없이 시연할 때

---

## 🔵 [B-2] `bg2_socket_sender.py` — C-to-C 소켓 송신 클라이언트

### exec() 명령 (BG2 PC에서)
```python
exec(open('/path/to/bg2_socket_sender.py', encoding='utf-8').read())

# 이후 상자 디스폰 시마다:
sender.send("PKG_20260612_001", "sg2_in_01")

# 또는 단발성:
send_box_event("PKG_001", "sg2_in_02")
```

### 기능
| 기능 | 설명 |
|:---|:---|
| TCP 클라이언트 | SH5 PC(9000포트)로 JSON 전송 |
| 단발성 전송 | `send_box_event()` — 매번 연결/해제 |
| 연결 유지 | `BoxEventSender` — 연결 유지하며 반복 전송 (고빈도용) |
| 자동 재연결 | 연결 끊김 시 1회 자동 재시도 |
| 응답 확인 | SH5 PC의 {"status":"ok"} 수신 확인 |

### 설정 (파일 상단에서 수정)
```python
SH5_PC_HOST = "192.168.10.XX"  # ← SH5/AMR PC의 실제 IP
SH5_PC_PORT = 9000
```

### 필요한 환경
```
✅ BG2 PC (다른 Isaac Sim)
✅ sh5_socket_server.py가 SH5 PC에서 실행 중
✅ 네트워크 연결 (ping 확인)
```

### 언제 사용?
> **BG2 PC에서 상자 디스폰할 때마다 SH5 PC로 신호 전송**

---

## 🟡 [C] `sh5_solo_demo.py` — 혼자 단독 테스트

### exec() 명령
```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_solo_demo.py', encoding='utf-8').read())
```

### 기능
| 기능 | 설명 |
|:---|:---|
| 수동 투입 | 콘솔에서 `demo.trigger_line('sg2_in_01')` 호출 |
| 자동 순환 | `demo.auto_start()` → 3개 라인 5초 간격 자동 투입 |
| 슬롯 리셋 | `demo.reset_all()` |
| 상태 확인 | `demo.print_status()` → [2/4] ■■□□ 형식 |
| HDF5 재생 | 상자 투입 시 궤적 자동 재생 후 홈 복귀 |
| 4슬롯 만석 | 5초 후 로컬 자동 리셋 (AMR 없이) |

### API (Script Editor 콘솔에서 직접 입력)
```python
demo.trigger_line('sg2_in_01')   # 1번 라인 상자 1개 투입
demo.trigger_line('sg2_in_02')   # 2번 라인
demo.trigger_line('sg2_in_03')   # 3번 라인
demo.auto_start()                 # 자동 순환 시작 (5초 간격)
demo.auto_stop()                  # 자동 중지
demo.reset_all()                  # 전체 슬롯 초기화
demo.print_status()               # 현재 슬롯 현황 출력
```

### 필요한 환경
```
✅ Isaac Sim만 있으면 됨
❌ DB 불필요
❌ ROS2 불필요
❌ BG2 PC 불필요
❌ AMR 불필요
```

### 언제 사용?
> **SH5 로봇 모션만 단독 검증** — 발표 전날 밤 혼자 테스트

---

## 🟠 [D] `sh5_integrated.py` — 범용 통합 컨트롤러 (레거시)

### exec() 명령
```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_integrated.py', encoding='utf-8').read())
```

### 수정 포인트 (71번 줄)
```python
PICK_AND_PLACE_MODE = "HDF5_REPLAY"    # VR 궤적 재생 (기본/보험)
# PICK_AND_PLACE_MODE = "DUMMY_TELEPORT" # 상자 순간이동 (통신 테스트)
# PICK_AND_PLACE_MODE = "ACT_MODEL"      # AI 비전 추론
```

### 기능
| 기능 | 설명 |
|:---|:---|
| 3모드 전환 | ACT_MODEL / HDF5_REPLAY / DUMMY_TELEPORT |
| FSM 상태머신 | SCANNING → ALLOCATE → PICK_PLACE → REPORT → 반복 |
| DB 통신 | CheckWarehouseStatus, ReportInboundProgress, MovePackage, ManageWorkstation |
| Mock QR | DB 없을 때 랜덤 QR ID 자동 생성 |
| ROS2 Mock | ROS2 없을 때 전체 Mock 모드 자동 전환 |
| HDF5 재생 | hdf5_replay_player.py 연동 |

### 필요한 환경
```
✅ Isaac Sim
△  ROS2 (없으면 Mock 자동 전환)
△  DB팀 노드 (없으면 Mock 자동 전환)
```

### 언제 사용?
> **기존 상태머신 구조를 유지하면서 통합 테스트** — [A]의 전신  
> 모드 전환(71번 줄)이 필요한 상황에서 유연하게 사용

---

## 🟣 [E] `sh5_qr_scanner.py` — Plan B QR 카메라 인식 (옵션 패치)

### exec() 명령 (단독 또는 패치로 사용)
```python
# 단독 로드 후 테스트
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_qr_scanner.py', encoding='utf-8').read())

# 카메라 경로 탐색
find_camera_prims()

# 단일 스캔 테스트
scanner = SH5QRScanner("sg2_in_01")
result = scanner.scan()

# 디버그 이미지 저장
scanner.debug_capture("/tmp/cam_check.png")

# 기존 컨트롤러에 패치
patch_sh5_integrated_qr(controller)
```

### 기능
| 기능 | 설명 |
|:---|:---|
| 카메라 이미지 | Isaac Sim Top-View 카메라 → numpy 이미지 |
| QR 디코딩 | OpenCV QRCodeDetector (다중/단일, 전처리 포함) |
| 좌표 변환 | 픽셀 XY → 월드 XY (카메라 FOV + 높이 기반 역투영) |
| Mock 폴백 | 스캔 실패 시 자동으로 Mock 데이터 반환 |
| 경로 탐색 | `find_camera_prims()` 로 Stage 내 카메라 자동 탐색 |
| 드롭인 패치 | `patch_sh5_integrated_qr(controller)` 한 줄로 교체 |

### 필요한 환경
```
✅ Isaac Sim (Top-View 카메라 Prim 존재)
✅ OpenCV (pip install opencv-python)
△  QR 코드가 상자에 부착되어 있어야 함
```

### 언제 사용?
> **상자의 실제 위치를 카메라로 정확히 측정해야 할 때**  
> Plan A(소켓/ROS2 신호)가 충분하면 불필요

---

## ⚡ 상황별 빠른 선택 가이드

| 상황 | 실행할 파일 | 비고 |
|:---|:---|:---|
| D-Day 실제 통합 시연 | `sh5_spawn_controller.py` | DB+AMR+BG2 전부 연결 |
| ROS2 없이 BG2 PC 직접 통신 | `sh5_socket_server.py` + `bg2_socket_sender.py` | IP:Port만 맞추면 됨 |
| 혼자 모션 검증 | `sh5_solo_demo.py` | 아무것도 없어도 됨 |
| 모드 전환 필요 (AI↔HDF5) | `sh5_integrated.py` | 71번 줄 수정 |
| QR 카메라 추가 | `sh5_qr_scanner.py` | 위 스크립트 실행 후 패치 |

---

## 🔁 실행 순서 (D-Day 기준)

```
Terminal:
  1. source ~/dev_ws/cobot3_ws_ref/install/setup.bash
  2. ros2 topic list | grep sg2_spawn   ← DB팀 확인

Isaac Sim Script Editor:
  3. exec(amr_live_...gpu.py)           ← AMR팀 먼저
  4. exec(sh5_spawn_controller.py)      ← 우리 메인

(옵션) Plan B QR 추가:
  5. exec(sh5_qr_scanner.py)
  6. patch_sh5_integrated_qr(controller)

(비상) 혼자 테스트:
  3. exec(sh5_solo_demo.py)
  4. demo.auto_start()
```

---

*저장 경로: `/home/rokey/dev_ws/md_file/SH5_SCRIPT_GUIDE.md`*
