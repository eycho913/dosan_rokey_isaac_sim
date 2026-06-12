# Developer Log (DevLog) - FFW SG2 물류 분류 로봇 RL 개발 일지

---

## 📅 [2026-06-02 AM] - 환경 기반 구성 및 하드웨어 호환성 해결

### 🚀 IsaacLab RL 환경 기본 구성 완료
- FFW SG2 로봇(URDF) → USD 에셋 변환 및 IsaacLab `ArticulationCfg` 등록 완료.
- `rsl_rl >= 5.0.0` 버전 호환성 이슈(`MLPModel __init__() stochastic` 오류)를 `train.py`에서 파라미터 하드 팝(pop) 방식으로 해결.
- 태스크 등록 (`Isaac-Sort-FFW-SG2-v0`), PPO Runner 연결, 학습 로그 저장 구조(`/rl_ws/logs/`) 구축.

### 🐛 Blackwell 아키텍처(RTX 5080) PhysX GPU 비호환 해결
- **문제:** RTX 5080 Blackwell GPU에서 PhysX GPU Pipeline 활성화 시 `Segmentation fault` 발생.
- **원인:** Isaac Sim 4.x의 PhysX 드라이버가 아직 Blackwell 아키텍처를 완전 지원하지 않음.
- **조치:** 물리 연산을 CPU (Intel Core Ultra 9 275HX)로 강제 우회, PPO 역전파(학습)만 GPU 활용.
- **상태:** NVIDIA 공식 패치 전까지 CPU 물리 파이프라인 유지.

---

## 📅 [2026-06-02 PM] - 1차 학습 실패 분석 및 보상 체계 전면 개편

### ❌ 1차 학습 결과 (Iteration 1999, 300만 스텝)
- `reaching_package = 0.0000` → 로봇이 2000번 이터레이션 내내 **완전히 가만히 서 있음**.
- **원인 분석:** 보상 함수 `std=0.1`로 인한 보상 기울기(Gradient) 소실(Vanishing Gradient). 로봇이 상자에서 30cm 이상 떨어진 위치에서는 보상이 0에 수렴하여 탐험 동기 제로.
- 또한 감점(`action_rate`, `joint_vel`)이 작은 보상보다 커서 "아무것도 안 하는 것이 최선"이라는 Local Minima 발생.

### ✅ 보상 체계 전면 개편
| 항목 | 변경 전 | 변경 후 | 효과 |
|---|---|---|---|
| `reaching_package` std | 0.1 | **0.5** | Dense Reward: 먼 거리에서도 기울기 전달 |
| `reaching_package` weight | 1.0 | **3.0** | 접근 동기 강화 |
| `action_rate` penalty | -1e-4 | **-1e-5** | 탐험(Exploration) 장려 |
| `joint_vel` penalty | -1e-4 | **-1e-5** | 탐험(Exploration) 장려 |

---

## 📅 [2026-06-02 PM] - 물리 환경 버그 대거 수정

### 🐛 Bug #1: 상자 복제(Cloner) 위치 오류
- **문제:** 64개 환경 모두의 상자(Package)가 전역 좌표 (0, 0, 0) 한 지점에 겹쳐 스폰됨.
- **원인:** Isaac Sim Cloner가 복제된 환경의 `default_root_state`에 각 환경 기준점(env_origins)을 정상 반영하지 못하는 엔진 버그.
- **해결:** `events.py`의 `reset_package_position`에서 `env_origins + local_pos`를 수동 명시적 계산하여 전역 좌표 강제 주입.

### 🐛 Bug #2: 테이블-작업대(Racker) 물리 충돌 폭발
- **문제:** 에피소드 시작 시 작업대가 넘어짐.
- **원인:** 테이블(X=0.6m)과 작업대(X=1.2m, 넓은 프레임)의 물리 경계가 겹쳐 초기화 순간 충돌 폭발.
- **해결:** 테이블과 상자를 로봇 쪽으로 당겨 X=0.45m로 안전거리 확보.

### 🐛 Bug #3: 로봇 바퀴 미끄러짐 (다른 환경 침범)
- **문제:** 로봇이 팔을 휘두르는 반작용으로 바퀴가 굴러서 이웃 환경으로 이동.
- **해결:** `ffw_sg2_cfg.py`에 `fix_root_link=True` 적용. 현재 학습 단계(1~4단계)에서는 몸통을 바닥에 고정.

### ✅ 상체 승강(Lift Joint) 학습 추가
- SG2 로봇의 `lift_joint` (Prismatic 상하 이동 관절)를 행동 공간에 추가.
- 기존 8차원(팔 7DOF + 그리퍼) → **9차원(리프트 + 팔 7DOF + 그리퍼)**으로 확장.

---

## 📅 [2026-06-02 PM] - 2차 학습 성공 (Local Minima 완전 탈출)

### ✅ 2차 학습 결과 (Iteration 1999, 300만 스텝)
| 지표 | 이전 | 이후 | 평가 |
|---|---|---|---|
| `reaching_package` | 0.0000 | **0.2571** | 상자 인식 및 접근 성공 |
| `Mean reward` | ~0 | **20.88** | 학습 방향성 확보 |
| 감점 | -0.0008 | **-0.0001** | 부드러운 동작 달성 |

---

## 📅 [2026-06-03 AM] - 3차 장기 학습 분석 및 물리적 파지 불가 버그 발견

### ⚠️ 3차 학습 결과 (Iteration 9999, 1500만 스텝)
- `reaching_package = 2.7333` (만점 근접): 상자 접근 **완벽 마스터**.
- `lifting_package = 0.0000`: 들어 올리기 **여전히 0**.

### 🐛 Bug #4: 그리퍼 최대 벌림 < 상자 크기 (물리적 파지 불가)
- **문제:** 어떻게 해도 상자를 쥘 수 없었던 진짜 이유 발견.
- **원인:** SG2 그리퍼 최대 벌림 폭 = **109mm**. 설정된 상자 크기 = **120mm 정육면체**.
  - 그리퍼가 상자보다 작아서 물리적으로 파지 자체가 불가능한 상태였음.
- **해결:** 상자 크기를 **80mm 정육면체**(`size=(0.08, 0.08, 0.08)`)로 축소.

### 🐛 Bug #5: `package_drop` 종료 판정 오류 (실질적 낙하 감지 불가)
- **문제:** 상자를 바닥에 떨어트려도 에피소드가 종료되지 않고 보상도 없음.
- **원인:** `minimum_height=-0.05m` 설정으로 바닥(0m)을 뚫지 않는 한 판정 안 됨.
- **해결:** `minimum_height=0.6m`으로 수정, 낙하 시 **-50점** 패널티 + 즉시 에피소드 종료.

### ✅ 중간 징검다리 보상(grasp_package) 추가
- `reaching → lifting` 사이 빈 구간에 **`grasp_package` (5점)** 추가.
- 조건: EE가 상자 4cm 이내 + 그리퍼 닫힘 + 상자가 초기 Z 위치 이상 유지.

---

## 📅 [2026-06-03 PM] - 보상 해킹 발견 및 EE 프레임 치명적 버그 수정

### 🐛 Bug #6: 보상 해킹 (Reward Hacking) 발견
- **문제:** 로봇이 상자를 집지 않고 상자 옆에서 그리퍼만 허공에 꽉 닫고 있음. `grasp_package = 4.09`이지만 실제 파지 없음.
- **원인:** `grasp_package` 보상이 "EE 근처 + 그리퍼 닫힘" 만 확인, 실제 물리 접촉 여부 미확인.
- **해결:** `not_dropped` 조건 추가 (상자 Z 위치가 초기 기준 이상 유지 시에만 보상).

### 🐛 Bug #7: `init_z` Cloner 버그 재발 (Lifting 판정 오류)
- **문제:** `lifting_package` 보상이 로봇이 실제로 들어 올려도 항상 0 반환.
- **원인:** `package_is_lifted`의 `init_z = obj.data.default_root_state[:, 2]` 가 Cloner 버그로 잘못된 값 반환.
- **해결:** 3개 함수 모두 `init_z = env.scene.env_origins[:, 2] + 0.86`으로 수동 계산으로 교체.

### 🐛 Bug #8 (치명적): EE 프레임이 그리퍼 팁이 아닌 팔꿈치를 추적
- **문제:** 로봇이 팔꿈치로 상자를 치고, 그 위에서 그리퍼를 열고 닫는 행동 반복.
- **원인:** `ee_frame` 설정이 `arm_r_link7`(손목 링크) + Z+15cm 오프셋으로 되어 있어, 실제로는 팔꿈치 근처 위치를 추적하고 있었음.
  ```python
  # ❌ 잘못된 설정 (팔꿈치 추적)
  prim_path="arm_r_link7", offset=pos=(0.0, 0.0, 0.15)
  
  # ✅ 수정된 설정 (실제 그리퍼 팁 추적)
  prim_path="gripper_r_rh_p12_rn_base", offset=pos=(0.0, 0.0, 0.10)
  ```
- **영향:** 지금까지의 모든 `reaching_package` 보상과 `grasp_package` 보상이 잘못된 위치를 기준으로 계산되었음. **전체 재학습 필요.**

### ✅ 초기 관절 자세 개선 (수평 파지 자세)
- URDF 분석으로 `lift_joint` 범위: `-0.5 ~ 0.0` (위가 음수) 확인.
- 팔이 위에서 내려꽂는 자세 → **수평 측면 파지 자세**로 개선.
  - `lift_joint: 0.0 → -0.3` (팔 높이를 상자 높이에 맞게)
  - `arm_r_joint2: -0.5 → -1.2` (어깨를 더 아래로)
  - `arm_r_joint4: -1.5 → -0.8` (팔꿈치를 덜 굽혀 앞으로 뻗은 자세)

---

## 🗺️ Curriculum Learning Roadmap (학습 단계별 마스터플랜)

- **[1단계] Reaching & Grasping (현재 진행 중):**
  수직 고정 자세에서 상자에 손을 뻗어(`reaching_package`) 실제로 그리퍼로 잡아 올리기(`lifting_package`).
- **[2단계] Fixed Placing (단일 바구니 넣기):**
  들어 올린 상자를 작업대의 정해진 바구니에 골인시키는 조작 능력(`package_to_bin`) 학습.
- **[3단계] Sorting & Classification (조건부 분류):**
  ArUco ID에 따라 4개의 바구니 중 올바른 곳에 분류(`sorting_success`) 학습.
- **[4단계] End-to-End Master (고정 상태 통합):**
  무작위 상자를 인식·파지·분류하는 전 과정 통합. (로봇 하체 고정 유지)
- **[5단계] Mobile Manipulation (AGV 주행 통합):**
  `fix_root_link=True` 해제, 바퀴 관절을 행동 공간에 추가. 컨베이어 벨트 ↔ 작업대 왕복 자율 이동 학습.
- **[6단계] Sim2Real & Vision AI 통합 (최종):**
  카메라 + YOLO/ArUco 비전 모듈을 RL Policy에 연결하여 실제 로봇에 Sim2Real 적용.

---

## 📅 [2026-06-04] - 로봇 기종 변경(SH5) 및 VR 기반 원격 조작(Teleoperation) 기획

### 🔄 로봇 플랫폼 변경: SG2 → SH5
- **변경 사항:** 기존 병렬 그리퍼 기반의 SG2 로봇에서 **다관절 로봇 핸드(Dexterous Hand)를 장착한 SH5**로 플랫폼 전면 교체.
- **기대 효과:** 단순 파지를 넘어, 다관절 손가락을 활용하여 물체의 형태에 맞춘 정교한 조작 및 다품종 분류 작업 가능.

### 🥽 VR 원격 조작(Teleoperation) 연동
- **시스템 구성:** Docker 기반 Vuer VR 원격 조작 노드(`vr_publisher_sh5`) ↔ Isaac Sim 시뮬레이션 브릿지(`sh5_dds_bringup`).
- **통신 환경:** ROS 2 (Domain ID: 119) 기반 DDS 토픽 통신.
- **주요 기능:**
  - VR 헤드셋 및 컨트롤러를 이용한 작업자의 **머리(HEAD) 및 손목(WRIST) 트래킹 데이터** 수집 및 전송.
  - 작업자의 손동작(Gesture) 기반 활성화 트리거를 통한 시뮬레이션 내 로봇 제어 동기화.
  - 궤적 명령(Trajectory Commands)을 시뮬레이션 상의 SH5 관절 및 다관절 핸드 모션으로 실시간 매핑.

### 📦 물류 분류 및 작업대 배치 기능 기획안 (SH5 버전)
- **작업 환경:** 다양한 물체가 놓인 작업대와 목적별 분류 구역(바구니 등).
- **작업 시나리오:**
  1. **인식 및 접근:** VR 트래킹을 통한 직접 조작(또는 향후 비전 인식)으로 목표 물체를 향해 다관절 핸드 접근.
  2. **정교한 파지 (Dexterous Grasping):** 물체의 크기와 모양에 맞추어 다관절 손가락을 독립 제어, 안정적인 그립 형태 생성.
  3. **분류 및 이동 (Sorting):** 대상을 파지 후 들어올려 지정된 분류 기준(모양, 크기 등)에 맞는 작업대 위 목적 영역으로 이동.
  4. **배치 (Placing):** 목표 위치에서 부드럽게 손을 펴서(Release) 안전하게 대상 내려놓기.
- **향후 방향성:** VR 원격 조작을 통해 양질의 시연(Demonstration) 데이터를 수집하고, 이를 모방 학습(IL) 및 RL 보상 체계에 접목하여 정밀 자율 조작 성능 향상.

---

## 📅 [2026-06-05] - VR 텔레오퍼레이션 영상 스트리밍 파이프라인 완성과 의사결정

### 🐛 Bug #9: VR(Vuer) 화면 송출 실패 및 데이터 타입 불일치 해결
- **문제:** Isaac Sim의 화면이 VR 헤드셋(Vuer)으로 송출되지 않고 검은 화면만 표시됨.
- **원인 1 (Isaac Sim 렌더링):** IsaacLab 4.5+ 환경에서 `sim.step()`이 물리 엔진만 계산하고 렌더링(ActionGraph의 ROS2 Camera Helper)을 트리거하지 않음.
- **원인 2 (데이터 타입 불일치):** Vuer 노드(`vr_publisher_sh5.py`)는 네트워크 대역폭을 위해 `CompressedImage`를 기대하지만, Isaac Sim은 `Image`(Raw)를 발행.
- **해결 1:** `sh5_dds_bringup.py` 내의 메인 루프에서 `sim.step(render=True)`로 강제 렌더링 활성화 및 `ROS_DOMAIN_ID=119` 환경 변수 동기화 확인.
- **해결 2:** 외부 압축 노드(`image_transport`)를 띄우는 대신, Docker 컨테이너 내부의 `vr_publisher_sh5.py`를 직접 패치하여 `cv_bridge`와 `cv2.imencode`를 통해 Raw 이미지를 내부적으로 자동 압축하도록 개선. (Docker 내 `ros-jazzy-cv-bridge` 패키지 설치 완료)

### ⚖️ 기술적 의사결정 (ADR): Vuer vs Native SteamVR (OpenXR)
- **제안:** ALVR + SteamVR을 활용한 Isaac Sim 네이티브 OpenXR 사용 안.
- **분석:** 네이티브 방식은 시각적 품질과 지연 시간 면에서 압도적이나, SH5의 26-DOF 다관절 핸드 IK 솔버가 Vuer의 WebXR 구조(JSON)에 맞춰 개발되어 있음. 이를 OpenXR 스켈레탈 데이터로 다시 맵핑하려면 대규모 수학적 변환과 코드 재작성이 필요함.
- **결정:** "당일 내 환경 구축 및 모방 학습 데이터 취득"이라는 시급성을 고려하여, 이미 검증된 **Vuer(WebXR) 기반 핸드 트래킹 파이프라인을 유지**하기로 확정.

---

## 📋 현재 상태 및 앞으로 할 일

- [x] (기존 SG2) 물리 환경 기반 구성 및 Cloner/충돌 버그 해결
- [x] (기존 SG2) RL 보상 튜닝 및 초기 학습 궤도 안착
- [x] **[완료]** SH5 로봇 및 다관절 핸드 시뮬레이션 모델 구성
- [x] **[완료]** VR 원격 조작 통신 연동 (ROS 2 Domain 119, `vr_publisher_sh5` ↔ `sh5_dds_bringup`)
- [x] **[완료]** VR 시각 피드백 파이프라인 완비 (Isaac Sim 카메라 → Vuer 렌더링) 및 버그 수정
- [ ] **[현재]** VR 환경에서 시각 피드백을 보며 SH5 로봇 조작 및 모방 학습(IL)용 데모 데이터 수집
- [ ] 수집된 시연 데이터를 활용한 모방 학습 연동 및 궤적 검증
- [ ] SH5 환경에 맞춘 강화 학습(RL) 파이프라인 재설계 및 고도화
- [ ] 비전 모듈 연동 및 자율 분류/배치 프로세스 통합


## 📅 [2026-06-06] - SH5 데이터 수집 환경 최적화 및 텔레오퍼레이션 고도화

### 🏗️ 물리 환경 및 리스폰 로직 고도화
- **작업대 및 상자 배치:** 피킹 작업을 위해 상자 크기와 초기 위치 조정, 작업대를 전방(X=1.2m)에 배치.
- **자동 리스폰(Respawn) 기능:** 
  - 상자가 바닥(Z < 0.5)으로 떨어졌을 때뿐만 아니라, **작업대 위(X > 0.85)에 무사히 안착하여 정지(Velocity < 0.02)했을 때**도 자동으로 다음 에피소드를 위해 리스폰되도록 로직 추가.

### 🎮 SH5 제어 노드(`mobile_teleop.py`) 확장
- 기존의 스워브 베이스 전후진 제어에 추가로, **좌우 병진 이동(Strafing, Q/E키)** 구현.
- 로봇의 시선을 상하좌우로 조절할 수 있도록 **Head Pan/Tilt (I/J/K/L키)** 제어 기능 추가 및 안전한 JointTrajectory 퍼블리시 연동.

### 🧲 물체 파지(Grasping) 물리 파라미터 최적화
- **문제:** 로봇이 상자를 들어올릴 때 마찰 부족과 토크 부족으로 뚫고 나가거나 놓치는 현상 (Tunneling & Slipping).
- **해결 (극한의 황금 밸런스 튜닝):**
  - **상자 무게:** `mass`를 0.2kg에서 **0.05kg**으로 경량화.
  - **마찰력:** 상자 표면 및 손가락 끝(`_SH5_FINGER_TIP_MATERIAL`)의 `static/dynamic friction`을 **10.0**으로 상향.
  - **악력 및 유지력:** 손가락 관절(`XM_335`)의 최대 토크(`effort_limit_sim`)를 **50.0**으로 상향하고, `stiffness`를 **1500.0**, `damping`을 **80.0**으로 설정하여 강한 파지 유지와 진동 억제 성공.

### 🛡️ VR 조작 안전장치 우회 및 로거(Logger) 개선
- **안전장치 완화:** VR 원격 조작 시 컨트롤러가 튈 때 암(Arm) 제어가 끊기는 현상(`Startup mismatch`, `Reference jump`)을 방지하기 위해 Docker 컨테이너 내의 `ai_worker_config.yaml` 환경 설정에서 거리 및 각도 점프 제한(Threshold)을 사실상 무제한으로 해제.
- **모방 학습 데이터 로깅 효율화:** 
  - 여러 에피소드를 하나의 HDF5 파일에 저장하는 Robomimic 표준 구조 유지.
  - 실패한 에피소드는 저장 없이 버릴 수 있도록 **녹화 취소 기능(C 키)** 추가 도입.
  - 우측 팔의 조작 데이터를 이용해 좌측 팔 데이터로 뻥튀기하는 **Mirroring Data Augmentation** 기법 도입 검토 완료.

## 📅 [2026-06-07] - 모방 학습(IL) 데이터 파이프라인 완성 및 훈련 스크립트 구축

### 🧲 'Magic Snapping' (자석 부착) 물리 안정화 로직 도입
- **문제:** 로봇이 물리적으로 상자를 파지할 때, 접촉점에서의 마찰력 한계로 인해 상자가 미끄러지거나 회전(팽이 현상)하는 등 IL 데이터의 질을 훼손하는 심각한 Sim2Real 갭 발생.
- **해결:** `coupang_sh5_bringup.py` 내에 Kinematic Holding(강제 고정) 로직을 구현하여, 손가락이 굽혀지고 상자와의 거리가 일정 이내일 때 상자의 3D Root State를 로봇 손바닥에 강제로 고정. 이로써 완벽하고 깔끔한 Expert Trajectory(전문가 시연 궤적) 수집 가능.

### 🧠 모방 학습(IL) & 강화 학습(RL)용 데이터 로거(Logger) 전면 개편
- **기존:** `joint_positions`, `joint_velocities`, `actions(joint_targets)` 등 팔 위주의 데이터만 로깅.
- **개편:** 이동 대차(모바일 베이스)를 포함한 상태 기반(State-Based) 학습이 가능하도록 HDF5 구조를 대폭 확장 (147차원 입력 -> 66차원 출력).
  - `obs/robot_pose` (7D): 모바일 베이스의 현재 위치/회전 상태 추가.
  - `obs/box_pose` (7D): 대상 물체(상자)의 목표 좌표 추가 (YOLO 비전 정보의 Ground Truth 역할).
  - `obs/rack_pose` (7D): 대상 도착지(작업대)의 목표 좌표 추가.
  - `actions/cmd_vel` (3D): 모바일 베이스 주행 명령(조이스틱) 추가.

### ⌨️ 비동기 터미널 키보드 폴링(Terminal Poller) 구현 및 버그 수정
- **문제:** 원격 환경(SSH, VNC)에서 Isaac Sim GUI에 포커스가 없을 경우 `carb.input` 이 키보드를 감지하지 못해 데이터 녹화(R, T) 및 리스폰(B, V) 불가.
- **해결:** `tty` 및 `termios`를 활용한 백그라운드 스레드 키보드 폴러(`TerminalKeyboard`)를 직접 구현하여 원격 SSH 터미널에서도 딜레이나 블로킹 없이 키 입력을 캡처하도록 개선.
- **버그 수정:** `V` 키로 로봇 리스폰 시 이전 명령된 높이로 되돌아가는 문제 발견. `SH5DdsBridge`에 `clear_pending_targets()` 메서드를 추가하고 PD Controller Target을 명시적으로 리셋하여 완벽한 상태 초기화 달성.

### 🚀 인공지능 훈련 스크립트 (`train_bc.py`) 신규 개발
- HDF5 파일의 데모 궤적들을 불러와 파이토치(PyTorch) 기반 다층 퍼셉트론(MLP)에 훈련시키는 Behavior Cloning 스크립트 작성 완료.
- **전략 의사결정 (4분할 모델):** 실험 데드라인(3일)의 제약을 극복하기 위해, 하나의 Goal-Conditioned 만능 AI를 디버깅하는 대신 **작업대의 4개 슬롯별로 데이터를 분할하여 4개의 전용 모델(Expert Policies)을 각각 학습**시키는 실용적이고 확실한 방법을 채택.
- **사용성:** CLI 인자(`--data_dir`, `--output`)를 추가하여 각 공간별 HDF5 데이터를 지정된 이름의 `.pth` 로 빠르고 독립적으로 추출할 수 있도록 최적화.

---

## 📅 [2026-06-08] - ACT 정책 구축, 정밀 데이터 증강 및 리플레이 검증 도구 개발

### 🧠 ACT (Action Chunking with Transformers) 정책 스크립트 (`train_act.py`) 신규 개발
- **배경:** 단일 프레임 MLP 기반 BC의 동작 불안정성(급작스러운 회전, 허우적댐 등)과 멀티모달성(동일 상태에서 다른 행동 매핑) 부족을 해결하기 위해 ACT 모델을 개발함.
- **아키텍처:**
  - **CVAE Encoder:** 학습 시 전문가의 액션 시퀀스를 입력받아 latent vector $z$를 추출 (KL Divergence loss 및 Annealing 적용).
  - **State Encoder:** 과거 N 프레임(기본 `context_len=10`)의 153차원 상태 히스토리를 Transformer Encoder로 통합 인코딩.
  - **Action Decoder:** 인코딩된 상태와 $z$를 cross-attention으로 입력받아 미래 K 프레임(기본 `chunk_size=20`)의 액션 시퀀스를 일괄 예측(Action Chunking).
- **검증:** 100에피소드 데이터셋(81,909프레임)으로 10에포크 단기 학습을 실행하여 loss 수렴과 체크포인트(`test_act_verify.pth`) 정상 생성을 검증 완료.

### 🧲 Magic Snapping 로컬 좌표계 개선 및 물리 튜닝
- **로컬 좌표계 오프셋 전환:** 기존 월드 좌표계 기반 고정에서 로봇 손바닥(Body-Local) 좌표계 오프셋 연산(`q_inv * world_offset`)으로 개선하여, 로봇 주행 및 360도 회전 시에도 상자가 탈조(이탈)하지 않고 자연스럽게 고정되도록 수정.
- **상자 터널링 방지:** 상자 최대 속도를 `1.5 m/s`로 클램핑하여 고속 이동 시 충돌 레이어가 뚫리는 문제를 물리적으로 방지함.

### 📈 슬롯별 좌표 기반 정밀 데이터 증강 (`augment_data.py` v2)
- **목표:** 슬롯 1(위층 좌측)에 상자를 넣는 100개 에피소드 시연 데이터를 가공하여, 슬롯 3(아래층 좌측) 데이터로 자동 증강.
- **정밀 변환 로직:**
  - 단순히 전체 궤적에 오프셋을 더할 경우 벨트 위의 상자 위치(접근 단계)까지 찌그러지는 현상 발생.
  - 이를 막기 위해 **단계 감지(Phase-Aware) 기법**을 적용하여, 손가락이 닫혀 상자를 완전히 쥐는 시점(`grasp_frame`)부터 코사인 보간 블렌딩으로 $Z$ 오프셋($-0.738\text{m}$)을 점진적으로 적용하도록 구현.
  - 이를 통해 접근 단계는 완벽히 호환되고 배치(Placing) 높이만 슬롯 3에 맞게 정밀하게 꺾이는 100개 증강 에피소드 자동 생성 성공.

### ⚠️ 물리적 한계점 규명 (Slot 1 → 3 증강의 한계)
- **상황:** 슬롯 3 증강 데이터 적용 시 로봇이 정상적으로 아래로 뻗지 않고 슬롯 1 높이에서 헛손질하는 현상 분석.
- **원인:** SH5 로봇의 URDF 상 `lift_joint` 가 prismatic joint이며 하향 가동 리밋이 **`-0.5m`**로 고정되어 있음. 슬롯 1에서 슬롯 3으로 낮추기 위해 필요한 오프셋은 **`-0.738m`**이므로 리밋을 초과하여 물리 시뮬레이션 내부에서 강제 클램핑됨.
- **결론:** 위층(슬롯 1, 2)과 아래층(슬롯 3, 4)은 높이 차이가 로봇 lift 작동 범위를 초과하므로 단순 궤적 오프셋 증강이 불가능하며, 아래층 슬롯에 대해서는 반드시 물리적 수집(시연)이 병행되어야 함을 증명.

### 🎬 데이터 리플레이 및 시각화 도구 (`replay_data.py`) 개발
- **목표:** HDF5에 저장된 전문가 시연 및 증강 데이터의 모션을 Isaac Sim 상에서 1:1로 재생하며 육안으로 무결성을 검증하는 독립 유틸리티.
- **기능:**
  - 프레임별 `robot_pose`를 루트 링크에 강제 강제 주입하여 모바일 베이스 주행 모션까지 정확하게 재현.
  - 리플레이 중에도 `Magic Snapping` 물리 고정 엔진을 활성화하여 그리퍼 파지 동작과 릴리즈 시점 시각 확인 가능.
  - 키보드 바인딩(N: 다음, P: 이전, R: 다시 재생, Space: 일시정지, 1~4: 각 슬롯 에피소드 점프)으로 높은 사용성 확보.


---

## 📅 [2026-06-09] - 관절 인덱스 오류 수정, 데이터 증강 완성, ACT 학습 실행

### 🐛 Bug Fix: 관절 인덱스 맵 전면 오류 수정 (Critical)

- **문제:** `train_act.py`, `train_bc.py`, `augment_slot3_to_slot4.py` 전체에서 SH5 관절 인덱스를 잘못 가정.
- **기존 (틀린) 가정:** arm_L=[6:13], arm_R=[13:20], finger_L=[20:40], finger_R=[40:60], lift=[62]
- **실제 Isaac Sim `joint_names` 순서 (디버그 출력 확인):**
  - swerve steering: [0, 1, 2] / lift_joint: **[3]** / swerve drive: [4, 5, 6] / head: [7, 10]
  - arm_L: **[8, 11, 13, 15, 17, 19, 21]** (arm_L/R 인터리브 배열!)
  - arm_R: **[9, 12, 14, 16, 18, 20, 22]**
  - finger_L: **[23-27, 33-37, 43-47, 53-57]**
  - finger_R: **[28-32, 38-42, 48-52, 58-62]**
- **해결:** `coupang_sh5_bringup.py`에 디버그 출력 코드 삽입 후 실제 순서 확인. `train_act.py`, `train_bc.py`, `augment_slot3_to_slot4.py` 전면 수정.

---

### 🔄 데이터 증강 (`augment_slot3_to_slot4.py`) 완성

- **목적:** Slot4(왼손, 아래층 좌측) 미취득 → Slot3(오른손, 아래층 우측) 75개로부터 증강.
- **slot1 vs slot2 실측 비교로 올바른 변환 규칙 도출:**

  | 항목 | 기존 가정 | 실측 결과 |
  |---|---|---|
  | robot/box/rack pose X | ❌ 반전 적용 | ✅ 그대로 유지 |
  | box/rack pose qz | 미적용 | ✅ 반전 |
  | cmd_vel wz | ❌ 반전 적용 | ✅ 그대로 유지 |
  | cmd_vel vy | ✅ 반전 | ✅ 반전 |
  | arm 관절 스왑 | 연속 블록 슬라이싱 | ✅ 인터리브 쌍별 스왑 |

- **팔 관절 부호 규칙 (실측 기반, joint1~7 순):** [유지, 반전, 반전, 유지, 반전, 유지, 유지]
- **검증:** box qz 반전 ✅, arm 스왑 ✅, finger 스왑 ✅, num_samples/slot_id attrs ✅

---

### 🧠 ACT 학습 루프 개선: Phase 가중치 도입

- **문제:** 기존 균일 MSE Loss → 파지·삽입 순간이 이동 구간에 묻힘.
- **해결:** `auto_detect_phase()` 단계 정보를 DataLoader 샘플에 포함, Phase별 Loss 가중치 적용:
  - Phase 1 (파지): 3.0배 / Phase 2 (리프트): 1.5배 / Phase 4 (삽입/해제): 3.0배

---

### 🚀 ACT 학습 실행 (Slot1, 2, 3) - RTX 5080

- **환경:** RTX 5080 Laptop GPU (16GB), CUDA 12.8, PyTorch 2.7.0
- **모니터링:** 온도 66°C, 전력 79W/80W, GPU 사용률 96% → 정상 학습 확인

| 슬롯 | 에피소드 | 사용 손 | batch | epoch | lr | 저장 경로 |
|---|---|---|---|---|---|---|
| Slot1 | 100 | 오른손 | 256 | 500 | 3e-4 | models/slot1_act_policy.pth |
| Slot2 | 100 | 왼손 | 256 | 500 | 3e-4 | models/slot2_act_policy.pth |
| Slot3 | 75 | 오른손 | 256 | 500 | 3e-4 | models/slot3_act_policy.pth |

---

### 📋 남은 작업 (D-2, D-1)

1. 학습 완료 확인 (`Epoch [500/500]` + `.pth` 파일 생성)
2. `eval_bc.py`로 시뮬레이션 정책 동작 검증
3. 슬롯 선택 로직 통합 (입력 슬롯 번호 → 모델 로드)
4. 최종 데모 완성



### [2026-06-10] 비전 데이터 파이프라인 버그 픽스 및 모델 제어 로직 고도화

1. **추론(Inference) 로직 고도화 (evaluate_act.py)**
   - **문제:** VR 조종 데이터에 사람의 미세한 오차가 포함되어, 학습된 모델이 홈(대기 상태)으로 완벽하게 복귀하지 못하거나, 상자 파지 후 불필요한 행동을 지속하는 문제 발생.
   - **해결:** 하이브리드 제어(Hybrid Control) 방식 도입. 상자를 놓거나 떨어뜨린 순간 AI 추론을 강제 종료하고, 로봇 팔 관절을 완벽한 0.0도(default_joint_pos)로 강제 이동시키는 '자동 홈 복귀(Auto-Home)' 하드코딩 로직을 적용.
   - **결과:** AI는 복잡한 조작(Manipulation)만 담당하고 단순 복귀는 프로그래밍이 담당하여 정확도와 효율성 극대화.

2. **학습 데이터 무작위 샘플링 적용 (train_act_v2.py)**
   - 8개의 HDF5 파일에서 각 25개씩, 총 200개의 에피소드를 무작위(Random)로 추출하여 학습하도록 데이터 로더 개편. 데이터의 편향을 줄이고 다양성을 확보.

3. **비전 데이터 증강 버그 픽스 (replay_and_capture_red.py)**
   - **문제:** 사후 렌더링을 진행할 때 Top View는 정상적으로 찍히나 양손 카메라(Left/Right Camera)가 완전히 까맣게(Black Screen) 렌더링되는 현상 발견.
   - **원인:** 기존 스크립트에 양손 카메라의 USD 경로가 하드코딩되어 있었으나, 시뮬레이션 환경 구성 변화로 실제 트리 경로와 일치하지 않아 Replicator가 카메라를 찾지 못함. (VR 실시간 캡처 스크립트 bringup_v.py에서도 물리 엔진과 Replicator 간의 동기화 버그로 깊게 중첩된 관절 내 카메라 캡처가 누락되는 문제 확인)
   - **해결:** _find_camera_prim_by_name() 함수를 구현하여 씬(Stage) 내에서 카메라 Prim을 동적으로 탐색하여 바인딩하도록 수정. 
   - **비전 대비 강화:** 상자는 빨간색, 바닥은 진회색으로 변경하여 고대비(High Contrast) 시각 데이터를 구축하여 허공 삽질 문제를 해결.

4. **기술적 통찰 및 향후 계획**
   - **사후 렌더링 딜레이:** 에피소드 전환 시 RTX 렌더러의 비동기 파이프라인 특성상 이전 에피소드의 마지막 장면이 2~3프레임 남는 고스트(Ghost) 현상 확인. 전체 프레임 대비 비중이 극히 적어 AI 학습에는 무해함을 검증.
   - **모바일 베이스 주차:** 로봇 바퀴(cmd_vel)가 컨베이어 벨트 앞 원위치로 완벽하게 돌아가지 못하는 현상을 보완하기 위해, Nav2 없이 오도메트리 기반 PID(비례) 제어기를 적용한 '미니 자율 주차 코드' 추가 도입을 검토 중.

### [2026-06-10 PM] Vision ACT 최적화 및 물류 DB 연동 완료

1. **Vision ACT 학습 최적화 및 투트랙(Two-Track) 가동**
   - **문제:** 기존 `train_act_v2.py`에 테스트용으로 하드코딩된 25개 에피소드 제한 때문에 로봇이 궤적 일반화(Generalization)에 실패하고 무한 루프에 빠지는 현상 발생.
   - **해결:** 제한을 해제하여 증강된 800개의 데이터를 100% 활용하도록 개선.
   - **상태:** RTX 5080을 활용하여 메인 데이터셋 20 에포크 학습 중이며, Colab T4를 활용하여 서브 데이터셋(Red Box)을 평행으로 투트랙 학습 진행 중.

2. **ROS 2 기반 물류 시나리오 컨트롤러 고도화 (`sh5_scenario_controller.py`)**
   - **문제:** AI 비전 모델 제어에만 의존하면 전체 물류 시스템 검증에 병목이 발생함.
   - **해결:** 독립적인 `CentralLogisticsDB` ROS 2 노드를 작성하여 `GetPackageRoute`, `CheckWarehouseStatus`, `ReportInboundProgress` 서비스 연동 완료.
   - **보안/안전:** `pause_status` 토픽을 구독하여 AMR 접근 시 인터락(Interlock)이 걸리는 로직 구현.

3. **Dummy Teleport 서브 플랜 구현 (`evaluate_test_vision.py`)**
   - **Dummy Teleport 서브 플랜 구현 (`evaluate_test_vision.py`):** AI 추론을 건너뛰고 큐브를 목표 슬롯으로 즉시 강제 이동시키는 `--dummy_teleport` 옵션 구현 완료. 모델 학습 여부와 무관하게 QR ↔ DB ↔ AMR 연동 테스트를 End-to-End로 즉시 진행할 수 있는 파이프라인 완성.

---

### [2026-06-10 Night] 통합 아키텍처 완성 및 HDF5 서브 플랜 구축

1. **SH5 통합 컨트롤러 완성 (`sh5_integrated.py`)**
   - AMR 팀과 완벽하게 동일한 Script Editor `exec()` 구조로 개발 완료. 하나의 `final_coupan.usd` 창에서 AMR과 SH5 3대가 동시 구동되는 구조 확립.
   - 3가지 작동 모드 스위칭 시스템 구축: `DUMMY_TELEPORT` (빠른 시스템 테스트), `HDF5_REPLAY` (모델 불안정 대비 보험), `ACT_MODEL` (완성된 AI 도입).

2. **HDF5 재생 서브 플랜 구현 (`hdf5_replay_player.py`)**
   - 낼 마감일 시연의 불확실성을 없애기 위해, VR로 수집된 완벽한 전문가 시연(Demonstration) 데이터를 시뮬레이션 로봇에 1:1로 직접 주입하는 재생 엔진 개발. 
   - `sh5_integrated.py` 내의 State Machine과 연동되어 각 슬롯에 해당하는 궤적을 실시간으로 실행.

3. **분산 시뮬레이션(SimSync) 연동 완성**
   - `ControlTowerNode`와의 통신망 구축: `CheckWarehouseStatus` (중복 검사), `ReportInboundProgress` (적재 완료 보고) 서비스 연동 코드 완성.
   - `SimSyncNode` 발견 및 연동: BG2 로봇 라인에서 상자가 밀려났을 때 발생하는 `/sim/sg2_spawn_trigger` 토픽을 SH5 3대가 구독하여 큐에 적재하도록 아키텍처 완비.
   - Pause 인터락: `robot_id`를 `sg2_in_01~03`으로 맞춰 관제탑의 제어 신호와 완벽 동기화.

4. **SH5 로봇 Articulation 자동 로더 구축**
   - Isaac Sim Stage 상에 이미 로봇이 있다면 자동으로 래핑(Wrap)하고, 없다면 USD 에셋을 불러와 정확한 상대 좌표에 스폰(Spawn)시키는 스크립팅 로직 구축 완료.

---

### 📋 앞으로의 남은 작업 (Todo List) - 내일 아침 결전!

**[🔴 0순위] 실측 좌표 교체 및 로봇 관절 바인딩 (내일 오전 가장 먼저)**
- [ ] `final_coupan.usd`를 열고 `find_coords.py`를 실행하여 3개 라인의 컨베이어 끝단(X,Y) 좌표 측정.
- [ ] 측정된 좌표를 `sh5_integrated.py`의 `CONVEYOR_ENDPOINTS`에 반영.
- [ ] `final_coupan.usd` 내의 SH5 로봇 Prim 경로 파악 후 `load_sh5_robot()`의 `existing_paths` 실측값으로 업데이트.
- [ ] (필요시) `cobot3` 및 `cobot3_interfaces` ROS2 노드 실행 테스트.

**[1순위] 전체 파이프라인 (HDF5 Replay 모드) E2E 테스트**
- [ ] Isaac Sim Script Editor에서 `exec(open('.../sh5_integrated.py').read())` 실행.
- [ ] `PICK_AND_PLACE_MODE = "HDF5_REPLAY"` 상태에서 가상/실제 QR 데이터를 받고, HDF5 궤적을 통해 로봇 팔이 관절을 움직여 상자를 배치하는지 확인.
- [ ] 배치 후 4슬롯을 채우면 관제탑에 보고가 가고, 관제탑이 자동으로 AMR에 회전/회수 명령을 내리는지 검증.

**[2순위] Vision ACT 도입 (선택 사항)**
- [ ] 현재 원활히 진행 중인 RTX 5080 학습(800 에피소드) 완료 시 `test_act_verify.pth` 추출.
- [ ] 모드를 `ACT_MODEL`로 변경하여 AI 추론으로 픽앤플레이스가 동작하는지 시도. 실패 시 즉각 `HDF5_REPLAY`로 롤백.

---

### [2026-06-10 Final] D-Day 최종 통합 완성 — 모든 파이프라인 구현 완료

#### 1. DB 통신 완전 구현 (`sh5_integrated.py`)
- **MovePackage.action 클라이언트**: 중복 감지 시 AMR에 직송 명령, 실시간 피드백 출력
- **ManageWorkstation.action 클라이언트**: 4슬롯 만석 시 AMR에 작업대 교체 명령
- **cobot3_ws 인터페이스 100% 호환 검증**: 서비스/액션/토픽명 소스 직접 대조 완료

#### 2. 홈 복귀 모션 구현 (`hdf5_replay_player.py`)
- `_return_to_home()` 추가: 픽앤플레이스 후 HDF5 첫 프레임(대기 자세)으로 50스텝/2.5초 선형 보간 복귀
- 로봇이 슬롯 앞에 뻗은 채 다음 상자를 기다리는 문제 해결

#### 3. BG2 리스폰 방식 컨트롤러 (`sh5_spawn_controller.py`)
- 실제 시연 아키텍처 반영: BG2 PC 디스폰 → SimSyncNode → SH5 PC 컨베이어 끝 리스폰
- Isaac Sim Update 콜백 기반, ROS2 Spin 스레드 분리

#### 4. 단독 테스트 스크립트 (`sh5_solo_demo.py`)
- DB/ROS2/BG2 없이 혼자 픽앤플레이스 검증
- 1/2/3 키 또는 `demo.trigger_line()` 직접 호출

#### 5. C-to-C 소켓 통신 (`sh5_socket_server.py`, `bg2_socket_sender.py`)
- TCP 소켓(포트 9000) + JSON으로 BG2 PC ↔ SH5 PC 직접 연결 (ROS2 불필요)
- 단발성/연결 유지 두 방식, 자동 재연결 내장

#### 6. 문서 정비
- `SH5_DEMO_EXECUTION_GUIDE.md`: 담당자 시연 실행 가이드
- `SH5_INTEGRATION_TRANSFER_GUIDE.md`: 통합 PC 이전 가이드

#### 📋 D-Day Todo
- [ ] SH5 PC IP 확인 → `bg2_socket_sender.py` `SH5_PC_HOST` 수정
- [ ] `final_coupan.usd` SH5 Prim 경로 확인 → `existing_paths` 업데이트
- [ ] `PICK_AND_PLACE_MODE = "HDF5_REPLAY"` E2E 테스트 1회
- [ ] 학습 완료 시 `"ACT_MODEL"` 전환, 불안정 시 즉시 롤백
