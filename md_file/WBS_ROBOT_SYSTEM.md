# WBS (Work Breakdown Structure)
# FFW SG2 자율 물류 분류 로봇 시스템 개발

> **프로젝트 목표:** 쿠팡(Coupang)형 물류 센터에서 FFW SG2 로봇이 컨베이어 벨트에서 택배 상자를 인식·파지·분류하여 작업대(Racker)에 자율적으로 넣는 완전 자율 로봇 시스템 구현.
> 
> **기술 스택:** Isaac Sim / IsaacLab · PPO(Proximal Policy Optimization) · ROS2 · YOLO / ArUco · Python

---

## 1. 📦 프로젝트 계획 및 환경 구축

### 1.1 하드웨어 및 소프트웨어 환경 구성
- [x] 1.1.1 개발 PC 환경 세팅 (Ubuntu 22.04, ROS2 Humble, CUDA)
- [x] 1.1.2 Isaac Sim / IsaacLab 설치 및 버전 확인
- [x] 1.1.3 Blackwell GPU (RTX 5080) PhysX 비호환 이슈 해결 → CPU 물리 파이프라인 사용
- [x] 1.1.4 rsl_rl >= 5.0.0 버전 호환성 패치

### 1.2 로봇 에셋 준비
- [x] 1.2.1 FFW SG2 URDF 검토 (관절 수, 범위, 좌표계 확인)
- [x] 1.2.2 SG2 URDF → Isaac Sim USD 변환
- [x] 1.2.3 그리퍼(rh_p12_rn) 최대 벌림 폭 확인 (109mm)
- [x] 1.2.4 작업대(Racker) USD 에셋 준비
- [ ] 1.2.5 컨베이어 벨트 USD 에셋 제작 / 연동 (5단계 필요)

---

## 2. 🏭 시뮬레이션 환경 설계 (Isaac Sim)

### 2.1 씬(Scene) 구성
- [x] 2.1.1 로봇 배치 및 초기 자세 설정 (`ffw_sg2_cfg.py`)
- [x] 2.1.2 받침대(Table, H=0.8m) 생성 및 배치
- [x] 2.1.3 작업대(Racker) 배치 (AGV 이동 고려, 고정 해제)
- [x] 2.1.4 상자(Package) 물리 속성 설정 (80mm 정육면체, 300g)
- [x] 2.1.5 지면(GroundPlane) 및 조명 설정
- [ ] 2.1.6 컨베이어 벨트 시뮬레이션 추가 (5단계)

### 2.2 물리 안정성 확보
- [x] 2.2.1 Cloner 복제 버그 수정 (env_origins 기반 좌표 명시 계산)
- [x] 2.2.2 테이블-Racker 충돌 폭발 방지 (안전거리 X=0.45m)
- [x] 2.2.3 로봇 바퀴 슬립 방지 (`fix_root_link=True`)
- [x] 2.2.4 그리퍼 ↔ 상자 물리 파지 가능 크기 검증

### 2.3 엔드이펙터(EE) 프레임 정확도
- [x] 2.3.1 FrameTransformer 센서 설정
- [x] 2.3.2 EE 추적 링크 오류 수정 (`arm_r_link7` → `gripper_r_rh_p12_rn_base`)
- [x] 2.3.3 EE 오프셋 튜닝 (그리퍼 집게 끝 방향 +10cm)
- [ ] 2.3.4 EE 추적 정확도 검증 (시뮬레이터 육안 확인)

---

## 3. 🤖 강화학습(RL) 환경 개발

### 3.1 행동 공간(Action Space) 설계
- [x] 3.1.1 우측 팔 7DOF 관절 제어 (`arm_r_joint[1-7]`)
- [x] 3.1.2 우측 그리퍼 바이너리 제어 (열기/닫기)
- [x] 3.1.3 상체 승강(Lift Joint) 제어 추가 → 총 9차원
- [ ] 3.1.4 바퀴(Wheel) 제어 추가 (5단계: Steering + Drive × 3)

### 3.2 관측 공간(Observation Space) 설계
- [x] 3.2.1 로봇 관절 위치/속도 (`joint_pos_rel`, `joint_vel_rel`)
- [x] 3.2.2 엔드이펙터 위치 (`ee_pos_in_robot_root_frame`)
- [x] 3.2.3 상자 위치 (`object_position_in_robot_root_frame`)
- [x] 3.2.4 목표 Bin 위치 (`target_bin_position_in_robot_root_frame`)
- [x] 3.2.5 ArUco ID One-Hot 벡터 (4차원)
- [x] 3.2.6 그리퍼 현재 상태 (`gripper_state`)
- [ ] 3.2.7 로봇 위치/속도 추가 (5단계 이동 학습 시)
- [ ] 3.2.8 카메라 RGB-D 이미지 추가 (6단계 Sim2Real 시)

### 3.3 보상 함수(Reward Function) 설계
- [x] 3.3.1 `reaching_package`: 상자 접근 보상 (std=0.5, weight=3.0)
- [x] 3.3.2 `grasp_package`: 실제 파지 징검다리 보상 (weight=5.0)
- [x] 3.3.3 `lifting_package`: 들어 올리기 보상 (weight=15.0)
- [x] 3.3.4 `package_to_bin`: 목표 Bin 이동 보상 (weight=16.0)
- [x] 3.3.5 `package_to_bin_fine`: 정밀 유도 보상 (weight=5.0)
- [x] 3.3.6 `sorting_success`: 분류 성공 보너스 (weight=50.0)
- [x] 3.3.7 `wrong_bin`: 오분류 패널티 (weight=-30.0)
- [x] 3.3.8 `package_drop_penalty`: 낙하 패널티 (weight=-50.0)
- [x] 3.3.9 `action_rate` / `joint_vel`: 부드러운 동작 유도 패널티
- [ ] 3.3.10 이동 경로 효율성 보상 (5단계)
- [ ] 3.3.11 충돌 패널티 (5단계 AGV 이동 시)

### 3.4 종료 조건(Termination) 설계
- [x] 3.4.1 시간 초과 (`time_out`, 400 스텝)
- [x] 3.4.2 상자 낙하 (`package_drop`, Z < 0.6m)
- [x] 3.4.3 분류 성공 (`sorting_success`)
- [ ] 3.4.4 로봇 충돌 감지 (5단계)

### 3.5 이벤트(리셋) 설계
- [x] 3.5.1 씬 전체 리셋 (`reset_scene_to_default`)
- [x] 3.5.2 로봇 관절 초기화 + 노이즈 추가 (`reset_robot_joints`)
- [x] 3.5.3 상자 위치 초기화 + 랜덤 오프셋 (`reset_package_position`)
- [x] 3.5.4 ArUco ID 랜덤 배정 (`randomize_aruco_id`)
- [ ] 3.5.5 도메인 랜덤화: 상자 질량·마찰 랜덤 (6단계 Sim2Real)

---

## 4. 📚 커리큘럼 학습 (Curriculum Learning)

### 4.1 [1단계] Reaching & Grasping (진행 중)
- [x] 4.1.1 상자 접근 보상 정상화 확인 (`reaching_package` = 2.7)
- [x] 4.1.2 그리퍼 파지 가능 물리 환경 검증 (상자 80mm)
- [x] 4.1.3 EE 프레임 정확도 수정
- [ ] **4.1.4 `lifting_package > 0` 달성 (상자 들어 올리기 최초 성공)** ← 현재 목표
- [ ] 4.1.5 안정적 파지 성공률 > 50% 달성

### 4.2 [2단계] Fixed Placing (단일 바구니 넣기)
- [ ] 4.2.1 `package_to_bin` 보상 우상향 확인
- [ ] 4.2.2 들어 올린 상자를 특정 Bin에 넣는 성공률 > 30%
- [ ] 4.2.3 다양한 초기 자세 노이즈에서도 넣기 성공 검증

### 4.3 [3단계] Sorting & Classification (ArUco 조건부 분류)
- [ ] 4.3.1 ArUco ID에 따른 올바른 Bin 선택 정확도 > 90%
- [ ] 4.3.2 `sorting_success` 보상 안정적 획득
- [ ] 4.3.3 오분류 패널티 작동 검증

### 4.4 [4단계] End-to-End Master (고정 상태 통합)
- [ ] 4.4.1 무작위 상자 위치에서도 전 과정 성공률 > 70%
- [ ] 4.4.2 에피소드 평균 보상 80점 이상 안정화
- [ ] 4.4.3 체크포인트 저장 및 Play 모드 검증

### 4.5 [5단계] Mobile Manipulation (AGV 주행)
- [ ] 4.5.1 `fix_root_link=False` 해제
- [ ] 4.5.2 바퀴 관절 (Steer × 3, Drive × 3) 행동 공간 추가
- [ ] 4.5.3 컨베이어 벨트 시뮬레이션 환경 구축
- [ ] 4.5.4 이동 경로 보상 함수 설계
- [ ] 4.5.5 컨베이어 ↔ 작업대 왕복 자율 이동 성공

### 4.6 [6단계] Sim2Real & Vision AI 통합
- [ ] 4.6.1 Domain Randomization 적용 (질량, 마찰, 조명)
- [ ] 4.6.2 관측 공간에 카메라 RGB-D 이미지 추가
- [ ] 4.6.3 YOLO 기반 상자 탐지 모듈 개발
- [ ] 4.6.4 ArUco 마커 인식 모듈 개발
- [ ] 4.6.5 비전 모듈 → RL Policy 좌표 주입 파이프라인 구축
- [ ] 4.6.6 실제 SG2 로봇 ROS2 인터페이스 연결
- [ ] 4.6.7 실제 로봇 테스트 및 성능 검증

---

## 5. 📡 VR 기반 지도학습 (Imitation Learning & ACT)
- [x] 5.1.1 Vuer WebXR 기반 핸드 트래킹 및 카메라 스트리밍 파이프라인 구축
- [x] 5.1.2 Native OpenXR(SteamVR) 기반 Differential IK 암(Arm) 제어 매핑
- [x] 5.1.3 HDF5 로거 패키지 구축 (obs, actions, cmd_vel 저장 및 취소 기능)
- [x] 5.1.4 Magic Snapping (로컬 좌표계 기반 자석 고정) 안정화 로직 적용
- [x] 5.1.5 100개 이상의 전문가 에피소드 데이터 취득 성공
- [x] 5.1.6 Phase-Aware Z 오프셋 데이터 증강 기법 구현 (`augment_data.py`)
- [x] 5.1.7 SH5 로봇 lift_joint 한계(-0.5m) 디버깅 및 아래층 슬롯 수집 필요성 규명
- [x] 5.1.8 HDF5 데이터 리플레이 시각화 검증 도구 개발 (`replay_data.py`)
- [x] 5.1.9 Behavior Cloning (MLP) 및 ACT (Action Chunking with Transformers) 학습 파이프라인 개발

---

## 6. 🚀 시스템 통합 및 배포

### 6.1 ROS2 인터페이스 및 추론
- [x] 6.1.1 학습 완료된 BC 및 ACT 가중치 추론 연동 (`eval_bc.py`)
- [x] 6.1.2 Temporal Ensembling 및 sliding window 추론 기법 적용
- [x] 6.1.3 통합 Goal-Conditioned ACT 정책을 사용한 단일 시뮬레이션 자율 추론 엔진(`evaluate_act.py`) 완성
- [ ] 6.1.4 실환경 SH5 로봇 모션 매핑 인터페이스 연결
- [ ] 6.1.5 비전(YOLO/ArUco) 데이터 ↔ 모델 입력 실시간 연동

### 6.2 안전 및 예외 처리
- [x] 6.2.1 VR 암 조작 안전장치 우회 및 한계 해제 (ai_worker_config 튜닝)
- [x] 6.2.2 물체 속도 클램핑(1.5m/s)을 통한 물리 터널링 방지
- [x] 6.2.3 추론(Inference) 중 비상 정지(Space) 및 환경 리셋 연동 완료

---

## 📊 전체 진척도 요약

| 단계 | 항목 | 완료 | 전체 | 진척도 |
|---|---|---|---|---|
| 1. 환경 구축 | 인프라/에셋 | 4 | 5 | 80% |
| 2. 시뮬레이션 환경 | 씬/물리/EE | 11 | 13 | 85% |
| 3. RL & IL 데이터 정의 | Action/Obs/Reward | 21 | 31 | 68% |
| 4. 커리큘럼 학습 | 6단계 전체 | 9 | 31 | 29% |
| 5. VR 지도학습 (IL/ACT) | 툴/데이터/학습 | 9 | 9 | **100%** |
| 6. 시스템 통합/배포 | ROS2/안전/검증 | 6 | 11 | 54% |
| **합계** | | **60** | **100** | **60.0%** |
