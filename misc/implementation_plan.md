# ROBOTIS Open-Source 환경 구축 계획 (SH5 VR Teleoperation)

공식 오픈소스(`robotis_lab`, `robotis_dds_python`, `ai_worker`, `cyclo`)를 기반으로 10일 프로젝트 타임라인에 맞춘 환경 구축 계획입니다. 기존에 구축하던 환경 대신, 검증된 공식 파이프라인으로 전환합니다.

## User Review Required

> [!IMPORTANT]  
> **설치 방식 선택 (Local vs Docker)**
> 공식 가이드는 Docker 사용을 권장합니다(Isaac Sim부터 모든 종속성 포함). 하지만 사용자님의 PC에는 이미 로컬에 Isaac Sim과 Isaac Lab이 세팅되어 있습니다.
> **본 계획은 기존 환경을 최대한 활용하여 속도를 높이기 위해 "Local 환경에 직접 설치"하는 방향으로 작성되었습니다.** 만약 Docker 환경으로 완전히 분리하고 싶으시다면 피드백을 통해 알려주세요!

## Open Questions

> [!WARNING]
> 현재 시스템에 `cmake`, `build-essential` 등 컴파일용 패키지가 설치되어 있나요? (CycloneDDS 소스 빌드에 필요합니다. 없으면 제가 설치 스크립트에 포함하겠습니다.)

## Proposed Changes

---

### 1. CycloneDDS 설치 (통신 미들웨어 코어)
VR 데이터를 Isaac Sim으로 실시간 전송하기 위한 고성능 통신 라이브러리를 소스 코드에서 빌드하여 설치합니다.

#### [NEW] 빌드 및 환경변수 설정
- `https://github.com/eclipse-cyclonedds/cyclonedds.git` (v0.10.2) 클론
- `/home/rokey/cyclonedds/install` 경로에 CMake 빌드 및 설치
- `~/.bashrc`에 `CYCLONEDDS_HOME` 및 `LD_LIBRARY_PATH` 추가

---

### 2. robotis_dds_python 설치 (파이썬 브릿지)
CycloneDDS를 파이썬(Isaac Sim 내부)에서 사용할 수 있게 해주는 SDK를 설치합니다.

#### [NEW] 패키지 설치
- `https://github.com/ROBOTIS-GIT/robotis_dds_python.git` 클론
- Isaac Lab의 Python 환경(`isaaclab.sh -p -m pip`)을 사용하여 패키지 설치

---

### 3. robotis_lab 설치 (환경 및 학습 파이프라인)
AI Worker 로봇과 SH5 손에 대한 물리 환경, VR 연동 스크립트, 모방학습 코드가 포함된 메인 레포지토리를 세팅합니다.

#### [NEW] 클론 및 연동
- `https://github.com/ROBOTIS-GIT/robotis_lab.git` 클론 (submodule 포함)
- `robotis_lab` 패키지를 Isaac Lab 환경에 설치 및 인식되도록 경로 설정 (`pip install -e .`)

---

### 4. ai_worker 최신화 검토
현재 워크스페이스(`/home/rokey/dev_ws/isaac_sim/src/ai_worker`)에 있는 로봇 모델과 최신 버전 간의 호환성을 검토합니다.

## Verification Plan

환경 구축이 완료된 후 다음 단계로 검증을 진행합니다.

### Automated Tests
- 통신 테스트: `robotis_dds_python`의 `example` 폴더에 있는 Publisher/Subscriber가 정상적으로 통신하는지 확인.

### Manual Verification
- **Isaac Sim 로봇 구동 테스트:** 
  ```bash
  python scripts/sim2real/bringup/sh5_dds_bringup.py --domain_id 30 --enable_gravity --enable_camera_views
  ```
  명령어를 실행하여 Isaac Sim 환경에 SH5 로봇이 정상적으로 렌더링되고 물리 엔진이 적용되는지 확인.
- **Quest 2 VR 테스트:** Quest 2에서 JointTrajectory 토픽을 쏘았을 때 시뮬레이터 안의 손가락이 실시간으로 반응하는지 육안 확인.
