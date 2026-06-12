# 📋 의연 담당 작업 타임라인 및 기여 정리

> **기간**: 2026년 5월 26일 ~ 2026년 6월 12일  
> **역할**: SH5 Isaac Sim 시뮬레이션 / 모방학습 데이터 파이프라인 / 3대 로봇 물류 자동화  
> **주의**: Isaac Sim 기반 SH5 쌍팔 로봇의 물류 자동화 전체 파이프라인(데이터 수집 → 전처리 → 학습 → 재생 시연)을 설계하고 구현함

---

<h2>의연 담당 작업 타임라인 (데이터 수집 및 환경 구축)</h2>

<table>
  <thead>
    <tr>
      <th nowrap>작업명</th>
      <th nowrap>날짜</th>
      <th nowrap>담당자</th>
      <th nowrap>파트</th>
      <th nowrap>단계</th>
      <th nowrap>완료 여부</th>
    </tr>
  </thead>
  <tbody>
    <tr><td nowrap>Isaac Sim + SH5 USD 로봇 스폰 환경 구축 (finalfac.usd)</td><td nowrap>5월 26일</td><td nowrap>조의연</td><td nowrap>환경 구축</td><td nowrap>씬 세팅</td><td nowrap>완료</td></tr>
    <tr><td nowrap>VR 컨트롤러 기반 SH5 양팔 원격 조작 인터페이스 구현</td><td nowrap>5월 27일</td><td nowrap>조의연</td><td nowrap>데이터 수집</td><td nowrap>조작 인터페이스</td><td nowrap>완료</td></tr>
    <tr><td nowrap>HDF5 에피소드 녹화 파이프라인 구현 (VRDemonstrationLogger)</td><td nowrap>5월 28일</td><td nowrap>조의연</td><td nowrap>데이터 수집</td><td nowrap>녹화 시스템</td><td nowrap>완료</td></tr>
    <tr><td nowrap>카메라 어노테이터 통합 (Left/Right/TopView RGB 160×120 동시 저장)</td><td nowrap>5월 28일</td><td nowrap>조의연</td><td nowrap>데이터 수집</td><td nowrap>비전 데이터 수집</td><td nowrap>완료</td></tr>
    <tr><td nowrap>슬롯별(1~4) HDF5 데이터 분리 저장 구조 설계</td><td nowrap>5월 29일</td><td nowrap>조의연</td><td nowrap>데이터 수집</td><td nowrap>데이터 구조화</td><td nowrap>완료</td></tr>
    <tr><td nowrap>Magic Snapping 파지 보조 로직 구현 (데이터 수집 중 파지 정확도 향상)</td><td nowrap>5월 30일</td><td nowrap>조의연</td><td nowrap>데이터 수집</td><td nowrap>파지 보조</td><td nowrap>완료</td></tr>
    <tr><td nowrap>슬롯 1~4 각 100+ 에피소드 수집 완료 (총 400+ 에피소드)</td><td nowrap>6월 1일</td><td nowrap>조의연</td><td nowrap>데이터 수집</td><td nowrap>에피소드 수집</td><td nowrap>완료</td></tr>
    <tr><td nowrap>TopView QR 카메라 기반 상자 위치 인식 및 좌표 변환 구현</td><td nowrap>6월 2일</td><td nowrap>조의연</td><td nowrap>비전 연동</td><td nowrap>QR 로컬라이제이션</td><td nowrap>완료</td></tr>
  </tbody>
</table>

---

<h2>의연 담당 작업 타임라인 (데이터 전처리 및 증강)</h2>

<table>
  <thead>
    <tr>
      <th nowrap>작업명</th>
      <th nowrap>날짜</th>
      <th nowrap>담당자</th>
      <th nowrap>파트</th>
      <th nowrap>단계</th>
      <th nowrap>완료 여부</th>
    </tr>
  </thead>
  <tbody>
    <tr><td nowrap>freeze_idle_arms.py: 비동작 팔 방해 제거 전처리 구현</td><td nowrap>6월 3일</td><td nowrap>조의연</td><td nowrap>데이터 전처리</td><td nowrap>팔 고정 처리</td><td nowrap>완료</td></tr>
    <tr><td nowrap>create_subset.py: frozen_set 서브셋 추출 (학습/재생용 정제 데이터)</td><td nowrap>6월 3일</td><td nowrap>조의연</td><td nowrap>데이터 전처리</td><td nowrap>서브셋 추출</td><td nowrap>완료</td></tr>
    <tr><td nowrap>augment_data.py: 좌우 미러링 + 관절 노이즈 데이터 증강 구현</td><td nowrap>6월 4일</td><td nowrap>조의연</td><td nowrap>데이터 증강</td><td nowrap>증강 파이프라인</td><td nowrap>완료</td></tr>
    <tr><td nowrap>augment_slot3_to_slot4.py: 슬롯 3→4 변환 증강 (좌우 반전)</td><td nowrap>6월 4일</td><td nowrap>조의연</td><td nowrap>데이터 증강</td><td nowrap>슬롯 변환</td><td nowrap>완료</td></tr>
    <tr><td nowrap>filter_dataset.py: 실패 에피소드 필터링 (trajectory 품질 기준)</td><td nowrap>6월 5일</td><td nowrap>조의연</td><td nowrap>데이터 전처리</td><td nowrap>품질 필터링</td><td nowrap>완료</td></tr>
    <tr><td nowrap>Google Colab A100 환경에서 train_act_v2.py 150 epoch 학습 완료</td><td nowrap>6월 6일</td><td nowrap>조의연</td><td nowrap>모방 학습</td><td nowrap>ACT 학습</td><td nowrap>완료</td></tr>
    <tr><td nowrap>HDF5 EpisodeLoader 구현: 슬롯별 랜덤 에피소드 로드 및 offset 보정</td><td nowrap>6월 7일</td><td nowrap>조의연</td><td nowrap>재생 시스템</td><td nowrap>에피소드 로더</td><td nowrap>완료</td></tr>
  </tbody>
</table>

---

<h2>의연 담당 작업 타임라인 (ROS 2 연동 및 다중 로봇 시연)</h2>

<table>
  <thead>
    <tr>
      <th nowrap>작업명</th>
      <th nowrap>날짜</th>
      <th nowrap>담당자</th>
      <th nowrap>파트</th>
      <th nowrap>단계</th>
      <th nowrap>완료 여부</th>
    </tr>
  </thead>
  <tbody>
    <tr><td nowrap>ros2_sh5_bridge.py: ROS 2 ↔ Isaac Sim 파일큐 브릿지 구현</td><td nowrap>6월 8일</td><td nowrap>조의연</td><td nowrap>ROS 2 연동</td><td nowrap>브릿지 구현</td><td nowrap>완료</td></tr>
    <tr><td nowrap>sh5_bringup_ros2_3robot.py: 3대 로봇 독립 상태머신 병렬 운영 구현</td><td nowrap>6월 8일</td><td nowrap>조의연</td><td nowrap>시뮬레이션</td><td nowrap>3-로봇 시스템</td><td nowrap>완료</td></tr>
    <tr><td nowrap>SlotRegistry: 고객별 슬롯 유지 할당 (같은 고객 → 항상 같은 슬롯)</td><td nowrap>6월 9일</td><td nowrap>조의연</td><td nowrap>시뮬레이션</td><td nowrap>슬롯 관리</td><td nowrap>완료</td></tr>
    <tr><td nowrap>report_inbound_progress: 입고 완료 보고 서비스 연동</td><td nowrap>6월 9일</td><td nowrap>조의연</td><td nowrap>ROS 2 연동</td><td nowrap>입고 보고</td><td nowrap>완료</td></tr>
    <tr><td nowrap>WARMUP_FRAMES=30: 텔레포트 제거 — 현재 자세→첫 프레임 선형 보간</td><td nowrap>6월 10일</td><td nowrap>조의연</td><td nowrap>재생 안정화</td><td nowrap>자세 보간</td><td nowrap>완료</td></tr>
    <tr><td nowrap>stay.hdf5 호밍: 복귀 시 쓰러짐 방지 안전 자세 적용</td><td nowrap>6월 10일</td><td nowrap>조의연</td><td nowrap>재생 안정화</td><td nowrap>안전 호밍</td><td nowrap>완료</td></tr>
    <tr><td nowrap>frozen_set 에피소드 사용: 방해 팔이 제거된 데이터로 재생 교체</td><td nowrap>6월 10일</td><td nowrap>조의연</td><td nowrap>재생 안정화</td><td nowrap>에피소드 교체</td><td nowrap>완료</td></tr>
  </tbody>
</table>

---

<h2>의연 담당 작업 타임라인 (파지 안정화 및 Spawn/Despawn)</h2>

<table>
  <thead>
    <tr>
      <th nowrap>작업명</th>
      <th nowrap>날짜</th>
      <th nowrap>담당자</th>
      <th nowrap>파트</th>
      <th nowrap>단계</th>
      <th nowrap>완료 여부</th>
    </tr>
  </thead>
  <tbody>
    <tr><td nowrap>HDF5-Guided Snapping: box_trajectory 기반 파지 링크 자동 선택</td><td nowrap>6월 11일</td><td nowrap>조의연</td><td nowrap>파지 안정화</td><td nowrap>HDF5 가이드 스냅</td><td nowrap>완료</td></tr>
    <tr><td nowrap>ATTACH_FACTOR=1.0, MAX_BOX_STEP=3.0: 딜레이 없는 즉시 부착 구현</td><td nowrap>6월 11일</td><td nowrap>조의연</td><td nowrap>파지 안정화</td><td nowrap>파라미터 튜닝</td><td nowrap>완료</td></tr>
    <tr><td nowrap>yo-yo 현상 원인 분석: kinematic 박스에 velocity 설정 → PhysX 에러 누적</td><td nowrap>6월 12일</td><td nowrap>조의연</td><td nowrap>물리 안정화</td><td nowrap>버그 분석</td><td nowrap>완료</td></tr>
    <tr><td nowrap>_write_box_pose() 구현: write_root_pose_to_sim / USD XFormable 직접 쓰기</td><td nowrap>6월 12일</td><td nowrap>조의연</td><td nowrap>물리 안정화</td><td nowrap>velocity 차단</td><td nowrap>완료</td></tr>
    <tr><td nowrap>WorkstationManager: 작업대 RACK prim 실시간 Despawn/Spawn</td><td nowrap>6월 12일</td><td nowrap>조의연</td><td nowrap>작업대 관리</td><td nowrap>Spawn/Despawn</td><td nowrap>완료</td></tr>
    <tr><td nowrap>WS_LOCATION_TO_RACK 매핑 확정: RACK_02~04 = sg2_in_01~03 라인</td><td nowrap>6월 12일</td><td nowrap>조의연</td><td nowrap>작업대 관리</td><td nowrap>prim 경로 매핑</td><td nowrap>완료</td></tr>
  </tbody>
</table>

---

## 📊 타임라인

```mermaid
gantt
    title 조의연 담당 작업 타임라인 (5/26~6/12)
    dateFormat  YYYY-MM-DD
    section 환경 구축 / 데이터 수집
    Isaac Sim 환경 구축 및 VR 조작 인터페이스     :done, a1, 2026-05-26, 3d
    HDF5 녹화 파이프라인 및 카메라 통합           :done, a2, 2026-05-28, 2d
    슬롯별 400+ 에피소드 수집 완료               :done, a3, 2026-05-29, 4d
    section 데이터 전처리 / 증강 / 학습
    freeze_idle_arms + create_subset 전처리      :done, b1, 2026-06-03, 1d
    augment_data 증강 파이프라인                 :done, b2, 2026-06-04, 1d
    ACT v2 학습 (Colab A100, 150 epoch)         :done, b3, 2026-06-06, 1d
    HDF5 EpisodeLoader 구현                     :done, b4, 2026-06-07, 1d
    section ROS 2 연동 / 3-로봇 시연
    ros2_sh5_bridge 구현                        :done, c1, 2026-06-08, 1d
    3대 로봇 ReplayController 병렬 구현          :done, c2, 2026-06-08, 1d
    SlotRegistry + report_inbound 연동           :done, c3, 2026-06-09, 1d
    section 재생 안정화
    WARMUP_FRAMES 보간 + stay.hdf5 호밍         :done, d1, 2026-06-10, 1d
    HDF5-Guided Snapping (파지 안정화)           :done, d2, 2026-06-11, 1d
    yo-yo 수정 + _write_box_pose                :done, d3, 2026-06-12, 1d
    WorkstationManager Spawn/Despawn            :done, d4, 2026-06-12, 1d
```

---

## 🔑 핵심 전환점

| # | 전환점 | 관련 내용 | 날짜 |
|---|--------|----------|------|
| 1 | HDF5 녹화 파이프라인 구축 | VR 조작 데이터를 joint/box/image 모두 동시 저장하는 완전한 녹화 시스템 완성 | 5월 28일 |
| 2 | 방해 팔 전처리 도입 | freeze_idle_arms로 비동작 팔 궤적을 stay 자세로 오버라이드 → 학습 품질 개선 | 6월 3일 |
| 3 | 3대 로봇 병렬 시연 | 독립 ReplayController 3개를 병렬로 운영, 각 라인이 독립적으로 pick & place 수행 | 6월 8일 |
| 4 | 워밍업 보간 도입 | 첫 프레임 텔레포트 제거 — 30프레임 선형 보간으로 자연스러운 시작 동작 구현 | 6월 10일 |
| 5 | HDF5-Guided Snapping | box_trajectory로 파지 링크를 자동 선택, 왼/오른손 오류 완전 해결 | 6월 11일 |
| 6 | yo-yo 현상 원인 규명 및 수정 | kinematic 박스의 velocity 설정이 PhysX 에러 누적 → 시뮬 종료 문제 완전 해결 | 6월 12일 |
| 7 | 작업대 실시간 Spawn/Despawn | WorkstationManager로 AMR 이동 시 RACK prim을 즉시 화면에서 제거/복원 | 6월 12일 |

---

## 🧩 담당 역할 요약

조의연는 SNFC 프로젝트에서 'SH5 Isaac Sim 물류 자동화' 파트를 전담하여, Isaac Sim 기반 SH5 쌍팔 로봇의 물류 자동화 전체 파이프라인을 설계하고 구현하였다.

5월 26일부터 6월 초까지는 Isaac Sim 환경 구축, VR 조작 인터페이스, HDF5 녹화 파이프라인을 구현하고 슬롯 1~4에 걸쳐 400여 개의 고품질 시연 데이터를 직접 수집하였다. freeze_idle_arms, create_subset, augment_data 등 전처리 및 증강 파이프라인을 구축하여 학습 데이터 품질을 최적화하고, Google Colab A100 환경에서 Vision-ACT 모델을 150 epoch 학습하였다.

6월 8일 이후에는 ROS 2 브릿지 및 3대 로봇 병렬 시연 시스템을 구현하고, 워밍업 보간·stay.hdf5 호밍·frozen_set 에피소드 등 재생 안정화 기법을 순차적으로 적용하였다.

6월 12일에는 yo-yo 현상의 근본 원인(kinematic 바디 velocity 설정 금지)을 규명하고 `_write_box_pose()` 헬퍼로 완전히 해결하였으며, 작업대 Spawn/Despawn 기능(`WorkstationManager`)을 완성하여 관제탑과의 실시간 연동을 완성하였다.

---

## ✅ 최종 정리

조의연의 주요 기여는 데이터 수집부터 전처리, 학습, Isaac Sim 시뮬레이션 재생 시연에 이르는 전체 모방학습 파이프라인의 설계와 구현에 있다.

특히, HDF5 guided snapping으로 왼/오른손 자동 선택을 구현하고, kinematic 물리 버그를 직접 분석·해결하여 yo-yo 현상 없는 안정적인 파지를 달성하였다. 3대 로봇이 독립적으로 pick & place를 수행하는 병렬 시연 시스템과 작업대 실시간 Spawn/Despawn 기능으로 관제탑과의 완전한 실시간 연동 인프라를 완성하였다. 자세한 디버깅 이력은 [DEBUGGING.md](DEBUGGING.md) 파일에서 확인할 수 있다.

---

## 📁 코드 설명

### 🚀 시연

---

#### `sh5_bringup_ros2_3robot.py` ★ 메인 시연 스크립트

3대의 SH5 쌍팔 로봇이 `sg2_in_01`, `sg2_in_02`, `sg2_in_03` 3개 라인에서 독립적으로 동시에 pick & place를 수행하는 Isaac Sim 메인 시연 스크립트.

**주요 클래스:**

| 클래스 | 역할 |
|--------|------|
| `BringupSceneCfg` | finalfac.usd 씬 + 3대 로봇 + 3개 상자 구성 |
| `SlotRegistry` | 고객별 슬롯 유지 할당 (동일 고객 → 동일 슬롯 보장) |
| `WorkstationManager` | RACK prim 실시간 Despawn/Spawn 관리 |
| `ReplayController` | 라인별 상태머신 (IDLE→SCANNING→WAITING_DB→REPLAYING→HOMING→DONE) |
| `FileQueueReader` | `/tmp/sh5_queue.jsonl` 파일큐 폴링 |

**상태머신 흐름:**
```
IDLE ──트리거──▶ SCANNING ──QR인식──▶ WAITING_DB ──DB응답──▶ REPLAYING ──완료──▶ HOMING ──▶ DONE ──▶ IDLE
```

**HDF5-Guided Snapping 파지 로직:**
```
매 프레임 (REPLAYING):
  1. HDF5 box_trajectory로 기준 위치 계산
  2. 손가락 상태 확인 (finger_pos_avg)
     - >= 0.80 (닫힘): 가장 가까운 로봇 링크에 즉시 부착
     - < 0.80 (열림): HDF5 원본 위치 사용 → 자연스러운 릴리즈
  3. _write_box_pose()로 velocity 없이 위치만 적용
```

**핵심 파라미터:**

| 파라미터 | 값 | 설명 |
|---|---|---|
| `PLAYBACK_SPEED` | `2` | 재생 배속 |
| `WARMUP_FRAMES` | `30` | 첫 프레임 보간 (~1초) |
| `ATTACH_FACTOR` | `1.0` | 링크 중심 완전 부착 |
| `GRASP_DIST` | `0.30 m` | 스냅 활성화 거리 |
| `FINGER_OPEN_THRESH` | `0.80 rad` | 손가락 열림 임계값 |
| `FROZEN_SET_DIR` | `datasets/train_data/frozen_set` | 재생 에피소드 경로 |
| `STAY_HDF5_PATH` | `datasets/stay.hdf5` | 호밍 자세 데이터 |

**실행:**
```bash
isaac-python scripts/sh5_bringup_ros2_3robot.py
```

---

#### `ros2_sh5_bridge.py` ★ ROS 2 브릿지

관제탑(WMS/AMR)과 Isaac Sim 시뮬레이터 사이의 ROS 2 ↔ 파일큐 양방향 브릿지.

**주요 기능:**
- `/sim/sg2_spawn_trigger` 구독 → `check_warehouse_status` 서비스 호출 → `/tmp/sh5_queue.jsonl` 기록
- `/tmp/sh5_report_req.jsonl` 모니터링 → `report_inbound_progress` 서비스 호출
- `/{robot_id}/pause_status` 구독 → `/tmp/sh5_pause.json` 기록 (일시정지 신호)

**파일큐 인터페이스:**

| 파일 | 방향 | 내용 |
|------|------|------|
| `/tmp/sh5_queue.jsonl` | Bridge → Isaac | 패키지 투입 트리거 |
| `/tmp/sh5_qr_req.jsonl` | Isaac → Bridge | QR DB 확인 요청 |
| `/tmp/sh5_qr_result.jsonl` | Bridge → Isaac | DB 중복 체크 결과 |
| `/tmp/sh5_report_req.jsonl` | Isaac → Bridge | 입고 완료 보고 |
| `/tmp/sh5_pause.json` | Bridge → Isaac | 일시정지 신호 |
| `/tmp/sh5_ws_trigger.jsonl` | Bridge → Isaac | 작업대 Spawn/Despawn |

**실행:**
```bash
python3 scripts/ros2_sh5_bridge.py
```

---

### 📦 데이터 수집

---

#### `coupang_sh5_bringup_v.py` ★ VR 조작 데이터 수집

VR 컨트롤러 + 키보드로 SH5 로봇을 직접 조작하며 HDF5 에피소드를 녹화하는 스크립트.

**주요 클래스:**
- `VRDemonstrationLogger` — HDF5 에피소드 녹화/저장/취소
- `TerminalKeyboard` — 키보드 조작 인터페이스

**저장 데이터 구조:**
```
episode_XXXXXX/
├── obs/joint_positions    (T, 14)  관절 각도 (rad)
├── obs/box_pose           (T,  7)  상자 위치 + 쿼터니언 (xyz + wxyz)
├── obs/robot_pose         (T,  7)  로봇 베이스 위치 + 쿼터니언
├── obs/images/left        (T, 120, 160, 3)  좌측 카메라 RGB
├── obs/images/right       (T, 120, 160, 3)  우측 카메라 RGB
└── obs/images/topview     (T, 120, 160, 3)  탑뷰 카메라 RGB (QR 스캔용)
```

**키보드 조작:**
| 키 | 동작 |
|---|---|
| W/S | 전진/후진 |
| A/D | 좌/우 회전 |
| R (또는 1) | 🔴 녹화 시작 |
| T (또는 2) | ⬛ 저장 (성공) |
| C (또는 3) | 🗑️ 취소 (실패) |
| B (또는 4) | 📦 상자 리스폰 |

---

### 🔧 데이터 전처리

---

#### `freeze_idle_arms.py` ★ 비동작 팔 고정

데이터 수집 시 키보드 조작에 의해 움직이는 반대쪽 팔(비동작 팔)의 궤적을 `stay.hdf5` 안전 자세로 대체하는 전처리 스크립트.

- **입력**: 원본 HDF5 에피소드
- **처리**: 타깃 슬롯의 반대쪽 팔 관절값 → stay 자세로 오버라이드
- **출력**: 방해 팔이 제거된 정제 에피소드

#### `create_subset.py` ★ 서브셋 추출

전처리된 에피소드 중 품질 좋은 것만 선별하여 `frozen_set` 디렉토리에 복사.
재생 시연과 학습 모두 이 폴더를 사용.

#### `augment_data.py` ★ 데이터 증강

좌우 미러링 및 관절 가우시안 노이즈로 에피소드 수를 늘리는 증강 스크립트.
- 좌우 미러링: 관절값 부호 반전 + 박스/로봇 Y좌표 반전
- 노이즈: 관절별 σ=0.01 rad 가우시안 노이즈 추가

#### `augment_slot3_to_slot4.py` 슬롯 변환 증강

슬롯 3 데이터를 좌우 반전하여 슬롯 4 데이터로 변환. 수집량이 적은 슬롯을 보완.

#### `filter_dataset.py` 품질 필터링

trajectory 길이, 관절 속도 이상치 등의 기준으로 실패 에피소드를 자동 제거.

---

### 🧠 학습

---

#### `train_act_v2.py` ★ ACT v2 학습

Vision-ACT 모델을 fine-tuning하는 학습 스크립트. Google Colab A100 환경에서 150 epoch 학습.

```bash
python3 scripts/train_act_v2.py \
  --data_dir /datasets/train_data/frozen_set \
  --output_dir /models/ \
  --epochs 150 \
  --batch_size 64
```

- 입력: joint_positions + image (topview) + slot 번호 (goal 조건)
- 출력: joint_targets (14D 행동 시퀀스)
- 체크포인트 자동 저장 및 resume 지원

#### `evaluate_test_vision.py` 추론 평가

Isaac Sim에서 학습된 ACT 모델을 실시간으로 추론하여 로봇을 제어하는 평가 스크립트.

---

### 🔄 HDF5 재생

---

#### `hdf5_replay_player.py` ★ 에피소드 로더

HDF5 파일에서 에피소드를 로드하여 Isaac Sim에 주입하는 로더.

```python
loader = HDF5EpisodeLoader(slot_num=1)
episode = loader.load_random_episode()
# episode["joint_trajectory"]  (T, 14) 관절 궤적
# episode["box_trajectory"]    (T,  7) 상자 궤적 (xyz + quat)
# episode["robot_trajectory"]  (T,  7) 로봇 베이스 궤적
# episode["box_initial_pose"]  초기 상자 위치 (offset 계산용)
# episode["robot_initial_pose"] 초기 로봇 위치 (offset 계산용)
```

---

### 📡 USD 에셋

---

#### `assets/scene/finalfac.usd` 물류 창고 씬 (47 MB)

Isaac Sim 메인 씬 파일. 물류 창고 환경 전체 포함.
- 컨베이어 벨트 3개 라인
- 작업대 RACK_01~10
- AMR 경로 및 마킹
- 조명, 바닥, 배경

#### `assets/scene/RACK.usd` 작업대 모델

`WorkstationManager`에서 SPAWN 시 신규 생성할 때 사용하는 작업대 단일 USD 에셋.

#### `assets/box_assets/PKG_*.usd` 상자 USD 모델

QR 코드가 부착된 쿠팡 택배 상자 USD 모델. 패키지 ID별로 개별 파일로 관리.

