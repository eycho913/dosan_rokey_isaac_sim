# PPT 슬라이드 프롬프트 모음 — SH5 물류 자동화 프로젝트

---

## Slide 1 — 표지

Create a clean and professional slide with a dark navy background (#1A237E).

At the center, place the following text:
- Main Title (large, white, bold): "SH5 쌍팔 로봇 물류 자동화 시스템"
- Subtitle (medium, light blue): "Isaac Sim 기반 모방학습 Pick & Place 시연"
- Below subtitle (small, gray): "기간: 2026.05.26 ~ 2026.06.12 | 담당: 조의연"
- Bottom right corner: Project logo placeholder

---

## Slide 2 — 목차

Create a clean slide with white background (#FFFFFF).

Header strip (light gray #F1F3F5):
- Title: "목차"
- Subtitle: "프로젝트 전체 구성"

Main area: Numbered list with icons, 2 columns:
Left:
1. 프로젝트 개요 및 목표
2. 시스템 아키텍처
3. 데이터 수집 파이프라인
4. 데이터 전처리 및 증강

Right:
5. ACT 모방학습 모델
6. HDF5 재생 시연 시스템
7. 물리 안정화 기술 (yo-yo 수정)
8. 작업대 Spawn/Despawn
9. ROS 2 연동 및 최종 시연
10. 향후 계획

---

## Slide 3 — 프로젝트 개요

Create a clean slide with white background.

Header strip:
- Title: "프로젝트 개요 및 목표"
- Subtitle: "Isaac Sim × SH5 쌍팔 로봇 × 모방학습"

Layout: 2 columns.

Left column (50%):
- Header: "해결하고자 한 문제"
- Bullets:
  * 물류 창고 Pick & Place 자동화
  * 다관절 손(Dexterous Hand) 정교한 파지
  * 3개 라인 동시 처리 (sg2_in_01~03)
  * WMS/AMR 관제탑 실시간 연동

Right column (50%):
- Header: "핵심 달성 목표"
- Bullets:
  * VR 조작으로 400+ 에피소드 수집
  * Vision-ACT 모델 150 epoch 학습
  * HDF5 궤적 재생으로 안정적 시연
  * yo-yo 현상 완전 제거 (PhysX 수정)
  * 작업대 실시간 Spawn/Despawn 구현

---

## Slide 4 — 전체 시스템 아키텍처

Create a clean slide with white background.

Header strip:
- Title: "전체 시스템 아키텍처"
- Subtitle: "관제탑 ↔ ROS 2 브릿지 ↔ Isaac Sim 연동 흐름"

Layout: Below header, full-width flow diagram area.

Left column (40%):
- Header: "구성 요소"
- Bullets:
  * **AMR/WMS 관제탑**: sg2_spawn_trigger 발행
  * **ros2_sh5_bridge.py**: ROS 2 ↔ 파일큐 변환
  * **sh5_bringup_ros2_3robot.py**: Isaac Sim 메인
  * **WorkstationManager**: RACK prim 관리
  * **파일큐 6종**: /tmp/sh5_*.jsonl

Right column (60%): Large image placeholder.
  [아키텍처 다이어그램: AMR→Bridge→Isaac Sim→3Robot 흐름도]

---

## Slide 5 — 데이터 수집: VR 텔레오퍼레이션

Create a clean slide with white background.

Header strip:
- Title: "데이터 수집 — VR 텔레오퍼레이션"
- Subtitle: "coupang_sh5_bringup_v.py | 총 400+ 에피소드"

Layout: 2 columns.

Left column (45%):
- Header: "수집 방식"
- Bullets:
  * VR 컨트롤러 + 키보드 하이브리드 조작
  * Magic Snapping으로 파지 정확도 보조
  * R키: 녹화 시작 / T키: 저장 / C키: 취소
  * 슬롯 1~4 각 100+ 에피소드

- Header: "저장 데이터 (HDF5)"
- Small table:
  | 키 | shape |
  |---|---|
  | joint_positions | (T, 14) |
  | box_pose | (T, 7) |
  | images/topview | (T,120,160,3) |

Right column (55%): 2 image placeholders stacked.
  [위: Isaac Sim VR 조작 화면 / 아래: TopView 카메라 + Left/Right 카메라 3분할 화면]

---

## Slide 6 — 데이터 전처리 파이프라인

Create a clean slide with white background.

Header strip:
- Title: "데이터 전처리 파이프라인"
- Subtitle: "freeze_idle_arms → create_subset → augment_data"

Layout: Full-width horizontal flow diagram (3 steps).

Step 1 box (light blue):
- Title: "① 방해 팔 고정"
- Script: freeze_idle_arms.py
- Content: 비동작 팔 궤적을 stay.hdf5 안전 자세로 대체

Step 2 box (light green):
- Title: "② 서브셋 추출"
- Script: create_subset.py
- Content: 품질 좋은 에피소드만 frozen_set으로 선별

Step 3 box (light orange):
- Title: "③ 데이터 증강"
- Script: augment_data.py
- Content: 좌우 미러링 + 관절 노이즈 → 에피소드 2배 확장

Below flow: small note box
- "문제: 수집 시 키보드 조작에 의해 반대쪽 팔이 움직여 학습 방해"
- "해결: 비동작 팔 관절값을 stay 자세로 강제 오버라이드"

---

## Slide 7 — ACT 모방학습 모델

Create a clean slide with white background.

Header strip:
- Title: "ACT 모방학습 모델 (Vision-ACT)"
- Subtitle: "train_act_v2.py | Google Colab A100 | 150 epoch"

Layout: 2 columns.

Left column (50%):
- Header: "모델 아키텍처"
- Bullets:
  * **CVAE Encoder**: 액션 시퀀스 → latent z 추출
  * **State Encoder**: 관절 14D + TopView 이미지 → Transformer 인코딩
  * **Action Decoder**: 미래 K프레임 액션 일괄 예측 (Action Chunking)
  * **목표 조건**: slot 번호로 1~4번 슬롯 구분

- Header: "학습 설정"
- Bullets:
  * batch_size=64, epochs=150
  * Phase 가중치: 파지(3.0x), 리프트(1.5x), 삽입(3.0x)
  * 데이터: frozen_set 400+ 에피소드

Right column (50%): Image placeholder.
  [Vision-ACT 아키텍처 다이어그램 또는 loss curve 그래프]

---

## Slide 8 — HDF5 궤적 재생 시스템

Create a clean slide with white background.

Header strip:
- Title: "HDF5 궤적 재생 시스템"
- Subtitle: "hdf5_replay_player.py | 고품질 시연 보험"

Layout: 2 columns.

Left column (45%):
- Header: "재생 방식"
- Bullets:
  * 수집된 VR 시연 데이터를 Isaac Sim에 1:1 주입
  * 슬롯별 랜덤 에피소드 자동 선택
  * offset 보정: 녹화 위치 → 실제 라인 위치
  * 2배속 재생 (PLAYBACK_SPEED=2)

- Header: "안정화 기법 3종"
- Numbered list:
  1. WARMUP_FRAMES=30: 텔레포트 → 선형 보간
  2. frozen_set 에피소드: 방해 팔 제거
  3. stay.hdf5 호밍: 쓰러짐 방지 복귀

Right column (55%): Image placeholder.
  [재생 중 Isaac Sim 화면: 로봇이 상자를 집어 슬롯에 넣는 장면]
