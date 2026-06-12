# Configuration Rules & Constraints (규칙 및 제약 사항)

이 문서는 본 프로젝트 환경을 셋업하고 개발할 때 반드시 준수해야 하는 시스템적 제약과 해결 규칙을 정리한 문서입니다. 특히 하드웨어 호환성 관련 규칙이 포함되어 있습니다.

## 1. 하드웨어 및 물리 엔진 (PhysX) 규칙
- **제약 사항:** 시스템에 장착된 그래픽 카드는 **NVIDIA RTX 5080 (Blackwell)** 입니다.
- **버그 요약:** 현재 Isaac Sim 4.5/5.1에 내장된 PhysX GPU 솔버는 Blackwell 아키텍처와 호환성 충돌이 있어, GPU 파이프라인을 켤 경우(`use_gpu_pipeline = True`) 즉시 `Segmentation fault (core dumped)` 와 함께 프로그램이 강제 종료됩니다.
- **해결 규칙 (필수):** NVIDIA에서 공식 호환 패치를 발표하기 전까지는 반드시 아래와 같은 분할 연산 모드를 유지해야 합니다.
  1. **물리 시뮬레이션 (Physics):** CPU(Intel Core Ultra 9 275HX)로 구동 
  2. **강화 학습 인공지능 연산 (NN):** GPU(`cuda:0`)로 구동

## 2. API 버전 호환성 규칙 (`rsl_rl`)
- **제약 사항:** `rsl_rl >= 5.0.0` 버전을 사용하고 있습니다.
- **버그 요약:** `stochastic`, `init_noise_std`, `noise_std_type`, `state_dependent_std` 와 같은 파라미터는 최신 버전의 MLPModel 생성자에서 삭제(Deprecated)되었습니다. IsaacLab의 `configclass`가 이 파라미터들을 기본값으로 자동 주입하면 `TypeError`가 발생하여 훈련이 시작조차 되지 않습니다.
- **해결 규칙 (필수):** `train.py` 코드 내부에서 PPO `OnPolicyRunner`를 초기화하기 직전에 `agent_cfg.to_dict()`를 호출하여 생성된 딕셔너리에서 해당 4개의 구버전 키값을 명시적으로 `pop()`하여 제거하는 과정을 유지해야 합니다. (이미 `train.py`에 적용 완료)

## 3. 학습 및 시각화 규칙
- 학습 명령어 실행 시 `--headless` 옵션 유무를 통해 시각적 시뮬레이터 구동 여부를 결정합니다.
- 대규모 장기 훈련 시에는 반드시 `--headless` 모드를 사용하여 CPU 렌더링 오버헤드를 막아야 합니다.
- Tensorboard 로그 디렉토리는 `/home/rokey/dev_ws/rl_ws/logs/rsl_rl/ffw_sg2_sort` 로 통일합니다.
