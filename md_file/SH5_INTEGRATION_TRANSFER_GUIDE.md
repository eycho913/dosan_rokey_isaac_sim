# 🔗 SH5 로봇 통합 PC 이전 가이드

> **목적:** 내 PC의 SH5 로봇 코드를 AMR+DB가 이미 구축된 통합 PC에 이식  
> **작성일:** 2026-06-10  

---

## 📦 상대방 PC에 전달해야 할 파일 목록

### 1. 스크립트 파일 (필수)

| 파일 | 내 PC 경로 | 상대 PC 저장 경로 |
|:---|:---|:---|
| `sh5_spawn_controller.py` | `~/dev_ws/coupang_ws/scripts/` | `~/dev_ws/coupang_ws/scripts/` |
| `sh5_solo_demo.py` | `~/dev_ws/coupang_ws/scripts/` | `~/dev_ws/coupang_ws/scripts/` |
| `hdf5_replay_player.py` | `~/dev_ws/coupang_ws/scripts/` | `~/dev_ws/coupang_ws/scripts/` |

### 2. SH5 USD 모델 (필수)

| 파일 | 내 PC 경로 | 상대 PC 저장 경로 |
|:---|:---|:---|
| `ffw_sh5_follower_custom.usd` | `~/dev_ws/assets/ffw_description/usd/` | `~/dev_ws/assets/ffw_description/usd/` |
| `ffw_sh5_follower.usd` | `~/dev_ws/assets/ffw_description/usd/` | `~/dev_ws/assets/ffw_description/usd/` |
| `configuration/` 폴더 전체 | `~/dev_ws/assets/ffw_description/usd/` | `~/dev_ws/assets/ffw_description/usd/` |

> ⚠️ **USD 내부 상대경로 의존성** 때문에 폴더 구조 그대로 복사해야 함

### 3. HDF5 데이터 파일 (픽앤플레이스 궤적)

| 파일 | 내 PC 경로 | 상대 PC 저장 경로 |
|:---|:---|:---|
| `slot1_1.hdf5`, `slot1_2.hdf5` | `~/dev_ws/datasets/train_data/` | `~/dev_ws/datasets/train_data/` |
| `slot2_1.hdf5`, `slot2_2.hdf5` | `~/dev_ws/datasets/train_data/` | `~/dev_ws/datasets/train_data/` |
| `slot3_1.hdf5`, `slot3_2.hdf5` | `~/dev_ws/datasets/train_data/` | `~/dev_ws/datasets/train_data/` |
| `slot4_1.hdf5`, `slot4_2.hdf5` | `~/dev_ws/datasets/train_data/` | `~/dev_ws/datasets/train_data/` |

> HDF5 파일 총 크기 확인: `du -sh ~/dev_ws/datasets/train_data/*.hdf5`

---

## 📤 파일 전송 명령 (내 PC → 통합 PC)

```bash
# 통합 PC의 IP가 192.168.10.XX 라고 가정
TARGET_IP="192.168.10.XX"   # ← 실제 IP로 교체

# 1. 스크립트 전송
scp ~/dev_ws/coupang_ws/scripts/sh5_spawn_controller.py rokey@$TARGET_IP:~/dev_ws/coupang_ws/scripts/
scp ~/dev_ws/coupang_ws/scripts/sh5_solo_demo.py        rokey@$TARGET_IP:~/dev_ws/coupang_ws/scripts/
scp ~/dev_ws/coupang_ws/scripts/hdf5_replay_player.py   rokey@$TARGET_IP:~/dev_ws/coupang_ws/scripts/

# 2. SH5 USD 모델 전송 (폴더째로)
scp -r ~/dev_ws/assets/ffw_description/ rokey@$TARGET_IP:~/dev_ws/assets/

# 3. HDF5 데이터 전송 (용량 주의 - 수백 MB)
scp ~/dev_ws/datasets/train_data/slot*.hdf5 rokey@$TARGET_IP:~/dev_ws/datasets/train_data/
```

---

## ✅ 통합 PC에서 의존성 검사 (상대방이 실행)

### STEP 1: Python 패키지 확인

```bash
# 필수 패키지 일괄 확인
python3 -c "
import h5py;    print('h5py ✅', h5py.__version__)
import numpy;   print('numpy ✅', numpy.__version__)
import rclpy;   print('rclpy ✅')
print('모든 의존성 OK')
"
```

실패 시 설치:
```bash
pip install h5py numpy
```

### STEP 2: ROS2 cobot3_interfaces 확인

```bash
source ~/dev_ws/cobot3_ws_ref/install/setup.bash

# 인터페이스 존재 확인
ros2 interface show cobot3_interfaces/srv/CheckWarehouseStatus
ros2 interface show cobot3_interfaces/srv/ReportInboundProgress
ros2 interface show cobot3_interfaces/action/MovePackage
ros2 interface show cobot3_interfaces/action/ManageWorkstation
```

> 이 명령들이 에러 없이 출력되면 ✅

### STEP 3: HDF5 파일 무결성 확인

```bash
python3 -c "
import h5py
import glob

files = glob.glob('/home/rokey/dev_ws/datasets/train_data/slot*.hdf5')
print(f'HDF5 파일 {len(files)}개 발견')
for f in sorted(files):
    with h5py.File(f, 'r') as h:
        demos = list(h['data'].keys())
        print(f'  {f.split(\"/\")[-1]}: {len(demos)}개 에피소드')
"
```

기대 출력:
```
HDF5 파일 8개 발견
  slot1_1.hdf5: 50개 에피소드
  slot1_2.hdf5: 50개 에피소드
  ...
```

### STEP 4: USD 파일 경로 확인

```bash
ls -lh ~/dev_ws/assets/ffw_description/usd/ffw_sh5_follower_custom.usd
ls -lh ~/dev_ws/assets/ffw_description/usd/configuration/
```

두 줄 모두 파일이 보여야 ✅

### STEP 5: 네트워크 ROS2 토픽 확인

```bash
source ~/dev_ws/cobot3_ws_ref/install/setup.bash

# DB팀/AMR팀 노드가 올라와 있는지 확인
ros2 topic list | grep -E "pause_status|sg2_spawn_trigger|warehouse"
ros2 service list | grep -E "check_warehouse|report_inbound"
```

기대 출력:
```
/sg2_in_01/pause_status
/sg2_in_02/pause_status
/sg2_in_03/pause_status
/sim/sg2_spawn_trigger
```

---

## 🔧 통합 PC에서 코드 경로 조정

스크립트 파일 3개에서 경로 관련 상수를 통합 PC 환경에 맞게 확인/수정:

### `sh5_spawn_controller.py` & `sh5_solo_demo.py`

```python
# 상단 경로 상수 확인 (대부분 그대로 사용 가능)
BOX_USD_PATH = "/home/rokey/dev_ws/assets/ffw_description/usd/ffw_sh5_follower_custom.usd"
```

### `hdf5_replay_player.py`

```python
# 41번 줄
HDF5_BASE_DIR = Path("/home/rokey/dev_ws/datasets/train_data")  # 경로 동일하면 수정 불필요
```

---

## 🚀 통합 PC Isaac Sim 실행 순서

### 순서 1: 환경 준비 (Terminal)

```bash
source ~/dev_ws/cobot3_ws_ref/install/setup.bash
ros2 topic list | grep sg2_spawn  # DB팀 노드 확인
```

### 순서 2: Isaac Sim 열기

1. Isaac Sim 실행
2. `File → Open` → `final_coupan.usd`
3. **▶ Play** 클릭

### 순서 3: AMR 팀 스크립트 실행 (AMR팀이 수행)

```python
exec(open('~/dev_ws/coupang_ws/scripts/amr_live_existing_stage_true8_qr_camera_controller_gpu.py', encoding='utf-8').read())
```

### 순서 4: SH5 컨트롤러 실행 (우리가 수행)

```python
exec(open('/home/rokey/dev_ws/coupang_ws/scripts/sh5_spawn_controller.py', encoding='utf-8').read())
```

---

## 🗂️ 전체 파일 전달 체크리스트

```
□ sh5_spawn_controller.py     전달 완료
□ sh5_solo_demo.py            전달 완료
□ hdf5_replay_player.py       전달 완료
□ ffw_sh5_follower_custom.usd 전달 완료
□ ffw_sh5_follower.usd        전달 완료
□ configuration/ 폴더 전체    전달 완료
□ slot1_1.hdf5 ~ slot4_2.hdf5 전달 완료 (8개)

□ 통합 PC: h5py 설치 확인
□ 통합 PC: cobot3_interfaces 빌드 확인
□ 통합 PC: HDF5 무결성 확인
□ 통합 PC: ROS2 토픽 연결 확인
□ 통합 PC: Isaac Sim Script Editor 실행 테스트
```

---

## 💡 핵심 주의사항

> [!WARNING]
> **USD 폴더 구조 유지 필수**  
> `ffw_sh5_follower_custom.usd`는 내부에서 `configuration/` 폴더의 하위 USD를 상대경로로 참조합니다.  
> 폴더 구조가 달라지면 Isaac Sim에서 로봇 메시가 깨져서 보입니다.

> [!NOTE]
> **HDF5 경로가 다를 경우**  
> `hdf5_replay_player.py` 41번 줄 `HDF5_BASE_DIR`만 수정하면 모든 슬롯 파일 경로가 자동으로 따라옵니다.

> [!TIP]
> **SH5 로봇이 final_coupan.usd에 없는 경우**  
> `sh5_spawn_controller.py`가 실행되면 `SH5_USD_PATH`의 USD를 로드해서  
> `ROBOT_POSITIONS` 좌표에 자동으로 3대를 스폰합니다. 별도 작업 불필요.
