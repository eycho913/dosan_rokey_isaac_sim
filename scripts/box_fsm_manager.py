import torch
import numpy as np

class BoxState:
    ON_MAIN_CONVEYOR = "ON_MAIN_CONVEYOR"
    WAIT_SG2_01 = "WAIT_SG2_01"
    PUSHED_BY_SG2_01 = "PUSHED_BY_SG2_01"
    ON_DAY1_LINE = "ON_DAY1_LINE"
    ON_DAY23_LINE = "ON_DAY23_LINE"
    WAIT_SG2_02 = "WAIT_SG2_02"
    PUSHED_BY_SG2_02 = "PUSHED_BY_SG2_02"
    ON_DAY2_LINE = "ON_DAY2_LINE"
    ON_DAY3_LINE = "ON_DAY3_LINE"
    WAIT_SH5 = "WAIT_SH5"
    SH5_PICKING = "SH5_PICKING"
    IN_WORKSTATION = "IN_WORKSTATION"
    DONE = "DONE"

class BoxLogisticsManager:
    """
    PM 요구사항에 맞춘 상자 상태 머신(FSM) 및 물리 최적화 관리자입니다.
    프레임 방어를 위해 대부분의 구간에서 상자를 Kinematic(script 이동)으로 관리하고,
    SG2가 미는 순간(PUSH)에만 Dynamic Rigid Body를 활성화합니다.
    """
    def __init__(self, scene, stage, box_prim_paths):
        self.scene = scene
        self.stage = stage
        self.box_prim_paths = box_prim_paths
        
        # 각 박스별 상태 관리
        self.boxes = {}
        for path in box_prim_paths:
            box_name = path.split("/")[-1] # 예: "Box_1"
            self.boxes[box_name] = {
                "state": BoxState.ON_MAIN_CONVEYOR,
                "rb_api": self._get_rb_api(path),
                "collider_api": self._get_collider_api(path),
                "name": box_name
            }
            
            # 초기 상태: 물리 OFF (Kinematic 이동)
            self._set_physics(box_name, dynamic=False)

    def _get_rb_api(self, prim_path):
        from pxr import UsdPhysics
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            return UsdPhysics.RigidBodyAPI(prim)
        return None

    def _get_collider_api(self, prim_path):
        from pxr import UsdPhysics
        prim = self.stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            return UsdPhysics.CollisionAPI(prim)
        return None

    def _set_physics(self, box_name, dynamic=False, collision=True):
        """
        상자의 물리(Kinematic/Dynamic) 및 충돌(Collision) 상태를 제어합니다.
        dynamic=False 이면 Kinematic 이동 모드입니다.
        """
        box = self.boxes[box_name]
        if box["rb_api"]:
            # Kinematic이 활성화되면(True), Dynamic Rigid Body가 꺼진 것입니다.
            box["rb_api"].GetKinematicEnabledAttr().Set(not dynamic)
        if box["collider_api"]:
            box["collider_api"].GetCollisionEnabledAttr().Set(collision)

    def zero_velocity(self, box_name):
        """속도 및 각속도 초기화"""
        if box_name in self.scene.keys():
            box_state = self.scene[box_name].data.root_state_w.clone()
            box_state[0, 7:13] = 0.0 # linear_vel (3) + angular_vel (3)
            self.scene[box_name].write_root_state_to_sim(box_state)

    def snap_to_pose(self, box_name, target_pos, target_quat):
        """지정된 위치로 포즈 강제 정렬"""
        if box_name in self.scene.keys():
            box_state = self.scene[box_name].data.root_state_w.clone()
            box_state[0, 0:3] = torch.tensor(target_pos, device=box_state.device)
            box_state[0, 3:7] = torch.tensor(target_quat, device=box_state.device)
            box_state[0, 7:13] = 0.0 # 정렬 시 속도 초기화 보장
            self.scene[box_name].write_root_state_to_sim(box_state)

    def script_move(self, box_name, direction, speed, dt):
        """Python 기반 (Kinematic) 이동"""
        if box_name in self.scene.keys():
            box_state = self.scene[box_name].data.root_state_w.clone()
            # direction: [x, y, z] 이동 벡터
            movement = torch.tensor(direction, device=box_state.device) * speed * dt
            box_state[0, 0:3] += movement
            self.scene[box_name].write_root_state_to_sim(box_state)

    def update(self, dt, global_zones):
        """
        매 시뮬레이션 Step마다 호출되는 핵심 FSM 업데이트.
        global_zones는 STOP_ZONE, ARRIVAL_ZONE 등의 위치 정보를 담은 딕셔너리입니다.
        """
        for box_name, box_info in self.boxes.items():
            state = box_info["state"]
            
            # 1단계: 메인 컨베이어 이동
            if state == BoxState.ON_MAIN_CONVEYOR:
                # Python으로 이동 처리 (물리 OFF)
                self.script_move(box_name, direction=[0, -1, 0], speed=0.3, dt=dt)
                
                # STOP_ZONE_1 도착 판정
                if self._check_arrival(box_name, global_zones["STOP_ZONE_1"]):
                    self.change_state(box_name, BoxState.WAIT_SG2_01)
                    self.snap_to_pose(box_name, global_zones["STOP_ZONE_1"]["pos"], global_zones["STOP_ZONE_1"]["quat"])
                    self.zero_velocity(box_name)

            # 2단계: STOP_ZONE_1 대기
            elif state == BoxState.WAIT_SG2_01:
                # 상자 정지/정렬 유지, SG2 작업 가능 신호 대기
                # (외부 컨트롤러가 PUSHED_BY_SG2_01로 상태 변경)
                pass

            # 3단계: SG2_01 푸시 (유일하게 물리가 켜지는 구간)
            elif state == BoxState.PUSHED_BY_SG2_01:
                # 분기 라인 입구 도착 확인 시 물리를 끄고 다음 상태로 전환
                if self._check_arrival(box_name, global_zones["BRANCH_1_ENTRANCE"]):
                    self.zero_velocity(box_name)
                    # 1일차인지 2/3일차인지 라우팅에 따라 분기
                    next_state = self._route_from_sg2_01(box_name) 
                    self.change_state(box_name, next_state)

            # 4단계: 분기 라인 이동
            elif state == BoxState.ON_DAY1_LINE:
                self.script_move(box_name, direction=[-1, 0, 0], speed=0.3, dt=dt)
                if self._check_arrival(box_name, global_zones["ARRIVAL_ZONE_1"]):
                    self.change_state(box_name, BoxState.WAIT_SH5)
                    self.snap_to_pose(box_name, global_zones["ARRIVAL_ZONE_1"]["pos"], global_zones["ARRIVAL_ZONE_1"]["quat"])
                    self.zero_velocity(box_name)
                    
            elif state == BoxState.ON_DAY23_LINE:
                self.script_move(box_name, direction=[0, -1, 0], speed=0.3, dt=dt)
                if self._check_arrival(box_name, global_zones["STOP_ZONE_2"]):
                    self.change_state(box_name, BoxState.WAIT_SG2_02)
                    self.snap_to_pose(box_name, global_zones["STOP_ZONE_2"]["pos"], global_zones["STOP_ZONE_2"]["quat"])
                    self.zero_velocity(box_name)
                    
            # 5단계: STOP_ZONE_2 및 SG2_02 처리
            elif state == BoxState.WAIT_SG2_02:
                pass # 외부에서 트리거 대기
                
            elif state == BoxState.PUSHED_BY_SG2_02:
                if self._check_arrival(box_name, global_zones["BRANCH_2_ENTRANCE"]):
                    self.zero_velocity(box_name)
                    next_state = self._route_from_sg2_02(box_name)
                    self.change_state(box_name, next_state)

            elif state in [BoxState.ON_DAY2_LINE, BoxState.ON_DAY3_LINE]:
                # 해당 라인으로 이동 후 ARRIVAL_ZONE에 도착하면 정지
                self.script_move(box_name, direction=[-1, 0, 0], speed=0.3, dt=dt)
                arrival_target = "ARRIVAL_ZONE_2" if state == BoxState.ON_DAY2_LINE else "ARRIVAL_ZONE_3"
                if self._check_arrival(box_name, global_zones[arrival_target]):
                    self.change_state(box_name, BoxState.WAIT_SH5)
                    self.snap_to_pose(box_name, global_zones[arrival_target]["pos"], global_zones[arrival_target]["quat"])
                    self.zero_velocity(box_name)

            # 6단계: ARRIVAL_ZONE 및 작업대
            elif state == BoxState.WAIT_SH5:
                # 정지 상태 유지 (물리 OFF), SH5가 픽업(Magic Snapping)하기를 대기
                pass

            elif state == BoxState.SH5_PICKING:
                # 기존 Magic Snapping 로직이 활성화된 상태
                pass

            elif state == BoxState.IN_WORKSTATION:
                # 작업대에 들어간 상자는 이동/회전 완전 고정, 충돌도 끌 수 있음
                pass


    def change_state(self, box_name, new_state):
        old_state = self.boxes[box_name]["state"]
        self.boxes[box_name]["state"] = new_state
        print(f"[FSM] {box_name}: {old_state} -> {new_state}")
        
        # 상태 전환에 따른 물리/충돌 토글 처리 (핵심 요구사항 반영)
        if new_state in [BoxState.PUSHED_BY_SG2_01, BoxState.PUSHED_BY_SG2_02]:
            self._set_physics(box_name, dynamic=True, collision=True)
        elif new_state == BoxState.IN_WORKSTATION:
            self._set_physics(box_name, dynamic=False, collision=False) # 완전 고정, 충돌 OFF
        else:
            # 기타 구간(이동, 대기)에서는 물리를 끄고 Kinematic으로 동작
            self._set_physics(box_name, dynamic=False, collision=True)

    def _check_arrival(self, box_name, zone_info):
        """박스가 목표 존에 도착했는지 거리 기반 계산"""
        if box_name not in self.scene.keys(): return False
        box_pos = self.scene[box_name].data.root_pos_w[0]
        target_pos = torch.tensor(zone_info["pos"], device=box_pos.device)
        dist = torch.norm(box_pos[:2] - target_pos[:2]) # XY 평면 거리
        return dist.item() < zone_info.get("threshold", 0.05)
        
    def _route_from_sg2_01(self, box_name):
        # 상자 ID 또는 바코드 속성에 따른 라우팅
        # 임시 로직
        return BoxState.ON_DAY1_LINE
        
    def _route_from_sg2_02(self, box_name):
        return BoxState.ON_DAY2_LINE
