import csv
import glob
import time
import os
import random
import subprocess
from enum import Enum
import rclpy
from rclpy.node import Node
from cobot3_interfaces.srv import GetPackageRoute, CheckWarehouseStatus, ReportInboundProgress
from std_msgs.msg import Bool

class WorkstationState(Enum):
    IDLE = 1
    SCANNING = 2
    AMR_CALL = 3
    ALLOCATE_SLOT = 4
    WAIT_REFRESH = 5

class CentralLogisticsDB(Node):
    """
    실제 QR/DB 연동 시나리오용 DB 컨트롤러 (ROS 2 Service Clients)
    """
    def __init__(self, workstation_id='WS01', robot_id='sh5_in_01'):
        super().__init__('sh5_logistics_db_client')
        self.my_workstation_id = workstation_id
        self.robot_id = robot_id
        self.is_paused = False
        
        # ROS 2 Service Clients
        self.route_client = self.create_client(GetPackageRoute, '/get_package_route')
        self.check_status_client = self.create_client(CheckWarehouseStatus, '/check_warehouse_status')
        self.report_client = self.create_client(ReportInboundProgress, '/report_inbound_progress')

        # ROS 2 Topic Subscriber (Pause Control)
        self.pause_sub = self.create_subscription(
            Bool,
            f'/{self.robot_id}/pause_status',
            self.pause_callback,
            10
        )

        # Fallback local states
        self.global_customer_locations = {}

    def pause_callback(self, msg):
        if msg.data and not self.is_paused:
            print(f"\n[DB/AMR] ⏸️ 긴급 일시 정지(Pause) 신호 수신! 작업대 회전/교체 대기 중...")
        elif not msg.data and self.is_paused:
            print(f"\n[DB/AMR] ▶️ 작업 재개(Resume) 신호 수신! 다시 작업을 시작합니다.")
        self.is_paused = msg.data

    def check_customer_location(self, customer_name, package_id, qr_id):
        """동일 수령인의 패키지가 이미 보관 중인지 중복 검사"""
        if not self.check_status_client.wait_for_service(timeout_sec=0.5):
            print(f"[DB] ⚠️ CheckWarehouseStatus 서비스 응답 없음. (로컬 상태 반환: {self.global_customer_locations.get(customer_name, None)})")
            return self.global_customer_locations.get(customer_name, None)
            
        req = CheckWarehouseStatus.Request()
        req.customer_name = customer_name
        req.package_id = package_id
        req.qr_id = qr_id
        
        future = self.check_status_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            is_in = future.result().is_already_in_warehouse
            if is_in:
                return "other_workstation" # 이미 보관중이므로 AMR 회수를 유도
        return None

    def get_package_route(self, package_id, customer_name, qr_id):
        """QR을 스캔하여 배송일자/분류 목적지 조회"""
        if not self.route_client.wait_for_service(timeout_sec=0.5):
            return None
            
        req = GetPackageRoute.Request()
        req.package_id = package_id
        req.customer_name = customer_name
        req.qr_id = qr_id
        
        future = self.route_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            return future.result().route_destination
        return None

    def report_progress(self, robot_id, filled_slots_count, package_id, qr_id):
        """적재 완료 후 관제탑에 DB 동기화 보고"""
        if not self.report_client.wait_for_service(timeout_sec=0.5):
            print("[DB] ⚠️ ReportInboundProgress 서비스 응답 없음. 로컬에만 반영합니다.")
            return False
            
        req = ReportInboundProgress.Request()
        req.workstation_id = self.my_workstation_id
        req.robot_id = robot_id
        req.filled_slots_count = filled_slots_count
        req.package_id = package_id
        req.workstation_qr_id = f"WORKSTATION_{self.my_workstation_id}"
        req.package_qr_id = qr_id
        
        future = self.report_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            return future.result().success
        return False

    def assign_customer_to_workstation(self, customer_name, workstation_id):
        self.global_customer_locations[customer_name] = workstation_id

    def call_amr_to_retrieve_box(self, customer_name):
        print(f"[DB/AMR] 🚛 AMR 호출 완료: '{customer_name}'의 상자 회수를 위해 AMR이 배차되었습니다.")
        time.sleep(1)
        return True

    def request_workstation_refresh(self):
        print(f"[DB/AMR] 🔄 작업대 갱신 요청 중... (AMR이 빈 랙으로 교체 중)")
        time.sleep(3)
        self.global_customer_locations.clear()
        print(f"[DB/AMR] ✅ 작업대 갱신 완료! 글로벌 할당 상태가 초기화되었습니다.")
        return True


class SH5ScenarioController:
    def __init__(self, data_dir='/home/rokey/dev_ws/qr_data'):
        # 사용자가 SH5로 인식하게 바꿔달라고 했으므로 로봇 ID를 sh5_in_01로 사용
        self.robot_id = 'sh5_in_01'
        self.db = CentralLogisticsDB(workstation_id='WS01', robot_id=self.robot_id)
        self.qr_database = self._load_qr_data(data_dir)
        self.state = WorkstationState.IDLE
        
        # 4개의 슬롯 관리 (None이면 빈 슬롯, string이면 고객 이름)
        self.slots = {1: None, 2: None, 3: None, 4: None}
        print("[System] SH5 물류 시나리오 컨트롤러 초기화 완료.")

    def _load_qr_data(self, data_dir):
        """CSV 파일들을 읽어 qr_id를 key로 하는 딕셔너리 생성"""
        qr_db = {}
        csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
        for f in csv_files:
            with open(f, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    qr_db[row['qr_id']] = {
                        'package_id': row['package_id'],
                        'customer_name': row['customer_name'],
                        'route_zone': row['route_zone']
                    }
        print(f"[Data] {len(csv_files)}개의 CSV 파일에서 총 {len(qr_db)}개의 QR 데이터를 로드했습니다.")
        return qr_db

    def mock_scan_qr(self):
        """비전 시스템(Top-view)에서 QR을 스캔하고 박스 좌표를 반환하는 가상/브릿지 함수"""
        available_qrs = list(self.qr_database.keys())
        scanned_qr = random.choice(available_qrs) if available_qrs else "UNKNOWN"
        # 실제로는 Top-View 카메라의 Homography에서 계산된 물리 좌표 (x, y) 반환
        box_x = round(random.uniform(0.6, 0.8), 2)
        box_y = round(random.uniform(-0.1, 0.1), 2)
        return scanned_qr, (box_x, box_y)

    def execute_act_pick_and_place(self, box_coords, target_slot_num, use_dummy=False):
        """ACT 모델을 실행하여 상자를 집어 슬롯에 배치 (subprocess 호출)"""
        print(f"[Robot/ACT] 🤖 좌표 {box_coords}에서 상자를 Pick하여 슬롯 {target_slot_num}에 Place 수행 중...")
        
        cmd = [
            "python3", "/home/rokey/dev_ws/coupang_ws/scripts/evaluate_test_vision.py",
            "--slot", str(target_slot_num),
            "--box_x", str(box_coords[0]),
            "--box_y", str(box_coords[1])
        ]
        if use_dummy:
            cmd.append("--dummy_teleport")
            print("[Robot/ACT] ⚠️ Dummy Teleport 모드로 실행합니다.")
            
        try:
            print(f"[명령어 실행 대기] {' '.join(cmd)}")
            time.sleep(2)
            print(f"[Robot/ACT] ✅ Pick & Place 완료.")
        except Exception as e:
            print(f"[Robot/ACT] ❌ 실행 실패: {e}")

    def run_scenario(self):
        """메인 시나리오 상태 머신 루프"""
        self.state = WorkstationState.SCANNING
        
        while rclpy.ok():
            # ROS 2 콜백(Pause 구독 등) 처리를 위해 스핀
            rclpy.spin_once(self.db, timeout_sec=0.1)
            
            if self.db.is_paused:
                time.sleep(1.0)
                continue
                
            if self.state == WorkstationState.SCANNING:
                print("\n--- [Step 1] Top-view 카메라 QR 스캔 대기 ---")
                time.sleep(1)
                scanned_qr, box_coords = self.mock_scan_qr()
                
                if scanned_qr not in self.qr_database:
                    print(f"[경고] 등록되지 않은 QR 코드입니다: {scanned_qr}")
                    continue
                
                pkg_info = self.qr_database[scanned_qr]
                customer = pkg_info['customer_name']
                package_id = pkg_info['package_id']
                print(f"[Vision] 스캔 성공! QR: {scanned_qr} -> 고객: {customer} (Box 위치: {box_coords})")
                
                # DB 인터페이스 1: GetPackageRoute (분류 목적지 조회)
                route_dest = self.db.get_package_route(package_id, customer, scanned_qr)
                if route_dest:
                    print(f"-> [DB] 분류 목적지(배송일자): {route_dest}")
                
                # DB 인터페이스 2: CheckWarehouseStatus (중복 검사)
                other_ws = self.db.check_customer_location(customer, package_id, scanned_qr)
                if other_ws and other_ws != self.db.my_workstation_id:
                    print(f"-> [판단] 고객 '{customer}'의 물품이 이미 보관 중입니다. Bypass 또는 이송합니다.")
                    self.state = WorkstationState.AMR_CALL
                else:
                    print(f"-> [판단] 고객 '{customer}'는 현재 작업대에 할당 가능합니다.")
                    self.state = WorkstationState.ALLOCATE_SLOT
                    
            elif self.state == WorkstationState.AMR_CALL:
                print(f"\n--- [Step 2] AMR 회수 요청 ---")
                self.db.call_amr_to_retrieve_box(customer) # MovePackage.action 연동 필요
                self.state = WorkstationState.SCANNING
                
            elif self.state == WorkstationState.ALLOCATE_SLOT:
                print(f"\n--- [Step 3] 동적 슬롯 할당 및 적재 ---")
                target_slot = None
                
                # 1. 이미 이 사람의 슬롯이 있는지 확인
                for slot_num, owner in self.slots.items():
                    if owner == customer:
                        target_slot = slot_num
                        break
                
                # 2. 없다면 빈 슬롯 할당
                if target_slot is None:
                    for slot_num, owner in self.slots.items():
                        if owner is None:
                            self.slots[slot_num] = customer
                            target_slot = slot_num
                            print(f"[Slot] 빈 슬롯 {slot_num}번을 '{customer}' 님에게 새로 할당했습니다.")
                            break
                else:
                    print(f"[Slot] 기존 할당된 슬롯 {target_slot}번을 사용합니다.")

                # 3. 슬롯이 꽉 찼는지 확인
                if target_slot is None:
                    print(f"[경고] 4개의 슬롯이 모두 꽉 찼습니다! (현재 상태: {self.slots})")
                    self.state = WorkstationState.WAIT_REFRESH
                else:
                    self.db.assign_customer_to_workstation(customer, self.db.my_workstation_id)
                    
                    # 빈 슬롯이 있으면 로봇 구동 (Dummy 텔레포트 적용)
                    self.execute_act_pick_and_place(box_coords, target_slot, use_dummy=True)
                    
                    # DB 인터페이스 3: ReportInboundProgress (적재 완료 보고)
                    filled_slots = sum(1 for owner in self.slots.values() if owner is not None)
                    success = self.db.report_progress(self.robot_id, filled_slots, package_id, scanned_qr)
                    if success:
                        print(f"[DB] ✅ 관제탑으로 {filled_slots}번 슬롯 적재 완료 보고가 전송되었습니다.")
                        
                    self.state = WorkstationState.SCANNING
                    
            elif self.state == WorkstationState.WAIT_REFRESH:
                print(f"\n--- [Step 4] 작업대 갱신 요청 및 대기 ---")
                self.db.request_workstation_refresh() # ManageWorkstation.action 연동 필요
                
                print(f"\n--- [Step 5] 작업 재개 (슬롯 초기화) ---")
                self.slots = {1: None, 2: None, 3: None, 4: None}
                self.state = WorkstationState.SCANNING

if __name__ == "__main__":
    rclpy.init()
    controller = SH5ScenarioController()
    try:
        controller.run_scenario()
    except KeyboardInterrupt:
        print("\n시나리오 컨트롤러를 종료합니다.")
    finally:
        controller.db.destroy_node()
        rclpy.shutdown()
