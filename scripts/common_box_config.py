import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

def get_yellow_box_cfg(prim_path="{ENV_REGEX_NS}/Box", pos_x=0.7, pos_y=0.0, pos_z=0.85):
    """
    팀원 공통으로 사용하는 모방학습용 노란색(오렌지색) 상자(Box) 에셋 설정입니다.
    
    [통합 시 주의사항]
    - 프레임 방어를 위해 초기 Z좌표(pos_z)는 컨베이어 높이(기본 0.85m)로 고정합니다.
    - 무겁고 마찰력이 강하므로, SG2가 미는 순간(PUSH) 외에는 물리 연산(Dynamic)을 
      끄고(Kinematic) 파이썬 스크립트로 좌표만 이동시키는 것을 권장합니다.
      
    Args:
        prim_path (str): 상자가 생성될 USD 경로
        pos_x (float): 초기 스폰 X 좌표
        pos_y (float): 초기 스폰 Y 좌표
        pos_z (float): 초기 스폰 Z 좌표 (컨베이어 표면 높이로 고정 권장)
    
    Returns:
        RigidObjectCfg: 물리, 마찰력, 질량이 모두 세팅된 상자 설정 객체
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 손가락 마찰력이 자연스럽게 작용하도록 감쇠 조절
                linear_damping=0.1,
                angular_damping=5.0,
                max_depenetration_velocity=0.3,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            # 안정적인 파지를 위해 질량을 1.5kg으로 묵직하게 세팅
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),  
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002, # 2mm: 빠른 충돌/관통 방지
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.38, 0.08)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", # 로봇과 접촉 시 최대 마찰력 적용
                static_friction=2.0,         # 강한 정지 마찰력
                dynamic_friction=1.8,        # 강한 동마찰력
                restitution=0.0,             # 통통 튀지 않음
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(pos_x, pos_y, pos_z),
            rot=(1.0, 0.0, 0.0, 0.0)
        )
    )
