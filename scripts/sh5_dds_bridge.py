import threading
import time
import isaaclab.utils.math as math_utils
from cyclonedds.core import Qos, Policy
from robotis_dds_python.idl.builtin_interfaces.msg import Time_
from robotis_dds_python.idl.geometry_msgs.msg import Point_, Pose_, Quaternion_, Transform_, TransformStamped_, Twist_, Vector3_
from robotis_dds_python.idl.nav_msgs.msg import Odometry_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.tf2_msgs.msg import TFMessage_
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.tools.topic_manager import TopicManager
from robotis_lab.controllers.swerve import SwerveDriveController, SwerveModule, SwerveOdometry

class SH5DdsBridge:
    def __init__(
        self,
        robot,
        topic_manager: TopicManager,
        topic_names: dict[str, str],
        joint_states_topic: str,
        odom_topic: str,
        tf_topic: str,
        base_frame: str,
        odom_frame: str,
        trajectory_qos: Qos,
        cmd_vel_topic: str | None,
        swerve_modules: list[SwerveModule],
        wheel_radius: float,
        cmd_vel_timeout: float,
    ):
        self.robot = robot
        self.base_frame = base_frame
        self.odom_frame = odom_frame
        self.swerve_modules = swerve_modules
        self.wheel_radius = wheel_radius
        self.cmd_vel_timeout = cmd_vel_timeout
        self.swerve_controller = (
            SwerveDriveController(swerve_modules, wheel_radius) if swerve_modules else None
        )
        self.odometry = (
            SwerveOdometry(
                [module.x_offset for module in swerve_modules],
                [module.y_offset for module in swerve_modules],
                wheel_radius,
            )
            if swerve_modules
            else None
        )
        self._last_swerve_update_time = time.monotonic()
        self.running = True
        self.lock = threading.Lock()
        self.pending_positions: dict[str, float] = {}
        self.latest_cmd_vel = (0.0, 0.0, 0.0)
        self.last_cmd_vel_time = 0.0
        self.unknown_joints: set[str] = set()
        self._warned_missing_base_frame = False
        self._warned_missing_swerve_joints: set[str] = set()
        self._body_names = list(self.robot.data.body_names)
        self._base_id = (
            self._body_names.index(self.base_frame) if self.base_frame in self._body_names else None
        )
        self._joint_name_to_index = {
            name: index for index, name in enumerate(self.robot.data.joint_names)
        }
        # [DEBUG] 관절 이름 → 인덱스 매핑 출력 (증강 스크립트 인덱스 확인용)
        print("\n[DEBUG] ===== JOINT INDEX MAP =====")
        for idx, name in enumerate(self.robot.data.joint_names):
            print(f"  [{idx:2d}] {name}")
        print("[DEBUG] ===========================\n")
        self._missing_swerve_joints = [
            joint_name
            for module in self.swerve_modules
            for joint_name in (module.steering_joint, module.wheel_joint)
            if joint_name not in self._joint_name_to_index
        ]

        self._swerve_steering_joint_ids = [
            self._joint_name_to_index[module.steering_joint]
            for module in self.swerve_modules
            if module.steering_joint in self._joint_name_to_index
        ]
        self._swerve_wheel_joint_ids = [
            self._joint_name_to_index[module.wheel_joint]
            for module in self.swerve_modules
            if module.wheel_joint in self._joint_name_to_index
        ]
        self.readers = []
        self.threads = []
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name=joint_states_topic,
            topic_type=JointState_,
        )
        self.odom_writer = topic_manager.topic_writer(
            topic_name=odom_topic,
            topic_type=Odometry_,
        )
        self.tf_writer = topic_manager.topic_writer(
            topic_name=tf_topic,
            topic_type=TFMessage_,
        )

        for label, topic_name in topic_names.items():
            if not topic_name:
                continue
            reader = topic_manager.topic_reader(topic_name=topic_name, topic_type=JointTrajectory_, qos=trajectory_qos)
            thread = threading.Thread(
                target=self._trajectory_loop,
                args=(label, reader),
                daemon=True,
            )
            self.readers.append(reader)
            self.threads.append(thread)
            thread.start()
            print(f"[DDS] Subscribing {label}: {topic_name}")

        if cmd_vel_topic:
            cmd_vel_reader = topic_manager.topic_reader(
                topic_name=cmd_vel_topic,
                topic_type=Twist_,
                qos=trajectory_qos,
            )
            cmd_vel_thread = threading.Thread(target=self._cmd_vel_loop, args=(cmd_vel_reader,), daemon=True)
            self.readers.append(cmd_vel_reader)
            self.threads.append(cmd_vel_thread)
            cmd_vel_thread.start()
            print(f"[DDS] Subscribing cmd_vel: {cmd_vel_topic}")

    # Run DDS reader loops
    def _trajectory_loop(self, label: str, reader):
        try:
            while self.running:
                for msg in reader.take_iter():
                    self._store_trajectory(label, msg)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] {label} subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    def _cmd_vel_loop(self, reader):
        try:
            while self.running:
                for msg in reader.take_iter():
                    self._store_cmd_vel(msg)
                time.sleep(0.001)
        except Exception as exc:
            print(f"[DDS] cmd_vel subscriber exception: {exc}")
        finally:
            try:
                reader.Close()
            except Exception:
                pass

    # Parse trajectory topics and match joints
    def _store_trajectory(self, label: str, msg):
        if msg is None or not msg.points:
            return

        point = msg.points[-1]
        joint_names = list(msg.joint_names)
        positions = list(point.positions)

        if label == "lift":
            lift_position = None
            if cfg.LIFT_JOINT_NAME in joint_names:
                lift_position = (
                    cfg.LIFT_POSITION_SCALE
                    * positions[joint_names.index(cfg.LIFT_JOINT_NAME)]
                )
            elif len(positions) == 1:
                lift_position = cfg.LIFT_POSITION_SCALE * positions[0]
            if lift_position is None:
                print(
                    f"[DDS] Ignoring lift message: '{cfg.LIFT_JOINT_NAME}' "
                    f"not found in joint_names={joint_names}"
                )
                return
            joint_names = [cfg.LIFT_JOINT_NAME]
            positions = [lift_position]

        if len(joint_names) != len(positions):
            print(
                f"[DDS] Ignoring {label} message: joint_names={len(joint_names)} "
                f"positions={len(positions)}"
            )
            return

        with self.lock:
            self.pending_positions.update(dict(zip(joint_names, positions)))

    def clear_pending_targets(self):
        with self.lock:
            self.pending_positions.clear()
            self.latest_cmd_vel = (0.0, 0.0, 0.0)

    # Apply swerve drive mobile base command
    def _store_cmd_vel(self, msg):
        if msg is None:
            return
        with self.lock:
            self.latest_cmd_vel = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self.last_cmd_vel_time = time.monotonic()

    def _current_cmd_vel(self) -> tuple[float, float, float]:
        with self.lock:
            command = self.latest_cmd_vel
            last_msg_time = self.last_cmd_vel_time

        if last_msg_time == 0.0:
            return 0.0, 0.0, 0.0
        if self.cmd_vel_timeout > 0.0 and time.monotonic() - last_msg_time > self.cmd_vel_timeout:
            return 0.0, 0.0, 0.0
        return command

    def apply_latest_targets(self):
        with self.lock:
            commands = dict(self.pending_positions)

        position_target = self.robot.data.joint_pos_target.clone()
        velocity_target = self.robot.data.joint_vel_target.clone()

        for name, position in commands.items():
            joint_id = self._joint_name_to_index.get(name)
            if joint_id is None:
                if name not in self.unknown_joints:
                    self.unknown_joints.add(name)
                    print(f"[DDS] Joint '{name}' is not in the SH5 USD articulation; ignoring it.")
                continue
            position_target[:, joint_id] = float(position)

        self._apply_swerve_targets(position_target, velocity_target)

        self.robot.set_joint_position_target(position_target)
        self.robot.set_joint_velocity_target(velocity_target)

    def _apply_swerve_targets(self, position_target, velocity_target):
        if not self.swerve_modules:
            return

        for joint_name in self._missing_swerve_joints:
            if joint_name not in self._warned_missing_swerve_joints:
                self._warned_missing_swerve_joints.add(joint_name)
                print(f"[DDS] Swerve joint '{joint_name}' is not in the SH5 USD articulation; ignoring cmd_vel.")
        if self._missing_swerve_joints:
            return

        current_steering = [
            float(value)
            for value in self.robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist()
        ]
        current_wheel_velocities = [
            float(value)
            for value in self.robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]
        linear_x, linear_y, angular_z = self._current_cmd_vel()
        now = time.monotonic()
        dt = now - self._last_swerve_update_time
        self._last_swerve_update_time = now

        if self.swerve_controller is None:
            return
        module_commands = self.swerve_controller.compute_commands(
            linear_x,
            linear_y,
            angular_z,
            current_steering_positions=current_steering,
            current_wheel_velocities=current_wheel_velocities,
            dt=dt,
        )
        for module_command, steering_id, wheel_id in zip(
            module_commands,
            self._swerve_steering_joint_ids,
            self._swerve_wheel_joint_ids,
        ):
            position_target[:, steering_id] = module_command.steering_position
            velocity_target[:, wheel_id] = module_command.wheel_velocity

    def update_odometry(self, dt: float):
        if self.odometry is None or not self.swerve_modules or self._missing_swerve_joints:
            return

        steering_positions = [
            float(value) + module.angle_offset
            for value, module in zip(
                self.robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist(),
                self.swerve_modules,
            )
        ]
        wheel_velocities = [
            float(value)
            for value in self.robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]
        self.odometry.update(steering_positions, wheel_velocities, dt)

    # Publish robot state and close DDS resources
    def publish_joint_states(self):
        stamp = _now_stamp()
        header = Header_(stamp=stamp, frame_id="base_link")

        joint_names = list(self.robot.data.joint_names)
        positions = self.robot.data.joint_pos.squeeze(0).detach().cpu().tolist()
        velocities = self.robot.data.joint_vel.squeeze(0).detach().cpu().tolist()
        efforts = [0.0] * len(joint_names)

        msg = JointState_(
            header=header,
            name=joint_names,
            position=positions,
            velocity=velocities,
            effort=efforts,
        )
        try:
            self.joint_state_writer.write(msg)
        except Exception as exc:
            print(f"[DDS] joint_states write error: {exc}")

    def publish_odometry(self):
        if self.odometry is None:
            return

        state = self.odometry.state()
        quat_x, quat_y, quat_z, quat_w = yaw_to_quaternion(state.yaw)
        covariance = [0.0] * 36
        for index in (0, 7, 14, 21, 28, 35):
            covariance[index] = 0.001

        stamp = _now_stamp()
        msg = Odometry_(
            header=Header_(stamp=stamp, frame_id=self.odom_frame),
            child_frame_id=self.base_frame,
            pose=PoseWithCovariance_(
                pose=Pose_(
                    position=Point_(x=state.x, y=state.y, z=0.0),
                    orientation=Quaternion_(x=quat_x, y=quat_y, z=quat_z, w=quat_w),
                ),
                covariance=covariance,
            ),
            twist=TwistWithCovariance_(
                twist=Twist_(
                    linear=Vector3_(x=state.vx, y=state.vy, z=0.0),
                    angular=Vector3_(x=0.0, y=0.0, z=state.wz),
                ),
                covariance=covariance,
            ),
        )
        try:
            self.odom_writer.write(msg)
        except Exception as exc:
            print(f"[DDS] odom write error: {exc}")

    def publish_tf(self):
        if self._base_id is None:
            if not self._warned_missing_base_frame:
                self._warned_missing_base_frame = True
                print(
                    f"[DDS] Cannot publish TF: base frame '{self.base_frame}' is not in SH5 body names. "
                    f"Available bodies: {self._body_names}"
                )
            return

        stamp = _now_stamp()
        body_pose_w = self.robot.data.body_link_state_w[0, :, :7]
        base_pose_w = body_pose_w[self._base_id]
        base_pos_w = base_pose_w[:3].unsqueeze(0)
        base_quat_w = base_pose_w[3:7].unsqueeze(0)

        transforms = []
        for body_id, child_frame in enumerate(self._body_names):
            if child_frame == self.base_frame:
                continue

            child_pose_w = body_pose_w[body_id]
            child_pos_b, child_quat_b = math_utils.subtract_frame_transforms(
                base_pos_w,
                base_quat_w,
                child_pose_w[:3].unsqueeze(0),
                child_pose_w[3:7].unsqueeze(0),
            )
            pos = child_pos_b.squeeze(0).detach().cpu().tolist()
            quat_wxyz = child_quat_b.squeeze(0).detach().cpu().tolist()

            transforms.append(
                TransformStamped_(
                    header=Header_(stamp=stamp, frame_id=self.base_frame),
                    child_frame_id=child_frame,
                    transform=Transform_(
                        translation=Vector3_(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])),
                        rotation=Quaternion_(
                            x=float(quat_wxyz[1]),
                            y=float(quat_wxyz[2]),
                            z=float(quat_wxyz[3]),
                            w=float(quat_wxyz[0]),
                        ),
                    ),
                )
            )

        try:
            self.tf_writer.write(TFMessage_(transforms=transforms))
        except Exception as exc:
            print(f"[DDS] tf write error: {exc}")

    def shutdown(self):
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
        for reader in self.readers:
            try:
                reader.Close()
            except Exception:
                pass
        try:
            self.joint_state_writer.Close()
        except Exception:
            pass
        try:
            self.odom_writer.Close()
        except Exception:
            pass
        try:
            self.tf_writer.Close()
        except Exception:
            pass


# ========== Robot State ==========

def _swerve_modules() -> list[SwerveModule]:
    return [
        SwerveModule(
            steering_joint=steering_joint,
            wheel_joint=wheel_joint,
            x_offset=SH5_SWERVE_MODULE_X_OFFSETS[index],
            y_offset=SH5_SWERVE_MODULE_Y_OFFSETS[index],
            angle_offset=SH5_SWERVE_MODULE_ANGLE_OFFSETS[index],
            steering_limit_lower=cfg.AI_WORKER_SWERVE_STEERING_LIMIT_LOWER,
            steering_limit_upper=cfg.AI_WORKER_SWERVE_STEERING_LIMIT_UPPER,
            wheel_speed_limit_lower=cfg.AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
            wheel_speed_limit_upper=cfg.AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
        )
        for index, (steering_joint, wheel_joint) in enumerate(
            zip(SH5_SWERVE_STEERING_JOINTS, SH5_SWERVE_WHEEL_JOINTS)
        )
    ]


def _write_default_joint_state(robot):
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)


# ========== Camera View ==========

def _find_camera_prim_by_name(stage, prim_name: str):
    for prim in stage.Traverse():
        if prim.GetName() == prim_name and prim.GetTypeName() == "Camera":
            return prim
    return None


def _ensure_camera_viewport_attrs(camera_prim):
    from pxr import Gf, Sdf

    coi_attr = camera_prim.GetProperty("omni:kit:centerOfInterest")
    if not coi_attr or not coi_attr.IsValid():
        coi_attr = camera_prim.CreateAttribute(
            "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
        )
    if coi_attr.Get() is None:
        coi_attr.Set(Gf.Vec3d(0.0, 0.0, -10.0))


def _position_window(window, width: int, height: int, x: int | None = None, y: int | None = None):
    for attr_name, value in (("width", width), ("height", height), ("position_x", x), ("position_y", y)):
        if value is None:
            continue
        try:
            setattr(window, attr_name, value)
        except Exception:
            pass
        try:
            frame = getattr(window, "frame", None)
            if frame is not None:
                setattr(frame, attr_name, value)
        except Exception:
            pass


def _set_viewport_camera(
    window_name: str,
    camera_path: str,
    width: int = 640,
    height: int = 480,
    x: int | None = None,
    y: int | None = None,
):
    try:
        from omni.kit.viewport.utility import create_viewport_window, get_viewport_from_window_name
        from pxr import Sdf

        viewport = get_viewport_from_window_name(window_name)
        if viewport is None:
            window = create_viewport_window(
                window_name,
                width=width,
                height=height,
                position_x=0 if x is None else x,
                position_y=0 if y is None else y,
                camera_path=Sdf.Path(camera_path),
            )
            cfg.AI_WORKER_CAMERA_VIEW_WINDOWS.append(window)
            _position_window(window, width, height, x, y)
            viewport = get_viewport_from_window_name(window_name)
        if viewport is not None:
            viewport.set_active_camera(camera_path)
            return True
    except Exception as exc:
        print(f"[WARN] Could not create viewport '{window_name}': {exc}")
    return False


