import os

file_path = '/home/rokey/dev_ws/coupang_ws/scripts/coupang_sh5_bringup_v2.py'
bridge_path = '/home/rokey/dev_ws/coupang_ws/scripts/sh5_dds_bridge.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

# find SH5DdsBridge
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith('class SH5DdsBridge:'):
        start_idx = i
    if line.startswith('def _setup_camera_views'):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    bridge_lines = lines[start_idx:end_idx]
    
    # write to sh5_dds_bridge.py
    with open(bridge_path, 'w') as f:
        f.write("import threading\n")
        f.write("import time\n")
        f.write("import isaaclab.utils.math as math_utils\n")
        f.write("from cyclonedds.core import Qos, Policy\n")
        f.write("from robotis_dds_python.idl.builtin_interfaces.msg import Time_\n")
        f.write("from robotis_dds_python.idl.geometry_msgs.msg import Point_, Pose_, Quaternion_, Transform_, TransformStamped_, Twist_, Vector3_\n")
        f.write("from robotis_dds_python.idl.nav_msgs.msg import Odometry_\n")
        f.write("from robotis_dds_python.idl.sensor_msgs.msg import JointState_\n")
        f.write("from robotis_dds_python.idl.std_msgs.msg import Header_\n")
        f.write("from robotis_dds_python.idl.tf2_msgs.msg import TFMessage_\n")
        f.write("from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_\n")
        f.write("from robotis_dds_python.tools.topic_manager import TopicManager\n")
        f.write("from robotis_lab.controllers.swerve import SwerveDriveController, SwerveModule, SwerveOdometry\n\n")
        f.writelines(bridge_lines)

    # remove from original and add import
    new_lines = lines[:start_idx] + ["from sh5_dds_bridge import SH5DdsBridge\n\n"] + lines[end_idx:]
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print("Bridge extracted successfully.")
else:
    print("Could not find bounds.")
