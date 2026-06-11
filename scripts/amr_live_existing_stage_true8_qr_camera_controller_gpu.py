"""
Isaac Sim Script Editor controller.

Purpose
- Use an already-open Isaac Sim/Nucleus Live stage.
- Control existing prims:
  /World/AMR_01~05
  /World/RACK_01~10
- Read Control Tower bridge commands from:
  /home/rokey/isaaclab_ws/isaac_aruco/amr/bridge_queue/commands
- Execute previous-style AMR logic:
  8-way Time A* + reservation table + global move arbiter + lookahead2
  Rack-zone 4-way-only traversal + dynamic visual-front movement-direction yaw alignment
- QR localization stage added:
  read downward camera images with OpenCV QRCodeDetector, map QR ID to grid cell, then feed the existing planner.
- Write bridge results/status back for fleet_manager_bridge_node.py.

Run inside Isaac Sim:
  Window -> Script Editor
  exec(open('/home/rokey/isaaclab_ws/isaac_aruco/amr/amr_live_existing_stage_true8_qr_camera_controller_gpu.py', encoding='utf-8').read())
"""

import json
import math
import os
import re
import random
import shutil
import socket
import time
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except Exception as _cv_err:
    cv2 = None
    np = None
    CV2_AVAILABLE = False
    _CV2_IMPORT_ERROR = _cv_err


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, None)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


# GPU usage policy. The navigation/planning algorithm is intentionally unchanged.
# Only OpenCV image pre-processing is moved to CUDA when the installed cv2 build supports it.
AMR_GPU_ENABLED = env_bool("AMR_GPU_ENABLED", True)
AMR_QR_GPU_PREPROCESS_ENABLED = env_bool("AMR_QR_GPU_PREPROCESS_ENABLED", True)
AMR_QR_CUDA_DEVICE_ID = env_int("AMR_QR_CUDA_DEVICE_ID", 0)
AMR_OPENCV_CPU_THREADS = env_int("AMR_OPENCV_CPU_THREADS", 1)

_OPENCV_RUNTIME_CONFIGURED = False
_OPENCV_CUDA_CHECKED = False
_OPENCV_CUDA_AVAILABLE = False
_OPENCV_CUDA_STATUS = "not_checked"


def configure_opencv_runtime():
    global _OPENCV_RUNTIME_CONFIGURED
    if _OPENCV_RUNTIME_CONFIGURED or not CV2_AVAILABLE:
        return
    _OPENCV_RUNTIME_CONFIGURED = True
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    try:
        if AMR_OPENCV_CPU_THREADS >= 0:
            cv2.setNumThreads(AMR_OPENCV_CPU_THREADS)
    except Exception:
        pass


def opencv_cuda_available() -> bool:
    global _OPENCV_CUDA_CHECKED, _OPENCV_CUDA_AVAILABLE, _OPENCV_CUDA_STATUS
    if _OPENCV_CUDA_CHECKED:
        return _OPENCV_CUDA_AVAILABLE
    _OPENCV_CUDA_CHECKED = True
    _OPENCV_CUDA_AVAILABLE = False

    if not CV2_AVAILABLE:
        _OPENCV_CUDA_STATUS = "cv2_unavailable"
        return False
    if not AMR_GPU_ENABLED or not AMR_QR_GPU_PREPROCESS_ENABLED:
        _OPENCV_CUDA_STATUS = "disabled_by_env"
        return False
    if not hasattr(cv2, "cuda"):
        _OPENCV_CUDA_STATUS = "cv2_cuda_module_missing"
        return False

    try:
        count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as e:
        _OPENCV_CUDA_STATUS = f"cuda_check_failed:{e}"
        return False

    if count <= 0:
        _OPENCV_CUDA_STATUS = "no_cuda_device_or_opencv_built_without_cuda"
        return False

    try:
        cv2.cuda.setDevice(max(0, min(AMR_QR_CUDA_DEVICE_ID, count - 1)))
    except Exception as e:
        _OPENCV_CUDA_STATUS = f"cuda_set_device_failed:{e}"
        return False

    _OPENCV_CUDA_AVAILABLE = True
    _OPENCV_CUDA_STATUS = f"available:{count}_device(s)"
    return True


try:
    import omni.replicator.core as rep
    REPLICATOR_AVAILABLE = True
except Exception as _rep_err:
    rep = None
    REPLICATOR_AVAILABLE = False
    _REPLICATOR_IMPORT_ERROR = _rep_err

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics


# ============================================================
# Stage mapping: current user stage
# ============================================================
WORLD_ROOT = "/World"

AMR_PRIM_PATHS = {
    "AMR_01": "/World/AMR_01",
    "AMR_02": "/World/AMR_02",
    "AMR_03": "/World/AMR_03",
    "AMR_04": "/World/AMR_04",
    "AMR_05": "/World/AMR_05",
}

RACK_PRIM_PATHS = {
    "WS_01": "/World/RACK_01",
    "WS_02": "/World/RACK_02",
    "WS_03": "/World/RACK_03",
    "WS_04": "/World/RACK_04",
    "WS_05": "/World/RACK_05",
    "WS_06": "/World/RACK_06",
    "WS_07": "/World/RACK_07",
    "WS_08": "/World/RACK_08",
    "WS_09": "/World/RACK_09",
    "WS_10": "/World/RACK_10",
    "RACK_01": "/World/RACK_01",
    "RACK_02": "/World/RACK_02",
    "RACK_03": "/World/RACK_03",
    "RACK_04": "/World/RACK_04",
    "RACK_05": "/World/RACK_05",
    "RACK_06": "/World/RACK_06",
    "RACK_07": "/World/RACK_07",
    "RACK_08": "/World/RACK_08",
    "RACK_09": "/World/RACK_09",
    "RACK_10": "/World/RACK_10",
}

# If target_x/target_y from Control Tower is 0/empty, target_location falls back to this map.
# Adjust these values to match your factory layout if needed.
LOCATION_TARGETS = {
    "sg2_in_01_A": (-6.0, 4.0, 0.0),
    "sg2_in_01_B": (-6.0, 2.5, 0.0),
    "sg2_in_02_A": (-3.0, 4.0, 0.0),
    "sg2_in_02_B": (-3.0, 2.5, 0.0),
    "sg2_in_03_A": (0.0, 4.0, 0.0),
    "sg2_in_03_B": (0.0, 2.5, 0.0),
    "sg2_out_00_A": (6.0, 4.0, 0.0),
    "sg2_out_00_B": (6.0, 2.5, 0.0),
    "staging": (8.0, -2.0, 0.0),
    "warehouse": (0.0, -5.0, 0.0),
    # v16 current QR-layout aliases used by bridge command managers.
    # SG targets are known-valid QR cells.  Stage targets are kept distinct from SG A cells;
    # if a stage cell is not present in the QR map the command manager should prefer SG/B cells.
    "stage_drop_01": (7.5, 3.0, 0.0),
    "stage_drop_02": (6.0, 0.0, 0.0),
    "stage_drop_03": (7.5, -6.0, 0.0),
    "stage_drop_04": (6.0, -9.0, 0.0),
}


# ============================================================
# Bridge queue paths
# ============================================================
BRIDGE_QUEUE_DIR = Path("/home/rokey/isaaclab_ws/isaac_aruco/amr/bridge_queue")
COMMAND_DIR = BRIDGE_QUEUE_DIR / "commands"
STATUS_DIR = BRIDGE_QUEUE_DIR / "status"
RESULT_DIR = BRIDGE_QUEUE_DIR / "results"
DONE_DIR = BRIDGE_QUEUE_DIR / "done"
CANCEL_DIR = BRIDGE_QUEUE_DIR / "cancel"


# ============================================================
# Redis realtime AMR status publishing
# ============================================================
# Control Tower / DB / Redis PC. Update if the other PC uses a different IP.
REDIS_STATUS_ENABLED = True
REDIS_HOST = "192.168.100.20"
REDIS_PORT = 6379
REDIS_KEY_PREFIX = "amr:"
REDIS_PUBLISH_PERIOD_SEC = float(os.environ.get("AMR_REDIS_PUBLISH_PERIOD_SEC", "0.20"))  # optimized default: 5 Hz
REDIS_CONNECT_TIMEOUT_SEC = 0.08
REDIS_RECONNECT_PERIOD_SEC = 3.0
REDIS_BATTERY_DEFAULT = "85.0"


# ============================================================
# Motion / planner parameters
# ============================================================
GRID_SPACING = 1.5
AMR_SIZE_M = 0.7
RACK_SIZE_M = 1.3
SAFETY_MARGIN_M = 0.12

AMR_SPEED_MPS = 1.35
AMR_CARRY_SPEED_MPS = 0.90
LIFT_HEIGHT_M = 0.10
LIFT_DURATION_SEC = 0.45
PLACE_DURATION_SEC = 0.45
ROTATE_DURATION_SEC = 1.20
ROTATE_WORKSTATION_TARGETS = {"ROTATE_WORKSTATION", "ROTATE", "ROTATE_180", "WORKSTATION_ROTATE", "TURN_WORKSTATION"}
ROTATE_LOCATION_KEYWORDS = {"ROTATING", "ROTATE_WORKSTATION", "ROTATE_180"}
ROTATE_WORKSTATION_DELTA_RAD = math.pi

MAX_TIME_HORIZON = 80
RESERVATION_HORIZON = 35
GOAL_HOLD_STEPS = 4
LOOKAHEAD_STEPS = 2
DECISION_PERIOD_SEC = 0.08
COMMAND_SCAN_PERIOD_SEC = 0.20
STATUS_PERIOD_SEC = 0.25
LOG_PERIOD_SEC = 5.0

# ============================================================
# QR camera localization parameters
# ============================================================
# This does not change the 8-way planner. It only changes how AMR current_cell is observed.
QR_CAMERA_LOCALIZATION_ENABLED = True
QR_CAMERA_SCAN_PERIOD_SEC = float(os.environ.get("AMR_QR_CAMERA_SCAN_PERIOD_SEC", "0.45"))
QR_CAMERA_RESOLUTION = (
    int(os.environ.get("AMR_QR_CAMERA_WIDTH", "320")),
    int(os.environ.get("AMR_QR_CAMERA_HEIGHT", "240")),
)
QR_DETECTOR_EVERY_N_SCANS = int(os.environ.get("AMR_QR_DETECTOR_EVERY_N_SCANS", "1"))
QR_LOG_PERIOD_SEC = 5.0
QR_SCAN_ROUND_ROBIN_ENABLED = True
QR_SCAN_MAX_AMRS_PER_CYCLE = 1

# Safe QR localization gate.
# QR navigation is kept enabled, but detected QR cells are accepted only when they
# are physically plausible. This prevents two-cell jumps caused by reading a
# forward/neighbor QR while the scripted grid motion is still between cells.
QR_SAFE_STABLE_DETECTIONS = int(os.environ.get("AMR_QR_SAFE_STABLE_DETECTIONS", "2"))
QR_SAFE_MAX_CELL_JUMP = int(os.environ.get("AMR_QR_SAFE_MAX_CELL_JUMP", "1"))
QR_SAFE_MAX_WORLD_DIST_M = float(os.environ.get("AMR_QR_SAFE_MAX_WORLD_DIST_M", "1.0"))
QR_SAFE_BLOCK_STATES = {"LIFTING", "PLACING", "ROTATING"}

# If True, the demo keeps moving even when the QR is temporarily missed.
# The log will clearly say source=TRANSFORM_FALLBACK.
# Keep this False for QR-based tests: fallback may hide QR misreads.
QR_ALLOW_TRANSFORM_FALLBACK = False

# Navigation must use QR-coded cells only.
# If the QR map is loaded, A* will never plan through cells that do not have a QR marker.
QR_ONLY_NAVIGATION = True
QR_SNAP_TARGET_TO_NEAREST = True
QR_MAX_SNAP_DISTANCE_M = 2.20

# If a Control Tower goal resolves to the same QR cell as the rack pickup cell,
# the AMR would lift and immediately place. Prevent that by forcing the drop
# target to a different QR cell at least this many cells away from pickup.
COMMAND_MIN_DROP_DISTANCE_CELLS = 3

# Rack is moved as a kinematic object while carried to prevent physics sliding.
RACK_LOCK_TO_AMR_WHILE_CARRIED = True
RACK_DISABLE_RIGID_BODY_WHILE_CARRIED = True
# For this live demo, racks are script-controlled objects.
# We keep rack physics/collision disabled and use the software planner for obstacles.
RACK_DISABLE_PHYSICS_AT_STARTUP = True
RACK_KEEP_COLLISION_DISABLED_AFTER_PLACE = True

# When an AMR passes through or very near a rack/workstation, diagonal movement is forbidden.
# This represents the 4-leg workstation constraint: entering/exiting under the workstation must be done
# with cardinal motion only, not diagonal crossing between legs.
RACK_FOUR_WAY_ZONE_ENABLED = True
RACK_FOUR_WAY_ZONE_RADIUS_CELLS = 1

# AMR heading convention. The controller assumes the visual front of the AMR points along +X when yaw=0.
# If your imported AMR model visually faces another direction, change only this value.
AMR_FRONT_YAW_OFFSET_DEG = 0.0
AMR_FRONT_YAW_OFFSET_RAD = math.radians(AMR_FRONT_YAW_OFFSET_DEG)

# v14 user-requested visual heading correction.
# No additional visual heading offset. Robot front is aligned directly with movement direction.
AMR_VISUAL_HEADING_CLOCKWISE_OFFSET_DEG = 0.0
AMR_VISUAL_HEADING_CLOCKWISE_OFFSET_RAD = math.radians(AMR_VISUAL_HEADING_CLOCKWISE_OFFSET_DEG)

# v11 stronger swept-path clearance. This prevents a diagonal AMR and a cardinal AMR
# from visually clipping during the same tick even if their destination cells differ.
DYNAMIC_SWEPT_COLLISION_MARGIN_M = 0.22
DIAGONAL_CARDINAL_CONFLICT_ENABLED = True

# v12 anti-clump / lookahead controls.
# LOOKAHEAD_STEPS was already configured as 2; v12 makes the second cell actively participate
# in global arbitration instead of only being kept in the planned path.
LOOKAHEAD2_ACTIVE_ARBITEER = True
LOOKAHEAD2_MIN_CLEARANCE_CELLS = 1

# v18 path-shape tuning.  The previous carrying turn penalty was too high, so
# the planner sometimes preferred a long detour over a simple 90-degree turn.
TURN_COST_EMPTY_90 = 0.03
TURN_COST_EMPTY_180 = 0.55
TURN_COST_CARRY_90 = 0.08
TURN_COST_CARRY_180 = 0.85

# v22 path-shape stability.  A waiting/following AMR should not suddenly back up
# when a short right/left turn or one-tick wait would be better.  U-turns and
# moves that increase the distance to the task goal are still allowed, but they
# are deliberately expensive so they are used only when no better route exists.
WAIT_COST_EMPTY = 0.75
WAIT_COST_CARRY = 0.90
REVERSE_MOVE_EXTRA_COST_EMPTY = 0.45
REVERSE_MOVE_EXTRA_COST_CARRY = 0.70
GOAL_REGRESSION_COST_EMPTY = 0.25
GOAL_REGRESSION_COST_CARRY = 0.35
KEEP_HEADING_SMALL_BONUS = 0.02

# v18/v19 convoy tuning. Same-direction following may use the cell vacated by the
# leading AMR in the same tick; otherwise straight-line traffic keeps an
# unnecessary 2-cell gap.  v19 generalizes this into tail-release following:
# a trailing AMR may enter a cell only when the AMR currently occupying that cell
# is also approved to leave it in the same tick.  This keeps current AMR cells
# as hard blocks unless they are explicitly vacated by the approved leader move.
ALLOW_SAME_DIRECTION_CONVOY_FOLLOWING = True
ALLOW_VACATED_CELL_TAIL_RELEASE = True
ALLOW_LOADED_STRAIGHT_TIGHT_CONVOY = True
ALLOW_LOADED_TURN_TAIL_RELEASE = False
# v25 safety rule:
# - Straight same-direction convoy may use tail-release.
# - Perpendicular/L-shaped tail-release must be conservative when any AMR is carrying
#   a rack/workstation. A loaded AMR has a wider swept footprint; the follower must
#   wait until the loaded leader has actually completed the move.
ALLOW_EMPTY_TURN_TAIL_RELEASE = True
ALLOW_LOADED_PERPENDICULAR_TAIL_RELEASE = False
# v25 target lock:
# Keep explicit SG2 A/B destinations from being rewritten by the defensive
# "away from pickup" fallback, unless the requested target is exactly the pickup cell.
LOCK_EXPLICIT_SG2_AB_TARGETS = True

# v26 5-AMR congestion recovery.
# With 5 simultaneous tasks, a few AMRs can remain motionless because carrying
# AMRs keep winning arbitration and waiting AMRs keep selecting one-tick waits.
# This keeps all previous safety rules, but adds aging: the longer an AMR waits,
# the higher its arbitration priority becomes and the more expensive an additional
# wait action becomes.
CONGESTION_AGING_ENABLED = True
CONGESTION_WAIT_PRIORITY_CAP = 120
CONGESTION_TO_RACK_EXTRA_BOOST_AFTER = 35
CONGESTION_TO_RACK_EXTRA_BOOST = 70
CONGESTION_WAIT_COST_STEP_EMPTY = 0.030
CONGESTION_WAIT_COST_STEP_CARRY = 0.020
CONGESTION_WAIT_COST_CAP_EMPTY = 2.50
CONGESTION_WAIT_COST_CAP_CARRY = 2.00

# v10 dense loaded-lane spread.
# When three or more rack-carrying AMRs form a tight vertical queue, a plain
# shortest-path policy can keep the middle AMR going straight one more cell and
# continue blocking the two side AMRs.  This local traffic rule only rewrites the
# first proposal in that narrow jam pattern: the center blocker yields left/out,
# the lower AMR goes straight toward SG2, and the upper AMR turns toward its goal.
# The global arbiter still performs every normal collision, footprint, tail-release
# and QR safety check after this rewrite.
LOADED_LANE_SPREAD_ENABLED = True
LOADED_LANE_SPREAD_MIN_GROUP_SIZE = 3
LOADED_LANE_SPREAD_WAIT_THRESHOLD = 8
LOADED_LANE_SPREAD_LOG_PERIOD = 20

# v11 bottleneck turn-first policy.
# In dense SG2 traffic, continuing straight one more cell can keep the bottleneck
# closed. Once an AMR has waited for several ticks, prefer a 90-degree turn/yield
# over another straight step. This is a preference, not a hard command: every
# rewritten proposal still passes the normal QR, footprint, tail-release and
# swept-collision checks in the global arbiter.
BOTTLENECK_TURN_FIRST_ENABLED = True
BOTTLENECK_TURN_FIRST_WAIT_THRESHOLD = 8
BOTTLENECK_TURN_BONUS_EMPTY = 0.18
BOTTLENECK_TURN_BONUS_CARRY = 0.42
BOTTLENECK_STRAIGHT_EXTRA_EMPTY = 0.10
BOTTLENECK_STRAIGHT_EXTRA_CARRY = 0.28
BOTTLENECK_TURN_PRIORITY_BONUS = 90.0
BOTTLENECK_TURN_FIRST_LOG_PERIOD = 20

# v12 preemptive turn-first policy.
# v11 waited until wait/no_path accumulated, so the AMRs could already be packed into
# a one-column loaded queue before the turn-first rule started.  v12 predicts that
# bottleneck earlier from local loaded-neighbor density and same-column loaded queues,
# then prefers a 90-degree yield before the straight step closes the corridor.
PREEMPTIVE_TURN_FIRST_ENABLED = True
PREEMPTIVE_TURN_ONLY_CARRYING = True
PREEMPTIVE_TURN_LOCAL_DENSITY_THRESHOLD = 3
PREEMPTIVE_TURN_LOADED_NEIGHBOR_THRESHOLD = 2
PREEMPTIVE_TURN_SAME_COLUMN_GROUP_SIZE = 3
PREEMPTIVE_TURN_MAX_GOAL_REGRESSION = 2
PREEMPTIVE_TURN_PRIORITY_BONUS = 60.0
PREEMPTIVE_TURN_LOG_PERIOD = 20

# v14 final-drop-lock return-home policy.
# Keep no-path escape for the AMR carrying its current workstation, but do NOT
# create internal lane-clearance jobs that re-pick a workstation after it has
# already been placed at its requested destination.  This matches the test goal:
# workstations stay where they are dropped; only the AMR returns home.
NO_PATH_ESCAPE_WAYPOINT_ENABLED = True
NO_PATH_ESCAPE_THRESHOLD = 35
NO_PATH_ESCAPE_RADIUS = 3
NO_PATH_ESCAPE_LOG_PERIOD = 20
DYNAMIC_SG2_LANE_CLEARANCE_ENABLED = False
RETURN_HOME_CLEARANCE_PREEMPT_ENABLED = False


# v15 traffic-level starvation/spacing diagnostics.
# The log analysis showed AMR_02 often had no_path=0 but wait kept increasing,
# which means A* found a route but the global arbiter rejected the one-step move.
# These rules keep all v14 final-drop-lock behavior, but add:
# 1) wait/no_path==0 starvation unlock priority,
# 2) loaded AMR spacing guard to avoid creating tight side-by-side rack clusters,
# 3) explicit reject-reason logs from the arbiter.
STARVATION_UNLOCK_ENABLED = True
# v16: lower the trigger a little because the user test now starts AMRs one-by-one.
# If wait increases while no_path remains 0, the robot has a valid route but is losing arbitration.
STARVATION_UNLOCK_WAIT_THRESHOLD = 18
STARVATION_UNLOCK_PRIORITY_BONUS = 500.0
STARVATION_UNLOCK_LOG_PERIOD = 20

# v16: command JSON may request a specific AMR.  This is used by the 1-3-5-7-9
# stagger manager so command 1 uses AMR_01, command 2 uses AMR_02, and so on.
PREFERRED_AMR_ASSIGNMENT_ENABLED = True

LOADED_AMR_MIN_SPACING_ENABLED = True
LOADED_AMR_MIN_CHEBYSHEV_DISTANCE = 2
LOADED_AMR_SPACING_ALLOW_STRAIGHT_CONVOY = True
LOADED_AMR_SPACING_LOG_PERIOD = 20

ARB_REJECT_REASON_LOG_ENABLED = True
ARB_REJECT_REASON_LOG_PERIOD = 20

ANTI_CLUMP_ENABLED = True
ANTI_CLUMP_RADIUS_CELLS = 2
ANTI_CLUMP_MIN_GROUP_SIZE = 3
ANTI_CLUMP_GOAL_MIN_SEPARATION_CELLS = 4
REJECTED_MOVE_GOAL_REASSIGN_STEPS = 2

# v16 path policy patch: Mode C two-step approval.
# - Carrying AMR: 4-way A* + center/up/down/left/right rack footprint check.
# - Empty AMR: true 8-way movement, no stationary-rack footprint obstacle, can pass under racks.
# - Move approval treats the second planned cell as soft lookahead only; it never blocks a safe one-cell move.
# - Goal-cell entry never requires lookahead cells beyond the goal.
PATH_POLICY_MODE_C_2STEP_APPROVAL = True

# v14 unique target control.
# Prevent two AMRs from being assigned the same final QR cell.
UNIQUE_RANDOM_GOALS_ENABLED = True
UNIQUE_TASK_TARGET_RESERVATION_ENABLED = True
DUPLICATE_TARGET_RESOLVE_ENABLED = True

# v9 heading calibration:
# The AMR visual front is NOT assumed to be root +X.
# It is measured from the line: AMR visual bbox center -> selected front reference bbox center.
# During every move, root yaw is adjusted by the delta between current visual-front yaw and desired movement yaw.
FRONT_REFERENCE_KEYWORDS = [
    "frontnameplate",
    "front_nameplate",
    "front-nameplate",
    "frontplate",
    "front_plate",
    "frontsignature",
    "signaturelight",
    "topthinbluelinefront",
    "frontamber",
    "frontblue",
    "nameplate",
]

# Camera prim search keywords under each AMR. Keep these broad because the USD naming may vary.
QR_CAMERA_NAME_KEYWORDS = ["downward", "qr", "camera"]

# Keep the same QR detection algorithm. This only changes where RGB->BGR conversion runs.
QR_PREPROCESS_BACKEND = "CUDA_RGB2BGR_IF_AVAILABLE"

RANDOM_DRIVE_ENABLED = False
RANDOM_GOAL_MIN_RADIUS_CELL = 4
RANDOM_GOAL_RANGE_CELL = 8

# v15 integration mode: wait until Control Tower command arrives.
# Commands left in bridge_queue before this controller starts are ignored to prevent old tests
# from moving racks by themselves.
STANDBY_UNTIL_COMMAND = True
IGNORE_STALE_COMMANDS_ON_STARTUP = True
STALE_COMMAND_GRACE_SEC = 2.0

random.seed(42)

GridCell = Tuple[int, int]
TimedCell = Tuple[int, int, int]
EdgeKey = Tuple[int, int, int, int, int]
Move = Tuple[int, int]

# Filled by ExistingStageTrue8Controller.build_qr_prim_map().
GLOBAL_VALID_QR_CELLS: Set[GridCell] = set()
GLOBAL_QR_CELL_WORLD_MAP: Dict[GridCell, Tuple[float, float]] = {}
GLOBAL_RACK_FOUR_WAY_CELLS: Set[GridCell] = set()


# ============================================================
# File helpers
# ============================================================
def ensure_dirs():
    for d in [COMMAND_DIR, STATUS_DIR, RESULT_DIR, DONE_DIR, CANCEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def safe_read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_write_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{int(time.time() * 1000)}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ============================================================
# USD transform helpers
# ============================================================
def get_stage():
    return omni.usd.get_context().get_stage()


def get_world_translation(prim) -> Gf.Vec3d:
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    tr = mat.ExtractTranslation()
    return Gf.Vec3d(float(tr[0]), float(tr[1]), float(tr[2]))


def get_bbox_center_world(prim) -> Gf.Vec3d:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn = box.GetMin()
    mx = box.GetMax()
    return Gf.Vec3d(
        float(mn[0] + mx[0]) * 0.5,
        float(mn[1] + mx[1]) * 0.5,
        float(mn[2] + mx[2]) * 0.5,
    )


def normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def get_root_yaw_rad(prim) -> float:
    try:
        api = UsdGeom.XformCommonAPI(prim)
        _translate, rotate, _scale, _pivot, _rot_order = api.GetXformVectors(Usd.TimeCode.Default())
        return math.radians(float(rotate[2]))
    except Exception:
        pass

    try:
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                value = op.Get(Usd.TimeCode.Default())
                return math.radians(float(value[2]))
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                value = op.Get(Usd.TimeCode.Default())
                return math.radians(float(value))
    except Exception:
        pass

    return 0.0


def find_front_reference_prim(stage, amr_root_path: str):
    root = stage.GetPrimAtPath(amr_root_path)
    if not root.IsValid():
        return None

    candidates = []
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not prim.IsActive():
            continue

        path = str(prim.GetPath())
        name = prim.GetName()
        compact_path = path.lower().replace("_", "").replace("-", "").replace(" ", "")
        compact_name = name.lower().replace("_", "").replace("-", "").replace(" ", "")
        type_name = prim.GetTypeName()

        score = 0

        # Strongly prefer actual front visual markers. In this AMR model,
        # FrontSignatureLight / TopThinBlueLineFront are more reliable than wheel front caps.
        preferred_terms = [
            "frontnameplate",
            "frontsignaturelight",
            "frontsignature",
            "signaturelight",
            "topthinbluelinefront",
            "frontplate",
            "frontblueedge",
            "frontamberglow",
            "frontamber",
            "frontblue",
            "nameplate",
        ]
        for idx, term in enumerate(preferred_terms):
            if term in compact_path or term in compact_name:
                score += 200 - idx * 5

        if "front" in compact_path or "front" in compact_name:
            score += 30
        if "nameplate" in compact_path or "nameplate" in compact_name:
            score += 40

        # Do not use wheel/pod/cap/housing meshes as the front reference.
        bad_terms = ["wheel", "pod", "housing", "cap", "rear", "back", "left", "right", "fl", "fr"]
        for term in bad_terms:
            if term in compact_path or term in compact_name:
                score -= 120

        if type_name in ("Mesh", "Cube", "Xform"):
            score += 2

        if score > 0:
            candidates.append((score, path, prim))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][2]

def calibrate_front_yaw_offset(stage, amr_prim, amr_name: str) -> Tuple[float, str]:
    front_prim = find_front_reference_prim(stage, str(amr_prim.GetPath()))
    if front_prim is None:
        print(
            f"AMR FRONT CALIBRATION FALLBACK | {amr_name} "
            f"front reference not found, offset_deg={AMR_FRONT_YAW_OFFSET_DEG:.3f}"
        )
        return AMR_FRONT_YAW_OFFSET_RAD, ""

    amr_center = get_bbox_center_world(amr_prim)
    front_center = get_bbox_center_world(front_prim)
    vx = float(front_center[0] - amr_center[0])
    vy = float(front_center[1] - amr_center[1])
    length = math.hypot(vx, vy)

    if length < 1e-5:
        print(
            f"AMR FRONT CALIBRATION FALLBACK | {amr_name} "
            f"front reference bbox too close path={front_prim.GetPath()}"
        )
        return AMR_FRONT_YAW_OFFSET_RAD, str(front_prim.GetPath())

    root_yaw = get_root_yaw_rad(amr_prim)
    front_world_yaw = math.atan2(vy, vx)
    front_offset = normalize_angle_rad(front_world_yaw - root_yaw)

    print(
        f"AMR FRONT CALIBRATED V11 | {amr_name} "
        f"front={front_prim.GetPath()} "
        f"root_yaw_deg={math.degrees(root_yaw):.2f} "
        f"front_bbox_yaw_deg={math.degrees(front_world_yaw):.2f} "
        f"front_offset_deg={math.degrees(front_offset):.2f}"
    )
    return front_offset, str(front_prim.GetPath())


def get_amr_visual_front_yaw_rad(amr) -> Optional[float]:
    front_path = getattr(amr, "front_reference_path", "")
    if not front_path:
        return None
    stage = get_stage()
    if stage is None:
        return None
    front_prim = stage.GetPrimAtPath(front_path)
    if not front_prim.IsValid() or not front_prim.IsActive():
        return None

    try:
        amr_center = get_bbox_center_world(amr.obj.prim)
        front_center = get_bbox_center_world(front_prim)
        vx = float(front_center[0] - amr_center[0])
        vy = float(front_center[1] - amr_center[1])
        if math.hypot(vx, vy) < 1e-5:
            return None
        return math.atan2(vy, vx)
    except Exception:
        return None

def get_xy(prim) -> Tuple[float, float]:
    p = get_world_translation(prim)
    return float(p[0]), float(p[1])


def get_z(prim) -> float:
    p = get_world_translation(prim)
    return float(p[2])


def set_translate(prim, x: float, y: float, z: float):
    # Current AMR/RACK roots are direct children of /World, so local translate works as world translate.
    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(x), float(y), float(z)))


def set_yaw_if_possible(prim, yaw_rad: float):
    xformable = UsdGeom.Xformable(prim)
    rotate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
            break
    if rotate_op is None:
        try:
            rotate_op = xformable.AddRotateXYZOp()
        except Exception:
            return
    rotate_op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(yaw_rad)))

def set_rigid_body_enabled_subtree(root_prim, enabled: bool):
    """Disable/enable rigid body simulation under a rack while keeping visual transform controllable."""
    try:
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid():
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                try:
                    api = UsdPhysics.RigidBodyAPI(prim)
                    api.CreateRigidBodyEnabledAttr(bool(enabled))
                    # While carrying, force kinematic behavior as an additional guard.
                    try:
                        api.CreateKinematicEnabledAttr(not bool(enabled))
                    except Exception:
                        pass
                except Exception:
                    pass
            # Zero velocities if the attributes exist. This reduces post-lift sliding.
            for attr_name, value in [
                ("physics:velocity", Gf.Vec3f(0.0, 0.0, 0.0)),
                ("physics:angularVelocity", Gf.Vec3f(0.0, 0.0, 0.0)),
            ]:
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    try:
                        attr.Set(value)
                    except Exception:
                        pass
    except Exception as e:
        print(f"RACK PHYSICS LOCK WARNING | {root_prim.GetPath()} enabled={enabled} err={e}")


def set_collision_enabled_subtree(root_prim, enabled: bool):
    """Temporarily disable rack collisions while it is carried to avoid physics separation/sliding."""
    try:
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid():
                continue
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                try:
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(bool(enabled))
                except Exception:
                    pass
    except Exception as e:
        print(f"RACK COLLISION LOCK WARNING | {root_prim.GetPath()} enabled={enabled} err={e}")


def hard_disable_physics_subtree(root_prim, disable_collision: bool = True):
    """Make an imported rack purely script-controlled.

    Imported racks can have RigidBodyAPI/CollisionAPI on nested mesh prims.
    If those remain active, PhysX may keep pulling the mesh downward even while
    the parent Xform is moved by script. This function disables/removes the
    physics APIs as much as the current editable layer allows.
    """
    try:
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid():
                continue

            for attr_name, value in [
                ("physics:velocity", Gf.Vec3f(0.0, 0.0, 0.0)),
                ("physics:angularVelocity", Gf.Vec3f(0.0, 0.0, 0.0)),
            ]:
                attr = prim.GetAttribute(attr_name)
                if attr and attr.IsValid():
                    try:
                        attr.Set(value)
                    except Exception:
                        pass

            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                try:
                    rb = UsdPhysics.RigidBodyAPI(prim)
                    rb.CreateRigidBodyEnabledAttr(False)
                    try:
                        rb.CreateKinematicEnabledAttr(True)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                except Exception:
                    pass

            if prim.HasAPI(UsdPhysics.MassAPI):
                try:
                    prim.RemoveAPI(UsdPhysics.MassAPI)
                except Exception:
                    pass

            if disable_collision and prim.HasAPI(UsdPhysics.CollisionAPI):
                try:
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                except Exception:
                    pass

            if disable_collision and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                try:
                    prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                except Exception:
                    pass
    except Exception as e:
        print(f"HARD PHYSICS DISABLE WARNING | {root_prim.GetPath()} err={e}")


def set_world_translate_preserve_direct_child(prim, x: float, y: float, z: float):
    set_translate(prim, x, y, z)


def compute_visual_center_offset_xy(prim) -> Tuple[float, float]:
    """Return world-space XY offset from prim origin to rendered bbox center."""
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        center = box.GetMidpoint()
        origin = get_world_translation(prim)
        return float(center[0] - origin[0]), float(center[1] - origin[1])
    except Exception:
        return 0.0, 0.0


def get_object_center_xy(obj) -> Tuple[float, float]:
    """Return the controllable root origin used as the AMR/RACK center.

    Do NOT use BBox center here. In this user stage, bbox center can be far from
    the actual AMR/RACK root because imported assets contain nested offsets.
    Using the root origin makes the AMR root center align exactly with QR centers.
    """
    return get_xy(obj.prim)


def set_object_center_xy(obj, x: float, y: float, z: float):
    """Place the AMR/RACK root center exactly on the requested QR center."""
    set_translate(obj.prim, float(x), float(y), float(z))


def qr_cell_to_world(cell: GridCell) -> Tuple[float, float]:
    mapped = GLOBAL_QR_CELL_WORLD_MAP.get(cell)
    if mapped is not None:
        return mapped
    return cell[0] * GRID_SPACING, cell[1] * GRID_SPACING


def distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cell_chebyshev_distance(a: GridCell, b: GridCell) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def cell_manhattan_distance(a: GridCell, b: GridCell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def world_to_grid(x: float, y: float) -> GridCell:
    return int(round(x / GRID_SPACING)), int(round(y / GRID_SPACING))


def grid_to_world(cell: GridCell) -> Tuple[float, float]:
    mapped = GLOBAL_QR_CELL_WORLD_MAP.get(cell)
    if mapped is not None:
        return mapped
    return cell[0] * GRID_SPACING, cell[1] * GRID_SPACING


def normalize_move(dx: int, dy: int) -> Move:
    return (0 if dx == 0 else int(math.copysign(1, dx)), 0 if dy == 0 else int(math.copysign(1, dy)))


def is_diagonal_move(move: Move) -> bool:
    return abs(move[0]) == 1 and abs(move[1]) == 1


def is_cardinal_move(move: Move) -> bool:
    return (abs(move[0]) + abs(move[1])) == 1


def yaw_from_move(move: Move) -> Optional[float]:
    if move == (0, 0):
        return None
    # Desired WORLD direction of the robot visual front.
    return math.atan2(move[1], move[0])


def set_amr_yaw_to_movement(amr) -> bool:
    desired_front_yaw = yaw_from_move(amr.heading)
    if desired_front_yaw is None:
        return False

    # v12: apply requested visual heading offset.
    # Negative clockwise value means counterclockwise offset.
    desired_front_yaw = normalize_angle_rad(desired_front_yaw - AMR_VISUAL_HEADING_CLOCKWISE_OFFSET_RAD)

    current_root_yaw = get_root_yaw_rad(amr.obj.prim)
    current_front_yaw = get_amr_visual_front_yaw_rad(amr)

    if current_front_yaw is not None:
        # Dynamic visual correction:
        # rotate the root by exactly the angular error between the currently visible
        # front direction and the desired movement direction.
        delta = normalize_angle_rad(desired_front_yaw - current_front_yaw)
        root_yaw = normalize_angle_rad(current_root_yaw + delta)
    else:
        # Fallback to static calibrated offset.
        offset = getattr(amr, "front_yaw_offset_rad", AMR_FRONT_YAW_OFFSET_RAD)
        root_yaw = normalize_angle_rad(desired_front_yaw - offset)

    set_yaw_if_possible(amr.obj.prim, root_yaw)
    return True


# ============================================================
# Redis status publisher helpers
# ============================================================
def redis_encode_command(*parts: str) -> bytes:
    chunks = [f"*{len(parts)}\r\n".encode("utf-8")]
    for part in parts:
        data = str(part).encode("utf-8")
        chunks.append(f"${len(data)}\r\n".encode("utf-8"))
        chunks.append(data)
        chunks.append(b"\r\n")
    return b"".join(chunks)


class RedisStatusClient:
    def __init__(self, host: str, port: int, timeout: float, reconnect_period: float):
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.reconnect_period = float(reconnect_period)
        self.sock = None
        self.last_connect_try = 0.0
        self.last_error = ""
        self.connected_once = False

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None

    def connect_if_needed(self) -> bool:
        if self.sock is not None:
            return True
        now = time.time()
        if now - self.last_connect_try < self.reconnect_period:
            return False
        self.last_connect_try = now
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout)
            s.settimeout(self.timeout)
            self.sock = s
            self.last_error = ""
            if not self.connected_once:
                print(f"REDIS STATUS CONNECTED | {self.host}:{self.port}")
                self.connected_once = True
            return True
        except Exception as e:
            self.last_error = str(e)
            self.close()
            return False

    def hset(self, key: str, mapping: Dict[str, str]) -> bool:
        if not mapping:
            return True
        if not self.connect_if_needed():
            return False
        parts = ["HSET", key]
        for k, v in mapping.items():
            parts.append(str(k))
            parts.append(str(v))
        try:
            self.sock.sendall(redis_encode_command(*parts))
            try:
                self.sock.recv(256)
            except socket.timeout:
                pass
            return True
        except Exception as e:
            self.last_error = str(e)
            self.close()
            return False


def cell_to_floor_qr_id_from_map(cell: Optional[GridCell], cell_world_map: Dict[GridCell, Tuple[float, float]]) -> str:
    if cell is None:
        return ""
    xy = cell_world_map.get(cell)
    if xy is None:
        # Fallback format. It is still a valid logical QR-style ID, but exact map coordinates are preferred.
        x, y = grid_to_world(cell)
    else:
        x, y = xy
    return f"FLOOR_X_{float(x):.3f}_Y_{float(y):.3f}"


# ============================================================
# QR camera localization helpers
# ============================================================
def normalize_qr_id(value: str) -> str:
    return str(value or "").strip().replace(" ", "").upper()


def find_descendant_camera(stage, root_path: str) -> Optional[str]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None

    candidates = []
    try:
        for prim in Usd.PrimRange(root):
            if not prim.IsValid() or not prim.IsActive():
                continue
            if prim.GetTypeName() != "Camera":
                continue
            name = prim.GetName().lower()
            path = str(prim.GetPath())
            score = 0
            for kw in QR_CAMERA_NAME_KEYWORDS:
                if kw in name or kw in path.lower():
                    score += 1
            # Prefer explicitly downward/qr named cameras.
            if "downward" in name or "downward" in path.lower():
                score += 5
            if "qr" in name or "qr" in path.lower():
                score += 3
            candidates.append((score, path))
    except Exception as e:
        print(f"QR camera scan failed under {root_path}: {e}")
        return None

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


class QRCameraReader:
    def __init__(self, amr_name: str, camera_path: str):
        self.amr_name = amr_name
        self.camera_path = camera_path
        self.render_product = None
        self.annotator = None
        self.detector = cv2.QRCodeDetector() if CV2_AVAILABLE else None
        self.ready = False
        self.last_qr_id = ""
        self.last_error = ""
        self.scan_count = 0
        self.cuda_preprocess_enabled = False
        self.cuda_disabled_reason = ""
        self.setup()

    def setup(self):
        configure_opencv_runtime()
        if not CV2_AVAILABLE:
            self.last_error = f"cv2 unavailable: {_CV2_IMPORT_ERROR}"
            return
        if not REPLICATOR_AVAILABLE:
            self.last_error = f"omni.replicator.core unavailable: {_REPLICATOR_IMPORT_ERROR}"
            return
        try:
            self.cuda_preprocess_enabled = opencv_cuda_available()
            self.cuda_disabled_reason = _OPENCV_CUDA_STATUS
            self.render_product = rep.create.render_product(self.camera_path, QR_CAMERA_RESOLUTION)
            self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.annotator.attach([self.render_product])
            self.ready = True
            backend = "opencv_cuda_rgb2bgr" if self.cuda_preprocess_enabled else f"opencv_cpu_rgb2bgr({_OPENCV_CUDA_STATUS})"
            print(f"QR CAMERA READY | {self.amr_name} camera={self.camera_path} res={QR_CAMERA_RESOLUTION} preprocess={backend} cv_threads={AMR_OPENCV_CPU_THREADS}")
        except Exception as e:
            self.last_error = str(e)
            print(f"QR CAMERA SETUP FAILED | {self.amr_name} camera={self.camera_path} err={e}")

    def get_rgb_array(self):
        if not self.ready or self.annotator is None:
            return None
        try:
            data = self.annotator.get_data()
            if data is None:
                return None
            if isinstance(data, dict):
                data = data.get("data", None)
            if data is None:
                return None
            arr = np.asarray(data)
            if arr.size == 0:
                return None
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            return arr
        except Exception as e:
            self.last_error = str(e)
            return None

    def rgb_to_bgr_for_qr(self, frame):
        # QRCodeDetector is still the same detector. Only RGB->BGR conversion is moved
        # to cv2.cuda when the local OpenCV build supports CUDA.
        if self.cuda_preprocess_enabled:
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_bgr = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2BGR)
                return gpu_bgr.download(), "CUDA"
            except Exception as e:
                self.cuda_preprocess_enabled = False
                self.cuda_disabled_reason = f"cuda_runtime_failed:{e}"
                self.last_error = self.cuda_disabled_reason
                print(f"QR CUDA PREPROCESS DISABLED | {self.amr_name} reason={self.cuda_disabled_reason}")

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), "CPU"

    def decode(self) -> Optional[str]:
        self.scan_count += 1
        if QR_DETECTOR_EVERY_N_SCANS > 1 and self.scan_count % QR_DETECTOR_EVERY_N_SCANS != 0:
            return self.last_qr_id or None

        frame = self.get_rgb_array()
        if frame is None:
            return None

        try:
            bgr, _backend = self.rgb_to_bgr_for_qr(frame)

            decoded = ""
            if hasattr(self.detector, "detectAndDecodeMulti"):
                ok, decoded_info, points, _ = self.detector.detectAndDecodeMulti(bgr)
                if ok and decoded_info:
                    decoded = next((s for s in decoded_info if str(s).strip()), "")

            if not decoded:
                decoded, points, _ = self.detector.detectAndDecode(bgr)

            decoded = normalize_qr_id(decoded)
            if decoded:
                self.last_qr_id = decoded
                return decoded
            return None
        except Exception as e:
            self.last_error = str(e)
            return None


# ============================================================
# Planner data structures
# ============================================================
@dataclass
class PlanRequest:
    robot_id: str
    start: GridCell
    goal: GridCell
    heading: Move = (1, 0)
    carrying_rack: bool = False
    priority: int = 0
    waiting_steps: int = 0
    allowed_goal_occupied: bool = False


@dataclass
class PlanResult:
    robot_id: str
    path: List[GridCell]
    timed_path: List[TimedCell]
    success: bool
    reason: str = ""


class ReservationTable:
    def __init__(self):
        self.reserved_cell: Dict[TimedCell, str] = {}
        self.reserved_edge: Dict[EdgeKey, str] = {}
        self.soft_reserved_cell: Dict[TimedCell, str] = {}

    def clear(self):
        self.reserved_cell.clear()
        self.reserved_edge.clear()
        self.soft_reserved_cell.clear()

    def reserve_cell(self, cell: GridCell, t: int, robot_id: str):
        self.reserved_cell[(cell[0], cell[1], t)] = robot_id

    def reserve_edge(self, from_cell: GridCell, to_cell: GridCell, t: int, robot_id: str):
        self.reserved_edge[(from_cell[0], from_cell[1], to_cell[0], to_cell[1], t)] = robot_id

    def reserve_soft_cell(self, cell: GridCell, t: int, robot_id: str):
        self.soft_reserved_cell[(cell[0], cell[1], t)] = robot_id

    def is_cell_reserved(self, cell: GridCell, t: int, robot_id: str) -> bool:
        owner = self.reserved_cell.get((cell[0], cell[1], t))
        return owner is not None and owner != robot_id

    def is_soft_reserved(self, cell: GridCell, t: int, robot_id: str) -> bool:
        owner = self.soft_reserved_cell.get((cell[0], cell[1], t))
        return owner is not None and owner != robot_id

    def is_soft_cell_reserved(self, cell: GridCell, t: int, robot_id: str) -> bool:
        return self.is_soft_reserved(cell, t, robot_id)

    def is_edge_swap(self, from_cell: GridCell, to_cell: GridCell, t: int, robot_id: str) -> bool:
        owner = self.reserved_edge.get((to_cell[0], to_cell[1], from_cell[0], from_cell[1], t))
        return owner is not None and owner != robot_id

    def reserve_path(self, robot_id: str, timed_path: List[TimedCell], carrying_rack: bool, heading_path: List[Move]):
        if not timed_path:
            return
        horizon = timed_path[:RESERVATION_HORIZON]
        for i, node in enumerate(horizon):
            x, y, t = node
            cell = (x, y)
            self.reserve_cell(cell, t, robot_id)
            if carrying_rack:
                heading = heading_path[i] if i < len(heading_path) else (0, 0)
                for c in self._rack_soft_cells(cell, heading):
                    self.reserve_soft_cell(c, t, robot_id)
            if i > 0:
                px, py, pt = horizon[i - 1]
                self.reserve_edge((px, py), cell, pt, robot_id)
        gx, gy, gt = timed_path[-1]
        for dt in range(GOAL_HOLD_STEPS):
            self.reserve_cell((gx, gy), gt + dt, robot_id)

    def _rack_soft_cells(self, cell: GridCell, heading: Move) -> List[GridCell]:
        x, y = cell
        # Soft reservation mirrors the carried-rack footprint model used by the planner:
        # the AMR center owns the hard path cell, while the four adjacent cells are
        # treated as soft pressure zones for other path candidates.
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


class TimeAStar8:
    def __init__(self):
        self.moves8: List[Move] = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (0, 0),
        ]
        self.moves4: List[Move] = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

    def plan(self, req: PlanRequest, reservation: ReservationTable, static_obstacles: Set[GridCell], start_time: int) -> PlanResult:
        if req.start == req.goal:
            return PlanResult(req.robot_id, [req.start], [(req.start[0], req.start[1], start_time)], True, "already_at_goal")

        open_heap = []
        came_from: Dict[Tuple[int, int, int, int, int], Tuple[int, int, int, int, int]] = {}
        g_score: Dict[Tuple[int, int, int, int, int], float] = {}

        sx, sy = req.start
        hx, hy = req.heading
        start_node = (sx, sy, start_time, hx, hy)
        g_score[start_node] = 0.0
        heappush(open_heap, (self.heuristic(req.start, req.goal), 0.0, start_node))

        while open_heap:
            _, current_g, current = heappop(open_heap)
            x, y, t, chx, chy = current
            current_cell = (x, y)

            if current_cell == req.goal:
                path, timed, _ = self.reconstruct(came_from, current)
                return PlanResult(req.robot_id, path, timed, True, "success")

            if t - start_time >= MAX_TIME_HORIZON:
                continue

            for move in self.neighbors(req.carrying_rack):
                dx, dy = move
                nx, ny = x + dx, y + dy
                nt = t + 1
                next_cell = (nx, ny)

                if not self.valid_cell(next_cell, req.goal, req.allowed_goal_occupied, static_obstacles):
                    continue
                if not self.transition_safe(req.robot_id, current_cell, next_cell, t, nt, move, req, reservation, static_obstacles):
                    continue

                nhx, nhy = (chx, chy) if move == (0, 0) else move
                next_node = (nx, ny, nt, nhx, nhy)
                step_cost = self.step_cost((chx, chy), move, current_cell, next_cell, req.goal, nt, req.carrying_rack, reservation, req.robot_id, req.waiting_steps)
                # Balanced Mode-C policy:
                #   - immediate next-cell safety remains a hard condition in transition_safe()
                #   - the second cell is not a hard blocker, but it adds cost so A* prefers
                #     smoother paths that do not immediately run into an obstacle/dead-end.
                step_cost += self.second_step_soft_penalty(next_cell, move, nt, req, reservation, static_obstacles)
                tentative_g = current_g + step_cost

                if tentative_g < g_score.get(next_node, float("inf")):
                    came_from[next_node] = current
                    g_score[next_node] = tentative_g
                    f = tentative_g + self.heuristic(next_cell, req.goal)
                    heappush(open_heap, (f, tentative_g, next_node))

        return PlanResult(req.robot_id, [], [], False, "no_path")

    def neighbors(self, carrying_rack: bool) -> List[Move]:
        # Previous improved policy: normal AMR true 8-way, rack-carrying AMR 4-way only.
        return self.moves4 if carrying_rack else self.moves8

    def valid_cell(self, cell: GridCell, goal: GridCell, allowed_goal_occupied: bool, static_obstacles: Set[GridCell]) -> bool:
        # QR-only navigation: the planner may step only on cells that have an actual QR marker.
        # The goal is also snapped to a QR cell before planning, so no non-QR goal exception is needed.
        if QR_ONLY_NAVIGATION and GLOBAL_VALID_QR_CELLS and cell not in GLOBAL_VALID_QR_CELLS:
            return False
        if cell in static_obstacles and not (allowed_goal_occupied and cell == goal):
            return False
        return True

    def transition_safe(self, robot_id: str, from_cell: GridCell, to_cell: GridCell, t: int, nt: int, move: Move, req: PlanRequest, reservation: ReservationTable, static_obstacles: Set[GridCell]) -> bool:
        if reservation.is_cell_reserved(to_cell, nt, robot_id):
            return False
        if reservation.is_edge_swap(from_cell, to_cell, t, robot_id):
            return False
        if self.is_diagonal(move):
            # Rack/workstation leg constraint.
            # If the AMR is entering, leaving, crossing, or moving adjacent to a rack zone,
            # diagonal motion is forbidden when it is carrying a workstation.
            #
            # Pickup approach fix:
            # A non-carrying AMR may normally move 8-way and pass under stationary racks,
            # but when it is moving to a rack pickup goal, the final approach must be
            # cardinal. Otherwise the AMR can cut diagonally into a liftable cell and start
            # lifting from an unrealistic diagonal pose.
            c1 = (from_cell[0] + move[0], from_cell[1])
            c2 = (from_cell[0], from_cell[1] + move[1])

            if req.allowed_goal_occupied:
                if to_cell == req.goal or cell_chebyshev_distance(to_cell, req.goal) <= 1:
                    return False
                if RACK_FOUR_WAY_ZONE_ENABLED and GLOBAL_RACK_FOUR_WAY_CELLS:
                    if (
                        from_cell in GLOBAL_RACK_FOUR_WAY_CELLS
                        or to_cell in GLOBAL_RACK_FOUR_WAY_CELLS
                        or c1 in GLOBAL_RACK_FOUR_WAY_CELLS
                        or c2 in GLOBAL_RACK_FOUR_WAY_CELLS
                    ):
                        return False

            if req.carrying_rack and RACK_FOUR_WAY_ZONE_ENABLED and GLOBAL_RACK_FOUR_WAY_CELLS:
                if (
                    from_cell in GLOBAL_RACK_FOUR_WAY_CELLS
                    or to_cell in GLOBAL_RACK_FOUR_WAY_CELLS
                    or c1 in GLOBAL_RACK_FOUR_WAY_CELLS
                    or c2 in GLOBAL_RACK_FOUR_WAY_CELLS
                ):
                    return False

            # Corner cutting prevention.
            if (c1 in static_obstacles) or (c2 in static_obstacles):
                return False
            if reservation.is_cell_reserved(c1, nt, robot_id) or reservation.is_cell_reserved(c2, nt, robot_id):
                return False
        if req.carrying_rack:
            # Carrying rack footprint policy.
            # A* checks only the candidate next-cell footprint. The second planned cell is
            # checked later by the global move arbiter (Mode C two-step approval), so A*
            # does not become too conservative and produce unnecessary no_path failures.
            # At the final goal, do not require extra lookahead/side cells beyond the
            # destination. The goal cell itself is still validated by valid_cell().
            # Footprint cells check physical conflicts such as stationary racks and AMR
            # reservations. They do not require QR markers, because QR cells represent
            # AMR center navigation points rather than the entire rack body footprint.
            for c in self.rack_occupied_cells(to_cell, move, req.goal):
                if c == to_cell:
                    if not self.valid_cell(c, req.goal, req.allowed_goal_occupied, static_obstacles):
                        return False
                else:
                    if c in static_obstacles:
                        return False
                if reservation.is_cell_reserved(c, nt, robot_id):
                    return False
        return True

    def rack_occupied_cells(self, center: GridCell, move: Move, goal: Optional[GridCell] = None) -> List[GridCell]:
        x, y = center
        if goal is not None and center == goal:
            return [(x, y)]

        cells = [
            (x, y),
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        # Preserve order while removing duplicates.
        deduped: List[GridCell] = []
        seen: Set[GridCell] = set()
        for c in cells:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        return deduped

    def step_cost(self, heading: Move, move: Move, current_cell: GridCell, next_cell: GridCell, goal: GridCell, nt: int, carrying_rack: bool, reservation: ReservationTable, robot_id: str, waiting_steps: int = 0) -> float:
        """Cost model tuned for dense AMR convoy behavior.

        V22 intent:
        - A 90-degree turn should be cheap, so the planner chooses a short right/left
          turn instead of a long detour.
        - A U-turn / backward move should be much more expensive than waiting or
          turning. This prevents a follower AMR from suddenly reversing when the
          leader or a temporary reservation blocks the forward cell.
        - Moving farther away from the task goal is allowed but penalized. This keeps
          unavoidable detours possible while strongly preferring forward/right-turn
          progress.
        """
        if move == (0, 0):
            cost = WAIT_COST_CARRY if carrying_rack else WAIT_COST_EMPTY
            if CONGESTION_AGING_ENABLED and waiting_steps > 0:
                if carrying_rack:
                    cost += min(waiting_steps * CONGESTION_WAIT_COST_STEP_CARRY, CONGESTION_WAIT_COST_CAP_CARRY)
                else:
                    cost += min(waiting_steps * CONGESTION_WAIT_COST_STEP_EMPTY, CONGESTION_WAIT_COST_CAP_EMPTY)
        elif self.is_diagonal(move):
            cost = 1.414
        else:
            cost = 1.0

        if move != (0, 0) and heading != (0, 0):
            dot = heading[0] * move[0] + heading[1] * move[1]
            if move == heading:
                cost = max(0.01, cost - KEEP_HEADING_SMALL_BONUS)
            elif dot < 0:
                cost += TURN_COST_CARRY_180 if carrying_rack else TURN_COST_EMPTY_180
                cost += REVERSE_MOVE_EXTRA_COST_CARRY if carrying_rack else REVERSE_MOVE_EXTRA_COST_EMPTY
            else:
                cost += TURN_COST_CARRY_90 if carrying_rack else TURN_COST_EMPTY_90

            # v11: after the AMR has actually waited in congestion, make a 90-degree
            # turn/yield cheaper than another straight step. This is intentionally
            # conditional, so normal open-road driving does not zigzag.
            if BOTTLENECK_TURN_FIRST_ENABLED and waiting_steps >= BOTTLENECK_TURN_FIRST_WAIT_THRESHOLD:
                if dot == 0:
                    bonus = BOTTLENECK_TURN_BONUS_CARRY if carrying_rack else BOTTLENECK_TURN_BONUS_EMPTY
                    cost = max(0.05, cost - bonus)
                elif move == heading:
                    cost += BOTTLENECK_STRAIGHT_EXTRA_CARRY if carrying_rack else BOTTLENECK_STRAIGHT_EXTRA_EMPTY

        # Penalize moves that increase distance to the task goal.  This is the
        # core fix for the observed behavior where a trailing AMR waits and then
        # reverses, even though a short right turn would reach the goal faster.
        if move != (0, 0):
            h_now = self.heuristic(current_cell, goal)
            h_next = self.heuristic(next_cell, goal)
            if h_next > h_now + 1e-9:
                regression = h_next - h_now
                cost += regression * (GOAL_REGRESSION_COST_CARRY if carrying_rack else GOAL_REGRESSION_COST_EMPTY)

        if reservation.is_soft_reserved(next_cell, nt, robot_id):
            cost += 0.20
        return cost

    def second_step_soft_penalty(self, next_cell: GridCell, move: Move, nt: int, req: PlanRequest, reservation: ReservationTable, static_obstacles: Set[GridCell]) -> float:
        """Return a non-blocking cost penalty for the cell after next_cell.

        This keeps the 2-step lookahead useful without freezing the AMR when the
        immediate next cell is safe. The second cell is treated as a risk signal,
        not as a hard safety veto. A* will prefer safer 2-step candidates when they
        exist, but it can still advance one safe cell and replan on the next tick.
        """
        # V21: do not let 2-step lookahead distort the actual route.
        # The second cell remains available as a soft risk probe in the arbiter/logs,
        # but A* cost should primarily follow shortest/turn-minimal paths.
        # This prevents the regression where a simple 90-degree turn path is avoided
        # because a future cell adds artificial penalty.
        if True:
            return 0.0

        if move == (0, 0) or next_cell == req.goal:
            return 0.0

        second = (next_cell[0] + move[0], next_cell[1] + move[1])
        t2 = nt + 1
        penalty = 0.0

        if req.carrying_rack:
            # Carrying AMR: evaluate the same 4-neighbor rack footprint, but as a
            # soft preference only. Center cell QR validity matters for future AMR
            # center navigation; side cells are physical clearance cells.
            for c in self.rack_occupied_cells(second, move, req.goal):
                if c == second:
                    if QR_ONLY_NAVIGATION and GLOBAL_VALID_QR_CELLS and c not in GLOBAL_VALID_QR_CELLS:
                        penalty += 0.45
                    if c in static_obstacles and c != req.goal:
                        penalty += 0.45
                else:
                    if c in static_obstacles:
                        penalty += 0.30
                if reservation.is_cell_reserved(c, t2, req.robot_id):
                    penalty += 0.55
        else:
            # Empty AMR: do not treat stationary racks as obstacles. Keep only AMR
            # center navigation and active reservation risks as soft cost.
            if QR_ONLY_NAVIGATION and GLOBAL_VALID_QR_CELLS and second not in GLOBAL_VALID_QR_CELLS:
                penalty += 0.25
            if second in static_obstacles:
                penalty += 0.25
            if reservation.is_cell_reserved(second, t2, req.robot_id):
                penalty += 0.55

        return penalty

    def heuristic(self, cell: GridCell, goal: GridCell) -> float:
        dx = abs(cell[0] - goal[0])
        dy = abs(cell[1] - goal[1])
        # Octile distance for 8-way grid.
        return (max(dx, dy) - min(dx, dy)) + 1.414 * min(dx, dy)

    def is_diagonal(self, move: Move) -> bool:
        return abs(move[0]) == 1 and abs(move[1]) == 1

    def reconstruct(self, came_from, current):
        nodes = [current]
        while current in came_from:
            current = came_from[current]
            nodes.append(current)
        nodes.reverse()
        path = [(n[0], n[1]) for n in nodes]
        timed = [(n[0], n[1], n[2]) for n in nodes]
        headings = [(n[3], n[4]) for n in nodes]
        return path, timed, headings


# ============================================================
# Simulation state
# ============================================================
@dataclass
class SimObject:
    name: str
    path: str
    prim: object
    base_z: float


@dataclass
class RackState:
    key: str
    obj: SimObject
    cell: GridCell
    carried_by: Optional[str] = None
    assigned: bool = False
    physics_locked: bool = False


@dataclass
class CommandTask:
    command_id: str
    workstation_id: str
    rack_key: str
    target_location: str
    target_xy: Tuple[float, float]
    target_cell: GridCell
    target_yaw: float = 0.0
    action_mode: str = "MOVE_WORKSTATION"
    rotate_start_yaw: float = 0.0
    rotate_target_yaw: float = 0.0
    amr_name: Optional[str] = None
    preferred_amr_name: Optional[str] = None
    require_preferred_amr: bool = False
    rack_attach_dx: float = 0.0
    rack_attach_dy: float = 0.0
    phase: str = "PENDING"
    phase_start: float = 0.0
    last_status_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    # Cell where the AMR was located when this task was assigned.
    # After dropping the workstation, the AMR returns here before the command result is published.
    return_cell: Optional[GridCell] = None
    return_xy: Optional[Tuple[float, float]] = None
    # Internal clearance tasks are generated by the controller to move a workstation
    # that is blocking an SG2 inner A lane. They must not create bridge result files,
    # otherwise the batch test script would count them as user commands.
    internal: bool = False
    parent_command_id: Optional[str] = None
    clearance_reason: str = ""
    # v13: if an internal lane-clearance task preempts an external RETURN_HOME task,
    # this stores the external command id so it can be completed after clearance return-home.
    resume_task_id: Optional[str] = None


@dataclass
class AmrState:
    name: str
    obj: SimObject
    cell: GridCell
    heading: Move = (1, 0)
    state: str = "IDLE"
    target_cell: Optional[GridCell] = None
    target_xy: Optional[Tuple[float, float]] = None
    task_id: Optional[str] = None
    carrying_rack: Optional[str] = None
    move_from: Optional[GridCell] = None
    move_to: Optional[GridCell] = None
    move_elapsed: float = 0.0
    wait_steps: int = 0
    no_path_steps: int = 0
    current_qr_id: str = ""
    localization_source: str = "TRANSFORM"
    qr_candidate_id: str = ""
    qr_candidate_cell: Optional[GridCell] = None
    qr_candidate_count: int = 0
    qr_last_rejected_id: str = ""
    qr_last_rejected_reason: str = ""
    camera_path: Optional[str] = None
    qr_reader: Optional[QRCameraReader] = None
    last_qr_log_at: float = 0.0
    front_reference_path: str = ""
    front_yaw_offset_rad: float = AMR_FRONT_YAW_OFFSET_RAD

    def is_moving(self) -> bool:
        return self.move_from is not None and self.move_to is not None


# ============================================================
# Main controller
# ============================================================
class ExistingStageTrue8Controller:
    def __init__(self):
        ensure_dirs()
        self.stage = get_stage()
        if self.stage is None:
            raise RuntimeError("No active USD stage. Open your Nucleus Live stage first.")

        self.amrs: Dict[str, AmrState] = {}
        self.racks: Dict[str, RackState] = {}
        self.qr_prim_map: Dict[str, Tuple[GridCell, Tuple[float, float], str]] = {}
        self.valid_qr_cells: Set[GridCell] = set()
        self.qr_cell_world_map: Dict[GridCell, Tuple[float, float]] = {}
        self.rack_four_way_cells: Set[GridCell] = set()
        self.tasks: Dict[str, CommandTask] = {}
        self.processed_commands: Set[str] = set()
        self.controller_start_time = time.time()
        self.planner = TimeAStar8()
        self.reservation = ReservationTable()
        self.tick = 0
        self.redis_client = RedisStatusClient(REDIS_HOST, REDIS_PORT, REDIS_CONNECT_TIMEOUT_SEC, REDIS_RECONNECT_PERIOD_SEC) if REDIS_STATUS_ENABLED else None
        self.last_redis_publish_at = 0.0
        self.last_redis_log_at = 0.0
        self.last_decision_at = 0.0
        self.last_command_scan_at = 0.0
        self.last_qr_scan_at = 0.0
        self.qr_round_robin_index = 0
        self.last_log_at = 0.0
        self.redis_last_payloads: Dict[str, Dict[str, str]] = {}
        self.collision_counts = {
            "CELL": 0,
            "EDGE_SWAP": 0,
            "DIAGONAL_CROSS": 0,
            "FOOTPRINT": 0,
            "SWEPT_FOOTPRINT": 0,
        }

        self.build_qr_prim_map()
        self.load_stage_objects()
        self.build_rack_four_way_zones()
        self.setup_amr_qr_readers()
        self.assign_initial_random_goals()

        self._subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self.on_update,
            name="existing_stage_true8_amr_controller",
        )

        print("\nExistingStageTrue8Controller V26-GPU-5AMR-CONGESTION-AGING started")
        print(f"OpenCV CUDA status: {_OPENCV_CUDA_STATUS}; GPU enabled={AMR_GPU_ENABLED}; QR GPU preprocess={AMR_QR_GPU_PREPROCESS_ENABLED}")
        print(f"AMRs loaded: {list(self.amrs.keys())}")
        print(f"Racks loaded: {list(self.racks.keys())}")
        print(f"Bridge queue: {BRIDGE_QUEUE_DIR}")
        print("Press Play. Control Tower commands will move existing /World AMR/RACK prims. V19: optimized QR camera 320x240 + round-robin decode + Redis 5Hz + V18 compatibility.\n")
        if REDIS_STATUS_ENABLED:
            print(f"REDIS STATUS TARGET | {REDIS_HOST}:{REDIS_PORT} key_prefix={REDIS_KEY_PREFIX} period={REDIS_PUBLISH_PERIOD_SEC}s")

    def build_qr_prim_map(self):
        global GLOBAL_VALID_QR_CELLS, GLOBAL_QR_CELL_WORLD_MAP
        self.qr_prim_map.clear()
        self.valid_qr_cells.clear()
        self.qr_cell_world_map.clear()

        floor_pattern = re.compile(r"FLOOR[_\-]?X[_\-]?(-?\d+(?:\.\d+)?)[_\-]?Y[_\-]?(-?\d+(?:\.\d+)?)", re.IGNORECASE)

        for prim in self.stage.Traverse():
            if not prim.IsValid() or not prim.IsActive():
                continue
            path = str(prim.GetPath())
            name = prim.GetName()
            key_src = f"{name} {path}"
            low = key_src.lower()

            is_marker_candidate = (
                "qr" in low
                or "aruco" in low
                or "marker" in low
                or "floor_x" in low
                or "floor-x" in low
                or bool(floor_pattern.search(key_src))
            )
            if not is_marker_candidate:
                continue

            try:
                xy = get_xy(prim)
            except Exception:
                continue

            # If the QR ID encodes the true floor coordinate, prefer that over prim transform.
            encoded_xy = None
            m = floor_pattern.search(key_src)
            if m:
                try:
                    encoded_xy = (float(m.group(1)), float(m.group(2)))
                except Exception:
                    encoded_xy = None

            source_xy = encoded_xy if encoded_xy is not None else xy
            cell = world_to_grid(*source_xy)
            self.valid_qr_cells.add(cell)
            self.qr_cell_world_map.setdefault(cell, (float(source_xy[0]), float(source_xy[1])))

            candidates = {
                normalize_qr_id(name),
                normalize_qr_id(path.split("/")[-1]),
                normalize_qr_id(path),
            }
            if encoded_xy is not None:
                candidates.add(normalize_qr_id(f"FLOOR_X_{encoded_xy[0]}_Y_{encoded_xy[1]}"))
                candidates.add(normalize_qr_id(f"FLOOR_X_{encoded_xy[0]:.3f}_Y_{encoded_xy[1]:.3f}"))

            for c in list(candidates):
                if c:
                    self.qr_prim_map[c] = (cell, source_xy, path)

        GLOBAL_VALID_QR_CELLS = set(self.valid_qr_cells)
        GLOBAL_QR_CELL_WORLD_MAP = dict(self.qr_cell_world_map)
        print(f"QR MAP LOADED V19 | ids={len(self.qr_prim_map)} valid_qr_cells={len(self.valid_qr_cells)} exact_centers={len(self.qr_cell_world_map)}")
        if QR_ONLY_NAVIGATION and not self.valid_qr_cells:
            print("QR MAP WARNING | QR_ONLY_NAVIGATION is enabled but no QR cells were loaded. Planner will fall back to unrestricted grid.")
        if not CV2_AVAILABLE:
            print(f"QR CAMERA WARNING | cv2 unavailable: {_CV2_IMPORT_ERROR}")
        if not REPLICATOR_AVAILABLE:
            print(f"QR CAMERA WARNING | replicator unavailable: {_REPLICATOR_IMPORT_ERROR}")

    def setup_amr_qr_readers(self):
        if not QR_CAMERA_LOCALIZATION_ENABLED:
            print("QR CAMERA LOCALIZATION DISABLED")
            return
        for amr in self.amrs.values():
            camera_path = find_descendant_camera(self.stage, amr.obj.path)
            amr.camera_path = camera_path
            if camera_path:
                amr.qr_reader = QRCameraReader(amr.name, camera_path)
            else:
                print(f"QR CAMERA NOT FOUND | {amr.name} under={amr.obj.path}")

    def parse_qr_cell_from_text(self, qr_id: str) -> Optional[GridCell]:
        q = normalize_qr_id(qr_id)
        if not q:
            return None
        if q in self.qr_prim_map:
            return self.qr_prim_map[q][0]

        # Current floor QR payload pattern: FLOOR_X_4.225_Y_14.475
        m = re.search(r"FLOOR[_\-]?X[_\-]?(-?\d+(?:\.\d+)?)[_\-]?Y[_\-]?(-?\d+(?:\.\d+)?)", q, re.IGNORECASE)
        if m:
            xy = (float(m.group(1)), float(m.group(2)))
            cell = world_to_grid(*xy)
            # Register newly observed QR cell as valid if the stage map missed it.
            self.valid_qr_cells.add(cell)
            self.qr_cell_world_map.setdefault(cell, (float(xy[0]), float(xy[1])))
            GLOBAL_VALID_QR_CELLS.add(cell)
            GLOBAL_QR_CELL_WORLD_MAP.setdefault(cell, (float(xy[0]), float(xy[1])))
            return cell

        # Common integer grid patterns: QR_12_08, QR_X12_Y08, QR_0012_0008.
        m = re.search(r"(?:X)?(-?\d+)[_\-:, ]+(?:Y)?(-?\d+)", q)
        if m:
            cell = (int(m.group(1)), int(m.group(2)))
            if not QR_ONLY_NAVIGATION or not self.valid_qr_cells or cell in self.valid_qr_cells:
                return cell
            return self.nearest_qr_cell_to_world(*grid_to_world(cell))

        # If IDs are sequential QR_0000~QR_9999, interpret as row-major only if no QR map is available.
        m = re.search(r"(\d+)$", q)
        if m and not self.qr_prim_map:
            idx = int(m.group(1))
            grid_w = 40
            return idx % grid_w, idx // grid_w

        return None

    def nearest_qr_cell_to_world(self, x: float, y: float) -> GridCell:
        if not self.valid_qr_cells:
            return world_to_grid(x, y)
        best_cell = None
        best_d = float("inf")
        for cell in self.valid_qr_cells:
            wx, wy = grid_to_world(cell)
            d = (wx - x) ** 2 + (wy - y) ** 2
            if d < best_d:
                best_d = d
                best_cell = cell
        return best_cell if best_cell is not None else world_to_grid(x, y)

    def snap_xy_to_nearest_qr(self, x: float, y: float) -> Tuple[float, float, GridCell]:
        cell = self.nearest_qr_cell_to_world(x, y)
        wx, wy = grid_to_world(cell)
        return wx, wy, cell

    def explicit_drop_target_locked(self, target_location: str, rack_key: str, target_cell: GridCell) -> bool:
        """Return True when the command explicitly requested an SG2 A/B cell.

        This prevents the defensive drop-target fallback from changing
        sg2_in_XX_A to sg2_in_XX_B after lift completion. The only exception is
        when the requested target is exactly the pickup cell; in that invalid case
        we still need a real carry destination.
        """
        if not LOCK_EXPLICIT_SG2_AB_TARGETS:
            return False
        loc = str(target_location or "").strip().lower()
        if not (loc.startswith("sg2_in_") and (loc.endswith("_a") or loc.endswith("_b"))):
            return False
        rack = self.racks.get(rack_key)
        if rack is None:
            return False
        if target_cell == rack.cell:
            return False
        if QR_ONLY_NAVIGATION and self.valid_qr_cells and target_cell not in self.valid_qr_cells:
            return False
        return True

    def choose_drop_cell_away_from_pickup(self, desired_xy: Tuple[float, float], pickup_cell: GridCell) -> Tuple[float, float, GridCell]:
        """Return a QR cell that is not effectively the pickup cell.

        The Control Tower can send target_x/y equal to the current rack position.
        In that case the state machine sees TO_TARGET already complete right after
        lifting and immediately places the rack. This selector keeps the user
        requested target as much as possible, but guarantees a meaningful carry
        distance on QR cells.
        """
        if not self.valid_qr_cells:
            base = world_to_grid(*desired_xy)
            if cell_chebyshev_distance(base, pickup_cell) >= COMMAND_MIN_DROP_DISTANCE_CELLS:
                wx, wy = grid_to_world(base)
                return wx, wy, base
            fallback = (pickup_cell[0] + COMMAND_MIN_DROP_DISTANCE_CELLS, pickup_cell[1])
            wx, wy = grid_to_world(fallback)
            return wx, wy, fallback

        candidates = []
        for cell in self.valid_qr_cells:
            sep = cell_chebyshev_distance(cell, pickup_cell)
            if sep < COMMAND_MIN_DROP_DISTANCE_CELLS:
                continue
            wx, wy = grid_to_world(cell)
            desired_dist = distance_xy((wx, wy), desired_xy)
            # Prefer cells close to the requested target, then farther from pickup.
            score = desired_dist - 0.03 * sep
            candidates.append((score, desired_dist, -sep, cell, wx, wy))

        if not candidates:
            # Very defensive fallback: use the farthest known QR cell.
            far = max(self.valid_qr_cells, key=lambda c: cell_chebyshev_distance(c, pickup_cell))
            wx, wy = grid_to_world(far)
            return wx, wy, far

        candidates.sort()
        _, desired_dist, neg_sep, cell, wx, wy = candidates[0]
        return wx, wy, cell

    def enforce_drop_target_not_pickup(self, command_id: str, rack_key: str, target_xy: Tuple[float, float], target_cell: GridCell) -> Tuple[Tuple[float, float], GridCell]:
        rack = self.racks.get(rack_key)
        if rack is None:
            return target_xy, target_cell

        pickup_cell = rack.cell
        sep = cell_chebyshev_distance(target_cell, pickup_cell)
        if sep >= COMMAND_MIN_DROP_DISTANCE_CELLS:
            return target_xy, target_cell

        old_xy = target_xy
        old_cell = target_cell
        new_x, new_y, new_cell = self.choose_drop_cell_away_from_pickup(target_xy, pickup_cell)
        print(
            f"TARGET REASSIGNED AWAY FROM PICKUP | command={command_id} "
            f"pickup_cell={pickup_cell} old_target=({old_xy[0]:.3f}, {old_xy[1]:.3f}) "
            f"old_cell={old_cell} new_target=({new_x:.3f}, {new_y:.3f}) "
            f"new_cell={new_cell} min_sep={COMMAND_MIN_DROP_DISTANCE_CELLS}"
        )
        return (new_x, new_y), new_cell

    def update_qr_candidate_stability(self, amr: AmrState, qr_id: str, cell: GridCell) -> int:
        if amr.qr_candidate_id == qr_id and amr.qr_candidate_cell == cell:
            amr.qr_candidate_count += 1
        else:
            amr.qr_candidate_id = qr_id
            amr.qr_candidate_cell = cell
            amr.qr_candidate_count = 1
        return amr.qr_candidate_count

    def qr_localization_reject(self, amr: AmrState, qr_id: str, cell: Optional[GridCell], reason: str, now: float, force_log: bool = False) -> bool:
        amr.localization_source = "QR_REJECTED_HOLD_LAST_CELL"
        amr.qr_last_rejected_id = qr_id or ""
        amr.qr_last_rejected_reason = reason
        if force_log or now - amr.last_qr_log_at >= QR_LOG_PERIOD_SEC:
            print(
                f"QR REJECTED | {amr.name} qr={qr_id} detected_cell={cell} "
                f"hold_cell={amr.cell} state={amr.state} moving={amr.is_moving()} reason={reason}"
            )
            amr.last_qr_log_at = now
        return False

    def is_qr_cell_update_safe(self, amr: AmrState, cell: GridCell) -> Tuple[bool, str]:
        if amr.is_moving():
            return False, "amr_is_moving"
        if amr.state in QR_SAFE_BLOCK_STATES:
            return False, f"blocked_state_{amr.state}"

        jump = cell_chebyshev_distance(amr.cell, cell)
        if jump > QR_SAFE_MAX_CELL_JUMP:
            return False, f"cell_jump_{jump}_gt_{QR_SAFE_MAX_CELL_JUMP}"

        amr_xy = get_object_center_xy(amr.obj)
        qr_xy = self.qr_cell_world_map.get(cell, grid_to_world(cell))
        dist = distance_xy(amr_xy, qr_xy)
        if dist > QR_SAFE_MAX_WORLD_DIST_M:
            return False, f"world_dist_{dist:.3f}_gt_{QR_SAFE_MAX_WORLD_DIST_M:.3f}"

        return True, "safe"

    def localize_amr_by_qr_camera(self, amr: AmrState, now: float, force_log: bool = False) -> bool:
        if not QR_CAMERA_LOCALIZATION_ENABLED:
            return False
        if amr.qr_reader is None:
            if QR_ALLOW_TRANSFORM_FALLBACK:
                xy = get_object_center_xy(amr.obj)
                amr.cell = self.nearest_qr_cell_to_world(*xy) if QR_ONLY_NAVIGATION else world_to_grid(*xy)
                amr.localization_source = "TRANSFORM_FALLBACK_NO_CAMERA_QR_SNAPPED" if QR_ONLY_NAVIGATION else "TRANSFORM_FALLBACK_NO_CAMERA"
            return False

        qr_id = amr.qr_reader.decode()
        if not qr_id:
            if QR_ALLOW_TRANSFORM_FALLBACK:
                xy = get_object_center_xy(amr.obj)
                amr.cell = self.nearest_qr_cell_to_world(*xy) if QR_ONLY_NAVIGATION else world_to_grid(*xy)
                amr.localization_source = "TRANSFORM_FALLBACK_QR_MISS_QR_SNAPPED" if QR_ONLY_NAVIGATION else "TRANSFORM_FALLBACK_QR_MISS"
            else:
                amr.localization_source = "QR_MISS_HOLD_LAST_CELL"
            if force_log or now - amr.last_qr_log_at >= QR_LOG_PERIOD_SEC:
                print(f"QR MISS | {amr.name} camera={amr.camera_path} hold_cell={amr.cell} fallback={QR_ALLOW_TRANSFORM_FALLBACK} err={amr.qr_reader.last_error}")
                amr.last_qr_log_at = now
            return False

        cell = self.parse_qr_cell_from_text(qr_id)
        if cell is None:
            if QR_ALLOW_TRANSFORM_FALLBACK:
                xy = get_object_center_xy(amr.obj)
                amr.cell = self.nearest_qr_cell_to_world(*xy) if QR_ONLY_NAVIGATION else world_to_grid(*xy)
                amr.localization_source = "TRANSFORM_FALLBACK_QR_UNMAPPED_QR_SNAPPED" if QR_ONLY_NAVIGATION else "TRANSFORM_FALLBACK_QR_UNMAPPED"
            else:
                amr.localization_source = "QR_UNMAPPED_HOLD_LAST_CELL"
            if force_log or now - amr.last_qr_log_at >= QR_LOG_PERIOD_SEC:
                print(f"QR UNMAPPED | {amr.name} qr={qr_id} hold_cell={amr.cell} fallback={QR_ALLOW_TRANSFORM_FALLBACK}")
                amr.last_qr_log_at = now
            amr.current_qr_id = qr_id
            return False

        stable_count = self.update_qr_candidate_stability(amr, qr_id, cell)
        if stable_count < max(1, QR_SAFE_STABLE_DETECTIONS):
            return self.qr_localization_reject(
                amr,
                qr_id,
                cell,
                f"stable_count_{stable_count}_lt_{QR_SAFE_STABLE_DETECTIONS}",
                now,
                force_log,
            )

        safe, reason = self.is_qr_cell_update_safe(amr, cell)
        if not safe:
            return self.qr_localization_reject(amr, qr_id, cell, reason, now, force_log)

        amr.current_qr_id = qr_id
        old_cell = amr.cell
        amr.cell = cell
        amr.localization_source = "QR_CAMERA_SAFE"
        if force_log or now - amr.last_qr_log_at >= QR_LOG_PERIOD_SEC:
            print(
                f"QR LOCALIZED SAFE | {amr.name} qr={qr_id} old_cell={old_cell} "
                f"new_cell={cell} stable={stable_count} camera={amr.camera_path}"
            )
            amr.last_qr_log_at = now
        return True

    def update_qr_localization(self, now: float):
        amr_list = list(self.amrs.values())
        if not amr_list:
            return

        # FPS optimization: do not decode all five camera render products on every QR cycle.
        # Each AMR keeps its last valid QR/cell; only a small round-robin subset is decoded per cycle.
        # This keeps QR-based navigation while reducing camera readback + OpenCV workload.
        if QR_SCAN_ROUND_ROBIN_ENABLED:
            count = max(1, min(int(QR_SCAN_MAX_AMRS_PER_CYCLE), len(amr_list)))
            selected = []
            for _ in range(count):
                selected.append(amr_list[self.qr_round_robin_index % len(amr_list)])
                self.qr_round_robin_index += 1
        else:
            selected = amr_list

        for amr in selected:
            # Avoid changing discrete planner cell while interpolating between cells.
            if amr.is_moving():
                continue
            self.localize_amr_by_qr_camera(amr, now)

    def load_stage_objects(self):
        for name, path in AMR_PRIM_PATHS.items():
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsActive():
                print(f"AMR NOT VALID/ACTIVE: {name} {path}")
                continue
            z = get_z(prim)
            obj = SimObject(name=name, path=path, prim=prim, base_z=z)
            xy = get_object_center_xy(obj)
            cell = self.nearest_qr_cell_to_world(*xy) if QR_ONLY_NAVIGATION else world_to_grid(*xy)
            front_offset, front_ref_path = calibrate_front_yaw_offset(self.stage, prim, name)
            self.amrs[name] = AmrState(
                name=name,
                obj=obj,
                cell=cell,
                localization_source="TRANSFORM_INIT_QR_SNAPPED" if QR_ONLY_NAVIGATION else "TRANSFORM_INIT",
                front_reference_path=front_ref_path,
                front_yaw_offset_rad=front_offset,
            )
            print(
                f"AMR OK: {name} path={path} center_xy={xy} cell={cell} "
                f"front_ref={front_ref_path or 'FALLBACK_OFFSET'} front_offset_deg={math.degrees(front_offset):.2f}"
            )

        # Load only WS_01~WS_10 keys to avoid duplicates from RACK_XX aliases.
        for i in range(1, 11):
            key = f"WS_{i:02d}"
            path = RACK_PRIM_PATHS[key]
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsActive():
                print(f"RACK NOT VALID/ACTIVE: {key} {path}")
                continue
            z = get_z(prim)
            obj = SimObject(name=key, path=path, prim=prim, base_z=z)
            if RACK_DISABLE_PHYSICS_AT_STARTUP:
                hard_disable_physics_subtree(prim, disable_collision=True)
                print(f"RACK SCRIPT CONTROLLED | {key} physics/collision disabled at startup")
            xy = get_object_center_xy(obj)
            cell = self.nearest_qr_cell_to_world(*xy) if QR_ONLY_NAVIGATION else world_to_grid(*xy)
            # Snap rack root to its nearest QR center at startup so rack origins are also aligned.
            if QR_ONLY_NAVIGATION and self.valid_qr_cells:
                qx, qy = grid_to_world(cell)
                set_object_center_xy(obj, qx, qy, obj.base_z)
                xy = get_object_center_xy(obj)
            self.racks[key] = RackState(key=key, obj=obj, cell=cell, physics_locked=RACK_DISABLE_PHYSICS_AT_STARTUP)
            print(f"RACK OK: {key} path={path} center_xy={xy} cell={cell}")

        if not self.amrs:
            raise RuntimeError("No AMR prims loaded. Check /World/AMR_01~05 paths.")
        if not self.racks:
            raise RuntimeError("No RACK prims loaded. Check /World/RACK_01~10 paths.")

    def build_rack_four_way_zones(self):
        """Build cells where diagonal motion is forbidden because of workstation legs.

        The workstation has four legs. When an AMR goes under/through or immediately around
        the rack, allowing diagonal motion can visually cut through the legs. This zone forces
        the planner and arbiter to use only north/east/south/west movement in and near racks.
        """
        global GLOBAL_RACK_FOUR_WAY_CELLS
        self.rack_four_way_cells.clear()

        if not RACK_FOUR_WAY_ZONE_ENABLED:
            GLOBAL_RACK_FOUR_WAY_CELLS = set()
            print("RACK FOUR-WAY ZONE DISABLED")
            return

        r = max(0, int(RACK_FOUR_WAY_ZONE_RADIUS_CELLS))
        for rack in self.racks.values():
            cx, cy = rack.cell
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    cell = (cx + dx, cy + dy)
                    if QR_ONLY_NAVIGATION and self.valid_qr_cells and cell not in self.valid_qr_cells:
                        continue
                    self.rack_four_way_cells.add(cell)

        GLOBAL_RACK_FOUR_WAY_CELLS = set(self.rack_four_way_cells)
        print(
            f"RACK FOUR-WAY ZONE LOADED | racks={len(self.racks)} "
            f"zone_cells={len(self.rack_four_way_cells)} radius={RACK_FOUR_WAY_ZONE_RADIUS_CELLS}"
        )

    def assign_initial_random_goals(self):
        if not RANDOM_DRIVE_ENABLED or STANDBY_UNTIL_COMMAND:
            for amr in self.amrs.values():
                if amr.state not in ["TO_RACK", "LIFTING", "TO_TARGET", "PLACING", "RETURN_HOME"]:
                    amr.state = "IDLE"
                    amr.target_cell = None
                    amr.target_xy = None
            print("STANDBY MODE | AMRs will stay idle until /manage_workstation command arrives")
            return
        for amr in self.amrs.values():
            self.assign_random_goal(amr)

    def assign_random_goal(self, amr: AmrState):
        candidates: List[GridCell] = []
        obstacles = self.static_obstacles_for(amr, allow_assigned_rack=False)
        occupied_targets = self.occupied_and_target_cells(exclude_name=amr.name) if UNIQUE_RANDOM_GOALS_ENABLED else set()
        blocked = obstacles | occupied_targets

        if QR_ONLY_NAVIGATION and self.valid_qr_cells:
            for cell in self.valid_qr_cells:
                if cell == amr.cell or cell in blocked:
                    continue
                if cell in self.rack_four_way_cells:
                    # keep random circulation goals out of the rack-leg zone.
                    continue
                d = cell_manhattan_distance(cell, amr.cell)
                if RANDOM_GOAL_MIN_RADIUS_CELL <= d <= RANDOM_GOAL_RANGE_CELL * 2:
                    candidates.append(cell)
            if not candidates:
                for cell in self.valid_qr_cells:
                    if cell != amr.cell and cell not in blocked and cell not in self.rack_four_way_cells:
                        candidates.append(cell)
            if candidates:
                # Prefer targets farther from other AMR targets to reduce identical-coordinate convergence.
                def score(cell: GridCell) -> Tuple[int, int]:
                    other_targets = [a.target_cell for a in self.amrs.values() if a.name != amr.name and a.target_cell is not None]
                    min_sep = min((cell_chebyshev_distance(cell, t) for t in other_targets), default=99)
                    dist_self = cell_manhattan_distance(cell, amr.cell)
                    return (min_sep, dist_self)

                ranked = sorted(candidates, key=score, reverse=True)
                top = ranked[: min(16, len(ranked))]
                cell = random.choice(top)
                amr.target_cell = cell
                amr.target_xy = grid_to_world(cell)
                if amr.state not in ["TO_RACK", "LIFTING", "TO_TARGET", "PLACING", "RETURN_HOME"]:
                    amr.state = "RANDOM"
                return

        for _ in range(120):
            dx = random.randint(-RANDOM_GOAL_RANGE_CELL, RANDOM_GOAL_RANGE_CELL)
            dy = random.randint(-RANDOM_GOAL_RANGE_CELL, RANDOM_GOAL_RANGE_CELL)
            if abs(dx) + abs(dy) < RANDOM_GOAL_MIN_RADIUS_CELL:
                continue
            cell = (amr.cell[0] + dx, amr.cell[1] + dy)
            if cell in blocked or cell in self.rack_four_way_cells:
                continue
            amr.target_cell = cell
            amr.target_xy = grid_to_world(cell)
            if amr.state not in ["TO_RACK", "LIFTING", "TO_TARGET", "PLACING", "RETURN_HOME"]:
                amr.state = "RANDOM"
            return
        amr.target_cell = None
        amr.target_xy = None

    def occupied_and_target_cells(self, exclude_name: Optional[str] = None) -> Set[GridCell]:
        cells: Set[GridCell] = set()
        for other in self.amrs.values():
            if other.name == exclude_name:
                continue
            cells.add(other.cell)
            if other.target_cell is not None:
                cells.add(other.target_cell)
            if other.is_moving() and other.move_to is not None:
                cells.add(other.move_to)
        for rack in self.racks.values():
            cells.add(rack.cell)
        if UNIQUE_TASK_TARGET_RESERVATION_ENABLED:
            for task in self.tasks.values():
                if task.target_cell is not None:
                    cells.add(task.target_cell)
        return cells

    def release_random_goals_conflicting_with(self, reserved_cell: GridCell, owner_name: Optional[str] = None, reason: str = "reserved_target"):
        if not UNIQUE_RANDOM_GOALS_ENABLED:
            return
        for other in self.amrs.values():
            if owner_name is not None and other.name == owner_name:
                continue
            if other.state != "RANDOM":
                continue
            if other.target_cell == reserved_cell:
                print(f"UNIQUE TARGET RELEASE | {other.name} reason={reason} released={reserved_cell}")
                other.target_cell = None
                other.target_xy = None
                self.assign_dispersed_random_goal(other, reason=reason)

    def resolve_duplicate_random_targets(self):
        if not DUPLICATE_TARGET_RESOLVE_ENABLED:
            return
        target_groups: Dict[GridCell, List[AmrState]] = {}
        for amr in self.amrs.values():
            if amr.state != "RANDOM" or amr.target_cell is None:
                continue
            target_groups.setdefault(amr.target_cell, []).append(amr)

        for cell, group in target_groups.items():
            if len(group) <= 1:
                continue
            # Keep the AMR closest to the duplicated target; reassign the rest.
            group.sort(key=lambda a: cell_manhattan_distance(a.cell, cell))
            keeper = group[0]
            for amr in group[1:]:
                print(f"DUPLICATE TARGET REASSIGN | keep={keeper.name} reassign={amr.name} duplicated_goal={cell}")
                amr.target_cell = None
                amr.target_xy = None
                self.assign_dispersed_random_goal(amr, reason=f"duplicate_goal_{cell}")

        # Also keep random AMR goals away from active task destination cells.
        task_targets = {task.target_cell for task in self.tasks.values() if task.target_cell is not None}
        for amr in self.amrs.values():
            if amr.state == "RANDOM" and amr.target_cell in task_targets:
                old = amr.target_cell
                print(f"TASK TARGET CONFLICT REASSIGN | {amr.name} old_goal={old}")
                amr.target_cell = None
                amr.target_xy = None
                self.assign_dispersed_random_goal(amr, reason=f"task_target_conflict_{old}")

    def assign_dispersed_random_goal(self, amr: AmrState, reason: str = "anti_clump") -> bool:
        obstacles = self.static_obstacles_for(amr, allow_assigned_rack=False)
        occupied = self.occupied_and_target_cells(exclude_name=amr.name) | obstacles

        source_cells = list(self.valid_qr_cells) if (QR_ONLY_NAVIGATION and self.valid_qr_cells) else []
        if not source_cells:
            # Fallback to a bounded search around the robot if QR map is unavailable.
            cx, cy = amr.cell
            for dx in range(-RANDOM_GOAL_RANGE_CELL * 2, RANDOM_GOAL_RANGE_CELL * 2 + 1):
                for dy in range(-RANDOM_GOAL_RANGE_CELL * 2, RANDOM_GOAL_RANGE_CELL * 2 + 1):
                    source_cells.append((cx + dx, cy + dy))

        ranked: List[Tuple[int, int, GridCell]] = []
        for cell in source_cells:
            if cell == amr.cell or cell in occupied:
                continue
            if cell in self.rack_four_way_cells:
                # do not choose a rack-leg constrained zone as a random circulation target.
                continue
            dist_from_self = cell_manhattan_distance(cell, amr.cell)
            if dist_from_self < RANDOM_GOAL_MIN_RADIUS_CELL:
                continue
            min_sep = min((cell_chebyshev_distance(cell, c) for c in occupied), default=99)
            if min_sep < ANTI_CLUMP_GOAL_MIN_SEPARATION_CELLS:
                continue
            # Prefer far, well-separated goals.
            ranked.append((min_sep, dist_from_self, cell))

        if not ranked:
            return False

        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        # Use one of the best few to avoid deterministic cycling.
        top = ranked[: min(12, len(ranked))]
        _, _, chosen = random.choice(top)
        amr.target_cell = chosen
        amr.target_xy = grid_to_world(chosen)
        if amr.state not in ["TO_RACK", "LIFTING", "TO_TARGET", "PLACING", "RETURN_HOME"]:
            amr.state = "RANDOM"
        amr.wait_steps = 0
        amr.no_path_steps = 0
        print(f"ANTI-CLUMP GOAL REASSIGNED | {amr.name} reason={reason} new_goal={chosen}")
        return True

    def resolve_random_clumps(self):
        if not ANTI_CLUMP_ENABLED:
            return
        random_amrs = [a for a in self.amrs.values() if a.state == "RANDOM" and not a.is_moving()]
        for amr in random_amrs:
            near = [
                other for other in self.amrs.values()
                if other.name != amr.name and cell_chebyshev_distance(other.cell, amr.cell) <= ANTI_CLUMP_RADIUS_CELLS
            ]
            if len(near) + 1 >= ANTI_CLUMP_MIN_GROUP_SIZE:
                self.assign_dispersed_random_goal(amr, reason=f"cluster_size_{len(near)+1}")

    def on_update(self, event):
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return

        now = time.time()
        dt = getattr(event, "dt", None)
        if dt is None or dt <= 0.0:
            dt = 1.0 / 60.0
        dt = min(float(dt), 0.05)

        if now - self.last_qr_scan_at >= QR_CAMERA_SCAN_PERIOD_SEC:
            self.update_qr_localization(now)
            self.last_qr_scan_at = now

        if now - self.last_command_scan_at >= COMMAND_SCAN_PERIOD_SEC:
            self.scan_commands()
            self.last_command_scan_at = now

        self.update_task_phases(now, dt)

        # Dynamic SG2 lane clearance: allow B placements, but evacuate a placed B rack
        # when a matching A placement needs to enter behind it.
        self.ensure_sg2_lane_clearance_tasks()

        if now - self.last_decision_at >= DECISION_PERIOD_SEC:
            self.plan_and_approve_moves()
            self.last_decision_at = now

        self.update_motion(dt)
        self.detect_collisions()
        self.publish_redis_status_if_due(now)

        if now - self.last_log_at >= LOG_PERIOD_SEC:
            self.print_summary()
            self.last_log_at = now

    # --------------------------------------------------------
    # Redis realtime status publishing
    # --------------------------------------------------------
    def amr_state_for_redis(self, amr: AmrState) -> str:
        if amr.state in ["ERROR", "FAILED"]:
            return "ERROR"
        if amr.is_moving() or amr.state in ["TO_RACK", "LIFTING", "TO_TARGET", "PLACING", "RETURN_HOME"]:
            return "MOVING"
        return "IDLE"

    def amr_target_qr_for_redis(self, amr: AmrState) -> str:
        if amr.target_cell is not None:
            return cell_to_floor_qr_id_from_map(amr.target_cell, self.qr_cell_world_map)
        return ""

    def amr_current_qr_for_redis(self, amr: AmrState) -> str:
        if amr.current_qr_id:
            return amr.current_qr_id
        return cell_to_floor_qr_id_from_map(amr.cell, self.qr_cell_world_map)

    def publish_redis_status_if_due(self, now: float):
        if not REDIS_STATUS_ENABLED or self.redis_client is None:
            return
        if now - self.last_redis_publish_at < REDIS_PUBLISH_PERIOD_SEC:
            return
        self.last_redis_publish_at = now

        ok_count = 0
        skipped_count = 0
        for amr in self.amrs.values():
            mapping = {
                "state": self.amr_state_for_redis(amr),
                "current_qr_id": self.amr_current_qr_for_redis(amr),
                "target_qr_id": self.amr_target_qr_for_redis(amr),
                "carrying_workstation_id": amr.carrying_rack or "",
                "battery": REDIS_BATTERY_DEFAULT,
            }
            key = f"{REDIS_KEY_PREFIX}{amr.name}"

            # Redis optimization: if nothing changed, do not write the same hash every cycle.
            # A full refresh is still forced at the reduced publish period by status changes.
            if self.redis_last_payloads.get(key) == mapping:
                skipped_count += 1
                continue

            if self.redis_client.hset(key, mapping):
                self.redis_last_payloads[key] = dict(mapping)
                ok_count += 1

        if ok_count == 0 and skipped_count == 0 and now - self.last_redis_log_at >= REDIS_RECONNECT_PERIOD_SEC:
            self.last_redis_log_at = now
            print(f"REDIS STATUS WAITING | {REDIS_HOST}:{REDIS_PORT} err={self.redis_client.last_error}")
        elif ok_count > 0 and now - self.last_redis_log_at >= 15.0:
            self.last_redis_log_at = now
            print(f"REDIS STATUS SENT | changed_amrs={ok_count} skipped_unchanged={skipped_count} host={REDIS_HOST}:{REDIS_PORT}")

    # --------------------------------------------------------
    # Dynamic SG2 lane clearance
    # --------------------------------------------------------
    def parse_sg2_target_location(self, target_location: str) -> Optional[Tuple[str, str]]:
        """Return (lane, slot) for names like sg2_in_02_A."""
        m = re.search(r"sg2[_\-]?in[_\-]?(\d+)[_\-]?([AB])", str(target_location or ""), re.IGNORECASE)
        if not m:
            return None
        return (m.group(1).zfill(2), m.group(2).upper())

    def sg2_outer_b_cell_for_inner_a_task(self, task: CommandTask) -> Optional[GridCell]:
        """Infer the outer B cell that can block the matching inner A cell.

        In the current Isaac grid used by the commands, A is one cell deeper in +x
        than B. Example: sg2_in_02_B=(4,-2), sg2_in_02_A=(5,-2).
        We intentionally infer from task.target_cell, not LOCATION_TARGETS, because
        incoming command JSON already provides the calibrated positive Isaac coordinates.
        """
        parsed = self.parse_sg2_target_location(task.target_location)
        if parsed is None:
            return None
        _, slot = parsed
        if slot != "A":
            return None
        return (task.target_cell[0] - 1, task.target_cell[1])

    def stationary_rack_at_cell(self, cell: GridCell, exclude_key: Optional[str] = None) -> Optional[RackState]:
        for rack in self.racks.values():
            if exclude_key is not None and rack.key == exclude_key:
                continue
            if rack.cell != cell:
                continue
            if rack.carried_by is not None:
                continue
            # If another active task is already assigned to this rack, do not steal it.
            if rack.assigned:
                return None
            return rack
        return None

    def active_clearance_for_rack(self, rack_key: str) -> bool:
        for task in self.tasks.values():
            if task.internal and task.action_mode == "LANE_CLEARANCE" and task.rack_key == rack_key:
                return True
        return False

    def preempt_return_home_for_lane_clearance(self, clearance_task: CommandTask, blocker: RackState, parent_task: CommandTask) -> bool:
        """Use a non-carrying RETURN_HOME AMR as the clearance carrier when no AMR is idle.

        This keeps the user's return-home policy: the preempted external command is
        not completed immediately.  It is paused, the AMR clears the blocking rack,
        then the internal task returns the same AMR to the original home cell and
        completes the paused external command.
        """
        if not RETURN_HOME_CLEARANCE_PREEMPT_ENABLED:
            return False

        candidates: List[Tuple[int, str, CommandTask]] = []
        for task in self.tasks.values():
            if task.internal:
                continue
            if task.phase != "RETURN_HOME":
                continue
            amr = self.amrs.get(task.amr_name or "")
            if amr is None:
                continue
            if amr.is_moving() or amr.carrying_rack:
                continue
            if amr.state != "RETURN_HOME":
                continue
            # Do not steal the A-bound AMR itself.
            if amr.name == parent_task.amr_name:
                continue
            candidates.append((cell_manhattan_distance(amr.cell, blocker.cell), amr.name, task))

        if not candidates:
            return False

        candidates.sort()
        _, amr_name, paused_task = candidates[0]
        amr = self.amrs[amr_name]

        paused_task.phase = "RETURN_HOME_PAUSED"
        paused_task.phase_start = time.time()

        clearance_task.amr_name = amr.name
        clearance_task.phase = "TO_RACK"
        clearance_task.phase_start = time.time()
        clearance_task.return_cell = paused_task.return_cell
        clearance_task.return_xy = paused_task.return_xy
        clearance_task.resume_task_id = paused_task.command_id

        blocker.assigned = True
        amr.state = "TO_RACK"
        amr.task_id = clearance_task.command_id
        amr.carrying_rack = None
        amr.target_cell = blocker.cell
        amr.target_xy = grid_to_world(blocker.cell)
        amr.wait_steps = 0
        amr.no_path_steps = 0
        self.release_random_goals_conflicting_with(blocker.cell, owner_name=amr.name, reason="lane_clearance_preempt_pickup")
        self.release_random_goals_conflicting_with(clearance_task.target_cell, owner_name=amr.name, reason="lane_clearance_preempt_drop")
        print(
            f"LANE CLEARANCE PREEMPT RETURN_HOME | parent={parent_task.command_id} blocker={blocker.key} "
            f"amr={amr.name} paused={paused_task.command_id} from={blocker.cell} to={clearance_task.target_cell} "
            f"return={clearance_task.return_cell}"
        )
        return True

    def active_external_targets(self) -> Set[GridCell]:
        cells: Set[GridCell] = set()
        for task in self.tasks.values():
            if task.internal:
                continue
            cells.add(task.target_cell)
            if task.return_cell is not None:
                cells.add(task.return_cell)
        return cells

    def choose_lane_clearance_cell(self, blocked_cell: GridCell, a_cell: GridCell, blocking_rack_key: str) -> Optional[GridCell]:
        """Choose a temporary cell for a rack that blocks an SG2 B->A lane.

        The target must be a QR-valid free cell, must not be an active task target,
        and should be outside the direct B/A lane.  The scoring prefers moving the
        rack left/outward and slightly aside rather than placing it directly in the
        same lane again.
        """
        occupied_racks = {rack.cell for key, rack in self.racks.items() if key != blocking_rack_key and rack.carried_by is None}
        occupied_amrs = {amr.cell for amr in self.amrs.values()}
        moving_to = {amr.move_to for amr in self.amrs.values() if amr.move_to is not None}
        forbidden = occupied_racks | occupied_amrs | moving_to | self.active_external_targets()
        forbidden.add(blocked_cell)
        forbidden.add(a_cell)
        # Keep the direct A/B lane clear after evacuation.
        for dx in range(-1, 2):
            forbidden.add((blocked_cell[0] + dx, blocked_cell[1]))

        candidates: List[GridCell] = []
        if self.valid_qr_cells:
            source = self.valid_qr_cells
        else:
            bx, by = blocked_cell
            source = {(x, y) for x in range(bx - 6, bx + 3) for y in range(by - 6, by + 7)}

        bx, by = blocked_cell
        for cell in source:
            if cell in forbidden:
                continue
            if cell in self.static_obstacles_for_virtual(blocking_rack_key):
                continue
            # Do not place the evacuation target deeper than the blocked SG2 B cell.
            if cell[0] >= blocked_cell[0]:
                continue
            candidates.append(cell)

        if not candidates:
            return None

        def score(cell: GridCell) -> Tuple[float, int, int, int]:
            x, y = cell
            dist = abs(x - bx) + abs(y - by)
            same_lane_penalty = 8 if y == by else 0
            near_a_penalty = 20 if cell_chebyshev_distance(cell, a_cell) <= 1 else 0
            # Prefer x <= B-2 so the B cell and its immediate approach stay clear.
            shallow_penalty = 5 if x == bx - 1 else 0
            return (dist + same_lane_penalty + near_a_penalty + shallow_penalty, abs(y - by), -x, y)

        candidates.sort(key=score)
        return candidates[0]

    def static_obstacles_for_virtual(self, exclude_rack_key: str) -> Set[GridCell]:
        obstacles: Set[GridCell] = set()
        for key, rack in self.racks.items():
            if key == exclude_rack_key:
                continue
            if rack.carried_by is not None:
                continue
            obstacles.add(rack.cell)
        return obstacles

    def ensure_sg2_lane_clearance_tasks(self):
        """Optionally create internal lane-clearance tasks.

        v14 default is OFF: after a workstation is placed at its final requested
        destination, the controller must not pick it again just to clear a lane.
        This keeps SG A/B and stage drops final for standalone 10-workstation tests.
        """
        if not DYNAMIC_SG2_LANE_CLEARANCE_ENABLED:
            return
        for task in list(self.tasks.values()):
            if task.internal:
                continue
            if task.phase not in {"TO_RACK", "LIFTING", "TO_TARGET", "PLACING"}:
                continue
            b_cell = self.sg2_outer_b_cell_for_inner_a_task(task)
            if b_cell is None:
                continue
            blocker = self.stationary_rack_at_cell(b_cell, exclude_key=task.rack_key)
            if blocker is None:
                continue
            if self.active_clearance_for_rack(blocker.key):
                continue
            clearance_cell = self.choose_lane_clearance_cell(b_cell, task.target_cell, blocker.key)
            if clearance_cell is None:
                now = time.time()
                print(f"LANE CLEARANCE WAIT | parent={task.command_id} blocker={blocker.key} at={b_cell} reason=no_clearance_cell")
                continue
            clearance_id = f"__CLEAR_{blocker.key}_FROM_{b_cell[0]}_{b_cell[1]}_FOR_{task.command_id}"
            if clearance_id in self.tasks:
                continue
            clearance_task = CommandTask(
                command_id=clearance_id,
                workstation_id=blocker.key,
                rack_key=blocker.key,
                target_location=f"lane_clearance_for_{task.target_location}",
                target_xy=grid_to_world(clearance_cell),
                target_cell=clearance_cell,
                target_yaw=0.0,
                action_mode="LANE_CLEARANCE",
                internal=True,
                parent_command_id=task.command_id,
                clearance_reason=f"blocking_{task.target_location}_via_{b_cell}",
            )
            if self.assign_task_to_amr(clearance_task):
                self.tasks[clearance_id] = clearance_task
                print(
                    f"LANE CLEARANCE START | parent={task.command_id} blocker={blocker.key} "
                    f"from={b_cell} to={clearance_cell} amr={clearance_task.amr_name}"
                )
            elif self.preempt_return_home_for_lane_clearance(clearance_task, blocker, task):
                self.tasks[clearance_id] = clearance_task
                print(
                    f"LANE CLEARANCE START | parent={task.command_id} blocker={blocker.key} "
                    f"from={b_cell} to={clearance_cell} amr={clearance_task.amr_name} mode=preempt_return_home"
                )
            else:
                # No idle/preemptable AMR yet. Retry on the next decision cycle; do not fail the user task.
                if int(time.time() * 10) % 30 == 0:
                    print(f"LANE CLEARANCE WAIT | parent={task.command_id} blocker={blocker.key} at={b_cell} reason=no_idle_or_preemptable_amr")

    # --------------------------------------------------------
    # Command queue
    # --------------------------------------------------------
    def scan_commands(self):
        for command_path in sorted(COMMAND_DIR.glob("*.json")):
            if command_path.name in self.processed_commands:
                continue
            data = safe_read_json(command_path)
            if not data:
                continue
            command_id = str(data.get("command_id", command_path.stem))
            created_at = float(data.get("created_at", 0.0) or 0.0)
            if IGNORE_STALE_COMMANDS_ON_STARTUP and created_at > 0.0:
                if created_at < self.controller_start_time - STALE_COMMAND_GRACE_SEC:
                    done_path = DONE_DIR / command_path.name
                    try:
                        shutil.move(str(command_path), str(done_path))
                    except Exception:
                        pass
                    self.processed_commands.add(command_path.name)
                    print(f"STALE COMMAND IGNORED | {command_id} created_at={created_at:.3f} controller_start={self.controller_start_time:.3f}")
                    continue
            if command_id in self.tasks:
                continue

            task = self.command_to_task(data)
            if task is None:
                safe_write_json(RESULT_DIR / f"{command_id}.json", {
                    "command_id": command_id,
                    "success": False,
                    "status": "FAILED",
                    "message": "invalid rack/workstation mapping",
                    "finished_at": time.time(),
                })
                self.processed_commands.add(command_path.name)
                continue

            if not self.assign_task_to_amr(task):
                safe_write_json(RESULT_DIR / f"{command_id}.json", {
                    "command_id": command_id,
                    "success": False,
                    "status": "FAILED",
                    "message": "no available AMR",
                    "finished_at": time.time(),
                })
                self.processed_commands.add(command_path.name)
                continue

            self.tasks[command_id] = task
            self.processed_commands.add(command_path.name)
            print(f"COMMAND ACCEPTED | {command_id} mode={task.action_mode} rack={task.rack_key} amr={task.amr_name} target={task.target_xy}")

    def command_to_task(self, data: Dict) -> Optional[CommandTask]:
        command_id = str(data.get("command_id", f"CMD_{int(time.time())}"))
        workstation_id = str(data.get("workstation_id", ""))
        rack_key = self.resolve_rack_key(workstation_id)
        if rack_key is None or rack_key not in self.racks:
            print(f"COMMAND REJECTED | unknown workstation_id={workstation_id}")
            return None

        start_location = str(data.get("start_location", ""))
        target_location = str(data.get("target_location", ""))
        start_location_key = start_location.strip().upper()
        target_location_key = target_location.strip().upper()
        tx = data.get("target_x", None)
        ty = data.get("target_y", None)
        yaw = float(data.get("target_yaw", 0.0) or 0.0)
        preferred_amr_name = str(
            data.get("preferred_amr_name", "")
            or data.get("preferred_amr", "")
            or data.get("amr_name", "")
            or data.get("robot_id", "")
        ).strip()
        require_preferred_amr = bool(data.get("require_preferred_amr", False))
        if preferred_amr_name and preferred_amr_name.upper().startswith("AMR") and not preferred_amr_name.startswith("AMR_"):
            digits = "".join(ch for ch in preferred_amr_name if ch.isdigit())
            if digits:
                preferred_amr_name = f"AMR_{int(digits):02d}"

        rack = self.racks.get(rack_key)
        if rack is None:
            return None

        rotate_by_target = target_location_key in ROTATE_WORKSTATION_TARGETS
        rotate_by_location = any(keyword in start_location_key for keyword in ROTATE_LOCATION_KEYWORDS) or any(keyword in target_location_key for keyword in ROTATE_LOCATION_KEYWORDS)
        action_mode = "ROTATE_WORKSTATION" if (rotate_by_target or rotate_by_location) else "MOVE_WORKSTATION"

        if action_mode == "ROTATE_WORKSTATION":
            target_xy = get_object_center_xy(rack.obj)
            target_cell = rack.cell
            rotate_start_yaw = get_root_yaw_rad(rack.obj.prim)
            rotate_target_yaw = normalize_angle_rad(rotate_start_yaw + ROTATE_WORKSTATION_DELTA_RAD)
            reason = "target_location" if rotate_by_target else "ROTATING_LOCATION"
            print(
                f"ROTATE_WORKSTATION COMMAND | command={command_id} reason={reason} "
                f"start_location={start_location} target_location={target_location} "
                f"rack={rack_key} cell={target_cell} start_yaw={rotate_start_yaw:.3f} target_yaw={rotate_target_yaw:.3f}"
            )
        else:
            if tx is not None and ty is not None and (abs(float(tx)) > 1e-6 or abs(float(ty)) > 1e-6):
                target_xy = (float(tx), float(ty))
            else:
                loc = LOCATION_TARGETS.get(target_location, LOCATION_TARGETS.get("warehouse", (0.0, -5.0, 0.0)))
                target_xy = (float(loc[0]), float(loc[1]))
                yaw = float(loc[2])

            target_cell = world_to_grid(*target_xy)
            if QR_ONLY_NAVIGATION and QR_SNAP_TARGET_TO_NEAREST and self.valid_qr_cells:
                snapped_x, snapped_y, snapped_cell = self.snap_xy_to_nearest_qr(*target_xy)
                snap_dist = distance_xy(target_xy, (snapped_x, snapped_y))
                if snap_dist <= QR_MAX_SNAP_DISTANCE_M:
                    print(f"TARGET SNAP TO QR | command={command_id} raw={target_xy} snapped=({snapped_x:.3f}, {snapped_y:.3f}) cell={snapped_cell} dist={snap_dist:.3f}")
                    target_xy = (snapped_x, snapped_y)
                    target_cell = snapped_cell
                else:
                    print(f"TARGET SNAP WARNING | command={command_id} raw={target_xy} nearest_cell={snapped_cell} dist={snap_dist:.3f}m > {QR_MAX_SNAP_DISTANCE_M}m")
                    target_cell = snapped_cell
                    target_xy = grid_to_world(target_cell)
            if not self.explicit_drop_target_locked(target_location, rack_key, target_cell):
                target_xy, target_cell = self.enforce_drop_target_not_pickup(command_id, rack_key, target_xy, target_cell)
            rotate_start_yaw = 0.0
            rotate_target_yaw = 0.0

        return CommandTask(
            command_id=command_id,
            workstation_id=workstation_id,
            rack_key=rack_key,
            target_location=target_location,
            target_xy=target_xy,
            target_cell=target_cell,
            target_yaw=yaw,
            preferred_amr_name=preferred_amr_name or None,
            require_preferred_amr=require_preferred_amr,
            action_mode=action_mode,
            rotate_start_yaw=rotate_start_yaw,
            rotate_target_yaw=rotate_target_yaw,
        )

    def resolve_rack_key(self, workstation_id: str) -> Optional[str]:
        if workstation_id in self.racks:
            return workstation_id
        if workstation_id.startswith("RACK_"):
            digits = "".join(ch for ch in workstation_id if ch.isdigit())
            if digits:
                key = f"WS_{int(digits):02d}"
                return key if key in self.racks else None
        if workstation_id.startswith("WS_"):
            return workstation_id if workstation_id in self.racks else None
        digits = "".join(ch for ch in workstation_id if ch.isdigit())
        if digits:
            key = f"WS_{int(digits):02d}"
            return key if key in self.racks else None
        return None

    def assign_task_to_amr(self, task: CommandTask) -> bool:
        rack = self.racks.get(task.rack_key)
        if rack is None or rack.assigned or rack.carried_by is not None:
            return False

        available = []
        rack_xy = get_object_center_xy(rack.obj)
        for amr in self.amrs.values():
            if amr.state in ["LIFTING", "ROTATING", "TO_RACK", "TO_TARGET", "PLACING", "RETURN_HOME"] or amr.carrying_rack:
                continue
            dist = distance_xy(get_object_center_xy(amr.obj), rack_xy)
            available.append((dist, amr.name))

        if not available:
            return False

        available.sort()
        available_names = {name for _, name in available}
        preferred = (task.preferred_amr_name or "").strip()
        if PREFERRED_AMR_ASSIGNMENT_ENABLED and preferred:
            if preferred in available_names:
                amr = self.amrs[preferred]
                print(f"PREFERRED AMR ASSIGNED | command={task.command_id} preferred={preferred} rack={task.rack_key}")
            elif task.require_preferred_amr:
                print(
                    f"PREFERRED AMR UNAVAILABLE | command={task.command_id} preferred={preferred} "
                    f"available={sorted(available_names)} require=True"
                )
                return False
            else:
                amr = self.amrs[available[0][1]]
                print(
                    f"PREFERRED AMR FALLBACK | command={task.command_id} preferred={preferred} "
                    f"assigned={amr.name} available={sorted(available_names)}"
                )
        else:
            amr = self.amrs[available[0][1]]
        task.amr_name = amr.name
        task.phase = "TO_RACK"
        task.phase_start = time.time()
        task.return_cell = amr.cell
        task.return_xy = grid_to_world(amr.cell)

        if QR_ONLY_NAVIGATION and self.valid_qr_cells:
            rack.cell = self.nearest_qr_cell_to_world(*get_object_center_xy(rack.obj))

        rack.assigned = True
        amr.state = "TO_RACK"
        amr.task_id = task.command_id
        amr.carrying_rack = None
        amr.target_cell = rack.cell
        amr.target_xy = grid_to_world(rack.cell)
        self.release_random_goals_conflicting_with(rack.cell, owner_name=amr.name, reason="task_pickup_target_reserved")
        if task.action_mode != "ROTATE_WORKSTATION":
            self.release_random_goals_conflicting_with(task.target_cell, owner_name=amr.name, reason="task_drop_target_reserved")
        amr.wait_steps = 0
        amr.no_path_steps = 0
        self.publish_status(task, "PICKING")
        return True

    # --------------------------------------------------------
    # Task state machine
    # --------------------------------------------------------
    def clear_stale_motion_reservation(self, amr: AmrState):
        # Non-moving task phases must not keep a future move_to reservation.
        # This prevents another AMR from waiting on a cell that the current AMR
        # is not physically moving into while LIFTING/ROTATING/PLACING.
        if not amr.is_moving():
            amr.move_from = None
            amr.move_to = None
            amr.move_elapsed = 0.0

    def update_task_phases(self, now: float, dt: float):
        for task in list(self.tasks.values()):
            amr = self.amrs.get(task.amr_name or "")
            rack = self.racks.get(task.rack_key)
            if amr is None or rack is None:
                self.finish_task(task, False, "missing amr or rack")
                continue

            if task.phase == "TO_RACK":
                if amr.cell == rack.cell and not amr.is_moving():
                    task.phase = "LIFTING"
                    task.phase_start = now
                    amr.state = "LIFTING"
                    self.clear_stale_motion_reservation(amr)
                    self.publish_status(task, "PICKING")

            elif task.phase == "LIFTING":
                alpha = min((now - task.phase_start) / LIFT_DURATION_SEC, 1.0)
                ax, ay = get_object_center_xy(amr.obj)

                if RACK_DISABLE_RIGID_BODY_WHILE_CARRIED and not rack.physics_locked:
                    hard_disable_physics_subtree(rack.obj.prim, disable_collision=True)
                    rack.physics_locked = True
                    print(f"RACK HARD ATTACHED | {rack.key} physics/collision removed for carry")

                # Hard root lock: rack root follows AMR root center exactly during lift.
                set_object_center_xy(rack.obj, ax, ay, rack.obj.base_z + LIFT_HEIGHT_M * alpha)
                if alpha >= 1.0:
                    task.rack_attach_dx = 0.0
                    task.rack_attach_dy = 0.0
                    rack.carried_by = amr.name
                    amr.carrying_rack = rack.key
                    if task.action_mode == "ROTATE_WORKSTATION":
                        task.rotate_start_yaw = get_root_yaw_rad(rack.obj.prim)
                        task.rotate_target_yaw = normalize_angle_rad(task.rotate_start_yaw + ROTATE_WORKSTATION_DELTA_RAD)
                        amr.state = "ROTATING"
                        self.clear_stale_motion_reservation(amr)
                        amr.target_cell = None
                        amr.target_xy = None
                        task.phase = "ROTATING"
                        task.phase_start = now
                        self.publish_status(task, "NAVIGATING")
                    else:
                        # Guard again at lift completion. If the drop target is still
                        # the same QR cell as pickup, choose a real carry destination.
                        # v25: explicit SG2 A/B destinations are locked so an A target
                        # is not rewritten to B after the AMR briefly stops at A.
                        if not self.explicit_drop_target_locked(task.target_location, task.rack_key, task.target_cell):
                            task.target_xy, task.target_cell = self.enforce_drop_target_not_pickup(
                                task.command_id, task.rack_key, task.target_xy, task.target_cell
                            )
                        amr.state = "TO_TARGET"
                        amr.target_cell = task.target_cell
                        amr.target_xy = task.target_xy
                        self.release_random_goals_conflicting_with(task.target_cell, owner_name=amr.name, reason="task_drop_target_reserved")
                        task.phase = "TO_TARGET"
                        task.phase_start = now
                        self.publish_status(task, "NAVIGATING")

            elif task.phase == "ROTATING":
                alpha = min((now - task.phase_start) / ROTATE_DURATION_SEC, 1.0)
                ax, ay = get_object_center_xy(amr.obj)
                set_object_center_xy(rack.obj, ax, ay, rack.obj.base_z + LIFT_HEIGHT_M)
                yaw_now = normalize_angle_rad(task.rotate_start_yaw + ROTATE_WORKSTATION_DELTA_RAD * alpha)
                set_yaw_if_possible(rack.obj.prim, yaw_now)
                if alpha >= 1.0:
                    set_yaw_if_possible(rack.obj.prim, task.rotate_target_yaw)
                    task.phase = "PLACING"
                    task.phase_start = now
                    amr.state = "PLACING"
                    self.clear_stale_motion_reservation(amr)
                    self.publish_status(task, "PLACING")

            elif task.phase == "TO_TARGET":
                # v13: if the AMR was temporarily routed to an escape/staging waypoint,
                # reaching that waypoint restores the original drop target instead of placing.
                if amr.target_cell is not None and amr.target_cell != task.target_cell and amr.cell == amr.target_cell and not amr.is_moving():
                    print(f"ESCAPE WAYPOINT DONE | {amr.name} task={task.command_id} at={amr.cell} resume_target={task.target_cell}")
                    amr.target_cell = task.target_cell
                    amr.target_xy = task.target_xy
                    amr.wait_steps = 0
                    amr.no_path_steps = 0
                    continue

                if amr.cell == task.target_cell and not amr.is_moving():
                    task.phase = "PLACING"
                    task.phase_start = now
                    amr.state = "PLACING"
                    self.clear_stale_motion_reservation(amr)
                    self.publish_status(task, "PLACING")

            elif task.phase == "PLACING":
                alpha = min((now - task.phase_start) / PLACE_DURATION_SEC, 1.0)
                ax, ay = get_object_center_xy(amr.obj)
                set_object_center_xy(rack.obj, ax, ay, rack.obj.base_z + LIFT_HEIGHT_M * (1.0 - alpha))
                if alpha >= 1.0:
                    set_object_center_xy(rack.obj, task.target_xy[0], task.target_xy[1], rack.obj.base_z)
                    if abs(task.target_yaw) > 1e-6:
                        set_yaw_if_possible(rack.obj.prim, task.target_yaw)
                    rack.cell = task.target_cell
                    rack.carried_by = None
                    rack.assigned = False
                    # Keep rack physics disabled after placement for demo stability.
                    # The planner still treats placed racks as software obstacles.
                    if not RACK_KEEP_COLLISION_DISABLED_AFTER_PLACE:
                        set_collision_enabled_subtree(rack.obj.prim, True)
                    else:
                        hard_disable_physics_subtree(rack.obj.prim, disable_collision=True)

                    amr.carrying_rack = None
                    self.clear_stale_motion_reservation(amr)

                    # Return-home policy:
                    # Do not finish the command while the AMR is still under/near the dropped workstation.
                    # The AMR returns to the cell where it was when the task was assigned, then the result is published.
                    if task.return_cell is not None and task.return_cell != amr.cell:
                        task.phase = "RETURN_HOME"
                        task.phase_start = now
                        amr.state = "RETURN_HOME"
                        amr.target_cell = task.return_cell
                        amr.target_xy = task.return_xy if task.return_xy is not None else grid_to_world(task.return_cell)
                        amr.wait_steps = 0
                        amr.no_path_steps = 0
                        self.release_random_goals_conflicting_with(task.return_cell, owner_name=amr.name, reason="task_return_home_reserved")
                        print(f"RETURN_HOME START | {amr.name} task={task.command_id} from={amr.cell} return={task.return_cell}")
                        self.publish_status(task, "RETURNING")
                    else:
                        amr.task_id = None
                        amr.state = "IDLE"
                        amr.target_cell = None
                        amr.target_xy = None
                        self.finish_task(task, True, "completed")

            elif task.phase == "RETURN_HOME":
                if amr.cell == task.return_cell and not amr.is_moving():
                    amr.task_id = None
                    amr.state = "IDLE"
                    amr.target_cell = None
                    amr.target_xy = None
                    amr.wait_steps = 0
                    amr.no_path_steps = 0
                    print(f"RETURN_HOME DONE | {amr.name} task={task.command_id} cell={amr.cell}")
                    resume_task_id = task.resume_task_id
                    if task.internal and resume_task_id:
                        resume_task = self.tasks.get(resume_task_id)
                        if resume_task is not None:
                            print(f"RETURN_HOME RESUME DONE | {amr.name} clearance={task.command_id} completed_paused={resume_task_id}")
                            self.finish_task(resume_task, True, "completed_returned_home_after_lane_clearance")
                    self.finish_task(task, True, "completed_returned_home")

            elif task.phase == "RETURN_HOME_PAUSED":
                # Waiting for a preempting internal lane-clearance task to return this AMR home.
                pass

    def publish_status(self, task: CommandTask, status: str):
        if task.internal:
            return
        now = time.time()
        if now - task.last_status_at < STATUS_PERIOD_SEC and status not in ["PICKING", "PLACING", "RETURNING"]:
            return
        task.last_status_at = now
        amr = self.amrs.get(task.amr_name or "")
        dist = 0.0
        if amr is not None:
            if task.phase in ["TO_TARGET", "PLACING"]:
                target = task.target_xy
            elif task.phase == "RETURN_HOME" and task.return_xy is not None:
                target = task.return_xy
            else:
                target = grid_to_world(self.racks[task.rack_key].cell)
            dist = distance_xy(get_xy(amr.obj.prim), target)
        safe_write_json(STATUS_DIR / f"{task.command_id}.json", {
            "command_id": task.command_id,
            "status": status,
            "distance_remaining": float(dist),
            "amr_name": task.amr_name,
            "rack_key": task.rack_key,
            "updated_at": now,
        })

    def finish_task(self, task: CommandTask, success: bool, message: str):
        if not task.internal:
            safe_write_json(RESULT_DIR / f"{task.command_id}.json", {
                "command_id": task.command_id,
                "success": bool(success),
                "status": "COMPLETED" if success else "FAILED",
                "message": message,
                "amr_name": task.amr_name,
                "rack_key": task.rack_key,
                "action_mode": task.action_mode,
                "return_cell": list(task.return_cell) if task.return_cell is not None else None,
                "finished_at": time.time(),
            })
        if task.command_id in self.tasks:
            del self.tasks[task.command_id]
        print(f"TASK RESULT | {task.command_id} success={success} message={message}")

    # --------------------------------------------------------
    # Planning and motion
    # --------------------------------------------------------
    def static_obstacles_for(self, amr: AmrState, allow_assigned_rack: bool) -> Set[GridCell]:
        obstacles: Set[GridCell] = set()

        # Empty AMR can pass under stationary workstations/racks.
        # Rack avoidance is applied only while the AMR is carrying a workstation.
        # Racks currently carried by another AMR are not duplicated here as rack
        # obstacles; the carrier AMR cell is already hard-reserved in the reservation
        # table and must remain the first-priority blocking condition.
        if not amr.carrying_rack:
            return obstacles

        for rack in self.racks.values():
            if rack.carried_by is not None:
                continue
            if allow_assigned_rack and amr.task_id:
                task = self.tasks.get(amr.task_id)
                if task and rack.key == task.rack_key and amr.state == "TO_RACK":
                    continue
            obstacles.add(rack.cell)
        return obstacles


    def congestion_priority_score(self, name: str) -> float:
        """Priority score for dense 5-AMR arbitration.

        Previous policy always put carrying AMRs first. That is safe, but with 5
        simultaneous tasks it can starve empty AMRs that need to reach a rack,
        producing a visual freeze of AMR_01~AMR_03. This score keeps carrying
        priority, but adds bounded aging so a robot that has waited for many
        ticks eventually gets a chance to move if the normal safety gates allow it.
        """
        amr = self.amrs[name]
        score = 0.0
        if amr.state != "RANDOM":
            score += 100.0
        if amr.carrying_rack:
            score += 45.0
        if CONGESTION_AGING_ENABLED:
            waited = min(float(getattr(amr, "wait_steps", 0)), float(CONGESTION_WAIT_PRIORITY_CAP))
            score += waited
            if amr.state == "TO_RACK" and waited >= CONGESTION_TO_RACK_EXTRA_BOOST_AFTER:
                score += float(CONGESTION_TO_RACK_EXTRA_BOOST)
        return score

    def is_manual_first_step_safe(self, name: str, dst: GridCell, allow_current_vacate: Optional[Set[GridCell]] = None) -> bool:
        """Conservative validator for a locally rewritten one-step proposal.

        This is not a replacement for global_move_arbiter().  It only prevents the
        lane-spread heuristic from proposing obviously invalid cells.  The arbiter
        still decides the final approved set with the existing collision rules.
        """
        allow_current_vacate = allow_current_vacate or set()
        amr = self.amrs[name]
        src = amr.cell
        move = normalize_move(dst[0] - src[0], dst[1] - src[1])
        if move == (0, 0):
            return False
        if cell_chebyshev_distance(src, dst) != 1:
            return False
        if QR_ONLY_NAVIGATION and self.valid_qr_cells and dst not in self.valid_qr_cells:
            return False
        if amr.carrying_rack and is_diagonal_move(move):
            return False
        if not amr.carrying_rack and self.must_use_four_way_near_rack(src, dst) and is_diagonal_move(move):
            return False

        # Exact current-cell occupancy is normally not allowed.  The only exception
        # here is when the cell belongs to another AMR in the same spread group and
        # that AMR is also being explicitly moved away by this heuristic; the normal
        # tail-release/arbiter checks will still make the final call.
        for other_name, other in self.amrs.items():
            if other_name == name:
                continue
            if other.cell == dst and dst not in allow_current_vacate:
                return False
            if other.is_moving() and other.move_to == dst:
                return False

        if amr.carrying_rack:
            obstacles = self.static_obstacles_for(amr, allow_assigned_rack=False)
            for c in self.planner.rack_occupied_cells(dst, move, amr.target_cell):
                if c in obstacles and c != amr.target_cell:
                    return False
                # Do not create a rack-footprint overlap with a stationary AMR.  Its
                # own source cell is allowed because the rack is moving with it.
                for other_name, other in self.amrs.items():
                    if other_name == name:
                        continue
                    if other.cell == c and c not in allow_current_vacate:
                        return False
                    if other.is_moving() and other.move_to == c:
                        return False
        return True

    def goal_biased_cardinal_candidates(self, name: str) -> List[GridCell]:
        amr = self.amrs[name]
        if amr.target_cell is None:
            return []
        x, y = amr.cell
        gx, gy = amr.target_cell
        primary: List[Move] = []
        if gx > x:
            primary.append((1, 0))
        elif gx < x:
            primary.append((-1, 0))
        if gy > y:
            primary.append((0, 1))
        elif gy < y:
            primary.append((0, -1))
        # Keep all four cardinal moves as fallback, but prefer goal-reducing moves.
        moves: List[Move] = []
        for m in primary + [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if m not in moves:
                moves.append(m)
        return [(x + dx, y + dy) for dx, dy in moves]

    def is_90_turn_move(self, amr: AmrState, dst: GridCell) -> bool:
        move = normalize_move(dst[0] - amr.cell[0], dst[1] - amr.cell[1])
        if move == (0, 0) or amr.heading == (0, 0):
            return False
        return (amr.heading[0] * move[0] + amr.heading[1] * move[1]) == 0

    def bottleneck_turn_priority_score(self, name: str, dst: GridCell) -> float:
        if not BOTTLENECK_TURN_FIRST_ENABLED:
            return 0.0
        amr = self.amrs[name]
        waited = max(int(getattr(amr, "wait_steps", 0)), int(getattr(amr, "no_path_steps", 0)))
        predicted = self.predicted_bottleneck_for_amr(name, dst)
        if waited < BOTTLENECK_TURN_FIRST_WAIT_THRESHOLD and not predicted:
            return 0.0
        move = normalize_move(dst[0] - amr.cell[0], dst[1] - amr.cell[1])
        if move == (0, 0) or amr.heading == (0, 0):
            return 0.0
        dot = amr.heading[0] * move[0] + amr.heading[1] * move[1]
        if dot == 0:
            pre_bonus = PREEMPTIVE_TURN_PRIORITY_BONUS if predicted else 0.0
            return BOTTLENECK_TURN_PRIORITY_BONUS + pre_bonus + float(min(waited, 120))
        if move == amr.heading:
            return -BOTTLENECK_TURN_PRIORITY_BONUS * 0.25
        if dot < 0:
            return -BOTTLENECK_TURN_PRIORITY_BONUS * 0.50
        return 0.0

    def local_density_score(self, cell: GridCell, ignore_names: Optional[Set[str]] = None) -> int:
        ignore_names = ignore_names or set()
        score = 0
        for other_name, other in self.amrs.items():
            if other_name in ignore_names:
                continue
            if cell_chebyshev_distance(cell, other.cell) <= 1:
                score += 1
            if other.is_moving() and other.move_to is not None and cell_chebyshev_distance(cell, other.move_to) <= 1:
                score += 1
        for rack in self.racks.values():
            if rack.carried_by is None and cell_chebyshev_distance(cell, rack.cell) <= 1:
                score += 1
        return score

    def loaded_neighbor_count(self, name: str, radius: int = 2) -> int:
        amr = self.amrs[name]
        count = 0
        for other_name, other in self.amrs.items():
            if other_name == name:
                continue
            if not other.carrying_rack:
                continue
            if cell_chebyshev_distance(amr.cell, other.cell) <= radius:
                count += 1
            elif other.is_moving() and other.move_to is not None and cell_chebyshev_distance(amr.cell, other.move_to) <= radius:
                count += 1
        return count

    def same_column_loaded_group_size(self, name: str, y_window: int = 2) -> int:
        amr = self.amrs[name]
        x, y = amr.cell
        count = 0
        for other in self.amrs.values():
            if not other.carrying_rack:
                continue
            ox, oy = other.cell
            if ox == x and abs(oy - y) <= y_window:
                count += 1
        return count

    def predicted_bottleneck_for_amr(self, name: str, dst: Optional[GridCell] = None) -> bool:
        if not PREEMPTIVE_TURN_FIRST_ENABLED:
            return False
        amr = self.amrs[name]
        if amr.is_moving():
            return False
        if amr.state not in ["TO_TARGET", "RETURN_HOME", "TO_RACK"]:
            return False
        if PREEMPTIVE_TURN_ONLY_CARRYING and not amr.carrying_rack:
            return False

        loaded_neighbors = self.loaded_neighbor_count(name, radius=2)
        if loaded_neighbors >= PREEMPTIVE_TURN_LOADED_NEIGHBOR_THRESHOLD:
            return True

        if amr.carrying_rack and self.same_column_loaded_group_size(name, y_window=2) >= PREEMPTIVE_TURN_SAME_COLUMN_GROUP_SIZE:
            return True

        density_cell = dst if dst is not None else amr.cell
        if self.local_density_score(density_cell, {name}) >= PREEMPTIVE_TURN_LOCAL_DENSITY_THRESHOLD:
            return True

        return False

    def turn_candidates_for(self, name: str) -> List[GridCell]:
        amr = self.amrs[name]
        hx, hy = amr.heading
        if (hx, hy) == (0, 0):
            return []
        x, y = amr.cell
        left = (-hy, hx)
        right = (hy, -hx)
        candidates = [(x + left[0], y + left[1]), (x + right[0], y + right[1])]
        if amr.target_cell is not None:
            # Prefer the turn that does not badly regress from the goal and that moves
            # into the less dense local area. This keeps the rule general while still
            # matching the observed SG2 jam where the middle AMR should rotate/yield
            # instead of driving straight one more cell.
            candidates.sort(key=lambda c: (
                self.local_density_score(c, {name}),
                cell_manhattan_distance(c, amr.target_cell),
                c[0],
                c[1],
            ))
        return candidates

    def apply_bottleneck_turn_first_overrides(self, proposals: Dict[str, GridCell], path2: Dict[str, Optional[GridCell]]):
        """Prefer a 90-degree turn over straight movement in a real bottleneck.

        This is broader than the specific AMR_01/02/03 lane-spread rule.  It only
        activates after an AMR has accumulated wait/no_path steps, and it only
        rewrites a one-step proposal if a safe 90-degree candidate exists.  The
        global arbiter still performs the final safety decision.
        """
        if not BOTTLENECK_TURN_FIRST_ENABLED:
            return

        changed: List[str] = []
        for name in sorted(proposals.keys()):
            amr = self.amrs[name]
            if amr.is_moving() or amr.state not in ["TO_TARGET", "RETURN_HOME", "TO_RACK"]:
                continue
            waited = max(int(getattr(amr, "wait_steps", 0)), int(getattr(amr, "no_path_steps", 0)))
            dst = proposals.get(name)
            predicted = self.predicted_bottleneck_for_amr(name, dst)
            if waited < BOTTLENECK_TURN_FIRST_WAIT_THRESHOLD and not predicted:
                continue
            if dst is None or dst == amr.cell:
                continue
            move = normalize_move(dst[0] - amr.cell[0], dst[1] - amr.cell[1])
            if amr.heading == (0, 0):
                continue
            # Already turning: keep it.
            if (amr.heading[0] * move[0] + amr.heading[1] * move[1]) == 0:
                continue
            # Do not turn-first when the next straight step is already the final goal.
            if amr.target_cell is not None and dst == amr.target_cell:
                continue

            vacating_cells = {a.cell for a in self.amrs.values() if a.name in proposals}
            for cand in self.turn_candidates_for(name):
                if not self.is_manual_first_step_safe(name, cand, allow_current_vacate=vacating_cells):
                    continue
                if amr.target_cell is not None:
                    # Allow a small detour, but do not select a turn that explodes the
                    # distance to the actual task target.
                    now_d = cell_manhattan_distance(amr.cell, amr.target_cell)
                    cand_d = cell_manhattan_distance(cand, amr.target_cell)
                    max_regression = PREEMPTIVE_TURN_MAX_GOAL_REGRESSION if predicted else 2
                    if cand_d > now_d + max_regression:
                        continue
                proposals[name] = cand
                path2[name] = None
                changed.append(f"{name}:{amr.cell}->{cand}")
                break

        if changed:
            max_wait = max(max(self.amrs[n].wait_steps, self.amrs[n].no_path_steps) for n in proposals.keys())
            if max_wait % BOTTLENECK_TURN_FIRST_LOG_PERIOD == 0 or max_wait <= BOTTLENECK_TURN_FIRST_WAIT_THRESHOLD + 2:
                print("PREEMPTIVE/BOTTLENECK TURN FIRST | " + ", ".join(changed))

    def apply_loaded_lane_spread_overrides(self, proposals: Dict[str, GridCell], path2: Dict[str, Optional[GridCell]]):
        """Rewrite only the first step of a tight loaded queue.

        User-requested behavior for the observed 5-AMR SG2 jam:
        - AMR_01 / center blocker should not go straight one more cell into the lane.
          It should yield left/outward.
        - AMR_02 should continue straight toward the open side.
        - AMR_03 should turn toward its destination instead of waiting behind the
          middle blocker.

        The rule is deliberately narrow: it activates only when at least three
        carrying TO_TARGET AMRs are stopped on the same x-column in adjacent y cells.
        """
        if not LOADED_LANE_SPREAD_ENABLED:
            return

        candidates = []
        for name, amr in self.amrs.items():
            if name not in proposals:
                continue
            if amr.is_moving() or not amr.carrying_rack:
                continue
            if amr.state != "TO_TARGET":
                continue
            if (
                amr.wait_steps < LOADED_LANE_SPREAD_WAIT_THRESHOLD
                and amr.no_path_steps < LOADED_LANE_SPREAD_WAIT_THRESHOLD
                and not self.predicted_bottleneck_for_amr(name, proposals.get(name))
            ):
                continue
            candidates.append((name, amr.cell))

        if len(candidates) < LOADED_LANE_SPREAD_MIN_GROUP_SIZE:
            return

        by_x: Dict[int, List[Tuple[int, str]]] = {}
        for name, (x, y) in candidates:
            by_x.setdefault(x, []).append((y, name))

        for x, items in by_x.items():
            if len(items) < LOADED_LANE_SPREAD_MIN_GROUP_SIZE:
                continue
            items.sort()
            # Find the densest consecutive triple in this column.
            for i in range(len(items) - 2):
                triple = items[i:i + 3]
                ys = [v[0] for v in triple]
                if ys[2] - ys[0] > 2:
                    continue
                lower_name = triple[0][1]
                middle_name = triple[1][1]
                upper_name = triple[2][1]
                lower = self.amrs[lower_name]
                middle = self.amrs[middle_name]
                upper = self.amrs[upper_name]

                # If AMR_01/02/03 are exactly involved, follow the user's intended
                # choreography explicitly: AMR_01 yields left, AMR_02 goes straight
                # to the right/east side, AMR_03 turns/proceeds toward the right/east
                # side.  Otherwise apply the same center-yield/general spread rule.
                desired: Dict[str, List[GridCell]] = {}
                group_names = {lower_name, middle_name, upper_name}
                if {"AMR_01", "AMR_02", "AMR_03"}.issubset(group_names):
                    for n in ["AMR_01", "AMR_02", "AMR_03"]:
                        a = self.amrs[n]
                        ax, ay = a.cell
                        if n == "AMR_01":
                            desired[n] = [(ax - 1, ay), (ax + 1, ay)] + self.goal_biased_cardinal_candidates(n)
                        elif n == "AMR_02":
                            desired[n] = [(ax + 1, ay)] + self.goal_biased_cardinal_candidates(n)
                        elif n == "AMR_03":
                            desired[n] = [(ax + 1, ay)] + self.goal_biased_cardinal_candidates(n)
                else:
                    # Generic fallback: middle AMR yields outward/left if possible;
                    # lower and upper AMRs prefer the SG2/right side when their target
                    # is to the right, otherwise their normal goal-biased cardinal move.
                    mx, my = middle.cell
                    desired[middle_name] = [(mx - 1, my), (mx + 1, my)] + self.goal_biased_cardinal_candidates(middle_name)
                    for n in (lower_name, upper_name):
                        a = self.amrs[n]
                        ax, ay = a.cell
                        if a.target_cell and a.target_cell[0] > ax:
                            desired[n] = [(ax + 1, ay)] + self.goal_biased_cardinal_candidates(n)
                        else:
                            desired[n] = self.goal_biased_cardinal_candidates(n)

                vacating_cells = {self.amrs[n].cell for n in desired.keys()}
                changed: List[str] = []
                for n, cells in desired.items():
                    for dst in cells:
                        if self.is_manual_first_step_safe(n, dst, allow_current_vacate=vacating_cells):
                            if proposals.get(n) != dst:
                                proposals[n] = dst
                                path2[n] = None
                                changed.append(f"{n}:{self.amrs[n].cell}->{dst}")
                            break

                if changed:
                    max_wait = max(self.amrs[n].wait_steps for n in desired.keys())
                    if max_wait % LOADED_LANE_SPREAD_LOG_PERIOD == 0 or max_wait <= LOADED_LANE_SPREAD_WAIT_THRESHOLD + 2:
                        print("LOADED LANE SPREAD | " + ", ".join(changed))
                    return

    def endpoint_clear_for_escape(self, name: str, candidate: GridCell) -> bool:
        amr = self.amrs[name]
        if QR_ONLY_NAVIGATION and self.valid_qr_cells and candidate not in self.valid_qr_cells:
            return False
        if candidate == amr.cell:
            return False
        static_obstacles = self.static_obstacles_for(amr, allow_assigned_rack=False)
        if candidate in static_obstacles:
            return False
        move = normalize_move(candidate[0] - amr.cell[0], candidate[1] - amr.cell[1])
        if amr.carrying_rack and is_diagonal_move(move):
            return False
        # Check endpoint footprint with full side clearance even when candidate is the
        # temporary goal. This avoids the normal final-goal relaxation from allowing
        # an escape waypoint whose carried rack would overlap a stationary rack/AMR.
        footprint_move = move if move != (0, 0) else amr.heading
        for c in self.planner.rack_occupied_cells(candidate, footprint_move, None):
            if c in static_obstacles:
                return False
            for other_name, other in self.amrs.items():
                if other_name == name:
                    continue
                # If another AMR is actually moving away from its current cell, do not
                # treat that old cell as a permanent endpoint blocker; do block its move_to.
                if other.is_moving():
                    if other.move_to == c:
                        return False
                elif other.cell == c:
                    return False
        return True

    def choose_escape_waypoint_for_no_path(self, name: str) -> Optional[GridCell]:
        amr = self.amrs[name]
        if amr.target_cell is None:
            return None
        task = self.tasks.get(amr.task_id or "")
        if task is None:
            return None

        cx, cy = amr.cell
        source: List[GridCell] = []
        if self.valid_qr_cells:
            for c in self.valid_qr_cells:
                if 1 <= cell_manhattan_distance(amr.cell, c) <= NO_PATH_ESCAPE_RADIUS:
                    source.append(c)
        else:
            for x in range(cx - NO_PATH_ESCAPE_RADIUS, cx + NO_PATH_ESCAPE_RADIUS + 1):
                for y in range(cy - NO_PATH_ESCAPE_RADIUS, cy + NO_PATH_ESCAPE_RADIUS + 1):
                    c = (x, y)
                    if 1 <= cell_manhattan_distance(amr.cell, c) <= NO_PATH_ESCAPE_RADIUS:
                        source.append(c)

        reservation = ReservationTable()
        for other in self.amrs.values():
            reservation.reserve_cell(other.cell, self.tick, other.name)
            if other.is_moving() and other.move_from is not None and other.move_to is not None:
                reservation.reserve_cell(other.move_to, self.tick + 1, other.name)
                reservation.reserve_edge(other.move_from, other.move_to, self.tick, other.name)

        static_obstacles = self.static_obstacles_for(amr, allow_assigned_rack=False)
        now_goal_dist = cell_manhattan_distance(amr.cell, task.target_cell)
        candidates: List[Tuple[float, GridCell, int]] = []

        for cell in source:
            if cell == task.target_cell:
                continue
            if not self.endpoint_clear_for_escape(name, cell):
                continue
            req = PlanRequest(
                robot_id=name,
                start=amr.cell,
                goal=cell,
                heading=amr.heading,
                carrying_rack=bool(amr.carrying_rack),
                priority=3,
                waiting_steps=amr.wait_steps,
                allowed_goal_occupied=False,
            )
            result = self.planner.plan(req, reservation, static_obstacles, self.tick)
            if not result.success or len(result.path) < 2:
                continue
            # Prefer cells that reduce density and move outward/left when an SG2-bound
            # carrying AMR is jammed. This directly addresses the AMR_02 case where
            # moving further into the packed +x lane keeps the bottleneck closed.
            density = self.local_density_score(cell, {name})
            path_len = max(1, len(result.path) - 1)
            goal_dist = cell_manhattan_distance(cell, task.target_cell)
            regression = max(0, goal_dist - now_goal_dist)
            outward_bonus = -5.0 if cell[0] < amr.cell[0] else 0.0
            deeper_penalty = 8.0 if cell[0] > amr.cell[0] else 0.0
            same_lane_penalty = 3.0 if cell[1] == amr.cell[1] and cell[0] > amr.cell[0] else 0.0
            score = density * 18.0 + path_len * 3.0 + regression * 2.5 + deeper_penalty + same_lane_penalty + outward_bonus
            candidates.append((score, cell, path_len))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[2], x[1][0], abs(x[1][1] - cy), x[1][1]))
        return candidates[0][1]

    def try_assign_escape_waypoint_for_no_path(self, name: str) -> bool:
        if not NO_PATH_ESCAPE_WAYPOINT_ENABLED:
            return False
        amr = self.amrs[name]
        if amr.state != "TO_TARGET" or not amr.carrying_rack:
            return False
        if amr.no_path_steps < NO_PATH_ESCAPE_THRESHOLD:
            return False
        task = self.tasks.get(amr.task_id or "")
        if task is None or task.phase != "TO_TARGET":
            return False
        # Do not stack escape waypoints. If the AMR is already going to a temporary
        # waypoint, wait until update_task_phases() restores the original target.
        if amr.target_cell != task.target_cell:
            return False

        escape = self.choose_escape_waypoint_for_no_path(name)
        if escape is None:
            if amr.no_path_steps % NO_PATH_ESCAPE_LOG_PERIOD == 0:
                print(f"ESCAPE WAYPOINT WAIT | {name} cell={amr.cell} target={task.target_cell} reason=no_safe_escape")
            return False

        amr.target_cell = escape
        amr.target_xy = grid_to_world(escape)
        amr.wait_steps = 0
        amr.no_path_steps = 0
        print(f"ESCAPE WAYPOINT SET | {name} task={task.command_id} from={amr.cell} escape={escape} final={task.target_cell}")
        return True


    def is_starvation_unlock_candidate(self, name: str) -> bool:
        """True when an AMR has a valid plan/proposal but repeatedly loses arbitration.

        wait_steps rising while no_path_steps stays zero means A* can find a path, but
        the global arbiter keeps rejecting the immediate move.  Give that AMR a short
        local priority window so it can clear the bottleneck instead of being starved
        behind nearby loaded AMRs.
        """
        if not STARVATION_UNLOCK_ENABLED:
            return False
        amr = self.amrs.get(name)
        if amr is None:
            return False
        if amr.is_moving():
            return False
        if amr.state not in ["TO_TARGET", "RETURN_HOME", "TO_RACK"]:
            return False
        if int(getattr(amr, "wait_steps", 0)) < STARVATION_UNLOCK_WAIT_THRESHOLD:
            return False
        # no_path==0 is important: the planner has a route, only the arbiter is blocking it.
        if int(getattr(amr, "no_path_steps", 0)) != 0:
            return False
        return True

    def starvation_priority_score(self, name: str) -> float:
        if not self.is_starvation_unlock_candidate(name):
            return 0.0
        amr = self.amrs[name]
        return STARVATION_UNLOCK_PRIORITY_BONUS + float(min(int(amr.wait_steps), 300))

    def loaded_amr_spacing_conflict(self, name: str, src: GridCell, dst: GridCell, approved: Dict[str, GridCell], proposals: Dict[str, GridCell]) -> bool:
        """Avoid forming tight clusters of rack-carrying AMRs.

        Carrying workstations are visually wider than the grid-center point.  When two
        loaded AMRs finish a tick in adjacent cells, the next arbitration cycle often
        blocks both of them.  This guard prevents new adjacent loaded placements unless
        it is a straight same-direction convoy case or the move is part of an approved
        starvation unlock that increases separation.
        """
        if not LOADED_AMR_MIN_SPACING_ENABLED:
            return False
        amr = self.amrs.get(name)
        if amr is None or not amr.carrying_rack:
            return False

        move = normalize_move(dst[0] - src[0], dst[1] - src[1])
        if move == (0, 0):
            return False

        starved = self.is_starvation_unlock_candidate(name)

        for other_name, other in self.amrs.items():
            if other_name == name or not other.carrying_rack:
                continue

            # Predict the other AMR's next cell as well as possible in this tick.
            if other_name in approved:
                other_next = approved[other_name]
                other_src = other.cell
            elif other_name in proposals:
                other_next = proposals[other_name]
                other_src = other.cell
            elif other.is_moving() and other.move_to is not None:
                other_next = other.move_to
                other_src = other.move_from if other.move_from is not None else other.cell
            else:
                other_next = other.cell
                other_src = other.cell

            new_dist = cell_chebyshev_distance(dst, other_next)
            if new_dist >= LOADED_AMR_MIN_CHEBYSHEV_DISTANCE:
                continue

            old_dist = cell_chebyshev_distance(src, other_src)
            other_move = normalize_move(other_next[0] - other_src[0], other_next[1] - other_src[1])

            # Keep the previously requested straight convoy behavior: loaded AMRs may
            # follow in-line when both are moving the same direction.  This exception
            # does not apply to perpendicular/L-shaped merges.
            if (
                LOADED_AMR_SPACING_ALLOW_STRAIGHT_CONVOY
                and move == other_move
                and is_cardinal_move(move)
                and self.vacated_cell_tail_release_allowed(name, src, dst, other_name, other_src, other_next)
            ):
                continue

            # v16 soft-spacing rule:
            # - Prevent creating a *new* tighter loaded cluster.
            # - If the AMRs are already too close, allow moves that keep or increase separation.
            #   A hard reject here trapped the system in a deadlock because escape moves were
            #   blocked by the same spacing guard that was meant to prevent clustering.
            if old_dist < LOADED_AMR_MIN_CHEBYSHEV_DISTANCE and new_dist >= old_dist:
                continue

            # Starvation unlock is slightly more permissive: if it increases separation, allow it.
            if starved and new_dist > old_dist:
                continue

            return True
        return False

    def format_reject_context(self, name: str, src: GridCell, dst: GridCell, reason: str) -> str:
        amr = self.amrs.get(name)
        if amr is None:
            return f"MOVE REJECT | {name} {src}->{dst} reason={reason}"
        return (
            f"MOVE REJECT | {name} {src}->{dst} reason={reason} "
            f"state={amr.state} carry={amr.carrying_rack} "
            f"wait={amr.wait_steps} no_path={amr.no_path_steps}"
        )

    def plan_and_approve_moves(self):
        if RANDOM_DRIVE_ENABLED and not STANDBY_UNTIL_COMMAND:
            self.resolve_random_clumps()
            self.resolve_duplicate_random_targets()
        elif self.tasks:
            self.resolve_duplicate_random_targets()
        requests: List[PlanRequest] = []
        static_by_robot: Dict[str, Set[GridCell]] = {}

        for amr in self.amrs.values():
            if amr.is_moving() or amr.state in ["LIFTING", "ROTATING", "PLACING"]:
                continue
            if amr.target_cell is None and amr.state == "RANDOM" and RANDOM_DRIVE_ENABLED and not STANDBY_UNTIL_COMMAND:
                self.assign_random_goal(amr)
            if amr.target_cell is None or amr.cell == amr.target_cell:
                if amr.state == "RANDOM" and RANDOM_DRIVE_ENABLED and not STANDBY_UNTIL_COMMAND:
                    self.assign_random_goal(amr)
                continue

            carrying = bool(amr.carrying_rack)
            allowed_goal_occupied = amr.state == "TO_RACK"
            req = PlanRequest(
                robot_id=amr.name,
                start=amr.cell,
                goal=amr.target_cell,
                heading=amr.heading,
                carrying_rack=carrying,
                priority=3 if carrying else 1,
                waiting_steps=amr.wait_steps,
                allowed_goal_occupied=allowed_goal_occupied,
            )
            requests.append(req)
            static_by_robot[amr.name] = self.static_obstacles_for(amr, allow_assigned_rack=allowed_goal_occupied)

        if not requests:
            return

        reservation = ReservationTable()
        # Current positions are hard-reserved.
        for other in self.amrs.values():
            reservation.reserve_cell(other.cell, self.tick, other.name)
            if other.is_moving():
                reservation.reserve_cell(other.move_to, self.tick + 1, other.name)
                reservation.reserve_edge(other.move_from, other.move_to, self.tick, other.name)

        ordered = sorted(requests, key=lambda r: (-self.congestion_priority_score(r.robot_id), r.robot_id))
        proposals: Dict[str, GridCell] = {}
        path2: Dict[str, Optional[GridCell]] = {}

        for req in ordered:
            result = self.planner.plan(req, reservation, static_by_robot.get(req.robot_id, set()), self.tick)
            if result.success and len(result.path) >= 2:
                next_cell = result.path[1]
                proposals[req.robot_id] = next_cell
                path2[req.robot_id] = result.path[2] if len(result.path) >= 3 else None
                # V24: do not reserve the entire planned future path before global approval.
                # Earlier versions reserved every successful path immediately during proposal
                # generation. If a high-priority AMR later failed in the arbiter, its
                # unapproved future path still blocked lower AMRs in the same planning cycle.
                # This produced the observed freeze: AMR_04/AMR_05 could both be carrying,
                # one unapproved path polluted the ReservationTable, and the other AMR got
                # no_path even though its immediate next cell was safe.
                # Current positions and real in-progress move_to cells remain hard-reserved;
                # future path conflicts are resolved by global_move_arbiter/tail-release.
                self.amrs[req.robot_id].no_path_steps = 0
            else:
                self.amrs[req.robot_id].no_path_steps += 1
                self.amrs[req.robot_id].wait_steps += 1
                if self.amrs[req.robot_id].no_path_steps in (20, 50, 100) or (self.amrs[req.robot_id].no_path_steps > 100 and self.amrs[req.robot_id].no_path_steps % 50 == 0):
                    a = self.amrs[req.robot_id]
                    print(f"CONGESTION NO_PATH | {req.robot_id} state={a.state} cell={a.cell} target={a.target_cell} carrying={a.carrying_rack} wait={a.wait_steps} no_path={a.no_path_steps}")
                # v13: carrying AMR no_path recovery.  Do not keep a loaded AMR parked
                # in the dense lane forever; move it to a safe staging/escape QR cell,
                # then restore the original drop target after the waypoint is reached.
                self.try_assign_escape_waypoint_for_no_path(req.robot_id)
                # For RANDOM mode, avoid standing forever by changing goal, not by unsafe forced move.
                if self.amrs[req.robot_id].state == "RANDOM" and RANDOM_DRIVE_ENABLED and not STANDBY_UNTIL_COMMAND and self.amrs[req.robot_id].no_path_steps >= 3:
                    self.assign_random_goal(self.amrs[req.robot_id])

        self.apply_bottleneck_turn_first_overrides(proposals, path2)
        self.apply_loaded_lane_spread_overrides(proposals, path2)

        approved = self.global_move_arbiter(proposals, path2)

        rejected_names = set(proposals.keys()) - set(approved.keys())
        for name in rejected_names:
            amr = self.amrs[name]
            amr.wait_steps += 1
            if amr.wait_steps in (40, 80, 120) or (amr.wait_steps > 120 and amr.wait_steps % 60 == 0):
                print(f"CONGESTION WAIT | {name} state={amr.state} cell={amr.cell} target={amr.target_cell} carrying={amr.carrying_rack} wait={amr.wait_steps} no_path={amr.no_path_steps}")
            if amr.state == "RANDOM" and RANDOM_DRIVE_ENABLED and not STANDBY_UNTIL_COMMAND and amr.wait_steps >= REJECTED_MOVE_GOAL_REASSIGN_STEPS:
                self.assign_dispersed_random_goal(amr, reason="rejected_by_lookahead_or_arbiter")

        for name, next_cell in approved.items():
            amr = self.amrs[name]
            if next_cell == amr.cell:
                amr.wait_steps += 1
                continue
            self.start_cell_move(amr, next_cell)

        self.tick += 1

    def heading_path_from_cells(self, path: List[GridCell], initial_heading: Move) -> List[Move]:
        if not path:
            return []
        headings = [initial_heading]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            headings.append(headings[-1] if (dx, dy) == (0, 0) else normalize_move(dx, dy))
        return headings

    def moving_segment_conflict(self, name: str, src: GridCell, dst: GridCell, approved: Optional[Dict[str, GridCell]] = None) -> bool:
        """Reject only real swept-motion conflicts.

        V21 tail-release fix:
        - Stationary AMRs are handled by current_cell_conflict_unless_tail_release();
          do not use a large segment-radius check against stationary AMRs because it
          creates artificial 1.5~2 cell gaps and makes AMRs stop around LIFTING/PLACING robots.
        - Approved moves in this same planning tick are treated as moving segments, so
          a follower can enter the leader's vacated tail cell when tail-release rules allow it.
        - AMRs already in an in-progress move are also treated as moving segments and can
          be followed into their vacated cell if the rule is safe.
        """
        approved = approved or {}
        src_w = grid_to_world(src)
        dst_w = grid_to_world(dst)
        radius = self.footprint_radius(self.amrs[name])
        move = normalize_move(dst[0] - src[0], dst[1] - src[1])

        for other_name, other in self.amrs.items():
            if other_name == name:
                continue

            if other_name in approved:
                other_src = other.cell
                other_dst = approved[other_name]
            elif other.is_moving() and other.move_from is not None and other.move_to is not None:
                other_src = other.move_from
                other_dst = other.move_to
            else:
                # Stationary AMR: exact current-cell conflict is handled earlier.
                # Do not apply swept-radius checks around a stationary/lifting AMR.
                continue

            other_src_w = grid_to_world(other_src)
            other_dst_w = grid_to_world(other_dst)
            other_radius = self.footprint_radius(other)
            clearance = radius + other_radius + DYNAMIC_SWEPT_COLLISION_MARGIN_M

            if self.vacated_cell_tail_release_allowed(name, src, dst, other_name, other_src, other_dst):
                continue

            if segment_distance(src_w, dst_w, other_src_w, other_dst_w) < clearance:
                return True

            other_move = normalize_move(other_dst[0] - other_src[0], other_dst[1] - other_src[1])
            if DIAGONAL_CARDINAL_CONFLICT_ENABLED and self.diagonal_cardinal_conflict(src, dst, other_src, other_dst, move, other_move):
                return True

        return False

    def diagonal_cardinal_conflict(
        self,
        src: GridCell,
        dst: GridCell,
        other_src: GridCell,
        other_dst: GridCell,
        move: Move,
        other_move: Move,
    ) -> bool:
        """Extra grid-topology guard for diagonal-vs-cardinal visual collisions.

        A line-segment test alone can miss visually large imported AMR bodies when one robot
        cuts diagonally through a 2x2 block while another moves cardinally along one edge of
        the same block. This rejects mixed diagonal/cardinal moves sharing the same local 2x2
        neighborhood or adjacent swept cells.
        """
        this_diag = is_diagonal_move(move)
        other_diag = is_diagonal_move(other_move)
        this_card = is_cardinal_move(move)
        other_card = is_cardinal_move(other_move)

        if not ((this_diag and other_card) or (this_card and other_diag)):
            return False

        def swept_cells(a: GridCell, b: GridCell, m: Move) -> Set[GridCell]:
            cells = {a, b}
            if is_diagonal_move(m):
                cells.add((a[0] + m[0], a[1]))
                cells.add((a[0], a[1] + m[1]))
            return cells

        s1 = swept_cells(src, dst, move)
        s2 = swept_cells(other_src, other_dst, other_move)

        # Same endpoint, adjacent corner, or edge of diagonal square is unsafe for visual AMR bodies.
        if s1 & s2:
            return True

        # If all swept cells are within one Chebyshev step, treat it as the same local crossing zone.
        for c1 in s1:
            for c2 in s2:
                if cell_chebyshev_distance(c1, c2) <= 1:
                    return True

        return False

    def global_move_arbiter(self, proposals: Dict[str, GridCell], path2: Optional[Dict[str, Optional[GridCell]]] = None) -> Dict[str, GridCell]:
        approved: Dict[str, GridCell] = {}
        approved_second: Dict[str, Optional[GridCell]] = {}
        target_to_robot: Dict[GridCell, str] = {}
        reject_reasons: Dict[str, str] = {}
        path2 = path2 or {}

        def reject(name: str, reason: str) -> None:
            reject_reasons[name] = reason

        # Deterministic priority: starvation-unlock AMRs first, then turn-first,
        # congestion aging and task/carrying priority.  Tail-release dependencies can
        # still move a leader before a follower when the follower wants the leader's cell.
        base_order = sorted(
            proposals.keys(),
            key=lambda n: (
                -self.starvation_priority_score(n),
                -self.bottleneck_turn_priority_score(n, proposals[n]),
                -self.congestion_priority_score(n),
                n,
            ),
        )
        ordered_names = self.order_for_tail_release(base_order, proposals)

        starved_names = [n for n in ordered_names if self.is_starvation_unlock_candidate(n)]
        if starved_names:
            msg = []
            for n in starved_names:
                a = self.amrs[n]
                msg.append(f"{n}@{a.cell}-> {proposals.get(n)} wait={a.wait_steps}")
            # Log sparsely to avoid flooding Script Editor.
            if any(self.amrs[n].wait_steps % STARVATION_UNLOCK_LOG_PERIOD == 0 for n in starved_names):
                print("STARVATION UNLOCK PRIORITY | " + "; ".join(msg))

        for name in ordered_names:
            amr = self.amrs[name]
            src = amr.cell
            dst = proposals[name]
            if dst == src:
                reject(name, "same_cell_wait")
                continue
            if QR_ONLY_NAVIGATION and self.valid_qr_cells and dst not in self.valid_qr_cells:
                reject(name, "dst_not_qr")
                continue
            move = normalize_move(dst[0] - src[0], dst[1] - src[1])
            if amr.carrying_rack and self.must_use_four_way_near_rack(src, dst) and is_diagonal_move(move):
                reject(name, "loaded_diagonal_or_near_rack_diagonal")
                continue
            if self.current_cell_conflict_unless_tail_release(name, src, dst, approved):
                reject(name, "current_cell_conflict_or_tail_release_denied")
                continue
            if dst in target_to_robot:
                reject(name, f"duplicate_target_with_{target_to_robot[dst]}")
                continue
            if self.edge_swap_conflict(name, src, dst, approved):
                reject(name, "edge_swap_conflict")
                continue
            if self.diagonal_cross_conflict(name, src, dst, approved):
                reject(name, "diagonal_cross_conflict")
                continue
            if self.loaded_amr_spacing_conflict(name, src, dst, approved, proposals):
                reject(name, "loaded_amr_min_spacing_conflict")
                continue
            if self.footprint_conflict(name, src, dst, approved):
                reject(name, "footprint_swept_conflict")
                continue
            if self.moving_segment_conflict(name, src, dst, approved):
                reject(name, "moving_segment_or_swept_conflict")
                continue
            # Mode C soft lookahead policy:
            # The immediate next-cell safety checks above are hard blockers.
            # The second planned cell is only a caution/replanning hint, so it must not
            # reject the current one-cell move. This prevents unnecessary waiting when
            # the next cell is safe but the cell after that is currently blocked.
            if LOOKAHEAD2_ACTIVE_ARBITEER:
                self.lookahead2_conflict(name, src, dst, path2.get(name), approved, approved_second)
            target_to_robot[dst] = name
            approved[name] = dst
            approved_second[name] = path2.get(name)

        self.last_move_reject_reasons = reject_reasons
        if ARB_REJECT_REASON_LOG_ENABLED:
            for name, reason in reject_reasons.items():
                amr = self.amrs.get(name)
                if amr is None:
                    continue
                if (
                    self.is_starvation_unlock_candidate(name)
                    or amr.wait_steps in (10, 20, 40, 80, 120)
                    or (amr.wait_steps > 120 and amr.wait_steps % ARB_REJECT_REASON_LOG_PERIOD == 0)
                ):
                    print(self.format_reject_context(name, amr.cell, proposals.get(name, amr.cell), reason))
        return approved

    def lookahead2_conflict(
        self,
        name: str,
        src: GridCell,
        dst: GridCell,
        second: Optional[GridCell],
        approved: Dict[str, GridCell],
        approved_second: Dict[str, Optional[GridCell]],
    ) -> bool:
        if second is None or second == dst:
            return False

        amr = self.amrs[name]
        move2 = normalize_move(second[0] - dst[0], second[1] - dst[1])

        # Soft lookahead policy:
        # The checks below intentionally do not block the current move. The hard
        # safety decision already happened for dst in global_move_arbiter() and in A*.
        # This method is kept as a centralized second-cell risk probe for debugging,
        # future speed control, or replanning hints.
        if QR_ONLY_NAVIGATION and self.valid_qr_cells and second not in self.valid_qr_cells:
            return True
        if self.hard_amr_cell_conflict(name, second):
            return True

        # Carrying AMR policy: Mode C two-step approval.
        # A* already checked the immediate next-cell footprint. Here we additionally
        # check the second planned cell with the same carried-rack 4-neighbor footprint.
        # If the second cell is the final goal, rack_occupied_cells() returns only the
        # goal center cell, so no lookahead beyond the final destination is required.
        if amr.carrying_rack:
            if self.must_use_four_way_near_rack(dst, second) and is_diagonal_move(move2):
                return True
            if not self.carrying_footprint_clear_for_second_step(name, second, move2):
                return True
        else:
            # Non-carrying AMRs are allowed to move diagonally and pass under stationary
            # racks/workstations. Keep only AMR-to-AMR safety checks here.
            pass

        # Avoid entering a near-future occupied or reserved visual zone.
        for other_name, other_dst in approved.items():
            other = self.amrs[other_name]
            other_src = other.cell
            other_second = approved_second.get(other_name)

            if second == other_dst or second == other_src:
                return True
            if cell_chebyshev_distance(second, other_dst) < LOOKAHEAD2_MIN_CLEARANCE_CELLS:
                return True
            if other_second is not None:
                if second == other_second:
                    return True
                # Two-step edge swap: this AMR's next->second conflicts with other next->other_second.
                if dst == other_second and second == other_dst:
                    return True
                if self.diagonal_cardinal_conflict(dst, second, other_dst, other_second, move2, normalize_move(other_second[0] - other_dst[0], other_second[1] - other_dst[1])):
                    return True
                if segment_distance(grid_to_world(dst), grid_to_world(second), grid_to_world(other_dst), grid_to_world(other_second)) < (self.footprint_radius(self.amrs[name]) + self.footprint_radius(other) + DYNAMIC_SWEPT_COLLISION_MARGIN_M):
                    return True

        # Also avoid planning second step through currently moving AMR segments.
        for other_name, other in self.amrs.items():
            if other_name == name:
                continue
            if other.is_moving() and other.move_from is not None and other.move_to is not None:
                if segment_distance(grid_to_world(dst), grid_to_world(second), grid_to_world(other.move_from), grid_to_world(other.move_to)) < (self.footprint_radius(self.amrs[name]) + self.footprint_radius(other) + DYNAMIC_SWEPT_COLLISION_MARGIN_M):
                    return True
        return False

    def hard_amr_cell_conflict(self, name: str, cell: GridCell) -> bool:
        """Return True when a cell is physically occupied by another AMR now or by
        another AMR that is actually moving into the cell. Stale move_to values from
        non-moving LIFTING/PLACING/ROTATING states are intentionally ignored.
        """
        for other_name, other in self.amrs.items():
            if other_name == name:
                continue
            if other.cell == cell:
                return True
            if other.is_moving() and other.move_to == cell:
                return True
        return False

    def carrying_footprint_clear_for_second_step(self, name: str, center: GridCell, move: Move) -> bool:
        """Check carried-rack footprint for the second planned cell in Mode C.

        Center cell must be QR-valid and free from rack obstacles. Adjacent footprint
        cells are physical clearance cells: they may be non-QR cells, but they must not
        overlap stationary rack obstacles or AMR hard-occupied cells. At the final goal,
        rack_occupied_cells() returns only the goal center cell, preventing the old
        failure where the planner required a non-existent cell beyond the destination.
        """
        amr = self.amrs[name]
        goal = amr.target_cell
        static_obstacles = self.static_obstacles_for(amr, allow_assigned_rack=False)
        occupied_cells = self.planner.rack_occupied_cells(center, move, goal)

        for c in occupied_cells:
            if c == center:
                if QR_ONLY_NAVIGATION and self.valid_qr_cells and c not in self.valid_qr_cells:
                    return False
                if c in static_obstacles and c != goal:
                    return False
            else:
                if c in static_obstacles:
                    return False
            if self.hard_amr_cell_conflict(name, c):
                return False
        return True

    def must_use_four_way_near_rack(self, src: GridCell, dst: GridCell) -> bool:
        if not RACK_FOUR_WAY_ZONE_ENABLED or not self.rack_four_way_cells:
            return False
        move = normalize_move(dst[0] - src[0], dst[1] - src[1])
        if move == (0, 0):
            return False
        if not is_diagonal_move(move):
            return False
        c1 = (src[0] + move[0], src[1])
        c2 = (src[0], src[1] + move[1])
        return (
            src in self.rack_four_way_cells
            or dst in self.rack_four_way_cells
            or c1 in self.rack_four_way_cells
            or c2 in self.rack_four_way_cells
        )

    def edge_swap_conflict(self, name: str, src: GridCell, dst: GridCell, approved: Dict[str, GridCell]) -> bool:
        for other_name, other_dst in approved.items():
            other_src = self.amrs[other_name].cell
            if other_src == dst and other_dst == src:
                return True
        return False

    def diagonal_cross_conflict(self, name: str, src: GridCell, dst: GridCell, approved: Dict[str, GridCell]) -> bool:
        move = (dst[0] - src[0], dst[1] - src[1])
        if abs(move[0]) != 1 or abs(move[1]) != 1:
            return False
        for other_name, other_dst in approved.items():
            other_src = self.amrs[other_name].cell
            if self.vacated_cell_tail_release_allowed(name, src, dst, other_name, other_src, other_dst):
                continue
            om = (other_dst[0] - other_src[0], other_dst[1] - other_src[1])
            # Crossing diagonals of the same square.
            if abs(om[0]) == 1 and abs(om[1]) == 1:
                if src == (other_dst[0], other_src[1]) and dst == (other_src[0], other_dst[1]):
                    return True

            # v11: mixed diagonal/cardinal conflict in the same local crossing zone.
            if DIAGONAL_CARDINAL_CONFLICT_ENABLED and self.diagonal_cardinal_conflict(src, dst, other_src, other_dst, move, om):
                return True
        return False

    def order_for_tail_release(self, base_order: List[str], proposals: Dict[str, GridCell]) -> List[str]:
        """Topologically order one-step proposals so a cell owner is approved
        before a follower enters that owner's currently occupied cell.

        Example straight convoy A-B-C-D:
            AMR_2 at B proposes B->C
            AMR_1 at A proposes A->B
        AMR_2 must be evaluated first; then AMR_1 can be approved into the
        tail cell B that AMR_2 is vacating in the same tick.

        Example turn release:
            AMR_1 at A proposes A->B
            AMR_2 at D proposes D->A
        AMR_1 is evaluated first; then AMR_2 can enter A after it is vacated.
        """
        if not ALLOW_VACATED_CELL_TAIL_RELEASE:
            return base_order

        base_index = {name: i for i, name in enumerate(base_order)}
        by_current = {amr.cell: name for name, amr in self.amrs.items()}
        deps: Dict[str, Set[str]] = {name: set() for name in base_order}

        for name in base_order:
            src = self.amrs[name].cell
            dst = proposals.get(name)
            if dst is None or dst == src:
                continue
            owner = by_current.get(dst)
            if owner is None or owner == name or owner not in proposals:
                continue
            owner_dst = proposals.get(owner)
            if owner_dst is None or owner_dst == self.amrs[owner].cell:
                continue
            # Edge swap is not tail release; keep it blocked by the normal edge-swap guard.
            if owner_dst == src:
                continue
            if self.vacated_cell_tail_release_allowed(name, src, dst, owner, self.amrs[owner].cell, owner_dst):
                deps[name].add(owner)

        ordered: List[str] = []
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(n: str):
            if n in visited:
                return
            if n in visiting:
                return
            visiting.add(n)
            for dep in sorted(deps.get(n, set()), key=lambda x: base_index.get(x, 9999)):
                visit(dep)
            visiting.remove(n)
            visited.add(n)
            ordered.append(n)

        for n in base_order:
            visit(n)
        return ordered

    def current_cell_conflict_unless_tail_release(self, name: str, src: GridCell, dst: GridCell, approved: Dict[str, GridCell]) -> bool:
        """Current AMR cells are hard blocks except for a verified tail release.

        V21 extension:
        1) If the occupant was already approved in this same tick to leave dst, a follower
           may enter dst if tail-release safety passes.
        2) If the occupant is already physically moving away from dst, a follower may also
           enter dst on the next decision tick if tail-release safety passes.
        3) LIFTING/PLACING/ROTATING stationary AMRs remain hard blocks only at their
           actual current cell; stale future move_to is ignored elsewhere.
        """
        for other_name, other in self.amrs.items():
            if other_name == name:
                continue
            if other.cell != dst:
                continue

            # Case 1: the owner has already been approved to leave this cell in this tick.
            other_dst = approved.get(other_name)
            if other_dst is not None:
                if other_dst == other.cell or other_dst == src:
                    return True
                return not self.vacated_cell_tail_release_allowed(name, src, dst, other_name, other.cell, other_dst)

            # Case 2: the owner is already in a physical move away from this cell.
            if other.is_moving() and other.move_from is not None and other.move_to is not None:
                if other.move_from == dst and other.move_to != src:
                    return not self.vacated_cell_tail_release_allowed(name, src, dst, other_name, other.move_from, other.move_to)

            return True
        return False

    def vacated_cell_tail_release_allowed(self, name: str, src: GridCell, dst: GridCell, other_name: str, other_src: GridCell, other_dst: GridCell) -> bool:
        """Allow a trailing AMR to enter a cell that another AMR is vacating in
        the same tick.

        This covers two user-requested cases:
        1) Straight convoy: A->B while B->C in the same direction.
        2) Turn/tail release: D->A while A->B, after A is vacated.

        The rule never allows edge swaps, never allows diagonal tail-release, and
        keeps loaded perpendicular tail-release conservative.  Straight loaded
        convoy can be enabled for demo density because both AMRs move with the
        same velocity along the same line.
        """
        if not ALLOW_VACATED_CELL_TAIL_RELEASE:
            return False
        if dst != other_src:
            return False
        if other_dst == src:
            return False

        move = normalize_move(dst[0] - src[0], dst[1] - src[1])
        other_move = normalize_move(other_dst[0] - other_src[0], other_dst[1] - other_src[1])
        if move == (0, 0) or other_move == (0, 0):
            return False
        # Tail-release must be grid-cardinal.  Diagonal entry into a vacated cell
        # can visually clip the leaving AMR or rack at the corner.
        if is_diagonal_move(move) or is_diagonal_move(other_move):
            return False

        same_direction = move == other_move
        this_amr = self.amrs[name]
        other_amr = self.amrs[other_name]

        if same_direction and ALLOW_SAME_DIRECTION_CONVOY_FOLLOWING:
            if this_amr.carrying_rack and other_amr.carrying_rack and not ALLOW_LOADED_STRAIGHT_TIGHT_CONVOY:
                return False
            return True

        # Non-straight / perpendicular tail release.
        # v25 collision fix: if either side is carrying a rack, do not allow
        # same-tick entry into the vacated cell. In an L-shaped move the loaded
        # rack sweeps a wider corner area, so the follower must wait until the
        # loaded AMR actually completes the move.
        if this_amr.carrying_rack or other_amr.carrying_rack:
            if not ALLOW_LOADED_PERPENDICULAR_TAIL_RELEASE:
                return False

        # Empty AMR L-turn tail-release is still allowed because the footprint is
        # much smaller and the user-requested ABC/D flow remains useful.
        return bool(ALLOW_EMPTY_TURN_TAIL_RELEASE)

    def same_direction_convoy_following_allowed(self, name: str, src: GridCell, dst: GridCell, other_name: str, other_src: GridCell, other_dst: GridCell) -> bool:
        """Backward-compatible wrapper for older call sites."""
        move = normalize_move(dst[0] - src[0], dst[1] - src[1])
        other_move = normalize_move(other_dst[0] - other_src[0], other_dst[1] - other_src[1])
        return move == other_move and self.vacated_cell_tail_release_allowed(name, src, dst, other_name, other_src, other_dst)

    def footprint_conflict(self, name: str, src: GridCell, dst: GridCell, approved: Dict[str, GridCell]) -> bool:
        src_w = grid_to_world(src)
        dst_w = grid_to_world(dst)
        radius = self.footprint_radius(self.amrs[name])
        for other_name, other_dst in approved.items():
            other_src = self.amrs[other_name].cell
            other_src_w = grid_to_world(other_src)
            other_dst_w = grid_to_world(other_dst)
            other_radius = self.footprint_radius(self.amrs[other_name])
            if self.vacated_cell_tail_release_allowed(name, src, dst, other_name, other_src, other_dst):
                continue

            min_dist = segment_distance(src_w, dst_w, other_src_w, other_dst_w)
            if min_dist < (radius + other_radius):
                return True
        return False

    def footprint_radius(self, amr: AmrState) -> float:
        base = RACK_SIZE_M if amr.carrying_rack else AMR_SIZE_M
        return base * 0.5 + SAFETY_MARGIN_M

    def start_cell_move(self, amr: AmrState, next_cell: GridCell):
        amr.move_from = amr.cell
        amr.move_to = next_cell
        amr.move_elapsed = 0.0
        dx = next_cell[0] - amr.cell[0]
        dy = next_cell[1] - amr.cell[1]
        if max(abs(dx), abs(dy)) > 1:
            print(f"MOVE REJECTED | {amr.name} non_adjacent_move from={amr.cell} to={next_cell}")
            amr.move_from = None
            amr.move_to = None
            amr.move_elapsed = 0.0
            amr.wait_steps += 1
            return
        if dx != 0 or dy != 0:
            amr.heading = normalize_move(dx, dy)
            set_amr_yaw_to_movement(amr)

    def update_motion(self, dt: float):
        for amr in self.amrs.values():
            if not amr.is_moving():
                continue
            src_w = grid_to_world(amr.move_from)
            dst_w = grid_to_world(amr.move_to)
            dist = max(distance_xy(src_w, dst_w), 1e-6)
            speed = AMR_CARRY_SPEED_MPS if amr.carrying_rack else AMR_SPEED_MPS
            duration = dist / speed
            amr.move_elapsed += dt
            alpha = min(amr.move_elapsed / duration, 1.0)
            x = src_w[0] * (1.0 - alpha) + dst_w[0] * alpha
            y = src_w[1] * (1.0 - alpha) + dst_w[1] * alpha
            set_object_center_xy(amr.obj, x, y, amr.obj.base_z)
            set_amr_yaw_to_movement(amr)

            if amr.carrying_rack:
                rack = self.racks.get(amr.carrying_rack)
                if rack:
                    # Hard transform lock: rack root follows AMR root every frame.
                    # Re-disable physics defensively because Live/PhysX can recompose APIs from referenced layers.
                    if not rack.physics_locked:
                        hard_disable_physics_subtree(rack.obj.prim, disable_collision=True)
                        rack.physics_locked = True
                    set_object_center_xy(rack.obj, x, y, rack.obj.base_z + LIFT_HEIGHT_M)
                    rack.cell = amr.move_to if alpha >= 1.0 else amr.cell

            if alpha >= 1.0:
                amr.cell = amr.move_to
                amr.move_from = None
                amr.move_to = None
                amr.move_elapsed = 0.0
                amr.wait_steps = 0
                # Keep the discrete planner cell deterministic.
                # QR-camera localization can read a forward/neighbor QR after a move and
                # overwrite amr.cell with a non-adjacent cell, which makes the next
                # scripted move visually jump over two cells. For this scripted demo
                # controller, cell state follows the commanded grid motion.
                amr.current_qr_id = cell_to_floor_qr_id_from_map(amr.cell, self.qr_cell_world_map)
                amr.localization_source = "SCRIPTED_GRID_MOTION"
                if amr.state == "RANDOM" and amr.target_cell == amr.cell and RANDOM_DRIVE_ENABLED and not STANDBY_UNTIL_COMMAND:
                    self.assign_random_goal(amr)
                elif amr.state == "RANDOM" and (not RANDOM_DRIVE_ENABLED or STANDBY_UNTIL_COMMAND) and amr.target_cell == amr.cell:
                    amr.state = "IDLE"
                    amr.target_cell = None
                    amr.target_xy = None

    # --------------------------------------------------------
    # Collision check and logging
    # --------------------------------------------------------
    def detect_collisions(self):
        # Current-cell collision.
        occupied: Dict[GridCell, str] = {}
        for amr in self.amrs.values():
            if amr.cell in occupied:
                self.log_collision("CELL", f"{occupied[amr.cell]} and {amr.name} at {amr.cell}")
            else:
                occupied[amr.cell] = amr.name

        # Approximate world footprint collision.
        names = list(self.amrs.keys())
        for i in range(len(names)):
            a = self.amrs[names[i]]
            aw = get_object_center_xy(a.obj)
            ar = self.footprint_radius(a)
            for j in range(i + 1, len(names)):
                b = self.amrs[names[j]]
                bw = get_object_center_xy(b.obj)
                br = self.footprint_radius(b)
                if distance_xy(aw, bw) < ar + br:
                    self.log_collision("FOOTPRINT", f"{a.name} and {b.name}")

    def log_collision(self, typ: str, msg: str):
        self.collision_counts[typ] = self.collision_counts.get(typ, 0) + 1
        total = sum(self.collision_counts.values())
        print(f"COLLISION DETECTED | type={typ} total={total} | {msg}")

    def print_summary(self):
        active_tasks = len(self.tasks)
        moving = sum(1 for a in self.amrs.values() if a.is_moving())
        carrying = sum(1 for a in self.amrs.values() if a.carrying_rack)
        print(
            f"LIVE_TRUE8 | tick={self.tick} moving={moving} active_tasks={active_tasks} "
            f"carrying={carrying} collisions={self.collision_counts} "
            f"qr={[a.current_qr_id or a.localization_source for a in self.amrs.values()]}"
        )


# ============================================================
# Geometry helper for swept footprint
# ============================================================
def segment_distance(a0, a1, b0, b1) -> float:
    # 2D line segment minimum distance.
    if segments_intersect(a0, a1, b0, b1):
        return 0.0
    return min(
        point_segment_distance(a0, b0, b1),
        point_segment_distance(a1, b0, b1),
        point_segment_distance(b0, a0, a1),
        point_segment_distance(b1, a0, a1),
    )


def point_segment_distance(p, a, b) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    qx = ax + t * vx
    qy = ay + t * vy
    return math.hypot(px - qx, py - qy)


def segments_intersect(a, b, c, d) -> bool:
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


# ============================================================
# Start / replace old controller
# ============================================================
def start_controller():
    global _LIVE_TRUE8_CONTROLLER
    try:
        old = globals().get("_LIVE_TRUE8_CONTROLLER")
        if old is not None and getattr(old, "_subscription", None) is not None:
            old._subscription = None
            print("Previous controller subscription released")
    except Exception as e:
        print("Previous controller cleanup warning:", e)
    _LIVE_TRUE8_CONTROLLER = ExistingStageTrue8Controller()
    return _LIVE_TRUE8_CONTROLLER


start_controller()
