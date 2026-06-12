import sys
from pathlib import Path
ROBOTIS_LAB_DIR = Path("/home/rokey/dev_ws/robotis_lab/scripts/sim2real/bringup")
if str(ROBOTIS_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOTIS_LAB_DIR))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({})
simulation_app = app_launcher.app

import omni.replicator.core as rep
from isaaclab.sensors import Camera, CameraCfg
print("Imports successful!")
simulation_app.close()
