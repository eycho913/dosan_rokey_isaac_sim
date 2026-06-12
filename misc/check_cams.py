import os
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)

import isaaclab.sim as sim_utils
import omni.usd

sim_cfg = sim_utils.SimulationCfg(device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)

# Spawn the robot USD
usd_path = "/home/rokey/dev_ws/robotis_lab/assets/robots/sh5/ffw_sh5_v2/sh5_v2.usd"
cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
cfg.func("/World/envs/env_0/Robot", cfg)

sim.reset()

stage = omni.usd.get_context().get_stage()
cams = [prim.GetPath() for prim in stage.Traverse() if prim.GetTypeName() == "Camera"]
print("FOUND CAMERAS:", cams)

app_launcher.app.close()
