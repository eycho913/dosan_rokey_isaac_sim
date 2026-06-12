from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app

import gymnasium as gym
import robotis_lab.simulation_tasks.manager_based.SH5.pick_place_rack
env_cfg = gym.envs.registration.registry['RobotisLab-PickPlace-SH5-Rack-IK-Rel-v0'].kwargs['env_cfg_entry_point']
print("Environment registered! Entry point:", env_cfg)

simulation_app.close()
