import sys
import omni
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

enable_extension("omni.isaac.ros2_bridge")

world = World()
cube = DynamicCuboid(prim_path="/World/Cube", name="cube", position=[0, 0, 0], scale=[1, 1, 1])

import omni.isaac.core.utils.prims as prim_utils
prim_utils.create_prim("/World/Camera", "Camera")

og.Controller.edit(
    {"graph_path": "/World/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("CreateRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
            ("ROS2CameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
            ("CreateRenderProduct.outputs:execOut", "ROS2CameraHelper.inputs:execIn"),
            ("CreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelper.inputs:renderProductPath"),
        ],
        og.Controller.Keys.SET_VALUES: [
            ("ROS2CameraHelper.inputs:topicName", "/test_camera"),
            ("ROS2CameraHelper.inputs:type", "rgb"),
        ],
    },
)
og.Controller.attribute("/World/ActionGraph/CreateRenderProduct.inputs:cameraPrim").set([og.SubGraph.Target("/World/Camera")])

world.reset()
for i in range(100):
    world.step(render=True)

simulation_app.close()
