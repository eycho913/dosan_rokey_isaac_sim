import sys
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni.graph.core as og
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.api.world import World
from isaacsim.core.prims.geometry_prim import GeometryPrim

enable_extension("omni.isaac.ros2_bridge")

world = World()

import isaacsim.core.utils.prims as prim_utils
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
