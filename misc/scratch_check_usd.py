from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, UsdShade
stage = Usd.Stage.Open("/home/rokey/dev_ws/assets/ffw_description/usd/ffw_sh5_follower_custom.usd")
bound = False
for prim in stage.Traverse():
    if UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath():
        print(f"Bound: {prim.GetPath()} -> {UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath()}")
        bound = True
if not bound:
    print("NO MATERIALS BOUND!")
for prim in stage.Traverse():
    if prim.GetTypeName() == "Camera":
        print(f"Found camera: {prim.GetPath()}")
simulation_app.close()
