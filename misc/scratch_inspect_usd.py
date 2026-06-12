from pxr import Usd, UsdGeom
stage = Usd.Stage.Open("/home/rokey/dev_ws/assets/ffw_description/usd/ffw_sh5_follower.usd")
for prim in stage.Traverse():
    if prim.GetTypeName() in ["Mesh", "Xform", "Camera"]:
        print(f"{prim.GetTypeName()}: {prim.GetPath()}")
