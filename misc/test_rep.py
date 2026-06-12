import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app_launcher = AppLauncher(parser.parse_args())

import isaaclab.sim as sim_utils
import omni.replicator.core as rep

sim_cfg = sim_utils.SimulationCfg(device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)
sim.reset()

cam = rep.create.camera(position=(0, 0, 1), look_at=(0,0,0))
render_product = rep.create.render_product(cam, (160, 120))
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach([render_product])

for i in range(5):
    sim.step(render=True)
    data = rgb_annot.get_data()
    print(f"Frame {i}, data shape: {data.shape if data is not None else 'None'}")

app_launcher.app.close()
