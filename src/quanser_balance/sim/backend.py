from __future__ import annotations
import mujoco
import numpy as np
from importlib.resources import files

def load_model():
    xml = files("quanser_balance.sim.assets").joinpath("cartpole.xml")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    return model, data

def step_free(model, data, steps=1000):
    for _ in range(steps):
        data.ctrl[:] = 0.0   # zero actions
        mujoco.mj_step(model, data)

model, data = load_model()
for _ in range(1000):
    data.ctrl[:] = 0.0
    mujoco.mj_step(model, data)