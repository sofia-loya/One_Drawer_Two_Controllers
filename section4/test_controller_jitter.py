import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "section2"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "section3"))

import numpy as np
import mujoco
from drawer_controller import (
    MODEL_PATH, TaskState, ARM_ACTUATORS, arm_addresses,
    object_id, desired_position, resolved_rate_target, TIME_LIMIT
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, object_id(model, mujoco.mjtObj.mjOBJ_KEY, "home"))

# Same jitter range as drawer_env.reset()
cabinet_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "cabinet")
cabinet_home = model.body_pos[cabinet_id].copy()
rng = np.random.default_rng(0)
jitter = rng.uniform([-0.02, -0.04, 0.0], [0.02, 0.04, 0.0])
model.body_pos[cabinet_id] = cabinet_home + jitter
mujoco.mj_forward(model, data)

_, dof_adr = arm_addresses(model)
act_ids = np.asarray([object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS])
state, success = TaskState(), False

while data.time < TIME_LIMIT and not success:
    target, success = desired_position(model, data, state)
    q_target, _ = resolved_rate_target(model, data, state, target)
    data.ctrl[act_ids] = q_target
    mujoco.mj_step(model, data)

print(f"jitter={jitter}, success={success}, t={data.time:.2f}s, drawer={data.qpos[object_id(model, mujoco.mjtObj.mjOBJ_JOINT, 'drawer_slide')]:.4f}")
