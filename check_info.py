import mujoco

model = mujoco.MjModel.from_xml_path("scene/drawer_scene_bad.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)  

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)
print("timestep:", model.opt.timestep)
print("integrator:", model.opt.integrator)

handle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "handle_site")
mujoco.mj_forward(model, data)
print("handle_site world pos:", data.site_xpos[handle_id])