import inspect
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
urdf_file = os.path.join(currentdir, "urdf/go1.urdf")


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
planeId = p.loadURDF("plane.urdf")
quad = p.loadURDF(urdf_file, basePosition=[0, 0, 0.5])
num_joints = p.getNumJoints(quad)
print("*****************************")
print(f"joint numbers = {num_joints}")
print(f"they are {num_joints}, one is the trunk/imu_joint 4 are dummy joints for collision on shoulders, 4 are the feet")
print("******************************")
theta0 = np.pi / 4
p.setGravity(0, 0, -9.81)


time_steps = 3000

for i in range(time_steps):
    p.stepSimulation()

    time.sleep(1.0 / 240.0)

p.disconnect()
