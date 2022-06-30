import copy

import gym
import numpy as np

from quadruped_spring.env.control_interface.utils import get_pose_from_phi_des


class InitialPoseWrapper(gym.Wrapper):
    """Wrapper for changing Initial Robot Pose based on the pitch angle."""

    def __init__(self, env, phi_des):
        super().__init__(env)
        self._phi_des = phi_des
        self.phi_des = np.deg2rad(phi_des)
        self.print_info()

    def print_info(self):
        print('*** Initial Pose Wrapper Info ***')
        print(f"initial pitch desired -> {self._phi_des} Degree")
        print('')

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def reset(self):
        self.env.reset()
        self._ac_interface = self.env.get_ac_interface()
        self.q_des = get_pose_from_phi_des(self.phi_des, self.env.robot)
        pose = self._ac_interface.get_last_reference()
        self.env._last_action = self._ac_interface._settle_robot_by_ramp(pose, self.q_des, intermediate_pose_param=1.0)
        return self.env.get_observation()
