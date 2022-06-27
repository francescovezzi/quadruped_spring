import gym
import os, glob
import numpy as np


class InitialPoseWrapper(gym.Wrapper):
    """Wrapper for changing Initial Robot Pose based on the pitch angle."""

    def __init__(self, env, experts_folder):
        super().__init__(env)
        self._experts_folder = experts_folder
        self.experts = {}

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def reset(self):
        obs = self.env.reset()
        return obs
    
    @staticmethod
    def expert_identikit(expert_path):
        models_path = os.path.join(expert_path, 'models')
        for model in glob.glob(models_path):
            pass
