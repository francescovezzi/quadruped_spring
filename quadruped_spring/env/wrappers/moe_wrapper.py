import glob
import os

import gym
import numpy as np
import yaml
from sb3_contrib import ARS
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from quadruped_spring.env.sensors.robot_sensors import SensorList
from quadruped_spring.env.sensors.sensor_collection import SensorCollection
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper

ENV = "QuadrupedSpring-v0"
MODEL = "best_model"


class MoEWrapper(gym.Wrapper):
    """Mixture of Experts ensembling."""

    def __init__(self, env, experts_folder):
        super().__init__(env)
        self._experts_folder = experts_folder
        self.experts = self._get_models()

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        self.step_expert_sensors()

        return obs, reward, done, infos

    def reset(self):
        obs = self.env.reset()
        self._reset_expert_sensors()
        return obs

    def _reset_expert_sensors(self):
        for _, expert_sensors in self.experts:
            expert_sensors._reset(self.env.robot)

    def _get_expert_observation(self, model_sensor):
        obs = model_sensor.get_noisy_obs()
        return ObsFlatteningWrapper._flatten_obs(obs)

    def step_expert_sensors(self):
        [expert_sensors._on_step() for _, expert_sensors in self.experts]

    def _build_model_sensors(self, model):
        observation_space_mode = model.env.env_method("get_observation_space_mode")[0]
        robot_sensors = SensorList(SensorCollection().get_el(observation_space_mode))
        robot_sensors._init(robot_config=self.env.get_robot_config())
        return robot_sensors

    def _get_models(self):
        models_path = os.path.join(self._experts_folder, "models/ars")
        model_list = []
        for model_folder in glob.glob(os.path.join(models_path, "*")):
            model = self.create_model(model_folder)
            model_sensors = self._build_model_sensors(model)
            model_list.append((model, model_sensors))
        return model_list

    @staticmethod
    def load_env_kwargs(model_folder):
        args_path = os.path.join(model_folder, f"{ENV}/args.yml")
        env_kwargs = {}
        if os.path.isfile(args_path):
            with open(args_path, "r") as f:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
                if loaded_args["env_kwargs"] is not None:
                    env_kwargs = loaded_args["env_kwargs"]
            return env_kwargs
        else:
            raise RuntimeError(f"{args_path} file not found.")

    def create_env(self, model_folder):
        stats_path = os.path.join(model_folder, f"{ENV}/vecnormalize.pkl")
        env_kwargs = self.load_env_kwargs(model_folder)
        env = lambda: gym.make(ENV, **env_kwargs)
        env = make_vec_env(env, n_envs=1)
        env = VecNormalize.load(stats_path, env)
        env.training = False  # do not update stats at test time
        env.norm_reward = False  # reward normalization is not needed at test time
        return env

    def create_model(self, model_folder):
        model_path = os.path.join(model_folder, MODEL)
        env = self.create_env(model_folder)
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "delta_std_schedule": lambda _: 0.0,
        }
        model = ARS.load(model_path, env, custom_objects=custom_objects)
        return model

    def get_experts_prediction(self):
        norm_obs = lambda model, obs: model.get_vec_normalize_env().normalize_obs(obs)
        predictions = [
            expert.predict(norm_obs(expert, self._get_expert_observation(expert_sensors)), deterministic=True)[0]
            for expert, expert_sensors in self.experts
        ]
        return predictions

    @staticmethod
    def _compute_action_ensemble(actions_pred):
        w0 = 1.0
        w1 = 0.0
        w2 = 0.0
        action_ensemble = actions_pred[0] * w0 + actions_pred[1] * w1  # + actions_pred[2] * w2
        return np.clip(action_ensemble, -1, 1)

    def get_action_ensemble(self):
        actions_pred = self.get_experts_prediction()
        return self._compute_action_ensemble(actions_pred)
