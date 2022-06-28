import gym
import os, glob
import numpy as np
import yaml

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import ARS


ENV = 'QuadrupedSpring-v0'
MODEL = 'best_model'


class MoEWrapper(gym.Wrapper):
    """Mixture of Experts ensembling."""

    def __init__(self, env, experts_folder):
        super().__init__(env)
        self._experts_folder = experts_folder
        self.experts = self._get_models()

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def reset(self):
        obs = self.env.reset()
        return obs
    
    def _get_models(self):
        models_path = os.path.join(self._experts_folder, 'models')
        model_list = []
        for model_folder in glob.glob(os.path.join(models_path, '*')):
            model = self.create_model(model_folder)
            model_list.append(model)
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
            raise RuntimeError(f'{args_path} file not found.')
        
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
    
    def get_experts_prediction(self, obs):
        norm_obs = lambda model, obs: model.get_vec_normalize_env().normalize_obs(obs)
        predictions = [expert.predict(norm_obs(expert, obs), deterministic=True)[0] for expert in self.experts]
        return predictions
    
    @staticmethod
    def _compute_action_ensemble(actions_pred):
        w0 = 0.0
        w1 = 1.0
        action_ensemble = actions_pred[0] * w0 + actions_pred[1] * w1
        return np.clip(action_ensemble, -1, 1)
    
    def get_action_ensemble(self, obs):
        actions_pred = self.get_experts_prediction(obs)
        return self._compute_action_ensemble(actions_pred)
         