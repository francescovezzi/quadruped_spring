import inspect
import os
import glob

import numpy as np
from matplotlib import pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from importlib import import_module

import yaml
from sb3_contrib import ARS
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from quadruped_spring.env.quadruped_gym_env import QuadrupedGymEnv
from quadruped_spring.env.wrappers.initial_pose_wrapper import InitialPoseWrapper
from quadruped_spring.env.wrappers.landing_wrapper import LandingWrapper
from quadruped_spring.env.wrappers.rest_wrapper import RestWrapper
from quadruped_spring.env.wrappers.moe_wrapper import ENV, MoEWrapper
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper


LEARNING_ALGS = {"ars": ARS}
LEARNING_ALG = "ars"
MAIN_FOLDER = "MoE_pitch_28_06"
ENV_ID = "QuadrupedSpring-v0"
ENV = QuadrupedGymEnv
MODEL_NAME = "best_model"

LOG_DIR = os.path.join(currentdir, 'logs', MAIN_FOLDER, 'models', LEARNING_ALG)
PLOT_PATH = os.path.join(currentdir, 'logs', MAIN_FOLDER, 'plots')


N_EVAL_EPISODES = 2
SEED = 37
set_random_seed(SEED)


class Agent():
    def __init__(self,
                 path: str,
                 ):
        self.path = path
        self.env_id_name = ENV_ID
        self.env_id_class = ENV
        self.number = self.get_agent_number()
        self.name = f'agent: {self.number}'
        self.model_name = MODEL_NAME
        self.n_eval_episodes = N_EVAL_EPISODES
        self.reward_list = []
        self.success_list = []
        self._init()
        
    def get_agent_number(self):
         return int((self.path.split('/')[-1]).split('_')[-1])
    
    def fill_performance(self):
        self.reward_list, self.success_list = self.test_agent()
        
    def _init(self):
        self._loaded_args = self.load_args()
        self.env_kwargs = self.load_env_kwargs()
        self.wrapper_list = self.load_wrapper_list()
        self.env = self.create_env()
        self.model = self.create_model()
        self.reward_list, self.success_list = self.test_agent()
    
    def run_episode(self):
        obs = self.env.reset()
        episode_reward = 0
        success = 0
        done = False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, done, info = self.env.step(action)
            episode_reward += rewards
        success = info[0]["TimeLimit.truncated"]
        return episode_reward[0], success
    
    def test_agent(self):
        reward_list = []
        success_list = []
        for _ in range(self.n_eval_episodes):
            episode_reward, success = self.run_episode()
            reward_list.append(episode_reward)
            success_list.append(1 if success else 0)
        return reward_list, success_list
        
    def load_args(self):
        args_path = os.path.join(self.path, f"{self.env_id_name}/args.yml")
        env_kwargs = {}
        if os.path.isfile(args_path):
            with open(args_path, "r") as f:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
                return loaded_args
        else:
            raise RuntimeError(f"{args_path} file not found.")
    
    def load_env_kwargs(self):
        env_kwargs = {}
        if self._loaded_args["env_kwargs"] is not None:
            env_kwargs = self._loaded_args["env_kwargs"]
            return env_kwargs

    def load_wrapper_list(self):
        wrapper_list = self._loaded_args["hyperparams"]["env_wrapper"]
        return wrapper_list
        
    def callable_env(self):
        def aux():
            env = self.env_id_class(**self.env_kwargs)
            env = RestWrapper(env)
            for wrapper in self.wrapper_list:
                is_dict = False
                if isinstance(wrapper, dict):
                    wrap_kwargs = list(wrapper.values())[0]
                    wrapper = list(wrapper.keys())[0]
                    is_dict = True
                module = ".".join(wrapper.split(".")[:-1])
                class_name = wrapper.split(".")[-1]
                module = import_module(module)
                wrap = getattr(module, class_name)
                if is_dict:
                    env = wrap(env, **wrap_kwargs)
                else:
                    env = wrap(env)
            return env
        return aux
    
    def create_env(self):
        stats_path = os.path.join(self.path, f"{self.env_id_name}/vecnormalize.pkl")
        env = self.callable_env()
        env = make_vec_env(env, n_envs=1)
        env = VecNormalize.load(stats_path, env)
        env.training = False  # do not update stats at test time
        env.norm_reward = False  # reward normalization is not needed at test time
        return env
    
    def create_model(self):
        model_path = os.path.join(self.path, self.model_name)
        env = self.create_env()
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "delta_std_schedule": lambda _: 0.0,
        }
        model = ARS.load(model_path, env, custom_objects=custom_objects)
        set_random_seed(SEED)
        return model
    
    @staticmethod
    def get_statistics(array):
        return np.mean(array), np.std(array)

    def get_reward_statistics(self):
        return self.get_statistics(self.reward_list)
    
    def get_success_rate(self):
        return np.sum(self.success_list) / self.n_eval_episodes
    
    
class AgentList():
    def __init__(self, path: str):
        self.path = path
        self.agent_list = []
        self.names_list = []
        self.num_experts = None
        self.mean_reward_list = []
        self.std_reward_list = []
        self.success_rate_list = []
        self._init()
    
    def _init(self):
        self.get_agents()
        self.num_experts = len(self.agent_list)
        self._fill_names()
        self._fill_reward_stats()
        self._fill_success_rate()
    
    def _fill_reward_stats(self):
        for agent in self.agent_list:
            mean, std = agent.get_reward_statistics()
            self.mean_reward_list.append(mean)
            self.std_reward_list.append(std)
    
    def _fill_success_rate(self):
        for agent in self.agent_list:
            self.success_rate_list.append(agent.get_success_rate())
    
    def _fill_names(self):
        for agent in self.agent_list:
            self.names_list.append(agent.name)
            
    def get_agents(self):
        for folder in glob.glob(os.path.join(self.path, '*')):
            self.agent_list.append(Agent(folder))
    
    def plot_performance(self):
        names = self.names_list
        mean_reward = self.mean_reward_list
        std_reward = self.std_reward_list
        success_rate = self.success_rate_list
        barWidth = 0.3
        
        fig, axes = plt.subplots(nrows=1, ncols=1)
        
        # The x position of bars
        r1 = np.arange(self.num_experts)
        r2 = [x + barWidth for x in r1]
        print(mean_reward, std_reward, success_rate)
        axes.bar(r1, mean_reward, width=barWidth, color='blue', edgecolor = 'black', yerr=std_reward, capsize=7, label=r'average reward')
        axes.bar(r2, success_rate, width=barWidth, color='orange', edgecolor = 'black', capsize=7, label=r'success rate')

        axes.set_xticks([r + barWidth / 2 for r in range(self.num_experts)], names)
        axes.set_xlim(-barWidth, 3 * barWidth * (self.num_experts))
        axes.set_ylim(0, 1.3)
        axes.legend()
        fig.suptitle(r'experts performances')
        
        return fig, axes    


def save_figure(fig, name):
    path = os.path.join(PLOT_PATH, name)
    fig.savefig(path)
            

if __name__ == '__main__':
    agents = AgentList(LOG_DIR)
    fig, axes = agents.plot_performance()
    save_figure(fig, 'performance_comparison')
    
    
    print('end')