import inspect
import os

import numpy as np
from matplotlib import pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from stable_baselines3.common.utils import set_random_seed

from quadruped_spring.env.quadruped_gym_env import QuadrupedGymEnv
from quadruped_spring.env.wrappers.initial_pose_wrapper import InitialPoseWrapper
from quadruped_spring.env.wrappers.landing_wrapper import LandingWrapper
from quadruped_spring.env.wrappers.moe_wrapper import MoEWrapper
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper
from quadruped_spring.env.wrappers.rest_wrapper import RestWrapper

SEED = 24

FOLDER = "MoE_pitch_28_06"
LOG_FOLDER = f"logs/{FOLDER}"
PLOT_PATH = f"{LOG_FOLDER}/plots"

N_EVAL_EPISODES = 15

RENDER = False
CURRICULUM_LEVEL = 0.2


def build_env():
    env_config = {
        "render": RENDER,
        "on_rack": False,
        "motor_control_mode": "PD",
        "action_repeat": 10,
        "enable_springs": False,
        "add_noise": False,
        "enable_action_interpolation": False,
        "enable_action_filter": True,
        "task_env": "JUMPING_IN_PLACE",
        "observation_space_mode": "PHI_DES",
        "action_space_mode": "SYMMETRIC",
        "enable_env_randomization": True,
        "env_randomizer_mode": "PITCH_RANDOMIZER",
        "curriculum_level": CURRICULUM_LEVEL,
    }
    env = QuadrupedGymEnv(**env_config)

    env = InitialPoseWrapper(env, phi_des=5.0)
    env = RestWrapper(env)
    env = MoEWrapper(env, LOG_FOLDER, seed=SEED)
    env = ObsFlatteningWrapper(env)
    env = LandingWrapper(env)

    return env


def run_episode(env, action):
    obs = env.reset()
    episode_reward = 0
    success = 0
    done = False
    while not done:
        obs, rewards, done, info = env.step(action)
        episode_reward += rewards
    success = info["TimeLimit.truncated"]
    return episode_reward, success


def test_agent(env, action):
    reward_list = []
    success_list = []
    for _ in range(N_EVAL_EPISODES):
        episode_reward, success = run_episode(env, action)
        reward_list.append(episode_reward)
        success_list.append(1 if success else 0)
    return reward_list, success_list


class Agent:
    def __init__(
        self,
        number=None,
        reward_list=None,
        success_list=None,
        action=None,
    ):
        self.number = number
        self.reward_list = np.asarray(reward_list)
        self.success_list = np.asarray(success_list)
        self.action = action

    @staticmethod
    def get_statistics(array):
        return np.mean(array), np.std(array)

    def get_reward_statistics(self):
        return self.get_statistics(self.reward_list)


def compute_agent_kwargs(env: QuadrupedGymEnv, idx: int, num_experts: int) -> dict:
    agent_number = idx
    action = np.zeros(num_experts)
    action[idx] = 1
    agent_action = action
    agent_rw, agent_success = test_agent(env, action)
    return {"number": agent_number, "reward_list": agent_rw, "success_list": agent_success, "action": agent_action}


def build_agent_list(env):
    num_experts = env.get_experts_number()
    agent_list = []
    for i in range(num_experts):
        agent_kwargs = compute_agent_kwargs(env, i, num_experts)
        agent_list.append(Agent(**agent_kwargs))
    return agent_list


class AgentList:
    def __init__(self, agent_list: list[Agent]):
        self.agent_list = agent_list
        self.names_list = []
        self.num_experts = len(self.agent_list)
        self.mean_reward_list = []
        self.std_reward_list = []
        self.success_rate_list = []
        self._init()

    def _init(self):
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
            success_rate = np.sum(agent.success_list) / N_EVAL_EPISODES
            self.success_rate_list.append(success_rate)

    def _fill_names(self):
        for agent in self.agent_list:
            self.names_list.append(agent.number)

    def plot_performance(self):
        names = [f"agent: {name}" for name in agents.names_list]
        mean_reward = self.mean_reward_list
        std_reward = self.std_reward_list
        success_rate = self.success_rate_list
        barWidth = 0.3

        fig, axes = plt.subplots(nrows=1, ncols=1)

        # The x position of bars
        r1 = np.arange(self.num_experts)
        r2 = [x + barWidth for x in r1]

        axes.bar(
            r1,
            mean_reward,
            width=barWidth,
            color="blue",
            edgecolor="black",
            yerr=std_reward,
            capsize=7,
            label=r"average reward",
        )
        axes.bar(r2, success_rate, width=barWidth, color="orange", edgecolor="black", capsize=7, label=r"success rate")

        axes.set_xticks([r + barWidth / 2 for r in range(self.num_experts)], names)
        axes.set_xlim(-barWidth, 3 * barWidth * (self.num_experts))
        axes.set_ylim(0, 1.3)
        axes.legend()
        fig.suptitle(r"experts performances")

        return fig, axes


def save_figure(fig, name):
    path = os.path.join(PLOT_PATH, name)
    fig.savefig(path)


if __name__ == "__main__":

    env = build_env()
    agents = AgentList(build_agent_list(env))

    # Agents performances
    fig, axes = agents.plot_performance()
    save_figure(fig, "agents_performance")

    plt.show()
    print("end")
