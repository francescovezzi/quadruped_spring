import inspect
import os

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

from importlib import import_module

import numpy as np
import yaml
from matplotlib import pyplot as plt
from sb3_contrib import ARS
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from env.quadruped_gym_env import QuadrupedGymEnv
from quadruped_spring.env.wrappers.obs_flattening_wrapper import ObsFlatteningWrapper
from quadruped_spring.env.wrappers.rest_wrapper import RestWrapper
from quadruped_spring.utils.monitor_state import MonitorState
from quadruped_spring.utils.video_recording import VideoRec

SEED = 24

# Agent selection
LEARNING_ALGS = {"ars": ARS}
LEARNING_ALG = "ars"
SUB_FOLDER = "jumping_in_place/07_15"
ENV_ID = "QuadrupedSpring-v0"
ID = "4"
MODEL = "best_model"

REC_VIDEO = False  # Enable for video recording
SAVE_PLOTS = False  # Enable for save plots of the episode
RENDER = True  # For rendering
EVAL_EPISODES = 1  # Number of episodes to simulate
ENABLE_ENV_RANDOMIZATION = False  # For enable env randomization
ENV_RANDOMIZER = "MASS_RANDOMIZER"  # Mass randomization

LOG_DIR = f"logs/{SUB_FOLDER}"


def callable_env(env_id, wrappers, kwargs):
    def aux():
        env = env_id(**kwargs)
        env = RestWrapper(env)
        if REC_VIDEO:
            video_folder = os.path.join(LOG_DIR, "videos")
            video_name = f"{LEARNING_ALG}_{ENV_ID}_{ID}"
            env = VideoRec(env, video_folder, video_name)
        for wrapper in wrappers:
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
            if SAVE_PLOTS:
                plot_folder = os.path.join(LOG_DIR, "plots", f"{LEARNING_ALG}_{ENV_ID}_{ID}")
                env = MonitorState(env, path=plot_folder)
        return env

    return aux


# define directories
aux_dir = os.path.join(LOG_DIR, "models")
model_dir = os.path.join(currentdir, aux_dir, LEARNING_ALG, f"{ENV_ID}_{ID}")
model_file = os.path.join(model_dir, MODEL)
args_file = os.path.join(model_dir, ENV_ID, "args.yml")
stats_file = os.path.join(model_dir, ENV_ID, "vecnormalize.pkl")
plot_dir = os.path.join(LOG_DIR, "plots")

# Load env kwargs
env_kwargs = {}
if os.path.isfile(args_file):
    with open(args_file, "r") as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
            env_kwargs = loaded_args["env_kwargs"]
            if RENDER:
                env_kwargs["render"] = True
            env_kwargs["task_env"] = "ENDLESS_JUMPING"

wrapper_list = loaded_args["hyperparams"]["env_wrapper"]

# build env
env_kwargs["enable_env_randomization"] = ENABLE_ENV_RANDOMIZATION
env_kwargs["env_randomizer_mode"] = ENV_RANDOMIZER
env = callable_env(QuadrupedGymEnv, wrapper_list, env_kwargs)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_file, env)
env.training = False  # do not update stats at test time
env.norm_reward = False  # reward normalization is not needed at test time

# load model
custom_objects = {
    "learning_rate": 0.0,
    "lr_schedule": lambda _: 0.0,
    "clip_range": lambda _: 0.0,
}
model = LEARNING_ALGS[LEARNING_ALG].load(model_file, env, custom_objects=custom_objects)
print(f"\nLoaded model: {model_file}\n")
set_random_seed(SEED)


#################################################################
# run model
#################################################################
obs = env.reset()  # Always reset enviornment before stepping in it
n_episodes = EVAL_EPISODES
total_reward = 0
total_success = 0
# Simulate episodes
for _ in range(n_episodes):
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
    total_reward += rewards[0]
    total_success += info[0]["TimeLimit.truncated"]

avg_reward = total_reward / n_episodes
avg_success = total_success / n_episodes

if REC_VIDEO:
    env.env_method("release_video", indices=0)
if SAVE_PLOTS:
    env.env_method("release_plots", indices=0)

env_randomizer = env.env_method("get_randomizer_mode", indices=0)[0]
print("\n")
print(f"over {n_episodes} episodes using {env_randomizer} env randomizer:")
print(f"average reward -> {avg_reward}")
print(f"average success -> {avg_success}")
print("\n")

env.close()
print("end")
