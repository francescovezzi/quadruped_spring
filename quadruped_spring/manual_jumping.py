import inspect
import os
import time

import gym
import numpy as np

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, current_dir)

import argparse

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.evaluate_metric import EvaluateMetricJumpOnPlace
from env.wrappers.landing_wrapper import LandingWrapper
from env.wrappers.rest_wrapper import RestWrapper
from utils.timer import Timer

from quadruped_spring.env.quadruped_gym_env import TASK_SPACE_MAP

# from utils.monitor_state import MonitorState


class JumpingStateMachine(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._settling_duration_time = 1  # seconds
        self._couching_duration_time = 4  # seconds
        self._settling_duration_steps = self._settling_duration_time / self.env._env_time_step
        self._couching_duration_steps = self._couching_duration_time / self.env._env_time_step
        assert self._couching_duration_steps >= 1000, "couching duration steps number should be >= 1000"
        self._states = {"settling": 0, "couching": 1, "jumping_ground": 2}
        self._state = self._states["settling"]
        self._flying_up_counter = 0
        self._actions = {
            0: self.settling_action,
            1: self.couching_action,
            2: self.jumping_explosive_action,
        }
        self._total_sim_steps = 9000
        self.max_height = 0.0
        self._step_counter = 0

        self._robot_config = self.env.get_robot_config()
        self._enable_springs = self.env._enable_springs
        self._jump_end = False
        self._flight_timer = Timer(dt=self.env._sim_time_step)

        self._default_action = self.env.compute_action_from_command(self._robot_config.INIT_MOTOR_ANGLES)

    def compute_action(self):
        return self._actions[self._state]()

    def update_state(self):
        if self._step_counter <= self._settling_duration_steps:
            actual_state = self._states["settling"]
        elif self._step_counter <= self._settling_duration_steps + self._couching_duration_steps:
            actual_state = self._states["couching"]
        else:
            actual_state = self._states["jumping_ground"]

        self.max_height = max(self.max_height, self.env.robot.GetBasePosition()[2])
        # if self._state != actual_state:
        #     print('********************')
        #     print(f'{self._state} -> {actual_state}')
        #     print(f'joint config is: {self.env.robot.GetMotorAngles()}')
        #     print(f'sim time is: {self.env.get_sim_time()}')
        #     print('********************')
        self._state = actual_state

    def settling_action(self):
        return self._default_action

    def couching_action(self):
        max_action_calf = -1
        min_action_calf = self._default_action[2]
        max_action_thigh = 0.9
        min_action_thigh = self._default_action[1]
        i = self._step_counter
        i_min = self._settling_duration_steps
        i_max = i_min + self._couching_duration_steps - 500
        action_thigh = self.generate_ramp(i, i_min, i_max, min_action_thigh, max_action_thigh)
        action_calf = self.generate_ramp(i, i_min, i_max, min_action_calf, max_action_calf)
        torques = np.array([0, action_thigh, action_calf] * 4)
        # torques = np.array([0, 0.5, 1] * 4)
        return torques

    def jumping_explosive_action(self):
        if self.env._enable_springs:
            coeff = 0.1
        else:
            coeff = 0.3
        action_front = np.array([0, 0, coeff * 1] * 2)
        action_rear = np.array([0, 0, 1] * 2)
        jump_action = np.concatenate((action_front, action_rear))
        return jump_action

    @staticmethod
    def generate_ramp(i, i_min, i_max, u_min, u_max) -> float:
        if i < i_min:
            return u_min
        elif i > i_max:
            return u_max
        else:
            return u_min + (u_max - u_min) * (i - i_min) / (i_max - i_min)

    def step(self, action):

        obs, reward, done, infos = self.env.step(action)
        self._step_counter += 1

        self.update_state()

        return obs, reward, done, infos

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)

    def reset(self):
        obs = self.env.reset()
        return obs

    def close(self):
        self.env.close()


def build_env(enable_springs=False):
    env_config = {
        "enable_springs": enable_springs,
        "render": False,
        "on_rack": False,
        "enable_joint_velocity_estimate": False,
        "isRLGymInterface": True,
        "motor_control_mode": "PD",
        "action_repeat": 1,
        "record_video": False,
        "action_space_mode": "DEFAULT",
        "task_env": "JUMPING_ON_PLACE_HEIGHT",
    }
    # env_config["enable_springs"] = True
    if fill_line:
        env_config["render"] = False
    env = QuadrupedGymEnv(**env_config)
    return env


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-springs", action="store_true", default=False, help="enable springs")
    parser.add_argument("--fill-line", action="store_true", default=False, help="fill line in report.txt")
    args = parser.parse_args()
    enable_springs = args.enable_springs
    fill_line = args.fill_line

    env = build_env(enable_springs=enable_springs)
    env = JumpingStateMachine(env)
    # sim_steps = env._total_sim_steps + 3000
    # env = MonitorState(env=env, path="logs/plots/manual_jumping_with_springs", rec_length=sim_steps)
    env = EvaluateMetricJumpOnPlace(env)
    env = RestWrapper(env)
    env = LandingWrapper(env)
    done = False
    while not done:
        action = env.compute_action()
        # action = np.zeros(12)
        obs, reward, done, info = env.step(action)
        # print(env.robot.GetMotorVelocities()-env.get_joint_velocity_estimation())
    # env.release_plots()
    print("******")
    print(f"reward -> {reward}")
    print(f"min_height -> {env.get_metric().height_min}")
    print("******")
    if fill_line:
        report_path = os.path.join(current_dir, "logs", "models", "performance_report.txt")
        with open(report_path, "w") as f:
            f.write(env.print_first_line_table())
            if enable_springs:
                f.write(env.fill_line(id="ManualWithSprings"))
            else:
                f.write(env.fill_line(id="ManualWithoutSprings"))
    else:
        env.print_metric()
    env.close()
    print("end")
