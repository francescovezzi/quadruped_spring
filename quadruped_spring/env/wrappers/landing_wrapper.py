import gym
import numpy as np
from stable_baselines3.common.env_util import is_wrapped

from quadruped_spring.env.wrappers.moe_wrapper import MoEWrapper
from quadruped_spring.utils.timer import Timer


class LandingWrapper(gym.Wrapper):
    """
    Wrapper to switch controller when robot starts taking off.
    Dear user please pay attention at the order of the wrapper you are using.
    It's recommended to use this one as the last one.
    """

    def __init__(self, env, step_interval=5):
        super().__init__(env)
        self._robot_config = self.env.get_robot_config()
        self._landing_action = self.env.get_landing_action()
        self.timer_jumping = Timer(dt=self.env.get_env_time_step())
        self.is_moe_wrapped = is_wrapped(self, MoEWrapper)
        self.env.set_landing_callback(LandingCallback(self.env, step_interval))

    def temporary_switch_motor_control_gain(foo):
        def wrapper(self, *args, **kwargs):
            """Temporary switch motor control gain"""
            if self.env.are_springs_enabled():
                kp = 60.0
                kd = 2.0
            else:
                kp = 60.0
                kd = 2.0
            tmp_save_motor_kp = self.env.robot._motor_model._kp
            tmp_save_motor_kd = self.env.robot._motor_model._kd
            self.env.robot._motor_model._kp = kp
            self.env.robot._motor_model._kd = kd
            ret = foo(self, *args, **kwargs)
            self.env.robot._motor_model._kp = tmp_save_motor_kp
            self.env.robot._motor_model._kd = tmp_save_motor_kd
            return ret

        return wrapper

    @temporary_switch_motor_control_gain
    def landing_phase(self):
        self.env.landing_callback.activate()
        action = self._landing_action
        done = False
        while not done:
            obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def take_off_phase(self, action):
        """Repeat last action until you rech the height peak"""
        done = False
        self.start_jumping_timer()
        while not (self.timer_jumping.time_up() or done):  # episode or timer end
            self.timer_jumping.step_timer()
            obs, reward, done, infos = self.env.step(action)
        return obs, reward, done, infos

    def start_jumping_timer(self):
        actual_time = self.env.get_sim_time()
        delta_time = self.env.task.compute_time_for_peak_heihgt()
        self.timer_jumping.reset_timer()
        self.timer_jumping.start_timer(timer_time=actual_time, start_time=actual_time, delta_time=delta_time)

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)

        if self.env.task.is_switched_controller() and not done:
            _, reward, done, infos = self.take_off_phase(action)
            if not done:
                _, reward, done, infos = self.landing_phase()

        return obs, reward, done, infos

    def reset(self):
        self.env.landing_callback.reset()
        obs = self.env.reset()
        return obs


class LandingCallback:
    def __init__(self, env, step_interval=5):
        self._env = env
        self.step_interval = step_interval  # It means (1000 / 5) -> 200 Hz
        self.torque_dim = self._env.get_robot_config().NUM_MOTORS  # 12
        self.desired_torques = np.zeros(self.torque_dim)
        self.enable_callback = False
        self.counter = 0

    def reset(self):
        self.enable_callback = False
        self.counter = 0
        self.robot = self._env.robot

    def activate(self):
        self.enable_callback = True

    def deactivate(self):
        self.enable_callback = False
        self.counter = 0  # Maybe irrelevant

    def _compute_torques(self):
        # raise RuntimeError('Please implement me :(')

        return np.ones(12) * 0

    def compute_torques(self):
        if self.counter % self.step_interval == 0:
            self.desired_torques = self._compute_torques()
        return self.desired_torques

    def _callback_step(self):
        des_torques = self.compute_torques()
        self.robot.apply_external_torque(des_torques)
        self.counter += 1

    def callback_step(self):
        if self.enable_callback:
            self._callback_step()

    def is_enabled(self):
        return self.enable_callback

    __call__ = callback_step
