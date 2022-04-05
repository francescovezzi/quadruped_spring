import gym

import numpy as np

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.monitor_state2 import MonitorState


class JumpingStateMachine(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        if self.env._isRLGymInterface:
            raise ValueError('disable RLGymInterface in env_configs')
        if self.env._motor_control_mode != "TORQUE":
            raise ValueError('motor control mode should be TORQUE')
        self._settling_duration_steps = 1000
        self._couching_duration_steps = 4000
        assert self._couching_duration_steps >= 1000, 'couching duration steps number should be >= 1000'
        self._states = {'settling': 0, 'couching': 1, 'jumping_ground': 2, 'jumping_air': 3, 'landing': 4}
        self._state = self._states['settling']
        self._flying_up_counter = 0
        self._actions = {0: self.settling_action, 1: self.couching_action, 2: self.jumping_explosive_action, 3: self.jumping_flying_action, 4: self.jumping_landing_action}
        self._total_sim_steps = 9000
        self.max_height = 0.0
        self._step_counter = 0
        
        self._time_step = self.env._time_step
        self._robot_config = self.env._robot_config
        self._enable_springs = self.env._enable_springs


    def compute_action(self):
        return self._actions[self._state]()
    
    def compensate_spring(self):
        spring_action = self.env.robot._spring_torque
        return -np.array(spring_action)

    def update_state(self):
        if self._step_counter <= self._settling_duration_steps:
            actual_state = self._states['settling']
        elif self._step_counter <= self._settling_duration_steps + self._couching_duration_steps:
            actual_state = self._states['couching']
            # print(self.env.robot.GetBasePosition()[2])
        else:
            if self.all_feet_in_contact():
                actual_state = self._states['jumping_ground']
            else:
                if self.is_landing():
                    actual_state = self._states['landing']
                else:
                    actual_state = self._states['jumping_air']
        self.max_height = max(self.max_height, self.env.robot.GetBasePosition()[2])
        self._state = actual_state
        
    def settling_action(self):
        if self.env._enable_springs:
            config_des = np.array(self.env._robot_config.SPRINGS_REST_ANGLE * 4)
            config_init = self.env._robot_config.INIT_MOTOR_ANGLES
            config_ref = self.generate_ramp(self._step_counter, 0, self._settling_duration_steps -200, config_init, config_des)
            action = self.angle_ref_to_command(config_ref)
        else:
            config_des = self.env._robot_config.INIT_MOTOR_ANGLES
            action = self.angle_ref_to_command(config_des)
        return action

    def couching_action(self):
        max_torque = 35.55*0.9
        min_torque = 0
        i = self._step_counter
        i_min = self._settling_duration_steps
        i_max = i_min + self._couching_duration_steps - 500
        torque_thigh = self.generate_ramp(i, i_min, i_max, 0, 16)
        torque_calf = self.generate_ramp(i, i_min, i_max, min_torque, max_torque)
        torques = np.array([0,torque_thigh,-torque_calf]*4)
        return torques
    
    def jumping_explosive_action(self):
        coeff = 1.0
        f_rear = 190
        f_front = coeff * f_rear
        jump_command = np.full(12, 0)
        for i in range(4):
            if i < 2:
                f = f_front
            else:
                f = f_rear
            jump_command[3 * i : 3 * (i + 1)] = self.map_force_to_tau([0, 0, -f], i)
            jump_command[3*i] = 0
        # print(jump_command)
        return jump_command
    
    def jumping_flying_action(self):
        action = np.full(12,0)
        return action
    
    def jumping_landing_action(self):
        config_des = np.array(self.env._robot_config.SPRINGS_REST_ANGLE * 4)
        config_des = np.array(self.env._robot_config.INIT_MOTOR_ANGLES)
        config_des += np.array([0, -0.2, -0.2]*2 + [0, 0, 0]*2)
        q = self.robot.GetMotorAngles()
        dq = self.robot.GetMotorVelocities()
        compensate_springs = np.full(12,0)
        if self.env._enable_springs:
            kp = 20
            kd = 10.0
            compensate_springs = self.compensate_spring()
        else:
            kp = 55
            kd = 0.8
        torque = -kp * (q - config_des) - kd * dq
        # action = self.angle_ref_to_command(config_des)
        action = torque #+ compensate_springs
        return action
        
    def all_feet_in_contact(self):
        _, _, _, feetInContactBool = self.env.robot.GetContactInfo()
        return np.all(feetInContactBool)
    
    def is_landing(self):
        if self._flying_up_counter >= 20:
            return True
        else:
            self._flying_up_counter += 1
            return False
    
    def generate_ramp(self, i, i_min, i_max, u_min, u_max) -> float:
        if i < i_min:
            return u_min
        elif i > i_max:
            return u_max
        else:
            return u_min + (u_max - u_min) * (i - i_min) / (i_max - i_min)

    def angle_ref_to_command(self, angles_ref):
        q = self.robot.GetMotorAngles()
        dq = self.robot.GetMotorVelocities()
        if self.env._enable_springs:
            kp = 70
            kd = 0.8
        else:
            kp = 55
            kd = 0.8
        torque = -kp * (q - angles_ref) - kd * dq
        return torque
    
    def height_to_theta_des(self, h):
        l = self.env._robot_config.THIGH_LINK_LENGTH
        theta_thigh = np.arccos(h / (2 * l))
        theta_des = np.array([0, theta_thigh, -2 * theta_thigh] * 4)
        return theta_des
    
    def map_force_to_tau(self, F_foot, i):
        J, _ = self.env.robot.ComputeJacobianAndPosition(i)
        tau = J.T @ F_foot
        return tau
        
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
    
def build_env():
    env_config = {}
    env_config["enable_springs"] = True
    env_config["render"] = False
    env_config["on_rack"] = False
    env_config["enable_joint_velocity_estimate"] = False
    env_config["isRLGymInterface"] = False
    env_config["robot_model"] = "GO1"
    env_config["motor_control_mode"] = "TORQUE"
    env_config["action_repeat"] = 1
    env_config["record_video"] = False
    
    env = QuadrupedGymEnv(**env_config)
    env = JumpingStateMachine(env)
    env = MonitorState(env=env, path='logs/plots/manual_jumping', rec_length=env._total_sim_steps)
    return env
    
if __name__ == '__main__':
    
    env = build_env()
    sim_steps = env.env._total_sim_steps

    for _ in range(sim_steps):
        action = env.compute_action()
        obs, reward, done, info = env.step(action)
        # print(env.robot.GetMotorVelocities()-env.get_joint_velocity_estimation())
    env.release_plots()
    
    env.close()
    # print(env.max_height)
    print("end")
        