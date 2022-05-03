"""This file implements the gym environment for a quadruped. """
import inspect
import os

# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import datetime
import time

# gym
import gym
import numpy as np

# pybullet
import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc
import quadruped
from gym import spaces
from gym.utils import seeding
from scipy.spatial.transform import Rotation as R

import quadruped_spring.go1.configs_go1_with_springs as go1_config_with_springs
import quadruped_spring.go1.configs_go1_without_springs as go1_config_without_springs
from quadruped_spring.utils import action_filter

from quadruped_spring.env.sensors import robot_sensors as rs
from quadruped_spring.env.sensors.robot_sensors import SensorList

ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
VIDEO_LOG_DIRECTORY = "videos/" + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")

# For the sensor equipment selectoin
OBS_SPACE_MAP = {"DEFAULT": [rs.IMU, rs.FeetPostion, rs.FeetVelocity, rs.GroundReactionForce]}

# Implemented action spaces for deep reinforcement learning:
#   - "DEFAULT": classic
#   - "SYMMETRIC" legs right side and left side move symmetrically
#   - "SYMMETRIC_NO_HIP" as symmetric but hips receive action = 0

# Tasks to be learned with reinforcement learning
#     - "FWD_LOCOMOTION"
#         reward forward progress only
#     - "LR_COURSE_TASK"
#         [TODO: what should you train for?]
#         Ideally we want to command GO1 to run in any direction while expending minimal energy
#         It is suggested to first train to run at 3 sample velocities (0.5 m/s, 1 m/s, 1.5 m/s)
#         How will you construct your reward function?
#     - "JUMPING_TASK"
#         Sparse reward, maximizing flight time + bonus forward_distance + malus on crashing
#     - "JUMPING_ON_PLACE_TASK"
#         Sparse reward, maximizing flight time + bonus maintin base position +
#         malus on crashing + malus on not allowed contacts
#     - "JUMPING_ON_PLACE_HEIGHT_TASK" (not working actually)
#         Sparse reward, maximizing maximum height relative to one jump +
#         bonus maintin base position + malus on crashing + malus on not allowed contacts
#     - "JUMPING_ON_PLACE_ABS_HEIGHT_TASK"
#         Sparse reward, maximizing absolute maximum height + bonus maintin base position +
#         malus on crashing + malus on not allowed contacts
#     - "LANDING_TASK" (not implemented yet)
#         Sparse reward, bonus mantain desired position + malus on crushing +
#         malus on not allowed contact + malus on feet not in contact
#     - "JUMPING_FORWARD"
#         Sparse reward, maximizing jump forward distance + bonus maximum height +
#         bonus maintain base orientation + malus on crashing + malus on not allowed contacts

# Motor control modes:
#   - "TORQUE":
#         supply raw torques to each motor (12)
#   - "PD":
#         supply desired joint positions to each motor (12)
#         torques are computed based on the joint position/velocity error
#   - "CARTESIAN_PD":
#         supply desired foot positions for each leg (12)
#         torques are computed based on the foot position/velocity error
#   - "INVKIN_CARTESIAN_PD":
#         supply desired foot positions for each leg (12)
#         torques are computed based on the joint position/velocity error


EPISODE_LENGTH = 10  # how long before we reset the environment (max episode length for RL)
MAX_FWD_VELOCITY = 5  # to avoid exploiting simulator dynamics, cap max reward for body velocity

class QuadrupedGymEnv(gym.Env):
    """The gym environment for a quadruped {Unitree GO1}.

    It simulates the locomotion of a quadrupedal robot.
    The state space, action space, and reward functions can be chosen with:
    observation_space_mode, motor_control_mode, task_env.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        isRLGymInterface=True,
        time_step=0.001,
        action_repeat=10,
        distance_weight=1e3,
        energy_weight=1e-4,  # 0.008,
        motor_control_mode="PD",
        task_env="FWD_LOCOMOTION",
        observation_space_mode="DEFAULT",
        action_space_mode="DEFAULT",
        on_rack=False,
        render=False,
        record_video=False,
        add_noise=True,
        enable_springs=False,
        enable_action_interpolation=False,
        enable_action_filter=False,
        enable_action_clipping=False,
        enable_joint_velocity_estimate=False,
        test_env=False,  # NOT ALLOWED FOR TRAINING!
    ):
        """Initialize the quadruped gym environment.

        Args:
          isRLGymInterface: If the gym environment is being run as RL or not. Affects
            if the actions should be scaled.
          time_step: Simulation time step.
          action_repeat: The number of simulation steps where the same actions are applied.
          distance_weight: The weight of the distance term in the reward.
          energy_weight: The weight of the energy term in the reward.
          motor_control_mode: Whether to use torque control, PD, control, etc.
          task_env: Task trying to learn (fwd locomotion, standup, etc.)
          observation_space_mode: what should be in here? Check available functions in quadruped.py
          action_space_mode: For action space dimension selecting
          on_rack: Whether to place the quadruped on rack. This is only used to debug
            the walking gait. In this mode, the quadruped's base is hanged midair so
            that its walking gait is clearer to visualize.
          render: Whether to render the simulation.
          record_video: Whether to record a video of each trial.
          add_noise: vary coefficient of friction
          test_env: add random terrain
          enable_springs: Whether to enable springs or not
          enable_action_interpolation: Whether to interpolate the current action
            with the previous action in order to produce smoother motions
          enable_action_filter: Boolean specifying if a lowpass filter should be
            used to smooth actions.
          enable_action_clipping: Boolean specifying if motor commands should be
            clipped or not. It's not implemented for pure torque control.
          enable_joint_velocity_estimate: Boolean specifying if it's used the
            estimated or the true joint velocity. Actually it affects only real
            observations space modes.
        """
        self.seed()
        self._enable_springs = enable_springs
        if self._enable_springs:
            self._robot_config = go1_config_with_springs
        else:
            self._robot_config = go1_config_without_springs
        self._isRLGymInterface = isRLGymInterface
        self._time_step = time_step
        self._action_repeat = action_repeat
        self.dt = self._action_repeat * self._time_step
        self._distance_weight = distance_weight
        self._energy_weight = energy_weight
        self._motor_control_mode = motor_control_mode
        self._TASK_ENV = task_env
        self._robot_sensors = SensorList(OBS_SPACE_MAP[observation_space_mode])
        self._action_space_mode = action_space_mode
        self._hard_reset = True  # must fully reset simulation at init
        self._on_rack = on_rack
        self._is_render = render
        self._is_record_video = record_video
        self._add_noise = add_noise
        self._enable_action_interpolation = enable_action_interpolation
        self._enable_action_filter = enable_action_filter
        self._enable_action_clipping = enable_action_clipping
        self._enable_joint_velocity_estimate = enable_joint_velocity_estimate
        self._using_test_env = test_env
        if test_env:
            self._add_noise = True
            self._observation_noise_stdev = 0.01  # TODO: check if increasing makes sense
        else:
            self._observation_noise_stdev = 0.01

        # other bookkeeping
        self._num_bullet_solver_iterations = int(300 / action_repeat)
        self._last_frame_time = 0.0  # for rendering
        self._MAX_EP_LEN = EPISODE_LENGTH  # max sim time in seconds, arbitrary
        self._action_bound = 1.0

        self.setupActionSpace()
        self.setupObservationSpace()

        if self._enable_action_filter:
            self._action_filter = self._build_action_filter()

        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()
        self._configure_visualizer()

        self.videoLogID = None
        self.reset()

    ######################################################################################
    # RL Observation and Action spaces
    ######################################################################################
    
    def setupObservationSpace(self):
        self._robot_sensors._init(robot_config=self._robot_config)
        obs_high = (self._robot_sensors._get_high_limits() + OBSERVATION_EPS)
        obs_low = (self._robot_sensors._get_low_limits() - OBSERVATION_EPS)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def setupActionSpace(self):
        """Set up action space for RL."""
        if self._motor_control_mode not in ["PD", "TORQUE", "CARTESIAN_PD", "INVKIN_CARTESIAN_PD"]:
            raise ValueError("motor control mode " + self._motor_control_mode + " not implemented yet.")

        if self._action_space_mode == "DEFAULT":
            action_dim = 12
        elif self._action_space_mode == "SYMMETRIC":
            action_dim = 6
        elif self._action_space_mode == "SYMMETRIC_NO_HIP":
            action_dim = 4
        else:
            raise ValueError(f"action space mode {self._action_space_mode} not implemented yet")

        action_high = np.array([1] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self._action_dim = action_dim

    def get_action_dim(self):
        return self._action_dim
    
    def get_observation(self):
        return self._robot_sensors.get_obs()

    ######################################################################################
    # Termination and reward
    ######################################################################################
    def is_fallen(self, dot_prod_min=0.85):
        """Decide whether the quadruped has fallen.

        If the up directions between the base and the world is larger (the dot
        product is smaller than 0.85) or the base is very low on the ground
        (the height is smaller than 0.13 meter), the quadruped is considered fallen.

        Returns:
          Boolean value that indicates whether the quadruped has fallen.
        """
        base_rpy = self.robot.GetBaseOrientationRollPitchYaw()
        orientation = self.robot.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.robot.GetBasePosition()
        return (
            np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < dot_prod_min or pos[2] < self._robot_config.IS_FALLEN_HEIGHT
        )

    def _not_allowed_contact(self):
        """
        Return True if the robot is performing some not allowed contact
        as touching the ground with knees
        """
        _, num_invalid_contacts, _, _ = self.robot.GetContactInfo()

        return num_invalid_contacts

    def _disoriented(self):
        roll, _, yaw = self.robot.GetBaseOrientationRollPitchYaw()
        max_angle = 25 * np.pi / 180
        return roll < max_angle or yaw < max_angle

    def _base_near_ground(self):
        _, _, z = self.robot.GetBasePosition()
        return z < 0.1

    def _termination(self):
        """Decide whether we should stop the episode and reset the environment."""
        self.terminated = False
        if self._TASK_ENV in ["JUMPING_TASK", "LR_COURSE_TASK", "FWD_LOCOMOTION"]:
            self.terminated = self.is_fallen()
            return self.terminated
        elif self._TASK_ENV in ["JUMPING_ON_PLACE_TASK", "JUMPING_ON_PLACE_HEIGHT_TASK", "JUMPING_ON_PLACE_ABS_HEIGHT_TASK"]:
            self.terminated = self.is_fallen() or self._not_allowed_contact()
            return self.terminated
        elif self._TASK_ENV == "LANDING_TASK":
            pass
        elif self._TASK_ENV == "JUMPING_FORWARD":
            return self.is_fallen() or self._not_allowed_contact()
        else:
            raise ValueError("This task mode {self._TASK_ENV} is not implemented yet.")

    def _reward_jumping(self):
        # Change is fallen height
        self._robot_config.IS_FALLEN_HEIGHT = 0.01

        _, _, _, feet_in_contact = self.robot.GetContactInfo()
        # no_feet_in_contact_reward = -np.mean(feet_in_contact)

        flight_time_reward = 0.0
        if np.all(1 - np.array(feet_in_contact)):
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self.get_sim_time()
                self._robot_pose_take_off = np.array(self.robot.GetBasePosition())
                self._robot_orientation_take_off = np.array(self.robot.GetBaseOrientationRollPitchYaw())
            else:
                # flight_time_reward = self.get_sim_time() - self._time_take_off
                pass
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self.get_sim_time() - self._time_take_off, self._max_flight_time)
                # Compute forward distance according to local frame (starting at take off)
                rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
                translation = -self._robot_pose_take_off
                pos_abs = np.array(self.robot.GetBasePosition())
                pos_relative = pos_abs + translation
                pos_relative = pos_relative @ rotation_matrix
                self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)

            self._all_feet_in_the_air = False

        return flight_time_reward
        # max_height_reward = self.robot.GetBasePosition()[2] / self._init_height
        # only_positive_height = max(self.robot.GetBasePosition()[2] - self._init_height, 0.0)
        # max_height_reward = only_positive_height ** 2 / self._init_height ** 2
        # return max_height_reward + flight_time_reward
        # return max_height_reward + no_feet_in_contact_reward

    def _reward_jumping_forward(self):
        _, _, _, feet_in_contact = self.robot.GetContactInfo()
        # no_feet_in_contact_reward = -np.mean(feet_in_contact)

        if np.all(1 - np.array(feet_in_contact)):
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self.get_sim_time()
                self._robot_pose_take_off = np.array(self.robot.GetBasePosition())
                self._robot_orientation_take_off = np.array(self.robot.GetBaseOrientationRollPitchYaw())
                self._robot_orientation_take_off_quat = np.array(self.robot.GetBaseOrientation())
            else:
                # flight_time_reward = self.get_sim_time() - self._time_take_off
                pass
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self.get_sim_time() - self._time_take_off, self._max_flight_time)
                # Compute forward distance according to local frame (starting at take off)
                rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
                translation = -self._robot_pose_take_off
                pos_abs = np.array(self.robot.GetBasePosition())
                pos_relative = pos_abs + translation
                pos_relative = pos_relative @ rotation_matrix
                self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)

            self._all_feet_in_the_air = False

        roll, _, yaw = self.robot.GetBaseOrientationRollPitchYaw()
        self._max_yaw = max(np.abs(yaw), self._max_yaw)
        self._max_roll = max(np.abs(roll), self._max_roll)
        delta_height = max(self.robot.GetBasePosition()[2] - self._init_height, 0.0)
        self._max_height = max(self._max_height, delta_height)

        return 0

    def _reward_jumping_on_place(self):
        """
        Reward maximum flight time, plus computing maximum orientation angles
        and forward distance
        """
        # Change is fallen height
        self._robot_config.IS_FALLEN_HEIGHT = 0.01

        _, _, _, feet_in_contact = self.robot.GetContactInfo()
        # no_feet_in_contact_reward = -np.mean(feet_in_contact)

        flight_time_reward = 0.0
        if np.all(1 - np.array(feet_in_contact)):
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self.get_sim_time()
                self._robot_pose_take_off = np.array(self.robot.GetBasePosition())
                self._robot_orientation_take_off = np.array(self.robot.GetBaseOrientationRollPitchYaw())
                self._robot_orientation_take_off_quat = np.array(self.robot.GetBaseOrientation())
            else:
                # flight_time_reward = self.get_sim_time() - self._time_take_off
                pass
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self.get_sim_time() - self._time_take_off, self._max_flight_time)
                # Compute forward distance according to local frame (starting at take off)
                rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
                translation = -self._robot_pose_take_off
                pos_abs = np.array(self.robot.GetBasePosition())
                pos_relative = pos_abs + translation
                pos_relative = pos_relative @ rotation_matrix
                self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)

            self._all_feet_in_the_air = False

        _, pitch, yaw = self.robot.GetBaseOrientationRollPitchYaw()
        self._max_yaw = max(np.abs(yaw), self._max_yaw)
        self._max_pitch = max(np.abs(pitch), self._max_pitch)

        return flight_time_reward

    def _reward_jumping_on_place_height(self):
        """
        Reward maximum height calculated from the jumping taking off, plus
        compute maximum orientation angles and forward distance
        """
        # Change is fallen height
        self._robot_config.IS_FALLEN_HEIGHT = 0.01

        _, _, _, feet_in_contact = self.robot.GetContactInfo()
        # no_feet_in_contact_reward = -np.mean(feet_in_contact)

        flight_time_reward = 0.0
        if np.all(1 - np.array(feet_in_contact)):
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self.get_sim_time()
                self._robot_pose_take_off = np.array(self.robot.GetBasePosition())
                self._robot_orientation_take_off = np.array(self.robot.GetBaseOrientationRollPitchYaw())
                self._robot_orientation_take_off_quat = np.array(self.robot.GetBaseOrientation())
                self._jump_init_height = self._robot_pose_take_off[2]
            else:
                delta_height = max(self.robot.GetBasePosition()[2] - self._jump_init_height, 0.0)
                self._max_height = max(self._max_height, delta_height)
                # print(self._max_height)
                # flight_time_reward = self.get_sim_time() - self._time_take_off
                pass
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self.get_sim_time() - self._time_take_off, self._max_flight_time)
                # Compute forward distance according to local frame (starting at take off)
                rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
                translation = -self._robot_pose_take_off
                pos_abs = np.array(self.robot.GetBasePosition())
                pos_relative = pos_abs + translation
                pos_relative = pos_relative @ rotation_matrix
                self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)

            self._all_feet_in_the_air = False

        _, pitch, yaw = self.robot.GetBaseOrientationRollPitchYaw()
        self._max_yaw = max(np.abs(yaw), self._max_yaw)
        self._max_pitch = max(np.abs(pitch), self._max_pitch)

        return flight_time_reward

    def _reward_jumping_on_place_abs_height(self):
        """
        Reward maximum peak height minus the initial height, plus
        compute maximum orientation angles and forward distance
        """
        # Change is fallen height
        self._robot_config.IS_FALLEN_HEIGHT = 0.01

        _, _, _, feet_in_contact = self.robot.GetContactInfo()
        # no_feet_in_contact_reward = -np.mean(feet_in_contact)

        flight_time_reward = 0.0
        if np.all(1 - np.array(feet_in_contact)):
            if not self._all_feet_in_the_air:
                self._all_feet_in_the_air = True
                self._time_take_off = self.get_sim_time()
                self._robot_pose_take_off = np.array(self.robot.GetBasePosition())
                self._robot_orientation_take_off = np.array(self.robot.GetBaseOrientationRollPitchYaw())
                self._robot_orientation_take_off_quat = np.array(self.robot.GetBaseOrientation())
                self._jump_init_height = self._robot_pose_take_off[2]
            else:
                # flight_time_reward = self.get_sim_time() - self._time_take_off
                pass
        else:
            if self._all_feet_in_the_air:
                self._max_flight_time = max(self.get_sim_time() - self._time_take_off, self._max_flight_time)
                # Compute forward distance according to local frame (starting at take off)
                rotation_matrix = R.from_euler("z", -self._robot_orientation_take_off[2], degrees=False).as_matrix()
                translation = -self._robot_pose_take_off
                pos_abs = np.array(self.robot.GetBasePosition())
                pos_relative = pos_abs + translation
                pos_relative = pos_relative @ rotation_matrix
                self._max_forward_distance = max(pos_relative[0], self._max_forward_distance)

            self._all_feet_in_the_air = False

        _, pitch, yaw = self.robot.GetBaseOrientationRollPitchYaw()
        self._max_yaw = max(np.abs(yaw), self._max_yaw)
        self._max_pitch = max(np.abs(pitch), self._max_pitch)
        delta_height = max(self.robot.GetBasePosition()[2] - self._init_height, 0.0)
        self._max_height = max(self._max_height, delta_height)
        # print(self._max_height)

        return flight_time_reward

    def _reward_fwd_locomotion(self):
        """Reward progress in the positive world x direction."""
        current_base_position = self.robot.GetBasePosition()
        forward_reward = current_base_position[0] - self._last_base_position[0]
        self._last_base_position = current_base_position
        # clip reward to MAX_FWD_VELOCITY (avoid exploiting simulator dynamics)
        if MAX_FWD_VELOCITY < np.inf:
            # calculate what max distance can be over last time interval based on max allowed fwd velocity
            max_dist = MAX_FWD_VELOCITY * (self._time_step * self._action_repeat)
            forward_reward = min(forward_reward, max_dist)

        v_x = max(min(self.robot.GetBaseLinearVelocity()[0], MAX_FWD_VELOCITY), -MAX_FWD_VELOCITY)
        v_y = max(min(self.robot.GetBaseLinearVelocity()[1], MAX_FWD_VELOCITY), -MAX_FWD_VELOCITY)
        v = np.array([v_x, v_y])

        return 1e-2 * np.linalg.norm(v)

    def _reward_lr_course(self):
        """Implement your reward function here. How will you improve upon the above?"""
        current_base_position = np.array(self.robot.GetBasePosition())
        v_ = (current_base_position[:2] - self._last_base_position[:2]) / (self._time_step * self._action_repeat)
        self._last_base_position = current_base_position
        # clip reward to MAX_FWD_VELOCITY (avoid exploiting simulator dynamics)
        if MAX_FWD_VELOCITY < np.inf:
            v_x = max(min(v_[0], MAX_FWD_VELOCITY), -MAX_FWD_VELOCITY)
            v_y = max(min(v_[1], MAX_FWD_VELOCITY), -MAX_FWD_VELOCITY)
            v = np.array([v_x, v_y])

        e = self._v_des - v

        v_norm = max(np.linalg.norm(v), 0.001)  # prevent CoT explosion for very small movements
        P = 0.0
        for i in range(len(self._dt_motor_torques)):
            P += np.array(self._dt_motor_torques[i]).dot(np.array(self._dt_motor_velocities[i]))
        P = P / len(self._dt_motor_torques)
        CoT = P / (5.0 * 10.0 * v_norm)

        ddq = (np.array(self._dt_motor_velocities[-1]) - np.array(self._dt_motor_velocities[0])) / (
            self._time_step * self._action_repeat
        )

        rpy = self.robot.GetBaseOrientationRollPitchYaw()

        r = (
            0.1 * v_norm
            - 0.0001 * abs(CoT)
            + 0.1 * np.exp(-1.0 * ddq.dot(ddq))
            - 0.05 * abs(rpy[2])
            + np.exp(-10.0 * e.dot(e))
        )

        return r

    def _reward(self):
        """Get reward depending on task"""
        if self._TASK_ENV == "FWD_LOCOMOTION":
            return self._reward_fwd_locomotion()
        elif self._TASK_ENV == "LR_COURSE_TASK":
            return self._reward_lr_course()
        elif self._TASK_ENV == "JUMPING_TASK":
            return self._reward_jumping()
        elif self._TASK_ENV == "JUMPING_ON_PLACE_TASK":
            return self._reward_jumping_on_place()
        elif self._TASK_ENV == "JUMPING_ON_PLACE_HEIGHT_TASK":
            return self._reward_jumping_on_place_height()
        elif self._TASK_ENV == "JUMPING_ON_PLACE_ABS_HEIGHT_TASK":
            return self._reward_jumping_on_place_abs_height()
        elif self._TASK_ENV == "LANDING_TASK":
            pass
        elif self._TASK_ENV == "JUMPING_FORWARD":
            return self._reward_jumping_forward()
        else:
            raise ValueError("This task mode is not implemented yet.")

    def reward_end_episode(self, reward):
        """add bonus and malus at the end of the episode"""
        if self._TASK_ENV == "JUMPING_TASK":
            return self._reward_end_jumping(reward)
        elif self._TASK_ENV == "JUMPING_ON_PLACE_TASK":
            return self._reward_end_jumping_on_place(reward)
        elif self._TASK_ENV in ["JUMPING_ON_PLACE_HEIGHT_TASK", "JUMPING_ON_PLACE_ABS_HEIGHT_TASK"]:
            return self._reward_end_jumping_on_place_height(reward)
        elif self._TASK_ENV == "LANDING_TASK":
            pass
        elif self._TASK_ENV == "JUMPING_FORWARD":
            return self._reward_end_jumping_forward(reward)
        else:
            # do nothing
            return reward

    def _reward_end_jumping_forward(self, reward):
        """Add bonus and malus at the end of the episode for jumping forawrd task"""
        if self._termination():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        max_fwd = 0.2
        max_height = 0.2
        normalize_fwd_distance = self._max_forward_distance / max_fwd
        reward += normalize_fwd_distance
        reward += normalize_fwd_distance * 0.05 * np.exp(-self._max_yaw**2 / 0.1)  # orientation
        reward += normalize_fwd_distance * 0.05 * np.exp(-self._max_roll**2 / 0.1)  # orientation

        reward += 0.1 * self._max_height / max_height  # bonus max_height

        if self._max_forward_distance > 0 and not self._termination():
            # Alive bonus proportional to the risk taken
            reward += normalize_fwd_distance * 0.1
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward

    def _reward_end_jumping(self, reward):
        """Add bonus and malus at the end of the episode for jumping task"""
        if self._termination():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        reward += self._max_flight_time
        max_distance = 0.2
        # Normalize forward distance reward
        reward += 0.1 * self._max_forward_distance / max_distance
        if self._max_flight_time > 0 and not self._termination():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * self._max_flight_time
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward

    def _reward_end_jumping_on_place(self, reward):
        """Add bonus and malus at the end of the episode for jumping on place task"""
        if self._termination():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        reward += self._max_flight_time
        reward += 0.05 * np.exp(-self._max_yaw**2 / 0.01)  # orientation
        reward += 0.05 * np.exp(-self._max_pitch**2 / 0.01)  # orientation

        reward += 0.1 * np.exp(-self._max_forward_distance**2 / 0.05)  # be on place

        if self._max_flight_time > 0 and not self._termination():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * self._max_flight_time
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward

    def _reward_end_jumping_on_place_height(self, reward):
        """Add bonus and malus at the end of the episode for jumping on place task"""
        if self._termination():
            # Malus for crashing
            # Optionally: no reward in case of crash
            reward -= 0.08
        max_height = 0.4
        max_height_normalized = self._max_height / max_height
        reward += max_height_normalized
        reward += max_height_normalized * 0.05 * np.exp(-self._max_yaw**2 / 0.01)  # orientation
        reward += max_height_normalized * 0.05 * np.exp(-self._max_pitch**2 / 0.01)  # orientation

        reward += max_height_normalized * 0.1 * np.exp(-self._max_forward_distance**2 / 0.05)  # be on place

        if self._max_height > 0 and not self._termination():
            # Alive bonus proportional to the risk taken
            reward += 0.1 * max_height_normalized
        # print(f"Forward dist: {self._max_forward_distance}")
        return reward

    ######################################################################################
    # Step simulation, map policy network actions to joint commands, etc.
    ######################################################################################
    def _interpolate_actions(self, action, substep_count):
        """If enabled, interpolates between the current and previous actions.

        Args:
        action: current action.
        substep_count: the step count should be between [0, self.__action_repeat).

        Returns:
        If interpolation is enabled, returns interpolated action depending on
        the current action repeat substep.
        """
        if self._enable_action_interpolation and self._last_action is not None:
            interp_fraction = float(substep_count + 1) / self._action_repeat
            interpolated_action = self._last_action + interp_fraction * (action - self._last_action)
        else:
            interpolated_action = action

        return interpolated_action

    def _transform_action_to_motor_command(self, action):
        """Map actions from RL (i.e. in [-1,1]) to joint commands based on motor_control_mode."""
        # clip actions to action bounds
        action = np.clip(action, -self._action_bound - ACTION_EPS, self._action_bound + ACTION_EPS)
        if self._motor_control_mode == "PD":
            action = self._scale_helper(
                action, self._robot_config.RL_LOWER_ANGLE_JOINT, self._robot_config.RL_UPPER_ANGLE_JOINT
            )
            action = np.clip(action, self._robot_config.RL_LOWER_ANGLE_JOINT, self._robot_config.RL_UPPER_ANGLE_JOINT)
            if self._enable_action_clipping:
                action = self._clip_motor_commands(action)
        elif self._motor_control_mode == "CARTESIAN_PD":
            action = self.ScaleActionToCartesianPos(action)
            # Here the clipping happens inside ScaleActionToCartesianPos
        elif self._motor_control_mode == "INVKIN_CARTESIAN_PD":
            action = self._invkin_action_to_command(action)
            action = np.clip(action, self._robot_config.RL_LOWER_ANGLE_JOINT, self._robot_config.RL_UPPER_ANGLE_JOINT)
        else:
            raise ValueError("RL motor control mode" + self._motor_control_mode + "not implemented yet.")
        return action

    def _invkin_action_to_command(self, actions):
        u = np.clip(actions, -1, 1)
        des_foot_pos = self._scale_helper(
            u, self._robot_config.RL_LOWER_CARTESIAN_POS, self._robot_config.RL_UPPER_CARTESIAN_POS
        )
        q_des = np.array(
            list(map(lambda i: self.robot.ComputeInverseKinematics(i, des_foot_pos[3 * i : 3 * (i + 1)]), range(4)))
        )
        return q_des.flatten()

    def _compute_action_from_command(self, command, min_command, max_command):
        """
        Helper to linearly scale from [min_command, max_command] to [-1, 1].
        """
        return -1 + 2 * (command - min_command) / (max_command - min_command)

    def _scale_helper(self, action, lower_lim, upper_lim):
        """Helper to linearly scale from [-1,1] to lower/upper limits."""
        new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
        return np.clip(new_a, lower_lim, upper_lim)

    def ScaleActionToCartesianPos(self, actions):
        """Scale RL action to Cartesian PD ranges.
        Edit ranges, limits etc., but make sure to use Cartesian PD to compute the torques.
        """
        # clip RL actions to be between -1 and 1 (standard RL technique)
        # print(actions)
        u = np.clip(actions, -1, 1)
        # scale to corresponding desired foot positions (i.e. ranges in x,y,z we allow the agent to choose foot positions)
        # [TODO: edit (do you think these should these be increased? How limiting is this?)]
        # scale_array = np.array([0.2, 0.05, 0.15] * 4)  # [0.1, 0.05, 0.08]*4)
        # add to nominal foot position in leg frame (what are the final ranges?)
        # des_foot_pos = self._robot_config.NOMINAL_FOOT_POS_LEG_FRAME + scale_array * u
        des_foot_pos = self._scale_helper(
            u, self._robot_config.RL_LOWER_CARTESIAN_POS, self._robot_config.RL_UPPER_CARTESIAN_POS
        )
        # get Cartesian kp and kd gains (can be modified)
        kpCartesian = self._robot_config.kpCartesian
        kdCartesian = self._robot_config.kdCartesian
        # get current motor velocities
        dq = self.robot.GetMotorVelocities()

        action = np.zeros(12)
        for i in range(4):
            # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
            J, xyz = self.robot.ComputeJacobianAndPosition(i)
            # Get current foot velocity in leg frame (Equation 2)
            dxyz = J @ dq[3 * i : 3 * (i + 1)]
            delta_foot_pos = xyz - des_foot_pos[3 * i : 3 * (i + 1)]

            # clamp the motor command by the joint limit, in case weired things happens
            if self._enable_action_clipping:
                delta_foot_pos = np.clip(
                    delta_foot_pos,
                    -self._robot_config.MAX_CARTESIAN_FOOT_POS_CHANGE_PER_STEP,
                    self._robot_config.MAX_CARTESIAN_FOOT_POS_CHANGE_PER_STEP,
                )

            F_foot = -kpCartesian @ delta_foot_pos - kdCartesian @ dxyz
            # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
            tau = J.T @ F_foot
            action[3 * i : 3 * (i + 1)] = tau  # TODO: add white noise
        return action

    def _clip_motor_commands(self, des_angles):
        """Clips motor commands.

        Args:
        des_angles: np.array. They are the desired motor angles (for PD control only)

        Returns:
        Clipped motor commands.
        """

        # clamp the motor command by the joint limit, in case weired things happens
        if self._motor_control_mode in ["INVKIN_CARTESIAN_PD", "PD"]:
            max_angle_change = self._robot_config.MAX_MOTOR_ANGLE_CHANGE_PER_STEP
            current_motor_angles = self.robot.GetMotorAngles()
            motor_commands = np.clip(
                des_angles, current_motor_angles - max_angle_change, current_motor_angles + max_angle_change
            )
            return motor_commands
        else:
            raise ValueError(f"Clipping angles available for PD control only, not in {self._motor_control_mode}")

    def adapt_action_dim_for_robot(self, action):
        """
        In according to the selected action spaced the action is converted to have the
        same dimension as the default one -> 12 in the properly way.
        """
        assert (
            len(action) == self._action_dim
        ), f"action dimension is {len(action)}, action space has dimension {self._action_dim} "
        if self._action_space_mode == "DEFAULT":
            return action
        elif self._action_space_mode == "SYMMETRIC":
            if self._motor_control_mode in ["TORQUE", "PD"]:
                symm_idx = 0  #  hip angle
            elif self._motor_control_mode in ["INVKIN_CARTESIAN_PD", "CARTESIAN_PD"]:
                symm_idx = 1  #  y cartesian pos
            else:
                raise ValueError(f"motor control mode {self._motor_control_mode} not implemented yet")
            leg_FR = action[0:3]
            leg_RR = action[3:6]

            leg_FL = np.copy(leg_FR)
            leg_FL[symm_idx] = -leg_FR[symm_idx]

            leg_RL = np.copy(leg_RR)
            leg_RL[symm_idx] = -leg_RR[symm_idx]

            leg = np.concatenate((leg_FR, leg_FL, leg_RR, leg_RL))

        elif self._action_space_mode == "SYMMETRIC_NO_HIP":
            if self._motor_control_mode == "PD":
                leg_FR = action[0:2]
                leg_RR = action[2:4]

                leg_FL = leg_FR = np.concatenate(([0], leg_FR))
                leg_RL = leg_RR = np.concatenate(([0], leg_RR))

                leg = np.concatenate((leg_FR, leg_FL, leg_RR, leg_RL))

            elif self._motor_control_mode in ["INVKIN_CARTESIAN_PD", "CARTESIAN_PD"]:
                # no motion along y
                leg_FL = leg_FR = np.array([action[0], 0, action[1]])
                leg_RL = leg_RR = np.array([action[2], 0, action[3]])

                leg = np.concatenate((leg_FR, leg_FL, leg_RR, leg_RL))

        else:
            raise ValueError(f"action space mode {self._action_space_mode} not implemented yet")

        return leg

    def adapt_command_to_action_dim(self, command):
        """Given a command it's been converted to a command with the same dimension
            of the Action Space. Actually used in LandingWrapper.

        Args:
            command (np.Array): the command you would like to apply to the robot
        """
        assert len(command) == 12, "command has not right dimensions. Should be (12,)"
        if self._action_space_mode == "DEFAULT":
            return command
        elif self._action_space_mode == "SYMMETRIC":
            leg_FR = command[0:3]
            leg_RR = command[6:9]
            return np.concatenate((leg_FR, leg_RR))
        elif self._action_space_mode == "SYMMETRIC_NO_HIP":
            leg_FR = command[1:3]
            leg_RR = command[7:9]
            return np.concatenate((leg_FR, leg_RR))
        else:
            raise ValueError(f"action space {self._action_space_mode} not supported yet")

    def step(self, action):
        """Step forward the simulation, given the action."""
        curr_act = action.copy()
        if self._enable_action_filter:
            curr_act = self._filter_action(curr_act)
        curr_act = self.adapt_action_dim_for_robot(curr_act)
        # save motor torques and velocities to compute power in reward function
        self._dt_motor_torques = []
        self._dt_motor_velocities = []
        for sub_step in range(self._action_repeat):
            if self._isRLGymInterface:
                if self._enable_action_interpolation:
                    curr_act = self._interpolate_actions(action, sub_step)
                proc_action = self._transform_action_to_motor_command(curr_act)
            else:
                proc_action = curr_act
            self.robot.ApplyAction(proc_action)
            self._pybullet_client.stepSimulation()
            # for joint velocity estimation
            self._last_joint_config = self._actual_joint_config
            self._actual_joint_config = self.robot.GetMotorAngles()
            self._sim_step_counter += 1
            self._dt_motor_torques.append(self.robot.GetMotorTorques())
            self._dt_motor_velocities.append(self.robot.GetMotorVelocities())

            if self._is_render:
                self._render_step_helper()

        self._last_action = curr_act
        self._env_step_counter += 1
        reward = self._reward()
        done = False
        infos = {"base_pos": self.robot.GetBasePosition()}

        if self._termination() or self.get_sim_time() > self._MAX_EP_LEN:
            infos["TimeLimit.truncated"] = not self._termination()
            done = True

        # Update the actual reward at the end of the episode with bonus or malus
        if done:
            reward = self.reward_end_episode(reward)
            # print(reward)
            
        self._robot_sensors._on_step()
        obs = self.get_observation()

        return obs, reward, done, infos

    ###################################################
    # Filtering to smooth actions
    ###################################################
    def _build_action_filter(self):
        sampling_rate = 1 / (self._time_step * self._action_repeat)
        num_joints = self._action_dim
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate, num_joints=num_joints)
        if self._enable_springs:
            a_filter.highcut = 2.5
        return a_filter

    def _reset_action_filter(self):
        self._action_filter.reset()

    def _filter_action(self, action):
        filtered_action = self._action_filter.filter(action)
        return filtered_action

    def _init_filter(self):
        # initialize the filter history, since resetting the filter will fill
        # the history with zeros and this can cause sudden movements at the start
        # of each episode
        default_action = self._compute_first_actions()
        self._action_filter.init_history(default_action)

    def _compute_first_actions(self):
        if self._motor_control_mode == "PD":
            # init_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS
            # default_action = self._map_command_to_action(init_angles)
            default_action = np.array([0] * self._action_dim)
        elif self._motor_control_mode in ["INVKIN_CARTESIAN_PD", "CARTESIAN_PD"]:
            # go toward NOMINAL_FOOT_POS_LEG_FRAME
            default_action = np.array([0] * self._action_dim)
        else:
            raise ValueError(f"The motor control mode {self._motor_control_mode} is not implemented for RL")

        return np.clip(default_action, -1, 1)

    ######################################################################################
    # Reset
    ######################################################################################
    def reset(self):
        """Set up simulation environment."""
        mu_min = 0.5
        if self._hard_reset:
            # set up pybullet simulation
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            self.plane = self._pybullet_client.loadURDF(
                pybullet_data.getDataPath() + "/plane.urdf", basePosition=[80, 0, 0]
            )  # to extend available running space (shift)
            self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
            self._pybullet_client.setGravity(0, 0, -9.8)
            self.robot = quadruped.Quadruped(
                pybullet_client=self._pybullet_client,
                robot_config=self._robot_config,
                motor_control_mode=self._motor_control_mode,
                on_rack=self._on_rack,
                render=self._is_render,
                enable_springs=self._enable_springs,
            )

            if self._add_noise:
                ground_mu_k = mu_min + (1 - mu_min) * np.random.random()
                self._ground_mu_k = ground_mu_k
                self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
                if self._is_render:
                    print("ground friction coefficient is", ground_mu_k)

            if self._using_test_env:
                pass
                # self.add_random_boxes()
                # self._add_base_mass_offset()
        else:
            self.robot.Reset(reload_urdf=False)

        self._robot_sensors._reset(self.robot)

        self._env_step_counter = 0
        self._sim_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self.init_action = self._compute_init_action()

        if self._is_render:
            self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])

        if self._enable_action_filter:
            self._reset_action_filter()
            self._init_filter()

        self._settle_robot()  # Settle robot after being spawned

        self._init_task_variables()

        # for joint velocity estimation
        self._last_joint_config = self.robot.GetMotorAngles()
        self._actual_joint_config = self.robot.GetMotorAngles()

        self._last_action = np.zeros(self._action_dim)
        if self._is_record_video:
            self.recordVideoHelper()

        return self.get_observation()

    def _settle_robot_by_action(self):
        """Settle robot in according to the used motor control mode in RL interface"""
        if self._is_render:
            time.sleep(0.2)
        for _ in range(1500):
            proc_action = self._transform_action_to_motor_command(self.init_action)
            self.robot.ApplyAction(proc_action)
            if self._is_render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()

    def _settle_robot_by_PD(self):
        """Settle robot and add noise to init configuration."""
        # change to PD control mode to set initial position, then set back..
        tmp_save_motor_control_mode_ENV = self._motor_control_mode
        tmp_save_motor_control_mode_ROB = self.robot._motor_control_mode
        self._motor_control_mode = "PD"
        self.robot._motor_control_mode = "PD"
        try:
            tmp_save_motor_control_mode_MOT = self.robot._motor_model._motor_control_mode
            self.robot._motor_model._motor_control_mode = "PD"
        except:
            pass
        init_motor_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS
        if self._is_render:
            time.sleep(0.2)
        for _ in range(1500):
            self.robot.ApplyAction(init_motor_angles)
            if self._is_render:
                time.sleep(0.001)
            self._pybullet_client.stepSimulation()

        # set control mode back
        self._motor_control_mode = tmp_save_motor_control_mode_ENV
        self.robot._motor_control_mode = tmp_save_motor_control_mode_ROB
        try:
            self.robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT
        except:
            pass

    def _settle_robot(self):
        if self._isRLGymInterface:
            self._settle_robot_by_action()
        else:
            self._settle_robot_by_PD()

    def compute_action_from_command(self, command):
        if self._motor_control_mode == "PD":
            action = self._compute_action_from_command(
                command, self._robot_config.RL_LOWER_ANGLE_JOINT, self._robot_config.RL_UPPER_ANGLE_JOINT
            )
        elif self._motor_control_mode in ["CARTESIAN_PD", "INVKIN_CARTESIAN_PD"]:
            action = self._compute_action_from_command(
                command, self._robot_config.RL_LOWER_CARTESIAN_POS, self._robot_config.RL_UPPER_CARTESIAN_POS
            )
        else:
            raise ValueError(f"motor control mode {self._motor_control_mode} not supported yet in RLGymInterface.")
        return action

    def _compute_init_action(self):
        if self._motor_control_mode == "PD":
            command = self._robot_config.INIT_MOTOR_ANGLES
            init_action = self.compute_action_from_command(command)
        elif self._motor_control_mode in ["CARTESIAN_PD", "INVKIN_CARTESIAN_PD"]:
            command = self._robot_config.NOMINAL_FOOT_POS_LEG_FRAME
            init_action = self.compute_action_from_command(command)
        elif self._motor_control_mode == 'TORQUE' and not self._isRLGymInterface:
            init_action = None
        else:
            raise ValueError(f"motor control mode {self._motor_control_mode} not supported yet in RLGymInterface.")
        return init_action

    def _init_task_variables(self):
        if self._TASK_ENV == "JUMPING_TASK":
            self._init_variables_jumping()
        elif self._TASK_ENV in ["LR_COURSE_TASK", "FWD_LOCOMOTION"]:
            self._v_des = np.array([2.0, 0.0])  # (np.random.uniform(), 2.*np.random.uniform()-1.)
        elif self._TASK_ENV == "JUMPING_ON_PLACE_TASK":
            self._init_variables_jumping_on_place()
        elif self._TASK_ENV in ["JUMPING_ON_PLACE_HEIGHT_TASK", "JUMPING_ON_PLACE_ABS_HEIGHT_TASK"]:
            self._init_variables_jumping_on_place_height()
        elif self._TASK_ENV == "LANDING_TASK":
            self._init_variables_landing()
        elif self._TASK_ENV == "JUMPING_FORWARD":
            self._init_variables_jumping_forward()
        else:
            raise ValueError(f"the task {self._TASK_ENV} is not implemented yet")

    def _init_variables_jumping_forward(self):
        self._v_des = np.array([3.0])
        self._init_height = self.robot.GetBasePosition()[2]
        self._all_feet_in_the_air = False
        self._time_take_off = self.get_sim_time()
        self._robot_pose_take_off = self.robot.GetBasePosition()
        self._robot_orientation_take_off = self.robot.GetBaseOrientationRollPitchYaw()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0
        self._max_yaw = 0.0
        self._max_roll = 0.0
        self._jump_init_height = self._robot_pose_take_off[2]
        self._max_height = 0.0

    def _init_variables_landing(self):
        pass

    def _init_variables_jumping(self):
        # For the jumping task
        self._v_des = np.array([2.0, 0.0])  # (np.random.uniform(), 2.*np.random.uniform()-1.)
        self._init_height = self.robot.GetBasePosition()[2]
        self._all_feet_in_the_air = False
        self._time_take_off = self.get_sim_time()
        self._robot_pose_take_off = self.robot.GetBasePosition()
        self._robot_orientation_take_off = self.robot.GetBaseOrientationRollPitchYaw()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0

    def _init_variables_jumping_on_place(self):
        self._v_des = np.array([3.0])
        self._init_height = self.robot.GetBasePosition()[2]
        self._all_feet_in_the_air = False
        self._time_take_off = self.get_sim_time()
        self._robot_pose_take_off = self.robot.GetBasePosition()
        self._robot_orientation_take_off = self.robot.GetBaseOrientationRollPitchYaw()
        self._robot_orientation_take_off_quat = self.robot.GetBaseOrientation()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0
        self._max_yaw = 0.0
        self._max_pitch = 0.0

    def _init_variables_jumping_on_place_height(self):
        self._v_des = np.array([3.0])
        self._init_height = self.robot.GetBasePosition()[2]
        self._all_feet_in_the_air = False
        self._time_take_off = self.get_sim_time()
        self._robot_pose_take_off = self.robot.GetBasePosition()
        self._robot_orientation_take_off = self.robot.GetBaseOrientationRollPitchYaw()
        self._robot_orientation_take_off_quat = self.robot.GetBaseOrientation()
        self._max_flight_time = 0.0
        self._max_forward_distance = 0.0
        self._max_yaw = 0.0
        self._max_pitch = 0.0
        self._jump_init_height = self._robot_pose_take_off[2]
        self._max_height = 0.0

    ######################################################################################
    # Render, record videos, bookkeping, and misc pybullet helpers.
    ######################################################################################
    def startRecordingVideo(self, name):
        self.videoLogID = self._pybullet_client.startStateLogging(self._pybullet_client.STATE_LOGGING_VIDEO_MP4, name)

    def stopRecordingVideo(self):
        self._pybullet_client.stopStateLogging(self.videoLogID)

    def close(self):
        if self._is_record_video:
            self.stopRecordingVideo()
        self._pybullet_client.disconnect()

    def recordVideoHelper(self, extra_filename=None):
        """Helper to record video, if not already, or end and start a new one"""
        # If no ID, this is the first video, so make a directory and start logging
        if self.videoLogID == None:
            directoryName = VIDEO_LOG_DIRECTORY
            assert isinstance(directoryName, str)
            os.makedirs(directoryName, exist_ok=True)
            self.videoDirectory = directoryName
        else:
            # stop recording and record a new one
            self.stopRecordingVideo()

        if extra_filename is not None:
            output_video_filename = (
                self.videoDirectory
                + "/"
                + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")
                + extra_filename
                + ".MP4"
            )
        else:
            output_video_filename = (
                self.videoDirectory + "/" + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + ".MP4"
            )
        logID = self.startRecordingVideo(output_video_filename)
        self.videoLogID = logID

    def configure(self, args):
        self._args = args

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _render_step_helper(self):
        """Helper to configure the visualizer camera during step()."""
        # Sleep, otherwise the computation takes less time than real time,
        # which will make the visualization like a fast-forward video.
        time_spent = time.time() - self._last_frame_time
        self._last_frame_time = time.time()
        # time_to_sleep = self._action_repeat * self._time_step - time_spent
        time_to_sleep = self._time_step - time_spent
        if time_to_sleep > 0 and (time_to_sleep < self._time_step):
            time.sleep(time_to_sleep)

        base_pos = self.robot.GetBasePosition()
        camInfo = self._pybullet_client.getDebugVisualizerCamera()
        curTargetPos = camInfo[11]
        distance = camInfo[10]
        yaw = camInfo[8]
        pitch = camInfo[9]
        targetPos = [0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1], curTargetPos[2]]
        self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)

    def _configure_visualizer(self):
        """Remove all visualizer borders, and zoom in"""
        # default rendering options
        self._render_width = 333
        self._render_height = 480
        self._cam_dist = 1.5
        self._cam_yaw = 20
        self._cam_pitch = -20
        # get rid of visualizer things
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI, 0)

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.robot.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def addLine(self, lineFromXYZ, lineToXYZ, lifeTime=0, color=[1, 0, 0]):
        """Add line between point A and B for duration lifeTime"""
        self._pybullet_client.addUserDebugLine(lineFromXYZ, lineToXYZ, lineColorRGB=color, lifeTime=lifeTime)

    def get_sim_time(self):
        """Get current simulation time."""
        return self._sim_step_counter * self._time_step

    def get_motor_control_mode(self):
        """Get current motor control mode."""
        return self._motor_control_mode

    def get_robot_config(self):
        """Get current robot config."""
        return self._robot_config

    def are_springs_enabled(self):
        """Get boolean specifying if springs are enabled or not."""
        return self._enable_springs

    def scale_rand(self, num_rand, low, high):
        """scale number of rand numbers between low and high"""
        return low + np.random.random(num_rand) * (high - low)

    def add_random_boxes(self, num_rand=100, z_height=0.04):
        """Add random boxes in front of the robot in x [0.5, 20] and y [-3,3]"""
        # x location
        x_low, x_upp = 0.5, 20
        # y location
        y_low, y_upp = -3, 3
        # block dimensions
        block_x_min, block_x_max = 0.1, 1
        block_y_min, block_y_max = 0.1, 1
        z_low, z_upp = 0.005, z_height
        # block orientations
        roll_low, roll_upp = -0.01, 0.01
        pitch_low, pitch_upp = -0.01, 0.01
        yaw_low, yaw_upp = -np.pi, np.pi

        x = x_low + np.random.random(num_rand) * (x_upp - x_low)
        y = y_low + np.random.random(num_rand) * (y_upp - y_low)
        z = z_low + np.random.random(num_rand) * (z_upp - z_low)
        block_x = self.scale_rand(num_rand, block_x_min, block_x_max)
        block_y = self.scale_rand(num_rand, block_y_min, block_y_max)
        roll = self.scale_rand(num_rand, roll_low, roll_upp)
        pitch = self.scale_rand(num_rand, pitch_low, pitch_upp)
        yaw = self.scale_rand(num_rand, yaw_low, yaw_upp)
        # loop through
        for i in range(num_rand):
            sh_colBox = self._pybullet_client.createCollisionShape(
                self._pybullet_client.GEOM_BOX, halfExtents=[block_x[i] / 2, block_y[i] / 2, z[i] / 2]
            )
            orn = self._pybullet_client.getQuaternionFromEuler([roll[i], pitch[i], yaw[i]])
            block2 = self._pybullet_client.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=sh_colBox, basePosition=[x[i], y[i], z[i] / 2], baseOrientation=orn
            )
            # set friction coeff
            self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

        # add walls
        orn = self._pybullet_client.getQuaternionFromEuler([0, 0, 0])
        sh_colBox = self._pybullet_client.createCollisionShape(
            self._pybullet_client.GEOM_BOX, halfExtents=[x_upp / 2, 0.5, 0.5]
        )
        block2 = self._pybullet_client.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=sh_colBox, basePosition=[x_upp / 2, y_low, 0.5], baseOrientation=orn
        )
        block2 = self._pybullet_client.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=sh_colBox, basePosition=[x_upp / 2, -y_low, 0.5], baseOrientation=orn
        )

    def _add_base_mass_offset(self, spec_mass=None, spec_location=None):
        """Attach mass to robot base."""
        quad_base = np.array(self.robot.GetBasePosition())
        quad_ID = self.robot.quadruped

        offset_low = np.array([-0.15, -0.05, -0.05])
        offset_upp = np.array([0.15, 0.05, 0.05])
        if spec_location is None:
            block_pos_delta_base_frame = self.scale_rand(3, offset_low, offset_upp)
        else:
            block_pos_delta_base_frame = np.array(spec_location)
        if spec_mass is None:
            base_mass = 8 * np.random.random()
        else:
            base_mass = spec_mass
        if self._is_render:
            print("=========================== Random Mass:")
            print("Mass:", base_mass, "location:", block_pos_delta_base_frame)
            # if rendering, also want to set the halfExtents accordingly
            # 1 kg water is 0.001 cubic meters
            boxSizeHalf = [(base_mass * 0.001) ** (1 / 3) / 2] * 3
            translationalOffset = [0, 0, 0.1]
        else:
            boxSizeHalf = [0.05] * 3
            translationalOffset = [0] * 3

        sh_colBox = self._pybullet_client.createCollisionShape(
            self._pybullet_client.GEOM_BOX, halfExtents=boxSizeHalf, collisionFramePosition=translationalOffset
        )
        base_block_ID = self._pybullet_client.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=sh_colBox,
            basePosition=quad_base + block_pos_delta_base_frame,
            baseOrientation=[0, 0, 0, 1],
        )

        cid = self._pybullet_client.createConstraint(
            quad_ID,
            -1,
            base_block_ID,
            -1,
            self._pybullet_client.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            -block_pos_delta_base_frame,
        )
        # disable self collision between box and each link
        for i in range(-1, self._pybullet_client.getNumJoints(quad_ID)):
            self._pybullet_client.setCollisionFilterPair(quad_ID, base_block_ID, i, -1, 0)


def test_env():

    env_config = {
        "render": True,
        "on_rack": False,
        "motor_control_mode": "INVKIN_CARTESIAN_PD",
        "action_repeat": 10,
        "enable_springs": True,
        "add_noise": False,
        "enable_action_interpolation": False,
        "enable_action_clipping": False,
        "enable_action_filter": True,
        "task_env": "JUMPING_FORWARD",
        "observation_space_mode": "DEFAULT",
        "action_space_mode": "SYMMETRIC",
        "enable_joint_velocity_estimate": True,
    }

    env = QuadrupedGymEnv(**env_config)
    
    # env = RestWrapper(env)
    # env = LandingWrapper(env)

    sim_steps = 500
    action_dim = env.get_action_dim()
    obs = env.reset()
    for i in range(sim_steps):
        action = np.random.rand(action_dim) * 2 - 1
        # action = np.full(action_dim, 0)
        obs, reward, done, info = env.step(action)
    env.close()
    print("end")


if __name__ == "__main__":
    # test out some functionalities
    test_env()
    os.sys.exit()
