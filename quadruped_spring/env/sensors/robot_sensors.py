import numpy as np

from quadruped_spring.env.sensors.sensor import Sensor


class SensorEncoderIMUBased(Sensor):
    """Sensor exploiting Encoder and IMU sensor."""

    def __init__(self, env):
        super().__init__(env)
        self._dummy_encoder = False
        self._dummy_IMU = False

    def _set_sensor(self, robot):
        super()._set_sensor(robot)
        if self._dummy_encoder:
            self._encoder._set_sensor(robot)
        if self._dummy_IMU:
            self._imu._set_sensor(robot)

    def _init_sensor(self, robot_config):
        super()._init_sensor(robot_config)
        self._get_encoder_imu()
        if self._dummy_encoder:
            self._encoder._init_sensor(robot_config)
        if self._dummy_IMU:
            self._imu._init_sensor(robot_config)

    def _update_sensor_info(self, high, low, noise_std):
        super()._update_sensor_info(high, low, noise_std)
        if self._dummy_IMU:
            self._imu._update_sensor_info()
        if self._dummy_encoder:
            self._encoder._update_sensor_info()

    def _reset_sensor(self):
        super()._reset_sensor()
        self._update_dummy_encoder_imu()

    def _on_step(self):
        super()._on_step()
        self._update_dummy_encoder_imu()

    def _get_encoder_imu(self):
        """
        Use the robot sensors if present otherwise create dummy ones.
        """
        robot_sensors = self._env._robot_sensors._sensor_list
        encoder_used = False
        imu_used = False
        for s in robot_sensors:
            if isinstance(s, JointPosition):
                encoder_used = True
                self._encoder = s
            if isinstance(s, OrientationRPY):
                imu_used = True
                self._imu = s
        if not encoder_used:
            self._encoder = JointPosition(self._env)
        if not imu_used:
            self._imu = OrientationRPY(self._env)
        self._dummy_encoder = not encoder_used
        self._dummy_IMU = not imu_used

    def _update_dummy_encoder_imu(self):
        if self._dummy_encoder:
            self._encoder._get_data()
            self._encoder._sample_noise()
        if self._dummy_IMU:
            self._imu._get_data()
            self._imu._sample_noise()

    def _get_encoder_measure(self):
        return self._encoder._read_dirty_data()

    def _get_imu_measure(self):
        return self._imu._read_dirty_data()
    

class BooleanContact(Sensor):
    """Boolean variables specifying if the feet are in contact with ground"""

    def __init__(self, env):
        super().__init__(env)
        self._name = "BoolContatc"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.CONTACT_BOOL_HIGH,
            low=self._robot_config.CONTACT_BOOL_LOW,
            noise_std=self._robot_config.CONTACT_BOOL_NOISE,
        )

    def _get_data(self):
        _, _, _, feet_in_contact = self._robot.GetContactInfo()
        self._data = np.array(feet_in_contact)

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()
        
        
class Height(Sensor):
    """robot height."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Height"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.HEIGHT_HIGH,
            low=self._robot_config.HEIGHT_LOW,
            noise_std=self._robot_config.HEIGHT_NOISE,
        )

    def _get_data(self):
        height = self._robot.GetBasePosition()[2]
        self._data = height

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class JointPosition(Sensor):
    """Joint Configuration"""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Encoder"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.JOINT_ANGLES_HIGH,
            low=self._robot_config.JOINT_ANGLES_LOW,
            noise_std=self._robot_config.JOINT_ANGLES_NOISE,
        )

    def _get_data(self):
        angles = self._robot.GetMotorAngles()
        self._data = angles

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class HeightRealistic(SensorEncoderIMUBased):
    """
    Robot base height obtained through direct kinematic.
    Not reliable while robot is flying.
    """

    def __init__(self, env):
        super().__init__(env)
        self._name = "Height"

    def _init_sensor(self, robot_config):
        super()._init_sensor(robot_config)
        # Geometric offsets from center of mass to Hip joint
        self._x_offset = self._robot_config.X_OFFSET
        self._y_offset = self._robot_config.Y_OFFSET
        self._feet_radius = 0.02

    def _update_sensor_info(self):
        super()._update_sensor_info(
            high=self._robot_config.HEIGHT_HIGH,
            low=self._robot_config.HEIGHT_LOW,
            # noise_std=self._robot_config.HEIGHT_NOISE,
            noise_std=0.0,  # The noise on the measure is inherited from joint angle and imu noise.
        )

    def _get_data(self):
        q = self._get_encoder_measure()
        q = self._env.robot.GetMotorAngles()
        rpy = self._get_imu_measure()
        # height = self._robot.GetBasePosition()[2]
        _, _, _, feet_in_contact = self._env.robot.GetContactInfo()
        feet_in_contact = np.asarray(feet_in_contact)
        # if np.all(feet_in_contact):
        #     height = self._compute_height_from_legs_average(q)
        # elif self._is_crossed_contact(feet_in_contact):
        #     height = self._compute_height_from_cross_average(q, feet_in_contact)
        # elif self._is_imu_contact(feet_in_contact):
        #     height = self._compute_height_from_imu(q, feet_in_contact)
        #     print(f'height_diff: {height - self._env.robot.GetBasePosition()[2]}')
        # else:
        #     print('using last data')
        #     height = self._last_data
        height = self._compute_height_from_imu(q, rpy, feet_in_contact)
        # print(f'height_diff: {height - self._env.robot.GetBasePosition()[2]}')
        self._data = height

    @staticmethod
    def _is_crossed_contact(feet_in_contact):
        a, b, c, d = feet_in_contact
        if (a and d) or (b and c):
            return True
        else:
            return False

    @staticmethod
    def _is_imu_contact(feet_in_contact):
        contact_list = [[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1]]
        ret = False
        for contact in contact_list:
            if np.all(feet_in_contact == np.asarray(contact)):
                ret = True
                break
        return ret

    def _compute_height_from_imu(self, q, rpy, feet_in_contact):
        robot = self._env.robot
        roll, pitch, _ = rpy
        contact_list = [[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1]]
        cases = dict(zip(["north", "east", "south", "west"], contact_list))
        leg_ids = list(range(self._env._robot_config.NUM_LEGS))
        h = 0
        avg = 0
        if np.all(feet_in_contact == cases["north"]):
            print("north")
            leg_id = [idx for idx in leg_ids if cases["north"][idx]]
            for i in leg_id:
                q_leg = q[3 * i : 3 * (i + 1)]
                _, pos = robot._compute_jacobian_and_position(q_leg, i)
                z = -pos[2]
                avg = avg + z + self._feet_radius
            avg = avg / 2
            h = avg + np.sin(pitch) * self._x_offset
        elif np.all(feet_in_contact == cases["south"]):
            print("south")
            leg_id = [idx for idx in leg_ids if cases["south"][idx]]
            for i in leg_id:
                q_leg = q[3 * i : 3 * (i + 1)]
                _, pos = robot._compute_jacobian_and_position(q_leg, i)
                print(i, pos)
                z = -pos[2]
                avg = avg + z + self._feet_radius
            avg = avg / 2
            h = avg - np.sin(pitch) * self._x_offset
        elif np.all(feet_in_contact == cases["east"]):
            print("east")
            leg_id = [idx for idx in leg_ids if cases["east"][idx]]
            for i in leg_id:
                q_leg = q[3 * i : 3 * (i + 1)]
                _, pos = robot._compute_jacobian_and_position(q_leg, i)
                z = -pos[2]
                avg = avg + z + self._feet_radius
            avg = avg / 2
            h = avg + np.sin(roll) * self._y_offset
        elif np.all(feet_in_contact == cases["west"]):
            print("west")
            leg_id = [idx for idx in leg_ids if cases["west"][idx]]
            for i in leg_id:
                q_leg = q[3 * i : 3 * (i + 1)]
                _, pos = robot._compute_jacobian_and_position(q_leg, i)
                z = -pos[2]
                avg = avg + z + self._feet_radius
            avg = avg / 2
            h = avg - np.sin(roll) * self._y_offset
        elif np.all(feet_in_contact == cases["south"]):
            print("south")
            leg_id = [idx for idx in leg_ids if cases["south"][idx]]
            for i in leg_id:
                q_leg = q[3 * i : 3 * (i + 1)]
                _, pos = robot._compute_jacobian_and_position(q_leg, i)
                print(i, pos)
                z = -pos[2]
                avg = avg + z + self._feet_radius
            avg = avg / 2
            h = avg - np.sin(pitch) * (self._x_offset)

        return h

    def _compute_height_from_cross_average(self, q, feet_in_contact):
        a, b, c, d = feet_in_contact
        if a and d:
            legs_id = [0, 3]
        elif b and c:
            legs_id = [1, 2]
        robot = self._env.robot
        avg = 0.0
        for i in legs_id:
            q_leg = q[3 * i : 3 * (i + 1)]
            _, pos = robot._compute_jacobian_and_position(q_leg, i)
            z = -pos[2]
            avg = avg + z + 0.02  # + feet_radius
        return avg / 2

    def _compute_height_from_legs_average(self, q):
        robot = self._env.robot
        avg = 0
        # feet_radius = 0.02  # Not needed, it would be just an offset.
        n_legs = robot._robot_config.NUM_LEGS
        for i in range(n_legs):
            q_leg = q[3 * i : 3 * (i + 1)]
            _, pos = robot._compute_jacobian_and_position(q_leg, i)
            z = -pos[2]
            avg = avg + z + 0.02  # + feet_radius
        return avg / n_legs

    def _reset_sensor(self):
        super()._reset_sensor()
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        super()._on_step()
        self._last_data = self._read_dirty_data()
        self._get_data()
        self._sample_noise()


class JointVelocity(Sensor):
    """Joint Vecloity"""

    def __init__(self, env):
        super().__init__(env)
        self._name = "JointVelocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.JOINT_VELOCITIES_HIGH,
            low=self._robot_config.JOINT_VELOCITIES_LOW,
            noise_std=self._robot_config.JOINT_VELOCITIES_NOISE,
        )

    def _get_data(self):
        velocities = self._robot.GetMotorVelocities()
        self._data = velocities

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class FeetPostion(Sensor):
    """Feet position in leg frame"""

    def __init__(self, env):
        super().__init__(env)
        self._name = "FeetPosition"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.FEET_POS_HIGH,
            low=self._robot_config.FEET_POS_LOW,
            noise_std=self._robot_config.FEET_POS_NOISE,
        )

    def _get_data(self):
        feet_pos, _ = self._robot.ComputeFeetPosAndVel()
        self._data = feet_pos

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()

    def __init__(self, env):
        super().__init__(env)
class FeetVelocity(Sensor):
    """Feet velocity in leg frame"""

    def __init__(self, env):
        super().__init__(env)
        self._name = "FeetVelocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.FEET_VEL_HIGH,
            low=self._robot_config.FEET_VEL_LOW,
            noise_std=self._robot_config.FEET_VEL_NOISE,
        )

    def _get_data(self):
        _, feet_vel = self._robot.ComputeFeetPosAndVel()
        self._data = feet_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class LinearVelocity(Sensor):
    """Base linear velocity."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Base Linear Velocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_LIN_HIGH,
            low=self._robot_config.VEL_LIN_LOW,
            noise_std=self._robot_config.VEL_LIN_NOISE,
        )

    def _get_data(self):
        lin_vel = self._robot.GetBaseLinearVelocity()
        self._data = lin_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class HeightVelocity(Sensor):
    """Base height linear velocity."""

    def __init__(self):
        super().__init__()
        self._name = "Base Height Velocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_LIN_HIGH[2],
            low=self._robot_config.VEL_LIN_LOW[2],
            noise_std=self._robot_config.VEL_LIN_NOISE[2],
        )

    def _get_data(self):
        height_vel = self._robot.GetBaseLinearVelocity()[2]
        self._data = height_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class HeightGround(Sensor):
    """Height base posiiton only if robot touches the ground."""

    def __init__(self):
        super().__init__()
        self._name = "Height Ground"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.HEIGHT_HIGH,
            low=self._robot_config.HEIGHT_LOW,
            noise_std=self._robot_config.HEIGHT_NOISE,
        )

    def _get_data(self):
        height = self._robot.GetBasePosition()[2]
        self._old_data = self._data
        if self._robot._is_flying():
            self._data = self._old_data
        else:
            self._data = height

    def _reset_sensor(self):
        self._old_data = self._robot.GetBasePosition()[2]
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class AngularVelocity(Sensor):
    """Base angular velocity."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Base Angular Velocity"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_ANG_HIGH,
            low=self._robot_config.VEL_ANG_LOW,
            noise_std=self._robot_config.VEL_ANG_NOISE,
        )

    def _get_data(self):
        ang_vel = self._robot.GetBaseAngularVelocity()
        self._data = ang_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class DesiredBaseLinearVelocityXZ(Sensor):
    """robot height."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Desired base linear velocity xz plane"
        self._desired_velocity = np.array([0.0, 0.01])

    def get_desired_velocity(self):
        return self._desired_velocity

    def set_desired_velocity(self, vel):
        self._desired_velocity = vel

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=np.array([0.0, 0.0]),
            low=np.array([0.0, 0.0]),
            noise_std=np.array([0.0, 0.0]),
        )

    def _get_data(self):
        self._data = self._desired_velocity

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class Quaternion(Sensor):
    """base_orientation (quaternion)"""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Quaternion"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.QUATERNION_HIGH,
            low=self._robot_config.QUATERNION_LOW,
            noise_std=self._robot_config.QUATERNION_NOISE,
        )

    def _get_data(self):
        base_orientation = self._robot.GetBaseOrientation()
        self._data = base_orientation

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class Pitch(Sensor):
    """pitch angle."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Pitch"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.PITCH_HIGH,
            low=self._robot_config.PITCH_LOW,
            noise_std=self._robot_config.PITCH_NOISE,
        )

    def _get_data(self):
        pitch_orientation = self._robot.GetBaseOrientationRollPitchYaw()[1]
        self._data = pitch_orientation

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class PitchRate(Sensor):
    """pitch orientation rate."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Pitch rate"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.PITCH_RATE_HIGH,
            low=self._robot_config.PITCH_RATE_LOW,
            noise_std=self._robot_config.PITCH_RATE_NOISE,
        )

    def _get_data(self):
        pitch_orientation_rate = self._robot.GetTrueBaseRollPitchYawRate()[1]
        self._data = pitch_orientation_rate

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class OrientationRPY(Sensor):
    """orientation Roll Pitch Yaw."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Orientation Roll Pitch Yaw"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.ORIENT_RPY_HIGH,
            low=self._robot_config.ORIENT_RPY_LOW,
            noise_std=self._robot_config.ORIENT_RPY_NOISE,
        )

    def _get_data(self):
        orientation = self._robot.GetBaseOrientationRollPitchYaw()
        self._data = orientation

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class LinearVelocity2D(Sensor):
    """Base linear velocity."""

    def __init__(self, env):
        super().__init__(env)
        self._name = "Base Linear Velocity xz plane"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_LIN_HIGH[[0, 2]],
            low=self._robot_config.VEL_LIN_LOW[[0, 2]],
            noise_std=self._robot_config.VEL_LIN_NOISE[[0, 2]],
        )

    def _get_data(self):
        lin_vel = self._robot.GetBaseLinearVelocity()[[0, 2]]
        self._data = lin_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class BaseHeightVelocity(Sensor):
    """Base height velocity."""

    def __init__(self):
        super().__init__()
        self._name = "Base Linear Velocity z direction"

    def _update_sensor_info(self):
        return super()._update_sensor_info(
            high=self._robot_config.VEL_LIN_HIGH[2],
            low=self._robot_config.VEL_LIN_LOW[2],
            noise_std=self._robot_config.VEL_LIN_NOISE[2],
        )

    def _get_data(self):
        lin_vel = self._robot.GetBaseLinearVelocity()[2]
        self._data = lin_vel

    def _reset_sensor(self):
        self._get_data()
        self._sample_noise()

    def _on_step(self):
        self._get_data()
        self._sample_noise()


class SensorList:
    """Manage all the robot sensors"""

    def __init__(self, sensor_list, env):
        if not isinstance(sensor_list, list):
            raise ValueError("Please use a list of sensors. Also if it is just one.")
        self._sensor_list = sensor_list
        self._env = env

    def _compute_obs_dim(self):
        dim = 0
        for s in self._sensor_list:
            dim += np.sum(np.array(s._dim))
        return dim

    def get_obs_dim(self):
        return self._obs_dim

    def _get_high_limits(self):
        high = []
        for s in self._sensor_list:
            high.append(s._high.flatten())
        return np.concatenate(high)

    def _get_low_limits(self):
        low = []
        for s in self._sensor_list:
            low.append(np.array(s._low).flatten())
        return np.concatenate(low)

    def get_obs(self):
        obs = {}
        for s in self._sensor_list:
            obs[s._name] = s._read_data()
        return obs

    def get_noisy_obs(self):
        obs = {}
        for s in self._sensor_list:
            obs[s._name] = s._read_dirty_data()
        return obs

    def _on_step(self):
        for s in self._sensor_list:
            s._on_step()

    def _reset(self, robot):
        for s in self._sensor_list:
            s._set_sensor(robot)
            s._reset_sensor()
        self._obs_dim = self._compute_obs_dim()

    def _init(self):
        for idx, s in enumerate(self._sensor_list):
            self._sensor_list[idx] = s(self._env)
        self._order_sensors()
        for s in self._sensor_list:
            s._init_sensor(self._env._robot_config)
            s._update_sensor_info()

    def _turn_on(self, robot):
        for s in self._sensor_list:
            s._set_sensor(robot)

    def get_desired_velocity(self):
        for s in self._sensor_list:
            if isinstance(s, DesiredBaseLinearVelocityXZ):
                return s.get_desired_velocity()
        raise ValueError("Desired Velocity not specified.")

    def _order_sensors(self):
        """
        Order the sensors list to ensure the height sensor use encoder measure after
        they have been updated. Just put the encoder and imu sensors at the top if present.
        """
        self._order_sensor(JointPosition)
        self._order_sensor(OrientationRPY)

    def _order_sensor(self, Sensor):
        sensor_present = False
        pos = None
        for idx, s in enumerate(self._sensor_list):
            if isinstance(s, Sensor):
                sensor_present = True
                pos = idx
        if sensor_present:
            sens = self._sensor_list.pop(pos)
            self._sensor_list.reverse()
            self._sensor_list.append(sens)
            self._sensor_list.reverse()
