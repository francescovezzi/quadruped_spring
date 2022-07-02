from quadruped_spring.env.sensors import robot_sensors as rs
from quadruped_spring.utils.base_collection import CollectionBase

# Implemented observation spaces for deep reinforcement learning:
#   "DEFAULT":  IMU (base linear velocity, base orientation, base angular rate) +
#               Feet position, Feet velociites, ground reaction force
#   "ENCODER":  IMU + Joint position and velocity + ground reaction force
#   "CARTESIAN_NO_IMU": Feet position, feet velocities, ground reaction force
#   "ANGLE_NO_IMU": Joint position, joint velocities, ground reaction force
#   "CARTESIAN_ANGLE_NO_IMU": CARTESIAN_NO_IMU + ANGLE_NO_IMU


class SensorCollection(CollectionBase):
    """Utility to collect all the implemented robot sensor equipments."""

    def __init__(self):
        super().__init__()
        self._ENCODER = [rs.JointPosition, rs.JointVelocity]
        self._ENCODER_2 = [rs.LinearVelocity, rs.AngularVelocity, rs.JointPosition, rs.JointVelocity]
        self._CARTESIAN_NO_IMU = [rs.FeetPostion, rs.FeetVelocity]
        self._ANGLE_NO_IMU = [rs.JointPosition, rs.JointVelocity]
        self._CUSTOM_3D = [
            rs.Quaternion,
            rs.DesiredBaseLinearVelocityXZ,
            rs.AngularVelocity,
            rs.LinearVelocity,
            rs.JointPosition,
            rs.JointVelocity,
        ]
        self._CUSTOM_2D = [
            rs.Pitch,
            rs.PitchRate,
            rs.DesiredBaseLinearVelocityXZ,
            rs.LinearVelocity2D,
            rs.JointPosition,
            rs.JointVelocity,
        ]
        self._ARS = [
            rs.Pitch,
            rs.PitchRate,
            rs.LinearVelocity2D,
            rs.JointPosition,
            rs.JointVelocity,
        ]
        self._ARS_HEIGHT = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.PitchRate,
            rs.LinearVelocity2D,
            rs.Height,
        ]
        self._ARS_CONTACT = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.HeightGround,
            rs.BooleanContact,
        ]
        self._ARS_BASIC = [
            rs.JointPosition,
            rs.JointVelocity,
            rs.Pitch,
            rs.Height,
            rs.BaseHeightVelocity,
        ]
        self._dict = {
            "ENCODER": self._ENCODER,
            "ENCODER_2": self._ENCODER_2,
            "CARTESIAN_NO_IMU": self._CARTESIAN_NO_IMU,
            "ANGLE_NO_IMU": self._ANGLE_NO_IMU,
            "CUSTOM_3D": self._CUSTOM_3D,
            "CUSTOM_2D": self._CUSTOM_2D,
            "ARS": self._ARS,
            "ARS_HEIGHT": self._ARS_HEIGHT,
            "ARS_CONTACT": self._ARS_CONTACT,
            "ARS_BASIC": self._ARS_BASIC,
        }
        self._element_type = "sensor package"
