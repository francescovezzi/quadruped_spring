import numpy as np


class Sensor:
    """A prototype class for a generic sensor"""

    def __init__(self):
        self._data = None
        self._robot = None
        self._name = "please give me a name"

    def _check_dimension(self):
        assert np.shape(self._high) == np.shape(self._low), "high limit and low limits are different in size"
        assert np.shape(self._high) == np.shape(self._data), "Observation different from sensor observation limits"

    def _update_sensor_info(self, high, low, noise_std):
        self._high = high
        self._low = low
        self._noise_std = noise_std
        self._dim = np.shape(self._high)

    def _sample_noise(self):
        if np.all(self._noise_std > 0.0):
            self._add_obs_noise = np.random.normal(scale=self._noise_std)
        elif np.all(self._noise_std == 0.0):
            self._add_obs_noise = np.zeros(np.shape(self._noise_std))
        else:
            raise ValueError(f"Noise standard deviation should be >= 0.0. not {self._noise_std}")

    def _set_sensor(self, robot):
        """Call it at init"""
        self._robot = robot

    def _init_sensor(self, robot_config):
        self._robot_config = robot_config

    def _reset_sensor(self):
        """Call it at reset"""
        pass

    def _read_data(self):
        """Get Sensor data without noise"""
        return self._data

    def _read_dirty_data(self):
        """Get Sensor data with noise"""
        return self._data + self._add_obs_noise

    def _on_step(self):
        """Callback for step method"""
        pass

    def _get_data(self):
        """Get sensor data"""
        pass
