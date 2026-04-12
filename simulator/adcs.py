"""
ADCS (Attitude Determination and Control System) simulator.

Models satellite pointing dynamics and actuator failures:
1. Reaction Wheel Friction (increased drag/current)
2. Gyroscope Drift (calibration bias)
3. Magnetorquer Anomalies (desaturation failure)

This follows ECSS-E-ST-60-30C standards for attitude control modeling.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ADCSTelemetry:
    """Telemetry outputs for ADCS subsystem."""
    time: np.ndarray
    pointing_error: np.ndarray      # Arcseconds of deviation from target
    wheel_speed: np.ndarray         # RPM of reaction wheels
    wheel_current: np.ndarray       # Amps drawn by motors
    gyro_bias: np.ndarray           # Degrees/hour estimated drift
    timestamp: np.ndarray

class ADCSSimulator:
    def __init__(self, duration_hours: float = 24, sampling_rate_hz: float = 0.1):
        self.num_samples = int(duration_hours * 3600 * sampling_rate_hz)
        self.time = np.linspace(0, duration_hours * 3600, self.num_samples)
        self.dt = self.time[1] - self.time[0]

    def simulate(
        self,
        wheel_friction_hour: float = None,
        gyro_drift_hour: float = None,
        magnetorquer_fault_hour: float = None,
    ) -> ADCSTelemetry:
        pointing_error = np.zeros(self.num_samples)
        wheel_speed = np.zeros(self.num_samples)
        wheel_current = np.zeros(self.num_samples)
        gyro_bias = np.zeros(self.num_samples)

        # Base nominal states
        curr_pointing = 2.0  # Nominal jitter in arcsec
        curr_speed = 2000.0  # Nominal RPM for bias stabilization
        curr_bias = 0.01     # Nominal gyro bias deg/hr

        for i in range(self.num_samples):
            t_hr = self.time[i] / 3600.0
            
            # 1. Gyro Drift Logic
            if gyro_drift_hour and t_hr >= gyro_drift_hour:
                curr_bias += 0.05 * self.dt / 3600.0 # Gradual drift accumulation
            
            # 2. Wheel Friction & Magnetorquer Logic
            friction_mult = 1.0
            if wheel_friction_hour and t_hr >= wheel_friction_hour:
                friction_mult = 2.5 # Friction increases drag
            
            # If magnetorquer fails, wheels don't desaturate, speed builds up
            if magnetorquer_fault_hour and t_hr >= magnetorquer_fault_hour:
                curr_speed += 1.0 * self.dt # Constant ramp up
            else:
                # Normal desaturation (simple model)
                curr_speed = 2000.0 + 50 * np.sin(2 * np.pi * t_hr / 1.5)
            
            # 3. Pointing Error Coupling
            # Bias causes fake error -> controller corrects -> real error induced
            curr_pointing = 2.0 + 100 * curr_bias + np.random.normal(0, 0.5)
            
            # Friction causes jitter and high current
            if friction_mult > 1.0:
                curr_pointing += 5.0 # Jitter from bearing friction
                curr_current = 0.5 * friction_mult + np.random.normal(0, 0.05)
            else:
                curr_current = 0.5 + 0.1 * (curr_speed / 5000.0) + np.random.normal(0, 0.02)

            pointing_error[i] = curr_pointing
            wheel_speed[i] = curr_speed
            wheel_current[i] = curr_current
            gyro_bias[i] = curr_bias

        return ADCSTelemetry(
            time=self.time,
            pointing_error=pointing_error,
            wheel_speed=wheel_speed,
            wheel_current=wheel_current,
            gyro_bias=gyro_bias,
            timestamp=np.arange(self.num_samples)
        )
