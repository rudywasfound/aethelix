"""
Communications subsystem simulator.

Models telemetry downlink and transponder health:
1. HPA (High Power Amplifier) degradation
2. Antenna Pointing Errors (link loss)
3. Bit Error Rate (BER) spikes (interference or weak signal)
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class CommsTelemetry:
    """Telemetry outputs for Comms subsystem."""
    time: np.ndarray
    downlink_power: np.ndarray      # dBm (signal strength)
    ber: np.ndarray                 # Bit Error Rate (10^-x)
    transponder_temp: np.ndarray    # Celsius
    timestamp: np.ndarray

class CommsSimulator:
    def __init__(self, duration_hours: float = 24, sampling_rate_hz: float = 0.1):
        self.num_samples = int(duration_hours * 3600 * sampling_rate_hz)
        self.time = np.linspace(0, duration_hours * 3600, self.num_samples)

    def simulate(
        self,
        hpa_fault_hour: float = None,
        pointing_error_hour: float = None,
        interference_hour: float = None,
    ) -> CommsTelemetry:
        downlink_power = np.zeros(self.num_samples)
        ber = np.zeros(self.num_samples)
        transponder_temp = np.zeros(self.num_samples)

        # Baseline nominal values
        base_power = -50.0  # dBm
        base_ber = 1e-7
        base_temp = 35.0

        for i in range(self.num_samples):
            t_hr = self.time[i] / 3600.0
            
            curr_power = base_power + np.random.normal(0, 0.5)
            curr_ber = base_ber
            curr_temp = base_temp + np.random.normal(0, 0.2)
            
            # HPA Fault (Power loss + Heat gain)
            if hpa_fault_hour and t_hr >= hpa_fault_hour:
                curr_power -= 10.0 # 10dB drop
                curr_temp += 15.0  # Heat rise from inefficiency
            
            # Antenna Pointing Error (Severe power loss -> BER spike)
            if pointing_error_hour and t_hr >= pointing_error_hour:
                curr_power -= 25.0
                curr_ber = 1e-2 # High error rate
            
            # External Interference (BER spike only)
            if interference_hour and t_hr >= interference_hour:
                curr_ber = 1e-4

            downlink_power[i] = curr_power
            ber[i] = curr_ber
            transponder_temp[i] = curr_temp

        return CommsTelemetry(
            time=self.time,
            downlink_power=downlink_power,
            ber=ber,
            transponder_temp=transponder_temp,
            timestamp=np.arange(self.num_samples)
        )
