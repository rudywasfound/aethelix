"""
OBC (Onboard Computer) subsystem simulator.

Models processing health and software stability:
1. Memory Errors (SEUs/Corruption)
2. Watchdog Resets (Software hangs)
3. Software Exceptions (CPU overloads)
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class OBCTelemetry:
    """Telemetry outputs for OBC subsystem."""
    time: np.ndarray
    cpu_load: np.ndarray        # Percentage
    memory_usage: np.ndarray    # Percentage
    reboot_count: np.ndarray    # Cumulative count
    timestamp: np.ndarray

class OBCSimulator:
    def __init__(self, duration_hours: float = 24, sampling_rate_hz: float = 0.1):
        self.num_samples = int(duration_hours * 3600 * sampling_rate_hz)
        self.time = np.linspace(0, duration_hours * 3600, self.num_samples)

    def simulate(
        self,
        memory_error_hour: float = None,
        watchdog_fault_hour: float = None,
        exception_hour: float = None,
    ) -> OBCTelemetry:
        cpu_load = np.zeros(self.num_samples)
        memory_usage = np.zeros(self.num_samples)
        reboot_count = np.zeros(self.num_samples)

        curr_cpu = 15.0  # Base idling load
        curr_mem = 40.0  # Base memory usage
        curr_reboots = 0

        for i in range(self.num_samples):
            t_hr = self.time[i] / 3600.0
            
            # Memory Leak / Corruption
            if memory_error_hour and t_hr >= memory_error_hour:
                curr_mem += 0.5 * (self.time[1] - self.time[0]) / 3600 # Accumulating leak
                curr_cpu += 0.1 # Processing overhead for ECC/repair
            
            # Watchdog Reset (Sudden increment)
            if watchdog_fault_hour and t_hr >= watchdog_fault_hour:
                # Trigger reset every 3 hours after fault starts
                if (t_hr - watchdog_fault_hour) % 3.0 < 0.1:
                    curr_reboots += 1
            
            # Software Exception (Transient spike)
            if exception_hour and abs(t_hr - exception_hour) < 0.2:
                curr_cpu = 95.0 # Max out CPU
            else:
                curr_cpu = 15.0 + np.random.normal(0, 2)

            cpu_load[i] = curr_cpu
            memory_usage[i] = curr_mem
            reboot_count[i] = curr_reboots

        return OBCTelemetry(
            time=self.time,
            cpu_load=cpu_load,
            memory_usage=memory_usage,
            reboot_count=reboot_count,
            timestamp=np.arange(self.num_samples)
        )
