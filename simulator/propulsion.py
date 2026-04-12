"""
Propulsion subsystem simulator.

Models fuel management and orbital maneuvers:
1. Thruster Valve Anomalies (stuck common/closed)
2. Fuel Pressure Deviations (leaks or regulator failure)
3. Propellant Depletion
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class PropulsionTelemetry:
    """Telemetry outputs for Propulsion subsystem."""
    time: np.ndarray
    tank_pressure: np.ndarray       # PSI (nominal ~300)
    thruster_temp: np.ndarray       # Celsius
    delta_v_measured: np.ndarray    # m/s (cumulative change)
    timestamp: np.ndarray

class PropulsionSimulator:
    def __init__(self, duration_hours: float = 24, sampling_rate_hz: float = 0.1):
        self.num_samples = int(duration_hours * 3600 * sampling_rate_hz)
        self.time = np.linspace(0, duration_hours * 3600, self.num_samples)

    def simulate(
        self,
        valve_fault_hour: float = None,
        pressure_leak_hour: float = None,
        depletion_hour: float = None,
    ) -> PropulsionTelemetry:
        tank_pressure = np.zeros(self.num_samples)
        thruster_temp = np.zeros(self.num_samples)
        delta_v = np.zeros(self.num_samples)

        curr_pressure = 300.0
        curr_temp = 20.0
        curr_dv = 0.0

        for i in range(self.num_samples):
            t_hr = self.time[i] / 3600.0
            
            # Baseline pressure decay (slow consumption)
            curr_pressure -= 0.001 * (self.time[1] - self.time[0]) / 3600.0
            
            # Fuel Pressure Leak (Rapid decay)
            if pressure_leak_hour and t_hr >= pressure_leak_hour:
                curr_pressure -= 5.0 * (self.time[1] - self.time[0]) / 3600.0
            
            # Propellant Depletion (Clip at ZERO)
            if depletion_hour and t_hr >= depletion_hour:
                curr_pressure = 0.0
            
            # Thruster Activity (Dummy burn at T+12h)
            if abs(t_hr - 12.0) < 0.05:
                if valve_fault_hour and t_hr >= valve_fault_hour:
                    # Valve stuck closed: no thrust, no temp rise
                    pass
                else:
                    curr_temp = 120.0 # Active thruster heat
                    curr_dv += 0.1     # Increment velocity
            else:
                curr_temp = 20.0 + np.random.normal(0, 1)

            tank_pressure[i] = max(0, curr_pressure + np.random.normal(0, 0.5))
            thruster_temp[i] = curr_temp
            delta_v[i] = curr_dv

        return PropulsionTelemetry(
            time=self.time,
            tank_pressure=tank_pressure,
            thruster_temp=thruster_temp,
            delta_v_measured=delta_v,
            timestamp=np.arange(self.num_samples)
        )
