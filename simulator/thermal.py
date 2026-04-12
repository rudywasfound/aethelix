"""
Thermal subsystem simulator for satellite.
Models temperature dynamics for solar panels, battery, and payload electronics.
Supports degradation modes for insulation and heatsinks.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ThermalTelemetry:
    """ Container for thermal subsystem telemetry outputs. """
    
    time: np.ndarray              # Seconds elapsed
    battery_temp: np.ndarray      # Celsius (typically 0-60C operating range)
    solar_panel_temp: np.ndarray  # Celsius (can reach 80C+ in sun)
    payload_temp: np.ndarray      # Celsius (electronics thermal limits ~70C)
    bus_current: np.ndarray       # Amps (proxy for power dissipation)
    timestamp: np.ndarray         # Sample indices


class ThermalSimulator:
    """
    Realistic thermal subsystem simulator.
    Models temperature based on heat balance: dT/dt = (heat_in - heat_out) / thermal_mass.
    """

    def __init__(
        self,
        duration_hours: float = 24,
        sampling_rate_hz: float = 0.1,
    ):
        """Initialize thermal simulator."""
        
        self.duration_hours = duration_hours
        self.sampling_rate_hz = sampling_rate_hz

        self.num_samples = int(duration_hours * 3600 * sampling_rate_hz)
        self.time = np.linspace(0, duration_hours * 3600, self.num_samples)
        self.dt = self.time[1] - self.time[0]

    def simulate_solar_panel_temp(
        self,
        base_temp: float = 45.0,
        eclipse_frequency_hours: float = 1.5,
        max_eclipse_temp: float = 5.0,
        degradation_start_hour: float = None,
        degradation_drift_rate: float = 0.5,
    ) -> np.ndarray:
        """Simulate solar panel temperature with orbital cycles and insulation drift."""
        
        orbital_phase = 2 * np.pi * self.time / (eclipse_frequency_hours * 3600)
        panel_temp = base_temp * (1 + 0.7 * np.cos(orbital_phase)) / 2 + max_eclipse_temp

        # Add orbital transients and noise
        panel_temp += 3 * np.sin(2 * orbital_phase) + np.random.normal(0, 1, len(panel_temp))

        # Inject insulation degradation
        if degradation_start_hour is not None:
            degrad_start_sample = int(degradation_start_hour * 3600 * self.sampling_rate_hz)
            degrad_start_sample = min(degrad_start_sample, len(self.time) - 1)
            if degrad_start_sample < len(self.time):
                time_since_degrad = (self.time[degrad_start_sample:] - self.time[degrad_start_sample]) / 3600
                drift = degradation_drift_rate * time_since_degrad

                panel_temp[degrad_start_sample:] += drift

        return panel_temp

    def simulate_battery_temp(
        self,
        solar_input: np.ndarray,
        battery_charge: np.ndarray,
        base_temp: float = 30.0,
        max_temp: float = 60.0,
        power_dissipation_factor: float = 0.1,
        thermal_mass: float = 1000.0,
        ambient_temp: float = 20.0,
        heat_dissipation_rate: float = 0.05,
        degradation_start_hour: float = None,
        degradation_factor: float = 0.5,
    ) -> np.ndarray:
        """Simulate battery temperature with heat generation and dissipation."""
        
        battery_temp = np.zeros(self.num_samples)
        temp = base_temp

        for i in range(self.num_samples):
            charge_stress = (1 - battery_charge[i] / 100.0) * (solar_input[i] / 500.0)
            heat_generation = power_dissipation_factor * charge_stress * 100

            cooling = heat_dissipation_rate
            
            if degradation_start_hour is not None:
                degrad_start_sample = int(
                    degradation_start_hour * 3600 * self.sampling_rate_hz
                )
                if i >= degrad_start_sample:
                    cooling *= degradation_factor

            temp_differential = temp - ambient_temp
            natural_cooling = cooling * temp_differential

            temp_change = (heat_generation - natural_cooling) * self.dt / thermal_mass

            temp = np.clip(temp + temp_change, ambient_temp, max_temp)

            battery_temp[i] = temp
            battery_temp[i] += np.random.normal(0, 0.3)

        return battery_temp

    def simulate_payload_temp(
        self,
        battery_voltage: np.ndarray,
        base_temp: float = 25.0,
        max_temp: float = 50.0,
        power_draw_factor: float = 0.05,
        degradation_start_hour: float = None,
        degradation_factor: float = 0.7,
    ) -> np.ndarray:
        """Simulate payload electronics temperature under load."""
        
        payload_temp = np.zeros(self.num_samples)
        temp = base_temp

        for i in range(self.num_samples):
            available_power = battery_voltage[i]
            heat = power_draw_factor * available_power

            cooling_rate = 0.03
            
            if degradation_start_hour is not None:
                degrad_start_sample = int(
                    degradation_start_hour * 3600 * self.sampling_rate_hz
                )
                if i >= degrad_start_sample:
                    cooling_rate *= degradation_factor

            temp_diff = temp - 20.0
            cooling = cooling_rate * temp_diff

            temp_change = (heat - cooling) * self.dt
            temp = np.clip(temp + temp_change, 20.0, max_temp)

            payload_temp[i] = temp + np.random.normal(0, 0.2)

        return payload_temp

    def simulate_bus_current(
        self,
        battery_charge: np.ndarray,
        battery_voltage: np.ndarray,
        base_current: float = 20.0,
    ) -> np.ndarray:
        """Simulate bus current draw based on charge and voltage stress."""
        
        charge_stress = 1.0 - battery_charge / 100.0
        voltage_stress = 1.0 - battery_voltage / 28.0

        current = base_current * (1.0 + 0.5 * charge_stress + 0.3 * voltage_stress)
        
        current += np.random.normal(0, 1, len(current))
        current = np.clip(current, 5, 50)

        return current

    def run_nominal(
        self,
        solar_input: np.ndarray,
        battery_charge: np.ndarray,
        battery_voltage: np.ndarray,
    ) -> ThermalTelemetry:
        """Simulate healthy temperature baseline based on power data."""
        
        panel_temp = self.simulate_solar_panel_temp(degradation_start_hour=None)
        batt_temp = self.simulate_battery_temp(
            solar_input, battery_charge, degradation_start_hour=None
        )
        payload_temp = self.simulate_payload_temp(
            battery_voltage, degradation_start_hour=None
        )
        bus_current = self.simulate_bus_current(battery_charge, battery_voltage)

        return ThermalTelemetry(
            time=self.time,
            battery_temp=batt_temp,
            solar_panel_temp=panel_temp,
            payload_temp=payload_temp,
            bus_current=bus_current,
            timestamp=np.arange(self.num_samples),
        )

    def run_degraded(
        self,
        solar_input: np.ndarray,
        battery_charge: np.ndarray,
        battery_voltage: np.ndarray,
        panel_degradation_hour: float = 6.0,
        panel_drift_rate: float = 0.6,
        battery_cooling_hour: float = 8.0,
        battery_cooling_factor: float = 0.5,
        payload_cooling_hour: float = None,
        payload_cooling_factor: float = 0.7,
    ) -> ThermalTelemetry:
        """Simulate degraded thermal behavior with insulation or heatsink failures."""
        
        panel_temp = self.simulate_solar_panel_temp(
            degradation_start_hour=panel_degradation_hour,
            degradation_drift_rate=panel_drift_rate,
        )
        batt_temp = self.simulate_battery_temp(
            solar_input,
            battery_charge,
            degradation_start_hour=battery_cooling_hour,
            degradation_factor=battery_cooling_factor,
        )
        payload_temp = self.simulate_payload_temp(
            battery_voltage,
            degradation_start_hour=payload_cooling_hour,
            degradation_factor=payload_cooling_factor,
        )
        bus_current = self.simulate_bus_current(battery_charge, battery_voltage)

        return ThermalTelemetry(
            time=self.time,
            battery_temp=batt_temp,
            solar_panel_temp=panel_temp,
            payload_temp=payload_temp,
            bus_current=bus_current,
            timestamp=np.arange(self.num_samples),
        )


if __name__ == "__main__":
    # Quick test of thermal simulator
    from simulator.power import PowerSimulator

    power_sim = PowerSimulator(duration_hours=24)
    power_nominal = power_sim.run_nominal()

    thermal_sim = ThermalSimulator(duration_hours=24)

    print("Simulating nominal thermal scenario...")
    thermal_nominal = thermal_sim.run_nominal(
        power_nominal.solar_input,
        power_nominal.battery_charge,
        power_nominal.battery_voltage,
    )
    print(
        f"  Battery: {thermal_nominal.battery_temp.mean():.1f}C (mean), "
        f"Panel: {thermal_nominal.solar_panel_temp.mean():.1f}C (mean)"
    )

    print("Simulating degraded thermal scenario...")
    thermal_degraded = thermal_sim.run_degraded(
        power_nominal.solar_input,
        power_nominal.battery_charge,
        power_nominal.battery_voltage,
        panel_degradation_hour=6.0,
        battery_cooling_hour=8.0,
    )
    print(
        f"  Battery: {thermal_degraded.battery_temp.mean():.1f}C (mean), "
        f"Panel: {thermal_degraded.solar_panel_temp.mean():.1f}C (mean)"
    )
