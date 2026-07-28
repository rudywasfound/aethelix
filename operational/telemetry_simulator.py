#!/usr/bin/env python3
"""
Satellite Telemetry Simulator

Generates realistic housekeeping measurements for testing Aethelix's
inference engine without needing real satellite data.

Usage:
    sim = TelemetrySimulator(scenario="solar_degradation")
    for t in range(3600):
        measurements = sim.generate(timestamp=datetime.now())
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class Measurement:
    """Single telemetry measurement with timestamp."""
    
    timestamp: datetime
    battery_voltage_measured: float
    battery_charge_measured: float
    battery_temp_measured: float
    bus_voltage_measured: float
    bus_current_measured: float
    solar_input_measured: float
    solar_panel_temp_measured: float
    payload_temp_measured: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for inference engine."""
        return {
            'battery_voltage_measured': self.battery_voltage_measured,
            'battery_charge_measured': self.battery_charge_measured,
            'battery_temp_measured': self.battery_temp_measured,
            'bus_voltage_measured': self.bus_voltage_measured,
            'bus_current_measured': self.bus_current_measured,
            'solar_input_measured': self.solar_input_measured,
            'solar_panel_temp_measured': self.solar_panel_temp_measured,
            'payload_temp_measured': self.payload_temp_measured,
        }


class TelemetrySimulator:
    """
    Realistic satellite telemetry generator.
    
    Simulates nominal operation and various failure modes.
    Includes sensor noise and environmental effects.
    """
    
    NOMINAL_RANGES = {
        'battery_voltage_measured': (28.0, 32.0),      # Volts
        'battery_charge_measured': (90.0, 100.0),      # Ah
        'battery_temp_measured': (25.0, 45.0),         # Celsius
        'bus_voltage_measured': (27.0, 31.0),          # Volts
        'bus_current_measured': (5.0, 30.0),           # Amps
        'solar_input_measured': (350.0, 450.0),        # Watts
        'solar_panel_temp_measured': (30.0, 60.0),     # Celsius
        'payload_temp_measured': (20.0, 50.0),         # Celsius
    }
    
    SENSOR_NOISE = {
        'battery_voltage_measured': 0.1,
        'battery_charge_measured': 0.5,
        'battery_temp_measured': 0.5,
        'bus_voltage_measured': 0.1,
        'bus_current_measured': 0.3,
        'solar_input_measured': 5.0,
        'solar_panel_temp_measured': 1.0,
        'payload_temp_measured': 0.8,
    }
    
    def __init__(self, scenario: str = "nominal", seed: int = 42):
        """
        Initialize simulator with specific failure scenario.
        
        Args:
            scenario: One of:
                - "nominal" - healthy satellite
                - "solar_degradation" - solar panel efficiency loss (GSAT-6A)
                - "battery_aging" - capacity loss over time
                - "battery_thermal" - overheating
                - "sensor_bias" - measurement drift
                - "multi_fault" - solar + thermal
            seed: Random seed for reproducibility
        """
        
        self.scenario = scenario
        self.rng = np.random.RandomState(seed)
        self.time_step = 0
        
        if scenario == "solar_degradation":
            self.degradation_rate = 0.02
        elif scenario == "battery_aging":
            self.capacity_loss_rate = 0.001
        elif scenario == "sensor_bias":
            self.bias_drift_rate = 0.05
        else:
            self.degradation_rate = 0.0
            self.capacity_loss_rate = 0.0
            self.bias_drift_rate = 0.0
    
    def generate(self, timestamp: Optional[datetime] = None) -> Measurement:
        """
        Generate a single measurement.
        
        Args:
            timestamp: Optional timestamp (uses auto-increment if None)
            
        Returns:
            Measurement object with all telemetry fields
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        battery_voltage = self.rng.uniform(*self.NOMINAL_RANGES['battery_voltage_measured'])
        battery_charge = self.rng.uniform(*self.NOMINAL_RANGES['battery_charge_measured'])
        battery_temp = self.rng.uniform(*self.NOMINAL_RANGES['battery_temp_measured'])
        bus_voltage = self.rng.uniform(*self.NOMINAL_RANGES['bus_voltage_measured'])
        bus_current = self.rng.uniform(*self.NOMINAL_RANGES['bus_current_measured'])
        solar_input = self.rng.uniform(*self.NOMINAL_RANGES['solar_input_measured'])
        solar_panel_temp = self.rng.uniform(*self.NOMINAL_RANGES['solar_panel_temp_measured'])
        payload_temp = self.rng.uniform(*self.NOMINAL_RANGES['payload_temp_measured'])
        
        if self.scenario == "solar_degradation":
            degradation_factor = 1.0 - (self.time_step * self.degradation_rate / 3600)
            solar_input *= degradation_factor
            
            battery_charge -= (1.0 - degradation_factor) * 5.0
            battery_charge = max(battery_charge, 60.0)  # Don't go negative
            
            bus_voltage -= (1.0 - degradation_factor) * 1.5
            
            battery_temp += (1.0 - degradation_factor) * 10.0
        
        elif self.scenario == "battery_aging":
            capacity_factor = 1.0 - (self.time_step * self.capacity_loss_rate / 3600)
            battery_charge *= capacity_factor
            battery_voltage -= (1.0 - capacity_factor) * 2.0
        
        elif self.scenario == "battery_thermal":
            battery_temp += (self.time_step * 0.05)
            battery_charge -= (battery_temp - 40.0) * 0.1
            bus_voltage -= (battery_temp - 40.0) * 0.05
        
        elif self.scenario == "sensor_bias":
            bias_factor = 1.0 + (self.time_step * self.bias_drift_rate / 3600)
            battery_charge *= bias_factor
            battery_voltage *= (1.0 + bias_factor * 0.01)
        
        elif self.scenario == "multi_fault":
            degradation_factor = 1.0 - (self.time_step * 0.01 / 3600)
            solar_input *= degradation_factor
            battery_charge -= (1.0 - degradation_factor) * 3.0
            battery_temp += 5.0
        
        battery_voltage = np.clip(battery_voltage, 20.0, 35.0)
        battery_charge = np.clip(battery_charge, 40.0, 105.0)
        battery_temp = np.clip(battery_temp, 0.0, 80.0)
        bus_voltage = np.clip(bus_voltage, 25.0, 32.0)
        bus_current = np.clip(bus_current, 0.0, 50.0)
        solar_input = np.clip(solar_input, 0.0, 500.0)
        solar_panel_temp = np.clip(solar_panel_temp, 20.0, 100.0)
        payload_temp = np.clip(payload_temp, 0.0, 80.0)
        
        battery_voltage += self.rng.normal(0, self.SENSOR_NOISE['battery_voltage_measured'])
        battery_charge += self.rng.normal(0, self.SENSOR_NOISE['battery_charge_measured'])
        battery_temp += self.rng.normal(0, self.SENSOR_NOISE['battery_temp_measured'])
        bus_voltage += self.rng.normal(0, self.SENSOR_NOISE['bus_voltage_measured'])
        bus_current += self.rng.normal(0, self.SENSOR_NOISE['bus_current_measured'])
        solar_input += self.rng.normal(0, self.SENSOR_NOISE['solar_input_measured'])
        solar_panel_temp += self.rng.normal(0, self.SENSOR_NOISE['solar_panel_temp_measured'])
        payload_temp += self.rng.normal(0, self.SENSOR_NOISE['payload_temp_measured'])
        
        self.time_step += 1
        
        return Measurement(
            timestamp=timestamp,
            battery_voltage_measured=battery_voltage,
            battery_charge_measured=battery_charge,
            battery_temp_measured=battery_temp,
            bus_voltage_measured=bus_voltage,
            bus_current_measured=bus_current,
            solar_input_measured=solar_input,
            solar_panel_temp_measured=solar_panel_temp,
            payload_temp_measured=payload_temp,
        )
    
    def generate_series(self, duration_seconds: int, sampling_rate: float = 1.0):
        """
        Generate a time series of measurements.
        
        Args:
            duration_seconds: How long to simulate
            sampling_rate: Measurements per second (1.0 = 1 Hz)
            
        Yields:
            Measurement objects
        """
        
        interval = 1.0 / sampling_rate
        now = datetime.now()
        
        for step in range(int(duration_seconds * sampling_rate)):
            timestamp = now + timedelta(seconds=step * interval)
            yield self.generate(timestamp)


def main():
    """Demo: generate and print telemetry for each scenario."""
    
    scenarios = ["nominal", "solar_degradation", "battery_aging", 
                 "battery_thermal", "sensor_bias", "multi_fault"]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.upper()}")
        print(f"{'='*70}")
        
        sim = TelemetrySimulator(scenario=scenario)
        
        print(f"\n{'Time':<12} {'Solar W':<10} {'Batt V':<10} {'Batt Ah':<10} "
              f"{'Batt T°C':<10} {'Bus V':<10}")
        print("-" * 70)
        
        for i, measurement in enumerate(sim.generate_series(3600, sampling_rate=0.5)):
            if i % 60 == 0:
                print(f"{measurement.timestamp.strftime('%H:%M:%S'):<12} "
                      f"{measurement.solar_input_measured:<10.1f} "
                      f"{measurement.battery_voltage_measured:<10.2f} "
                      f"{measurement.battery_charge_measured:<10.1f} "
                      f"{measurement.battery_temp_measured:<10.1f} "
                      f"{measurement.bus_voltage_measured:<10.2f}")


if __name__ == "__main__":
    main()
