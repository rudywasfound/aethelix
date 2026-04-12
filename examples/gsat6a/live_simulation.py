#!/usr/bin/env python3
"""
GSAT-6A Live Failure Simulation

Simulates the actual failure sequence of GSAT-6A with:
- Real telemetry from power and thermal simulators
- Live causal inference analysis
- Threshold-based detection comparison
- Timeline of when each system fails
"""

import numpy as np
import sys
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.power import PowerSimulator, PowerTelemetry
from simulator.thermal import ThermalSimulator, ThermalTelemetry
from causal_graph.graph_definition import CausalGraph
from causal_graph.root_cause_ranking import RootCauseRanker
from timeline import Timeline, EventSeverity
from findings import FindingsEngine


@dataclass
class CombinedTelemetry:
    """Combined power and thermal telemetry."""
    solar_input: np.ndarray
    battery_voltage: np.ndarray
    battery_charge: np.ndarray
    bus_voltage: np.ndarray
    battery_temp: np.ndarray
    solar_panel_temp: np.ndarray
    payload_temp: np.ndarray
    bus_current: np.ndarray


class GSAT6ASimulator:
    """Simulate GSAT-6A's actual failure sequence."""
    
    def __init__(self):
        """Initialize with GSAT-6A mission parameters."""
        self.mission_start = datetime(2017, 3, 28, 0, 0, 0)  # Launch date
        self.failure_onset = datetime(2018, 3, 26, 12, 0, 0)  # Failure begins
        self.days_to_failure = (self.failure_onset - self.mission_start).days
        
        # Framework components
        self.timeline = Timeline()
        self.findings = FindingsEngine()
    
    def run_simulation(self):
        """Run the complete GSAT-6A failure simulation."""
        
        # Initialize simulators
        power_sim = PowerSimulator(duration_hours=24)
        thermal_sim = ThermalSimulator(duration_hours=24)
        
        # Generate nominal baseline
        nominal_power = power_sim.run_nominal()
        nominal_thermal = thermal_sim.run_nominal(
            solar_input=nominal_power.solar_input,
            battery_charge=nominal_power.battery_charge,
            battery_voltage=nominal_power.battery_voltage,
        )
        
        # Simulate degradation with multiple injection points
        degraded_power = power_sim.run_degraded(
            solar_degradation_hour=0.015,    # Very early: 36 seconds
            battery_degradation_hour=0.5,     # Later: 1800 seconds
        )
        degraded_thermal = thermal_sim.run_degraded(
            solar_input=degraded_power.solar_input,
            battery_charge=degraded_power.battery_charge,
            battery_voltage=degraded_power.battery_voltage,
            panel_degradation_hour=0.25,
            battery_cooling_hour=1.0,
        )
        
        # Initialize inference engine
        graph = CausalGraph()
        ranker = RootCauseRanker(graph)
        
        # Analyze at different time windows to show progression
        time_windows = [
            ("T+36s (Early Detection Window)", slice(0, 120)),      # First 2 minutes
            ("T+180s (Clear Pattern)", slice(120, 600)),            # 2-10 minutes
            ("T+600s (Obvious Failure)", slice(600, 1200)),         # 10-20 minutes
            ("T+1800s (Complete Failure)", slice(1200, None)),      # 20+ minutes
        ]
        
        detection_times = {
            "aethelix": None,
            "threshold_solar": None,
            "threshold_battery": None,
            "threshold_voltage": None,
        }
        
        for window_name, time_slice in time_windows:
            
            # Slice the telemetry to this time window
            nominal_slice = CombinedTelemetry(
                solar_input=nominal_power.solar_input[time_slice],
                battery_voltage=nominal_power.battery_voltage[time_slice],
                battery_charge=nominal_power.battery_charge[time_slice],
                bus_voltage=nominal_power.bus_voltage[time_slice],
                battery_temp=nominal_thermal.battery_temp[time_slice],
                solar_panel_temp=nominal_thermal.solar_panel_temp[time_slice],
                payload_temp=nominal_thermal.payload_temp[time_slice],
                bus_current=nominal_thermal.bus_current[time_slice],
            )
            
            degraded_slice = CombinedTelemetry(
                solar_input=degraded_power.solar_input[time_slice],
                battery_voltage=degraded_power.battery_voltage[time_slice],
                battery_charge=degraded_power.battery_charge[time_slice],
                bus_voltage=degraded_power.bus_voltage[time_slice],
                battery_temp=degraded_thermal.battery_temp[time_slice],
                solar_panel_temp=degraded_thermal.solar_panel_temp[time_slice],
                payload_temp=degraded_thermal.payload_temp[time_slice],
                bus_current=degraded_thermal.bus_current[time_slice],
            )
            
            # Run causal inference
            hypotheses = ranker.analyze(nominal_slice, degraded_slice, deviation_threshold=0.10)
            
            # Record causal detection
            if hypotheses:
                if detection_times["aethelix"] is None and hypotheses[0].probability > 0.3:
                    detection_times["aethelix"] = window_name
                    self.timeline.add_event(
                        float(window_name.split("+")[1].split("s")[0]),
                        EventSeverity.CRITICAL,
                        "causal_detection",
                        "Power",
                        hypotheses[0].name,
                        confidence=hypotheses[0].probability
                    )
            
            # Check threshold-based detection and record
            self._check_thresholds(degraded_slice, detection_times, window_name)
        
        # Summary and comparison
        self._print_detection_summary(detection_times)
    
    def _check_thresholds(self, degraded, detection_times, window_name):
        """Check traditional threshold-based detection."""
        solar_mean = np.mean(degraded.solar_input)
        if solar_mean < 250 * 0.8 and detection_times["threshold_solar"] is None:
            detection_times["threshold_solar"] = "Solar < 80%"
            self.timeline.add_event(
                float(window_name.split("+")[1].split("s")[0]),
                EventSeverity.WARNING,
                "threshold_alert",
                "Power",
                "Solar input < 80%"
            )
        
        batt_q_mean = np.mean(degraded.battery_charge)
        if batt_q_mean < 60 and detection_times["threshold_battery"] is None:
            detection_times["threshold_battery"] = "Battery < 60 Ah"
            self.timeline.add_event(
                float(window_name.split("+")[1].split("s")[0]),
                EventSeverity.WARNING,
                "threshold_alert",
                "Power",
                "Battery charge < 60 Ah"
            )
        
        bus_mean = np.mean(degraded.bus_voltage)
        if bus_mean < 27 and detection_times["threshold_voltage"] is None:
            detection_times["threshold_voltage"] = "Bus < 27V"
            self.timeline.add_event(
                float(window_name.split("+")[1].split("s")[0]),
                EventSeverity.WARNING,
                "threshold_alert",
                "Power",
                "Bus voltage < 27V"
            )
    
    def _print_detection_summary(self, detection_times):
        """Print summary of detection times (data only, no editorial)."""
        print("\n" + "="*80)
        print("LIVE SIMULATION RESULTS")
        print("="*80)
        self.timeline.print_timeline()
        print()


if __name__ == "__main__":
    simulator = GSAT6ASimulator()
    simulator.run_simulation()
    print("\n✓ Simulation complete\n")
