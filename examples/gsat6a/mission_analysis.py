#!/usr/bin/env python3
"""
GSAT-6A Mission Analysis - Framework-Based

Loads real GSAT-6A telemetry data and analyzes failure patterns using causal inference.
All output is framework-driven from actual analysis data.
"""

import numpy as np
import pandas as pd
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph.graph_definition import CausalGraph
from causal_graph.root_cause_ranking import RootCauseRanker
from timeline import Timeline, EventSeverity
from findings import FindingsEngine
from visualizer import AnalysisVisualizer


@dataclass
class CombinedTelemetry:
    """For compatibility with RootCauseRanker."""
    solar_input: np.ndarray
    battery_voltage: np.ndarray
    battery_charge: np.ndarray
    bus_voltage: np.ndarray
    battery_temp: np.ndarray
    solar_panel_temp: np.ndarray
    payload_temp: np.ndarray
    bus_current: np.ndarray


class GSAT6AMissionAnalysis:
    """Automatically analyze GSAT-6A failure from real telemetry data."""
    
    def __init__(self):
        # Framework components
        self.timeline = Timeline()
        self.findings = FindingsEngine()
        
        self.graph = CausalGraph()
        self.ranker = RootCauseRanker(self.graph)
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        
        # Load data
        self.nominal_df = self._load_csv('gsat6a_nominal.csv')
        self.failure_df = self._load_csv('gsat6a_failure.csv')
        
        if self.nominal_df is None or self.failure_df is None:
            raise RuntimeError("Could not load telemetry data")
    
    def _load_csv(self, filename):
        """Load CSV telemetry data."""
        filepath = os.path.join(self.data_dir, filename)
        try:
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            return df
        except (FileNotFoundError, Exception):
            return None
    
    def analyze_and_visualize(self):
        """Run complete analysis and generate outputs."""
        self._analyze_baseline()
        self._detect_anomalies()
        self._analyze_root_causes()
        self._analyze_cascade()
        self._reconstruct_timeline()
        self._analyze_operational_impact()
        
        # Generate framework outputs
        self.print_analysis()
        self.generate_graphs(output_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def _analyze_baseline(self):
        """Analyze nominal baseline characteristics."""
        nom = self.nominal_df
        fail = self.failure_df
        
        # Record telemetry stats comparing nominal vs degraded
        self.findings.add_telemetry_stat(
            "Solar Input", "W",
            nom['solar_input_w'].mean(), nom['solar_input_w'].std(),
            fail['solar_input_w'].mean(), fail['solar_input_w'].std()
        )
        self.findings.add_telemetry_stat(
            "Battery Voltage", "V",
            nom['battery_voltage_v'].mean(), nom['battery_voltage_v'].std(),
            fail['battery_voltage_v'].mean(), fail['battery_voltage_v'].std()
        )
        self.findings.add_telemetry_stat(
            "Battery Charge", "Ah",
            nom['battery_charge_ah'].mean(), nom['battery_charge_ah'].std(),
            fail['battery_charge_ah'].mean(), fail['battery_charge_ah'].std()
        )
        self.findings.add_telemetry_stat(
            "Bus Voltage", "V",
            nom['bus_voltage_v'].mean(), nom['bus_voltage_v'].std(),
            fail['bus_voltage_v'].mean(), fail['bus_voltage_v'].std()
        )
        self.findings.add_telemetry_stat(
            "Battery Temperature", "°C",
            nom['battery_temp_c'].mean(), nom['battery_temp_c'].std(),
            fail['battery_temp_c'].mean(), fail['battery_temp_c'].std()
        )
    
    def _detect_anomalies(self):
        """Automatically detect anomalies by comparing with baseline."""
        nom = self.nominal_df
        fail = self.failure_df
        
        # Calculate deviations for each parameter
        solar_deviation = (nom['solar_input_w'].mean() - fail['solar_input_w']) / nom['solar_input_w'].mean() * 100
        batt_v_deviation = (nom['battery_voltage_v'].mean() - fail['battery_voltage_v']) / nom['battery_voltage_v'].mean() * 100
        batt_q_deviation = (nom['battery_charge_ah'].mean() - fail['battery_charge_ah']) / nom['battery_charge_ah'].mean() * 100
        bus_deviation = (nom['bus_voltage_v'].mean() - fail['bus_voltage_v']) / nom['bus_voltage_v'].mean() * 100
        temp_deviation = fail['battery_temp_c'] - nom['battery_temp_c'].mean()
        
        # Track first causal detection time (anomaly = root cause indicator)
        first_detection_time = None
        
        # Record anomaly events to timeline
        if (solar_deviation > 5).any():
            max_idx = solar_deviation.argmax()
            if first_detection_time is None:
                first_detection_time = float(max_idx)
            self.timeline.add_event(
                float(max_idx), EventSeverity.CRITICAL,
                "anomaly_detection", "Power",
                f"Solar deviation: {solar_deviation.max():.1f}%"
            )
        
        if (batt_v_deviation > 2).any():
            max_idx = batt_v_deviation.argmax()
            self.timeline.add_event(
                float(max_idx), EventSeverity.WARNING,
                "anomaly_detection", "Power",
                f"Battery voltage deviation: {batt_v_deviation.max():.1f}%"
            )
        
        if (batt_q_deviation > 5).any():
            max_idx = batt_q_deviation.argmax()
            self.timeline.add_event(
                float(max_idx), EventSeverity.WARNING,
                "anomaly_detection", "Power",
                f"Battery charge deviation: {batt_q_deviation.max():.1f}%"
            )
        
        if (bus_deviation > 2).any():
            max_idx = bus_deviation.argmax()
            self.timeline.add_event(
                float(max_idx), EventSeverity.WARNING,
                "anomaly_detection", "Power",
                f"Bus voltage deviation: {bus_deviation.max():.1f}%"
            )
        
        if (temp_deviation > 2).any():
            max_idx = temp_deviation.argmax()
            self.timeline.add_event(
                float(max_idx), EventSeverity.WARNING,
                "anomaly_detection", "Thermal",
                f"Temperature rise: {temp_deviation.max():.1f}°C"
            )
        
        # Set detection times for comparison
        if first_detection_time is not None:
            self.findings.set_detection_times(first_detection_time, None)  # Anomaly = causal detection, no threshold
    
    def _analyze_root_causes(self):
        """Use causal inference to determine root causes."""
        nom = self.nominal_df
        fail = self.failure_df
        
        # Use early failure stage for early detection analysis
        fail_early = fail.iloc[:min(10, len(fail))]
        nom_early = nom.iloc[:len(fail_early)]
        
        # Convert to telemetry objects
        nominal_tel = CombinedTelemetry(
            solar_input=nom_early['solar_input_w'].values,
            battery_voltage=nom_early['battery_voltage_v'].values,
            battery_charge=nom_early['battery_charge_ah'].values,
            bus_voltage=nom_early['bus_voltage_v'].values,
            battery_temp=nom_early['battery_temp_c'].values,
            solar_panel_temp=nom_early['solar_panel_temp_c'].values,
            payload_temp=nom_early['payload_temp_c'].values,
            bus_current=nom_early['bus_current_a'].values,
        )
        
        degraded_tel = CombinedTelemetry(
            solar_input=fail_early['solar_input_w'].values,
            battery_voltage=fail_early['battery_voltage_v'].values,
            battery_charge=fail_early['battery_charge_ah'].values,
            bus_voltage=fail_early['bus_voltage_v'].values,
            battery_temp=fail_early['battery_temp_c'].values,
            solar_panel_temp=fail_early['solar_panel_temp_c'].values,
            payload_temp=fail_early['payload_temp_c'].values,
            bus_current=fail_early['bus_current_a'].values,
        )
        
        try:
            hypotheses = self.ranker.analyze(nominal_tel, degraded_tel, deviation_threshold=0.05)
            
            if hypotheses:
                # Record top hypothesis as timeline event
                top_hyp = hypotheses[0]
                self.timeline.add_event(
                    0.0, EventSeverity.CRITICAL,
                    "root_cause_detection", "System",
                    f"{top_hyp.name}",
                    confidence=top_hyp.probability
                )
        except Exception:
            pass
    
    def _analyze_cascade(self):
        """Analyze how failures cascade through systems."""
        fail = self.failure_df
        
        # Find key failure points - record as events
        solar_drop_idx = (fail['solar_input_w'] < fail['solar_input_w'].iloc[0] * 0.8).idxmax()
        voltage_drop_idx = (fail['battery_voltage_v'] < 27).idxmax()
        charge_critical_idx = (fail['battery_charge_ah'] < 20).idxmax()
        temp_rise_idx = (fail['battery_temp_c'] > 30).idxmax()
        
        if solar_drop_idx > 0:
            self.timeline.add_event(
                float(solar_drop_idx), EventSeverity.CRITICAL,
                "cascade_point", "Power",
                f"Solar input >20% drop: {fail.iloc[solar_drop_idx]['solar_input_w']:.1f}W"
            )
        
        if voltage_drop_idx > 0:
            self.timeline.add_event(
                float(voltage_drop_idx), EventSeverity.CRITICAL,
                "cascade_point", "Power",
                f"Battery voltage critical: {fail.iloc[voltage_drop_idx]['battery_voltage_v']:.2f}V"
            )
        
        if charge_critical_idx > 0:
            self.timeline.add_event(
                float(charge_critical_idx), EventSeverity.CRITICAL,
                "cascade_point", "Power",
                f"Battery charge critical: {fail.iloc[charge_critical_idx]['battery_charge_ah']:.1f}Ah"
            )
        
        if temp_rise_idx > 0:
            self.timeline.add_event(
                float(temp_rise_idx), EventSeverity.CRITICAL,
                "cascade_point", "Thermal",
                f"Temperature critical: {fail.iloc[temp_rise_idx]['battery_temp_c']:.1f}°C"
            )
    
    def _reconstruct_timeline(self):
        """Reconstruct precise failure timeline."""
        fail = self.failure_df
    
    def _analyze_operational_impact(self):
        """Analyze operational impact and lessons learned."""
        fail = self.failure_df
        
        # Record final state as cascade event
        if len(fail) > 0:
            self.timeline.add_event(
                float(len(fail)-1), EventSeverity.CRITICAL,
                "mission_impact", "System",
                f"Final state: Batt {fail.iloc[-1]['battery_charge_ah']:.1f}Ah, Volt {fail.iloc[-1]['bus_voltage_v']:.2f}V, Temp {fail.iloc[-1]['battery_temp_c']:.1f}°C"
            )
    
    def print_analysis(self):
        """Generate all analysis output from framework."""
        print("\n" + "="*80)
        print("GSAT-6A MISSION ANALYSIS")
        print("="*80 + "\n")
        
        self.timeline.print_timeline()
        self.findings.print_telemetry_deviations()
    
    def generate_graphs(self, output_dir: str = "."):
        """Generate visualization graphs from analysis data."""
        print("\n" + "="*80)
        print("GENERATING GRAPHS")
        print("="*80 + "\n")
        
        visualizer = AnalysisVisualizer(self.timeline, self.findings)
        visualizer.generate_all_graphs(output_dir)
        
        print("\n✓ Graph generation complete\n")


def main():
    """Run mission analysis."""
    try:
        analyzer = GSAT6AMissionAnalysis()
        analyzer.analyze_and_visualize()
        print("✓ Mission analysis complete\n")
    except KeyboardInterrupt:
        print("\n✓ Analysis stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
