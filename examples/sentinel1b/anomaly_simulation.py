#!/usr/bin/env python3
"""
Sentinel-1B CAPS Anomaly Simulation
Models the December 2021 28V regulated bus failure.
"""

import sys
import os
import numpy as np
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.power import PowerSimulator
from simulator.thermal import ThermalSimulator
from causal_graph.graph_definition import CausalGraph, NodeType
from causal_graph.root_cause_ranking import RootCauseRanker

import pandas as pd

def load_csv(filename):
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', filename)
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return CombinedTelemetry(
        df['solar_input_w'].values,
        df['battery_voltage_v'].values,
        df['battery_charge_ah'].values,
        df['bus_voltage_v'].values,
        df['battery_temp_c'].values,
        df['solar_panel_temp_c'].values,
        df['payload_temp_c'].values,
        df['bus_current_a'].values,
    )

class CombinedTelemetry:
    def __init__(self, solar_input, battery_voltage, battery_charge, bus_voltage, battery_temp, solar_panel_temp, payload_temp, bus_current):
        self.solar_input = solar_input
        self.battery_voltage = battery_voltage
        self.battery_charge = battery_charge
        self.bus_voltage = bus_voltage
        self.battery_temp = battery_temp
        self.solar_panel_temp = solar_panel_temp
        self.payload_temp = payload_temp
        self.bus_current = bus_current

def run_simulation():
    print("="*60)
    print("SENTINEL-1B CAPS REGULATOR ANOMALY SIMULATION")
    print("="*60)
    
    print("Loading telemetry from CSV...")
    nominal = load_csv('sentinel1b_nominal.csv')
    degraded = load_csv('sentinel1b_failure.csv')
    
    # Run causal inference
    graph = CausalGraph()
    # Add a custom node for the bus regulator failure since this is a specific case study
    graph.add_node("caps_regulator_failure", NodeType.ROOT_CAUSE, "C-SAR Antenna Power Supply Unit Failure")
    graph.add_edge("caps_regulator_failure", "bus_regulation", weight=0.95, mechanism="Complete failure of the regulated 28V bus")
    graph.add_edge("caps_regulator_failure", "payload_temp", weight=0.8, mechanism="Loss of payload power causes temperature drop")
    
    ranker = RootCauseRanker(graph)
    # Monkeypatch the consistency dict for this case study
    original_check_consistency = ranker._check_consistency
    
    def consistency_patch(root_cause, anomalies):
        if root_cause == "caps_regulator_failure":
            expected = {"bus_voltage", "payload_temp"}
            observed = set(anomalies.keys())
            if not expected: return 0.5
            return len(expected & observed) / len(expected)
        return original_check_consistency(root_cause, anomalies)
        
    ranker._check_consistency = consistency_patch
    
    original_explain = ranker._explain_mechanism
    def explain_patch(root_cause, evidence, anomalies):
        if root_cause == "caps_regulator_failure":
            base = "Severe loss of the 28V regulated bus and subsequent payload temp drop indicates CAPS failure."
            if evidence: return f"{base}\\nEvidence: {'; '.join(evidence)}"
            return base
        return original_explain(root_cause, evidence, anomalies)
        
    ranker._explain_mechanism = explain_patch

    print("\\nRunning Causal Inference Engine on Degraded Telemetry...")
    hyps = ranker.analyze(nominal, degraded, deviation_threshold=0.15)
    ranker.print_report(hyps)

if __name__ == "__main__":
    run_simulation()
