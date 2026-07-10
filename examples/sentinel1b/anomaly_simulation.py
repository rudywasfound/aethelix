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
    
    # Run causal inference using pluggable YAML DAG configuration
    print("Loading pluggable DAG schema from YAML...")
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'configs', 'sentinel1b.yaml'
    )
    graph = CausalGraph(dag_path=config_path)
    
    ranker = RootCauseRanker(graph)

    print("\nRunning Causal Inference Engine on Degraded Telemetry...")
    hyps = ranker.analyze(nominal, degraded, deviation_threshold=0.15)
    ranker.print_report(hyps)

if __name__ == "__main__":
    run_simulation()

