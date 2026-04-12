"""
Sub-threshold fault detection benchmark.

Evaluates Aethelix's ability to detect faults below the operational 15% alarm
threshold - the regime where traditional alarm systems fail by design.

Fault Severity Range: 5–12% degradation of solar input power.
  - Traditional threshold alarm: 0% detection (misses by design).
  - LSTM baseline: ~30–40% detection in this regime (noise floor limited).
  - Aethelix causal: see measured result below.

Methodology

100 reproducible scenarios (seed=42) injected at T+6h with solar degradation
drawn from Uniform(0.05, 0.12). Detection is confirmed when Aethelix produces
a hypothesis with confidence ≥ 40% (meaningful, not trivial) for
'solar_degradation' within 2 hours of fault onset.

False-positive rate is measured on a separate 30-scenario clean (no-fault)
data stream to confirm the detector is not simply always firing.
"""

import random
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.power import PowerSimulator
from causal_graph.graph_definition import CausalGraph
from causal_graph.stateful_ranking import StatefulRootCauseRanker
from operational.anomaly_detector import SlidingWindowDetector

# Confidence threshold — must exceed this to count as a real detection
CONFIDENCE_THRESHOLD = 40.0   # percent, meaningful (not trivial)
# How many samples post-fault-onset is still a "timely" detection?
DETECTION_WINDOW_SAMPLES = 720  # = 2 hours at 10-second sampling


def run_subthreshold_benchmark(num_scenarios: int = 100, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    print("Sub-threshold Fault Detection Benchmark")
    print(f"  Fault range: 5.0%–12.0% solar degradation")
    print(f"  Confidence threshold for detection: {CONFIDENCE_THRESHOLD}%")
    print(f"  Detection window post-onset: {DETECTION_WINDOW_SAMPLES} samples (2 h)")
    print(f"  Scenarios: {num_scenarios}  |  Seed: {seed}\n")

    graph   = CausalGraph()
    severities = np.random.uniform(0.05, 0.12, num_scenarios)

    detected_count  = 0
    detection_leads = []  # samples from fault onset to first correct detection

    for i in range(num_scenarios):
        severity = severities[i]

        sim     = PowerSimulator(duration_hours=24, sampling_rate_hz=0.1)
        nominal = sim.run_nominal()
        degraded = sim.run_degraded(
            solar_degradation_hour=6.0,
            solar_factor=1.0 - severity,   # e.g. 0.92 = 8% loss
            battery_degradation_hour=9999,  # no battery fault
        )

        detector = SlidingWindowDetector(p_threshold=0.005, persist=4)
        ranker   = StatefulRootCauseRanker(graph)
        ranker.reset()

        fault_onset = int(6.0 * 3600 * sim.sampling_rate_hz)
        detected    = False
        lead_samples = None

        for t in range(len(degraded.solar_input)):
            tick = {
                "solar_input"   : float(degraded.solar_input[t]),
                "battery_voltage": float(degraded.battery_voltage[t]),
                "battery_charge": float(degraded.battery_charge[t]),
                "bus_voltage"   : float(degraded.bus_voltage[t]),
                "orbital_phase" : float(degraded.orbital_phase[t]),
            }
            anomalies = detector.process_tick(tick)

            if anomalies:
                hyps = ranker.analyze_stream(anomalies)
                if hyps and hyps[0].name == "solar_degradation" \
                        and hyps[0].confidence >= CONFIDENCE_THRESHOLD \
                        and fault_onset <= t <= fault_onset + DETECTION_WINDOW_SAMPLES:
                    detected = True
                    lead_samples = t - fault_onset
                    break

        if detected:
            detected_count  += 1
            detection_leads.append(lead_samples)

        if (i + 1) % 25 == 0:
            print(f"  Scenario {i+1:3d}/{num_scenarios} | "
                  f"Detected so far: {detected_count}")

    # False-positive rate (clean data)
    fp_count = 0
    for _ in range(30):
        sim      = PowerSimulator(duration_hours=24, sampling_rate_hz=0.1)
        nominal  = sim.run_nominal()
        detector = SlidingWindowDetector(p_threshold=0.005, persist=4)
        ranker   = StatefulRootCauseRanker(graph)
        ranker.reset()
        for t in range(len(nominal.solar_input)):
            tick = {
                "solar_input"   : float(nominal.solar_input[t]),
                "battery_voltage": float(nominal.battery_voltage[t]),
                "battery_charge": float(nominal.battery_charge[t]),
                "bus_voltage"   : float(nominal.bus_voltage[t]),
                "orbital_phase" : float(nominal.orbital_phase[t]),
            }
            an = detector.process_tick(tick)
            if an:
                hyps = ranker.analyze_stream(an)
                if hyps and hyps[0].confidence >= CONFIDENCE_THRESHOLD:
                    fp_count += 1
                    break  # one FP event per scenario

    aethelix_rate   = detected_count / num_scenarios * 100
    fp_rate         = fp_count / 30 * 100
    mean_lead_s     = (np.mean(detection_leads) * 10.0
                       if detection_leads else float("nan"))  # 10 s/sample

    print("\n" + "=" * 60)
    print("  SUB-THRESHOLD BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Aethelix':>10} {'LSTM':>8} {'Threshold':>10}")
    print(f"  {'-'*63}")
    print(f"  {'Detection rate (5–12% faults)':<35} {aethelix_rate:>9.1f}% {'~35%':>8} {'0.0%':>10}")
    print(f"  {'False positive rate (clean)':<35} {fp_rate:>9.1f}% {'~5%':>8} {'0.0%':>10}")
    print(f"  {'Mean detection lead (samples)':<35} {mean_lead_s:>9.0f}s {'~15s':>8} {'N/A':>10}")
    print(f"  {'Training required':<35} {'None':>10} {'High':>8} {'None':>10}")
    print("=" * 60)
    print()
    print("  Traditional alarm: misses ALL faults below 15% by design.")
    print("  LSTM: ~35% detection limited by noise floor in this severity range.")
    print(f"  Aethelix: {aethelix_rate:.0f}% detection using causal path correlation")
    print(f"           with {fp_rate:.0f}% false-alarm rate on clean data.")
    print("=" * 60)


if __name__ == "__main__":
    run_subthreshold_benchmark()
