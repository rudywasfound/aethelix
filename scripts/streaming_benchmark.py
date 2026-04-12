"""
Detection lead-time benchmark.

Measures the time advantage of Aethelix causal inference over a traditional
fixed-threshold alarm system for solar degradation faults.

Lead-Time Definition
--------------------
Lead time = t_threshold_alarm – t_aethelix_detection

Where:
  t_aethelix_detection  = first sample at which Aethelix produces correct
                          top-1 hypothesis with confidence ≥ 40%.
  t_threshold_alarm     = first sample where the degraded channel deviation
                          exceeds 15% of the nominal mean (OOL trigger).

A positive lead time means Aethelix detects the fault EARLIER than the
threshold alarm. The threshold alarm is guaranteed to miss sub-15% faults
(lead time = undefined / +∞ advantage).

Methodology
-----------
50 scenarios (seed=42), solar degradation 15–40% injected at T=6h.
Each sample is 10 seconds (0.1 Hz). Results in seconds.
"""

import random
import sys
import time
import numpy as np
import pandas as pd
import queue
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator.power import PowerSimulator
from causal_graph.graph_definition import CausalGraph
from causal_graph.stateful_ranking import StatefulRootCauseRanker
from operational.anomaly_detector import SlidingWindowDetector

CONFIDENCE_THRESHOLD = 40.0    # % — meaningful detection
SAMPLE_RATE_HZ       = 0.1    # 1 sample / 10 seconds
FAULT_HOUR           = 6.0
THRESHOLD_FRACTION   = 0.15   # 15% deviation = OOL alarm fires


def _nominal_solar_mean(sim: PowerSimulator) -> float:
    """Get mean solar input in nominal (pre-fault) window."""
    nom = sim.run_nominal()
    fault_idx = int(FAULT_HOUR * 3600 * sim.sampling_rate_hz)
    return float(np.mean(nom.solar_input[:fault_idx]))


def run_lead_time_benchmark(num_scenarios: int = 50, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    print("Detection Lead-Time Benchmark")
    print(f"  Fault type:   Solar degradation (15%–40%)")
    print(f"  Fault onset:  T = {FAULT_HOUR:.0f}h")
    print(f"  Scenarios:    {num_scenarios}  |  Seed: {seed}")
    print(f"  Aethelix confidence threshold: {CONFIDENCE_THRESHOLD}%")
    print(f"  OOL threshold: {THRESHOLD_FRACTION*100:.0f}% deviation\n")

    graph      = CausalGraph()
    severities = np.random.uniform(0.15, 0.40, num_scenarios)

    lead_times_s = []    # seconds Aethelix fires before threshold
    aethelix_miss = 0
    threshold_miss = 0

    for i in range(num_scenarios):
        severity = severities[i]
        factor   = 1.0 - severity  # e.g. 0.75 for 25% loss

        sim      = PowerSimulator(duration_hours=24, sampling_rate_hz=SAMPLE_RATE_HZ)
        nom_mean = _nominal_solar_mean(sim)
        degraded = sim.run_degraded(
            solar_degradation_hour=FAULT_HOUR,
            solar_factor=factor,
            battery_degradation_hour=9999,
        )

        fault_idx = int(FAULT_HOUR * 3600 * SAMPLE_RATE_HZ)

        detector     = SlidingWindowDetector(p_threshold=0.005, persist=4)
        ranker       = StatefulRootCauseRanker(graph)
        ranker.reset()

        t_aethelix   = None
        t_threshold  = None

        for t in range(len(degraded.solar_input)):
            solar_val = float(degraded.solar_input[t])

            # Threshold alarm: fires when deviation > 15% from nominal mean
            if t_threshold is None and t >= fault_idx:
                deviation = abs(solar_val - nom_mean) / nom_mean
                if deviation > THRESHOLD_FRACTION:
                    t_threshold = t

            # Aethelix detection
            if t_aethelix is None:
                tick = {
                    "solar_input"    : solar_val,
                    "battery_voltage": float(degraded.battery_voltage[t]),
                    "battery_charge" : float(degraded.battery_charge[t]),
                    "bus_voltage"    : float(degraded.bus_voltage[t]),
                    "orbital_phase"  : float(degraded.orbital_phase[t]),
                }
                anomalies = detector.process_tick(tick)
                if anomalies:
                    hyps = ranker.analyze_stream(anomalies)
                    if (hyps and hyps[0].name == "solar_degradation"
                            and hyps[0].confidence >= CONFIDENCE_THRESHOLD
                            and t >= fault_idx):
                        t_aethelix = t

            if t_aethelix is not None and t_threshold is not None:
                break

        # Convert sample indices to seconds
        dt_per_sample = 1.0 / SAMPLE_RATE_HZ  # = 10 s

        if t_aethelix is None:
            aethelix_miss += 1
            lead_s = None
        elif t_threshold is None:
            # Threshold never fired — Aethelix-only detection (infinite advantage)
            threshold_miss += 1
            lead_s = None  # handled separately
        else:
            lead_s = (t_threshold - t_aethelix) * dt_per_sample
            lead_times_s.append(lead_s)

    # ── Summary ─────────────────────────────────────────────────────────────
    if lead_times_s:
        mean_lead   = np.mean(lead_times_s)
        median_lead = np.median(lead_times_s)
        p75_lead    = np.percentile(lead_times_s, 75)
        positive    = sum(1 for l in lead_times_s if l > 0)
    else:
        mean_lead = median_lead = p75_lead = float("nan")
        positive  = 0

    print("=" * 60)
    print("  DETECTION LEAD-TIME RESULTS")
    print("=" * 60)
    print(f"  Scenarios run:               {num_scenarios}")
    print(f"  Aethelix detected:           {num_scenarios - aethelix_miss}")
    print(f"  Threshold fired:             {num_scenarios - threshold_miss}")
    print(f"  Threshold-only misses:       {threshold_miss} (severity too mild)")
    print()
    print(f"  Lead-time statistics (Aethelix vs OOL threshold):")
    print(f"    Mean lead time:            {mean_lead:+.1f} s")
    print(f"    Median lead time:          {median_lead:+.1f} s")
    print(f"    75th percentile:           {p75_lead:+.1f} s")
    print(f"    Scenarios Aethelix faster: {positive}/{len(lead_times_s)}")
    print()
    print(f"  Published comparisons:")
    print(f"    LSTM Telemanom lead time:  ~+10 to +20 s  (requires training)")
    print(f"    OOL threshold lead time:   0 s            (baseline)")
    print(f"    Aethelix lead time:        {mean_lead:+.1f} s (zero training)")
    print("=" * 60)


if __name__ == "__main__":
    run_lead_time_benchmark()
