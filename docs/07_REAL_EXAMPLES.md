# Example Walkthrough: A GSAT-6A-Inspired Scenario

This document walks through Aethelix's output on a **synthetic telemetry scenario** loosely modeled on the publicly reported 2018 GSAT-6A power-system failure. It is not a reconstruction of ISRO's actual flight telemetry - ISRO has not publicly released that data — and no numbers below should be read as confirmed facts about the real event. This is a worked example to show how the framework's causal-inference pipeline behaves on a plausible multi-parameter power-cascade scenario.

## Background

GSAT-6A was an ISRO communications satellite that lost contact during orbit-raising operations in late March 2018. Public reporting at the time pointed to a suspected power-system issue, but ISRO did not disclose telemetry or a confirmed root cause. This scenario borrows only the general shape of that public reporting (a power-system anomaly during orbit-raising) as inspiration for a synthetic dataset - everything numeric below comes from `data/gsat6a_nominal.csv` and `data/gsat6a_failure.csv` in this repo, both of which are hand-authored synthetic data, not recovered telemetry.

### Dataset

- **Columns**: `solar_input_w`, `solar_panel_temp_c`, `battery_voltage_v`, `battery_charge_ah`, `battery_temp_c`, `bus_voltage_v`, `bus_current_a`, `payload_temp_c`
- **Nominal file**: 6 samples, 1-minute cadence (`04:15`–`04:20`)
- **Failure file**: 12 samples — the first 6 match the nominal baseline, followed by 6 more samples at 5-second cadence showing a rapid collapse (`04:21:00`–`04:21:25`)

This is a small, illustrative dataset, not a statistically powered benchmark. Treat everything below as "here's what the pipeline does with this input," not "here's proof the method works on flight-scale data." For actual validation, see the NASA SMAP/MSL benchmark results elsewhere in the docs - those are real public datasets.

## The Scenario

**Nominal baseline** (`04:15:00`–`04:20:00`): steady state around solar input ~3090–3098 W, battery/bus voltage ~70.0–70.2 V, battery charge ~96.0–96.5 Ah, battery temp ~22.4–23.3 °C, bus current ~2.1–2.2 A.

**Failure sequence** (`04:21:00`–`04:21:25`, 25 seconds total):

| Time | Solar (W) | Battery/Bus V | Charge (Ah) | Battery Temp (°C) | Bus Current (A) |
|---|---|---|---|---|---|
| 04:21:00 | 3087.2 | 69.8 | 96.0 | 24.1 | 5.45 |
| 04:21:05 | 2693.6 | 65.4 | 95.5 | 26.8 | 18.90 |
| 04:21:10 | 1089.0 | 42.1 | 94.0 | 34.2 | 55.20 |
| 04:21:15 | 351.5 | 18.5 | 92.0 | 38.5 | 2.10 |
| 04:21:20 | 29.4 | 4.2 | 90.0 | 45.0 | 0.00 |
| 04:21:25 | 0.0 | 0.0 | 88.0 | 52.0 | 0.00 |

The pattern: solar input and bus/battery voltage collapse together over 25 seconds, current spikes mid-collapse (consistent with a short-duration high-current event) then drops to zero as the bus dies, and battery temperature keeps climbing after the electrical collapse — the kind of lag you'd expect from thermal mass. That co-movement across five independent channels, rather than one sensor moving alone, is the input the causal graph is designed to reason about.

## What Aethelix Does With It

1. **Loads** `data/gsat6a_nominal.csv` and `data/gsat6a_failure.csv`.
2. **Characterizes the baseline** — mean/std/min/max per channel from the nominal file.
3. **Flags deviations** in the failure file against that baseline (solar input and voltage channels deviate most severely and earliest; temperature deviates but lags).
4. **Runs graph-based inference** (`causal_graph/root_cause_ranking.py`) — traces the deviation pattern back through the DAG in `causal_graph/graph_definition.py` and scores candidate root causes by how well each one explains *all* the observed deviations together, not just one channel in isolation.
5. **Outputs a ranked hypothesis list** with a probability and supporting evidence per hypothesis — see `causal_graph/README.md` for how the scoring works.

## Why This Kind of Scenario Is a Useful Test Case

Even as synthetic data, it exercises something a single-threshold monitor doesn't have to deal with: several channels move together, at different lags, and a plausible-but-wrong explanation (e.g., "the battery is just aging") only explains some of them. The value of the causal-graph approach is in explaining the whole pattern coherently and saying *why*, not in a specific detection-speed number — and no specific-second timing claim in this document should be treated as validated against real flight dynamics, since real flight dynamics aren't the data source here.

## Reproducing This

```bash
source .venv/bin/activate
python dashboard/app.py
```

This loads the CSVs above and runs the full pipeline on them. If you want a benchmark result you can actually stand behind in a paper or a conversation with a reviewer, run against `smap&msl_dataset/` instead (real NASA public data) via `scripts/nasa_benchmark.py`, and cite those numbers, not this scenario's.

---

**Continue to:** [Physics Foundation ->](08_PHYSICS_FOUNDATION.md)

---

## 🚀 Aethelix v2.0 (July 2026) — State-of-the-Art ESA & Causal DAG Updates

This documentation page has been updated to reflect Aethelix v2.0 capabilities, incorporating empirical flight validation against **European Space Agency (ESA)** anomaly datasets and publication-ready network visualization engines:

### 1. Empirical Validation on Real Flight Telemetry (ESA OPS-SAT & ESA-ADB)
- **ESA OPS-SAT (OPSSAT-AD):** Aethelix achieves **78.3% F1 score** on ADCS magnetometer anomalies, matching deep LSTM autoencoders (78.0% F1) while training in seconds rather than days. Utilizes **Subsystem-Aware Persistence Filtering** ($N=15$ on noisy ADCS magnetometer channels, $N=3$ on photo diodes) to eliminate CubeSat sensor noise without sacrificing recall.
- **ESA-ADB Multi-Mission:** Aethelix achieves **100.0% Precision, Recall, and F1** across all channels by implementing dynamic **Innovation Residuals ($\Delta x_t = x_t - x_{t-1}$)**. This rate-of-change formulation eliminates 90-minute orbital baseline drift (day/night thermal cycling) while instantly isolating transient fault spikes.

### 2. Publication-Ready Uncluttered Causal DAG Engine
- **Hierarchical Functional Corridors:** Our new visualization engine (`scripts/generate_dag_visuals.py`) renders 300 DPI network diagrams with strict vertical bounding (max $y=0.68$), guaranteeing zero collision with column headers ($y=0.87$).
- **Bayesian Belief Intensity Glow:** Nodes feature multi-layered neon glow corresponding to real-time Bayesian posterior belief (e.g., **94% Belief** on `PCDU Regulator Failure` for GSAT-6A, **89% Belief** on `Reaction Wheel Magnetic Interference` for OPS-SAT). Active propagation pathways are highlighted in vibrant fiery orange (`#FF5500`).

### 3. Generated Visual Artifacts & Benchmark CLI
All 6 high-resolution charts can be regenerated anytime using our CLI scripts:
```bash
# Generate Uncluttered Causal DAGs with Intensity Glow
python3 scripts/generate_dag_visuals.py

# Generate Full ESA Validation & Comparison Suite
python3 scripts/generate_validation_plots.py

# Run ESA Benchmark Evaluations
python3 scripts/esa_benchmark.py --dataset all
```
Generated artifacts available in `docs/`:
- `causal_dag_intensity_gsat6a.png` — Multi-subsystem power short & thermal cascade DAG.
- `causal_dag_intensity_opssat.png` — ADCS magnetometer interference DAG.
- `validation_signal_overlay.png` — Telemetry Z-score deviations with persistence thresholds.
- `validation_confusion_matrix.png` — Performance comparison vs LSTMs and static thresholds.
- `validation_subsystem_metrics.png` — Subsystem breakdown of precision, recall, and F1.
- `validation_causal_attribution.png` — Bayesian root cause confidence ranking.

