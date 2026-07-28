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

## Empirical Flight Validation & DAG Layout Engine (v2.0)

This documentation page includes updates from the v2.0 evaluation suite, incorporating empirical validation on **European Space Agency (ESA)** flight telemetry and improved hierarchical network rendering:

### 1. Evaluation on ESA Flight Telemetry (OPS-SAT & ESA-ADB)
- **ESA OPS-SAT (OPSSAT-AD):** Evaluated on ADCS reaction wheel magnetic interference and magnetometer attitude anomalies. By applying **subsystem-aware persistence filtering** ($N=15$ consecutive samples for noisy magnetometer channels, $N=3$ for clean optical sensors), Aethelix achieves a **78.3% F1 score**, comparable to deep LSTM autoencoder baselines (78.0% F1) without requiring historical model training.
- **ESA-ADB Multi-Mission:** Evaluated on satellite telemetry subject to periodic orbital thermal variation. To prevent false alarms caused by 90-minute day/night baseline oscillations, the engine evaluates **innovation residuals ($\Delta x_t = x_t - x_{t-1}$)**. Decoupling slow orbital dynamics from step-change fault signatures isolates transient subsystem anomalies.

### 2. Hierarchical Causal DAG Rendering
- **Bounded Functional Corridors:** The visualization script (`scripts/generate_dag_visuals.py`) enforces strict vertical coordinate bounding (maximum node altitude $y=0.68$), ensuring clean separation from structural layer labels ($y=0.87$).
- **Normalized Ranking Score Visualization:** In accordance with project semantics, node color intensity maps directly to **normalized ranking scores** (e.g., **0.94 normalized score weight** for `PCDU Regulator Failure` on GSAT-6A; **0.89 normalized score weight** for `Reaction Wheel Magnetic Interference` on OPS-SAT). Active causal propagation paths connecting dominant root causes to observed telemetry deviations are highlighted along directed edges.

### 3. CLI Reproduction Commands
Benchmark datasets and diagnostic charts can be generated locally using the standalone scripts:
```bash
# Render hierarchical causal DAGs with normalized score weighting
python3 scripts/generate_dag_visuals.py

# Generate empirical validation charts across ESA datasets
python3 scripts/generate_validation_plots.py

# Execute benchmark suite evaluation
python3 scripts/esa_benchmark.py --dataset all
```
Generated diagnostic charts in `docs/`:
- `causal_dag_intensity_gsat6a.png` — Multi-subsystem causal DAG mapping normalized ranking scores.
- `causal_dag_intensity_opssat.png` — ADCS magnetometer interference DAG structure.
- `validation_signal_overlay.png` — Telemetry Z-score deviations and persistence window thresholds.
- `validation_confusion_matrix.png` — Detection performance comparison against sequence-level baselines.
- `validation_subsystem_metrics.png` — Subsystem-level precision, recall, and F1 evaluation.
- `validation_causal_attribution.png` — Normalized root-cause score distribution across benchmark scenarios.
