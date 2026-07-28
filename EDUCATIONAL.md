# Aethelix: Causal Intelligence for Satellite Fault Management

Aethelix represents a shift from **statistical anomaly detection** (which asks "is this data weird?") to **causal diagnostic reasoning** (which asks "why is this happening and what is the physical root cause?").

## Why Causal Graphs?

Modern satellites are complex, interconnected systems. A failure in one subsystem (e.g., a power drop) often cascades into others (e.g., thermal fluctuations, software reboots). 

Traditional systems use **Fixed Thresholds**:
- Simple to implement.
- **Problem**: Misses "sub-threshold" faults (e.g., a 5% solar degradation) that are still critical but below the 15% alarm line.
- **Problem**: Causes "alarm fatigue" through cascading alerts (one fault triggers 50 alarms).

Aethelix uses **Directed Acyclic Graphs (DAGs)**:
1. **Physics-First**: Relationships are derived from spacecraft design, not just data history.
2. **Consolidation**: Instead of 50 alarms, Aethelix points to the single root cause that explains all 50 deviations.
3. **Sub-threshold Sensitivity**: By summing "weak signals" along causal paths, Aethelix can detect a 5% fault with 90% confidence because the *pattern* across multiple sensors matches the causal model.

## Core Concepts

### 1. Root Causes
These are the physical failures (e.g., `solar_degradation`, `wheel_friction`). They have no parents in the graph.

### 2. Intermediate States
Unobservable physical states (e.g., `battery_efficiency`). They help bridge the gap between root causes and sensors.

### 3. Observables
The telemetry nodes (e.g., `battery_voltage_measured`). These are mapped to actual sensor data.

### 4. Bayesian Ranking
Aethelix uses a rule-based Bayesian approach:
- **Posterior Probability**: Which cause most likely explains the *current* set of anomalies?
- **Confidence**: How certain are we given the *completeness* and *consistency* of the evidence?

## Performance Summary
- **Zero-Shot Detection**: 100% detection rate on NASA SMAP/MSL dataset without any training.
- **Sub-threshold Advantage**: Detects 100% of 5-12% severity faults that traditional 15% thresholds miss entirely.
- **Lead Time**: Provides 30-120 seconds of early warning by detecting the "onset" of a fault before it reaches critical limits.

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
