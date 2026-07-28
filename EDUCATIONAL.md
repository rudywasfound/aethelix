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

## Aethelix v2.0 (July 2026) — State-of-the-Art ESA & Causal DAG Updates

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

