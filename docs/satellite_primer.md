# Satellite Fault Management Primer: Aethelix Guide

Welcome to the Aethelix operational environment. This guide is designed for Ground Segment Engineers and Satellite Operators.

## The Operational Workflow

1. **Uplink/Ingestion**:
   - Aethelix ingests telemetry via the **Hardware Abstraction Layer (HAL)**.
   - For flight operations, use the **CCSDS Adapter** to pipe raw Space Packets (CCSDS 133.0-B) directly into the engine.
   - For mission reconstruction, use the **CSV Adapter**.

2. **Automated Anomaly Detection**:
   - Aethelix uses **Sliding Window Normalization**. It learns the "normal" variance of your specific satellite over the last 50-100 ticks.
   - No hard-coded thresholds are required, though a 15% sensitivity is recommended for noisy channels.

3. **Causal Reasoning**:
   - When deviations are detected, the **Stateful Root Cause Ranker** activates.
   - It traces evidence back through the Causal DAG to identify the most likely root cause.
   - **Soft Streak Recovery**: Aethelix maintains a "memory" of faults. A single noisy tick will not reset the diagnosis.

4. **Response Strategy**:
   - Aethelix provides a **3-Tier Action Plan** for every detected fault:
     - **Immediate**: Actions to stabilize the spacecraft.
     - **Short-term**: Diagnostic steps for the next orbital pass.
     - **Escalation**: Triggers for safe-hold or hardware swap.

## Understanding the Dashboard

- **Suppressed Alarms**: Represents the number of secondary sensor alarms that Aethelix correctly identified as "consequential" to a single root cause.
- **Lead Time Advantage**: The time gained by Aethelix detecting the fault "sub-threshold" versus a standard 15% alarm system.
- **Causal Vector Space**: A live visualization of fault propagation through your satellite's subsystems.

## Best Practices
- **Sensor Faults**: If a sensor goes to zero or NaN, Aethelix flags it as a `Sensor Fault`. Do not interpret this as a physical failure unless confirmed by cross-subsystem evidence.
- **Eclipse Transitions**: Aethelix is eclipse-aware. It suppresses solar-panel alarms during UMBRA to avoid false positives during normal orbital transitions.

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
