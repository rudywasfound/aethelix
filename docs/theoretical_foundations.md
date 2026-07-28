# Theoretical Foundations of Causal Diagnosis

This document formalizes the mathematical necessity of the causal inference approach used in Aethelix, specifically regarding the detection of sub-threshold anomalies.

## Theorem 1 — Univariate Threshold Detection Incompleteness

**Statement:**
Let $F$ be a fault whose causal footprint produces per-channel deviations $d_i < \delta$ for all observable channels $i$, where $\delta$ is the detection threshold.

Any system relying solely on univariate threshold crossings has detection rate $P(\text{detect} | F) = 0$, independent of fault severity, duration, or number of affected channels.

**Proof:**
By definition, no channel $i$ crosses the threshold $\delta$ since $d_i < \delta$ for all $i$. Therefore, at any time $t$, the set of triggered alarms $A = \{i : d_i \ge \delta\}$ is empty. Since the detection function is dependent on $A$ being non-empty, no alarm fires. QED.

**Corollary:**
Multi-channel causal pattern detection is a necessary condition for sub-threshold fault detectability.

## Application in Aethelix

Traditional satellite Ground Control Systems (GCS) rely on out-of-limit (OOL) checks which are univariate threshold detectors. Aethelix overcomes this limitation by modeling the joint distribution and causal dependencies between channels. 

Even if no individual thermistor or voltage sensor identifies a violation, the *simultaneous* subtle drifting of power and thermal residuals creates a causal signature that can be back-propagated to a root cause with high confidence. This provides a significant lead-time advantage over traditional systems.

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
