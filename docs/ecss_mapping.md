# ECSS Fault Mode Mapping

Aethelix aligns its diagnostic output with the **ECSS-E-ST-10-04C** (Space Engineering: Space Environment) and **ECSS-M-ST-30-01C** (Risk Management) standards.

This mapping ensures that Aethelix reports can be directly ingested into agency FMECA (Failure Mode, Effects, and Criticality Analysis) databases.

## EPS (Electrical Power Subsystem)

| Aethelix Identifier | ECSS Fault ID | Description |
|:---|:---|:---|
| `solar_degradation` | **EPS-FM-001** | Solar Array Power Output Below Nominal |
| `battery_aging` | **EPS-FM-003** | Battery Cell Capacity Degradation |
| `pcdu_regulator_failure` | **EPS-FM-007** | Power Control and Distribution Unit Regulator Fault |

## TCS (Thermal Control Subsystem)

| Aethelix Identifier | ECSS Fault ID | Description |
|:---|:---|:---|
| `battery_heatsink_failure` | **TCS-FM-002** | Battery Interface Thermal Resistance Increase |
| `payload_radiator_degradation` | **TCS-FM-005** | Surface Emissivity Loss / Radiator Fouling |

## ADCS (Attitude Determination & Control)

| Aethelix Identifier | ECSS Fault ID | Description |
|:---|:---|:---|
| `wheel_friction` | **ADC-FM-012** | Reaction Wheel Bearing Friction Increase |
| `gyro_drift` | **ADC-FM-005** | Gyroscope Bias Stability Out of Spec |

## PROP (Propulsion Subsystem)

| Aethelix Identifier | ECSS Fault ID | Description |
|:---|:---|:---|
| `thruster_valve_fault` | **PRP-FM-008** | Thruster Valve Stiction / Leakage |

## Implementation in Aethelix
ECSS identifiers are embedded as metadata within the `CausalGraph` definition. When a diagnosis is generated, the identifier is surfaced in the `RootCauseHypothesis` report, enabling automated cross-referencing with Ground Segment mission databases.

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
