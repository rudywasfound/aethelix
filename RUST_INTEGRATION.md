# Rust Integration with Aethelix Framework

## Architecture

```
Python Framework (causal_graph, gsat6a)
         ↓
    Detects dropout in telemetry
         ↓
    Calls Rust binary (aethelix_core)
         ↓
Rust: Kalman Filter + Hidden State Inference
         ↓
    Returns JSON: hidden state estimates
         ↓
    Python updates causal inference
         ↓
    Diagnosis with confidence adjustment
```

## When Rust is Invoked

1. **Telemetry Gap Detected**: Consecutive samples missing for 5+ seconds
2. **Call Rust Core**: With gap duration and load power
3. **Get Predictions**: Kalman Filter fills missing samples
4. **Update Graph**: Hidden states constrain causal inference
5. **Resume Inference**: When telemetry resumes, Kalman update corrects predictions

## Usage Example

In `gsat6a/live_simulation.py`:

```python
from causal_graph.kalman_integration import DropoutHandler

# Initialize once
dropout_handler = DropoutHandler()

# In analysis loop
def analyze_telemetry_window(telemetry, sample_indices):
    # Detect gaps
    gaps = dropout_handler.detect_gaps(sample_indices)
    
    if gaps:
        # Get Rust predictions for missing samples
        hidden_states = dropout_handler.fill_gaps(
            gaps=gaps,
            load_power=300.0
        )
        
        # Use in causal inference
        ranker = RootCauseRanker()
        diagnosis = ranker.analyze_with_hidden_states(
            telemetry_dict=telemetry,
            hidden_state_estimates=hidden_states,
            confidence_adjustment=dropout_handler.confidence_degradation
        )
        
        return diagnosis
```

## Building the Rust Core

From project root:

```bash
# Build debug
cd rust_core && cargo build

# Build optimized release
cd rust_core && cargo build --release

# Run tests
cd rust_core && cargo test

# Run demo
./rust_core/target/release/aethelix_core
```

## Output Format

Rust binary outputs JSON on stdout:

```json
{
  "gap_duration_samples": 5,
  "confidence_factor": 0.78,
  "hidden_states": {
    "battery_state": {
      "estimated_value": 0.919,
      "lower_bound": 0.875,
      "upper_bound": 0.963,
      "confidence": 0.78
    },
    "solar_input": {
      "estimated_value": 361.568,
      "lower_bound": 335.866,
      "upper_bound": 387.270,
      "confidence": 0.23
    },
    "battery_efficiency": {
      "estimated_value": 1.0,
      "lower_bound": 0.95,
      "upper_bound": 1.0,
      "confidence": 0.23
    }
  },
  "filled_samples": [
    {"sample": 50, "charge": 80.6, "voltage": 26.91, "solar": 350.0},
    {"sample": 51, "charge": 81.1, "voltage": 26.94, "solar": 350.0},
    ...
  ]
}
```

## FFI Future Work

For tighter integration without subprocess calls:

```python
# PyO3 bindings (future)
from aethelix_core import PowerSystemKalmanFilter, infer_hidden_states

kf = PowerSystemKalmanFilter(nominal_voltage=28.0, nominal_capacity=50.0)
predictions = infer_hidden_states(kf, gap_duration=5, load_power=300.0)
```

## Performance

- Kalman prediction: ~1ms per sample (negligible)
- Subprocess overhead: ~50ms startup
- Total for 5-sample dropout: <100ms

For real-time use with frequent dropouts, FFI bindings recommended.

## Safety & Correctness

[OK] Type-safe matrix operations (nalgebra)
[OK] Bounds checking on all physical quantities
[OK] Covariance matrices guaranteed positive-definite
[OK] Numerical stability through symmetric updates
[OK] Deterministic (seeded) for reproducible tests

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
