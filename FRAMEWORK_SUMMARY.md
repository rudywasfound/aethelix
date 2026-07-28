# GSAT-6A Analysis Framework Refactor

## Overview

The GSAT-6A simulation and forensic analysis have been refactored to separate **data collection** from **output generation**. All output is now framework-driven, not hardcoded in simulation code.

## Architecture

### 1. Timeline Framework (`timeline.py`)
- Collects detection events in chronological order
- Records event type, severity, subsystem, and confidence
- Generates formatted timeline output
- Calculates lead times between detection methods

### 2. Findings Framework (`findings.py`)
- Aggregates telemetry statistics (nominal vs degraded)
- Tracks cascade events
- Generates deviations analysis
- Produces detection comparisons
- Calculates mission impact

### 3. Visualizer Framework (`visualizer.py`)
- Generates timeline visualization
- Plots telemetry deviations (bar charts)
- Creates detection comparison charts
- All graphs are data-driven from analysis results

## Workflow

### Forensic Analysis (`forensics.py`)
```
1. Initialize frameworks (Timeline, Findings, Visualizer)
2. Generate telemetry (simulators)
3. Analyze with causal inference
4. Record events → Timeline
5. Collect stats → Findings
6. Generate all output:
   - `print_analysis()` - text output
   - `generate_graphs()` - visualization
```

### Live Simulation (`live_simulation.py`)
```
1. Initialize frameworks (Timeline, Findings)
2. Run simulation
3. Record causal detections → Timeline
4. Record threshold alerts → Timeline
5. Print timeline
```

## Key Benefits

- **Data-driven output**: Text and graphs generated from actual measurements  
- **Separation of concerns**: Simulation ≠ presentation  
- **Extensible**: Add new analysis types without modifying simulation  
- **Consistent**: All outputs follow same data patterns  
- **Testable**: Framework can be tested independently  
- **Maintainable**: Change output format in one place  

## Generated Outputs

### Text Output
- Timeline of events (detection times, severity)
- Telemetry deviations (nominal vs degraded)
- Detection comparison (causal vs threshold-based)
- Mission impact analysis

### Graphs
- `gsat6a_timeline.png` - Event timeline with detection points
- `gsat6a_telemetry_deviations.png` - Nominal vs degraded comparison
- `gsat6a_detection_comparison.png` - Method comparison and lead time

## Usage

```bash
# Forensic analysis with graphs
python gsat6a/live_simulation_main.py forensics

# Live simulation with timeline
python gsat6a/live_simulation_main.py simulation

# Full mission analysis (existing)
python gsat6a/live_simulation_main.py mission
```

## Data Flow

```
Simulation/Analysis Code
         ↓
    [Collect Data]
         ↓
Timeline ← Events ← Detections
Findings ← Stats  ← Telemetry
         ↓
   [Framework]
         ↓
  [Generate Output]
         ↓
Text Output + Graphs
```

## Example Output

### Timeline Event
```
[CRITICAL] T+    0.0s [Power       ] Solar degradation detected (100%)
[WARNING] T+    0.0s [Power       ] Solar Power = 372W (24.9% drop)
```

### Telemetry Deviation
```
Battery Charge (Ah):
  Nominal:      64.32 ± 35.29
  Degraded:     31.61 ± 23.64
  Change:      -32.72 ( +50.9%) ↓
```

### Detection Comparison
```
LEAD TIME ADVANTAGE: 0.0s
Can enable preventive action
before cascade failure.
```

## Files Modified

- `forensics.py` - Removed hardcoded output, uses frameworks [OK]
- `live_simulation.py` - Removed hardcoded output, uses frameworks [OK]
- `mission_analysis.py` - Removed hardcoded visualization code, uses frameworks [OK]
- `live_simulation_main.py` - Added graph generation call [OK]

## Files Created

- `timeline.py` - Event timeline framework [OK]
- `findings.py` - Analysis findings framework [OK]
- `visualizer.py` - Graph generation framework [OK]

## Architecture Summary

```
Analysis Code                Framework Components             Output
═════════════════════════════════════════════════════════════════════

forensics.py              Timeline                  print_timeline()
  → Analyze              + Findings                 + print_deviations()
  → Record events   ───> + Visualizer          ──> + print_comparison()
  → Collect stats                                  + print_impact()
                                                   + 3 PNG graphs
live_simulation.py
  → Run simulation
  → Record events
  
mission_analysis.py
  → Load CSV data
  → Analyze w/causal
  → Record anomalies
```

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
