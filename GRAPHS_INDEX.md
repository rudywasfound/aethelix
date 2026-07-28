# Aethelix GSAT-6A Analysis - Graphs & Documentation Index

## Quick Links

### Generated Visualization Files (Framework-Based & ESA Suite)
All PNG files are generated from actual analysis and benchmark data:

#### A. Causal DAG Intensity & Network Visualizations (`docs/`)
1. **[docs/causal_dag_intensity_gsat6a.png](docs/causal_dag_intensity_gsat6a.png)** (750 KB)
   - Hierarchical multi-subsystem Causal DAG for GSAT-6A power short & thermal cascade.
   - Highlights 94% normalized score weight on `PCDU Regulator Failure` with active causal propagation paths highlighted.
2. **[docs/causal_dag_intensity_opssat.png](docs/causal_dag_intensity_opssat.png)** (722 KB)
   - Hierarchical ADCS subsystem Causal DAG for ESA OPS-SAT spacecraft.
   - Traces multi-axis magnetometer spikes back to `Reaction Wheel Magnetic Interference` (0.89 normalized score weight).

#### B. ESA Validation & Benchmark Charts (`docs/`)
3. **[docs/validation_signal_overlay.png](docs/validation_signal_overlay.png)** (681 KB)
   - Multi-panel sensor telemetry Z-score waveforms with persistence streak alarm thresholds ($N=15$ and $N=3$).
4. **[docs/validation_confusion_matrix.png](docs/validation_confusion_matrix.png)** (202 KB)
   - Side-by-side confusion matrices comparing Aethelix vs LSTMs and static thresholds on ESA anomaly datasets.
5. **[docs/validation_subsystem_metrics.png](docs/validation_subsystem_metrics.png)** (220 KB)
   - Grouped bar chart breaking down precision, recall, and F1 across spacecraft subsystems (ADCS, Power, Thermal).
6. **[docs/validation_causal_attribution.png](docs/validation_causal_attribution.png)** (283 KB)
   - Horizontal bar chart ranking Dominant root cause posterior belief across anomaly scenarios.

#### C. GSAT-6A Timeline & Forensics (`root` & `docs/`)
7. **[gsat6a_timeline.png](gsat6a_timeline.png)** (89 KB)
   - Detection timeline showing chronological anomalies and critical events.
8. **[gsat6a_telemetry_deviations.png](gsat6a_telemetry_deviations.png)** (141 KB)
   - Bar charts comparing nominal vs degraded telemetry loss percentages.
9. **[gsat6a_detection_comparison.png](gsat6a_detection_comparison.png)** (53 KB)
   - Method comparison showing lead-time advantage of causal inference over static OOL limits.

### Documentation Files

#### Main Analysis Documents
- **[FRAMEWORK_SUMMARY.md](FRAMEWORK_SUMMARY.md)** [START HERE]
  - Complete refactoring summary
  - Framework architecture (Timeline, Findings, Visualizer)
  - Data-driven output explanation
  - Workflow diagrams

- **[README.md](README.md)**
  - Quick start instructions
  - Framework overview
  - Key results summary

#### Framework Documentation
- **[docs/01_INTRODUCTION.md](docs/01_INTRODUCTION.md)**
  - Framework overview
  - Real-world GSAT-6A example
  - Why causal inference matters

### Source Data
- **[data/gsat6a_nominal.csv](data/gsat6a_nominal.csv)** - 25 samples (healthy)
- **[data/gsat6a_failure.csv](data/gsat6a_failure.csv)** - 38 samples (failure cascade)

---

## Quick Start

1. **View the graphs**: Look at the 3 PNG files above
2. **Understand the approach**: Read [FRAMEWORK_SUMMARY.md](FRAMEWORK_SUMMARY.md)
3. **Regenerate yourself**:
   ```bash
   source .venv/bin/activate
   python gsat6a/live_simulation_main.py forensics    # Forensic analysis
   python gsat6a/live_simulation_main.py simulation   # Live simulation
   python gsat6a/live_simulation_main.py mission      # Mission analysis
   ```

---

## What Each Graph Shows

### Timeline Graph
**Shows**: Chronological detection of anomalies and critical events
- Critical events marked in red (root causes, failures)
- Warnings marked in orange (threshold violations)
- Time-stamped with event descriptions
- Covers all detected events from analysis

**Use**: Understand when failures are detected and progression over time

### Telemetry Deviations Graph
**Shows**: Nominal vs degraded state comparison with loss percentages
- Side-by-side bar charts for each parameter
- Loss percentages clearly labeled (↓ for loss, ↑ for rise)
- 6 key parameters: solar, voltage, charge, bus, temperature
- Generated from actual simulation/mission data

**Use**: Quantify parameter changes during failure

### Detection Comparison Graph
**Shows**: Method comparison with lead time advantage
- Causal inference vs threshold-based detection timing
- Lead time in seconds between methods (shown only when both methods detect)
- Analysis summary box with advantages of each method
- Data-driven from analysis results
- Intelligently handles partial data:
  - If only causal inference detects: shows green bar + summary explaining threshold wasn't triggered
  - If only threshold detects: shows orange bar + explanation
  - If both detect: shows both bars with lead time annotation between them

**Use**: Demonstrate early warning benefit of causal inference and explain why certain detection methods may not trigger

---

## Key Analysis Results

### Detection Advantage (Forensics)
- **Causal Inference**: T+0 seconds
- **Threshold-Based**: T+0 seconds (in ideal simulation)
- **Lead Time**: Demonstrates causal root cause vs symptom detection

### Detection Advantage (Mission Analysis)
- **Causal Inference**: T+36 seconds (first solar deviation >5%)
- **Threshold-Based**: Never triggered (CSV data shows no threshold crossing)
- **Root Cause**: Solar degradation identified early via causal inference
- **Graph Rendering**: Detection comparison displays causal bar only, with summary explaining why threshold wasn't triggered

### Failure Timeline (Mission Data)
- T+4s: Solar input >20% drop detected
- T+20s: Battery voltage critical (<27V)
- T+27s: Temperature critical (>30°C)
- T+30s: Battery charge critical (<20Ah)
- T+36s: Solar deviation 53.6% detected
- T+37s: Final state (Batt 0.1Ah, Volt 15.2V)

---

## How Graphs Are Generated

### Forensics Mode
```bash
python gsat6a/live_simulation_main.py forensics
```
- Generates nominal and degraded power/thermal telemetry
- Runs causal inference analysis
- Records timeline events
- Generates 3 graphs

### Live Simulation Mode
```bash
python gsat6a/live_simulation_main.py simulation
```
- Simulates failure sequence in real-time
- Detects with both causal and threshold methods
- Records all detection events to timeline
- Prints timeline of events

### Mission Analysis Mode
```bash
python gsat6a/live_simulation_main.py mission
```
- Loads real GSAT-6A CSV data
- Analyzes with causal inference
- Detects anomalies automatically
- Generates timeline and deviations
- Outputs 3 framework-based graphs

---

## Framework Architecture

```
Analysis Code                Framework Components             Output
═════════════════════════════════════════════════════════════════════

forensics.py              Timeline                  print_timeline()
  → Analyze              + Findings                 + print_deviations()
  → Record events   ───> + Visualizer          ──> + print_comparison()
  → Collect stats                                  + 3 PNG graphs

live_simulation.py
  → Run simulation
  → Record events
  
mission_analysis.py
  → Load CSV data
  → Analyze w/causal
  → Record anomalies
```

### Key Components

1. **timeline.py** - Timeline framework
   - Records detection events with severity
   - Generates formatted timeline output
   - Calculates lead times

2. **findings.py** - Findings framework
   - Aggregates telemetry statistics
   - Tracks deviations and anomalies
   - Generates deviation analysis

3. **visualizer.py** - Visualizer framework
    - Generates timeline graph
    - Creates telemetry comparison charts
    - Produces detection comparison plot
    - Intelligent rendering: handles partial detection data (one method triggered, other not)
    - Left panel: Bar chart of available detection times
    - Right panel: Analysis summary explaining advantages/limitations of each method

---

## Detection Comparison Graph - Partial Data Handling

### The Issue (Fixed)
Previously, the detection comparison graph would render as empty when only one detection method triggered:
- If `causal_detection_time` was set but `threshold_detection_time` was `None`, the entire left panel would be blank
- Condition was too strict: `if causal_time is not None AND threshold_time is not None`

### The Solution
Updated `visualizer.py` to intelligently render available data:
- Changed condition to: `if causal_time is not None OR threshold_time is not None`
- Dynamically builds bar chart with only available methods
- Lead time annotation shown only when both times exist
- Right panel (analysis summary) always displays, explaining why methods did/didn't trigger

### Example Behavior
**Mission Analysis Output** (typical case):
- Causal Inference detects at T+36s [OK] (shown as green bar)
- Threshold-Based never triggers [X] (not shown in bar chart)
- Analysis summary explains: "[OK] Causal Inference: T+36.0s" + "[OK] Threshold-Based: Not triggered"
- No lead time arrow (only one method detected)

---

## Verification

All graphs are completely data-driven from analysis results:
- No hardcoded values or explanations
- All output calculated from actual data
- Framework separates analysis from presentation
- Graphs automatically regenerated on each run

---

## Advanced Usage

### Customize Analysis
Edit the analysis parameters in:
- `gsat6a/forensics.py` - Detection thresholds
- `gsat6a/mission_analysis.py` - Deviation detection levels
- `gsat6a/live_simulation.py` - Failure injection timing

### Extend Framework
Add new analysis types by:
1. Creating analysis event records
2. Feeding them to `timeline.add_event()` 
3. Framework handles visualization automatically

---

Generated: 2025-01-26
Data Source: GSAT-6A failure telemetry (March 26, 2018)
Analysis: Automated Aethelix causal inference framework (refactored to framework-based output)

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
