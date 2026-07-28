# Quick Start Guide (5 Minutes)

Get Aethelix running in 5 minutes with the default example.

## Prerequisites

- Python 3.8+ installed
- 2 GB RAM available
- Terminal/command prompt

## Step-by-Step

### 1. Clone and Setup (2 minutes)

```bash
# Clone repository
git clone https://github.com/rudywasfound/aethelix.git
cd aethelix

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Framework (1 minute)

```bash
python main.py
```

You'll see:
```
======================================================================
Causal Inference for Satellite Fault Diagnosis
======================================================================

[1] Initializing simulators...
[2] Running nominal scenario...
[3] Running degraded scenario (multi-fault)...
[4] Analyzing deviations...
[5] Generating plots...
[6] Building causal graph...
[7] Ranking root causes...

ROOT CAUSE RANKING ANALYSIS
========================================================================

Most Likely Root Causes (by posterior probability):

1. solar_degradation         P= 46.3%  Confidence=93.3%
2. battery_aging             P= 18.8%  Confidence=71.7%
3. battery_thermal           P= 18.7%  Confidence=75.0%
...

Outputs saved to 'output/'
```

### 3. View Results (2 minutes)

```bash
# List generated files
ls -la output/

# Open plots
open output/comparison.png        # macOS
xdg-open output/comparison.png    # Linux
start output\comparison.png       # Windows
```

Expected files:
- `comparison.png` - Nominal vs degraded telemetry side-by-side
- `residuals.png` - Deviation analysis

## What Just Happened?

```
+----------------------------------------------+
| STEP 1: Simulate                             |
| • Generated 24 hours of nominal telemetry    |
| • Generated same with 3 simultaneous faults: |
|   - Solar panel degradation (t=6h)           |
|   - Battery aging (t=8h)                     |
|   - Battery cooling failure (t=8h)           |
+--------------+-------------------------------+
               |
               (down)
+----------------------------------------------+
| STEP 2: Analyze                              |
| • Detected anomalies (>15% deviation)        |
| • Quantified severity scores                 |
| • Identified onset times                     |
+--------------+-------------------------------+
               |
               (down)
+----------------------------------------------+
| STEP 3: Reason                               |
| • Built causal graph (23 nodes, 29 edges)    |
| • Traced paths from causes -> effects         |
| • Scored hypotheses by consistency           |
| • Ranked by posterior probability            |
+--------------+-------------------------------+
               |
               (down)
+----------------------------------------------+
| STEP 4: Report                               |
| • Output ranked root causes                  |
| • Confidence and evidence for each           |
| • Visualization of telemetry changes         |
+----------------------------------------------+
```

## Real Output Examples

### Example 1: GSAT6A Telemetry Deviations

Below is actual output from the Aethelix framework analyzing a GSAT6A satellite scenario with solar array degradation:

![GSAT6A Telemetry Deviations](gsat6a_telemetry_deviations.png)

This graph shows:

**LEFT**: Nominal (healthy) operation
**RIGHT**: Degraded operation with solar failure

**Lines**:
Green dashed = Expected healthy behavior
Red solid = What actually happened

Key observations from the graphs:

Solar Array Power drops from 500W to 350W
Battery State of Charge falls from 100% to 20%
Power Bus Voltage drops from 12V to 10V (critical threshold)
Thermal Status: Battery temperature rises to 44C

### Example 2: GSAT6A Detection Comparison

This shows how Aethelix's causal inference compares to traditional threshold-based detection:

![GSAT6A Detection Comparison](gsat6a_detection_comparison.png)

The analysis includes:

Mission timeline and failure cascade
Causal inference results (probability = 46.3%)
Detection methodology using graph traversal
Comparison with traditional threshold-based detection

Result: Solar array deployment failure correctly identified as root cause in 36-90 seconds. Traditional threshold systems take 2-5 minutes.

### Example 3: Residual Analysis

The framework produces deviation plots showing magnitude and timing of anomalies.

When solar panels degrade:

Solar input drops 60W from nominal
Battery charge deviates -23%
Bus voltage deviates -1.5V
All deviations start within minutes of the fault

## Understanding the Output

### Console Report

```
ROOT CAUSE RANKING ANALYSIS
========================================================================

Most Likely Root Causes (by posterior probability):

1. solar_degradation         P= 46.3%  Confidence=93.3%
   Evidence: solar_input deviation, battery_charge deviation
   Mechanism: Reduced solar input propagates through power subsystem...

2. battery_aging             P= 18.8%  Confidence=71.7%
   Evidence: battery_charge deviation, battery_voltage deviation
   Mechanism: Aged battery cells have reduced capacity...
```

What this means:
- **P = 46.3%**: Probability that solar_degradation caused the observed anomalies
- **Confidence = 93.3%**: How certain we are (based on evidence quality)
- **Mechanism**: Plain-English explanation of how the cause produces effects
- **Evidence**: Which sensor readings support this hypothesis

### Telemetry Plot (comparison.png)

Two panels:
- **Left**: Nominal operation (healthy satellite)
- **Right**: Degraded operation (with faults)

Red shaded area: Period when faults were active

You'll see:
- Solar input drops at 6 hours
- Battery charge drops at 8 hours  
- Battery temperature rises at 8 hours

### Residual Plot (residuals.png)

Shows deviation from nominal:
- Positive = higher than normal
- Negative = lower than normal
- Larger = more significant

## Key Observations

From the default run, you should observe:

1. **Multi-fault diagnosis works**: Even though 3 faults are active, the framework correctly identifies solar degradation as most likely (46.3%).

2. **Secondary effects are explained**: Battery temperature rise is correctly attributed to solar degradation (not a direct fault), via reduced charging cycles.

3. **Confidence scores vary**: Hypotheses with more evidence have higher confidence.

4. **Mechanisms are explicit**: Each cause includes an English explanation, not just probability.

## Next Steps

Now that you have it running:

### Option 1: Understand How It Works
Read [Architecture Guide](07_ARCHITECTURE.md) to understand the causal reasoning process.

### Option 2: Customize Parameters
Read [Configuration Guide](05_CONFIGURATION.md) to:
- Inject different faults
- Change simulation duration
- Adjust detection thresholds
- Tune scoring weights

### Option 3: Use as Python Library
Read [Python Library Usage](11_PYTHON_LIBRARY.md) to integrate into your own code:

```python
from simulator.power import PowerSimulator
from causal_graph.root_cause_ranking import RootCauseRanker
from causal_graph.graph_definition import CausalGraph

# Your own scenario
power_sim = PowerSimulator(duration_hours=12)
nominal = power_sim.run_nominal()
degraded = power_sim.run_degraded(
    solar_degradation_hour=2.0,
    solar_factor=0.6
)

# Infer root causes
graph = CausalGraph()
ranker = RootCauseRanker(graph)
hypotheses = ranker.analyze(nominal, degraded)

# Get results
for h in hypotheses:
    print(f"{h.name}: {h.probability:.1%}")
```

### Option 4: Run Tests
```bash
python -m unittest discover tests/ -v
```

This verifies all components work correctly.

### Option 5: Explore Examples
Check example scripts:
- `gsat6a/live_simulation.py` - Real satellite scenario
- `operational/telemetry_simulator.py` - Custom scenarios
- `tests/test_*.py` - Unit test examples

## Common Customizations

### Run for 12 Hours Instead of 24
Edit `main.py`, line 102:
```python
power_sim = PowerSimulator(duration_hours=12, sampling_rate_hz=0.1)
thermal_sim = ThermalSimulator(duration_hours=12, sampling_rate_hz=0.1)
```

### Inject Different Faults
Edit `main.py`, lines 124-135:
```python
power_deg = power_sim.run_degraded(
    solar_degradation_hour=2.0,      # Start earlier
    solar_factor=0.5,                 # Worse degradation
    battery_degradation_hour=4.0,     # Start earlier
    battery_factor=0.6,               # Worse aging
)
```

### Change Detection Threshold
Edit `main.py`, line 144:
```python
analyzer = ResidualAnalyzer(deviation_threshold=0.10)  # Stricter: 10%
```

## Troubleshooting

### Error: "No module named 'simulator'"
```bash
# Make sure you're in the right directory
pwd  # should show .../aethelix
ls   # should see simulator/, causal_graph/, etc.

# Make sure virtual environment is activated
which python  # should show .../aethelix/.venv/bin/python
```

### Plots not displaying
```bash
# Plots are saved to output/ directory, not displayed
ls output/comparison.png
```

### Memory usage is high
- Reduce simulation duration from 24 to 12 hours
- Increase sampling_rate_hz from 0.1 to 1 (fewer data points)

### Installation issues
See [Installation Troubleshooting](02_INSTALLATION.md#troubleshooting-installation)

## What's Next?

- **Learn more**: [Running the Framework](04_RUNNING_FRAMEWORK.md)
- **Understand design**: [Architecture](07_ARCHITECTURE.md)
- **Use as library**: [Python Library](11_PYTHON_LIBRARY.md)
- **Deploy**: [Deployment Guide](16_DEPLOYMENT.md)

---

**Continue to:** [Running the Framework ->](04_RUNNING_FRAMEWORK.md)

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
