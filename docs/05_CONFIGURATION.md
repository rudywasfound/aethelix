# Configuration & Parameters

Complete reference for tuning Aethelix's behavior.

## Configuration Hierarchy

```
Default values (in source code)
         (down)
Configuration file (if present)
         (down)
Runtime parameters (in function calls)
         (down)
Environment variables (optional)
```

Each level overrides the one above it.

## Simulation Configuration

### Power Simulator

```python
from simulator.power import PowerSimulator

sim = PowerSimulator(
    duration_hours=24,           # Simulation length
    sampling_rate_hz=0.1,        # Telemetry frequency
    initial_soc=95.0,            # Initial battery state of charge (%)
    nominal_solar_input=600.0,   # Nominal solar power (W)
    nominal_bus_voltage=28.0,    # Nominal bus voltage (V)
)

# Nominal scenario (healthy satellite)
nominal = sim.run_nominal(
    eclipse_duration_hours=0.5,  # Orbital eclipse duration
    eclipse_depth=1.0,           # Eclipse depth (1.0 = total darkness)
)

# Degraded scenario (with faults)
degraded = sim.run_degraded(
    # Solar panel fault
    solar_degradation_hour=6.0,  # Start time (hours)
    solar_factor=0.7,            # Remaining efficiency (0.7 = 30% loss)
    
    # Battery aging fault
    battery_degradation_hour=8.0,
    battery_factor=0.8,          # Remaining efficiency (0.8 = 20% loss)
)
```

#### Parameter Details

| Parameter | Type | Default | Range | Effect |
|-----------|------|---------|-------|--------|
| `duration_hours` | float | 24 | 0.1-720 | Total simulation time |
| `sampling_rate_hz` | float | 0.1 | 0.01-10 | Telemetry sample frequency |
| `initial_soc` | float | 95.0 | 0-100 | Starting battery charge |
| `nominal_solar_input` | float | 600.0 | 100-1000 | Healthy solar power |
| `nominal_bus_voltage` | float | 28.0 | 20-36 | Nominal voltage |
| `eclipse_duration_hours` | float | 0.5 | 0-12 | Darkness time per orbit |
| `eclipse_depth` | float | 1.0 | 0-1.0 | Darkness intensity |
| `solar_degradation_hour` | float | 6.0 | 0-duration | Fault start time |
| `solar_factor` | float | 0.7 | 0-1.0 | Efficiency multiplier |
| `battery_degradation_hour` | float | 8.0 | 0-duration | Fault start time |
| `battery_factor` | float | 0.8 | 0-1.0 | Efficiency multiplier |

### Thermal Simulator

```python
from simulator.thermal import ThermalSimulator

sim = ThermalSimulator(
    duration_hours=24,
    sampling_rate_hz=0.1,
    ambient_temp=3.0,            # Space temperature (K)
    battery_capacity=100.0,      # Battery Wh
)

# Nominal thermal scenario
nominal = sim.run_nominal(
    solar_input,     # From power simulator
    battery_charge,  # From power simulator
    battery_voltage, # From power simulator
)

# Degraded with cooling failure
degraded = sim.run_degraded(
    solar_input,
    battery_charge,
    battery_voltage,
    battery_cooling_hour=8.0,    # Start time
    battery_cooling_factor=0.5,  # Effectiveness (0.5 = 50% loss)
)
```

#### Parameter Details

| Parameter | Type | Default | Range | Effect |
|-----------|------|---------|-------|--------|
| `ambient_temp` | float | 3.0 | 1-300 | Absolute space temperature |
| `battery_capacity` | float | 100.0 | 10-1000 | Watt-hours |
| `battery_cooling_hour` | float | 8.0 | 0-duration | Cooling fault start |
| `battery_cooling_factor` | float | 0.5 | 0-1.0 | Cooling effectiveness |

## Analysis Configuration

### Residual Analyzer

```python
from analysis.residual_analyzer import ResidualAnalyzer

analyzer = ResidualAnalyzer(
    deviation_threshold=0.15,    # Anomaly threshold (15% = 15% deviation)
    smoothing_window=10,         # Moving average window size
    severity_scaling=1.0,        # Severity score multiplier
)

stats = analyzer.analyze(nominal, degraded)
```

#### Parameter Details

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `deviation_threshold` | float (0-1) | 0.15 | What's considered an anomaly |
| `smoothing_window` | int | 10 | Samples for moving average |
| `severity_scaling` | float | 1.0 | Multiply all severity scores |

**Deviation Threshold Guidance:**
- `0.05` (5%): Very sensitive, many false positives
- `0.10` (10%): Sensitive, good for real-time monitoring
- `0.15` (15%): Standard, balances sensitivity and specificity
- `0.20` (20%): Conservative, misses subtle anomalies
- `0.30` (30%): Very conservative, only major faults

## Causal Graph Configuration

### Graph Definition

The causal graph is configured in `causal_graph/graph_definition.py`:

```python
from causal_graph.graph_definition import CausalGraph

graph = CausalGraph()

# Inspect configuration
print(f"Root causes: {[n.name for n in graph.root_causes]}")
print(f"Intermediates: {[n.name for n in graph.intermediates]}")
print(f"Observables: {[n.name for n in graph.observables]}")
```

### Node Types

1. **Root Causes** (7 nodes)
   - Solar degradation
   - Battery aging
   - Battery thermal stress
   - Sensor bias
   - Panel insulation failure
   - Heatsink failure
   - Radiator degradation

2. **Intermediates** (8 nodes)
   - Solar input
   - Battery state
   - Battery temperature
   - Bus regulation
   - Battery efficiency
   - Thermal stress
   - Payload state
   - Bus current

3. **Observables** (8 nodes)
   - Measured solar input
   - Measured battery voltage
   - Measured battery charge
   - Measured bus voltage
   - Measured battery temperature
   - Measured solar panel temperature
   - Measured payload temperature
   - Measured bus current

### Modifying the Graph

To extend or customize the causal graph:

```python
from causal_graph.graph_definition import CausalGraph, Node, Edge

# Create custom graph
class CustomGraph(CausalGraph):
    def __init__(self):
        super().__init__()
        
        # Add new node
        new_cause = Node(
            name="radiator_degradation_new",
            node_type="root_cause"
        )
        self.root_causes.append(new_cause)
        self.nodes.append(new_cause)
        
        # Add new edge
        new_edge = Edge(
            source="radiator_degradation_new",
            target="battery_temp",
            weight=0.7,
            mechanism="Poor radiator efficiency reduces heat dissipation"
        )
        self.edges.append(new_edge)

# Use custom graph
graph = CustomGraph()
ranker = RootCauseRanker(graph)
```

See [Causal Graph Design](08_CAUSAL_GRAPH.md) for detailed structure.

## Inference Configuration

### Root Cause Ranker

```python
from causal_graph.root_cause_ranking import RootCauseRanker

ranker = RootCauseRanker(
    graph,
    prior_probabilities=None,  # Uniform by default
    consistency_weight=1.0,    # How much consistency affects score
    severity_weight=1.0,       # How much severity affects score
)

hypotheses = ranker.analyze(
    nominal,
    degraded,
    deviation_threshold=0.15,
    confidence_threshold=0.5,  # Minimum confidence to report
)
```

#### Prior Probabilities

Set custom prior probabilities (before evidence):

```python
priors = {
    "solar_degradation": 0.3,      # 30% prior (more likely a priori)
    "battery_aging": 0.2,          # 20%
    "battery_thermal": 0.1,        # 10%
    # ... others
}

ranker = RootCauseRanker(graph, prior_probabilities=priors)
```

Use cases:
- Historical data shows solar faults are more common: increase solar prior
- In winter, thermal faults are rare: decrease thermal prior
- New satellite with known issues: adjust based on fleet data

#### Scoring Weights

Customize how scores are computed:

```python
ranker = RootCauseRanker(
    graph,
    consistency_weight=2.0,    # Consistency is more important
    severity_weight=0.5,       # Severity is less important
)
```

- High `consistency_weight`: Favor hypotheses consistent with graph
- High `severity_weight`: Favor hypotheses with strong evidence

## Visualization Configuration

### Telemetry Plotter

```python
from visualization.plotter import TelemetryPlotter

plotter = TelemetryPlotter(
    figsize=(14, 10),           # Figure size in inches
    dpi=150,                    # Resolution
    style="default",            # Matplotlib style
)

plotter.plot_comparison(
    nominal,
    degraded,
    degradation_hours=(6, 24),  # Highlight period
    save_path="output/plot.png",
)

plotter.plot_residuals(
    nominal,
    degraded,
    save_path="output/residuals.png",
)
```

#### Parameter Details

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `figsize` | tuple | (14, 10) | Width  x  height in inches |
| `dpi` | int | 150 | Resolution (dots per inch) |
| `style` | str | "default" | Matplotlib style |
| `degradation_hours` | tuple | (6, 24) | Highlight period |

## Configuration File (Optional)

Create `aethelix_config.yaml`:

```yaml
# Simulation
simulation:
  duration_hours: 24
  sampling_rate_hz: 0.1
  initial_soc: 95.0

# Power faults
power_faults:
  solar_degradation_hour: 6.0
  solar_factor: 0.7
  battery_degradation_hour: 8.0
  battery_factor: 0.8

# Thermal faults
thermal_faults:
  battery_cooling_hour: 8.0
  battery_cooling_factor: 0.5

# Analysis
analysis:
  deviation_threshold: 0.15
  smoothing_window: 10

# Visualization
visualization:
  figsize: [14, 10]
  dpi: 150
  style: default

# Inference
inference:
  consistency_weight: 1.0
  severity_weight: 1.0
```

Load configuration:

```python
import yaml

with open("aethelix_config.yaml") as f:
    config = yaml.safe_load(f)

power_sim = PowerSimulator(**config["simulation"])
power_deg = power_sim.run_degraded(**config["power_faults"])
analyzer = ResidualAnalyzer(**config["analysis"])
```

## Environment Variables

Set options via environment variables:

```bash
export PRAVAHA_OUTPUT_DIR="./results"
export PRAVAHA_DEVIATION_THRESHOLD="0.10"
export PRAVAHA_SAMPLING_RATE_HZ="1.0"
```

Access in code:

```python
import os

output_dir = os.getenv("PRAVAHA_OUTPUT_DIR", "output")
threshold = float(os.getenv("PRAVAHA_DEVIATION_THRESHOLD", "0.15"))
sampling_rate = float(os.getenv("PRAVAHA_SAMPLING_RATE_HZ", "0.1"))
```

## Parameter Recommendations

### For Real-Time Monitoring

```python
PowerSimulator(
    duration_hours=0.5,      # Last 30 minutes
    sampling_rate_hz=1.0,    # 1 Hz (real-time)
)

analyzer = ResidualAnalyzer(deviation_threshold=0.10)  # Sensitive
```

### For Forensic Analysis

```python
PowerSimulator(
    duration_hours=720,      # Last 30 days
    sampling_rate_hz=0.01,   # 1 sample/100 seconds (low data volume)
)

analyzer = ResidualAnalyzer(deviation_threshold=0.20)  # Conservative
```

### For Research / Benchmarking

```python
PowerSimulator(
    duration_hours=168,      # One week
    sampling_rate_hz=0.1,    # Standard sampling
)

analyzer = ResidualAnalyzer(deviation_threshold=0.15)  # Standard
```

### For Development / Testing

```python
PowerSimulator(
    duration_hours=6,        # Short, fast
    sampling_rate_hz=0.1,    # Standard sampling
)

analyzer = ResidualAnalyzer(deviation_threshold=0.15)
```

## Troubleshooting Configuration

### Symptom: All hypotheses have low probability (<20%)

**Cause**: Faults too subtle or deviation threshold too high

**Solution**:
```python
# Reduce threshold
analyzer = ResidualAnalyzer(deviation_threshold=0.10)

# Or increase fault severity
power_deg = power_sim.run_degraded(solar_factor=0.5)  # Worse degradation
```

### Symptom: False positives (wrong cause ranked high)

**Cause**: Deviation threshold too low or inconsistent priors

**Solution**:
```python
# Increase threshold
analyzer = ResidualAnalyzer(deviation_threshold=0.20)

# Or adjust priors based on known failure modes
priors = {
    "solar_degradation": 0.1,  # Less likely for this satellite
    "battery_aging": 0.5,       # More likely
}
ranker = RootCauseRanker(graph, prior_probabilities=priors)
```

### Symptom: Inference runs slowly

**Cause**: Large simulation duration or high sampling rate

**Solution**:
```python
# Reduce duration
PowerSimulator(duration_hours=12)  # Instead of 24

# Or reduce sampling rate
PowerSimulator(sampling_rate_hz=0.5)  # Instead of 1.0
```

## Next Steps

- **Run with custom config**: [Running the Framework](04_RUNNING_FRAMEWORK.md)
- **Understand inference**: [Inference Algorithm](09_INFERENCE_ALGORITHM.md)
- **Extend causal graph**: [Causal Graph Design](08_CAUSAL_GRAPH.md)
- **Optimize performance**: [Performance Tuning](15_PERFORMANCE.md)

---

**Continue to:** [Output Interpretation ->](06_OUTPUT_INTERPRETATION.md)

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
