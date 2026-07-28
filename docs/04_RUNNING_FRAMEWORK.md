# Running the Framework

Complete guide to executing Aethelix workflows and understanding the results.

## Overview

The Aethelix workflow consists of 5 phases:

```
1. SIMULATION      -> Generate realistic telemetry
2. ANALYSIS        -> Quantify anomalies
3. VISUALIZATION   -> Plot deviations
4. GRAPH BUILDING  -> Construct causal model
5. INFERENCE       -> Rank root causes
```

## Default Workflow

### Quick Run
```bash
python main.py
```

Generates:
- `output/comparison.png` - Telemetry comparison
- `output/residuals.png` - Deviation analysis
- Console report - Root cause ranking

### What It Does

**Phase 1: Simulation (5 seconds)**
- Creates power simulator (24 hours, 0.1 Hz sampling)
- Creates thermal simulator
- Runs nominal scenario (healthy satellite)
- Runs degraded scenario (3 simultaneous faults):
  - Solar degradation at 6 hours (30% loss)
  - Battery aging at 8 hours (20% loss)
  - Battery cooling failure at 8 hours (50% loss)

**Phase 2: Analysis (1 second)**
- Compares degraded vs nominal
- Detects anomalies (>15% deviation)
- Quantifies severity
- Identifies onset times

**Phase 3: Visualization (2 seconds)**
- Plots all 8 telemetry channels
- Highlights fault period (6-24 hours)
- Generates residual deviation plot

**Phase 4: Graph Building (1 second)**
- Loads causal graph (23 nodes, 29 edges)
- Validates consistency
- Prepares for inference

**Phase 5: Inference (2 seconds)**
- Traces paths through causal graph
- Scores hypotheses by consistency
- Normalizes to probabilities
- Computes confidence scores

**Total time**: ~15 seconds

## Advanced Workflows

### Custom Fault Scenarios

Create a new Python file `custom_scenario.py`:

```python
from simulator.power import PowerSimulator
from simulator.thermal import ThermalSimulator
from visualization.plotter import TelemetryPlotter
from analysis.residual_analyzer import ResidualAnalyzer
from causal_graph.root_cause_ranking import RootCauseRanker
from causal_graph.graph_definition import CausalGraph
import os

# Setup
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Create simulators
power_sim = PowerSimulator(duration_hours=12, sampling_rate_hz=0.5)
thermal_sim = ThermalSimulator(duration_hours=12, sampling_rate_hz=0.5)

# Nominal
power_nom = power_sim.run_nominal()
thermal_nom = thermal_sim.run_nominal(
    power_nom.solar_input,
    power_nom.battery_charge,
    power_nom.battery_voltage,
)

# Degraded: Only solar degradation
power_deg = power_sim.run_degraded(
    solar_degradation_hour=3.0,
    solar_factor=0.5,           # 50% efficiency (50% loss)
    battery_degradation_hour=999,  # Disable (set to future time)
    battery_factor=1.0,
)
thermal_deg = thermal_sim.run_degraded(
    power_deg.solar_input,
    power_deg.battery_charge,
    power_deg.battery_voltage,
    battery_cooling_hour=999,
    battery_cooling_factor=1.0,
)

# Analyze
analyzer = ResidualAnalyzer(deviation_threshold=0.15)
nominal = CombinedTelemetry(power_nom, thermal_nom)
degraded = CombinedTelemetry(power_deg, thermal_deg)
stats = analyzer.analyze(nominal, degraded)
analyzer.print_report(stats)

# Visualize
plotter = TelemetryPlotter()
plotter.plot_comparison(nominal, degraded, save_path=f"{output_dir}/custom.png")

# Infer
graph = CausalGraph()
ranker = RootCauseRanker(graph)
hypotheses = ranker.analyze(nominal, degraded, deviation_threshold=0.15)
ranker.print_report(hypotheses)

class CombinedTelemetry:
    def __init__(self, power_telem, thermal_telem):
        self.time = power_telem.time
        self.solar_input = power_telem.solar_input
        self.battery_voltage = power_telem.battery_voltage
        self.battery_charge = power_telem.battery_charge
        self.bus_voltage = power_telem.bus_voltage
        self.battery_temp = thermal_telem.battery_temp
        self.solar_panel_temp = thermal_telem.solar_panel_temp
        self.payload_temp = thermal_telem.payload_temp
        self.bus_current = thermal_telem.bus_current
        self.timestamp = power_telem.timestamp
```

Run it:
```bash
python custom_scenario.py
```

### Batch Processing

Process multiple scenarios:

```python
# batch_analysis.py
from simulator.power import PowerSimulator
from causal_graph.root_cause_ranking import RootCauseRanker
from causal_graph.graph_definition import CausalGraph
import json

scenarios = [
    {"name": "solar_only", "solar_factor": 0.5},
    {"name": "battery_only", "battery_factor": 0.7},
    {"name": "thermal_only", "cooling_factor": 0.3},
    {"name": "multi_fault", "solar_factor": 0.7, "battery_factor": 0.8, "cooling_factor": 0.5},
]

results = []

for scenario in scenarios:
    power_sim = PowerSimulator(duration_hours=24)
    power_nom = power_sim.run_nominal()
    power_deg = power_sim.run_degraded(
        solar_factor=scenario.get("solar_factor", 1.0),
        battery_factor=scenario.get("battery_factor", 1.0),
    )
    
    # ... thermal sim, analysis, etc.
    
    # Infer
    graph = CausalGraph()
    ranker = RootCauseRanker(graph)
    hypotheses = ranker.analyze(nominal, degraded)
    
    results.append({
        "scenario": scenario["name"],
        "top_cause": hypotheses[0].name,
        "probability": hypotheses[0].probability,
        "confidence": hypotheses[0].confidence,
    })

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Integration with Rust Core

For high-frequency data processing:

```python
import aethelix_core  # Rust bindings
from simulator.power import PowerSimulator

# Generate telemetry
power_sim = PowerSimulator(duration_hours=1, sampling_rate_hz=100)  # 100 Hz
power_data = power_sim.run_nominal()

# Use Rust Kalman filter
kf = aethelix_core.KalmanFilter(dt=0.01)  # 10 ms timestep

estimates = []
for i in range(len(power_data.time)):
    measurement = aethelix_core.Measurement()
    measurement.battery_voltage = float(power_data.battery_voltage[i])
    measurement.battery_charge = float(power_data.battery_charge[i])
    measurement.battery_temp = 35.0
    # ... set other fields
    
    kf.update(measurement)
    estimate_json = kf.get_estimate()
    estimates.append(estimate_json)
```

See [Rust Integration](12_RUST_INTEGRATION.md) for details.

## Modular Usage

Use individual components:

### Just Simulation
```python
from simulator.power import PowerSimulator

sim = PowerSimulator(duration_hours=24)
nominal = sim.run_nominal()
degraded = sim.run_degraded(solar_factor=0.7)

# Access data
print(f"Nominal solar input: {nominal.solar_input}")
print(f"Degraded solar input: {degraded.solar_input}")
```

### Just Analysis
```python
from analysis.residual_analyzer import ResidualAnalyzer

analyzer = ResidualAnalyzer(deviation_threshold=0.15)
stats = analyzer.analyze(nominal, degraded)

# Access results
print(f"Severity: {stats['overall_severity']:.1%}")
print(f"Most affected variable: {stats['max_deviation_variable']}")
```

### Just Visualization
```python
from visualization.plotter import TelemetryPlotter

plotter = TelemetryPlotter()
plotter.plot_comparison(nominal, degraded, save_path="plot.png")
plotter.plot_residuals(nominal, degraded, save_path="residuals.png")
```

### Just Inference
```python
from causal_graph.root_cause_ranking import RootCauseRanker
from causal_graph.graph_definition import CausalGraph

graph = CausalGraph()
ranker = RootCauseRanker(graph)
hypotheses = ranker.analyze(nominal, degraded)

for h in hypotheses:
    print(f"{h.name}: {h.probability:.1%} (confidence: {h.confidence:.1%})")
```

## Configuration Parameters

### Simulation Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `duration_hours` | 24 | Simulation length in hours |
| `sampling_rate_hz` | 0.1 | Telemetry frequency (0.1 Hz = 1 sample/10 sec) |
| `solar_degradation_hour` | 6.0 | When solar fault begins |
| `solar_factor` | 0.7 | Solar efficiency (0.7 = 30% loss) |
| `battery_degradation_hour` | 8.0 | When battery aging begins |
| `battery_factor` | 0.8 | Battery efficiency (0.8 = 20% loss) |
| `battery_cooling_hour` | 8.0 | When cooling failure begins |
| `battery_cooling_factor` | 0.5 | Cooling effectiveness (0.5 = 50% loss) |

### Analysis Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `deviation_threshold` | 0.15 | Anomaly threshold (15% deviation) |

Lower threshold = detect smaller anomalies (more false positives)
Higher threshold = only major anomalies (might miss subtle faults)

### Inference Parameters

Built into CausalGraph - see [Configuration Guide](05_CONFIGURATION.md)

## Output Structure

```
output/
+-- comparison.png          # Nominal vs degraded telemetry
+-- residuals.png          # Deviation analysis plot
+-- (console reports)      # Printed to stdout
```

### Extending Output

Generate additional plots:

```python
from visualization.plotter import TelemetryPlotter
import matplotlib.pyplot as plt

plotter = TelemetryPlotter()

# Custom plot: just solar variables
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(nominal.time, nominal.solar_input, label="Nominal")
axes[0].plot(degraded.time, degraded.solar_input, label="Degraded")
axes[0].set_ylabel("Solar Input (W)")
axes[0].legend()

axes[1].plot(nominal.time, nominal.battery_charge, label="Nominal")
axes[1].plot(degraded.time, degraded.battery_charge, label="Degraded")
axes[1].set_ylabel("Battery Charge (%)")
axes[1].set_xlabel("Time (hours)")
axes[1].legend()

plt.tight_layout()
plt.savefig("output/custom_solar.png", dpi=150)
plt.close()
```

## Performance Considerations

### Timing Breakdown

| Phase | Time (24h, 0.1 Hz) | Time (12h, 1 Hz) |
|-------|-------------------|-----------------|
| Simulation | 3-5 sec | 2-3 sec |
| Analysis | 0.5 sec | 0.5 sec |
| Visualization | 1-2 sec | 1-2 sec |
| Graph building | 0.5 sec | 0.5 sec |
| Inference | 1-2 sec | 1-2 sec |
| **Total** | **~10 sec** | **~7 sec** |

### Optimization Tips

1. **Reduce simulation duration**: 24 hours -> 12 hours (saves 2 sec)
2. **Increase sampling rate**: 0.1 Hz -> 1 Hz (less data, faster analysis)
3. **Use Rust core**: ~10x speedup for high-frequency data
4. **Parallel batch processing**: Process multiple scenarios simultaneously

See [Performance Tuning](15_PERFORMANCE.md) for detailed optimization.

## Debugging & Logging

### Enable Verbose Output

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all modules print detailed logs
```

### Print Intermediate Values

```python
from simulator.power import PowerSimulator

sim = PowerSimulator()
nominal = sim.run_nominal()

print(f"Nominal solar input shape: {nominal.solar_input.shape}")
print(f"Mean solar input: {nominal.solar_input.mean():.1f} W")
print(f"Min/Max: {nominal.solar_input.min():.1f}/{nominal.solar_input.max():.1f} W")
```

### Inspect Causal Graph

```python
from causal_graph.graph_definition import CausalGraph

graph = CausalGraph()
print(f"Nodes: {len(graph.nodes)}")
print(f"Edges: {len(graph.edges)}")

# Print node details
for node in graph.nodes:
    print(f"  {node.name} ({node.node_type})")

# Print edge details
for edge in graph.edges[:5]:  # First 5 edges
    print(f"  {edge.source} -> {edge.target} (weight: {edge.weight})")
```

## Common Workflows

### Workflow 1: Sensitivity Analysis

How does severity affect detection accuracy?

```python
from simulator.power import PowerSimulator
from causal_graph.root_cause_ranking import RootCauseRanker
from causal_graph.graph_definition import CausalGraph

results = {}
for solar_factor in [0.3, 0.5, 0.7, 0.9]:
    power_sim = PowerSimulator()
    power_nom = power_sim.run_nominal()
    power_deg = power_sim.run_degraded(solar_factor=solar_factor)
    # ... thermal sim, analysis, inference
    
    graph = CausalGraph()
    ranker = RootCauseRanker(graph)
    hypotheses = ranker.analyze(nominal, degraded)
    
    results[solar_factor] = {
        "top_cause": hypotheses[0].name,
        "probability": hypotheses[0].probability,
    }
```

### Workflow 2: Multi-fault Comparison

How do different fault combinations behave?

```python
scenarios = [
    {"solar": 0.7, "battery": 1.0, "cooling": 1.0},  # Solar only
    {"solar": 1.0, "battery": 0.8, "cooling": 1.0},  # Battery only
    {"solar": 1.0, "battery": 1.0, "cooling": 0.5},  # Cooling only
    {"solar": 0.7, "battery": 0.8, "cooling": 0.5},  # All three
]

for scenario in scenarios:
    # Run simulation with this scenario
    # Infer root causes
    # Record which cause was ranked highest
```

### Workflow 3: Streaming Data Processing

Process real-time telemetry:

```python
def process_telemetry_stream(telemetry_source, window_hours=1):
    """Process streaming telemetry in rolling windows"""
    
    from collections import deque
    from causal_graph.root_cause_ranking import RootCauseRanker
    from causal_graph.graph_definition import CausalGraph
    
    graph = CausalGraph()
    ranker = RootCauseRanker(graph)
    
    buffer = deque(maxlen=int(window_hours * 3600))  # 1 hour window
    
    for telemetry_point in telemetry_source:
        buffer.append(telemetry_point)
        
        # Every 10 minutes, analyze
        if len(buffer) % 600 == 0:
            # Convert buffer to nominal/degraded
            hypotheses = ranker.analyze(nominal_baseline, buffer_data)
            
            # Alert if high-probability fault detected
            for h in hypotheses:
                if h.probability > 0.5:
                    alert(f"High-confidence fault: {h.name}")
```

## Next Steps

- **Customize scenarios**: [Configuration Guide](05_CONFIGURATION.md)
- **Understand output**: [Output Interpretation](06_OUTPUT_INTERPRETATION.md)
- **Learn internals**: [Architecture Guide](07_ARCHITECTURE.md)
- **Optimize performance**: [Performance Tuning](15_PERFORMANCE.md)

---

**Continue to:** [Configuration Guide ->](05_CONFIGURATION.md)

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
