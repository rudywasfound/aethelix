# Quick Start

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run Full Analysis

```bash
python main.py
```

Simulates 24h of nominal and degraded satellite telemetry, detects deviations, ranks root causes. Output: plots + console report.

## Test Suite

```bash
python -m unittest discover tests/ -v
```

Expected: 27 tests passing.

## Using the Causal Graph Framework

```python
from causal_graph import CausalGraph, DAGVisualizer

# Load graph and visualize
graph = CausalGraph()
viz = DAGVisualizer(graph)
viz.save("dag.png")  # Outputs PNG image

# Analyze structure
from causal_graph.dag_visualization import print_structure_by_type
print_structure_by_type(graph)
```

## Customize Fault Scenarios

Edit `main.py`:

```python
power_deg = power_sim.run_degraded(
    solar_degradation_hour=6.0,   # When fault starts
    solar_factor=0.7,              # Severity (0-1)
    battery_degradation_hour=8.0,
    battery_factor=0.8,
)
```

## Key Modules

| Module | Purpose |
|--------|---------|
| `causal_graph/graph_definition.py` | DAG: 23 nodes, 29 edges |
| `causal_graph/visualizer.py` | Render graphs to PNG/PDF/SVG |
| `causal_graph/root_cause_ranking.py` | Bayesian inference |
| `simulator/power.py` | Power subsystem simulator |
| `simulator/thermal.py` | Thermal subsystem simulator |
| `main.py` | Full workflow orchestration |

See `README.md` for detailed architecture.

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
