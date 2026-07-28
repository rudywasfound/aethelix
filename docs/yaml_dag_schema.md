# Aethelix YAML/JSON Causal DAG Schema

To support satellite-agnostic analyses, Aethelix lets developers and satellite operators define custom causal graphs in self-contained YAML or JSON configuration files. This means you can add support for new spacecraft platforms, sensors, and anomaly signatures without writing or modifying any Python code.

---

## File Structure

An Aethelix DAG configuration file has the following top-level keys:

```yaml
aethelix_dag_version: "1.0"

satellite:
  name: "Sentinel-1B"
  agency: "ESA"
  # Optional descriptive metadata ...

nodes:
  # List of vertices in the DAG ...

edges:
  # List of directed causal links between nodes ...
```

### 1. `aethelix_dag_version` (String, Optional)
Specifies the version of the Aethelix schema used. The default is `"1.0"`. Major-version mismatches will trigger a validation error.

### 2. `satellite` (Mapping, Optional)
A free-form documentation block for tracking provenance, spacecraft specifications, or mission dates.

### 3. `nodes` (List of Mappings, Required)
Defines the variables in your causal model. Every node represents a failure mode, a physical state, or a telemetry stream.

Each node mapping has:
- **`id`** (String, Required): A unique identifier (e.g. `solar_degradation`). It must not contain spaces.
- **`type`** (String, Required): One of three functional roles in the causal model:
  - `root_cause` (aliases: `fault`, `cause`): A primary failure source you want to diagnose.
  - `intermediate` (aliases: `inter`, `effect`): An unobservable propagation state (e.g. `solar_input` power).
  - `observable` (aliases: `obs`): A measured sensor parameter (e.g. `solar_input_measured` from telemetry).
- **`description`** (String, Optional): A human-readable label shown in operator reports and graph diagrams.
- **`degradation_modes`** (List of Strings, Optional): Lists specific ways the node can degrade (only applicable to `root_cause`).
- **`subsystem`** (String, Optional): Grouping label (e.g. `EPS`, `ADCS`) for visualizer categorization.

### 4. `edges` (List of Mappings, Required)
Defines directed causal edges from parents to children (`source -> target`), representing failure propagation.

Each edge mapping has:
- **`source`** (String, Required): The `id` of the causing node. Must be declared in the `nodes` block.
- **`target`** (String, Required): The `id` of the affected node. Must be declared in the `nodes` block.
- **`weight`** (Float, Optional): Causal strength from `0.0` to `1.0`. Defaults to `1.0`. Higher values denote stronger propagation probability.
- **`mechanism`** (String, Optional): Plain English description of the physical coupling. Shown to operators to justify diagnostic findings.

---

## Schema Validation Rules

When Aethelix loads a DAG configuration, the loader validates the following constraints:
1. **Valid JSON/YAML syntax** and structure.
2. **Uniqueness**: No duplicate node `id`s are permitted in the same file.
3. **No Dangling References**: Every edge `source` and `target` must map to a defined node `id`.
4. **Valid Weights**: Every edge `weight` must be a float between `0.0` and `1.0` inclusive.
5. **Acyclicity**: The graph must not contain any loops or cyclic paths (enforced using DFS topological sorting). If a loop is found, a `DAGLoadError` is raised.

---

## Usage Example

### Loading from Python

You can load a custom DAG file directly into a `CausalGraph`:

```python
from causal_graph import CausalGraph, RootCauseRanker

# Initialize a CausalGraph directly from your YAML config
graph = CausalGraph(dag_path="configs/sentinel1b.yaml")

# Run ranking analysis using this custom graph
ranker = RootCauseRanker(graph)
hypotheses = ranker.analyze(nominal_telemetry, degraded_telemetry)
ranker.print_report(hypotheses)
```

### Validating via CLI

Use the Aethelix CLI to check a schema config for correctness:

```bash
aethelix validate configs/sentinel1b.yaml
```
If the file is valid, the CLI outputs the total node and edge counts. If invalid, it displays the specific schema rule or cycle path violated.

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
