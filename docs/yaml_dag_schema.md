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
