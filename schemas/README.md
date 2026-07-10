# Aethelix Causal DAG Schemas

This directory contains the schema definitions for the pluggable JSON/YAML DAG (Directed Acyclic Graph) configs.

## Files

- [aethelix_dag.schema.yaml](aethelix_dag.schema.yaml): Human-readable YAML schema defining validation rules, keys, node types, and edge definitions.
- [dag_schema.json](dag_schema.json): Strict JSON Schema (draft-07) for programmatic validation. Use with any JSON Schema validator (e.g. `jsonschema`, `ajv`, IDE plugins).

### Programmatic Validation Example

```bash
# Python (requires: pip install jsonschema pyyaml)
python -c "
import json, yaml
from jsonschema import validate
schema = json.load(open('schemas/dag_schema.json'))
with open('configs/sentinel1b.yaml') as f:
    doc = yaml.safe_load(f)
validate(doc, schema)
print('✓ Valid')
"
```

## Quick Start: Bring Your Own Satellite Config

To describe your own satellite subsystem failures, copy this minimal example into a `.yaml` file:

```yaml
aethelix_dag_version: "1.0"

satellite:
  name: "My Cubesat"
  agency: "My Organization"
  orbit: "LEO"

nodes:
  - id: battery_undercharge
    type: root_cause
    description: "Battery fails to receive charge"
    degradation_modes: ["aging", "cell_failure"]

  - id: payload_power
    type: intermediate
    description: "Power available to payload"

  - id: payload_temp_measured
    type: observable
    description: "Measured payload temperature"

edges:
  - source: battery_undercharge
    target: payload_power
    weight: 0.85
    mechanism: "Low battery charge starves payload of regulated power"

  - source: payload_power
    target: payload_temp_measured
    weight: 0.90
    mechanism: "Payload shut down due to low power causes temperature to drop"
```

Then load it directly in Python:

```python
from causal_graph import CausalGraph

graph = CausalGraph(dag_path="my_cubesat.yaml")
print(f"Loaded {len(graph.nodes)} nodes!")
```
