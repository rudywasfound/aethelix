"""
dag_loader.py — Pluggable YAML/JSON DAG loader for Aethelix.

Allows any satellite's causal fault graph to be described in a plain
YAML or JSON file without touching Python source code.  This is what
turns Aethelix from a GSAT-6A tool into a satellite-agnostic framework.

Usage (Python API)
------------------
    from causal_graph.dag_loader import load_dag, DAGLoadError

    graph = load_dag("configs/my_sat.yaml")         # returns CausalGraph
    graph = load_dag("configs/my_sat.json")         # JSON works too

Usage (CausalGraph kwarg)
-------------------------
    from causal_graph.graph_definition import CausalGraph

    graph = CausalGraph(dag_path="configs/sentinel1b.yaml")

Schema reference
----------------
    See schemas/aethelix_dag.schema.yaml or docs/yaml_dag_schema.md.

Validation rules enforced
-------------------------
1.  Required top-level keys: aethelix_dag_version, nodes, edges
2.  Each node must have: id (str), type (root_cause|intermediate|observable)
3.  Each edge must have: source (str), target (str), weight (0-1 float)
4.  All edge endpoints must reference a declared node id
5.  The graph must be acyclic (DFS cycle detection)
6.  Node ids must be unique within a file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Standard library — no extra deps needed beyond PyYAML (already in requirements)
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:                         # pragma: no cover
    _YAML_AVAILABLE = False

SCHEMA_VERSION = "1.0"

# Legal values for the 'type' field of a node
_LEGAL_NODE_TYPES = {"root_cause", "intermediate", "observable"}

# Mapping from YAML type string → NodeType enum value
_NODE_TYPE_ALIASES = {
    "root_cause":   "root_cause",
    "intermediate": "intermediate",
    "observable":   "observable",
    # Friendly abbreviations people might reach for
    "fault":        "root_cause",
    "cause":        "root_cause",
    "obs":          "observable",
    "inter":        "intermediate",
    "effect":       "intermediate",
}


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class DAGLoadError(ValueError):
    """
    Raised when a DAG config file fails validation.

    Attributes
    ----------
    path : str | None
        The file that caused the error (if known).
    hint : str
        A human-readable description of what is wrong and where.
    """

    def __init__(self, hint: str, path: Optional[Union[str, Path]] = None):
        self.path = str(path) if path else "<unknown>"
        self.hint = hint
        super().__init__(f"[DAGLoadError] {self.path}: {hint}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dag(path: Union[str, Path]) -> "CausalGraph":  # noqa: F821
    """
    Load a YAML or JSON satellite DAG config and return a populated CausalGraph.

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to a ``.yaml``, ``.yml``, or ``.json`` file.

    Returns
    -------
    CausalGraph
        A fully populated graph ready for root-cause inference.

    Raises
    ------
    DAGLoadError
        If the file is missing, unparseable, or fails schema validation.
    FileNotFoundError
        If the file does not exist (re-raised with context).
    """
    # Deferred import to avoid circular imports (dag_loader ↔ graph_definition)
    from causal_graph.graph_definition import CausalGraph, NodeType

    path = Path(path)

    raw = _read_file(path)
    validated = validate_schema(raw, path)

    # Build a bare CausalGraph (no hardcoded subsystems) then populate it
    graph = CausalGraph.__new__(CausalGraph)        # skip __init__
    _init_empty_graph(graph)

    _populate_nodes(graph, validated["nodes"], path)
    _populate_edges(graph, validated["edges"], path)

    # Attach metadata as a plain dict for introspection
    graph.dag_meta = {
        "aethelix_dag_version": validated.get("aethelix_dag_version", SCHEMA_VERSION),
        "satellite": validated.get("satellite", {}),
        "source_file": str(path.resolve()),
        "expected_evidence": validated.get("expected_evidence", {}),
    }

    return graph


def validate_schema(raw: Dict[str, Any], path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """
    Validate a parsed YAML/JSON dict against the Aethelix DAG schema.

    Can be called independently of ``load_dag`` — useful for CI lint checks.

    Parameters
    ----------
    raw : dict
        Parsed content of a satellite config file.
    path : str | Path | None
        File path shown in error messages (optional, for display only).

    Returns
    -------
    dict
        The validated (and lightly normalised) config dict.

    Raises
    ------
    DAGLoadError
        On any schema violation.
    """
    p = path  # for error messages

    # 1 — Must be a mapping
    if not isinstance(raw, dict):
        raise DAGLoadError("File must be a YAML/JSON object (mapping), not a list or scalar.", p)

    # 2 — Version check (warn but allow minor mismatches; hard-error on major)
    version = str(raw.get("aethelix_dag_version", SCHEMA_VERSION))
    major = version.split(".")[0]
    if major != SCHEMA_VERSION.split(".")[0]:
        raise DAGLoadError(
            f"Unsupported schema version '{version}'. "
            f"This loader supports v{SCHEMA_VERSION}. "
            "See docs/yaml_dag_schema.md for migration notes.",
            p,
        )

    # 3 — Required top-level keys
    for key in ("nodes", "edges"):
        if key not in raw:
            raise DAGLoadError(f"Required top-level key '{key}' is missing.", p)

    # 4 — Nodes list
    nodes_raw = raw["nodes"]
    if not isinstance(nodes_raw, list) or len(nodes_raw) == 0:
        raise DAGLoadError("'nodes' must be a non-empty list.", p)

    seen_ids: set[str] = set()
    nodes_validated: List[Dict] = []

    for i, node in enumerate(nodes_raw):
        if not isinstance(node, dict):
            raise DAGLoadError(f"nodes[{i}] must be a mapping, got {type(node).__name__}.", p)

        # Required fields
        if "id" not in node:
            raise DAGLoadError(f"nodes[{i}] is missing required field 'id'.", p)
        if "type" not in node:
            raise DAGLoadError(f"nodes[{i}] (id='{node.get('id')}') is missing required field 'type'.", p)

        node_id = str(node["id"])
        raw_type = str(node["type"]).lower()

        # Uniqueness
        if node_id in seen_ids:
            raise DAGLoadError(f"Duplicate node id '{node_id}' at nodes[{i}].", p)
        seen_ids.add(node_id)

        # Type normalisation
        if raw_type not in _NODE_TYPE_ALIASES:
            raise DAGLoadError(
                f"nodes[{i}] (id='{node_id}'): unknown type '{raw_type}'. "
                f"Valid types: {sorted(_LEGAL_NODE_TYPES)}.",
                p,
            )
        canonical_type = _NODE_TYPE_ALIASES[raw_type]

        validated_node = {
            "id": node_id,
            "type": canonical_type,
            "description": str(node.get("description", "")),
            "degradation_modes": [str(m) for m in node.get("degradation_modes", [])],
            "subsystem": str(node.get("subsystem", "")),
        }

        # Optional structural_equation parsing
        if "structural_equation" in node:
            seq = node["structural_equation"]
            if not isinstance(seq, dict):
                raise DAGLoadError(f"nodes[{i}].structural_equation must be a mapping.", p)
            validated_node["structural_equation"] = {
                "coefficients": {str(k): float(v) for k, v in seq.get("coefficients", {}).items()},
                "noise_std": float(seq.get("noise_std", 0.1)),
            }

        nodes_validated.append(validated_node)

    # 5 — Edges list
    edges_raw = raw["edges"]
    if not isinstance(edges_raw, list):
        raise DAGLoadError("'edges' must be a list.", p)

    edges_validated: List[Dict] = []
    for i, edge in enumerate(edges_raw):
        if not isinstance(edge, dict):
            raise DAGLoadError(f"edges[{i}] must be a mapping.", p)

        for field in ("source", "target"):
            if field not in edge:
                raise DAGLoadError(f"edges[{i}] is missing required field '{field}'.", p)

        src = str(edge["source"])
        dst = str(edge["target"])

        # Dangling reference checks
        if src not in seen_ids:
            raise DAGLoadError(
                f"edges[{i}]: source '{src}' is not defined in nodes. "
                "Add it to the 'nodes' list first.",
                p,
            )
        if dst not in seen_ids:
            raise DAGLoadError(
                f"edges[{i}]: target '{dst}' is not defined in nodes. "
                "Add it to the 'nodes' list first.",
                p,
            )

        # Weight validation
        weight_raw = edge.get("weight", 1.0)
        try:
            weight = float(weight_raw)
        except (TypeError, ValueError):
            raise DAGLoadError(
                f"edges[{i}] ({src}→{dst}): weight must be a float, got '{weight_raw}'.",
                p,
            )
        if not (0.0 <= weight <= 1.0):
            raise DAGLoadError(
                f"edges[{i}] ({src}→{dst}): weight {weight} is out of range [0, 1].",
                p,
            )

        edges_validated.append({
            "source": src,
            "target": dst,
            "weight": weight,
            "mechanism": str(edge.get("mechanism", "")),
        })

    # 6 — Acyclicity (DFS)
    _assert_acyclic(nodes_validated, edges_validated, p)

    # 7 — Expected evidence (optional)
    expected_evidence_validated = {}
    if "expected_evidence" in raw:
        ev_raw = raw["expected_evidence"]
        if not isinstance(ev_raw, dict):
            raise DAGLoadError("'expected_evidence' must be a mapping of root_cause -> list[str].", p)
        for rc, obs_list in ev_raw.items():
            if not isinstance(obs_list, list):
                raise DAGLoadError(f"'expected_evidence.{rc}' must be a list of observable IDs.", p)
            expected_evidence_validated[str(rc)] = [str(o) for o in obs_list]

    result = {
        **raw,
        "nodes": nodes_validated,
        "edges": edges_validated,
    }
    if expected_evidence_validated:
        result["expected_evidence"] = expected_evidence_validated
        
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> Dict[str, Any]:
    """Read and parse a YAML or JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"DAG config not found: {path}")

    suffix = path.suffix.lower()

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DAGLoadError(f"Cannot read file: {exc}", path) from exc

    if suffix in (".yaml", ".yml"):
        if not _YAML_AVAILABLE:
            raise DAGLoadError(
                "PyYAML is not installed. Run: pip install PyYAML>=6.0", path
            )
        try:
            return yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise DAGLoadError(f"YAML parse error: {exc}", path) from exc

    elif suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise DAGLoadError(f"JSON parse error: {exc}", path) from exc

    else:
        raise DAGLoadError(
            f"Unsupported file extension '{suffix}'. Use .yaml, .yml, or .json.",
            path,
        )


def _init_empty_graph(graph: Any) -> None:
    """Initialise an empty CausalGraph (no hardcoded subsystems)."""
    # Import here to avoid circular refs at module load time
    from causal_graph.graph_definition import RUST_CORE_AVAILABLE

    graph.nodes = {}
    graph.edges = []
    graph.dag_meta = {}

    try:
        from aethelix.rust_core import PyCausalGraph  # type: ignore[import-not-found]
        graph.rust_graph = PyCausalGraph()
    except ImportError:
        try:
            from aethelix_core import PyCausalGraph  # type: ignore[import-not-found]
            graph.rust_graph = PyCausalGraph()
        except ImportError:
            graph.rust_graph = None


def _populate_nodes(graph: Any, nodes: List[Dict], path: Path) -> None:
    """Add validated node dicts to graph."""
    from causal_graph.graph_definition import NodeType

    _type_map = {
        "root_cause":   NodeType.ROOT_CAUSE,
        "intermediate": NodeType.INTERMEDIATE,
        "observable":   NodeType.OBSERVABLE,
    }

    for node in nodes:
        graph.add_node(
            node["id"],
            _type_map[node["type"]],
            node["description"],
            degradation_modes=node["degradation_modes"],
        )
        if "structural_equation" in node:
            seq = node["structural_equation"]
            graph.set_structural_equation(
                node_name=node["id"],
                coefficients=seq["coefficients"],
                noise_std=seq["noise_std"],
            )


def _populate_edges(graph: Any, edges: List[Dict], path: Path) -> None:
    """Add validated edge dicts to graph."""
    for edge in edges:
        graph.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            mechanism=edge["mechanism"],
        )


def _assert_acyclic(
    nodes: List[Dict],
    edges: List[Dict],
    path: Union[str, Path, None],
) -> None:
    """
    Detect cycles using iterative DFS (Kahn-style topological sort).

    Raises DAGLoadError if a cycle is found.
    """
    # Build adjacency list
    adj: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    in_degree: Dict[str, int] = {n["id"]: 0 for n in nodes}

    for edge in edges:
        adj[edge["source"]].append(edge["target"])
        in_degree[edge["target"]] += 1

    # Kahn's algorithm
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    visited = 0

    while queue:
        nid = queue.pop()
        visited += 1
        for child in adj[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited != len(nodes):
        # Find a node still in a cycle for a helpful error message
        cyclic = [nid for nid, deg in in_degree.items() if deg > 0]
        raise DAGLoadError(
            f"Graph contains a cycle involving nodes: {cyclic[:5]}. "
            "A causal DAG must be acyclic.",
            path,
        )


# ---------------------------------------------------------------------------
# Convenience: export an existing CausalGraph back to YAML
# ---------------------------------------------------------------------------

def dump_dag(graph: Any, path: Union[str, Path], satellite_meta: Optional[Dict] = None) -> Path:
    """
    Serialize an in-memory CausalGraph to a YAML config file.

    Useful for exporting the built-in Python graphs to canonical YAML.

    Parameters
    ----------
    graph : CausalGraph
        A populated graph.
    path : str | Path
        Output file path (should end in .yaml).
    satellite_meta : dict | None
        Optional ``satellite:`` block (name, agency, orbit, mission).

    Returns
    -------
    Path
        The file that was written.
    """
    if not _YAML_AVAILABLE:
        raise RuntimeError("PyYAML is required to dump YAML. Run: pip install PyYAML>=6.0")

    from causal_graph.graph_definition import NodeType

    _type_reverse = {
        NodeType.ROOT_CAUSE:   "root_cause",
        NodeType.INTERMEDIATE: "intermediate",
        NodeType.OBSERVABLE:   "observable",
    }

    nodes_out = []
    for name, node in graph.nodes.items():
        entry: Dict[str, Any] = {
            "id": name,
            "type": _type_reverse[node.node_type],
            "description": node.description,
        }
        if node.degradation_modes:
            entry["degradation_modes"] = node.degradation_modes
        nodes_out.append(entry)

    edges_out = []
    for edge in graph.edges:
        entry = {
            "source": edge.source,
            "target": edge.target,
            "weight": round(edge.weight, 4),
        }
        if edge.mechanism:
            entry["mechanism"] = edge.mechanism
        edges_out.append(entry)

    doc: Dict[str, Any] = {
        "aethelix_dag_version": SCHEMA_VERSION,
    }
    if satellite_meta:
        doc["satellite"] = satellite_meta
    doc["nodes"] = nodes_out
    doc["edges"] = edges_out

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(doc, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
