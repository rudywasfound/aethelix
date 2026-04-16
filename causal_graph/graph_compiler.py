"""
Causal Graph Binary Compiler for Aethelix Flight Firmware.

Compiles the Python CausalGraph definition into a compact binary format
suitable for embedding in LEON3 firmware via Rust's include_bytes!().

Binary format (little-endian):
  Bytes 0–3:  Magic: 0xCA 0x05 0xAE 0x01
  Byte  4:    Number of nodes (u8, max 255)
  Byte  5:    Number of edges (u8, max 255)
  Bytes 6..:  Node table — for each node: [node_id: u8, node_type: u8]
              node_type — 0=ROOT_CAUSE, 1=INTERMEDIATE, 2=OBSERVABLE
  Remaining:  Edge table — for each edge: [src_id: u8, dst_id: u8, weight_u8: u8]
              weight_u8 — maps [0.0, 1.0] → [0, 255]

Total size: ~350 bytes for the current 50-node, 80-edge graph (well under 4 KB budget).

Output files:
  causal_graph/causal_graph.bin  — embedded in Rust via include_bytes!()
  causal_graph/graph_ids.json    — node_name → node_id mapping (for debugging)
  include/aethelix_graph_ids.h   — C header with FAULT_* constants (for aethelix.h)

Usage:
    python causal_graph/graph_compiler.py
    # Regenerate whenever graph_definition.py changes.
"""

import struct
import json
from pathlib import Path
import sys

# Allow running from repo root or from causal_graph/ directory
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from causal_graph.graph_definition import CausalGraph, NodeType

# ── Constants ──────────────────────────────────────────────────────────────────
MAGIC = bytes([0xCA, 0x05, 0xAE, 0x01])

NODE_TYPE_MAP = {
    NodeType.ROOT_CAUSE:   0,
    NodeType.INTERMEDIATE: 1,
    NodeType.OBSERVABLE:   2,
}

MAX_BINARY_SIZE = 4096   # Flash budget for the embedded graph


# ── Compiler ───────────────────────────────────────────────────────────────────

def compile_graph(
    output_bin:  Path = None,
    output_ids:  Path = None,
    output_hdr:  Path = None,
) -> tuple[dict, int]:
    """
    Compile CausalGraph to binary.

    Returns:
        (node_id_map, binary_size_bytes)
    """
    script_dir = Path(__file__).parent

    output_bin = output_bin or script_dir / "causal_graph.bin"
    output_ids = output_ids or script_dir / "graph_ids.json"
    output_hdr = output_hdr or REPO_ROOT / "include" / "aethelix_graph_ids.h"

    print("Loading Aethelix CausalGraph...")
    graph = CausalGraph()

    # Assign sequential IDs sorted by name (deterministic across Python versions)
    node_names  = sorted(graph.nodes.keys())
    node_id_map = {name: idx for idx, name in enumerate(node_names)}

    n_nodes = len(node_names)
    n_edges = len(graph.edges)

    if n_nodes > 255:
        raise ValueError(f"Too many nodes ({n_nodes} > 255). Upgrade to u16 format.")
    if n_edges > 255:
        raise ValueError(f"Too many edges ({n_edges} > 255). Upgrade to u16 format.")

    print(f"  Nodes: {n_nodes}  |  Edges: {n_edges}")

    # ── Build binary ──────────────────────────────────────────────────────────
    buf = bytearray()

    # Header
    buf.extend(MAGIC)
    buf.append(n_nodes)
    buf.append(n_edges)

    # Node table: [node_id: u8, node_type: u8]
    for node_name in node_names:
        node_id   = node_id_map[node_name]
        node_type = NODE_TYPE_MAP[graph.nodes[node_name].node_type]
        buf.append(node_id)
        buf.append(node_type)

    # Edge table: [src_id: u8, dst_id: u8, weight_u8: u8]
    skipped = 0
    edge_count_actual = 0
    for edge in graph.edges:
        if edge.source not in node_id_map or edge.target not in node_id_map:
            skipped += 1
            continue
        src_id    = node_id_map[edge.source]
        dst_id    = node_id_map[edge.target]
        weight_u8 = max(0, min(255, round(edge.weight * 255.0)))
        buf.append(src_id)
        buf.append(dst_id)
        buf.append(weight_u8)
        edge_count_actual += 1

    if skipped:
        print(f"  WARNING: {skipped} edges skipped (unknown node reference)")

    # ── Write binary ──────────────────────────────────────────────────────────
    output_bin.write_bytes(bytes(buf))
    size_bytes = len(buf)
    budget_ok  = size_bytes <= MAX_BINARY_SIZE

    print(f"  Written: {output_bin}  ({size_bytes} bytes)")
    print(f"  Budget:  ≤{MAX_BINARY_SIZE} bytes → {'✓ OK' if budget_ok else '✗ EXCEEDS BUDGET'}")

    # ── Node ID mapping JSON ──────────────────────────────────────────────────
    root_cause_ids = {
        name: node_id_map[name]
        for name in node_names
        if graph.nodes[name].node_type == NodeType.ROOT_CAUSE
    }
    id_info = {
        "format_version":  "1",
        "magic":           "0xCA 0x05 0xAE 0x01",
        "n_nodes":         n_nodes,
        "n_edges":         edge_count_actual,
        "node_ids":        node_id_map,
        "node_types": {
            name: NODE_TYPE_MAP[graph.nodes[name].node_type]
            for name in node_names
        },
        "root_causes": root_cause_ids,
        "observables": {
            name: node_id_map[name]
            for name in node_names
            if graph.nodes[name].node_type == NodeType.OBSERVABLE
        },
    }
    output_ids.write_text(json.dumps(id_info, indent=2))
    print(f"  Written: {output_ids}")

    # ── C header with fault ID constants ─────────────────────────────────────
    output_hdr.parent.mkdir(parents=True, exist_ok=True)
    _write_c_header(output_hdr, root_cause_ids, n_nodes, n_edges)
    print(f"  Written: {output_hdr}")

    return node_id_map, size_bytes


def _write_c_header(path: Path, root_cause_ids: dict, n_nodes: int, n_edges: int):
    """Generate C preprocessor constants for all root cause node IDs."""
    lines = [
        "/* aethelix_graph_ids.h",
        " * Auto-generated by causal_graph/graph_compiler.py — DO NOT EDIT.",
        " * Root-cause fault IDs matching the compiled causal_graph.bin.",
        " */",
        "",
        "#ifndef AETHELIX_GRAPH_IDS_H",
        "#define AETHELIX_GRAPH_IDS_H",
        "",
        f"#define AETHELIX_GRAPH_N_NODES  {n_nodes}U",
        f"#define AETHELIX_GRAPH_N_EDGES  {n_edges}U",
        "",
        "/* Root cause node IDs (match compiled graph binary) */",
    ]
    for name, node_id in sorted(root_cause_ids.items(), key=lambda x: x[1]):
        macro_name = f"AETHELIX_FAULT_{name.upper()}"
        lines.append(f"#define {macro_name:<52} 0x{node_id:02X}U")

    lines += [
        "",
        "#define AETHELIX_FAULT_NONE  0xFFU  /* No fault detected */",
        "",
        "#endif /* AETHELIX_GRAPH_IDS_H */",
        "",
    ]
    path.write_text("\n".join(lines))


# ── Verifier ───────────────────────────────────────────────────────────────────

def verify_binary(bin_path: Path = None) -> bool:
    """Read back and verify structural integrity of the compiled binary."""
    bin_path = bin_path or Path(__file__).parent / "causal_graph.bin"

    data = bin_path.read_bytes()

    # Header checks
    assert data[:4] == MAGIC, f"Magic mismatch! Got {data[:4].hex()}"
    n_nodes = data[4]
    n_edges = data[5]

    expected_len = 6 + n_nodes * 2 + n_edges * 3
    assert len(data) == expected_len, (
        f"Size mismatch: expected {expected_len} bytes, got {len(data)}"
    )

    # Node type validity
    node_section_start = 6
    for i in range(n_nodes):
        node_type = data[node_section_start + i * 2 + 1]
        assert node_type in (0, 1, 2), (
            f"Invalid node type {node_type} at node index {i}"
        )

    # Edge validity
    edge_section_start = node_section_start + n_nodes * 2
    for i in range(n_edges):
        base   = edge_section_start + i * 3
        src_id = data[base]
        dst_id = data[base + 1]
        weight = data[base + 2]
        assert src_id < n_nodes, f"Edge {i}: src {src_id} out of range"
        assert dst_id < n_nodes, f"Edge {i}: dst {dst_id} out of range"
        assert 0 <= weight <= 255

    print(f"  Verification PASSED — {n_nodes} nodes, {n_edges} edges, {len(data)} bytes")
    return True


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Aethelix Causal Graph Binary Compiler")
    print("=" * 60)

    node_id_map, size = compile_graph()
    verify_binary()

    print()
    print("Root cause IDs (for FDIR reference / aethelix_graph_ids.h):")
    graph = CausalGraph()
    node_names  = sorted(graph.nodes.keys())
    nid_map     = {name: idx for idx, name in enumerate(node_names)}
    for name in node_names:
        if graph.nodes[name].node_type == NodeType.ROOT_CAUSE:
            print(f"  0x{nid_map[name]:02X}  {name}")

    print()
    print("Done. causal_graph.bin is ready for include_bytes!() in Rust firmware.")
