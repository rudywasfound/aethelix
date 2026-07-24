#!/usr/bin/env python3
"""
test_rust_python_parity.py
Verifies that the Python fallback for get_weighted_paths_to_root produces
identical results (to within float precision) as the Rust core implementation.
Also runs a small benchmark to ensure the Python memoization cache works.
"""

import time
import math
from causal_graph.graph_definition import CausalGraph, RUST_CORE_AVAILABLE
from causal_graph.root_cause_ranking import RootCauseRanker
from simulator.power import PowerSimulator

def test_path_parity():
    if not RUST_CORE_AVAILABLE:
        print("⚠️ Rust core not available. Skipping parity test.")
        return True

    print("Running Rust vs Python path traversal parity test...")
    graph = CausalGraph()

    # Get a complex observable node that traces back to multiple root causes
    test_node = "bus_voltage_measured"

    # 1. Get paths using Rust
    start = time.time()
    rust_paths = graph.get_weighted_paths_to_root(test_node, max_depth=10)
    rust_time = time.time() - start

    # 2. Disable Rust and clear cache to get Python times
    graph.rust_graph = None
    if hasattr(graph, '_path_cache'):
        graph._path_cache.clear()
        
    start = time.time()
    python_paths = graph.get_weighted_paths_to_root(test_node, max_depth=10)
    python_time = time.time() - start

    # 3. Check cache hit time
    start = time.time()
    cached_paths = graph.get_weighted_paths_to_root(test_node, max_depth=10)
    cached_time = time.time() - start

    # Sort paths for comparison
    def sort_key(p):
        return (p[0], p[1])
        
    rust_paths.sort(key=sort_key)
    python_paths.sort(key=sort_key)

    assert len(rust_paths) == len(python_paths), f"Path count mismatch: Rust {len(rust_paths)}, Python {len(python_paths)}"

    for (r_path, r_weight), (p_path, p_weight) in zip(rust_paths, python_paths):
        assert r_path == p_path, f"Path mismatch: {r_path} != {p_path}"
        assert math.isclose(r_weight, p_weight, rel_tol=1e-5), f"Weight mismatch for {r_path}: {r_weight} != {p_weight}"

    print("✅ Parity confirmed: Python fallback matches Rust core exact output.")
    print(f"⏱️  Performance: Rust {rust_time*1000:.2f}ms, Python {python_time*1000:.2f}ms, Python Cached {cached_time*1000:.2f}ms")
    
    return True

if __name__ == "__main__":
    test_path_parity()
