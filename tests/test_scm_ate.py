#!/usr/bin/env python3
"""
test_scm_ate.py
Verification gate for the SCM layer. Constructs a known 3-node linear-Gaussian
chain and verifies that the Monte Carlo ATE matches the analytical true ATE.
"""

import math
from causal_graph.graph_definition import CausalGraph, NodeType, StructuralEquation
from causal_graph.scm import ate

def test_chain_ate():
    print("Running SCM ATE Correctness Test...")
    graph = CausalGraph.__new__(CausalGraph)
    graph.nodes = {}
    graph.edges = []
    graph.dag_meta = {}
    graph.rust_graph = None
    
    graph.add_node("A", NodeType.ROOT_CAUSE, "Node A")
    graph.add_node("B", NodeType.INTERMEDIATE, "Node B")
    graph.add_node("C", NodeType.OBSERVABLE, "Node C")
    
    graph.add_edge("A", "B", weight=1.0)
    graph.add_edge("B", "C", weight=1.0)
    
    
    graph.set_structural_equation("A", coefficients={}, noise_std=1.0)
    graph.set_structural_equation("B", coefficients={"A": 0.8}, noise_std=0.5)
    graph.set_structural_equation("C", coefficients={"B": 0.5}, noise_std=0.5)
    
    true_ate = 0.4
    
    estimated_ate = ate(graph, cause="A", effect="C", x0=0.0, x1=1.0, n_samples=50000)
    
    print(f"True ATE: {true_ate:.4f}")
    print(f"Estimated ATE: {estimated_ate:.4f}")
    
    assert math.isclose(estimated_ate, true_ate, abs_tol=0.02), f"ATE estimate {estimated_ate} too far from true {true_ate}"
    print("✅ SCM ATE test passed.")

if __name__ == "__main__":
    test_chain_ate()
