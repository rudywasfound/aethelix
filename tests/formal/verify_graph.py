import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from causal_graph.graph_definition import CausalGraph, NodeType

def test_dag_properties():
    """
    Formal verification of Causal DAG properties.
    Ensures the graph is technically sound for inference.
    """
    graph = CausalGraph()
    print(f"--- Verifying DAG: {len(graph.nodes)} nodes, {len(graph.edges)} edges ---")

    # 1. Cycle Detection (Must be a DAG)
    def has_cycle():
        visited = set()
        stack = set()
        
        def visit(node):
            if node in stack: return True
            if node in visited: return False
            visited.add(node)
            stack.add(node)
            for child in graph.get_children(node):
                if visit(child): return True
            stack.remove(node)
            return False

        for node in graph.nodes:
            if visit(node): return True
        return False

    if has_cycle():
        raise AssertionError("CRITICAL: Causal Graph contains cycles. It must be a Directed Acyclic Graph (DAG).")
    print("[PASS] No cycles detected.")

    # 2. Reachability (Every root cause must reach at least one observable)
    root_causes = graph.get_root_causes()
    observables = set(graph.get_observables())
    
    for root in root_causes:
        # Simple BFS for reachability
        reached_observables = False
        todo = [root]
        seen = {root}
        while todo:
            curr = todo.pop(0)
            if curr in observables:
                reached_observables = True
                break
            for child in graph.get_children(curr):
                if child not in seen:
                    seen.add(child)
                    todo.append(child)
        
        if not reached_observables:
            raise AssertionError(f"CRITICAL: Root cause '{root}' is unreachable from any telemetry observable.")
    print("[PASS] All root causes have observable paths.")

    # 3. Connectivity (All nodes must be part of the graph)
    # Check if any nodes are isolated (no parents and no children)
    for name in graph.nodes:
        if not graph.get_parents(name) and not graph.get_children(name):
            raise AssertionError(f"WARNING: Isolated node detected: '{name}'")
    print("[PASS] All nodes are connected to the causal structure.")

if __name__ == "__main__":
    try:
        test_dag_properties()
        print("\n--- FORMAL VERIFICATION SUCCESSFUL ---")
    except Exception as e:
        print(f"\n--- FORMAL VERIFICATION FAILED: {str(e)} ---")
        sys.exit(1)
