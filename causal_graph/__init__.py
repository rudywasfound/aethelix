from causal_graph.graph_definition import (
    CausalGraph, NodeType, Node, Edge,
    StructuralEquation, SCMNotConfiguredError,
)
from causal_graph.root_cause_ranking import RootCauseRanker, RootCauseHypothesis
from causal_graph.dag_loader import load_dag, dump_dag, DAGLoadError

try:
    from causal_graph.scm import ate, interventional_distribution
except ImportError:
    ate = None
    interventional_distribution = None

try:
    from causal_graph.visualizer import DAGVisualizer
except ImportError:
    class DAGVisualizer:
        def __init__(self, *args, **kwargs):
            raise ImportError("DAGVisualizer requires matplotlib and networkx. Please install them to use visualization.")

__all__ = [
    "CausalGraph",
    "NodeType",
    "Node",
    "Edge",
    "StructuralEquation",
    "SCMNotConfiguredError",
    "DAGVisualizer",
    "RootCauseRanker",
    "RootCauseHypothesis",
    # Pluggable DAG loader
    "load_dag",
    "dump_dag",
    "DAGLoadError",
    # SCM utilities
    "ate",
    "interventional_distribution",
]

