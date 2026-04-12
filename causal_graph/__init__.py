from causal_graph.graph_definition import CausalGraph, NodeType, Node, Edge
from causal_graph.root_cause_ranking import RootCauseRanker, RootCauseHypothesis

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
    "DAGVisualizer",
    "RootCauseRanker",
    "RootCauseHypothesis",
]
