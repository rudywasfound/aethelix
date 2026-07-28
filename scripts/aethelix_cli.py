#!/usr/bin/env python3
"""
CLI entry point for Aethelix pluggable DAG validation and management.
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_graph.dag_loader import load_dag, validate_schema, dump_dag, DAGLoadError
from causal_graph.graph_definition import CausalGraph

def validate_cmd(args):
    """Validate a YAML/JSON DAG file."""
    path = Path(args.config_path)
    if not path.exists():
        print(f"Error: File '{path}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    try:
        graph = load_dag(path)
        print(f"✓ Validated Aethelix DAG: {path}")
        print(f"  Nodes: {len(graph.nodes)}")
        print(f"  Edges: {len(graph.edges)}")
        
        from causal_graph.graph_definition import NodeType
        root_causes = sum(1 for n in graph.nodes.values() if n.node_type == NodeType.ROOT_CAUSE)
        intermediates = sum(1 for n in graph.nodes.values() if n.node_type == NodeType.INTERMEDIATE)
        observables = sum(1 for n in graph.nodes.values() if n.node_type == NodeType.OBSERVABLE)
        
        print(f"  - Root Causes: {root_causes}")
        print(f"  - Intermediates: {intermediates}")
        print(f"  - Observables: {observables}")
        
    except DAGLoadError as e:
        print(f"✗ Validation Failed: {e.hint}", file=sys.stderr)
        print(f"  File: {e.path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

def dump_gsat6a_cmd(args):
    """Dump the built-in GSAT-6A CausalGraph to a YAML file."""
    output_path = Path(args.output_path)
    try:
        g = CausalGraph()
        dump_dag(g, output_path, satellite_meta={
            'name': 'GSAT-6A',
            'agency': 'ISRO',
            'orbit': 'GEO',
            'mission': 'Communications (S-band)',
            'failure_date': '2018-03-26',
            'description': 'Multi-subsystem causal fault graph for GSAT-6A. Covers EPS, TCS, ADCS, COMMS, OBC, and Propulsion.',
        })
        print(f"✓ Dumped built-in GSAT-6A graph to {output_path}")
    except Exception as e:
        print(f"✗ Failed to dump GSAT-6A graph: {e}", file=sys.stderr)
        sys.exit(1)

def run_cmd(args):
    """Load a config file and print its structure."""
    path = Path(args.config_path)
    try:
        graph = load_dag(path)
        graph.print_structure()
    except Exception as e:
        print(f"✗ Failed to load or run DAG: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Aethelix Causal Inference Framework CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    val_parser = subparsers.add_parser("validate", help="Validate a satellite DAG YAML/JSON config file")
    val_parser.add_argument("config_path", help="Path to the DAG configuration file")
    
    dump_parser = subparsers.add_parser("dump-gsat6a", help="Dump the built-in GSAT-6A graph to a YAML file")
    dump_parser.add_argument("output_path", help="Target output path (e.g. configs/gsat6a.yaml)")

    run_parser = subparsers.add_parser("run", help="Load and print the structure of a DAG configuration")
    run_parser.add_argument("config_path", help="Path to the DAG configuration file")

    args = parser.parse_args()

    if args.command == "validate":
        validate_cmd(args)
    elif args.command == "dump-gsat6a":
        dump_gsat6a_cmd(args)
    elif args.command == "run":
        run_cmd(args)

if __name__ == "__main__":
    main()
