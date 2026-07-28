"""
run_aethelix.py — CLI entry point for the Aethelix causal inference framework.

Lets users run the full diagnosis pipeline against any satellite config
described in a YAML or JSON file.  No Python source changes required.

Examples
--------
Validate a config without running simulation:
    python scripts/run_aethelix.py --config configs/sentinel1b.yaml --validate-only

Full run with custom output directory:
    python scripts/run_aethelix.py --config configs/gsat6a.yaml --output-dir results/gsat6a/

Use the built-in GSAT-6A Python graph (backward-compat mode):
    python scripts/run_aethelix.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_aethelix",
        description="Aethelix — Physics-Based Causal Inference for Satellite Fault Diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_aethelix.py --config configs/gsat6a.yaml
  python scripts/run_aethelix.py --config configs/sentinel1b.yaml --validate-only
  python scripts/run_aethelix.py --config configs/cubesat_3u.yaml --output-dir out/cubesat/
  python scripts/run_aethelix.py
        """,
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a satellite DAG config (.yaml / .yml / .json). "
             "If omitted, the built-in GSAT-6A Python graph is used.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Parse and validate the config, print a summary, then exit. "
             "Does not run simulation or inference.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="output",
        help="Directory for output plots and reports (default: output/).",
    )
    parser.add_argument(
        "--deviation-threshold",
        metavar="FLOAT",
        type=float,
        default=0.15,
        help="Residual deviation threshold for anomaly detection (default: 0.15).",
    )
    return parser.parse_args()


def _load_graph(config_path: str | None):
    """Load CausalGraph from YAML/JSON or fall back to built-in Python graph."""
    if config_path:
        from causal_graph.dag_loader import load_dag, DAGLoadError
        try:
            graph = load_dag(config_path)
        except FileNotFoundError:
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
        except DAGLoadError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
    else:
        from causal_graph.graph_definition import CausalGraph
        graph = CausalGraph()

    return graph


def _print_graph_summary(graph, source: str) -> None:
    """Print a structured summary of the loaded graph."""
    from causal_graph.graph_definition import NodeType

    root_causes   = [n for n, node in graph.nodes.items() if node.node_type == NodeType.ROOT_CAUSE]
    intermediates = [n for n, node in graph.nodes.items() if node.node_type == NodeType.INTERMEDIATE]
    observables   = [n for n, node in graph.nodes.items() if node.node_type == NodeType.OBSERVABLE]

    meta = getattr(graph, "dag_meta", {})
    sat  = meta.get("satellite", {})

    print()
    print("━" * 60)
    print("  AETHELIX — Satellite DAG")
    print("━" * 60)

    if sat.get("name"):
        print(f"  Satellite : {sat['name']}")
    if sat.get("agency"):
        print(f"  Agency    : {sat['agency']}")
    if sat.get("orbit"):
        print(f"  Orbit     : {sat['orbit']}")
    if sat.get("mission"):
        print(f"  Mission   : {sat['mission']}")
    if meta.get("aethelix_dag_version"):
        print(f"  Schema    : Aethelix DAG v{meta['aethelix_dag_version']}")

    print(f"  Source    : {source}")
    print()
    print(f"  Graph stats")
    print(f"    Root causes   : {len(root_causes)}")
    print(f"    Intermediates : {len(intermediates)}")
    print(f"    Observables   : {len(observables)}")
    print(f"    Total nodes   : {len(graph.nodes)}")
    print(f"    Edges         : {len(graph.edges)}")
    print()
    print("  Root causes (diagnosis targets):")
    for rc in sorted(root_causes):
        desc = graph.nodes[rc].description
        print(f"    • {rc:<35s}  {desc}")
    print()
    print("  Observables (telemetry):")
    for obs in sorted(observables):
        desc = graph.nodes[obs].description
        print(f"    • {obs:<35s}  {desc}")
    print("━" * 60)


def main() -> None:
    args = _parse_args()

    source = args.config if args.config else "<built-in GSAT-6A Python graph>"
    print(f"\nAethelix Causal Inference Framework")
    print(f"Loading graph from: {source}")

    graph = _load_graph(args.config)
    _print_graph_summary(graph, source)

    if args.validate_only:
        print("[OK] Validation passed. Exiting (--validate-only mode).")
        return

    print("\n[1] Initializing simulators...")
    from simulator.power   import PowerSimulator
    from simulator.thermal import ThermalSimulator

    power_sim   = PowerSimulator(duration_hours=24, sampling_rate_hz=0.1)
    thermal_sim = ThermalSimulator(duration_hours=24, sampling_rate_hz=0.1)

    print("[2] Running nominal scenario...")
    power_nom   = power_sim.run_nominal()
    thermal_nom = thermal_sim.run_nominal(
        power_nom.solar_input,
        power_nom.battery_charge,
        power_nom.battery_voltage,
    )
    from main import CombinedTelemetry
    nominal = CombinedTelemetry(power_nom, thermal_nom)

    print("[3] Running degraded scenario (multi-fault)...")
    power_deg = power_sim.run_degraded(
        solar_degradation_hour=6.0,
        solar_factor=0.7,
        battery_degradation_hour=8.0,
        battery_factor=0.8,
    )
    thermal_deg = thermal_sim.run_degraded(
        power_deg.solar_input,
        power_deg.battery_charge,
        power_deg.battery_voltage,
        battery_cooling_hour=8.0,
        battery_cooling_factor=0.5,
    )
    degraded = CombinedTelemetry(power_deg, thermal_deg)

    print("[4] Analyzing deviations...")
    from analysis.residual_analyzer import ResidualAnalyzer
    analyzer = ResidualAnalyzer(deviation_threshold=args.deviation_threshold)
    stats    = analyzer.analyze(nominal, degraded)
    analyzer.print_report(stats)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("[5] Generating plots...")
    from visualization.plotter import TelemetryPlotter
    plotter = TelemetryPlotter()
    plotter.plot_residuals(nominal, degraded, save_path=f"{output_dir}/residuals.png")

    print("[6] Ranking root causes...")
    from causal_graph.root_cause_ranking import RootCauseRanker
    ranker      = RootCauseRanker(graph)
    hypotheses  = ranker.analyze(nominal, degraded, deviation_threshold=args.deviation_threshold)
    ranker.print_report(hypotheses)

    print(f"\nOutputs saved to '{output_dir}/'")
    print("Workflow complete.")


if __name__ == "__main__":
    main()
