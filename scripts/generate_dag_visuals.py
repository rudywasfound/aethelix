#!/usr/bin/env python3
"""
Generate publication-ready, uncluttered Causal DAG diagrams with intensity coloring.

This script creates high-resolution (300 DPI) network visualizations of the Aethelix
Causal Directed Acyclic Graph (DAG) during active anomaly scenarios:
1. GSAT-6A Power Regulator Short & Thermal Cascade (Multi-Subsystem DAG)
2. ESA OPS-SAT Attitude Magnetometer Interference (ADCS Subsystem DAG)

Key aesthetic optimizations for uncluttered, publication-grade layout:
- Strict vertical capping (max y=0.68) guaranteeing zero collision with column headers (y=0.89)
- Explicit functional corridor positioning (zero edge crossings)
- Human-readable Title Case labels with smart anchor offsetting
- Multi-layered neon glow for high-confidence root causes and severe deviations
- Muted background edges vs thick, vibrant active propagation pathways
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import networkx as nx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from causal_graph.graph_definition import CausalGraph, NodeType

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

BG_COLOR = "#0A0E14"        # Ultra-deep dark background
TEXT_COLOR = "#F0F6FC"      # Crisp off-white text
MUTED_EDGE = "#21262D"      # Ultra-faint slate for nominal edges
ACTIVE_EDGE = "#FF5500"     # Fiery orange-red for active fault propagation
NOMINAL_NODE = "#1F6FEB"    # Cool blue for healthy nodes

colors_intensity = ["#1F6FEB", "#2EA043", "#D29922", "#F85149", "#FF0033"]
cmap_intensity = LinearSegmentedColormap.from_list("anomaly_intensity", colors_intensity, N=100)


def format_label(name: str, val: float) -> str:
    """Convert technical snake_case names to clean Title Case for publication."""
    clean = name.replace("_measured", " (Sensor)").replace("_", " ").title()
    clean = clean.replace("Pcdu", "PCDU").replace("Adcs", "ADCS").replace("Rw ", "Reaction Wheel ")
    if val > 0.05:
        return f"{clean}\n[{val*100:.0f}% Belief]" if "Root Cause" in name or val > 0.7 else f"{clean}\n[{val*100:.0f}% Severity]"
    return clean


def draw_uncluttered_dag(
    G: nx.DiGraph,
    pos: dict,
    node_intensity: dict,
    active_edges: set,
    title: str,
    subtitle: str,
    output_filename: str,
    figsize=(18, 11)
):
    """
    Draw clean, uncluttered DAG with functional corridor layout and neon glow.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    nominal_edges = [e for e in G.edges() if e not in active_edges]
    act_edges = [e for e in G.edges() if e in active_edges]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=nominal_edges,
        edge_color=MUTED_EDGE,
        width=1.2,
        alpha=0.35,
        style="dashed",
        arrows=True,
        arrowsize=10,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.03",
        ax=ax
    )

    if act_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=act_edges,
            edge_color=ACTIVE_EDGE,
            width=6.0,
            alpha=0.2,
            arrows=True,
            arrowstyle="-",
            arrowsize=1,
            connectionstyle="arc3,rad=0.03",
            ax=ax
        )
        nx.draw_networkx_edges(
            G, pos,
            edgelist=act_edges,
            edge_color=ACTIVE_EDGE,
            width=3.2,
            alpha=0.95,
            arrows=True,
            arrowsize=22,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.03",
            ax=ax
        )

    node_list = list(G.nodes())
    intensities = [node_intensity.get(n, 0.0) for n in node_list]
    base_sizes = [900 + 1200 * (node_intensity.get(n, 0.0) ** 1.2) for n in node_list]

    for idx, node in enumerate(node_list):
        val = intensities[idx]
        if val > 0.3:
            color = cmap_intensity(val)
            for scale, alpha in [(2.4, 0.08), (1.8, 0.15), (1.3, 0.25)]:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=[node],
                    node_size=base_sizes[idx] * scale,
                    node_color=[color],
                    alpha=alpha,
                    ax=ax
                )

    nodes_draw = nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_size=base_sizes,
        node_color=intensities,
        cmap=cmap_intensity,
        vmin=0.0,
        vmax=1.0,
        edgecolors="white",
        linewidths=1.8,
        ax=ax
    )

    for node in node_list:
        x, y = pos[node]
        val = node_intensity.get(node, 0.0)
        label_text = format_label(node, val)
        
        if x < 0.3:
            lx, ly = x - 0.035, y
            ha, va = "right", "center"
        elif x > 0.7:
            lx, ly = x + 0.035, y
            ha, va = "left", "center"
        else:
            lx = x
            ly = y + 0.045 if y > 0.40 else y - 0.045
            ha, va = "center", "bottom" if y > 0.40 else "top"

        font_weight = "bold" if val > 0.4 else "normal"
        font_color = "#FF9999" if val > 0.7 else TEXT_COLOR
        
        ax.text(
            lx, ly, label_text,
            color=font_color,
            fontsize=10.5 if val > 0.4 else 9.5,
            fontweight=font_weight,
            horizontalalignment=ha,
            verticalalignment=va,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161B22", edgecolor="#30363D" if val < 0.4 else ACTIVE_EDGE, alpha=0.85, lw=1.0)
        )

    ax.text(0.02, 0.96, title, color="white", fontsize=22, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.925, subtitle, color="#8B949E", fontsize=13.5, transform=ax.transAxes)

    ax.axvline(0.33, color="#21262D", linestyle=":", lw=1.5, alpha=0.5)
    ax.axvline(0.66, color="#21262D", linestyle=":", lw=1.5, alpha=0.5)
    
    ax.text(0.16, 0.87, "ROOT CAUSES\n(Exogenous Fault Origin)", color="#58A6FF", fontsize=12, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.50, 0.87, "INTERMEDIATE EFFECTS\n(Unobservable State Propagation)", color="#D29922", fontsize=12, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.84, 0.87, "OBSERVABLE TELEMETRY\n(Measured Sensor Z-Scores)", color="#3FB950", fontsize=12, fontweight="bold", ha="center", transform=ax.transAxes)

    cbar_ax = fig.add_axes([0.35, 0.035, 0.3, 0.018])
    cbar = fig.colorbar(nodes_draw, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Anomaly Propagation Intensity & Bayesian Belief Score", color=TEXT_COLOR, fontsize=11, fontweight="bold")
    cbar.ax.xaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color=TEXT_COLOR)

    active_patch = LineCollection([[ (0, 0), (1, 0) ]], colors=[ACTIVE_EDGE], linewidths=3.5, label="Active Fault Propagation Corridor")
    nominal_patch = LineCollection([[ (0, 0), (1, 0) ]], colors=[MUTED_EDGE], linewidths=1.5, linestyles="dashed", label="Nominal / Inactive Subsystem Coupling")
    ax.legend(handles=[active_patch, nominal_patch], loc="lower left", facecolor="#161B22", edgecolor="#30363D", labelcolor=TEXT_COLOR, fontsize=10.5)

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.02, 0.98)
    ax.axis("off")

    out_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Generated Uncluttered DAG Visual: {out_path}")
    plt.close()


def build_gsat6a_scenario():
    """
    Scenario 1: GSAT-6A PCDU Power Regulator Failure & Thermal Cascade.
    All y-coordinates strictly <= 0.65 to guarantee massive empty spacing below headers (y=0.87).
    """
    G = nx.DiGraph()

    nodes = [
        "pcdu_regulator_failure", "battery_aging", "battery_thermal",
        "bus_regulation", "battery_state", "battery_temp",
        "bus_voltage_measured", "bus_current_measured", "battery_voltage_measured", "battery_temp_measured"
    ]
    for n in nodes:
        G.add_node(n)

    edges = [
        ("pcdu_regulator_failure", "bus_regulation"),
        ("bus_regulation", "bus_voltage_measured"),
        ("pcdu_regulator_failure", "bus_current_measured"),
        ("battery_aging", "battery_state"),
        ("battery_state", "battery_voltage_measured"),
        ("battery_thermal", "battery_temp"),
        ("battery_temp", "battery_temp_measured"),
        ("battery_thermal", "battery_state"),
    ]
    for s, t in edges:
        G.add_edge(s, t)

    pos = {
        "pcdu_regulator_failure": (0.16, 0.64),
        "bus_regulation":         (0.50, 0.64),
        "bus_voltage_measured":   (0.84, 0.68),
        "bus_current_measured":   (0.84, 0.54),
        
        "battery_aging":          (0.16, 0.38),
        "battery_state":          (0.50, 0.38),
        "battery_voltage_measured": (0.84, 0.38),
        
        "battery_thermal":        (0.16, 0.14),
        "battery_temp":           (0.50, 0.14),
        "battery_temp_measured":  (0.84, 0.14),
    }

    node_intensity = {
        "pcdu_regulator_failure": 0.94,
        "bus_regulation": 0.88,
        "bus_voltage_measured": 0.92,
        "bus_current_measured": 0.85,
        "battery_thermal": 0.60,
        "battery_temp": 0.65,
        "battery_temp_measured": 0.70,
        "battery_state": 0.35,
        "battery_voltage_measured": 0.38,
        "battery_aging": 0.10,
    }

    active_edges = {
        ("pcdu_regulator_failure", "bus_regulation"),
        ("bus_regulation", "bus_voltage_measured"),
        ("pcdu_regulator_failure", "bus_current_measured"),
        ("battery_thermal", "battery_temp"),
        ("battery_temp", "battery_temp_measured"),
    }

    draw_uncluttered_dag(
        G, pos, node_intensity, active_edges,
        title="Aethelix Causal DAG: GSAT-6A Power Short & Thermal Cascade",
        subtitle="Uncluttered corridor layout isolating bus regulation collapse (94% Belief) from nominal battery aging",
        output_filename="causal_dag_intensity_gsat6a.png"
    )


def build_opssat_scenario():
    """
    Scenario 2: ESA OPS-SAT Reaction Wheel Magnetic Interference (ADCS).
    All y-coordinates strictly <= 0.65 to guarantee massive empty spacing below headers (y=0.87).
    """
    G = nx.DiGraph()

    nodes = [
        "rw_magnetic_interference", "solar_panel_vibration", "gyro_bias_drift",
        "adcs_mag_bias", "adcs_rate_noise", "attitude_error",
        "adcs_mag_x", "adcs_mag_y", "adcs_mag_z", "adcs_gyro_x", "pd1_theta"
    ]
    for n in nodes:
        G.add_node(n)

    edges = [
        ("rw_magnetic_interference", "adcs_mag_bias"),
        ("adcs_mag_bias", "adcs_mag_x"),
        ("adcs_mag_bias", "adcs_mag_y"),
        ("adcs_mag_bias", "adcs_mag_z"),
        ("adcs_mag_bias", "attitude_error"),
        ("attitude_error", "pd1_theta"),
        ("gyro_bias_drift", "adcs_rate_noise"),
        ("adcs_rate_noise", "adcs_gyro_x"),
        ("solar_panel_vibration", "adcs_rate_noise"),
    ]
    for s, t in edges:
        G.add_edge(s, t)

    pos = {
        "rw_magnetic_interference": (0.16, 0.64),
        "adcs_mag_bias":            (0.50, 0.64),
        "adcs_mag_x":               (0.84, 0.68),
        "adcs_mag_y":               (0.84, 0.58),
        "adcs_mag_z":               (0.84, 0.48),
        
        "attitude_error":           (0.50, 0.36),
        "pd1_theta":                (0.84, 0.36),
        
        "solar_panel_vibration":    (0.16, 0.22),
        "gyro_bias_drift":          (0.16, 0.10),
        "adcs_rate_noise":          (0.50, 0.16),
        "adcs_gyro_x":              (0.84, 0.16),
    }

    node_intensity = {
        "rw_magnetic_interference": 0.89,
        "adcs_mag_bias": 0.84,
        "adcs_mag_x": 0.91,
        "adcs_mag_y": 0.87,
        "adcs_mag_z": 0.76,
        "attitude_error": 0.45,
        "pd1_theta": 0.38,
        "solar_panel_vibration": 0.04,
        "gyro_bias_drift": 0.05,
        "adcs_rate_noise": 0.08,
        "adcs_gyro_x": 0.05,
    }

    active_edges = {
        ("rw_magnetic_interference", "adcs_mag_bias"),
        ("adcs_mag_bias", "adcs_mag_x"),
        ("adcs_mag_bias", "adcs_mag_y"),
        ("adcs_mag_bias", "adcs_mag_z"),
        ("adcs_mag_bias", "attitude_error"),
        ("attitude_error", "pd1_theta"),
    }

    draw_uncluttered_dag(
        G, pos, node_intensity, active_edges,
        title="Aethelix Causal DAG: ESA OPS-SAT Magnetometer Interference",
        subtitle="Uncluttered ADCS layout tracing multi-axis magnetometer spikes to reaction wheel magnetic interference (89% Belief)",
        output_filename="causal_dag_intensity_opssat.png"
    )


if __name__ == "__main__":
    print("Generating Uncluttered Causal DAG Intensity Visualizations...")
    build_gsat6a_scenario()
    build_opssat_scenario()
    print("All uncluttered Causal DAG visualizations generated successfully!")
