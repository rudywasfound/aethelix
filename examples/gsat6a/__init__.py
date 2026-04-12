"""
GSAT-6A Failure Analysis Module

Demonstrates Aethelix's capability to diagnose the actual GSAT-6A
failure from March 26, 2018 using real-time telemetry simulation.

Usage:
  python -m gsat6a.live_simulation     # Run live telemetry analysis
  python -m gsat6a.visualization_3d    # Run 3D interactive visualization
"""

from gsat6a.live_simulation import GSAT6ASimulator

__all__ = ["GSAT6ASimulator"]
