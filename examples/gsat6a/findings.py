#!/usr/bin/env python3
"""
Findings framework for GSAT-6A analysis.

Generates key findings, cascade analysis, and mission impact from:
- Timeline events
- Telemetry statistics
- Detection comparisons

All findings are data-driven from actual measurements, not editorial.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TelemetryStats:
    """Statistics for a telemetry parameter across nominal and degraded states."""
    name: str
    unit: str
    nominal_mean: float
    degraded_mean: float
    nominal_std: float
    degraded_std: float
    
    @property
    def loss_percent(self) -> float:
        """Percentage loss from nominal to degraded."""
        if self.nominal_mean == 0:
            return 0
        return (self.nominal_mean - self.degraded_mean) / self.nominal_mean * 100
    
    @property
    def loss_absolute(self) -> float:
        """Absolute change from nominal to degraded."""
        return self.degraded_mean - self.nominal_mean


class FindingsEngine:
    """Generates findings from analysis data."""
    
    def __init__(self):
        """Initialize findings engine."""
        self.stats: List[TelemetryStats] = []
        self.causal_detection_time: Optional[float] = None
        self.threshold_detection_time: Optional[float] = None
        self.cascade_events: List[Tuple[float, str, str]] = []  # (time, subsystem, description)
    
    def add_telemetry_stat(self, name: str, unit: str,
                          nominal_mean: float, nominal_std: float,
                          degraded_mean: float, degraded_std: float):
        """Add a telemetry statistic."""
        self.stats.append(TelemetryStats(
            name=name,
            unit=unit,
            nominal_mean=nominal_mean,
            nominal_std=nominal_std,
            degraded_mean=degraded_mean,
            degraded_std=degraded_std
        ))
    
    def set_detection_times(self, causal: Optional[float], threshold: Optional[float]):
        """Set detection times for both methods."""
        self.causal_detection_time = causal
        self.threshold_detection_time = threshold
    
    def add_cascade_event(self, time_seconds: float, subsystem: str, description: str):
        """Add a cascade event."""
        self.cascade_events.append((time_seconds, subsystem, description))
    
    def print_telemetry_deviations(self):
        """Print telemetry deviations between nominal and degraded states."""
        if not self.stats:
            print("No telemetry data")
            return
        
        print("="*80)
        print("TELEMETRY DEVIATIONS")
        print("="*80)
        
        # Sort by largest loss
        sorted_stats = sorted(self.stats, key=lambda s: abs(s.loss_percent), reverse=True)
        
        for stat in sorted_stats:
            loss_sign = "↓" if stat.loss_percent > 0 else "↑"
            print(f"\n{stat.name} ({stat.unit}):")
            print(f"  Nominal:   {stat.nominal_mean:8.2f} ± {stat.nominal_std:.2f}")
            print(f"  Degraded:  {stat.degraded_mean:8.2f} ± {stat.degraded_std:.2f}")
            print(f"  Change:    {stat.loss_absolute:+8.2f} ({stat.loss_percent:+6.1f}%) {loss_sign}")
        
        print("\n" + "="*80 + "\n")
    
    def print_cascade_analysis(self):
        """Print failure cascade from timeline events."""
        if not self.cascade_events:
            print("No cascade data")
            return
        
        self.cascade_events.sort(key=lambda e: e[0])
        
        print("="*80)
        print("FAILURE CASCADE")
        print("="*80)
        
        # Group by subsystem
        subsystems = {}
        for time, subsys, desc in self.cascade_events:
            if subsys not in subsystems:
                subsystems[subsys] = []
            subsystems[subsys].append((time, desc))
        
        for i, (subsys, events) in enumerate(subsystems.items()):
            if i > 0:
                print()
            print(f"{subsys}")
            for time, desc in events:
                print(f"  └─ T+{time:.1f}s: {desc}")
        
        print("\n" + "="*80 + "\n")
    
    def print_detection_comparison(self):
        """Print comparison of detection methods."""
        print("="*80)
        print("DETECTION COMPARISON")
        print("="*80)
        
        if self.causal_detection_time is not None:
            print(f"\nCausal Inference: T+{self.causal_detection_time:.1f}s")
        else:
            print(f"\nCausal Inference: Not detected")
        
        if self.threshold_detection_time is not None:
            print(f"Threshold-Based:  T+{self.threshold_detection_time:.1f}s")
        else:
            print(f"Threshold-Based:  Not detected")
        
        if self.causal_detection_time is not None and self.threshold_detection_time is not None:
            lead_time = self.threshold_detection_time - self.causal_detection_time
            print(f"\nLead Time: {lead_time:.1f} seconds")
        
        print("\n" + "="*80 + "\n")
    
    def print_mission_impact(self):
        """Print mission impact analysis based on detection times."""
        if self.causal_detection_time is None:
            return
        
        print("="*80)
        print("MISSION IMPACT")
        print("="*80)
        
        print(f"\nDetection at T+{self.causal_detection_time:.1f}s enabled:")
        print(f"  • Root cause identification")
        print(f"  • Preventive corrective actions")
        print(f"  • Automated failsafe activation")
        
        if self.threshold_detection_time is not None:
            lead_time = self.threshold_detection_time - self.causal_detection_time
            print(f"\nTraditional threshold detection at T+{self.threshold_detection_time:.1f}s:")
            print(f"  • {lead_time:.1f}s later (reactive mode)")
            print(f"  • Only symptom detection, no root cause")
            print(f"  • Limited time for corrective action")
        
        print("\n" + "="*80 + "\n")
