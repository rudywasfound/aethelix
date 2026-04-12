"""
Residual and anomaly analysis for satellite telemetry.
Quantifies deviations between nominal and degraded scenarios to bridge raw data and causal inference.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from simulator.power import PowerTelemetry


@dataclass
class ResidualStats:
    """ Container for residual analysis results. """
    
    mean_deviation: Dict[str, float]  # Mean absolute deviation per metric
    max_deviation: Dict[str, float]   # Maximum deviation encountered
    onset_time: Dict[str, float]      # Time (hours) when deviation exceeds threshold
    severity_score: float             # Overall degradation severity (0-1)


class ResidualAnalyzer:
    """ Analyze deviations between nominal and degraded telemetry. """

    def __init__(self, deviation_threshold: float = 0.1):
        """Initialize analyzer with sensitivity threshold."""
        
        self.deviation_threshold = deviation_threshold

    def analyze(
        self, nominal: PowerTelemetry, degraded: PowerTelemetry
    ) -> ResidualStats:
        """Compute residual statistics between nominal and degraded scenarios."""
        
        # Define which metrics to analyze
        metrics = {
            "solar_input": (nominal.solar_input, degraded.solar_input),
            "battery_voltage": (nominal.battery_voltage, degraded.battery_voltage),
            "battery_charge": (nominal.battery_charge, degraded.battery_charge),
            "bus_voltage": (nominal.bus_voltage, degraded.bus_voltage),
        }

        # Storage for computed statistics
        mean_dev = {}
        max_dev = {}
        onset = {}

        # Compute statistics for each metric
        for name, (nom, deg) in metrics.items():
            residual = np.abs(deg - nom)
            
            mean_dev[name] = float(np.mean(residual))
            max_dev[name] = float(np.max(residual))

            threshold = self.deviation_threshold * np.mean(nom)
            exceeds = np.where(residual > threshold)[0]
            
            if len(exceeds) > 0:
                onset[name] = float(nominal.time[exceeds[0]] / 3600)
            else:
                onset[name] = float("inf")

        severity = self._compute_severity(mean_dev, max_dev, nominal)


        return ResidualStats(
            mean_deviation=mean_dev,
            max_deviation=max_dev,
            onset_time=onset,
            severity_score=severity,
        )

    def _compute_severity(
        self,
        mean_dev: Dict[str, float],
        max_dev: Dict[str, float],
        nominal: PowerTelemetry,
    ) -> float:
        """Compute overall degradation severity score (0-1)."""
        
        fractions = []
        
        for name in mean_dev.keys():
            if "voltage" in name:
                baseline = nominal.battery_voltage.mean() if "battery" in name else nominal.bus_voltage.mean()
            elif "solar" in name:
                baseline = nominal.solar_input.mean()
            else:  # charge
                baseline = 50.0

            frac = mean_dev[name] / (baseline if baseline > 0 else 1.0)
            fractions.append(frac)

        # Average fractional deviations across all metrics
        severity = np.clip(np.mean(fractions), 0, 1)
        return float(severity)

    def print_report(self, stats: ResidualStats):
        """Pretty-print residual analysis report for operators."""
        
        print("\nRESIDUAL ANALYSIS REPORT")

        # Overall severity at the top for quick decision making
        print(f"\nOverall Severity Score: {stats.severity_score:.2%}")
        
        # Mean deviations show typical magnitude of change
        print("\nMean Deviations:")
        for metric, value in sorted(stats.mean_deviation.items()):
            print(f"  {metric:20s}: {value:8.2f}")

        # Max deviations show worst-case impact
        print("\nMaximum Deviations:")
        for metric, value in sorted(stats.max_deviation.items()):
            print(f"  {metric:20s}: {value:8.2f}")

        # Onset times help operators understand fault timeline
        print("\nDegradation Onset Times (hours):")
        for metric, onset_h in sorted(stats.onset_time.items()):
            if np.isinf(onset_h):
                print(f"  {metric:20s}: No significant deviation detected")
            else:
                print(f"  {metric:20s}: {onset_h:6.2f}h")

        print("")


if __name__ == "__main__":
    # Quick test of residual analyzer
    from simulator.power import PowerSimulator

    sim = PowerSimulator(duration_hours=24)
    nominal = sim.run_nominal()
    degraded = sim.run_degraded(solar_degradation_hour=6.0, battery_degradation_hour=8.0)

    analyzer = ResidualAnalyzer(deviation_threshold=0.15)
    stats = analyzer.analyze(nominal, degraded)
    analyzer.print_report(stats)
