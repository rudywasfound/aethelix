"""
Root cause ranking algorithms for multi-fault diagnosis.
Infers likely causes from telemetry deviations using Bayesian reasoning over a causal graph.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from simulator.power import PowerTelemetry
from causal_graph.graph_definition import CausalGraph


@dataclass
class RootCauseHypothesis:
    """ Ranked hypothesis for a root cause diagnosis. """

    name: str                        # Root cause name (e.g., "solar_degradation")
    probability: float               # Posterior probability this is the cause (0-1, sums to 1.0)
    evidence: List[str]              # Observable deviations supporting this hypothesis
    mechanism: str                   # Explanation of causal mechanism
    confidence: float                # Confidence in this hypothesis (0-1, independent of probability)
    causal_paths: List[List[str]] = None  # Causal chains from root cause to observables
    recommendations: Dict[str, str] = None # Actionable steps for operators


class RootCauseRanker:
    """
    Infer and rank root causes using a causal graph.
    Identifies deviations, traces them back to roots, and ranks by probability and confidence.
    """

    def __init__(self, graph: CausalGraph):
        """
        Initialize ranker with causal graph.

        Args:
            graph: CausalGraph instance containing domain knowledge
        """

        self.graph = graph

        # Mapping from physical quantity names to observable node names
        self.observables_map = {
            "solar_input":       "solar_input_measured",
            "battery_voltage":   "battery_voltage_measured",
            "battery_charge":    "battery_charge_measured",
            "bus_voltage":       "bus_voltage_measured",
            "solar_panel_temp":  "solar_panel_temp_measured",
            "battery_temp":      "battery_temp_measured",
            "payload_temp":      "payload_temp_measured",
            "bus_current":       "bus_current_measured",
            # ADCS
            "pointing_error":    "pointing_error_measured",
            "wheel_speed":       "wheel_speed_measured",
            "wheel_current":     "wheel_current_measured",
            "gyro_bias":         "gyro_bias_observed",
            # Comms
            "downlink_power":    "downlink_power_measured",
            "ber":               "ber_measured",
            "transponder_temp":  "transponder_temp_measured",
            # OBC
            "cpu_load":          "cpu_load_measured",
            "memory_usage":      "memory_usage_measured",
            "reboot_count":      "reset_count_measured",
            # Propulsion
            "tank_pressure":     "tank_pressure_measured",
            "thruster_temp":     "thruster_temp_measured",
        }

        self._expected_evidence: Dict[str, List[str]] = {
            # EPS
            "solar_degradation":          ["solar_input", "battery_charge", "bus_voltage", "battery_voltage"],
            "battery_aging":              ["battery_voltage", "battery_charge", "bus_voltage"],
            "battery_thermal":            ["battery_voltage", "battery_charge", "battery_temp"],
            "sensor_bias":                ["battery_voltage", "battery_charge"],
            "pcdu_regulator_failure":     ["bus_voltage", "bus_current", "payload_temp"],
            
            # TCS
            "panel_insulation_degradation": ["solar_panel_temp", "battery_temp"],
            "battery_heatsink_failure":   ["battery_temp", "bus_current"],
            "payload_radiator_degradation": ["payload_temp"],
            
            # ADCS
            "wheel_friction":             ["wheel_current", "pointing_error"],
            "gyro_drift":                 ["gyro_bias", "pointing_error"],
            "magnetorquer_anomaly":       ["wheel_speed"],
            
            # COMMS
            "transponder_fault":          ["downlink_power", "transponder_temp"],
            "antenna_pointing_error":     ["downlink_power", "ber"],
            "ber_spike":                  ["ber"],
            
            # OBC
            "memory_corruption":          ["memory_usage", "cpu_load"],
            "watchdog_reset_fault":       ["reboot_count"],
            "software_exception":         ["cpu_load"],
            
            # PROP
            "thruster_valve_fault":       ["thruster_temp"],
            "fuel_pressure_anomaly":      ["tank_pressure"],
        }

        # Fault onset tracker for lead-time calculation
        self._onset_timestamps: Dict[str, float] = {}
        
        # Sensor sticky-fault history (count of consecutive NaNs/Zeros)
        self._sensor_dead_counts: Dict[str, int] = {}



    def analyze(
        self,
        nominal,
        degraded,
        deviation_threshold: float = 0.15,
    ) -> List[RootCauseHypothesis]:
        """
        Analyze deviations and rank root causes.

        Args:
            nominal: Healthy telemetry (baseline for comparison)
            degraded: Faulty telemetry (what we're diagnosing)
            deviation_threshold: Fractional threshold for flagging an anomaly.

        Returns:
            Sorted list of root cause hypotheses, ranked by probability (highest first).
        """

        
        orbital_phase = getattr(degraded, 'orbital_phase', [0.0])[0] if hasattr(degraded, 'orbital_phase') else 0.5
        anomalies = self._detect_anomalies(nominal, degraded, deviation_threshold, orbital_phase=orbital_phase)
        return self.analyze_anomalies(anomalies)

    def analyze_anomalies(self, anomalies: Dict[str, float]) -> List[RootCauseHypothesis]:
        """
        Rank root causes given a pre-computed dictionary of anomaly severities.
        """
       
        root_cause_scores: Dict[str, float] = {}
        root_cause_evidence: Dict[str, List[str]] = {}
        root_cause_paths: Dict[str, List] = {}

        for observable, severity in anomalies.items():
            contributing_causes, cause_paths = self._trace_back_to_roots(
                observable, severity, anomalies
            )
            
            for cause_name, cause_score in contributing_causes.items():
                if cause_name not in root_cause_scores:
                    root_cause_scores[cause_name] = 0.0
                    root_cause_evidence[cause_name] = []
                    root_cause_paths[cause_name] = []

                root_cause_scores[cause_name] += cause_score
                root_cause_evidence[cause_name].append(f"{observable} deviation")
                if cause_name in cause_paths:
                    root_cause_paths[cause_name].extend(cause_paths[cause_name])

        # normalise raw scores to posteriors
        total_score = sum(root_cause_scores.values())
        if total_score == 0:
            return []

        # compute normalised posteriors first (needed by confidence)
        posteriors: Dict[str, float] = {
            c: s / total_score for c, s in root_cause_scores.items()
        }

        # we sort causes by posterior so that we can compute the margin between rank-1 and rank-2
        sorted_causes = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        top_posterior   = sorted_causes[0][1] if len(sorted_causes) >= 1 else 0.0
        second_posterior = sorted_causes[1][1] if len(sorted_causes) >= 2 else 0.0

        hypotheses: List[RootCauseHypothesis] = []
        for cause_name, probability in posteriors.items():
            mechanism = self._explain_mechanism(
                cause_name, root_cause_evidence[cause_name], anomalies
            )
            confidence = self._compute_confidence(
                cause_name=cause_name,
                evidence=root_cause_evidence[cause_name],
                anomalies=anomalies,
                posterior=probability,
                top_posterior=top_posterior,
                second_posterior=second_posterior,
            )

            # Recommendations Engine
            recommendations = self.get_recommendations(cause_name, confidence)

            hypotheses.append(
                RootCauseHypothesis(
                    name=cause_name,
                    probability=probability,
                    evidence=root_cause_evidence[cause_name],
                    mechanism=mechanism,
                    confidence=confidence,
                    causal_paths=root_cause_paths.get(cause_name, []),
                    recommendations=recommendations,
                )
            )

        hypotheses.sort(key=lambda h: h.probability, reverse=True)
        return hypotheses



    def _detect_anomalies(
        self,
        nominal,
        degraded,
        threshold: float,
        orbital_phase: float = 0.5,
    ) -> Dict[str, float]:
        """
        Detect which observables deviate from nominal with direction and context awareness.
        """

        anomalies: Dict[str, float] = {}
        
        # Predicted eclipse window: 0.42 <= orbital_phase <= 0.58
        is_eclipse = 0.42 <= orbital_phase <= 0.58

        # Define all candidate channels (collected from available attributes)
        candidate_channels = [
            # EPS
            "solar_input", "battery_voltage", "battery_charge", "bus_voltage", 
            # TCS
            "battery_temp", "solar_panel_temp", "payload_temp", "bus_current",
            # ADCS
            "pointing_error", "wheel_speed", "wheel_current", "gyro_bias",
            # COMMS
            "downlink_power", "ber", "transponder_temp",
            # OBC
            "cpu_load", "memory_usage", "reboot_count",
            # PROP
            "tank_pressure", "thruster_temp"
        ]

        for name in candidate_channels:
            if not hasattr(degraded, name) or not hasattr(nominal, name):
                continue
                
            deg_values = getattr(degraded, name)
            nom_values = getattr(nominal, name)

            # Sensor Fault Detection (3+ consecutive zeros or NaNs)
            latest_val = deg_values[-1] if len(deg_values) > 0 else np.nan
            if np.isnan(latest_val) or latest_val == 0.0:
                self._sensor_dead_counts[name] = self._sensor_dead_counts.get(name, 0) + 1
            else:
                self._sensor_dead_counts[name] = 0

            if self._sensor_dead_counts[name] >= 3:
                continue

            # Eclipse Awareness
            if is_eclipse and name in ["solar_input", "solar_panel_temp"]:
                continue

            # Direction-Aware Deviation
            deg_mean = np.nanmean(deg_values)
            nom_mean = np.nanmean(nom_values)
            residual = deg_mean - nom_mean
            
            if name == "bus_voltage" and residual > 0:
                continue
                
            fractional_dev = abs(residual) / (nom_mean if nom_mean != 0 else 1.0)

            if fractional_dev > threshold:
                severity = np.clip(fractional_dev / 0.5, 0.0, 1.0)
                anomalies[name] = severity

        return anomalies

    def get_recommendations(self, cause_name: str, confidence: float) -> Dict[str, str]:
        """
        Generate 3-tier actionable recommendations based on fault type and confidence.
        """
        
        if confidence < 20.0: return {} 

        recs = {
            "solar_degradation": {
                "immediate": "Disable non-critical secondary payloads to reduce load.",
                "short_term": "Schedule a detailed solar array IV-curve sweep.",
                "escalation": "If battery SOC < 40%, initiate low-power safe mode."
            },
            "pcdu_regulator_failure": {
                "immediate": "Command switch to redundant PCDU regulator string B.",
                "short_term": "Analyze thermal telemetry for regulator board hot spots.",
                "escalation": "If bus voltage < 26.5V, prepare for emergency battery direct-connect."
            },
            "wheel_friction": {
                "immediate": "Increase wheel heater setpoint by 5C to thin lubricant.",
                "short_term": "Switch attitude control to magnetic-only desaturation mode.",
                "escalation": "If wheel current > 0.8A, command wheel shutdown and use thrusters."
            },
            "memory_corruption": {
                "immediate": "Initiate task-level reset for affected service.",
                "short_term": "Perform full memory scrub and checksum validation.",
                "escalation": "If SEU frequency > 5/hour, command full system cold reboot."
            }
        }
        
        default = {
            "immediate": "Monitor relevant telemetry channels at high sample rate.",
            "short_term": "Review historical trend data for similar signatures.",
            "escalation": "Consult subsystem domain expert if confidence exceeds 60%."
        }
        
        return recs.get(cause_name, default)



    def _trace_back_to_roots(
        self,
        observable: str,
        severity: float,
        anomalies: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, list]]:
        """
        Trace from observable back to root causes via the causal graph.
        """

        observable_node = self.observables_map.get(observable, observable)
        weighted_results = self.graph.get_weighted_paths_to_root(observable_node)

        root_scores: Dict[str, float] = {}
        root_paths:  Dict[str, list]  = {}

        for path, path_strength in weighted_results:
            root_cause = path[0]

            if root_cause not in root_scores:
                root_scores[root_cause] = 0.0
                root_paths[root_cause]  = []

            consistency = self._check_consistency(root_cause, anomalies)
            
            # Weighted scoring: 
            # path_strength (physical coupling) * severity (magnitude) * consistency (pattern match)
            # We use consistency with a baseline of 0.4 to ensure we don't zero out early detections
            score = path_strength * severity * (0.4 + 0.6 * consistency)
            
            # Unique Path Bonus: 
            # If this root cause is the only path to the observable, it gets a 20% boost.
            # This helps distinguish solar failures from battery failures sharing charge symptoms.
            if len(weighted_results) == 1:
                score *= 1.2

            # Use additive scoring for disjoint paths (converging mechanisms) 
            # while ensuring we don't exceed 1.0 for a single observable-root coupling
            root_scores[root_cause] = min(1.0, root_scores[root_cause] + score)
            root_paths[root_cause].append(path)

        return root_scores, root_paths



    def _check_consistency(
        self,
        root_cause: str,
        anomalies: Dict[str, float],
    ) -> float:
        """
        Fraction of expected anomalies that were actually observed.
        """

        if root_cause not in self._expected_evidence:
            return 0.5

        expected = self._expected_evidence.get(root_cause, [])
        if not expected:
            return 0.5

        observed = set(anomalies.keys())
        matches = len([e for e in expected if e in observed])
        missing = len(expected) - matches

        # Weighted Support Model:
        # High reward for confirmed symptoms, gentle penalty for missing ones.
        # Reduced missing multiplier from 0.3 to 0.15 for better early-phase detection.
        score = matches / (matches + 0.15 * missing) if (matches + missing) > 0 else 0.5
        return score



    def _explain_mechanism(
        self,
        root_cause: str,
        evidence: List[str],
        anomalies: Dict[str, float],
    ) -> str:
        """Generate plain-text explanation for operators."""

        explanations = {
            "solar_degradation": (
                "Reduced solar input is propagating through the power subsystem. "
                "This suggests solar panel degradation or shadowing, which reduces "
                "available power for charging the battery."
            ),
            "battery_aging": (
                "Battery voltage and charge deviations indicate internal degradation. "
                "This suggests increased internal resistance or cell aging, reducing "
                "charging efficiency and available capacity."
            ),
            "battery_thermal": (
                "Battery voltage droop under nominal load suggests thermal stress. "
                "Elevated temperature is degrading electrochemical performance "
                "and increasing internal losses."
            ),
            "sensor_bias": (
                "Anomalies in voltage and charge measurements may be due to sensor "
                "calibration drift rather than actual physical degradation. "
                "Cross-check with other subsystems before taking action."
            ),
            "panel_insulation_degradation": (
                "Elevated solar panel temperature indicates loss of thermal insulation "
                "or radiator fouling. This reduces panel efficiency and increases "
                "heat-induced stress on power electronics."
            ),
            "battery_heatsink_failure": (
                "High battery temperature with elevated current draw indicates the "
                "primary thermal management system has failed. This accelerates battery "
                "aging and risks thermal runaway if not corrected."
            ),
            "payload_radiator_degradation": (
                "Elevated payload temperature indicates radiator coating degradation "
                "or micrometeorite damage. Payload must operate at reduced power to "
                "avoid thermal shutdown."
            ),
            "pcdu_regulator_failure": (
                "A collapse in regulated bus voltage and current indicates a PCDU "
                "regulator failure. This is a critical electrical fault that may "
                "permanently disable payloads dependent on the regulated bus."
            ),
        }

        base = explanations.get(root_cause, "Unknown root cause mechanism.")
        if evidence:
            return f"{base}\nEvidence: {'; '.join(evidence)}"
        return base



    def _compute_confidence(
        self,
        cause_name: str,
        evidence: List[str],
        anomalies: Dict[str, float],
        posterior: float,
        top_posterior: float,
        second_posterior: float,
    ) -> float:
        """
        Compute calibrated confidence for a root-cause hypothesis.
        Uses a multiplicative model factoring in posterior probability, symptoms consistency,
        evidence saturation, and the margin between top hypotheses.
        """

        # 1. Model Posterior
        posterior_factor = float(np.sqrt(np.clip(posterior, 0.0, 1.0)))

        # Path Consistency

        consistency = self._check_consistency(cause_name, anomalies)
        # Consistency alone ranging 0–1 is fine; use it directly.
        consistency_factor = consistency

        # Evidence Saturation

        expected_count = len(self._expected_evidence.get(cause_name, []))
        # Number of *unique* observed channels that match expected evidence
        observed_matching = len(
            set(self._expected_evidence.get(cause_name, [])) & set(anomalies.keys())
        )
        if expected_count > 0:
            saturation = observed_matching / expected_count
        else:
            saturation = 0.5  # unknown cause: neutral

        # Apply a soft penalty for low saturation: sqrt keeps it non-zero
        # even with partial evidence, but penalises incompleteness.
        saturation_factor = float(np.sqrt(np.clip(saturation, 0.0, 1.0)))

        # Posterior Margin

        # margin = how much more probable is this hypothesis than the runner-up?
        # Range: [0, 1].  A tie (margin=0) → margin_factor=0 (maximally uncertain).
        if top_posterior > 0:
            margin = np.clip(
                (top_posterior - second_posterior) / top_posterior, 0.0, 1.0
            )
        else:
            margin = 0.0

        # Use a softer sqrt so partial separation still gives some confidence
        margin_factor = float(np.sqrt(margin))

        # Combine
        # Switched to a less aggressive combination to populate higher confidence bins.
        raw_confidence = (
            0.4 * posterior_factor +
            0.2 * consistency_factor +
            0.2 * saturation_factor +
            0.2 * margin_factor
        )

        # Baseline floor: any hypothesis with high posterior should have some confidence
        confidence = np.clip(raw_confidence, posterior * 0.1, 1.0)

        return float(np.clip(confidence, 0.0, 1.0))



    def print_report(self, hypotheses: List[RootCauseHypothesis]):
        """Pretty-print root cause analysis report for operators."""

        print("\nROOT CAUSE RANKING ANALYSIS")

        if not hypotheses:
            print("\nNo significant root causes detected.")
            return

        print("\nMost Likely Root Causes (by posterior probability):\n")
        for rank, hyp in enumerate(hypotheses, 1):
            print(
                f"{rank}. {hyp.name:25s} "
                f"P={hyp.probability:6.1%}  "
                f"Confidence={hyp.confidence:5.1%}"
            )


        print("DETAILED EXPLANATIONS:\n")

        for hyp in hypotheses:
            print(f"• {hyp.name} (P={hyp.probability:.1%})")

            if hyp.causal_paths:
                unique_paths = list(set([tuple(p) for p in hyp.causal_paths]))
                if unique_paths:
                    print(f"  Causal Paths:")
                    for path in unique_paths[:3]:
                        path_str = " → ".join(reversed(path))
                        print(f"    {path_str}")

            print(f"  Evidence: {', '.join(hyp.evidence)}")
            print(f"  Mechanism: {hyp.mechanism}")
            print()

        print("")


if __name__ == "__main__":
    from simulator.power import PowerSimulator

    sim = PowerSimulator(duration_hours=24)
    nominal = sim.run_nominal()
    degraded = sim.run_degraded(
        solar_degradation_hour=6.0,
        battery_degradation_hour=8.0,
    )

    graph = CausalGraph()
    ranker = RootCauseRanker(graph)
    hypotheses = ranker.analyze(nominal, degraded, deviation_threshold=0.15)
    ranker.print_report(hypotheses)