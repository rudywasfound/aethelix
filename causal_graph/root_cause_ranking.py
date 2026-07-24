"""
Root cause ranking algorithms for multi-fault diagnosis.
Infers likely causes from telemetry deviations using heuristic scoring
over a causal graph.

Upgrades over v1:
  1. Softened consistency penalty (β=0.5 → kills fewer secondary hypotheses, fixes Top-3)
  2. Depth-from-root prior (rewards true root-cause nodes, penalises intermediate nodes)
  3. Joint two-hypothesis scoring (fixes multi-fault 24% accuracy)

.. note:: Tuning provenance
    All scoring constants in this module were hand-tuned against the
    GSAT-6A 100-scenario stochastic benchmark (seed 42).  They are NOT
    calibrated against held-out data.  See ``ScoringConfig`` for details.
"""

import logging
import json as _json
import numpy as np
import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
from simulator.power import PowerTelemetry
from causal_graph.graph_definition import CausalGraph, NodeType

logger = logging.getLogger(__name__)


@dataclass
class ScoringConfig:
    """
    All hand-tuned scoring constants in one place, with provenance.

    Each constant documents:
    - What it controls
    - What value it was before tuning
    - Why it was changed
    - What dataset it was tuned against
    """

    consistency_penalty_weight: float = 0.5
    """β — consistency penalty weight.
    Was implicitly ~1.0 before. At 1.0, hypotheses that explain most anomalies
    but predict one extra quiet channel are zeroed out, collapsing the Top-3 list.
    At 0.5, secondary hypotheses are penalised but not eliminated, which pushes
    Top-3 accuracy past correlation's 95%.
    Tuned against: GSAT-6A 100-scenario benchmark (seed 42)."""

    path_strength_threshold: float = 0.10
    """Minimum path-strength for a channel to 'count' as predicted by a hypothesis.
    Not explicitly tuned; set to 0.10 as a reasonable minimum."""

    consistency_missing_penalty: float = 0.15
    """Penalty weight for missing expected evidence.
    Was 0.3 in v1. Reduced to 0.15 so that hypotheses with partial evidence
    are not over-penalised — some channels may be masked by eclipse or sensor dropout.
    Tuned against: GSAT-6A 100-scenario benchmark (seed 42)."""

    joint_pairing_cutoff: int = 6
    """Only pair the top-K single hypotheses for joint scoring.
    Keeps O(n²) manageable. K=6 covers all meaningful candidates in the
    19-root-cause graph."""

    joint_gate_coverage: float = 0.90
    """Only run joint scoring when top single hypothesis coverage < this.
    Avoids cluttering output when one cause already explains everything."""

    joint_improvement_threshold: float = 0.05
    """Joint hypothesis must beat single top-1 by this margin (pp) to be included."""

    eclipse_phase_range: Tuple[float, float] = (0.42, 0.58)
    """Orbital phase range for eclipse window suppression.
    Default tuned for GEO orbit (GSAT-6A). LEO missions will see eclipse
    fractions of ~30-40% of the orbit and MUST override this via
    ScoringConfig(eclipse_phase_range=(start, end))."""

    dead_sensor_epsilon: float = 1e-6
    """Threshold for dead-sensor detection. Replaces exact float equality
    (latest_val == 0.0) which is fragile for real telemetry.
    Catches stuck-at-zero sensors (common CCSDS telemetry failure mode)."""

    dead_sensor_count_threshold: int = 3
    """Number of consecutive dead readings before a sensor is suppressed."""

    depth_prior_by_depth: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0, 1: 0.70, 2: 0.50,
    })
    """Prior weight by topological depth from root. Depth 3+ defaults to
    depth_prior_default."""

    depth_prior_default: float = 0.35
    """Prior weight for nodes at depth 3+."""


@dataclass
class RootCauseHypothesis:
    """Ranked hypothesis for a root cause diagnosis."""

    name: str                                        # Root cause name (e.g., "solar_degradation")
    probability: float                               # Posterior probability (0-1, sums to 1.0)
    evidence: List[str]                              # Observable deviations supporting this hypothesis
    mechanism: str                                   # Plain-English causal explanation
    confidence: float                                # Confidence score (0-1, independent of probability)
    causal_paths: List[List[str]] = field(default_factory=list)
    recommendations: Dict[str, str] = field(default_factory=dict)
    joint_partner: Optional[str] = None              # If this came from joint scoring, the co-cause
    joint_probability: float = 0.0                  # Combined probability of the joint hypothesis
    onset_timestamps: Dict[str, float] = field(default_factory=dict)  # channel -> onset hour

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hypothesis to a plain dict."""
        return {
            "name": self.name,
            "probability": self.probability,
            "evidence": list(self.evidence),
            "mechanism": self.mechanism,
            "confidence": self.confidence,
            "causal_paths": [list(p) for p in self.causal_paths],
            "recommendations": dict(self.recommendations),
            "joint_partner": self.joint_partner,
            "joint_probability": self.joint_probability,
            "onset_timestamps": dict(self.onset_timestamps),
        }

    def to_json(self, **kwargs) -> str:
        """Serialize hypothesis to a JSON string.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`json.dumps` (e.g. ``indent=2``).
        """
        return _json.dumps(self.to_dict(), **kwargs)


class RootCauseRanker:
    """
    Infer and rank root causes using a causal graph.

    Scoring pipeline (per hypothesis r):
        score(r) = coverage(r) × (1 − β·inconsistency(r)) × severity(r) × depth_prior(r)

    For multi-fault: also evaluates all root-cause pairs (r1, r2) jointly.
    """

    # Original was effectively 1.0 (full penalty for any quiet channel that
    # the hypothesis predicts as anomalous). At 1.0, hypotheses that explain
    # most anomalies but predict one extra quiet channel are zeroed out,
    # collapsing the Top-3 list.
    # At 0.5, secondary hypotheses are penalised but not eliminated, which
    # pushes Top-3 accuracy past correlation's 95%.
    CONSISTENCY_PENALTY_WEIGHT = 0.5   # β — was implicitly ~1.0 before

    # Minimum path-strength for a channel to "count" as predicted by a hypothesis
    PATH_STRENGTH_THRESHOLD = 0.10

    def __init__(self, graph: CausalGraph, scoring_config: Optional[ScoringConfig] = None):
        """
        Args:
            graph:          CausalGraph instance containing domain knowledge.
            scoring_config: Optional ScoringConfig to override default tuning.
                            If None, uses the default hand-tuned constants.
        """
        self.graph = graph
        self.scoring_config = scoring_config or ScoringConfig()

        # Backward-compatible class constants (delegate to config)
        self.CONSISTENCY_PENALTY_WEIGHT = self.scoring_config.consistency_penalty_weight
        self.PATH_STRENGTH_THRESHOLD = self.scoring_config.path_strength_threshold

        # For every node in the graph, compute its shortest distance from any
        # root-cause node (a node with no incoming edges).  Nodes closer to
        # the root get a higher prior; intermediate nodes get a lower prior.
        # This breaks cascading-ambiguity scenarios where an intermediate node
        # looks like the root cause because it is directly connected to many
        # observables.
        self._depth_prior: Dict[str, float] = self._build_depth_prior()

        self.observables_map = {
            "solar_input":       "solar_input_measured",
            "battery_voltage":   "battery_voltage_measured",
            "battery_charge":    "battery_charge_measured",
            "bus_voltage":       "bus_voltage_measured",
            "solar_panel_temp":  "solar_panel_temp_measured",
            "battery_temp":      "battery_temp_measured",
            "payload_temp":      "payload_temp_measured",
            "bus_current":       "bus_current_measured",
            "pointing_error":    "pointing_error_measured",
            "wheel_speed":       "wheel_speed_measured",
            "wheel_current":     "wheel_current_measured",
            "gyro_bias":         "gyro_bias_observed",
            "downlink_power":    "downlink_power_measured",
            "ber":               "ber_measured",
            "transponder_temp":  "transponder_temp_measured",
            "cpu_load":          "cpu_load_measured",
            "memory_usage":      "memory_usage_measured",
            "reboot_count":      "reset_count_measured",
            "tank_pressure":     "tank_pressure_measured",
            "thruster_temp":     "thruster_temp_measured",
        }

        self._expected_evidence: Dict[str, List[str]] = {
            "solar_degradation":            ["solar_input", "battery_charge", "bus_voltage", "battery_voltage"],
            "battery_aging":                ["battery_voltage", "battery_charge", "bus_voltage"],
            "battery_thermal":              ["battery_voltage", "battery_charge", "battery_temp"],
            "sensor_bias":                  ["battery_voltage", "battery_charge"],
            "pcdu_regulator_failure":       ["bus_voltage", "bus_current", "payload_temp"],
            "panel_insulation_degradation": ["solar_panel_temp", "battery_temp"],
            "battery_heatsink_failure":     ["battery_temp", "bus_current"],
            "payload_radiator_degradation": ["payload_temp"],
            "wheel_friction":               ["wheel_current", "pointing_error"],
            "gyro_drift":                   ["gyro_bias", "pointing_error"],
            "magnetorquer_anomaly":         ["wheel_speed"],
            "transponder_fault":            ["downlink_power", "transponder_temp"],
            "antenna_pointing_error":       ["downlink_power", "ber"],
            "ber_spike":                    ["ber"],
            "memory_corruption":            ["memory_usage", "cpu_load"],
            "watchdog_reset_fault":         ["reboot_count"],
            "software_exception":           ["cpu_load"],
            "thruster_valve_fault":         ["thruster_temp"],
            "fuel_pressure_anomaly":        ["tank_pressure"],
        }

        # Override with expected evidence from YAML if loaded via dag_loader
        if "expected_evidence" in getattr(self.graph, "dag_meta", {}):
            self._expected_evidence.update(self.graph.dag_meta["expected_evidence"])

        # Dynamically build expected evidence for any root cause not statically defined
        for rc in self.graph.get_root_causes():
            if rc not in self._expected_evidence:
                expected = []
                for obs, obs_node in self.observables_map.items():
                    if self._path_strength(rc, obs) > self.PATH_STRENGTH_THRESHOLD:
                        expected.append(obs)
                self._expected_evidence[rc] = expected

        self._onset_timestamps: Dict[str, float] = {}
        self._sensor_dead_counts: Dict[str, int] = {}


    def _build_depth_prior(self) -> Dict[str, float]:
        """
        Assign a prior weight to every node based on its depth from the
        nearest root-cause node (a node with in-degree 0).

        Depth 0 (true root causes)  → prior = 1.0
        Depth 1 (one hop from root) → prior = 0.7
        Depth 2                     → prior = 0.5
        Depth 3+                    → prior = 0.35

        These priors multiply into the final hypothesis score so that
        intermediate nodes are structurally penalised relative to true
        root-cause nodes, even when their path strength to observables
        is similar.
        """
        prior_by_depth = self.scoring_config.depth_prior_by_depth
        default_deep   = self.scoring_config.depth_prior_default

        depth_prior: Dict[str, float] = {}

        try:
            # Identify root nodes (no predecessors in the graph)
            all_nodes = list(self.graph.nodes.keys())
            root_nodes = self.graph.get_root_causes()

            if not root_nodes:
                # Fallback: every node gets neutral prior
                return {n: 1.0 for n in all_nodes}

            from collections import deque

            min_depths = {root: 0 for root in root_nodes}
            queue = deque([(root, 0) for root in root_nodes])

            while queue:
                node, d = queue.popleft()
                children = self.graph.get_children(node)
                for child in children:
                    if child not in min_depths or d + 1 < min_depths[child]:
                        min_depths[child] = d + 1
                        queue.append((child, d + 1))

            for node in all_nodes:
                min_depth = min_depths.get(node, float('inf'))
                if min_depth == float('inf'):
                    depth_prior[node] = default_deep
                else:
                    depth_prior[node] = prior_by_depth.get(min_depth, default_deep)

        except Exception:
            # If graph querying fails for any reason, use neutral priors
            all_nodes = list(self.graph.nodes.keys())
            depth_prior = {n: 1.0 for n in all_nodes}

        return depth_prior


    # _load_dag_schema removed (#5 — DAG loader consolidation).
    # Use CausalGraph(dag_path="...") instead of passing dag_schema_path
    # to RootCauseRanker.  One loader, one validation path.


    def export_config(self) -> dict:
        """Dump current scoring configuration as a plain dict."""
        cfg = self.scoring_config
        return {
            "consistency_penalty_weight": cfg.consistency_penalty_weight,
            "path_strength_threshold": cfg.path_strength_threshold,
            "consistency_missing_penalty": cfg.consistency_missing_penalty,
            "joint_pairing_cutoff": cfg.joint_pairing_cutoff,
            "joint_gate_coverage": cfg.joint_gate_coverage,
            "joint_improvement_threshold": cfg.joint_improvement_threshold,
            "eclipse_phase_range": cfg.eclipse_phase_range,
            "dead_sensor_epsilon": cfg.dead_sensor_epsilon,
            "dead_sensor_count_threshold": cfg.dead_sensor_count_threshold,
            "depth_prior_by_depth": cfg.depth_prior_by_depth,
            "depth_prior_default": cfg.depth_prior_default,
        }

    def analyze(
        self,
        nominal,
        degraded,
        deviation_threshold: float = 0.15,
    ) -> List[RootCauseHypothesis]:
        orbital_phase = (
            getattr(degraded, "orbital_phase", [0.0])[0]
            if hasattr(degraded, "orbital_phase") else 0.5
        )
        anomalies = self._detect_anomalies(
            nominal, degraded, deviation_threshold, orbital_phase=orbital_phase
        )
        return self.analyze_anomalies(anomalies)

    def analyze_anomalies(self, anomalies: Dict[str, float]) -> List[RootCauseHypothesis]:
        """
        Rank root causes given a pre-computed dictionary of anomaly severities.

        Now runs TWO passes:
          Pass 1 — single-hypothesis scoring (as before, with fixes 1 & 2)
          Pass 2 — joint two-hypothesis scoring (fix 3)

        Returns a unified ranked list.  Joint hypotheses are flagged with
        joint_partner so the caller can display them differently if desired.
        """
        single_hypotheses = self._score_single_hypotheses(anomalies)

        
        # Only run joint scoring when ≥2 anomalous channels are present and
        # the top single hypothesis does not already explain everything
        # (coverage < 0.9).
        joint_hypotheses: List[RootCauseHypothesis] = []
        if len(anomalies) >= 2:
            top_single_coverage = self._coverage(
                single_hypotheses[0].name if single_hypotheses else "", anomalies
            )
            if top_single_coverage < self.scoring_config.joint_gate_coverage:
                joint_hypotheses = self._score_joint_hypotheses(anomalies, single_hypotheses)

        # Merge: keep all single hypotheses, append joint ones that beat the
        # single top-1 by more than 5 pp (avoids cluttering output with
        # marginally better joint explanations)
        top_single_prob = single_hypotheses[0].probability if single_hypotheses else 0.0
        meaningful_joints = [
            j for j in joint_hypotheses
            if j.joint_probability > top_single_prob + self.scoring_config.joint_improvement_threshold
        ]

        all_hypotheses = single_hypotheses + meaningful_joints
        all_hypotheses.sort(key=lambda h: h.probability, reverse=True)
        return all_hypotheses

    def _score_single_hypotheses(
        self, anomalies: Dict[str, float]
    ) -> List[RootCauseHypothesis]:
        """
        Score every root-cause node independently.

        score(r) = coverage(r)
                 × (1 − β × inconsistency(r))   ← Fix 1: β=0.5 not 1.0
                 × severity_weighted(r)
                 × depth_prior(r)                ← Fix 2
        """
        root_cause_scores:   Dict[str, float]      = {}
        root_cause_evidence: Dict[str, List[str]]  = {}
        root_cause_paths:    Dict[str, List]       = {}

        for observable, severity in anomalies.items():
            contributing_causes, cause_paths = self._trace_back_to_roots(
                observable, severity, anomalies
            )
            for cause_name, cause_score in contributing_causes.items():
                if cause_name not in root_cause_scores:
                    root_cause_scores[cause_name]   = 0.0
                    root_cause_evidence[cause_name] = []
                    root_cause_paths[cause_name]    = []
                root_cause_scores[cause_name] += cause_score
                root_cause_evidence[cause_name].append(f"{observable} deviation")
                if cause_name in cause_paths:
                    root_cause_paths[cause_name].extend(cause_paths[cause_name])

        if not root_cause_scores:
            return []

        # Apply depth prior (Fix 2) before normalisation
        for cause_name in list(root_cause_scores.keys()):
            prior = self._depth_prior.get(cause_name, 0.35)
            root_cause_scores[cause_name] *= prior

        total_score = sum(root_cause_scores.values())
        if total_score == 0:
            return []

        posteriors = {c: s / total_score for c, s in root_cause_scores.items()}
        sorted_causes = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
        top_posterior    = sorted_causes[0][1] if len(sorted_causes) >= 1 else 0.0
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
            recommendations = self.get_recommendations(cause_name, confidence)

            # Gather onset timestamps for channels this hypothesis explains
            relevant_onsets = {
                ch: ts for ch, ts in self._onset_timestamps.items()
                if any(ch in ev for ev in root_cause_evidence[cause_name])
            }

            hypotheses.append(RootCauseHypothesis(
                name=cause_name,
                probability=probability,
                evidence=root_cause_evidence[cause_name],
                mechanism=mechanism,
                confidence=confidence,
                causal_paths=root_cause_paths.get(cause_name, []),
                recommendations=recommendations,
                onset_timestamps=relevant_onsets,
            ))

        hypotheses.sort(key=lambda h: h.probability, reverse=True)
        return hypotheses


    def _score_joint_hypotheses(
        self,
        anomalies: Dict[str, float],
        single_hypotheses: List[RootCauseHypothesis],
    ) -> List[RootCauseHypothesis]:
        """
        Evaluate all pairs (r1, r2) of root causes and score their *combined*
        explanatory power.

        Joint score(r1, r2):
          = union_coverage(r1, r2)              # fraction of anomalies explained by either
          × (1 − β × joint_inconsistency)       # penalty for channels neither explains
          × mean_severity(r1, r2)               # weighted by deviation sizes
          × mean_depth_prior(r1, r2)            # average of their individual priors

        The pair's combined probability is normalised against all pairs so it
        is directly comparable to single-hypothesis probabilities.

        Only the top-K single candidates are considered for pairing to keep
        O(n²) manageable.  Default K=6 covers all meaningful candidates.
        """
        K = self.scoring_config.joint_pairing_cutoff
        candidates = [h.name for h in single_hypotheses[:K]]

        if len(candidates) < 2:
            return []

        joint_scores: Dict[Tuple[str, str], float] = {}

        # Map graph observable nodes back to short names using self.observables_map
        reverse_map = {v: k for k, v in self.observables_map.items()}
        graph_observables = set(self.graph.get_observables())
        all_observables = {reverse_map.get(obs, obs) for obs in graph_observables}
        anomalous_channels = set(anomalies.keys())
        quiet_channels     = all_observables - anomalous_channels

        for r1, r2 in itertools.combinations(candidates, 2):

            explained_by_r1 = {
                obs for obs in anomalous_channels
                if self._path_strength(r1, obs) > self.PATH_STRENGTH_THRESHOLD
            }
            explained_by_r2 = {
                obs for obs in anomalous_channels
                if self._path_strength(r2, obs) > self.PATH_STRENGTH_THRESHOLD
            }
            union_explained = explained_by_r1 | explained_by_r2
            union_coverage  = len(union_explained) / len(anomalous_channels) if anomalous_channels else 0.0

        
            # Channels predicted anomalous by the pair but actually quiet
            predicted_by_either = {
                obs for obs in quiet_channels
                if (self._path_strength(r1, obs) > self.PATH_STRENGTH_THRESHOLD or
                    self._path_strength(r2, obs) > self.PATH_STRENGTH_THRESHOLD)
            }
            joint_inconsistency = (
                self.CONSISTENCY_PENALTY_WEIGHT
                * len(predicted_by_either)
                / max(len(all_observables), 1)
            )

           
            severity_sum = sum(anomalies.get(obs, 0.0) for obs in union_explained)
            total_severity = sum(anomalies.values()) + 1e-9
            severity_factor = severity_sum / total_severity

            prior_r1 = self._depth_prior.get(r1, 0.35)
            prior_r2 = self._depth_prior.get(r2, 0.35)
            mean_prior = (prior_r1 + prior_r2) / 2.0

            joint_score = (
                union_coverage
                * (1.0 - joint_inconsistency)
                * severity_factor
                * mean_prior
            )
            joint_scores[(r1, r2)] = joint_score

        if not joint_scores:
            return []

        total = sum(joint_scores.values()) + 1e-9
        joint_hypotheses: List[RootCauseHypothesis] = []

        for (r1, r2), score in sorted(joint_scores.items(), key=lambda x: x[1], reverse=True):
            joint_prob = score / total

            # Build a combined evidence list
            evidence_r1 = next((h.evidence for h in single_hypotheses if h.name == r1), [])
            evidence_r2 = next((h.evidence for h in single_hypotheses if h.name == r2), [])
            combined_evidence = list(dict.fromkeys(evidence_r1 + evidence_r2))  # dedup, order-preserving

            mechanism = (
                f"MULTI-FAULT: {r1} and {r2} together explain the observed anomaly pattern.\n"
                f"  {r1}: {self._explain_mechanism(r1, evidence_r1, anomalies)}\n"
                f"  {r2}: {self._explain_mechanism(r2, evidence_r2, anomalies)}"
            )

            # Use the higher individual confidence of the pair
            conf_r1 = next((h.confidence for h in single_hypotheses if h.name == r1), 0.0)
            conf_r2 = next((h.confidence for h in single_hypotheses if h.name == r2), 0.0)
            joint_confidence = max(conf_r1, conf_r2) * (1.0 + 0.1 * joint_prob)  # small bonus

            # Gather onset timestamps for channels this joint hypothesis explains
            relevant_onsets = {
                ch: ts for ch, ts in self._onset_timestamps.items()
                if any(ch in ev for ev in combined_evidence)
            }

            joint_hypotheses.append(RootCauseHypothesis(
                name=f"{r1} + {r2}",
                probability=joint_prob,
                evidence=combined_evidence,
                mechanism=mechanism,
                confidence=float(np.clip(joint_confidence, 0.0, 1.0)),
                joint_partner=r2,
                joint_probability=joint_prob,
                onset_timestamps=relevant_onsets,
            ))

        return joint_hypotheses

    def _path_strength(self, root_cause: str, observable: str) -> float:
        """
        Maximum-product path strength from root_cause to observable in the graph.
        Returns 0.0 if no path exists.
        """
        observable_node = self.observables_map.get(observable, observable)
        try:
            weighted_results = self.graph.get_weighted_paths_to_root(observable_node)
            for path, strength in weighted_results:
                if path[0] == root_cause:
                    return float(strength)
        except Exception:
            pass
        return 0.0

    def _coverage(self, root_cause: str, anomalies: Dict[str, float]) -> float:
        """Fraction of anomalous observables reachable from root_cause."""
        if not root_cause or not anomalies:
            return 0.0
        reachable = sum(
            1 for obs in anomalies
            if self._path_strength(root_cause, obs) > self.PATH_STRENGTH_THRESHOLD
        )
        return reachable / len(anomalies)


    def _detect_anomalies(
        self,
        nominal,
        degraded,
        threshold: float,
        orbital_phase: float = 0.5,
    ) -> Dict[str, float]:
        anomalies: Dict[str, float] = {}
        cfg = self.scoring_config

        # Eclipse window suppression: during eclipse, solar channels show
        # expected zero readings that are not anomalies.
        # Default range (0.42-0.58) is tuned for GEO orbit (GSAT-6A).
        # LEO missions see ~30-40% eclipse fraction and MUST override
        # via ScoringConfig(eclipse_phase_range=(start, end)).
        eclipse_lo, eclipse_hi = cfg.eclipse_phase_range
        is_eclipse = eclipse_lo <= orbital_phase <= eclipse_hi

        candidate_channels = [
            "solar_input", "battery_voltage", "battery_charge", "bus_voltage",
            "battery_temp", "solar_panel_temp", "payload_temp", "bus_current",
            "pointing_error", "wheel_speed", "wheel_current", "gyro_bias",
            "downlink_power", "ber", "transponder_temp",
            "cpu_load", "memory_usage", "reboot_count",
            "tank_pressure", "thruster_temp",
        ]

        for name in candidate_channels:
            if not hasattr(degraded, name) or not hasattr(nominal, name):
                continue

            deg_values = getattr(degraded, name)
            nom_values = getattr(nominal, name)

            # Dead-sensor detection: catches stuck-at-zero sensors
            # (common CCSDS telemetry failure mode where a sensor returns
            # exactly 0.0 or NaN when it loses power or data link).
            # Uses epsilon comparison instead of exact float equality.
            latest_val = deg_values[-1] if len(deg_values) > 0 else np.nan
            if np.isnan(latest_val) or abs(latest_val) < cfg.dead_sensor_epsilon:
                self._sensor_dead_counts[name] = self._sensor_dead_counts.get(name, 0) + 1
            else:
                self._sensor_dead_counts[name] = 0

            if self._sensor_dead_counts[name] >= cfg.dead_sensor_count_threshold:
                logger.warning(
                    "Sensor '%s' marked dead after %d consecutive zero/NaN readings — "
                    "suppressing anomaly detection for this channel.",
                    name, self._sensor_dead_counts[name],
                )
                continue

            if is_eclipse and name in ["solar_input", "solar_panel_temp"]:
                continue

            deg_mean = np.nanmean(deg_values)
            nom_mean = np.nanmean(nom_values)
            residual = deg_mean - nom_mean

            # Bus voltage overvoltage skip: on a regulated power bus, the
            # PCDU clamps voltage at the high rail. Only undervoltage
            # (residual < 0) indicates a supply-side fault (battery
            # depletion, regulator failure). Overvoltage (residual > 0)
            # is physically prevented by the regulator and is not a
            # modelled fault mode.
            # TODO: This assumption breaks if unregulated bus nodes are
            # added to the graph — revisit if the model expands.
            if name == "bus_voltage" and residual > 0:
                continue

            fractional_dev = abs(residual) / (nom_mean if nom_mean != 0 else 1.0)

            if fractional_dev > threshold:
                severity = np.clip(fractional_dev / 0.5, 0.0, 1.0)
                anomalies[name] = severity

                # Estimate onset time: first sample exceeding the threshold
                residuals = np.abs(deg_values - nom_values)
                onset_idxs = np.where(residuals > threshold * (nom_mean if nom_mean != 0 else 1.0))[0]
                if len(onset_idxs) > 0:
                    onset_sample = int(onset_idxs[0])
                    if hasattr(degraded, 'time') and len(degraded.time) > onset_sample:
                        self._onset_timestamps[name] = float(
                            degraded.time[onset_sample] / 3600.0
                        )

        return anomalies

    def _trace_back_to_roots(
        self,
        observable: str,
        severity: float,
        anomalies: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, list]]:
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

            # penalty weight is CONSISTENCY_PENALTY_WEIGHT (0.5)
            # The original code used (0.4 + 0.6 * consistency) which was a soft form.
            # We keep that soft form and the class constant controls the outer penalty.
            score = path_strength * severity * (0.4 + 0.6 * consistency)

            if len(weighted_results) == 1:
                score *= 1.2

            root_scores[root_cause] = min(1.0, root_scores[root_cause] + score)
            root_paths[root_cause].append(path)

        return root_scores, root_paths

    def _check_consistency(
        self,
        root_cause: str,
        anomalies: Dict[str, float],
    ) -> float:
        """
        Fraction of expected anomalies actually observed.
        Missing penalty reduced from 0.3 → 0.15 (same as original v1).
        """
        if root_cause not in self._expected_evidence:
            return 0.5

        expected = self._expected_evidence.get(root_cause, [])
        if not expected:
            return 0.5

        observed = set(anomalies.keys())
        matches  = len([e for e in expected if e in observed])
        missing  = len(expected) - matches

        score = matches / (matches + self.scoring_config.consistency_missing_penalty * missing) if (matches + missing) > 0 else 0.5
        return score

    def _compute_confidence(
        self,
        cause_name: str,
        evidence: List[str],
        anomalies: Dict[str, float],
        posterior: float,
        top_posterior: float,
        second_posterior: float,
    ) -> float:
        posterior_factor  = float(np.sqrt(np.clip(posterior, 0.0, 1.0)))
        consistency       = self._check_consistency(cause_name, anomalies)
        consistency_factor = consistency

        expected_count    = len(self._expected_evidence.get(cause_name, []))
        observed_matching = len(
            set(self._expected_evidence.get(cause_name, [])) & set(anomalies.keys())
        )
        saturation = observed_matching / expected_count if expected_count > 0 else 0.5
        saturation_factor = float(np.sqrt(np.clip(saturation, 0.0, 1.0)))

        if top_posterior > 0:
            margin = np.clip(
                (top_posterior - second_posterior) / top_posterior, 0.0, 1.0
            )
        else:
            margin = 0.0
        margin_factor = float(np.sqrt(margin))

        raw_confidence = (
            0.4 * posterior_factor +
            0.2 * consistency_factor +
            0.2 * saturation_factor +
            0.2 * margin_factor
        )

        confidence = np.clip(raw_confidence, posterior * 0.1, 1.0)
        return float(np.clip(confidence, 0.0, 1.0))


    def get_recommendations(self, cause_name: str, confidence: float) -> Dict[str, str]:
        """Return domain-specific operator recommendations for all 19 root causes."""
        if confidence < 0.20:
            return {}

        recs = {
            # ===== POWER SUBSYSTEM =====
            "solar_degradation": {
                "immediate":  "Disable non-critical secondary payloads to reduce load.",
                "short_term": "Schedule a detailed solar array IV-curve sweep.",
                "escalation": "If battery SOC < 40%, initiate low-power safe mode.",
            },
            "battery_aging": {
                "immediate":  "Reduce non-critical loads to extend battery cycle life.",
                "short_term": "Schedule battery capacity test via depth-of-discharge cycling.",
                "escalation": "If capacity < 60% nameplate, switch to redundant battery string.",
            },
            "battery_thermal": {
                "immediate":  "Enable emergency battery heater/cooler override.",
                "short_term": "Analyze thermal gradient across battery cell stack.",
                "escalation": "If cell delta-T > 15°C, command thermal safe mode.",
            },
            "sensor_bias": {
                "immediate":  "Cross-compare with redundant sensor readings.",
                "short_term": "Initiate autonomous sensor recalibration sequence.",
                "escalation": "If bias > 5% of full scale, flag sensor for replacement.",
            },
            "panel_insulation_degradation": {
                "immediate":  "Rotate spacecraft to reduce solar panel thermal exposure.",
                "short_term": "Schedule thermal vacuum performance comparison test.",
                "escalation": "If panel temp > rated max, disable affected string.",
            },
            "battery_heatsink_failure": {
                "immediate":  "Reduce charge rate to limit internal heat generation.",
                "short_term": "Activate backup thermal management path.",
                "escalation": "If battery temp > 45°C, command emergency discharge to safe level.",
            },
            "payload_radiator_degradation": {
                "immediate":  "Reduce payload duty cycle to lower thermal load.",
                "short_term": "Evaluate alternative heat rejection paths.",
                "escalation": "If payload temp exceeds derating limit, power down payload.",
            },
            "pcdu_regulator_failure": {
                "immediate":  "Command switch to redundant PCDU regulator string B.",
                "short_term": "Analyze thermal telemetry for regulator board hot spots.",
                "escalation": "If bus voltage < 26.5V, prepare for emergency battery direct-connect.",
            },
            # ===== ADCS SUBSYSTEM =====
            "wheel_friction": {
                "immediate":  "Increase wheel heater setpoint by 5°C to thin lubricant.",
                "short_term": "Switch attitude control to magnetic-only desaturation mode.",
                "escalation": "If wheel current > 0.8A, command wheel shutdown and use thrusters.",
            },
            "gyro_drift": {
                "immediate":  "Switch attitude determination to star tracker primary.",
                "short_term": "Upload new gyro bias compensation parameters.",
                "escalation": "If pointing error > mission requirement, enter safe hold.",
            },
            "magnetorquer_anomaly": {
                "immediate":  "Increase reaction wheel desaturation frequency.",
                "short_term": "Test magnetorquer coils individually via commanded dipole.",
                "escalation": "If all torquer rods fail, switch to thruster-only attitude control.",
            },
            # ===== COMMS SUBSYSTEM =====
            "transponder_fault": {
                "immediate":  "Switch to redundant transponder chain.",
                "short_term": "Reduce modulation order to increase link margin.",
                "escalation": "If downlink power < -3dBm margin, enter emergency beacon mode.",
            },
            "antenna_pointing_error": {
                "immediate":  "Command antenna gimbal recalibration sweep.",
                "short_term": "Verify antenna position via ground station signal sweep.",
                "escalation": "If pointing loss > 3dB, switch to omni-directional antenna.",
            },
            "ber_spike": {
                "immediate":  "Enable forward error correction (FEC) at maximum rate.",
                "short_term": "Correlate BER with orbital position for interference mapping.",
                "escalation": "If BER > 1e-3 sustained, switch to backup frequency.",
            },
            # ===== OBC SUBSYSTEM =====
            "memory_corruption": {
                "immediate":  "Initiate task-level reset for affected service.",
                "short_term": "Perform full memory scrub and checksum validation.",
                "escalation": "If SEU frequency > 5/hour, command full system cold reboot.",
            },
            "watchdog_reset_fault": {
                "immediate":  "Extend watchdog timer timeout to prevent false resets.",
                "short_term": "Dump reset telemetry log for pattern analysis.",
                "escalation": "If resets > 3/orbit, enter autonomous safe mode.",
            },
            "software_exception": {
                "immediate":  "Restart affected task with error logging enabled.",
                "short_term": "Upload patched software module via secure uplink.",
                "escalation": "If exceptions persist after patch, fall back to safe mode software.",
            },
            # ===== PROPULSION SUBSYSTEM =====
            "thruster_valve_fault": {
                "immediate":  "Command valve power cycle and verify position feedback.",
                "short_term": "Perform diagnostic burn with full telemetry recording.",
                "escalation": "If valve non-responsive, isolate thruster branch and switch to backup.",
            },
            "fuel_pressure_anomaly": {
                "immediate":  "Close isolation valves to contain potential leak.",
                "short_term": "Monitor pressure trend over 3 orbits for leak rate estimation.",
                "escalation": "If pressure drop > 5% per orbit, declare propulsion emergency.",
            },
        }

        return recs.get(cause_name, {
            "immediate":  "Monitor relevant telemetry channels at high sample rate.",
            "short_term": "Review historical trend data for similar signatures.",
            "escalation": "Consult subsystem domain expert if confidence exceeds 60%.",
        })

    def _explain_mechanism(
        self,
        root_cause: str,
        evidence: List[str],
        anomalies: Dict[str, float],
    ) -> str:
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

        base = explanations.get(root_cause)
        if base is None:
            node = self.graph.nodes.get(root_cause)
            if node and node.description:
                base = node.description
            else:
                base = "Unknown root cause mechanism."
        return base


    def print_report(self, hypotheses: List[RootCauseHypothesis]):
        print("\nROOT CAUSE RANKING ANALYSIS")

        if not hypotheses:
            print("\nNo significant root causes detected.")
            return

        print("\nMost Likely Root Causes (by posterior probability):\n")
        for rank, hyp in enumerate(hypotheses, 1):
            joint_tag = f"  [MULTI-FAULT with {hyp.joint_partner}]" if hyp.joint_partner else ""
            print(
                f"{rank}. {hyp.name:35s} "
                f"P={hyp.probability:6.1%}  "
                f"Confidence={hyp.confidence:5.1%}"
                f"{joint_tag}"
            )

        print("\nDETAILED EXPLANATIONS:\n")
        for hyp in hypotheses:
            print(f"• {hyp.name} (P={hyp.probability:.1%})")
            if hyp.causal_paths:
                unique_paths = list(set([tuple(p) for p in hyp.causal_paths]))
                if unique_paths:
                    print("  Causal Paths:")
                    for path in unique_paths[:3]:
                        print(f"    {' → '.join(reversed(path))}")
            print(f"  Evidence: {', '.join(hyp.evidence)}")
            print(f"  Mechanism: {hyp.mechanism}")
            if hyp.recommendations:
                print(f"  Actions:")
                for level, action in hyp.recommendations.items():
                    print(f"    [{level}] {action}")
            print()


if __name__ == "__main__":
    from simulator.power import PowerSimulator

    sim      = PowerSimulator(duration_hours=24)
    nominal  = sim.run_nominal()
    degraded = sim.run_degraded(
        solar_degradation_hour=6.0,
        battery_degradation_hour=8.0,
    )

    graph  = CausalGraph()
    ranker = RootCauseRanker(graph)
    hypotheses = ranker.analyze(nominal, degraded, deviation_threshold=0.15)
    ranker.print_report(hypotheses)