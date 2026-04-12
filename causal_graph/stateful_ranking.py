"""
Stateful Root Cause Ranker — Bayesian Markov-linked temporal inference.

Key fixes over the original:

1. Prior stabilisation
   The original prior update formula was:
       decayed = prior * decay + uniform * (1 - decay)
   This is a convex blend that converges all priors toward 1/N after a few
   timesteps, collapsing the margin between causes and driving confidence → 0.

   New formula keeps the dominant hypothesis's advantage:
       decayed = prior * decay  (pure exponential decay toward zero)
   Normalisation then re-sharpens the distribution rather than flattening it.

2. Temporally-aware confidence
   The base _compute_confidence is designed for single-shot analysis where
   margin is the only ambiguity signal.  In a streaming context we have an
   additional strong signal: how many consecutive timesteps has this hypothesis
   been the top-ranked cause?  High streak -> confidence grows organically.

   New formula:
       temporal_factor  = tanh(streak / STREAK_SCALE)   # 0 -> 1 as streak grows
       margin_factor    = sqrt(margin)                   # ambiguity suppression
       posterior_factor = sqrt(posterior)                # probability anchor
       saturation       = sqrt(sat)                      # evidence completeness
       consistency      = check_consistency(...)         # expected symptoms match

       confidence = posterior_factor
                  * consistency
                  * saturation
                  * margin_factor
                  * (BASE_CONF + (1 - BASE_CONF) * temporal_factor)

   The temporal_factor term means:
   - At streak=0  -> multiplier = BASE_CONF (≈0.4), honest uncertainty
   - At streak=3  -> multiplier ≈ 0.70
   - At streak=7  -> multiplier ≈ 0.88
   - At streak=15 -> multiplier ≈ 0.97
   So confidence builds as the same cause keeps winning, which is exactly
   the right behaviour for a live satellite monitoring system.

3. Posterior floor
   Any hypothesis that achieves > POSTERIOR_FLOOR (0.35) posterior gets a
   minimum confidence of posterior * MIN_CONF_RATIO, preventing the product
   of small factors from flooring a genuinely dominant hypothesis at ~0%.

4. Confidence is stored and displayed as a PERCENTAGE (0–100).
   The dashboard compares hyp.confidence against thresholds like 50.0 and
   20.0 — so confidence must live in the 0–100 range, not 0–1.
   All internal computation stays in 0–1; we multiply by 100 at the end.
"""

import numpy as np
from typing import Dict, List
from causal_graph.root_cause_ranking import RootCauseRanker, RootCauseHypothesis


# Tunable constants -change here, nowhere else
_DECAY           = 0.92   # Exponential prior decay rate per timestep
_STREAK_SCALE    = 4.0    # Timesteps to reach ~63 % of max temporal boost (faster build)
_BASE_CONF       = 0.55   # Minimum confidence multiplier before temporal boost (raised)
_POSTERIOR_FLOOR = 0.25   # Posterior above which we apply the confidence floor (lowered)
_MIN_CONF_RATIO  = 0.45   # Floor = posterior * this ratio (raised for stronger floor)


class StatefulRootCauseRanker(RootCauseRanker):
    """
    Temporal Bayesian ranker that retains posteriors from T as priors at T+1.

    Compared to the base RootCauseRanker.analyze(), this class:
    - Maintains a running prior distribution over root causes
    - Applies exponential decay so stale evidence gradually forgives
    - Tracks a per-cause streak counter to reward consistent top rankings
    - Uses a temporally-aware confidence formula that builds over time
    - Returns confidence as a PERCENTAGE (0–100) to match dashboard thresholds
    """

    def __init__(self, graph, decay: float = _DECAY):
        super().__init__(graph)
        self.decay = decay

        # Running prior distribution. Initialised to uniform; updated each call.
        self.priors: Dict[str, float] = {}

        # How many consecutive timesteps has each cause been the #1 ranked cause?
        self._streak: Dict[str, int] = {}

        # Cached name of the top cause at the previous timestep (for streak tracking)
        self._prev_top: str = ""

    # Public API


    def reset(self):
        """Clear all memory — call when starting a new telemetry session."""
        self.priors = {}
        self._streak = {}
        self._prev_top = ""

    def analyze_stream(
        self,
        anomalies: Dict[str, float],
    ) -> List[RootCauseHypothesis]:
        """
        Rank root causes using Bayesian Markov-linked probabilistic memory.

        Args:
            anomalies: Pre-computed anomaly dict {channel_name: severity (0-1)}
                       produced by the sliding-window detector upstream.

        Returns:
            Sorted list of RootCauseHypothesis, highest probability first.
            hyp.confidence is in PERCENTAGE units (0–100).
        """

        # Handle empty observation window

        if not anomalies:
            for c in list(self.priors.keys()):
                self.priors[c] *= self.decay
            self._streak = {c: max(0, v - 1) for c, v in self._streak.items()}
            self._prev_top = ""
            return []

        # Backward tracing

        root_cause_scores:   Dict[str, float]      = {}
        root_cause_evidence: Dict[str, List[str]]  = {}
        root_cause_paths:    Dict[str, List]        = {}

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

        total_score = sum(root_cause_scores.values())
        if total_score == 0:
            for c in list(self.priors.keys()):
                self.priors[c] *= self.decay
            return []

        # Normalise likelihoods

        likelihoods: Dict[str, float] = {
            c: s / total_score for c, s in root_cause_scores.items()
        }

        # Bayesian prior update

        n_known = len(self._expected_evidence)
        uniform = 1.0 / n_known if n_known else 0.1

        unnorm: Dict[str, float] = {}
        for cause, likelihood in likelihoods.items():
            # Pure exponential decay preserves distribution shape, but floor prevents Cromwell's rule
            prior = max(1e-4, self.priors.get(cause, uniform) * self.decay)
            unnorm[cause] = likelihood * prior

        total_unnorm = sum(unnorm.values())
        if total_unnorm == 0:
            return []

        posteriors: Dict[str, float] = {
            c: v / total_unnorm for c, v in unnorm.items()
        }

        # Persist posteriors as next-step priors

        for cause in self._expected_evidence:
            if cause in posteriors:
                self.priors[cause] = posteriors[cause]
            else:
                self.priors[cause] = self.priors.get(cause, uniform) * self.decay

        # Streak update

        sorted_posterior = sorted(
            posteriors.items(), key=lambda x: x[1], reverse=True
        )
        current_top = sorted_posterior[0][0] if sorted_posterior else ""

        for cause in self._expected_evidence:
            if cause == current_top:
                # Increment streak if it's the winner
                self._streak[cause] = self._streak.get(cause, 0) + 1
            else:
                # Soft decay: -1 per missed tick so noisy timesteps don't wipe memory
                # This makes the system robust to transient noise/ambiguity
                self._streak[cause] = max(0, self._streak.get(cause, 0) - 1)

        self._prev_top = current_top

        # Build hypotheses

        top_posterior    = sorted_posterior[0][1] if len(sorted_posterior) >= 1 else 0.0
        second_posterior = sorted_posterior[1][1] if len(sorted_posterior) >= 2 else 0.0

        hypotheses: List[RootCauseHypothesis] = []
        for cause_name, probability in posteriors.items():
            mechanism = self._explain_mechanism(
                cause_name,
                root_cause_evidence.get(cause_name, []),
                anomalies,
            )
            # Returns 0–100 percentage
            confidence = self._compute_stateful_confidence(
                cause_name       = cause_name,
                evidence         = root_cause_evidence.get(cause_name, []),
                anomalies        = anomalies,
                posterior        = probability,
                top_posterior    = top_posterior,
                second_posterior = second_posterior,
                streak           = self._streak.get(cause_name, 0),
            )

            hypotheses.append(
                RootCauseHypothesis(
                    name         = cause_name,
                    probability  = probability,
                    evidence     = root_cause_evidence.get(cause_name, []),
                    mechanism    = mechanism,
                    confidence   = confidence,
                    causal_paths = root_cause_paths.get(cause_name, []),
                )
            )

        hypotheses.sort(key=lambda h: h.probability, reverse=True)
        return hypotheses

    # Override base _compute_confidence so analyze() also works correctly

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
        Override parent — routes through stateful confidence with current streak.
        Returns percentage (0–100) to match dashboard display thresholds.
        """
        return self._compute_stateful_confidence(
            cause_name       = cause_name,
            evidence         = evidence,
            anomalies        = anomalies,
            posterior        = posterior,
            top_posterior    = top_posterior,
            second_posterior = second_posterior,
            streak           = self._streak.get(cause_name, 0),
        )

    # Temporally-aware confidence (returns 0–100 PERCENTAGE)
    
    def _compute_stateful_confidence(
        self,
        cause_name:       str,
        evidence:         List[str],
        anomalies:        Dict[str, float],
        posterior:        float,
        top_posterior:    float,
        second_posterior: float,
        streak:           int,
    ) -> float:
        """
        Confidence formula for streaming/stateful context.

        Returns a value in 0–100 (percentage) so the dashboard thresholds
        (> 50.0 for critical, > 20.0 for warning, > 30.0 for event log)
        work correctly without any conversion.

        Four base factors:
          posterior_factor  = √posterior
          consistency       = fraction of expected symptoms observed
          saturation_factor = √(observed_matching / expected_count)
          margin_factor     = √((top - second) / top)

        Temporal factor (builds confidence over consecutive top-ranked timesteps):
          temporal  = tanh(streak / STREAK_SCALE)
          time_mult = BASE_CONF + (1 - BASE_CONF) * temporal

        Posterior floor:
          If posterior >= POSTERIOR_FLOOR, confidence floor = posterior * MIN_CONF_RATIO * 100
        """

        posterior_factor = float(np.sqrt(np.clip(posterior, 0.0, 1.0)))


        consistency = self._check_consistency(cause_name, anomalies)


        expected_count = len(self._expected_evidence.get(cause_name, []))

        observed_match = len(
            set(self._expected_evidence.get(cause_name, [])) & set(anomalies.keys())
        )
        # Posterior Bypass: if posterior >= 0.55, set saturation_factor = 1.0 
        # so a clearly dominant hypothesis isn't penalised for incomplete evidence
        if posterior >= 0.55:
            saturation_factor = 1.0
        else:
            saturation = (observed_match / expected_count) if expected_count > 0 else 0.5
            saturation_factor = float(np.sqrt(np.clip(saturation, 0.0, 1.0)))

        if top_posterior > 0:

            margin = np.clip(
                (top_posterior - second_posterior) / top_posterior, 0.0, 1.0
            )
        else:
            margin = 0.0
        margin_factor = float(np.cbrt(margin))

        temporal  = float(np.tanh(streak / _STREAK_SCALE))

        time_mult = _BASE_CONF + (1.0 - _BASE_CONF) * temporal

        raw = (

            posterior_factor
            * float(np.sqrt(consistency))
            * saturation_factor
            * margin_factor
            * time_mult
        )

        if posterior >= _POSTERIOR_FLOOR:

            floor = posterior * _MIN_CONF_RATIO
            raw   = max(raw, floor)

        return float(np.clip(raw * 100.0, 0.0, 100.0))