# Theoretical Foundations of Causal Diagnosis

This document formalizes the mathematical necessity of the causal inference approach used in Aethelix, specifically regarding the detection of sub-threshold anomalies.

## Theorem 1 — Univariate Threshold Detection Incompleteness

**Statement:**
Let $F$ be a fault whose causal footprint produces per-channel deviations $d_i < \delta$ for all observable channels $i$, where $\delta$ is the detection threshold.

Any system relying solely on univariate threshold crossings has detection rate $P(\text{detect} | F) = 0$, independent of fault severity, duration, or number of affected channels.

**Proof:**
By definition, no channel $i$ crosses the threshold $\delta$ since $d_i < \delta$ for all $i$. Therefore, at any time $t$, the set of triggered alarms $A = \{i : d_i \ge \delta\}$ is empty. Since the detection function is dependent on $A$ being non-empty, no alarm fires. QED.

**Corollary:**
Multi-channel causal pattern detection is a necessary condition for sub-threshold fault detectability.

## Application in Aethelix

Traditional satellite Ground Control Systems (GCS) rely on out-of-limit (OOL) checks which are univariate threshold detectors. Aethelix overcomes this limitation by modeling the joint distribution and causal dependencies between channels. 

Even if no individual thermistor or voltage sensor identifies a violation, the *simultaneous* subtle drifting of power and thermal residuals creates a causal signature that can be back-propagated to a root cause with high confidence. This provides a significant lead-time advantage over traditional systems.
