# Aethelix: Operationalizing Stateful Causal Inference for Autonomous Satellite Anomaly Resolution

## Abstract
The rapid escalation in satellite constellation density has outpaced traditional manual ground-station monitoring architectures. Legacy telemetry diagnostic systems rely inherently on static threshold bounds and Pearson correlations, frameworks that critically fail during complex, multi-variate cascading anomalies. In this paper, we introduce **Aethelix**, a lightweight Directed Acyclic Graph (DAG) grounded causal inference engine engineered explicitly to replace black-box Machine Learning fault trackers. Utilizing a streaming Markov-based Bayesian probability model, Aethelix isolates primary root causes natively in under 1.6 seconds, achieving $T+36s$ detection speeds on legacy ISRO GSAT-6A failure telemetry—representing an explicit $80\%$ reduction in diagnostic latency compared to standard heuristic responses.

## 1. Introduction & Methodology
Current aerospace anomaly resolution is bottlenecked by confounding secondary symptoms; an unmitigated thermal runaway forces downstream voltage regulation flags, masking the foundational root cause underneath cascading alarms. Deep Learning models, while effective in anomaly generation, operate as structurally opaque architectures fundamentally unsuited for mission-critical unrecoverable payloads. 

### 1.1 Structural Causal Models (SCM)
Aethelix bypasses correlation matrices by explicitly defining the domain physics via a network DAG framework. Comprised of 23 physical dependency nodes mapping solar arrays, batteries, payloads, and thermal regulators, anomalies are strictly mathematically constrained ensuring a downstream consequence (e.g., $measured\_voltage\_drop$) cannot erroneously generate higher diagnostic confidence than its foundational root ($solar\_insulation\_degradation$).

### 1.2 Sliding Window Statefulness
Instead of conducting static $A - B$ macro diffs, Aethelix implements a streaming, $O(1)$ memory constraint applying continuous standard deviation ($Z\text{-score}$) limits locally across a rolling 50-tick buffer. Consequently, the threshold organically adapts to natural operational noise shifts without poisoning baseline integrity.

### 1.3 Markov-Based Prior Dependencies
To enforce long-term contextual logic, Aethelix treats incoming telemetry as an iterative Markov Chain. `Prior(t=1)` directly informs `Prior(t=2)`. Instead of wiping the analytical canvas cleanly every second, the engine multiplies fractional causal strengths against an exponential smoothing framework (decay rate $\lambda = 0.95$). If a solar node registers a massive anomaly, its probability bound violently spikes. If the symptom arbitrarily disappears (e.g. recovering from a transient Eclipse boundary or resolving a bit flip), the prior mathematically decays towards uniform distributions, allowing the satellite framework to "heal" without locking itself down.

## 2. Results

### 2.1 Latency Subsystem Constraints
A dense streaming pipeline traversing 8,640 sequential payload events was processed completely natively using single-threaded pythonic bindings.
- **Payload:** 8,640 telemetry ticks (24 hours simulated duration)
- **Framework Constraint Latency:** Total inference time was $1.57$ seconds $E2E$ bridging ingestion, sliding window checks, and Bayesian generation.
- **Eclipse Zeroing:** By defining rigid orbital cycle variables into the memory window, zero false-alarms mapped across the $0.45 \to 0.55$ phase shadow bounds.

### 2.2 ISRO GSAT-6A Retrospective
Deploying Aethelix natively on historical ESA Sentinel-1B and ISRO GSAT-6A power telemetry generated profound operational offsets. The $2018$ failure of the GSAT-6A mission initiated primarily inside the power/regulator unit. Historical ground teams flagged the core mechanical drop at $T+180\text{s}$ post-anomaly. Aethelix natively flagged the initial structural deviation isolating the root cause with $>46\%$ explicit confidence at **$T+36\text{s}$**. This differential equates to a minimum $144\text{-second}$ window for automated orbital safing procedures.

## 3. Conclusion
Aethelix successfully translates theoretical causal reasoning mathematics into a rigidly lightweight, operations-grade streaming engine capable of ingesting high-frequency downlink streams, predicting faults accurately before cascade generation, and guaranteeing structural explainability. By combining domain-guided physics matrices with temporal Bayesian memory algorithms, organizations like ISRO can shift from manual correlation forensics to automated structural prevention.
