# What Aethelix Output Numbers Actually Mean
 
> This document precisely defines the mathematical meaning of every value
> produced by the Aethelix inference engine. Read this before interpreting
> any result, before integrating any output into a downstream system, and
> before extending the engine with new features.
 
---
 
## 1. The Core Commitment
 
Aethelix currently solves a single, well-defined problem:
 
> **Given a set of observed telemetry anomalies and a manually curated causal DAG,
> rank the root-cause hypotheses from most to least consistent with the evidence.**
 
That is the full scope of the current output. Nothing more, nothing less.
Understanding what this implies - and what it does not imply - is the
foundation of correct interpretation.
 
---
 
## 2. What the Output Values Are
 
### 2.1 Single Hypothesis Scoring Formula
 
The engine computes a raw score for each independent root-cause hypothesis `r` as:
 
```
score(r) = coverage(r) × (1 − β × inconsistency(r)) × severity_weighted(r) × depth_prior(r)
```
 
where:
 
- **`coverage(r)`**: The fraction of active anomalous observables reachable from `r` in the DAG.
- **`inconsistency(r)`**: The fraction of inactive (quiet) observables that `r` predicts should be anomalous.
- **`β`**: The consistency softening penalty weight (set to `0.5` in the v2 engine to avoid over-penalising secondary hypotheses).
- **`severity_weighted(r)`**: Weighted sum of observed anomaly severities that are causally explained by `r`.
- **`depth_prior(r)`**: Structural prior based on topological depth from root causes in the DAG. Rewarding true root cause nodes (`depth 0`: `1.0`, `depth 1`: `0.7`, `depth 2`: `0.5`, `depth 3+`: `0.35`) and penalising intermediate nodes to resolve cascading ambiguity.
 
### 2.2 Joint Two-Hypothesis Scoring Formula
 
To handle multi-fault scenarios, the engine also evaluates candidate pairs of root causes `(r1, r2)` and scores their combined explanatory power:
 
```
joint_score(r1, r2) = union_coverage(r1, r2) × (1 − β × joint_inconsistency) × severity_factor × mean_prior
```
 
where:
- **`union_coverage(r1, r2)`**: Fraction of active anomalies explained by either `r1` or `r2`.
- **`joint_inconsistency`**: Softened penalty for quiet channels predicted anomalous by *either* cause.
- **`severity_factor`**: Fraction of total anomaly severity explained by the pair.
- **`mean_prior`**: The average of the individual depth priors of `r1` and `r2` `(depth_prior(r1) + depth_prior(r2)) / 2.0`.
 
### 2.3 Normalization
 
Single hypotheses probabilities are normalized to sum to 1.0:
```
P(h | evidence) = raw_score(h) / Σ raw_score(h_i)
```
 
Joint hypotheses probabilities are normalized among candidate pairs:
```
P(r1, r2 | evidence) = joint_score(r1, r2) / Σ joint_score(a, b)
```
 
A joint hypothesis `r1 + r2` is exposed in the unified ranked list if its joint probability is at least 5% higher than the top single candidate's probability.
 
### 2.4 Confidence
 
The `Confidence` value reported alongside each hypothesis is:
```
confidence(h) = 0.4 × posterior_factor + 0.2 × consistency_factor + 0.2 × saturation_factor + 0.2 × margin_factor
```
For joint hypotheses, the confidence is the maximum of individual confidences multiplied by a joint probability bonus `(1.0 + 0.1 × joint_prob)`. It reflects how strongly the evidence supports the hypothesis independent of other candidates.
 
---
 
## 3. What These Values Are NOT
 
This is the most important section of this document.
 
### 3.1 Not Calibrated Bayesian Posteriors
 
The values produced after normalization **look** like posterior probabilities —
they are positive, they sum to 1.0, and they are labeled with `P=`. They are
**not** calibrated Bayesian posteriors in the formal sense.
 
A calibrated posterior `P(h | e)` requires:
- A prior `P(h)` over hypotheses, derived from historical base rates or
  engineering priors,
- A likelihood `P(e | h)` that models how likely each observation is
  *given that hypothesis h is the true cause*,
- Bayes' rule applied over a normalized hypothesis space.
Aethelix does not implement any of these steps. There are no priors. There are
no likelihoods. The normalization step is a rescaling operation applied to
heuristic scores, not an application of Bayes' theorem. Calling the output
`P=46.3%` is shorthand for "this hypothesis holds 46.3% of the total score
mass." It is a proportion, not a probability.
 
**Concretely:** you cannot say "there is a 46.3% chance that solar degradation
is the root cause." You can say "solar degradation is the strongest-scoring
hypothesis, accounting for roughly 46% of the total evidence weight."
 
### 3.2 Not Marginal Probabilities Over Independent Faults
 
Although Aethelix now explicitly scores pairs of faults via joint scoring, the output values do **not** represent independent marginal probabilities like `P(A is active)` and `P(B is active)`. The probability values are rescaled heuristic scores, meaning that increasing one candidate's score mechanically decreases the others in the normalised list.
 
### 3.3 Not a Solution to the General Multi-Fault (K >= 3) Diagnosis Problem
 
Aethelix now implements a targeted **joint two-hypothesis scoring pass** to diagnose double-fault scenarios. It evaluates pairs of root causes and ranks their combined explanatory power. While this successfully solves the two-fault co-occurrence problem (raising two-fault accuracy significantly), it does not search for higher-order subsets (3+ simultaneous faults) or solve the general multi-fault partition search.
 
---
 
## 4. What the Output Is Good For
 
| Use Case | Appropriate? | Notes |
|---|---|---|
| Identifying the single most likely dominant root cause | Yes | Core design intent |
| Ordering hypotheses for manual investigation | Yes | Ranking is reliable |
| Eliminating clearly unsupported hypotheses | Yes | Low-scoring hypotheses are weak candidates |
| Detecting that *something* is anomalous | Yes | Severity scores are directly useful |
| Estimating the probability that a specific fault is active | No | Values are not calibrated |
| Concluding that two faults are both present simultaneously | Yes | Supported via joint two-hypothesis scoring when joint probability significantly exceeds single probability |
| Computing a probability threshold for automatic action | No | Requires calibration first |
| Formal certification or safety-critical automated decisions | No | Requires a full probabilistic model |
 
---
 
## 5. Interpretation Guide for the Example Output
 
```
1. solar_degradation    P=46.3%  Confidence=93.3%
2. battery_aging        P=18.8%  Confidence=71.7%
3. battery_thermal      P=18.7%  Confidence=75.0%
4. sensor_bias          P=16.3%  Confidence=75.0%
```
 
Correct interpretation:
 
> Solar degradation is by far the strongest-scoring hypothesis. It accounts
> for nearly half the total evidence weight and is supported with very high
> consistency. Battery aging and battery thermal effects are secondary
> candidates of roughly equal relative weight. Sensor bias is the weakest
> candidate. The engine's primary output is that solar degradation should be
> investigated first.
 
Incorrect interpretations to avoid:
 
> ~~"There is a 46.3% probability that solar degradation is the cause."~~
> ~~"There is a 54% probability that solar degradation is NOT the cause."~~
> ~~"Battery aging and battery thermal are equally likely to be co-occurring."~~
> ~~"Since scores sum to 1.0, only one fault is present."~~
 
The high confidence values (93.3%, 75%) are separately meaningful: they say
that the evidence strongly engages with these hypotheses, not just that they
scored higher in the ranking.
 
---
 
## 6. The Diagnostic Objective Being Solved
 
To be fully explicit, the Aethelix engine implements a dual-pass objective:
 
1. **Single Hypothesis Pass:** Find the candidate `r` that maximizes:
   `score(r) = coverage(r) × (1 − β × inconsistency(r)) × severity_weighted(r) × depth_prior(r)`
2. **Joint Hypothesis Pass:** Evaluate all combinations of top candidates `(r1, r2)` and rank them using:
   `joint_score(r1, r2) = union_coverage(r1, r2) × (1 − β × joint_inconsistency) × severity_factor × mean_prior`
 
This objective is appropriate when:
- The operator wants to identify the dominant single fault or pair of faults quickly.
- The cascading failure path is well-represented in the Causal DAG.
This objective is insufficient when:
- Three or more independent faults occur simultaneously.
- Downstream systems require safety-certified, mathematically calibrated marginal probabilities.
---
 
## 7. What Would Be Needed for Stronger Guarantees
 
For reference, here is what each stronger semantic would require. These are
directions for future work, not current claims.
 
### 7.1 Calibrated Posteriors
 
Would require: empirical prior probabilities over faults (e.g., from historical
satellite failure records), a likelihood model `P(telemetry deviation | fault
active)` per node, and validation against held-out cases to confirm that a
stated `P=40%` is correct in approximately 40% of similar cases.
 
### 7.2 True Multi-Fault Identification
 
Would require: scoring all 2ⁿ subsets of hypotheses (or a smarter search),
a joint likelihood that accounts for fault interactions, and a criterion such
as MAP subset selection or a credible interval over the fault set.
 
### 7.3 Counterfactual Reasoning
 
Would require: structural equations on each node (not just edge weights),
the ability to intervene on a node (set it to a specific value) and propagate
the effect, and a formal do-calculus implementation on the DAG.
 
None of these are prerequisites for the current engine to be useful. They are
noted here to make clear what the current engine does not claim.
 
---
 
## 8. Terminology Conventions
 
To avoid ambiguity in code, comments, and documentation, use the following
terms consistently:
 
| Term | Correct Usage |
|---|---|
| **normalized score** | The value printed as `P=xx%` in output |
| **ranking score** | The raw pre-normalization score for a hypothesis |
| **consistency** | The evidence-quality × consistency scalar |
| **dominant root cause** | The highest-ranking hypothesis output by the engine |
| **causal path** | A directed path in the DAG from a root cause to an observable |
| ~~posterior probability~~ | Do not use for the current output |
| ~~Bayesian probability~~ | Do not use for the current output |
| ~~probability of fault~~ | Do not use for the current output |
 
---
 
## 9. Versioning
 
This document describes the semantics of Aethelix as of the current `master`
branch. If the scoring formula, normalization method, or output structure
changes, this document must be updated before the change is merged.
 
Any feature that would change the semantic category of the output values -
for example, introduction of calibrated priors, a likelihood model, or joint
multi-fault scoring - must be accompanied by a corresponding update to this
document and to the output labels in the code.
 
---
 
*Precision about what a model proves is not a limitation - it is what makes
the model trustworthy.*
