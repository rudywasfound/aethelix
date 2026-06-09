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
 
### 2.1 The Scoring Formula
 
The engine computes a raw score for each root-cause hypothesis `h` as:
 
```
raw_score(h) = path_strength(h) × consistency(h) × severity(h)
```
 
where:
 
- **`path_strength(h)`**: the product of edge weights along the strongest causal
  path from `h` to the observed anomalies in the DAG. These weights are
  hand-encoded engineering priors; they are not learned from data.
- **`consistency(h)`**: a measure of how many of the anomalies active in the
  current observation are reachable from `h` via the DAG, relative to how many
  `h` would predict. A hypothesis that explains all active anomalies and
  predicts no inactive ones scores 1.0.
- **`severity(h)`**: a scalar derived from the magnitude of the observed
  deviations on nodes that `h` influences. Larger deviations yield higher
  severity.
### 2.2 Normalization
 
After all hypotheses are scored, the raw scores are normalized:
 
```
P(h | evidence) = raw_score(h) / Σ raw_score(h_i)
```
 
This forces the output values to sum to 1.0.
 
### 2.3 Confidence
 
The `Confidence` value reported alongside each hypothesis is:
 
```
confidence(h) = evidence_quality(h) × consistency(h)
```
 
It is a separate scalar, not derived from the normalized probability. It
reflects how strongly the available evidence speaks to `h`, independent of
the relative ranking.
 
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
 
In a multi-fault scenario where faults A and B are both active simultaneously,
the output values do **not** represent `P(A is active)` and `P(B is active)`
as independent marginals. Because scores are normalized to sum to 1.0, the
values are *compositional*: increasing the score of one hypothesis mechanically
decreases the others, even if those hypotheses are not causally related.
 
The normalization creates an implicit mutual-exclusivity assumption — the
output behaves as if at most one hypothesis is true. This is a useful fiction
for ranking dominant causes. It is an incorrect model for estimating which
combination of faults is active.
 
### 3.3 Not a Solution to the Full Multi-Fault Diagnosis Problem
 
The full multi-fault diagnosis problem asks:
 
> Which subset S ⊆ {h₁, h₂, ..., hₙ} of hypotheses is simultaneously active,
> and is consistent with the observed evidence?
 
Aethelix does not solve this problem. It does not enumerate joint hypothesis
combinations, does not score subsets, and does not search for a maximally
consistent fault set. What it does is identify **the single most explanatory
hypothesis** and rank the others relative to it.
 
This is a valid and practically useful objective — identifying the dominant
root cause quickly is valuable for operators under time pressure. But it
should not be confused with a complete multi-fault identification.
 
---
 
## 4. What the Output Is Good For
 
| Use Case | Appropriate? | Notes |
|---|---|---|
| Identifying the single most likely dominant root cause | Yes | Core design intent |
| Ordering hypotheses for manual investigation | Yes | Ranking is reliable |
| Eliminating clearly unsupported hypotheses | Yes | Low-scoring hypotheses are weak candidates |
| Detecting that *something* is anomalous | Yes | Severity scores are directly useful |
| Estimating the probability that a specific fault is active | No | Values are not calibrated |
| Concluding that two faults are both present simultaneously | No | Normalization masks multi-fault structure |
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
 
To be fully explicit, the current Aethelix engine implements:
 
> **Hypothesis H₁:** Find the hypothesis h* that maximizes
> `path_strength(h) × consistency(h) × severity(h)`, and return all
> hypotheses ranked by this score, normalized to a unit-sum distribution.
 
This objective is appropriate when:
- The operator wants to know *where to look first*,
- One fault is significantly more explanatory than the others,
- The DAG structure correctly encodes the causal mechanism.
This objective is insufficient when:
- Two or more faults are simultaneously active and similarly explanatory,
- The goal is to certify that a specific fault is absent,
- Downstream systems require calibrated probabilities for risk computation.
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
