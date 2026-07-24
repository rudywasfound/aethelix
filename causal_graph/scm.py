"""
scm.py — Structural Causal Model utilities for Aethelix.

Provides interventional queries (ATE, interventional distributions) on top
of a CausalGraph that has structural equations configured.

.. warning:: Calibration status
    All default structural equation coefficients in the built-in GSAT-6A
    graph are derived from hand-set edge weights, NOT from empirical data.
    Outputs from ``ate()``, ``interventional_distribution()``, and
    ``CausalGraph.sample()`` are syntactically valid causal queries —
    the truncated factorisation is mathematically legitimate — but the
    numerical results reflect guesses about causal strength, not validated
    causal effect estimates.

    **Use these outputs for hypothesis generation and what-if exploration,
    not for automated decision-making, until calibrated against real
    interventional or sufficient observational data.**
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from causal_graph.graph_definition import CausalGraph, SCMNotConfiguredError


def interventional_distribution(
    graph: CausalGraph,
    interventions: Dict[str, float],
    target_nodes: List[str],
    n_samples: int = 10000,
) -> Dict[str, np.ndarray]:
    """
    Estimate the post-intervention distribution P(targets | do(interventions)).

    Parameters
    ----------
    graph : CausalGraph
        Must have structural equations configured on all nodes.
    interventions : dict
        Mapping from node name to intervention value (do(X=x)).
    target_nodes : list of str
        Which nodes to return samples for.
    n_samples : int
        Number of Monte Carlo samples.

    Returns
    -------
    dict
        Mapping from target node name to numpy array of samples.

    Raises
    ------
    SCMNotConfiguredError
        If any node lacks a structural equation.
    """
    all_samples = graph.sample(n=n_samples, interventions=interventions)
    return {t: all_samples[t] for t in target_nodes if t in all_samples}


def ate(
    graph: CausalGraph,
    cause: str,
    effect: str,
    x0: float = 0.0,
    x1: float = 1.0,
    n_samples: int = 50000,
) -> float:
    """
    Estimate the Average Treatment Effect of do(cause=x1) vs do(cause=x0) on effect.

    ATE = E[effect | do(cause=x1)] - E[effect | do(cause=x0)]

    For linear-Gaussian SCMs, the true ATE equals the product of path
    coefficients times (x1 - x0).  This Monte Carlo estimator converges
    to that value as n_samples → ∞.

    Parameters
    ----------
    graph : CausalGraph
        Must have structural equations configured on all nodes.
    cause : str
        The node to intervene on.
    effect : str
        The node to measure the effect at.
    x0 : float
        Baseline intervention value.
    x1 : float
        Treatment intervention value.
    n_samples : int
        Number of Monte Carlo samples per intervention.

    Returns
    -------
    float
        Estimated ATE.

    Raises
    ------
    SCMNotConfiguredError
        If any node lacks a structural equation.

    .. warning::
        The returned ATE is only as valid as the structural equations.
        With default (uncalibrated) coefficients, this is a structured
        hypothesis, not a validated causal effect estimate.
    """
    samples_x0 = graph.sample(n=n_samples, interventions={cause: x0})
    samples_x1 = graph.sample(n=n_samples, interventions={cause: x1})

    if effect not in samples_x0 or effect not in samples_x1:
        raise ValueError(
            f"Effect node '{effect}' not found in samples. "
            f"Available nodes: {sorted(samples_x0.keys())}"
        )

    mean_under_x0 = np.mean(samples_x0[effect])
    mean_under_x1 = np.mean(samples_x1[effect])

    return float(mean_under_x1 - mean_under_x0)
