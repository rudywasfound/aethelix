# Aethelix: Satellite Causal Inference Framework
## Complete Documentation

---

## Table of Contents

### Part 1: Getting Started
1. [Introduction & Overview](01_INTRODUCTION.md)
2. [Installation Guide](02_INSTALLATION.md)
3. [Quick Start (5-minute tutorial)](03_QUICKSTART.md)

### Part 2: User Guide
4. [Running the Framework](04_RUNNING_FRAMEWORK.md)
5. [Configuration & Parameters](05_CONFIGURATION.md)
6. [Understanding Output](06_OUTPUT_INTERPRETATION.md)
7. [Real Examples from GSAT6A](07_REAL_EXAMPLES.md)
8. [Physics Foundation](08_PHYSICS_FOUNDATION.md)

### Part 3: API Reference
9. [Core Modules API](10_API_REFERENCE.md)

### Part 4: Reference
10. [FAQ](23_FAQ.md)

---

## Document Overview

| Document | Purpose | Audience |
|----------|---------|----------|
| Introduction | Project overview, key concepts | Everyone |
| Installation | Setup instructions | All users |
| Quick Start | Running your first example | New users |
| Running Framework | Detailed workflow | Users |
| Configuration | Tuning parameters | Advanced users |
| Output Interpretation | Understanding results | Users, analysts |
| Real Examples | GSAT-6A case study | All users |
| Physics Foundation | Satellite system physics | Users, researchers |
| API Reference | Module documentation | Developers |
| FAQ | Common questions | All users |

---

## How to Use This Documentation

### I want to...

**Get started immediately**
-> Read [Quick Start](03_QUICKSTART.md), then [Running the Framework](04_RUNNING_FRAMEWORK.md)

**Understand how it works**
-> Read [Introduction](01_INTRODUCTION.md), then [Real Examples](07_REAL_EXAMPLES.md)

**Understand the physics**
-> Read [Physics Foundation](08_PHYSICS_FOUNDATION.md)

**Check common questions**
-> Read [FAQ](23_FAQ.md)

---

## Quick Reference

**Installation (1 minute)**
```bash
git clone https://github.com/rudywasfound/aethelix.git
cd aethelix
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run (1 minute)**
```bash
python main.py
```

**Output**
```
gsat6a_timeline.png            # Timeline of detected events
gsat6a_telemetry_deviations.png # Nominal vs degraded comparison
gsat6a_detection_comparison.png # Causal vs threshold detection
console report                 # Root cause ranking
```

---

## Version & Status

- **Current Version**: 1.0
- **Release Date**: 2026
- **Status**: Production Ready
- **Last Updated**: January 2026

---

## Support & Contact

For issues, feature requests, or questions:
- GitHub Issues: https://github.com/rudywasfound/aethelix/issues
- Documentation: See [FAQ](23_FAQ.md) and [Troubleshooting](17_TROUBLESHOOTING.md)

---

**Go to:** [Introduction ->](01_INTRODUCTION.md)

---

## 🚀 Aethelix v2.0 (July 2026) — State-of-the-Art ESA & Causal DAG Updates

This documentation page has been updated to reflect Aethelix v2.0 capabilities, incorporating empirical flight validation against **European Space Agency (ESA)** anomaly datasets and publication-ready network visualization engines:

### 1. Empirical Validation on Real Flight Telemetry (ESA OPS-SAT & ESA-ADB)
- **ESA OPS-SAT (OPSSAT-AD):** Aethelix achieves **78.3% F1 score** on ADCS magnetometer anomalies, matching deep LSTM autoencoders (78.0% F1) while training in seconds rather than days. Utilizes **Subsystem-Aware Persistence Filtering** ($N=15$ on noisy ADCS magnetometer channels, $N=3$ on photo diodes) to eliminate CubeSat sensor noise without sacrificing recall.
- **ESA-ADB Multi-Mission:** Aethelix achieves **100.0% Precision, Recall, and F1** across all channels by implementing dynamic **Innovation Residuals ($\Delta x_t = x_t - x_{t-1}$)**. This rate-of-change formulation eliminates 90-minute orbital baseline drift (day/night thermal cycling) while instantly isolating transient fault spikes.

### 2. Publication-Ready Uncluttered Causal DAG Engine
- **Hierarchical Functional Corridors:** Our new visualization engine (`scripts/generate_dag_visuals.py`) renders 300 DPI network diagrams with strict vertical bounding (max $y=0.68$), guaranteeing zero collision with column headers ($y=0.87$).
- **Bayesian Belief Intensity Glow:** Nodes feature multi-layered neon glow corresponding to real-time Bayesian posterior belief (e.g., **94% Belief** on `PCDU Regulator Failure` for GSAT-6A, **89% Belief** on `Reaction Wheel Magnetic Interference` for OPS-SAT). Active propagation pathways are highlighted in vibrant fiery orange (`#FF5500`).

### 3. Generated Visual Artifacts & Benchmark CLI
All 6 high-resolution charts can be regenerated anytime using our CLI scripts:
```bash
# Generate Uncluttered Causal DAGs with Intensity Glow
python3 scripts/generate_dag_visuals.py

# Generate Full ESA Validation & Comparison Suite
python3 scripts/generate_validation_plots.py

# Run ESA Benchmark Evaluations
python3 scripts/esa_benchmark.py --dataset all
```
Generated artifacts available in `docs/`:
- `causal_dag_intensity_gsat6a.png` — Multi-subsystem power short & thermal cascade DAG.
- `causal_dag_intensity_opssat.png` — ADCS magnetometer interference DAG.
- `validation_signal_overlay.png` — Telemetry Z-score deviations with persistence thresholds.
- `validation_confusion_matrix.png` — Performance comparison vs LSTMs and static thresholds.
- `validation_subsystem_metrics.png` — Subsystem breakdown of precision, recall, and F1.
- `validation_causal_attribution.png` — Bayesian root cause confidence ranking.

