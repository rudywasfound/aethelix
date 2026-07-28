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

## Empirical Flight Validation & DAG Layout Engine (v2.0)

This documentation page includes updates from the v2.0 evaluation suite, incorporating empirical validation on **European Space Agency (ESA)** flight telemetry and improved hierarchical network rendering:

### 1. Evaluation on ESA Flight Telemetry (OPS-SAT & ESA-ADB)
- **ESA OPS-SAT (OPSSAT-AD):** Evaluated on ADCS reaction wheel magnetic interference and magnetometer attitude anomalies. By applying **subsystem-aware persistence filtering** ($N=15$ consecutive samples for noisy magnetometer channels, $N=3$ for clean optical sensors), Aethelix achieves a **78.3% F1 score**, comparable to deep LSTM autoencoder baselines (78.0% F1) without requiring historical model training.
- **ESA-ADB Multi-Mission:** Evaluated on satellite telemetry subject to periodic orbital thermal variation. To prevent false alarms caused by 90-minute day/night baseline oscillations, the engine evaluates **innovation residuals ($\Delta x_t = x_t - x_{t-1}$)**. Decoupling slow orbital dynamics from step-change fault signatures isolates transient subsystem anomalies.

### 2. Hierarchical Causal DAG Rendering
- **Bounded Functional Corridors:** The visualization script (`scripts/generate_dag_visuals.py`) enforces strict vertical coordinate bounding (maximum node altitude $y=0.68$), ensuring clean separation from structural layer labels ($y=0.87$).
- **Normalized Ranking Score Visualization:** In accordance with project semantics, node color intensity maps directly to **normalized ranking scores** (e.g., **0.94 normalized score weight** for `PCDU Regulator Failure` on GSAT-6A; **0.89 normalized score weight** for `Reaction Wheel Magnetic Interference` on OPS-SAT). Active causal propagation paths connecting dominant root causes to observed telemetry deviations are highlighted along directed edges.

### 3. CLI Reproduction Commands
Benchmark datasets and diagnostic charts can be generated locally using the standalone scripts:
```bash
# Render hierarchical causal DAGs with normalized score weighting
python3 scripts/generate_dag_visuals.py

# Generate empirical validation charts across ESA datasets
python3 scripts/generate_validation_plots.py

# Execute benchmark suite evaluation
python3 scripts/esa_benchmark.py --dataset all
```
Generated diagnostic charts in `docs/`:
- `causal_dag_intensity_gsat6a.png` — Multi-subsystem causal DAG mapping normalized ranking scores.
- `causal_dag_intensity_opssat.png` — ADCS magnetometer interference DAG structure.
- `validation_signal_overlay.png` — Telemetry Z-score deviations and persistence window thresholds.
- `validation_confusion_matrix.png` — Detection performance comparison against sequence-level baselines.
- `validation_subsystem_metrics.png` — Subsystem-level precision, recall, and F1 evaluation.
- `validation_causal_attribution.png` — Normalized root-cause score distribution across benchmark scenarios.
