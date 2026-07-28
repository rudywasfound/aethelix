# Aethelix Documentation

Complete documentation for the Aethelix Satellite Causal Inference Framework.

## Quick Links

### Getting Started (Start here!)
1. **[Table of Contents](00_TABLE_OF_CONTENTS.md)** - Full documentation structure
2. **[Introduction](01_INTRODUCTION.md)** - What is Aethelix and why use it
3. **[Installation](02_INSTALLATION.md)** - Set up your environment
4. **[Quick Start](03_QUICKSTART.md)** - Run your first example (5 min)

### Using Aethelix
5. **[Running the Framework](04_RUNNING_FRAMEWORK.md)** - How to execute workflows
6. **[Configuration](05_CONFIGURATION.md)** - Tune parameters
7. **[Output Interpretation](06_OUTPUT_INTERPRETATION.md)** - Understand the results

### Deep Dive
8. **[Architecture](07_ARCHITECTURE.md)** - System design overview
9. **[Causal Graph](08_CAUSAL_GRAPH.md)** - Graph structure and design
10. **[Inference Algorithm](09_INFERENCE_ALGORITHM.md)** - How reasoning works

### Reference
11. **[API Reference](10_API_REFERENCE.md)** - Module documentation
12. **[Python Library](11_PYTHON_LIBRARY.md)** - Use as a library
13. **[Rust Integration](12_RUST_INTEGRATION.md)** - High-performance features

### Advanced Topics
14. **[Simulation & Testing](13_SIMULATION.md)** - Create test scenarios
15. **[Custom Scenarios](14_CUSTOM_SCENARIOS.md)** - Domain-specific use cases
16. **[Performance Tuning](15_PERFORMANCE.md)** - Optimize speed
17. **[Deployment](16_DEPLOYMENT.md)** - Production setup
18. **[Troubleshooting](17_TROUBLESHOOTING.md)** - Fix issues
19. **[Monitoring](18_MONITORING.md)** - Runtime observation

### Development
20. **[Development Setup](19_DEVELOPMENT.md)** - Local development
21. **[Contributing](20_CONTRIBUTING.md)** - Contribute code
22. **[Testing Framework](21_TESTING.md)** - Test infrastructure

### Reference
23. **[Glossary](22_GLOSSARY.md)** - Terminology
24. **[FAQ](23_FAQ.md)** - Common questions
25. **[Bibliography](24_REFERENCES.md)** - Academic references

## Usage Paths

### I'm new to Aethelix
-> Read: [Introduction](01_INTRODUCTION.md) -> [Installation](02_INSTALLATION.md) -> [Quick Start](03_QUICKSTART.md)

### I want to run it
-> Read: [Running the Framework](04_RUNNING_FRAMEWORK.md) -> [Configuration](05_CONFIGURATION.md)

### I want to understand it
-> Read: [Architecture](07_ARCHITECTURE.md) -> [Causal Graph](08_CAUSAL_GRAPH.md) -> [Inference Algorithm](09_INFERENCE_ALGORITHM.md)

### I want to use it as a library
-> Read: [Installation](02_INSTALLATION.md) -> [Python Library](11_PYTHON_LIBRARY.md) -> [API Reference](10_API_REFERENCE.md)

### I want to deploy it
-> Read: [Deployment](16_DEPLOYMENT.md) -> [Monitoring](18_MONITORING.md) -> [Troubleshooting](17_TROUBLESHOOTING.md)

### I want to contribute
-> Read: [Development Setup](19_DEVELOPMENT.md) -> [Contributing](20_CONTRIBUTING.md) -> [Testing Framework](21_TESTING.md)

## Document Overview

| # | Document | Pages | Purpose |
|---|----------|-------|---------|
| 0 | Table of Contents | 1 | Navigation guide |
| 1 | Introduction | 4 | Overview and concepts |
| 2 | Installation | 5 | Setup instructions |
| 3 | Quick Start | 4 | 5-minute tutorial |
| 4 | Running Framework | 6 | Execution workflows |
| 5 | Configuration | 7 | Parameter tuning |
| 6 | Output Interpretation | 6 | Understanding results |
| 7 | Architecture | 6 | System design |
| 8 | Causal Graph | 6 | Graph structure |
| 9 | Inference Algorithm | 6 | Mathematical foundation |
| 10 | API Reference | 8 | Module documentation |
| 11 | Python Library | 5 | Library integration |
| 12 | Rust Integration | 5 | Performance features |
| 13 | Simulation & Testing | 6 | Test scenarios |
| 14 | Custom Scenarios | 5 | Domain-specific use |
| 15 | Performance Tuning | 5 | Optimization |
| 16 | Deployment | 7 | Production setup |
| 17 | Troubleshooting | 6 | Problem solving |
| 18 | Monitoring | 5 | Runtime observation |
| 19 | Development Setup | 5 | Local development |
| 20 | Contributing | 5 | Code contribution |
| 21 | Testing Framework | 5 | Test infrastructure |
| 22 | Glossary | 4 | Terminology |
| 23 | FAQ | 5 | Common questions |
| 24 | Bibliography | 3 | Academic references |
| | **TOTAL** | **~150 pages** | **Complete guide** |

## Converting to PDF

### Option 1: Using Pandoc

Install pandoc: https://pandoc.org/installing.html

```bash
# Generate single PDF from all documents
pandoc 00_TABLE_OF_CONTENTS.md 01_INTRODUCTION.md 02_INSTALLATION.md \
       03_QUICKSTART.md 04_RUNNING_FRAMEWORK.md 05_CONFIGURATION.md \
       06_OUTPUT_INTERPRETATION.md 07_ARCHITECTURE.md 08_CAUSAL_GRAPH.md \
       09_INFERENCE_ALGORITHM.md 10_API_REFERENCE.md 11_PYTHON_LIBRARY.md \
       12_RUST_INTEGRATION.md 13_SIMULATION.md 14_CUSTOM_SCENARIOS.md \
       15_PERFORMANCE.md 16_DEPLOYMENT.md 17_TROUBLESHOOTING.md \
       18_MONITORING.md 19_DEVELOPMENT.md 20_CONTRIBUTING.md \
       21_TESTING.md 22_GLOSSARY.md 23_FAQ.md 24_REFERENCES.md \
       -o aethelix_documentation.pdf
```

### Option 2: Using Python

```python
import os
import subprocess

docs = [
    "00_TABLE_OF_CONTENTS.md",
    "01_INTRODUCTION.md",
    "02_INSTALLATION.md",
    # ... all other documents
]

# Concatenate all documents
full_content = ""
for doc in docs:
    with open(doc, "r") as f:
        full_content += f.read() + "\n\n"

with open("FULL_DOCUMENTATION.md", "w") as f:
    f.write(full_content)

# Convert to PDF
subprocess.run([
    "pandoc",
    "FULL_DOCUMENTATION.md",
    "-o", "aethelix_documentation.pdf",
    "--toc",
    "--toc-depth=2",
    "-V", "papersize=a4",
    "-V", "geometry:margin=1in",
])
```

### Option 3: Using MkDocs

Create `mkdocs.yml`:

```yaml
site_name: Aethelix Documentation
site_description: Satellite Causal Inference Framework
site_author: Your Name
site_url: https://example.com

nav:
  - Home: index.md
  - Getting Started:
    - Introduction: "01_INTRODUCTION.md"
    - Installation: "02_INSTALLATION.md"
    - Quick Start: "03_QUICKSTART.md"
  - User Guide:
    - Running Framework: "04_RUNNING_FRAMEWORK.md"
    - Configuration: "05_CONFIGURATION.md"
    - Output: "06_OUTPUT_INTERPRETATION.md"
  # ... rest of structure

theme:
  name: material
  
plugins:
  - search
  - pdf-export

markdown_extensions:
  - toc
  - codehilite
```

Then:
```bash
mkdocs build
# PDF available in site/ directory
```

## File Structure

```
DOCUMENTATION/
+-- README.md                          <- You are here
+-- 00_TABLE_OF_CONTENTS.md
+-- 01_INTRODUCTION.md
+-- 02_INSTALLATION.md
+-- 03_QUICKSTART.md
+-- 04_RUNNING_FRAMEWORK.md
+-- 05_CONFIGURATION.md
+-- 06_OUTPUT_INTERPRETATION.md
+-- 07_ARCHITECTURE.md
+-- 08_CAUSAL_GRAPH.md
+-- 09_INFERENCE_ALGORITHM.md
+-- 10_API_REFERENCE.md
+-- 11_PYTHON_LIBRARY.md
+-- 12_RUST_INTEGRATION.md
+-- 13_SIMULATION.md
+-- 14_CUSTOM_SCENARIOS.md
+-- 15_PERFORMANCE.md
+-- 16_DEPLOYMENT.md
+-- 17_TROUBLESHOOTING.md
+-- 18_MONITORING.md
+-- 19_DEVELOPMENT.md
+-- 20_CONTRIBUTING.md
+-- 21_TESTING.md
+-- 22_GLOSSARY.md
+-- 23_FAQ.md
+-- 24_REFERENCES.md
```

## Version Info

- **Documentation Version**: 1.0
- **Last Updated**: January 2026
- **Aethelix Version**: 1.0
- **Status**: Complete & Production-Ready

## Support

For issues or questions:
- **GitHub Issues**: https://github.com/rudywasfound/aethelix/issues
- **Documentation**: See FAQ and Troubleshooting sections
- **Email**: Contact repository maintainers

## License

Documentation is provided under the same license as Aethelix.

---

**Start here:** [Introduction ->](01_INTRODUCTION.md)

**Or jump to:** [Table of Contents ->](00_TABLE_OF_CONTENTS.md)

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
