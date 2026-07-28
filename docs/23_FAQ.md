# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is Aethelix used for?

A: Aethelix diagnoses root causes of satellite failures. Unlike simple threshold-based systems, it uses causal reasoning to distinguish between causes and their effects. For example, if solar panels degrade, battery temperature may rise as a secondary effect - Aethelix correctly attributes both to solar degradation, not battery thermal issues.

### Q: Do I need to be a researcher to use Aethelix?

A: No. If you can install Python and run a command, you can use Aethelix. We provide:
- Simple CLI (`python main.py`)
- Python library for integration
- Detailed documentation
- Example scenarios

For advanced customization (adding subsystems, modifying the graph), some Python knowledge helps, but you can start simple.

### Q: Is Aethelix a machine learning model?

A: No. Aethelix uses explicit causal graphs backed by aerospace physics equations.

Key differences from ML:

**Transparent**: You can see exactly why it makes each decision

**Explainable**: Every diagnosis includes the physics mechanism and supporting evidence

**No black box**: No hidden neural network parameters or learned weights

**Works without training data**: Uses physics equations, not learned patterns

**Deterministic**: Same inputs always produce same reasoning (not probabilistic guessing)

### Q: How accurate is Aethelix?

A: Accuracy depends on:
1. **Quality of causal graph**: How well does it represent reality?
2. **Quality of data**: Are measurements accurate and complete?
3. **Similarity to design**: Works best for scenarios matching the graph

In controlled tests with simulated data: 85-95% accuracy for single faults, 70-85% for multi-fault scenarios.

**Real accuracy depends on your specific satellite and environment.**

### Q: How does Aethelix differ from simple monitoring?

A: 

| Feature | Threshold | Correlation | Causal Inference |
|---------|-----------|-------------|------------------|
| Find anomalies | [OK] | [OK] | [OK] |
| Multi-fault diagnosis | [NO] | [NO] | [OK] |
| Explainability | [OK] | [OK] | [OK] |
| Causal reasoning | [NO] | [NO] | [OK] |
| Confidence scores | [NO] | [NO] | [OK] |

## Installation Questions

### Q: Do I need Rust installed?

A: No. Rust is optional for high-performance features. Pure Python works fine for most use cases.

### Q: What Python versions are supported?

A: Python 3.8+. We test on:
- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

### Q: Can I use Anaconda instead of venv?

A: Yes. Replace:
```bash
python -m venv .venv
source .venv/bin/activate
```

With:
```bash
conda create -n aethelix python=3.10
conda activate aethelix
```

### Q: What if pip install fails?

A: See [Troubleshooting](17_TROUBLESHOOTING.md#pip-installation-fails). Common solutions:
- Upgrade pip: `pip install --upgrade pip`
- Clear cache: `pip install --no-cache-dir -r requirements.txt`
- Use system Python package manager (apt, brew, etc.)

## Running Questions

### Q: How long does a run take?

A: Typically:
- 24-hour simulation at 0.1 Hz: ~10-15 seconds total
- 12-hour simulation at 1 Hz: ~7-10 seconds total

Breakdown:
- Simulation: 3-5 sec
- Analysis: <1 sec
- Visualization: 1-2 sec
- Inference: 1-2 sec

### Q: Can I speed it up?

A: Yes. See [Performance Tuning](15_PERFORMANCE.md). Options:
- Reduce duration: 24h -> 12h (saves ~2 sec)
- Increase sampling interval: 0.1 Hz -> 1 Hz (less data)
- Use Rust core: ~10x speedup
- Parallelize: Process multiple scenarios simultaneously

### Q: Can I use real telemetry data?

A: Currently, Aethelix uses simulated data. To use real data:

```python
# Load your telemetry data
import numpy as np
from main import CombinedTelemetry

time_series = np.load("your_telemetry.npy")
nominal = CombinedTelemetry.from_array(time_series)
# ... rest of workflow
```

See [Custom Scenarios](14_CUSTOM_SCENARIOS.md) for details.

### Q: What if the output doesn't match my expectations?

A: Check:
1. **Nominal baseline correct?** `print(nominal.solar_input.mean())`
2. **Fault severity high enough?** Try `solar_factor=0.3`
3. **Threshold too high?** Try `deviation_threshold=0.10`
4. **Graph applicable to your system?** Check [Causal Graph](08_CAUSAL_GRAPH.md)

## Configuration Questions

### Q: How do I set custom parameters?

A: Three ways:

**1. Direct parameters:**
```python
sim = PowerSimulator(duration_hours=12)
```

**2. Configuration file:**
```yaml
# aethelix_config.yaml
simulation:
  duration_hours: 12
  sampling_rate_hz: 0.1
```

**3. Environment variables:**
```bash
export PRAVAHA_DURATION_HOURS=12
```

### Q: What do prior probabilities do?

A: They bias the inference toward certain causes. Example:

```python
priors = {
    "solar_degradation": 0.5,   # 50% prior (very likely)
    "battery_aging": 0.3,
    "battery_thermal": 0.15,
    "sensor_bias": 0.05,
}
```

Use when:
- Historical data shows certain faults are more common
- Satellite design makes certain failures more likely
- You want to penalize or favor certain hypotheses

### Q: What does consistency_weight do?

A: Controls how much the causal graph structure affects scoring.

- **High consistency_weight** (e.g., 2.0): Favor hypotheses that fit the graph well
- **Low consistency_weight** (e.g., 0.5): Rely more on raw evidence

Use high values when:
- You trust the graph structure
- You want conservative, consistent diagnoses

Use low values when:
- You're unsure about the graph
- You want raw data to dominate

## Output Questions

### Q: What does probability mean?

A: Posterior probability - given the observed data, what's the chance this is the root cause?

If solar_degradation has P=46%, it means:
- Most likely cause (compared to alternatives)
- But not certain (not 90%+)
- Need more data to be sure

Probabilities sum to 100% across all hypotheses.

### Q: What does confidence mean?

A: Certainty in the probability estimate, not in the cause itself.

- **High confidence + high probability**: "Probably this cause, we're sure"
- **High confidence + low probability**: "Probably not this, we're sure"
- **Low confidence + high probability**: "Maybe this, but evidence is weak"
- **Low confidence + low probability**: "Very uncertain about this one"

### Q: Why do multiple causes have similar probability?

A: Causes have similar effects (ambiguity). This is actually correct - the evidence doesn't clearly distinguish them.

**Solution**: Collect more data or request specific diagnostics to differentiate.

### Q: What's a good confidence threshold?

A: Depends on your use case:

- **Real-time monitoring**: >70% confidence (trust it)
- **Forensic analysis**: >50% confidence (investigate)
- **Research**: >30% confidence (publish with caveats)
- **Critical systems**: >90% confidence (very conservative)

## Data & Integration Questions

### Q: Can I integrate with existing monitoring systems?

A: Yes. Aethelix outputs JSON/CSV:

```python
import json

output = {
    "hypotheses": [
        {
            "name": h.name,
            "probability": h.probability,
            "confidence": h.confidence,
        }
        for h in hypotheses
    ],
}

with open("diagnosis.json", "w") as f:
    json.dump(output, f)
```

Then ingest into your system via API, message queue, or file polling.

### Q: How do I handle missing data?

A: Currently, Aethelix requires complete telemetry. For gaps:

1. **Interpolate**: Use scipy or pandas
```python
import pandas as pd
df = pd.DataFrame({"measurement": data})
df_filled = df.interpolate()
```

2. **Use Rust Kalman filter**: Estimates hidden states during gaps

See [Rust Integration](12_RUST_INTEGRATION.md).

### Q: Can I add custom fault modes?

A: Yes. Modify `causal_graph/graph_definition.py`:

```python
class CustomGraph(CausalGraph):
    def __init__(self):
        super().__init__()
        # Add your nodes and edges
        self.add_node("my_fault", "root_cause")
        self.add_edge("my_fault", "some_observable", weight=0.8)
```

See [Causal Graph](08_CAUSAL_GRAPH.md) for details.

## Deployment Questions

### Q: Can I deploy to production?

A: Yes, Aethelix is production-ready. See [Deployment](16_DEPLOYMENT.md) for:
- Docker containerization
- Performance optimization
- Monitoring and logging
- Scaling strategies

### Q: Is Aethelix cloud-compatible?

A: Yes. Deploy to:
- AWS Lambda (serverless)
- Kubernetes (containerized)
- Google Cloud / Azure
- Traditional servers

See [Deployment](16_DEPLOYMENT.md) for recipes.

### Q: What are resource requirements?

A: Minimal:
- RAM: 100 MB typical
- CPU: Single core sufficient
- Disk: ~50 MB for code + dependencies
- Network: Not required (works offline)

### Q: How do I monitor a deployed instance?

A: See [Monitoring](18_MONITORING.md). Aethelix can emit:
- Diagnosis results to log files
- Metrics (probability, confidence) to monitoring systems
- Alerts when high-probability faults detected

## Troubleshooting Questions

### Q: The plots aren't showing

A: Plots are saved to files, not displayed in terminal. Check:
```bash
ls -la output/comparison.png
ls -la output/residuals.png
```

To display:
```python
import matplotlib.pyplot as plt
plt.show()
```

### Q: All hypotheses have equal probability

A: Causes have identical evidence. This means:
1. Evidence is ambiguous (correct diagnosis)
2. Graph is disconnected (might need refinement)
3. Faults are too subtle (increase severity)

**Solution**: Collect more/better data or inject stronger faults.

### Q: I get different results each time

A: Aethelix's results are deterministic (no randomness). If different:
1. Your input data changed
2. You changed parameters
3. You're comparing different scenarios

Check logs and parameters carefully.

### Q: Inference is slow

A: Check [Performance Tuning](15_PERFORMANCE.md):
- Reduce simulation duration
- Increase sampling interval
- Use Rust core for high-frequency data
- Run on faster hardware

## Advanced Questions

### Q: Can I modify the causal graph?

A: Yes, see [Causal Graph](08_CAUSAL_GRAPH.md). You can:
- Add new nodes (root causes, intermediates, observables)
- Add edges (causal mechanisms)
- Change edge weights
- Customize node descriptions

### Q: Can I use different inference algorithms?

A: Currently, Aethelix uses Bayesian graph traversal. To experiment:
1. Fork the repository
2. Modify `RootCauseRanker` class
3. Implement alternative algorithm
4. See [Contributing](20_CONTRIBUTING.md)

### Q: Can I contribute improvements?

A: Absolutely. See [Contributing](20_CONTRIBUTING.md) for:
- Code of conduct
- Pull request process
- Testing requirements
- Documentation guidelines

### Q: How is Aethelix licensed?

A: Check LICENSE file in repository for details.

## Getting Help

**Still have questions?**

1. Check [Table of Contents](00_TABLE_OF_CONTENTS.md) for more detailed docs
2. Search [Troubleshooting](17_TROUBLESHOOTING.md)
3. Review example code in `tests/` directory
4. File an issue: https://github.com/rudywasfound/aethelix/issues
5. Check project README: https://github.com/rudywasfound/aethelix

---

**Continue to:** [Bibliography ->](24_REFERENCES.md)

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
