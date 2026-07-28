# Physics Foundation: Why It's Not Guessing

Aethelix is backed by real aerospace physics, not machine learning pattern recognition. This document explains the physics equations that power the inference engine.

## Core Principle

**Aethelix is deterministic engineering, not probabilistic guessing.**

When the causal graph traces:
```
Solar degradation -> Reduced solar input -> Low battery charge -> High temperature
```

Each arrow represents a physics equation that MUST hold, not a learned correlation.

## Power System Physics

### Energy Balance Equation

At the core of every satellite power system:

```
Energy from sun = Stored in battery + Energy consumed by payload + Losses

P_solar = dQ/dt * eta_charge + P_payload + P_loss
```

Where:
- P_solar: Solar array output power (Watts)
- dQ/dt: Battery charging rate (Amp-hours per second)
- eta_charge: Charging efficiency (0-1)
- P_payload: Payload power consumption
- P_loss: Resistance losses in bus and wiring

### What This Means for Diagnosis

When solar panels degrade 30%:

P_solar decreases by 30% (deterministic - no guessing)
  |
  (down)
  |
dQ/dt must decrease (physics equation)
  |
  (down)
  |
Battery charge accumulates slower (inevitable consequence)

This isn't pattern matching from training data. It's thermodynamics.

## Battery State Equations

### State of Charge Dynamics

The battery state changes according to:

```
dSOC/dt = (I_charge - I_discharge) / Q_capacity

Where:
- SOC: State of charge (0-100%)
- I_charge: Current flowing into battery (Amps)
- I_discharge: Current flowing out
- Q_capacity: Battery capacity (Amp-hours)
```

### Example Calculation

Nominal operation:
- Solar provides 500W at 28V = 17.8 Amps available for charging
- Payload consumes 10A
- Net charging current = 17.8 - 10 = 7.8A
- Battery capacity = 100 Ah
- dSOC/dt = (7.8 / 100) * 3600 sec/hour = 281% per hour (reaches 100% in ~20 min)

With 30% solar degradation:
- Solar provides 350W at 28V = 12.5 Amps available
- Net charging current = 12.5 - 10 = 2.5A
- dSOC/dt = (2.5 / 100) * 3600 = 90% per hour (takes ~67 min to charge)

**This is physics, not a pattern:**

The battery MUST charge slower if less power is available. There's no way around it.

## Voltage Drop Physics

### Ohm's Law in Series

Bus voltage is determined by:

```
V_bus = V_battery - I * R_wiring - V_regulation_drop
```

As battery voltage falls (from reduced charging):

```
V_battery(t) = V_nominal * f(SOC(t))

Where f(SOC) is nonlinear:
- SOC = 100%: V = 28.5V
- SOC = 75%: V = 27.8V
- SOC = 50%: V = 26.5V
- SOC = 25%: V = 24.0V
- SOC = 0%: V = 22.0V (cutoff)
```

When solar degrades:
1. SOC drops slower (less charging current)
2. During eclipse, SOC drops faster (no charging)
3. Average SOC decreases
4. V_battery decreases
5. V_bus violates minimum threshold (10V minimum)

**Again: Pure physics. No guessing involved.**

## Thermal Physics

### Heat Transfer Equation

Battery temperature is governed by:

```
dT/dt = (Q_in - Q_rad - Q_cond) / (m * c)

Where:
- Q_in: Heat generated inside battery (resistive heating from discharge)
- Q_rad: Heat radiated to space (Stefan-Boltzmann: Q_rad = sigma * A * epsilon * (T^4 - T_space^4))
- Q_cond: Heat conducted through thermal connections
- m: Battery mass
- c: Specific heat capacity
```

### Power-Thermal Coupling

With solar degradation:

1. Battery can't charge fully during sun periods
2. During eclipse, battery discharges heavily
3. Discharge current drives resistive heating: Q = I^2 * R
4. Higher current = more heat
5. Higher temperature damages battery

**Mathematical chain:**

```
Solar degradation
  |
  (down) P_solar decreases
  |
  (down) Battery can't charge
  |
  (down) SOC drops during eclipse
  |
  (down) Discharge current I increases (payload needs more amperes from low-SOC battery)
  |
  (down) Heat Q = I^2 * R increases
  |
  (down) Temperature rises (Stefan-Boltzmann radiation can't dissipate fast enough)
```

Each step follows from physics equations. No pattern recognition needed.

## Why This Defeats Machine Learning

### ML Problem 1: Correlation Confusion

ML might learn: "Solar low AND Battery hot means solar failure (95% of training data)"

But what if:
- Battery heating element fails -> high heat with normal solar
- ML system confuses this as solar degradation
- Recommends solar panel rotation (wrong action)

**Physics doesn't get confused:**
- Solar heating element failure: Direct causal path solar -> heating element -> temp (no effect on battery charge)
- Solar panel failure: solar -> charge -> temp (cascading effects match the data)

### ML Problem 2: Extrapolation

ML trained on 30% solar loss might fail at 60% loss or at different temperatures.

**Physics works everywhere:**
- Equations work at 30%, 60%, 90% loss
- Work at -40C, +50C, or any temperature
- Scale from small cubesat to large geostationary satellite

### ML Problem 3: No Data

You don't have thousands of satellite failures to train on.

**Physics needs no training data:**
- Use the equations
- Done

## Causal Graph Validation

The causal graph is validated against physics equations:

```
Does edge "Solar degradation -> Battery charge" exist?
Check: Does physics predict this?
- Solar degradation -> reduced P_solar
- Reduced P_solar -> reduced dQ/dt
- Reduced dQ/dt -> lower SOC
Answer: YES, keep edge in graph

Does edge "Battery aging -> Solar input" exist?
Check: Does physics predict this?
- Battery aging doesn't affect solar panels
Answer: NO, remove edge from graph
```

Every edge in the causal graph is validated against aerospace physics.

## Bayesian Inference Over Physics, Not Instead Of

Aethelix uses Bayes' theorem to combine:

1. **Physics predictions**: "If solar degrades 30%, we MUST see X behavior"
2. **Observed deviations**: "We actually observed Y behavior"
3. **Consistency scoring**: "How well does physics prediction match observation?"

Bayes tells us: P(solar degradation | observation) is high if physics predictions match.

This is NOT ML pattern matching. It's:

```
Hypothesis: Solar degradation
Prediction from physics: "Solar input drop, battery charge drop, temperature rise"
Observation: "Solar input drop by 45W, battery charge drop by 20%, temperature rise by 3C"
Consistency: "Prediction matches observation closely"
Conclusion: P(solar degradation | data) = 46%

Hypothesis: Sensor bias
Prediction from physics: "All readings biased together (no physical correlation)"
Observation: "Multiple sensors show correlated deviations (strong causal chain)"
Consistency: "Prediction doesn't match observation"
Conclusion: P(sensor bias | data) = 5%
```

## Equations Used in Aethelix

### Power System (simulator/power.py)

```
Battery SOC dynamics:
  dSOC/dt = (I_charge - I_load) / Q_capacity

Battery voltage model:
  V(SOC) = V_nominal * (0.8 + 0.2 * SOC / 100)

Bus voltage:
  V_bus = V_battery - I * R_bus

Solar power degradation:
  P_solar(t) = P_nominal * (1 - degradation_factor) for t > t_fault
```

### Thermal System (simulator/thermal.py)

```
Battery heat generation:
  Q_gen = I^2 * R_internal + P_parasitic

Radiative heat loss (Stefan-Boltzmann):
  Q_rad = sigma * A * epsilon * (T^4 - T_space^4)

Temperature dynamics:
  dT/dt = (Q_gen - Q_rad) / (m * c_p)

Thermal-electrical coupling:
  Battery efficiency = 1 - (T - T_nominal) * thermal_coeff
```

## Why Operators Should Trust This

1. **Physics is proven**: These equations work in practice (used by ISRO, NASA, ESA)
2. **Transparent**: Every conclusion traces back to specific equations
3. **Auditable**: Engineers can verify the graph against textbooks
4. **Safe**: Can't hallucinate false diagnoses from pattern matching
5. **Offline**: Works without internet, machine learning servers, or cloud dependencies

## Next Steps

- See implementation: [simulator/power.py](../simulator/power.py)
- See thermal model: [simulator/thermal.py](../simulator/thermal.py)
- See real examples: [Real Examples](07_REAL_EXAMPLES.md)
- Run it yourself: [Quick Start](03_QUICKSTART.md)

---

**This is aerospace engineering, not data science guessing.**

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
