# ECSS Fault Mode Mapping

Aethelix aligns its diagnostic output with the **ECSS-E-ST-10-04C** (Space Engineering: Space Environment) and **ECSS-M-ST-30-01C** (Risk Management) standards.

This mapping ensures that Aethelix reports can be directly ingested into agency FMECA (Failure Mode, Effects, and Criticality Analysis) databases.

## EPS (Electrical Power Subsystem)

| Aethelix Identifier | ECSS Fault ID | Description |
|:---|:---|:---|
| `solar_degradation` | **EPS-FM-001** | Solar Array Power Output Below Nominal |
| `battery_aging` | **EPS-FM-003** | Battery Cell Capacity Degradation |
| `pcdu_regulator_failure` | **EPS-FM-007** | Power Control and Distribution Unit Regulator Fault |

## TCS (Thermal Control Subsystem)

| Aethelix Identifier | ECSS Fault ID | Description |
|:---|:---|:---|
| `battery_heatsink_failure` | **TCS-FM-002** | Battery Interface Thermal Resistance Increase |
| `payload_radiator_degradation` | **TCS-FM-005** | Surface Emissivity Loss / Radiator Fouling |

## ADCS (Attitude Determination & Control)

| Aethelix Identifier | ECSS Fault ID | Description |
|:---|:---|:---|
| `wheel_friction` | **ADC-FM-012** | Reaction Wheel Bearing Friction Increase |
| `gyro_drift` | **ADC-FM-005** | Gyroscope Bias Stability Out of Spec |

## PROP (Propulsion Subsystem)

| Aethelix Identifier | ECSS Fault ID | Description |
|:---|:---|:---|
| `thruster_valve_fault` | **PRP-FM-008** | Thruster Valve Stiction / Leakage |

## Implementation in Aethelix
ECSS identifiers are embedded as metadata within the `CausalGraph` definition. When a diagnosis is generated, the identifier is surfaced in the `RootCauseHypothesis` report, enabling automated cross-referencing with Ground Segment mission databases.
