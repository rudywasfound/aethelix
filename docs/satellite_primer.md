# Satellite Fault Management Primer: Aethelix Guide

Welcome to the Aethelix operational environment. This guide is designed for Ground Segment Engineers and Satellite Operators.

## The Operational Workflow

1. **Uplink/Ingestion**:
   - Aethelix ingests telemetry via the **Hardware Abstraction Layer (HAL)**.
   - For flight operations, use the **CCSDS Adapter** to pipe raw Space Packets (CCSDS 133.0-B) directly into the engine.
   - For mission reconstruction, use the **CSV Adapter**.

2. **Automated Anomaly Detection**:
   - Aethelix uses **Sliding Window Normalization**. It learns the "normal" variance of your specific satellite over the last 50-100 ticks.
   - No hard-coded thresholds are required, though a 15% sensitivity is recommended for noisy channels.

3. **Causal Reasoning**:
   - When deviations are detected, the **Stateful Root Cause Ranker** activates.
   - It traces evidence back through the Causal DAG to identify the most likely root cause.
   - **Soft Streak Recovery**: Aethelix maintains a "memory" of faults. A single noisy tick will not reset the diagnosis.

4. **Response Strategy**:
   - Aethelix provides a **3-Tier Action Plan** for every detected fault:
     - **Immediate**: Actions to stabilize the spacecraft.
     - **Short-term**: Diagnostic steps for the next orbital pass.
     - **Escalation**: Triggers for safe-hold or hardware swap.

## Understanding the Dashboard

- **Suppressed Alarms**: Represents the number of secondary sensor alarms that Aethelix correctly identified as "consequential" to a single root cause.
- **Lead Time Advantage**: The time gained by Aethelix detecting the fault "sub-threshold" versus a standard 15% alarm system.
- **Causal Vector Space**: A live visualization of fault propagation through your satellite's subsystems.

## Best Practices
- **Sensor Faults**: If a sensor goes to zero or NaN, Aethelix flags it as a `Sensor Fault`. Do not interpret this as a physical failure unless confirmed by cross-subsystem evidence.
- **Eclipse Transitions**: Aethelix is eclipse-aware. It suppresses solar-panel alarms during UMBRA to avoid false positives during normal orbital transitions.
