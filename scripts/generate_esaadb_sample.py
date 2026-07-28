#!/usr/bin/env python3
"""
generate_esaadb_sample.py — Synthetic Multi-Mission ESA-ADB Dataset Generator.

Constructs a representative multi-subsystem mission dataset under data/esa/esa_adb/mission1/
matching the schema expected by ingestion/esa_adb_adapter.py without requiring a 31 GB zip download.
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
OUTPUT_DIR = os.path.join(REPO_ROOT, "data/esa/esa_adb/mission1")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")


def main():
    logger.info("Generating synthetic ESA-ADB multi-mission dataset (mission1)...")
    os.makedirs(DATA_DIR, exist_ok=True)

    np.random.seed(42)
    n_samples = 5000
    timestamps = np.arange(n_samples, dtype=float)

    channels = {
        "adcs_mag_x": {"freq": 0.0002, "amp": 40.0, "base": 20.0, "noise": 1.5},
        "adcs_mag_y": {"freq": 0.00025, "amp": 35.0, "base": -15.0, "noise": 1.5},
        "adcs_mag_z": {"freq": 0.00018, "amp": 45.0, "base": 10.0, "noise": 1.5},
        "adcs_gyro_x": {"freq": 0.0005, "amp": 2.0, "base": 0.0, "noise": 0.3},
        "power_solar_v": {"freq": 0.0002, "amp": 12.0, "base": 28.0, "noise": 0.4},
        "power_battery_v": {"freq": 0.0002, "amp": 3.0, "base": 24.0, "noise": 0.2},
        "power_bus_i": {"freq": 0.0003, "amp": 5.0, "base": 15.0, "noise": 0.5},
        "thermal_battery_temp": {"freq": 0.00015, "amp": 8.0, "base": 20.0, "noise": 0.3},
        "thermal_panel_temp": {"freq": 0.0002, "amp": 25.0, "base": 5.0, "noise": 0.8},
        "thermal_payload_temp": {"freq": 0.00015, "amp": 5.0, "base": 22.0, "noise": 0.2},
    }

    events = [
        {"channel": "adcs_mag_x", "start": 1000.0, "end": 1200.0, "type": "anomaly", 
         "desc": "Magnetometer bias drift from solar array magnetic moment"},
        {"channel": "power_battery_v", "start": 2500.0, "end": 2800.0, "type": "anomaly", 
         "desc": "Battery cell voltage sag under high payload load"},
        {"channel": "thermal_battery_temp", "start": 4000.0, "end": 4300.0, "type": "anomaly", 
         "desc": "Battery thermal runaway during rapid eclipse transition"},
        {"channel": "adcs_gyro_x", "start": 3200.0, "end": 3300.0, "type": "rare", 
         "desc": "Transient rate sensor noise spike"},
    ]

    for ch_name, params in channels.items():
        val = (params["base"] + 
               params["amp"] * np.sin(2 * np.pi * params["freq"] * timestamps) + 
               np.random.normal(0, params["noise"], n_samples))
        
        for ev in events:
            if ev["channel"] == ch_name and ev["type"] == "anomaly":
                idx = np.where((timestamps >= ev["start"]) & (timestamps <= ev["end"]))[0]
                if "mag" in ch_name:
                    val[idx] += 35.0
                elif "power" in ch_name:
                    val[idx] -= 6.0
                elif "thermal" in ch_name:
                    val[idx] += 15.0
            elif ev["channel"] == ch_name and ev["type"] == "rare":
                idx = np.where((timestamps >= ev["start"]) & (timestamps <= ev["end"]))[0]
                val[idx] += np.random.normal(0, 4.0, len(idx))

        df = pd.DataFrame({"timestamp": timestamps, "value": val})
        csv_path = os.path.join(DATA_DIR, f"{ch_name}.csv")
        df.to_csv(csv_path, index=False)
        logger.debug(f"Saved channel {ch_name} to {csv_path}")

    labels_df = pd.DataFrame(events)
    labels_path = os.path.join(OUTPUT_DIR, "labels.csv")
    labels_df.to_csv(labels_path, index=False)
    
    readme_path = os.path.join(REPO_ROOT, "data/esa/esa_adb/README.md")
    with open(readme_path, "w") as f:
        f.write("# ESA-ADB Synthetic Multi-Mission Sample\n\n")
        f.write("Generated automatically by `scripts/generate_esaadb_sample.py` for testing.\n")
        f.write("Contains 10 channels across ADCS, Power, and Thermal subsystems with annotated anomalies.\n")

    logger.info(f"Successfully generated ESA-ADB mission1 dataset in {OUTPUT_DIR}/")
    logger.info(f"  Channels: {len(channels)}")
    logger.info(f"  Samples per channel: {n_samples:,}")
    logger.info(f"  Annotated events: {len(events)}")


if __name__ == "__main__":
    main()
