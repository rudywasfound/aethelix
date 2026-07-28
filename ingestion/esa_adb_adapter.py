"""
ESA Anomaly Detection Benchmark (ESA-ADB) data adapter for Aethelix.

Provides lightweight loading utilities for the ESA-ADB dataset (Zenodo
record 12528696). The full ESA-ADB is ~31 GB across 3 ESA missions with
224 channels and 1430 annotated events.

This adapter handles the preprocessed format produced by the ESA-ADB
GitHub repository (kplabs-pl/ESA-ADB).

Usage::

    from ingestion.esa_adb_adapter import load_esa_adb_mission
    channels, events = load_esa_adb_mission("data/esa/esa_adb/mission1")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)



@dataclass
class ESAADBChannel:
    """A single telemetry channel from the ESA-ADB dataset."""

    channel_id: str
    subsystem: str
    values: np.ndarray
    timestamps: np.ndarray
    sampling_rate: float

    @property
    def duration_hours(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0]) / 3600.0


@dataclass
class ESAADBEvent:
    """An annotated anomaly event from the ESA-ADB dataset."""

    event_id: int
    event_type: str             # "anomaly" or "rare"
    channel_id: str
    start_time: float
    end_time: float
    description: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


@dataclass
class ESAADBMission:
    """Container for a single ESA-ADB mission's data."""

    mission_name: str
    channels: Dict[str, ESAADBChannel] = field(default_factory=dict)
    events: List[ESAADBEvent] = field(default_factory=list)

    @property
    def anomaly_events(self) -> List[ESAADBEvent]:
        return [e for e in self.events if e.event_type == "anomaly"]

    @property
    def rare_events(self) -> List[ESAADBEvent]:
        return [e for e in self.events if e.event_type == "rare"]

    def get_channel_events(self, channel_id: str) -> List[ESAADBEvent]:
        return [e for e in self.events if e.channel_id == channel_id]



def load_esa_adb_mission(
    mission_dir: str | Path,
    mission_name: str = "unknown",
    max_channels: Optional[int] = None,
) -> ESAADBMission:
    """
    Load a single ESA-ADB mission from the preprocessed directory structure.

    The ESA-ADB preprocessed format (after running the ESA-ADB GitHub
    notebooks/data-prep scripts) typically contains:
      - ``data/`` directory with per-channel ``.csv`` files
      - ``labels.csv`` with anomaly annotations
      - ``anomaly_types.csv`` with event metadata

    Parameters
    ----------
    mission_dir : str | Path
        Path to the mission directory (e.g. ``data/esa/esa_adb/mission1``).
    mission_name : str
        Human-readable mission name for reporting.
    max_channels : int | None
        If given, limit the number of channels loaded (for quick testing).

    Returns
    -------
    ESAADBMission
        Loaded mission with channels and events.
    """
    mission_dir = Path(mission_dir)
    if not mission_dir.exists():
        raise FileNotFoundError(
            f"ESA-ADB mission directory not found: {mission_dir}\n"
            "Download from: https://zenodo.org/records/12528696\n"
            "Preprocess with: https://github.com/kplabs-pl/ESA-ADB"
        )

    mission = ESAADBMission(mission_name=mission_name)

    labels_path = mission_dir / "labels.csv"
    if labels_path.exists():
        mission.events = _load_events(labels_path)
        logger.info("Loaded %d events from %s", len(mission.events), labels_path)

    data_dir = mission_dir / "data"
    if data_dir.exists() and data_dir.is_dir():
        csv_files = sorted(data_dir.glob("*.csv"))
        if max_channels:
            csv_files = csv_files[:max_channels]

        for csv_file in csv_files:
            try:
                channel = _load_channel(csv_file)
                mission.channels[channel.channel_id] = channel
            except Exception as exc:
                logger.warning("Failed to load channel %s: %s", csv_file.name, exc)

        logger.info(
            "Loaded %d channels for mission '%s'",
            len(mission.channels), mission_name,
        )
    else:
        consolidated = mission_dir / "telemetry.csv"
        if consolidated.exists():
            mission.channels = _load_consolidated(consolidated, max_channels)
            logger.info(
                "Loaded %d channels from consolidated file for '%s'",
                len(mission.channels), mission_name,
            )
        else:
            logger.warning(
                "No data found in %s. Expected 'data/' directory or 'telemetry.csv'.",
                mission_dir,
            )

    return mission


def _load_events(labels_path: Path) -> List[ESAADBEvent]:
    """Load anomaly/rare event annotations from labels.csv."""
    events = []
    try:
        df = pd.read_csv(labels_path)

        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in ("chan_id", "channel", "channel_id"):
                col_map["channel"] = col
            elif lower in ("start_time", "start", "start_idx"):
                col_map["start"] = col
            elif lower in ("end_time", "end", "end_idx"):
                col_map["end"] = col
            elif lower in ("type", "anomaly_type", "event_type", "label"):
                col_map["type"] = col
            elif lower in ("description", "desc", "comment"):
                col_map["desc"] = col

        if "channel" not in col_map or "start" not in col_map:
            logger.warning(
                "labels.csv has unexpected columns: %s. Skipping event loading.",
                list(df.columns),
            )
            return events

        for idx, row in df.iterrows():
            event_type = "anomaly"
            if "type" in col_map:
                raw_type = str(row[col_map["type"]]).lower()
                if "rare" in raw_type:
                    event_type = "rare"

            events.append(ESAADBEvent(
                event_id=int(idx),
                event_type=event_type,
                channel_id=str(row[col_map["channel"]]),
                start_time=float(row[col_map["start"]]),
                end_time=float(row.get(col_map.get("end", ""), row[col_map["start"]])),
                description=str(row.get(col_map.get("desc", ""), "")),
            ))

    except Exception as exc:
        logger.warning("Failed to parse labels.csv: %s", exc)

    return events


def _load_channel(csv_path: Path) -> ESAADBChannel:
    """Load a single channel CSV file."""
    df = pd.read_csv(csv_path)

    channel_id = csv_path.stem

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        t0 = ts.iloc[0] if not pd.isna(ts.iloc[0]) else pd.Timestamp(0)
        timestamps = (ts - t0).dt.total_seconds().values.astype(float)
    else:
        timestamps = np.arange(len(df), dtype=float)

    value_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in ("timestamp", "time", "ts", "index", "id")]
    if not value_cols:
        raise ValueError(f"No numeric columns in {csv_path.name}")

    values = df[value_cols[0]].values.astype(float)

    if len(timestamps) > 1:
        dt = np.median(np.diff(timestamps))
        sampling_rate = 1.0 / max(dt, 1e-6)
    else:
        sampling_rate = 1.0

    return ESAADBChannel(
        channel_id=channel_id,
        subsystem="unknown",
        values=values,
        timestamps=timestamps,
        sampling_rate=sampling_rate,
    )


def _load_consolidated(csv_path: Path, max_channels: Optional[int]) -> Dict[str, ESAADBChannel]:
    """Load channels from a single consolidated CSV with one column per channel."""
    df = pd.read_csv(csv_path)
    channels = {}

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        t0 = ts.iloc[0] if not pd.isna(ts.iloc[0]) else pd.Timestamp(0)
        timestamps = (ts - t0).dt.total_seconds().values.astype(float)
        data_cols = [c for c in df.columns if c != "timestamp"]
    else:
        timestamps = np.arange(len(df), dtype=float)
        data_cols = list(df.columns)

    if max_channels:
        data_cols = data_cols[:max_channels]

    for col in data_cols:
        values = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        if len(timestamps) > 1:
            dt = np.median(np.diff(timestamps))
            sr = 1.0 / max(dt, 1e-6)
        else:
            sr = 1.0

        channels[col] = ESAADBChannel(
            channel_id=col,
            subsystem="unknown",
            values=values,
            timestamps=timestamps,
            sampling_rate=sr,
        )

    return channels


def get_mission_stats(mission: ESAADBMission) -> Dict:
    """Compute summary statistics for a loaded mission."""
    total_samples = sum(len(ch.values) for ch in mission.channels.values())
    return {
        "mission_name": mission.mission_name,
        "total_channels": len(mission.channels),
        "total_samples": total_samples,
        "total_events": len(mission.events),
        "anomaly_events": len(mission.anomaly_events),
        "rare_events": len(mission.rare_events),
        "total_duration_hours": sum(
            ch.duration_hours for ch in mission.channels.values()
        ),
    }
