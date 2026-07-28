"""
OPS-SAT Anomaly Detection (OPSSAT-AD) data adapter for Aethelix.

Parses the OPSSAT-AD ``segments.csv`` format from Zenodo (record 10850228)
and converts it into Aethelix-compatible data structures for both anomaly
detection (SlidingWindowDetector) and root-cause ranking (RootCauseRanker).

Dataset schema (segments.csv):
    timestamp  — ISO date string
    channel    — telemetry channel code (e.g. CADC0872)
    value      — float measurement
    label      — "anomaly" or "nominal"
    anomaly    — 0 (nominal) or 1 (anomalous) ground-truth flag
    segment    — consecutive integer segment ID
    sampling   — sampling rate of the segment
    train      — 1 = training, 0 = testing

Channel mapping (9 channels):
    CADC0872 → I_B_FB_MM_0 → mag_x    (magnetometer X)
    CADC0873 → I_B_FB_MM_1 → mag_y    (magnetometer Y)
    CADC0874 → I_B_FB_MM_2 → mag_z    (magnetometer Z)
    CADC0884 → I_PD1_THETA → pd1_theta (photo diode 1)
    CADC0886 → I_PD2_THETA → pd2_theta (photo diode 2)
    CADC0888 → I_PD3_THETA → pd3_theta (photo diode 3)
    CADC0890 → I_PD4_THETA → pd4_theta (photo diode 4)
    CADC0892 → I_PD5_THETA → pd5_theta (photo diode 5)
    CADC0894 → I_PD6_THETA → pd6_theta (photo diode 6)

Usage::

    from ingestion.opssat_adapter import load_opssat_segments, CHANNEL_MAP
    segments = load_opssat_segments("data/esa/opssat/segments.csv")

    for seg in segments:
        print(seg.segment_id, seg.channel, seg.is_train, seg.has_anomaly)
        print(seg.timestamps.shape, seg.values.shape, seg.labels.shape)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CHANNEL_MAP: Dict[str, str] = {
    "CADC0872": "mag_x",
    "CADC0873": "mag_y",
    "CADC0874": "mag_z",
    "CADC0884": "pd1_theta",
    "CADC0886": "pd2_theta",
    "CADC0888": "pd3_theta",
    "CADC0890": "pd4_theta",
    "CADC0892": "pd5_theta",
    "CADC0894": "pd6_theta",
}

CHANNEL_MAP_REV: Dict[str, str] = {v: k for k, v in CHANNEL_MAP.items()}

OBSERVABLE_MAP: Dict[str, str] = {
    "mag_x":     "mag_x_measured",
    "mag_y":     "mag_y_measured",
    "mag_z":     "mag_z_measured",
    "pd1_theta": "pd1_theta_measured",
    "pd2_theta": "pd2_theta_measured",
    "pd3_theta": "pd3_theta_measured",
    "pd4_theta": "pd4_theta_measured",
    "pd5_theta": "pd5_theta_measured",
    "pd6_theta": "pd6_theta_measured",
}

MAGNETOMETER_CHANNELS = {"mag_x", "mag_y", "mag_z"}
PHOTODIODE_CHANNELS = {
    "pd1_theta", "pd2_theta", "pd3_theta",
    "pd4_theta", "pd5_theta", "pd6_theta",
}



@dataclass
class TelemetrySegment:
    """A single contiguous telemetry segment from the OPSSAT-AD dataset."""

    segment_id: int
    channel: str            # Aethelix name (e.g. "mag_x")
    channel_code: str       # Original OPSSAT code (e.g. "CADC0872")
    is_train: bool
    sampling_rate: float
    timestamps: np.ndarray
    values: np.ndarray
    labels: np.ndarray

    @property
    def has_anomaly(self) -> bool:
        """Whether this segment contains any anomalous samples."""
        return bool(np.any(self.labels > 0))

    @property
    def anomaly_fraction(self) -> float:
        """Fraction of samples labelled anomalous."""
        if len(self.labels) == 0:
            return 0.0
        return float(np.mean(self.labels > 0))

    @property
    def duration_seconds(self) -> float:
        """Duration of the segment in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def to_streaming_rows(self) -> List[Dict[str, float]]:
        """
        Convert to a list of single-channel row dicts suitable for
        SlidingWindowDetector.process_tick().
        """
        rows = []
        for i in range(len(self.values)):
            rows.append({
                "timestamp": float(self.timestamps[i]),
                self.channel: float(self.values[i]),
            })
        return rows


@dataclass
class MultiChannelWindow:
    """
    A time-aligned multi-channel snapshot for causal analysis.

    Groups concurrent segments from different channels into a single
    window that RootCauseRanker can analyze.
    """

    window_id: int
    channels: Dict[str, np.ndarray] = field(default_factory=dict)
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    time: Optional[np.ndarray] = None

    @property
    def has_anomaly(self) -> bool:
        return any(np.any(lbl > 0) for lbl in self.labels.values())

    @property
    def anomalous_channels(self) -> List[str]:
        return [ch for ch, lbl in self.labels.items() if np.any(lbl > 0)]



def load_opssat_segments(
    csv_path: str | Path,
    channels: Optional[List[str]] = None,
) -> List[TelemetrySegment]:
    """
    Load OPSSAT-AD segments.csv into a list of TelemetrySegment objects.

    Parameters
    ----------
    csv_path : str | Path
        Path to the segments.csv file.
    channels : list[str] | None
        If given, only load these Aethelix channel names (e.g. ["mag_x"]).
        Default: load all 9 channels.

    Returns
    -------
    list[TelemetrySegment]
        One TelemetrySegment per (channel, segment) pair, sorted by
        (segment_id, channel).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"OPSSAT-AD segments.csv not found at {csv_path}.\n"
            f"Run: bash scripts/download_esa_datasets.sh"
        )

    logger.info("Loading OPSSAT-AD segments from %s ...", csv_path)
    df = pd.read_csv(csv_path)

    expected_cols = {"timestamp", "channel", "value", "anomaly", "segment", "sampling", "train"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"segments.csv is missing columns: {missing}. "
            f"Got: {list(df.columns)}"
        )

    df["aethelix_channel"] = df["channel"].map(CHANNEL_MAP)

    known = df["aethelix_channel"].notna()
    if not known.all():
        unknown_codes = df.loc[~known, "channel"].unique()
        logger.warning(
            "Skipping %d rows with unknown channel codes: %s",
            (~known).sum(), unknown_codes,
        )
    df = df[known].copy()

    if channels is not None:
        df = df[df["aethelix_channel"].isin(channels)]

    df["ts_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")

    segments: List[TelemetrySegment] = []

    for (seg_id, ch_code), group in df.groupby(["segment", "channel"]):
        group = group.sort_values("timestamp")

        aethelix_ch = CHANNEL_MAP.get(ch_code, ch_code)

        ts = group["ts_parsed"].values.astype("datetime64[ns]")
        if len(ts) > 0 and not pd.isna(ts[0]):
            t0 = ts[0]
            ts_seconds = (ts - t0).astype("timedelta64[ms]").astype(float) / 1000.0
        else:
            ts_seconds = np.arange(len(group), dtype=float)

        segments.append(TelemetrySegment(
            segment_id=int(seg_id),
            channel=aethelix_ch,
            channel_code=ch_code,
            is_train=bool(group["train"].iloc[0]),
            sampling_rate=float(group["sampling"].iloc[0]),
            timestamps=ts_seconds,
            values=group["value"].values.astype(float),
            labels=group["anomaly"].values.astype(int),
        ))

    segments.sort(key=lambda s: (s.segment_id, s.channel))

    logger.info(
        "Loaded %d segments (%d train, %d test) across %d channels.",
        len(segments),
        sum(1 for s in segments if s.is_train),
        sum(1 for s in segments if not s.is_train),
        len({s.channel for s in segments}),
    )

    return segments


def group_segments_by_id(
    segments: List[TelemetrySegment],
) -> Dict[int, List[TelemetrySegment]]:
    """Group segments by segment_id for multi-channel analysis."""
    groups: Dict[int, List[TelemetrySegment]] = {}
    for seg in segments:
        groups.setdefault(seg.segment_id, []).append(seg)
    return groups


def segments_to_windows(
    segments: List[TelemetrySegment],
    test_only: bool = True,
) -> List[MultiChannelWindow]:
    """
    Group segments by segment_id into MultiChannelWindow objects
    suitable for causal analysis.
    """
    groups = group_segments_by_id(segments)
    windows: List[MultiChannelWindow] = []

    for seg_id, seg_list in sorted(groups.items()):
        if test_only and all(s.is_train for s in seg_list):
            continue

        window = MultiChannelWindow(window_id=seg_id)
        for seg in seg_list:
            window.channels[seg.channel] = seg.values
            window.labels[seg.channel] = seg.labels
            if window.time is None:
                window.time = seg.timestamps

        windows.append(window)

    return windows


def get_dataset_stats(segments: List[TelemetrySegment]) -> Dict:
    """Compute summary statistics for the loaded dataset."""
    train_segs = [s for s in segments if s.is_train]
    test_segs = [s for s in segments if not s.is_train]

    return {
        "total_segments": len(segments),
        "train_segments": len(train_segs),
        "test_segments": len(test_segs),
        "channels": sorted({s.channel for s in segments}),
        "anomalous_test_segments": sum(1 for s in test_segs if s.has_anomaly),
        "nominal_test_segments": sum(1 for s in test_segs if not s.has_anomaly),
        "total_samples": sum(len(s.values) for s in segments),
        "anomalous_samples": sum(int(s.labels.sum()) for s in segments),
    }
