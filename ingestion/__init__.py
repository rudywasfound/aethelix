"""Aethelix ingestion module — data adapters for various telemetry formats."""

from ingestion.opssat_adapter import (
    load_opssat_segments,
    TelemetrySegment,
    MultiChannelWindow,
    CHANNEL_MAP,
)

try:
    from ingestion.esa_adb_adapter import (
        load_esa_adb_mission,
        ESAADBMission,
        ESAADBChannel,
        ESAADBEvent,
    )
except ImportError:
    pass

__all__ = [
    "load_opssat_segments",
    "TelemetrySegment",
    "MultiChannelWindow",
    "CHANNEL_MAP",
    "load_esa_adb_mission",
    "ESAADBMission",
    "ESAADBChannel",
    "ESAADBEvent",
]
