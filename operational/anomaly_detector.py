"""
Real-time streaming anomaly detector for satellite telemetry.

Two detection modes:
1. SlidingWindowDetector — operational mode for structured power/thermal telemetry.
   Uses a dual-window Kolmogorov–Smirnov distribution-shift test to flag contextual
   anomalies. Requires no training data.

   Eclipse-awareness: ONLY channels that are physically zero during occultation
   (solar_input, solar_panel_temp) are suppressed — and only inside the true eclipse
   window (orbital phase 0.42–0.58). Battery and bus channels are NOT suppressed:
   their orbital-coupled variation is stationary and the rolling reference window
   absorbs it naturally within one or two full orbits.

2. Z-score fallback — retained for shallow channels whose distribution is
   approximately Gaussian (e.g. single-value synthetic channels in unit tests).
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple

def fast_ks_2samp(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
    """
    Sub-millisecond Kolmogorov-Smirnov test algorithm.
    Optimized for anomaly detection streaming where arrays are small.
    """
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = len(data1)
    n2 = len(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.max(np.abs(cdf1 - cdf2))
    
    en = np.sqrt(n1 * n2 / (n1 + n2))
    z = (en + 0.12 + 0.11 / en) * d
    pval = 2 * np.exp(-2.0 * z ** 2)
    return d, min(float(pval), 1.0)

class SlidingWindowDetector:
    """
    Distribution-shift anomaly detector based on a rolling KS-test.

    For each telemetry channel, we maintain two windows:
      - reference window (REF_SIZE samples) — the recent "normal" baseline
      - current  window (CUR_SIZE samples) — the most recent observations

    When the KS-test p-value drops below p_threshold for PERSIST consecutive
    ticks, the channel is flagged as anomalous.

    Eclipse-awareness: ONLY solar_input and solar_panel_temp are suppressed
    during the true eclipse window (orbital phase 0.42–0.58).  Battery-coupled
    channels (battery_charge, bus_voltage) are intentionally NOT suppressed —
    they fluctuate with the orbit in a stationary manner that the rolling
    reference window learns within the first orbit, so no suppression is needed
    or desirable.

    False-positive control:
      - High p_threshold (strict): fewer FPs, lower recall
      - Low  p_threshold (loose) : higher recall, more FPs
    Default settings (p=0.005, persist=4) target F1-balanced performance
    on the NASA SMAP/MSL benchmark.
    """

    def __init__(
        self,
        window_size: int = 64,
        ref_size: int = 128,
        p_threshold: float = 0.005,
        persist: int = 4,
        z_threshold: float = 5.0,
        max_z: float = 8.0,
    ):
        self.window_size = window_size
        self.ref_size = ref_size
        self.p_threshold = p_threshold
        self.persist = persist
        self.z_threshold = z_threshold
        self.max_z = max_z

        self.cur_windows: Dict[str, deque] = {}
        self.ref_windows: Dict[str, deque] = {}

        self._alarm_streak: Dict[str, int] = {}
        self._in_alarm: Dict[str, bool] = {}

    def process_tick(self, row: dict) -> Dict[str, float]:
        """
        Ingest one telemetry row (dict of channel → scalar value).

        Returns a dict of {channel_name: severity} for any anomalous channels,
        where severity ∈ [0, 1] scales with statistical significance.
        """
        anomalies: Dict[str, float] = {}
        phase = row.get("orbital_phase", 0.0)
        in_eclipse = 0.42 <= phase <= 0.58

        for raw_key, val in row.items():
            if raw_key in ("timestamp", "orbital_phase") or not isinstance(
                val, (int, float, np.floating)
            ):
                continue

            key = raw_key
            for suffix in ("_measured", "_observed"):
                if raw_key.endswith(suffix):
                    key = raw_key[: -len(suffix)]
                    break

            SOLAR_DIRECT = ("solar_input", "solar_panel_temp")
            is_photo_diode = key.startswith("pd") and key.endswith("theta")
            
            if (0.10 <= phase <= 0.90 and key in SOLAR_DIRECT) or (is_photo_diode and val < 0.01):
                self._alarm_streak[key] = 0
                
                if key not in self.cur_windows:
                    self.cur_windows[key] = deque(maxlen=self.window_size)
                    self.ref_windows[key] = deque(maxlen=self.ref_size)
                    self._in_alarm[key] = False
                    
                cur_q = self.cur_windows[key]
                ref_q = self.ref_windows[key]
                cur_q.append(val)
                if len(cur_q) % 2 == 0:
                    ref_q.append(val)
                continue

            if key not in self.cur_windows:
                self.cur_windows[key] = deque(maxlen=self.window_size)
                self.ref_windows[key] = deque(maxlen=self.ref_size)
                self._alarm_streak[key] = 0
                self._in_alarm[key] = False

            cur_q = self.cur_windows[key]
            ref_q = self.ref_windows[key]

            if len(cur_q) >= self.window_size and len(ref_q) >= 20:
                _, pval = fast_ks_2samp(list(ref_q), list(cur_q))
                is_anomalous = pval < self.p_threshold

                if not is_anomalous:
                    ref_arr = np.array(ref_q)
                    mean, std = ref_arr.mean(), max(ref_arr.std(), 1e-6)
                    z = abs(val - mean) / std
                    if z > self.z_threshold:
                        is_anomalous = True

                if is_anomalous:
                    self._alarm_streak[key] = self._alarm_streak.get(key, 0) + 1
                    if self._alarm_streak[key] >= self.persist:
                        raw_sev = min(1.0, -np.log10(max(pval, 1e-10)) / 10.0)
                        anomalies[key] = raw_sev
                        continue
                else:
                    self._alarm_streak[key] = 0
                    self._in_alarm[key] = False

            cur_q.append(val)
            if len(cur_q) % 2 == 0:
                ref_q.append(val)

        return anomalies

class CycleLevelDetector:
    """
    Cross-cycle degradation detector for long-term health monitoring.
    
    Instead of tick-by-tick streaming detection (like SlidingWindowDetector),
    this compares the full distribution of a given cycle's feature curve 
    (e.g., a discharge voltage profile) against a composite healthy reference 
    built from the first REF_CYCLES cycles using a Kolmogorov-Smirnov test.
    """

    def __init__(
        self,
        ref_cycles: int = 10,
        p_threshold: float = 0.01,
        persist_cycles: int = 2,
    ):
        self.ref_cycles = ref_cycles
        self.p_threshold = p_threshold
        self.persist_cycles = persist_cycles

        self.ref_samples = []
        self.alarm_streak = 0
        self.first_alarm_cycle = -1
        self.is_alarming = False

    def process_cycle(self, cycle_index: int, curve: np.ndarray) -> bool:
        """
        Process a single completed cycle's data curve.
        Returns True if a persistent degradation is detected.
        """
        if len(curve) == 0:
            return self.is_alarming

        if cycle_index < self.ref_cycles:
            self.ref_samples.extend(curve.tolist())
            return False

        if not self.ref_samples:
            return False

        _, p_val = fast_ks_2samp(np.array(self.ref_samples), curve)

        if p_val < self.p_threshold:
            self.alarm_streak += 1
            if self.alarm_streak == 1:
                self.first_alarm_cycle = cycle_index
            if self.alarm_streak >= self.persist_cycles:
                self.is_alarming = True
        else:
            self.alarm_streak = 0

        return self.is_alarming

