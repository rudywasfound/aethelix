import sys
import os
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def fast_ks_2samp(data1: np.ndarray, data2: np.ndarray):
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = len(data1)
    n2 = len(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.max(np.abs(cdf1 - cdf2))
    
    en = np.sqrt(n1 * n2 / (n1 + n2))
    # Asymptotic approximation of the true KS p-value formula
    z = (en + 0.12 + 0.11 / en) * d
    pval = 2 * np.exp(-2.0 * z ** 2)
    return d, min(float(pval), 1.0)

class CycleLevelDetector:
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

def aethelix_detection_cycle(data: dict) -> int:
    detector = CycleLevelDetector(ref_cycles=10, p_threshold=0.01, persist_cycles=2)
    volts = data["volts"]
    
    if len(volts) < 15:
        return len(volts) - 1

    for ci, vp in enumerate(volts):
        if vp is None or len(vp) == 0:
            continue
        
        is_alarming = detector.process_cycle(ci, np.asarray(vp, dtype=float))
        if is_alarming:
            return detector.first_alarm_cycle
            
    return len(volts) - 1

from scripts.pcoe_benchmark import synthetic_battery

data = synthetic_battery("B0005", 42)
cycle = aethelix_detection_cycle(data)
print("Detected at cycle:", cycle)

