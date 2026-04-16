import sys
import os
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from operational.anomaly_detector import fast_ks_2samp
from scripts.pcoe_benchmark import _ks_2samp, ALL_BATTERIES, benchmark_battery

data1 = np.random.normal(3.6, 0.1, 80)
data2 = np.random.normal(3.5, 0.1, 80)

d1, p1 = fast_ks_2samp(data1, data2)
d2, p2 = _ks_2samp(data1, data2)
print("fast_ks:", d1, p1)
print("_ks_2samp:", d2, p2)
