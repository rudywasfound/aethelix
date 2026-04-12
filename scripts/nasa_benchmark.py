"""
NASA SMAP / MSL anomaly detection benchmark.

Evaluates Aethelix SlidingWindowDetector against the published NASA SMAP/MSL
dataset (Hundman et al. 2018, KDD).  We report Precision, Recall, and F1 at
the anomaly-sequence level — the same evaluation protocol used in the original
Telemanom LSTM paper so results are directly comparable.

Evaluation Protocol (sequence-level, industry standard)

- True Positive  (TP): at least one alarm fires inside an anomaly window.
- False Positive (FP): an alarm fires with no overlap to any anomaly window.
  Consecutive alarms in the same non-anomaly region count as ONE FP event.
- False Negative (FN): a labelled anomaly window with zero alarm overlap.

Reference baselines (Hundman et al. 2018 / NASA Telemanom):
  - Fixed threshold (OOL rule): ~50–60% Recall, very high FP rate
  - LSTM Telemanom:  Precision≈85%, Recall≈85%, F1≈85% (requires training)
  - Aethelix (zero-shot): see results below
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from operational.anomaly_detector import SlidingWindowDetector

DATA_DIR    = "smap&msl_dataset/data/data/test"
LABELS_PATH = "smap&msl_dataset/labeled_anomalies.csv"

# Baselines from published literature
LSTM_PRECISION = 0.851
LSTM_RECALL    = 0.853
LSTM_F1        = 0.852

THRESHOLD_PRECISION = 0.28   # typical OOL fixed-limit precision on SMAP/MSL
THRESHOLD_RECALL    = 0.53
THRESHOLD_F1        = 0.37


def run_nasa_benchmark():
    if not os.path.exists(LABELS_PATH):
        print(f"ERROR: Dataset not found at {LABELS_PATH}")
        print("Download: https://s3-us-west-2.amazonaws.com/telemanom/data.zip")
        return

    labels_df = pd.read_csv(LABELS_PATH)

    total_seqs     = 0
    tp_seqs        = 0
    fp_events      = 0
    fn_seqs        = 0
    per_chan       = []

    print(f"NASA SMAP/MSL Benchmark  —  {len(labels_df)} channels")
    print(f"Evaluation: sequence-level Precision / Recall / F1\n")

    for idx, row in labels_df.iterrows():
        chan_id   = row["chan_id"]
        test_path = os.path.join(DATA_DIR, f"{chan_id}.npy")

        if not os.path.exists(test_path):
            continue

        test_data       = np.load(test_path)
        anomaly_seqs    = ast.literal_eval(row["anomaly_sequences"])
        total_seqs     += len(anomaly_seqs)

        detector = SlidingWindowDetector(
            window_size=64,
            ref_size=128,
            p_threshold=0.005,
            persist=4,
        )

        detected_seqs   = set()
        in_fp_event     = False
        chan_fp         = 0

        for t in range(len(test_data)):
            val   = float(test_data[t, 0]) if test_data.ndim > 1 else float(test_data[t])
            tick  = {"value": val}
            alarms = detector.process_tick(tick)

            if alarms:
                in_anomaly_window = any(s <= t <= e for s, e in anomaly_seqs)

                if in_anomaly_window:
                    for si, (s, e) in enumerate(anomaly_seqs):
                        if s <= t <= e:
                            detected_seqs.add(si)
                    in_fp_event = False
                else:
                    if not in_fp_event:
                        chan_fp += 1
                        fp_events += 1
                        in_fp_event = True
            else:
                in_fp_event = False

        chan_tp = len(detected_seqs)
        chan_fn = len(anomaly_seqs) - chan_tp
        tp_seqs += chan_tp
        fn_seqs += chan_fn

        per_chan.append({
            "chan": chan_id,
            "seqs": len(anomaly_seqs),
            "tp"  : chan_tp,
            "fp"  : chan_fp,
        })

        if idx % 10 == 0:
            print(f"  [{idx:3d}/{len(labels_df)}] {chan_id} — "
                  f"TP={chan_tp}/{len(anomaly_seqs)}  FP_events={chan_fp}")

    precision = tp_seqs / (tp_seqs + fp_events) if (tp_seqs + fp_events) > 0 else 0.0
    recall    = tp_seqs / total_seqs             if total_seqs > 0             else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fp_per_ch = fp_events / len(labels_df)

    print("\n" + "=" * 60)
    print("  FINAL NASA SMAP/MSL BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Total channels evaluated: {len(labels_df)}")
    print(f"  Total labelled sequences: {total_seqs}")
    print(f"  True  Positives (seqs):   {tp_seqs}")
    print(f"  False Negatives (seqs):   {fn_seqs}")
    print(f"  False Positive events:    {fp_events}  ({fp_per_ch:.1f}/channel)")
    print()
    print(f"  {'Metric':<28} {'Aethelix':>12} {'LSTM (trained)':>16} {'Threshold':>12}")
    print(f"  {'-'*68}")
    print(f"  {'Precision':<28} {precision:>11.1%} {LSTM_PRECISION:>15.1%} {THRESHOLD_PRECISION:>11.1%}")
    print(f"  {'Recall':<28} {recall:>11.1%} {LSTM_RECALL:>15.1%} {THRESHOLD_RECALL:>11.1%}")
    print(f"  {'F1 Score':<28} {f1:>11.1%} {LSTM_F1:>15.1%} {THRESHOLD_F1:>11.1%}")
    print(f"  {'FP events / channel':<28} {fp_per_ch:>11.1f} {'N/A (trained)':>16} {'~High':>12}")
    print(f"  {'Training required':<28} {'None':>12} {'Days–weeks':>16} {'None':>12}")
    print(f"  {'Explainability':<28} {'Causal paths':>12} {'None':>16} {'Alert only':>12}")
    print("=" * 60)
    print()
    print("  NOTE: LSTM baseline (Telemanom) requires days of training data and")
    print("  produces no causal explanation. Aethelix is zero-shot.")
    print("  Aethelix's primary advantage is explainability + zero training,")
    print("  not raw F1 on this benchmark (which is LSTM's home turf).")
    print("=" * 60)


if __name__ == "__main__":
    run_nasa_benchmark()
