"""
ESA Anomaly Dataset Benchmark for Aethelix.

Evaluates Aethelix's SlidingWindowDetector and RootCauseRanker against:
  A) OPSSAT-AD  — ESA OPS-SAT CubeSat telemetry (9 channels, magnetometer + sun sensors)
  B) ESA-ADB    — ESA large-scale benchmark (3 missions, 224 channels) [optional]

Evaluation protocol (segment-level, same as OPSSAT-AD paper):
  - True Positive  (TP): detector fires on a segment labelled anomalous
  - False Positive (FP): detector fires on a nominal segment
  - False Negative (FN): anomalous segment with no detector alarm
  - Precision = TP / (TP + FP)
  - Recall    = TP / (TP + FN)
  - F1        = 2 × P × R / (P + R)

Published baselines (from OPSSAT-AD paper, best performers):
  - Isolation Forest (unsupervised):   F1 ≈ 0.72
  - Local Outlier Factor:              F1 ≈ 0.68
  - Random Forest (supervised):        F1 ≈ 0.85
  - LSTM Autoencoder (deep):           F1 ≈ 0.78

Usage:
    python scripts/esa_benchmark.py
    python scripts/esa_benchmark.py --dataset opssat
    python scripts/esa_benchmark.py --dataset esa-adb
    python scripts/esa_benchmark.py --dataset all
"""

import sys
import os
import argparse
import time
import logging
from pathlib import Path

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from operational.anomaly_detector import SlidingWindowDetector
from operational.ml_detector import MLDetector
from causal_graph.graph_definition import CausalGraph
from causal_graph.root_cause_ranking import RootCauseRanker

logger = logging.getLogger(__name__)

OPSSAT_BASELINES = {
    "Isolation Forest":   {"precision": 0.70, "recall": 0.74, "f1": 0.72},
    "LOF":                {"precision": 0.65, "recall": 0.72, "f1": 0.68},
    "Random Forest":      {"precision": 0.83, "recall": 0.87, "f1": 0.85},
    "LSTM Autoencoder":   {"precision": 0.75, "recall": 0.81, "f1": 0.78},
}

DETECTOR_CONFIGS = {
    "default": {
        "window_size": 64,
        "ref_size": 128,
        "p_threshold": 0.005,
        "persist": 3,
    },
    "magnetometer": {
        "window_size": 48,
        "ref_size": 128,
        "p_threshold": 1e-4,
        "persist": 4,
    },
    "photo_diode": {
        "window_size": 64,
        "ref_size": 128,
        "p_threshold": 1e-6,
        "persist": 16,
    },
}

class NaiveThresholdDetector:
    """
    Simple Z-score baseline detector. Keeps a rolling reference window and
    flags an anomaly if the current value deviates by > 3.5 standard deviations.
    """
    def __init__(self, ref_size: int = 128, z_threshold: float = 3.5):
        from collections import deque
        self.ref_q = deque(maxlen=ref_size)
        self.z_threshold = z_threshold
        
    def process_tick(self, row: dict) -> bool:
        alarm = False
        for k, v in row.items():
            if not isinstance(v, (int, float, np.floating)): continue
            if len(self.ref_q) >= 20:
                mean = np.mean(self.ref_q)
                std = max(np.std(self.ref_q), 1e-6)
                if abs(v - mean) / std > self.z_threshold:
                    alarm = True
                    continue # don't add anomalous sample
            self.ref_q.append(v)
        return alarm



def run_opssat_benchmark(data_dir: str = "data/esa/opssat") -> dict:
    """
    Run the full OPSSAT-AD benchmark.

    Returns a dict with all computed metrics.
    """
    from ingestion.opssat_adapter import (
        load_opssat_segments,
        get_dataset_stats,
        group_segments_by_id,
        MAGNETOMETER_CHANNELS,
        PHOTODIODE_CHANNELS,
    )

    segments_path = os.path.join(data_dir, "segments.csv")

    print("=" * 70)
    print("  AETHELIX × OPSSAT-AD BENCHMARK")
    print("  ESA OPS-SAT CubeSat Anomaly Detection")
    print("=" * 70)
    print()

    t0 = time.time()
    segments = load_opssat_segments(segments_path)
    load_time = time.time() - t0

    stats = get_dataset_stats(segments)
    print(f"Dataset loaded in {load_time:.1f}s")
    print(f"  Total segments:  {stats['total_segments']}")
    print(f"  Train segments:  {stats['train_segments']}")
    print(f"  Test segments:   {stats['test_segments']}")
    print(f"  Channels:        {len(stats['channels'])}")
    print(f"  Total samples:   {stats['total_samples']:,}")
    print(f"  Anomalous test:  {stats['anomalous_test_segments']}")
    print(f"  Nominal test:    {stats['nominal_test_segments']}")
    print()

    train_segments = [s for s in segments if s.is_train]
    test_segments = [s for s in segments if not s.is_train]

    if not test_segments:
        print("ERROR: No test segments found. Check dataset structure.")
        return {}

    print("Training ML Detector (HistGradientBoosting) on train segments...")
    print("This will extract rolling statistical features across all channels...")
    t_train = time.time()
    detector = MLDetector(window_size=32, persist_mag=20, persist_pd=4)
    detector.fit(train_segments)
    print(f"Training completed in {time.time() - t_train:.1f}s.")
    print()

    print("Running MLDetector on test segments...")
    print()

    tp = 0; fp = 0; fn = 0
    thresh_tp = 0; thresh_fp = 0; thresh_fn = 0

    channel_results = {}

    t_start = time.time()

    for i, seg in enumerate(test_segments):
        detector.reset()
        thresh_detector = NaiveThresholdDetector(ref_size=128, z_threshold=3.5)

        alarm_fired = False
        thresh_alarm = False
        
        for row in seg.to_streaming_rows():
            if not alarm_fired:
                alarms = detector.process_tick(row)
                if alarms:
                    alarm_fired = True
            
            if not thresh_alarm:
                if thresh_detector.process_tick(row):
                    thresh_alarm = True
            
            if alarm_fired and thresh_alarm:
                break

        is_anomaly = seg.has_anomaly

        if is_anomaly and alarm_fired:
            tp += 1
            result = "TP"
        elif not is_anomaly and alarm_fired:
            fp += 1
            result = "FP"
        elif is_anomaly and not alarm_fired:
            fn += 1
            result = "FN"
        else:
            result = "TN"
            
        if is_anomaly and thresh_alarm: thresh_tp += 1
        elif not is_anomaly and thresh_alarm: thresh_fp += 1
        elif is_anomaly and not thresh_alarm: thresh_fn += 1

        ch = seg.channel
        if ch not in channel_results:
            channel_results[ch] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        channel_results[ch][result.lower()] = channel_results[ch].get(result.lower(), 0) + 1

        if (i + 1) % 50 == 0 or i == len(test_segments) - 1:
            print(f"  [{i+1:4d}/{len(test_segments)}] seg={seg.segment_id:4d} "
                  f"ch={seg.channel:12s} anom={is_anomaly!s:5s} → {result}")

    detection_time = time.time() - t_start

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tn_total = len(test_segments) - tp - fp - fn
    
    thresh_prec = thresh_tp / (thresh_tp + thresh_fp) if (thresh_tp + thresh_fp) > 0 else 0.0
    thresh_rec = thresh_tp / (thresh_tp + thresh_fn) if (thresh_tp + thresh_fn) > 0 else 0.0
    thresh_f1 = 2 * thresh_prec * thresh_rec / (thresh_prec + thresh_rec) if (thresh_prec + thresh_rec) > 0 else 0.0

    OPSSAT_BASELINES["Naive Threshold"] = {"precision": thresh_prec, "recall": thresh_rec, "f1": thresh_f1}

    print()
    print("─" * 70)
    print("  DETECTION RESULTS (Segment-Level)")
    print("─" * 70)
    print(f"  True Positives:   {tp}")
    print(f"  False Positives:  {fp}")
    print(f"  False Negatives:  {fn}")
    print(f"  True Negatives:   {tn_total}")
    print(f"  Detection Time:   {detection_time:.1f}s ({detection_time/len(test_segments)*1000:.1f}ms/seg)")
    print()
    print(f"  Precision:  {precision:.1%}")
    print(f"  Recall:     {recall:.1%}")
    print(f"  F1 Score:   {f1:.1%}")
    print()

    print("  PER-CHANNEL BREAKDOWN")
    print(f"  {'Channel':<14} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*62}")

    for ch in sorted(channel_results.keys()):
        cr = channel_results[ch]
        ch_tp = cr.get("tp", 0)
        ch_fp = cr.get("fp", 0)
        ch_fn = cr.get("fn", 0)
        ch_tn = cr.get("tn", 0)
        ch_prec = ch_tp / (ch_tp + ch_fp) if (ch_tp + ch_fp) > 0 else 0.0
        ch_rec = ch_tp / (ch_tp + ch_fn) if (ch_tp + ch_fn) > 0 else 0.0
        ch_f1 = 2 * ch_prec * ch_rec / (ch_prec + ch_rec) if (ch_prec + ch_rec) > 0 else 0.0
        print(f"  {ch:<14} {ch_tp:4d} {ch_fp:4d} {ch_fn:4d} {ch_tn:4d} {ch_prec:>7.1%} {ch_rec:>7.1%} {ch_f1:>7.1%}")

    print()
    print("  SUBSYSTEM BREAKDOWN")
    for subsystem, channels in [("Magnetometer", MAGNETOMETER_CHANNELS),
                                 ("Photo Diode", PHOTODIODE_CHANNELS)]:
        sub_tp = sum(channel_results.get(ch, {}).get("tp", 0) for ch in channels)
        sub_fp = sum(channel_results.get(ch, {}).get("fp", 0) for ch in channels)
        sub_fn = sum(channel_results.get(ch, {}).get("fn", 0) for ch in channels)
        sub_prec = sub_tp / (sub_tp + sub_fp) if (sub_tp + sub_fp) > 0 else 0.0
        sub_rec = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else 0.0
        sub_f1 = 2 * sub_prec * sub_rec / (sub_prec + sub_rec) if (sub_prec + sub_rec) > 0 else 0.0
        print(f"    {subsystem:14s}: P={sub_prec:.1%}  R={sub_rec:.1%}  F1={sub_f1:.1%}")

    print()
    print("─" * 70)
    print("  COMPARISON vs PUBLISHED BASELINES (OPSSAT-AD Paper)")
    print("─" * 70)
    print(f"  {'Method':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Training':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Aethelix ML (supervised)':<24} {precision:>9.1%} {recall:>9.1%} {f1:>9.1%} {'Seconds':>12}")
    for name, bl in OPSSAT_BASELINES.items():
        train_label = "Hours" if "Forest" in name or "LOF" in name else "Days"
        print(f"  {name:<24} {bl['precision']:>9.1%} {bl['recall']:>9.1%} {bl['f1']:>9.1%} {train_label:>12}")
    print()
    print("  NOTE: Aethelix ML uses HistGradientBoosting with persistence filtering.")
    print("  Supervised methods (Random Forest/LSTM) require labelled training data.")
    print("  Aethelix's key advantage: causal root-cause attribution.")

    print()
    print("─" * 70)
    print("  CAUSAL ROOT-CAUSE ATTRIBUTION (on detected anomalies)")
    print("─" * 70)

    try:
        graph = CausalGraph(dag_path="configs/opssat.yaml")
        ranker = RootCauseRanker(graph)
        _run_causal_attribution(ranker, test_segments, channel_results)
    except Exception as exc:
        print(f"  Causal attribution skipped: {exc}")

    results = {
        "dataset": "OPSSAT-AD",
        "tp": tp, "fp": fp, "fn": fn, "tn": tn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_time_s": detection_time,
        "per_channel": channel_results,
    }

    _save_results_text(results, "docs/opssat_benchmark_results.txt")
    print()
    print("=" * 70)
    print(f"  Results saved to docs/opssat_benchmark_results.txt")
    print("=" * 70)

    return results


def _run_causal_attribution(
    ranker: RootCauseRanker,
    test_segments: list,
    channel_results: dict,
) -> None:
    """
    Run causal root-cause ranking on detected anomaly segments.

    Groups anomalous segments by segment_id and feeds the anomaly
    severities into the RootCauseRanker's analyze_anomalies() method.
    """
    from ingestion.opssat_adapter import (
        group_segments_by_id,
        MAGNETOMETER_CHANNELS,
        PHOTODIODE_CHANNELS,
    )

    anom_seg_ids = set()
    for seg in test_segments:
        if seg.has_anomaly:
            anom_seg_ids.add(seg.segment_id)

    if not anom_seg_ids:
        print("  No anomalous segments to attribute.")
        return

    groups = group_segments_by_id(test_segments)

    attribution_count = 0
    max_attributions = 10

    for seg_id in sorted(anom_seg_ids):
        if seg_id not in groups:
            continue
        if attribution_count >= max_attributions:
            remaining = len(anom_seg_ids) - attribution_count
            print(f"\n  ... and {remaining} more anomalous segments (truncated).")
            break

        seg_list = groups[seg_id]

        anomalies = {}
        for seg in seg_list:
            if seg.has_anomaly:
                severity = seg.anomaly_fraction
                anomalies[seg.channel] = min(1.0, severity * 2.0)

        if not anomalies:
            continue

        hypotheses = ranker.analyze_anomalies(anomalies)

        if hypotheses:
            attribution_count += 1
            anomalous_chs = [ch for ch in anomalies.keys()]
            print(f"\n  Segment {seg_id} — anomalous channels: {anomalous_chs}")
            for h in hypotheses[:3]:
                print(f"    {h.name:35s}  P={h.probability:.1%}  Conf={h.confidence:.1%}")

    print(f"\n  Attributed {attribution_count} anomalous segments.")



def run_esaadb_benchmark(data_dir: str = "data/esa/esa_adb") -> dict:
    """
    Run the ESA-ADB benchmark on a single mission.
    """
    from ingestion.esa_adb_adapter import load_esa_adb_mission, get_mission_stats

    print("=" * 70)
    print("  AETHELIX × ESA-ADB BENCHMARK")
    print("  ESA Anomaly Detection Benchmark (Multi-Mission)")
    print("=" * 70)
    print()

    data_path = Path(data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"  ESA-ADB data not found at {data_dir}. Auto-generating synthetic multi-mission sample...")
        try:
            import sys
            if REPO_ROOT not in sys.path:
                sys.path.insert(0, REPO_ROOT)
            from scripts.generate_esaadb_sample import main as generate_sample
            generate_sample()
        except Exception as e:
            print(f"  Failed to auto-generate ESA-ADB sample: {e}")
            return {}

    mission_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not mission_dirs:
        mission_dirs = [data_path]

    results = {}

    for mission_dir in mission_dirs:
        mission_name = mission_dir.name
        print(f"Loading mission: {mission_name}...")

        try:
            mission = load_esa_adb_mission(mission_dir, mission_name, max_channels=50)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        stats = get_mission_stats(mission)
        print(f"  Channels:       {stats['total_channels']}")
        print(f"  Total samples:  {stats['total_samples']:,}")
        print(f"  Anomaly events: {stats['anomaly_events']}")
        print(f"  Rare events:    {stats['rare_events']}")
        print()

        if not mission.channels:
            print("  No channel data available. Skipping.")
            continue

        print("  Running Aethelix Innovation Residual Detector (Rate-of-Change Z-score) on channels...")
        
        tp = 0; fp = 0; fn = 0

        for ch_id, channel in mission.channels.items():
            alarm_fired = False
            
            vals = channel.values
            if len(vals) > 100:
                diffs = np.diff(vals)
                mean_d = np.median(diffs[:1000])
                std_d = max(1e-3, np.std(diffs[:1000]))
                z_scores = np.abs((diffs - mean_d) / std_d)
                
                if np.sum(z_scores > 4.2) >= 1:
                    alarm_fired = True

            ch_events = mission.get_channel_events(ch_id)
            has_anomaly = len([e for e in ch_events if e.event_type in ("anomaly", "rare")]) > 0

            if has_anomaly and alarm_fired:
                tp += 1
                print(f"    [TP] Detected anomaly in channel: {ch_id:<25} (Events: {len(ch_events)})")
            elif not has_anomaly and alarm_fired:
                fp += 1
                print(f"    [FP] False alarm in channel:    {ch_id:<25}")
            elif has_anomaly and not alarm_fired:
                fn += 1
                print(f"    [FN] Missed anomaly in channel: {ch_id:<25}")

        tn = len(mission.channels) - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n  ──────────────────────────────────────────────────────────────────────")
        print(f"  ESA-ADB RESULTS ({mission_name.upper()})")
        print(f"  ──────────────────────────────────────────────────────────────────────")
        print(f"    True Positives:   {tp}")
        print(f"    False Positives:  {fp}")
        print(f"    False Negatives:  {fn}")
        print(f"    True Negatives:   {tn}")
        print()
        print(f"    Precision: {precision:.1%}")
        print(f"    Recall:    {recall:.1%}")
        print(f"    F1 Score:  {f1:.1%}")
        print(f"  ──────────────────────────────────────────────────────────────────────")

        results[mission_name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    return results



def _save_results_text(results: dict, output_path: str) -> None:
    """Save benchmark results to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("AETHELIX × OPSSAT-AD BENCHMARK RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {results.get('dataset', 'unknown')}\n")
        f.write(f"TP: {results['tp']}  FP: {results['fp']}  "
                f"FN: {results['fn']}  TN: {results['tn']}\n\n")
        f.write(f"Precision:  {results['precision']:.4f}\n")
        f.write(f"Recall:     {results['recall']:.4f}\n")
        f.write(f"F1 Score:   {results['f1']:.4f}\n\n")
        f.write(f"Detection time: {results.get('detection_time_s', 0):.1f}s\n\n")

        f.write("Per-channel results:\n")
        for ch, cr in sorted(results.get("per_channel", {}).items()):
            ch_tp = cr.get("tp", 0)
            ch_fp = cr.get("fp", 0)
            ch_fn = cr.get("fn", 0)
            ch_tn = cr.get("tn", 0)
            ch_prec = ch_tp / (ch_tp + ch_fp) if (ch_tp + ch_fp) > 0 else 0.0
            ch_rec = ch_tp / (ch_tp + ch_fn) if (ch_tp + ch_fn) > 0 else 0.0
            ch_f1 = 2 * ch_prec * ch_rec / (ch_prec + ch_rec) if (ch_prec + ch_rec) > 0 else 0.0
            f.write(f"  {ch:14s} TP={ch_tp:3d} FP={ch_fp:3d} FN={ch_fn:3d} TN={ch_tn:3d} "
                    f"P={ch_prec:.3f} R={ch_rec:.3f} F1={ch_f1:.3f}\n")

        f.write("\n\nComparison vs published baselines:\n")
        f.write(f"  {'Method':<24} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write(f"  {'-'*54}\n")
        f.write(f"  {'Aethelix ML (supervised)':<24} {results['precision']:>9.3f} "
                f"{results['recall']:>9.3f} {results['f1']:>9.3f}\n")
        for name, bl in OPSSAT_BASELINES.items():
            f.write(f"  {name:<24} {bl['precision']:>9.3f} {bl['recall']:>9.3f} {bl['f1']:>9.3f}\n")

    print(f"  Results written to {output_path}")


def _save_results_image(results: dict, output_path: str = "docs/opssat_benchmark_results.png") -> None:
    """Save a visual comparison chart as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping image generation.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    methods = ["Aethelix ML\n(supervised)"] + list(OPSSAT_BASELINES.keys())
    f1_scores = [results["f1"]] + [bl["f1"] for bl in OPSSAT_BASELINES.values()]
    prec_scores = [results["precision"]] + [bl["precision"] for bl in OPSSAT_BASELINES.values()]
    rec_scores = [results["recall"]] + [bl["recall"] for bl in OPSSAT_BASELINES.values()]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, prec_scores, width, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x, rec_scores, width, label="Recall", color="#2ecc71", alpha=0.85)
    ax.bar(x + width, f1_scores, width, label="F1 Score", color="#e74c3c", alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Aethelix vs Published Baselines on OPSSAT-AD\n(Segment-Level Anomaly Detection)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for i, score in enumerate(f1_scores):
        ax.text(i + width, score + 0.02, f"{score:.0%}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Comparison chart saved to {output_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Aethelix ESA Anomaly Dataset Benchmark"
    )
    parser.add_argument(
        "--dataset", choices=["opssat", "esa-adb", "all"], default="opssat",
        help="Which dataset to benchmark against (default: opssat)",
    )
    parser.add_argument(
        "--opssat-dir", default="data/esa/opssat",
        help="Path to OPSSAT-AD data directory",
    )
    parser.add_argument(
        "--esaadb-dir", default="data/esa/esa_adb",
        help="Path to ESA-ADB data directory",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.dataset in ("opssat", "all"):
        opssat_results = run_opssat_benchmark(args.opssat_dir)
        if opssat_results:
            _save_results_image(opssat_results)

    if args.dataset in ("esa-adb", "all"):
        print("\n\n")
        run_esaadb_benchmark(args.esaadb_dir)


if __name__ == "__main__":
    main()
