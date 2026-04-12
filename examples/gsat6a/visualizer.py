#!/usr/bin/env python3
"""
Graph visualization framework for GSAT-6A analysis.

Generates graphs from timeline, findings, and telemetry data.
All output is data-driven from the analysis results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Optional
from timeline import Timeline, EventSeverity
from findings import FindingsEngine, TelemetryStats


class AnalysisVisualizer:
    """Generate visualizations from analysis data."""
    
    def __init__(self, timeline: Timeline, findings: FindingsEngine):
        """Initialize visualizer with analysis results."""
        self.timeline = timeline
        self.findings = findings
        self.figsize = (16, 12)
    
    def generate_all_graphs(self, output_dir: str = "."):
        """Generate all analysis graphs."""
        self.plot_timeline(output_dir)
        self.plot_telemetry_deviations(output_dir)
        self.plot_detection_comparison(output_dir)
    
    def plot_timeline(self, output_dir: str = "."):
        """Plot timeline of events."""
        if not self.timeline.events:
            print("No timeline events to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        timeline_sorted = sorted(self.timeline.events, key=lambda e: e.time_seconds)
        
        # Separate by severity for visualization
        critical_events = [e for e in timeline_sorted if e.severity == EventSeverity.CRITICAL]
        warning_events = [e for e in timeline_sorted if e.severity == EventSeverity.WARNING]
        
        # Plot critical events
        if critical_events:
            critical_times = [e.time_seconds for e in critical_events]
            critical_y = [1.5] * len(critical_times)
            ax.scatter(critical_times, critical_y, s=300, c='red', marker='o', 
                      label='Critical Events', zorder=3)
            for event in critical_events:
                ax.annotate(f"T+{event.time_seconds:.0f}s\n{event.message}", 
                           xy=(event.time_seconds, 1.5),
                           xytext=(10, 20), textcoords='offset points',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=8)
        
        # Plot warning events
        if warning_events:
            warning_times = [e.time_seconds for e in warning_events]
            warning_y = [0.5] * len(warning_times)
            ax.scatter(warning_times, warning_y, s=300, c='orange', marker='s',
                      label='Warnings', zorder=3)
            for event in warning_events:
                ax.annotate(f"T+{event.time_seconds:.0f}s\n{event.message}",
                           xy=(event.time_seconds, 0.5),
                           xytext=(10, -30), textcoords='offset points',
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                           fontsize=8)
        
        # Timeline axis
        all_times = [e.time_seconds for e in timeline_sorted]
        if all_times:
            ax.set_xlim(0, max(all_times) * 1.1)
        ax.set_ylim(0, 2)
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Warnings', 'Critical'], fontsize=10)
        ax.set_title('GSAT-6A Detection Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper left', fontsize=11)
        
        plt.tight_layout()
        output_path = f"{output_dir}/gsat6a_timeline.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Timeline graph saved: {output_path}")
        plt.close()
    
    def plot_telemetry_deviations(self, output_dir: str = "."):
        """Plot telemetry deviations between nominal and degraded states."""
        if not self.findings.stats:
            print("No telemetry statistics to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('GSAT-6A Telemetry Deviations (Nominal vs Degraded)', 
                    fontsize=14, fontweight='bold')
        
        # Sort by largest loss
        sorted_stats = sorted(self.findings.stats, 
                            key=lambda s: abs(s.loss_percent), reverse=True)
        
        axes_flat = axes.flatten()
        
        for idx, stat in enumerate(sorted_stats[:6]):
            ax = axes_flat[idx]
            
            # Bar chart comparing nominal vs degraded
            categories = ['Nominal', 'Degraded']
            values = [stat.nominal_mean, stat.degraded_mean]
            colors = ['green', 'red']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Add loss percentage
            loss_text = f"Loss: {stat.loss_percent:.1f}%"
            ax.text(0.5, 0.95, loss_text, transform=ax.transAxes,
                   fontsize=11, fontweight='bold', ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            ax.set_ylabel(f'{stat.name} ({stat.unit})', fontweight='bold')
            ax.set_title(stat.name, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = f"{output_dir}/gsat6a_telemetry_deviations.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Telemetry deviations graph saved: {output_path}")
        plt.close()
    
    def plot_detection_comparison(self, output_dir: str = "."):
        """Plot detection method comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Detection times
        causal_time = self.findings.causal_detection_time
        threshold_time = self.findings.threshold_detection_time
        
        if causal_time is not None or threshold_time is not None:
            methods = []
            times = []
            colors = []
            
            if causal_time is not None:
                methods.append('Causal\nInference')
                times.append(causal_time)
                colors.append('green')
            
            if threshold_time is not None:
                methods.append('Threshold-\nBased')
                times.append(threshold_time)
                colors.append('orange')
            
            bars = ax1.bar(methods, times, color=colors, alpha=0.7, 
                          edgecolor='black', linewidth=2, width=0.5)
            
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'T+{time:.1f}s', ha='center', va='bottom', 
                        fontsize=12, fontweight='bold')
            
            # Add lead time annotation only if both exist
            if causal_time is not None and threshold_time is not None:
                lead_time = threshold_time - causal_time
                ax1.annotate('', xy=(0, causal_time), xytext=(0, threshold_time),
                            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
                ax1.text(0.15, (causal_time + threshold_time)/2, 
                        f'Lead Time:\n{lead_time:.1f}s',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9))
            
            ax1.set_ylabel('Detection Time (seconds)', fontweight='bold', fontsize=12)
            ax1.set_title('Detection Time Comparison', fontweight='bold', fontsize=12)
            ax1.set_ylim(0, max(times) * 1.3)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Right: Detection advantage
        ax2.axis('off')
        
        advantage_text = "ANALYSIS SUMMARY\n\n"
        
        if causal_time is not None:
            advantage_text += f"✓ Causal Inference:\n  T+{causal_time:.1f}s\n"
            advantage_text += "  • Root cause identification\n"
            advantage_text += "  • Early detection\n"
            advantage_text += "  • Actionable insights\n\n"
        else:
            advantage_text += "✗ Causal Inference: Not detected\n\n"
        
        if threshold_time is not None:
            advantage_text += f"✗ Threshold-Based:\n  T+{threshold_time:.1f}s\n"
            advantage_text += "  • Symptom detection\n"
            advantage_text += "  • Late response\n"
            advantage_text += "  • Limited insight\n\n"
        else:
            advantage_text += "✓ Threshold-Based: Not triggered\n\n"
        
        if causal_time is not None and threshold_time is not None:
            lead_time = threshold_time - causal_time
            advantage_text += f"LEAD TIME ADVANTAGE: {lead_time:.1f}s\n"
            advantage_text += f"Can enable preventive action\n"
            advantage_text += f"before cascade failure."
        
        ax2.text(0.1, 0.9, advantage_text, transform=ax2.transAxes,
                fontsize=11, family='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#e5f5ff', alpha=0.9))
        
        plt.tight_layout()
        output_path = f"{output_dir}/gsat6a_detection_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Detection comparison graph saved: {output_path}")
        plt.close()
