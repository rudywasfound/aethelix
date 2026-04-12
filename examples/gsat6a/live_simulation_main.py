#!/usr/bin/env python3
"""
GSAT-6A Demo Entry Point

Run forensic analysis (lead time measurement - our core selling point):
    python live_simulation_main.py forensics

Run live simulation (real-time failure sequence):
    python live_simulation_main.py simulation

Run full mission analysis (comprehensive visualization):
    python live_simulation_main.py mission
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "forensics"  # Default to forensics (lead time analysis)
    
    if mode == "forensics":
        from forensics import GSAT6AForensics
        forensics = GSAT6AForensics()
        forensics.analyze()
        forensics.print_analysis()
        forensics.generate_graphs(output_dir=os.path.dirname(os.path.abspath(__file__)).rsplit('/', 1)[0])
        print("✓ Forensic analysis complete\n")
    
    elif mode == "simulation":
        from live_simulation import GSAT6ASimulator
        simulator = GSAT6ASimulator()
        simulator.run_simulation()
    
    elif mode == "mission":
        from mission_analysis import GSAT6AMissionAnalysis
        analyzer = GSAT6AMissionAnalysis()
        analyzer.analyze_and_visualize()
        print("\n✓ Mission analysis complete\n")
    
    else:
        print(f"Unknown mode: {mode}")
        print("\nUsage:")
        print("  python live_simulation_main.py forensics   # Lead time analysis (default)")
        print("  python live_simulation_main.py simulation  # Live failure sequence")
        print("  python live_simulation_main.py mission     # Full mission analysis")
        sys.exit(1)

if __name__ == "__main__":
    main()
