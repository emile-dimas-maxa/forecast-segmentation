#!/usr/bin/env python3
"""
Example usage of the backtesting system.

This script demonstrates how to use the backtesting system programmatically
and via command line.
"""

import subprocess
import json
import pandas as pd
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'=' * 60}")
    print(f"🔧 {description}")
    print(f"{'=' * 60}")
    print(f"Command: {cmd}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Run example commands."""
    print("🚀 Radio Forecast Segmentation - Backtesting Examples")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("run_backtest.py").exists():
        print("❌ Error: run_backtest.py not found. Please run from the project root.")
        return

    # Example 1: Quick test
    success1 = run_command(
        "python run_backtest.py --sample-size 3 --min-iterations 1 --test-size 1 --val-size 1",
        "Quick Test (3 combinations, 1 iteration each)",
    )

    # Example 2: Custom fixed segments
    success2 = run_command(
        'python run_backtest.py --sample-size 2 --min-iterations 1 --test-size 1 --val-size 1 --fix-segments \'{"CONTINUOUS_STABLE": {"name": "arima", "order": [1,1,1]}, "RARE_STALE": {"name": "null"}}\'',
        "Custom Fixed Segments (ARIMA for CONTINUOUS_STABLE, null for RARE_STALE)",
    )

    # Example 3: Show help
    success3 = run_command("python run_backtest.py --help", "Command Line Help")

    # Example 4: Check results
    if Path("outputs/backtest_results.json").exists():
        print(f"\n{'=' * 60}")
        print("📊 Results Summary")
        print(f"{'=' * 60}")

        try:
            with open("outputs/backtest_results.json", "r") as f:
                results = json.load(f)

            print(f"Total combinations completed: {len(results)}")

            # Show top 3 results
            if results:
                print("\n🏆 Top 3 Results (by Test MAE):")
                sorted_results = sorted(results.items(), key=lambda x: x[1].get("test_mae", float("inf")))
                for i, (name, data) in enumerate(sorted_results[:3], 1):
                    test_mae = data.get("test_mae", "N/A")
                    val_mae = data.get("val_mae", "N/A")
                    print(f"  {i}. {name}: Test MAE={test_mae}, Val MAE={val_mae}")

        except Exception as e:
            print(f"Error reading results: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("📋 Summary")
    print(f"{'=' * 60}")
    print(f"✅ Quick test: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Custom segments: {'PASSED' if success2 else 'FAILED'}")
    print(f"✅ Help command: {'PASSED' if success3 else 'FAILED'}")

    print("\n🎯 Next Steps:")
    print("1. Run: python run_backtest.py --sample-size 50")
    print("2. Check results in: outputs/backtest_results.json")
    print("3. Try different fixed segments with --fix-segments")
    print("4. Run all combinations with: python run_backtest.py --sample-size 0")


if __name__ == "__main__":
    main()
