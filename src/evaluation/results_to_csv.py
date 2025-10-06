#!/usr/bin/env python3
"""
Convert all evaluation JSON results to a single CSV file.

Usage:
    python src/evaluation/results_to_csv.py --results_dir ./results --output evaluation_results.csv
    python src/evaluation/results_to_csv.py  # Uses defaults
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def collect_json_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Recursively find and load all *_results.json files.

    Args:
        results_dir: Directory to search for JSON files

    Returns:
        List of result dictionaries
    """
    results = []

    # Find all JSON files that end with _results.json
    json_files = list(results_dir.rglob("*_results.json"))

    print(f"Found {len(json_files)} result files in {results_dir}")

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"  ✓ Loaded: {json_file.name}")
        except Exception as e:
            print(f"  ✗ Error loading {json_file.name}: {e}")

    return results


def results_to_csv(results: List[Dict[str, Any]], output_file: Path):
    """
    Convert results to CSV format.

    Args:
        results: List of result dictionaries
        output_file: Output CSV file path
    """
    if not results:
        print("No results to write!")
        return

    # Determine all unique fields across all results
    all_fields = set()
    for result in results:
        all_fields.update(result.keys())

    # Define preferred column order
    preferred_order = [
        "model",
        "benchmark",
        "subtask",
        "timestamp",
        "duration_seconds",
        "success_rate",
        "num_episodes"
    ]

    # Start with preferred columns, then add any extras
    fieldnames = [f for f in preferred_order if f in all_fields]
    extra_fields = sorted(all_fields - set(fieldnames))
    fieldnames.extend(extra_fields)

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ CSV saved to: {output_file}")
    print(f"  Columns: {', '.join(fieldnames)}")
    print(f"  Rows: {len(results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert evaluation JSON results to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default directories
  python src/evaluation/results_to_csv.py

  # Specify custom paths
  python src/evaluation/results_to_csv.py --results_dir ./my_results --output my_eval.csv

  # Search multiple directories
  python src/evaluation/results_to_csv.py --results_dir ./results --results_dir ./experiments/logs
        """
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("./results"),
        help="Directory containing result JSON files (default: ./results)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./evaluation_results.csv"),
        help="Output CSV file path (default: ./evaluation_results.csv)"
    )

    args = parser.parse_args()

    # Collect results
    results = collect_json_results(args.results_dir)

    if results:
        # Convert to CSV
        results_to_csv(results, args.output)
    else:
        print("\n⚠ No results found. Make sure you have run evaluations with the standardized evaluator.")


if __name__ == "__main__":
    main()
