"""
Analyze and visualize benchmark results.

Load saved benchmark results and generate analysis reports.
"""

import json
import sys
import os
from typing import Dict, Any, List
import argparse


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_oolong_analysis(results: Dict[str, Any]):
    """Print analysis of OOLONG benchmark results."""
    print("\n" + "="*80)
    print("OOLONG BENCHMARK ANALYSIS")
    print("="*80)

    model_results = results.get('results', {})

    print(f"\nNumber of queries: {results.get('num_queries', 'N/A')}")
    print(f"\nModel Comparison:")
    print(f"{'Model':<35} {'Avg Score':>12} {'Success%':>10} {'Avg Time(s)':>12} {'Total Cost':>12}")
    print("-"*80)

    for model_name, metrics in model_results.items():
        avg_score = metrics.get('avg_score', 0)
        success_rate = metrics.get('success_rate', 0) * 100
        avg_time = metrics.get('avg_time', 0)
        total_cost = metrics.get('total_cost', 0)  # If available

        print(
            f"{model_name:<35} "
            f"{avg_score:>12.3f} "
            f"{success_rate:>9.1f}% "
            f"{avg_time:>12.2f} "
            f"{'N/A':>12}"  # Cost tracking TBD
        )

    print("="*80)


def print_browsecomp_analysis(results: Dict[str, Any]):
    """Print analysis of BrowseComp-Plus benchmark results."""
    print("\n" + "="*80)
    print("BROWSECOMP-PLUS BENCHMARK ANALYSIS")
    print("="*80)

    model_results = results.get('results', {})

    print(f"\nNumber of queries: {results.get('num_queries', 'N/A')}")
    print(f"Documents per query: {results.get('num_documents', 'N/A')}")
    print(f"\nModel Comparison:")
    print(
        f"{'Model':<35} {'Avg Score':>12} {'Exact Match':>12} "
        f"{'Success%':>10} {'Avg Time(s)':>12}"
    )
    print("-"*80)

    for model_name, metrics in model_results.items():
        avg_score = metrics.get('avg_score', 0)
        avg_exact = metrics.get('avg_exact_match', 0)
        success_rate = metrics.get('success_rate', 0) * 100
        avg_time = metrics.get('avg_time', 0)

        print(
            f"{model_name:<35} "
            f"{avg_score:>12.3f} "
            f"{avg_exact:>12.3f} "
            f"{success_rate:>9.1f}% "
            f"{avg_time:>12.2f}"
        )

    print("="*80)


def compare_multiple_runs(file_paths: List[str]):
    """Compare multiple benchmark runs."""
    print("\n" + "="*80)
    print("COMPARING MULTIPLE BENCHMARK RUNS")
    print("="*80)

    all_results = []
    for path in file_paths:
        try:
            results = load_results(path)
            all_results.append((path, results))
            print(f"✓ Loaded: {path}")
        except Exception as e:
            print(f"✗ Failed to load {path}: {e}")

    if not all_results:
        print("No valid results loaded!")
        return

    # Analyze each
    for path, results in all_results:
        print(f"\n{'='*80}")
        print(f"Results from: {path}")
        print(f"{'='*80}")

        if 'oolong' in path or results.get('num_documents') is None:
            print_oolong_analysis(results)
        else:
            print_browsecomp_analysis(results)


def export_csv(results: Dict[str, Any], output_path: str):
    """Export results to CSV format."""
    import csv

    model_results = results.get('results', {})

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'Model',
            'Avg Score',
            'Success Rate',
            'Avg Time',
            'Total Time'
        ])

        # Write data
        for model_name, metrics in model_results.items():
            writer.writerow([
                model_name,
                metrics.get('avg_score', 0),
                metrics.get('success_rate', 0),
                metrics.get('avg_time', 0),
                metrics.get('total_time', 0)
            ])

    print(f"\nExported to CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results"
    )

    parser.add_argument(
        "results_file",
        nargs="+",
        help="Path(s) to results JSON file(s)"
    )

    parser.add_argument(
        "--export-csv",
        help="Export to CSV file"
    )

    args = parser.parse_args()

    # Load and analyze results
    if len(args.results_file) == 1:
        # Single file analysis
        results = load_results(args.results_file[0])

        # Determine benchmark type and print analysis
        if 'num_documents' in results:
            print_browsecomp_analysis(results)
        else:
            print_oolong_analysis(results)

        # Export if requested
        if args.export_csv:
            export_csv(results, args.export_csv)

    else:
        # Multiple files comparison
        compare_multiple_runs(args.results_file)


if __name__ == "__main__":
    main()
