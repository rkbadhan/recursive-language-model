"""
Main benchmark runner script.

Run OOLONG and BrowseComp-Plus benchmarks with various models/agents:
- RLM(GPT-4o)
- RLM(GPT-4o-mini)
- RLM(GPT-4o) without recursion
- Direct GPT-4o
- Direct GPT-4o-mini
- Direct GPT-4o with BM25 pre-retrieval
- ReAct + GPT-4o + BM25
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlm.rlm_repl import RLM_REPL
from benchmarks.oolong import OOLONGBenchmark
from benchmarks.browsecomp_plus import BrowseCompPlusBenchmark
from benchmarks.baselines import (
    DirectGPT,
    DirectGPTMini,
    DirectGPTTruncated
)
from benchmarks.baselines.react_agent import ReActAgent, DirectGPTWithBM25
from benchmarks.utils import save_results


def create_rlm_wrapper(
    model: str = "gpt-4o",
    recursive_model: str = "gpt-4o-mini",
    enable_recursion: bool = True,
    max_iterations: int = 10
):
    """
    Create a wrapper function for RLM that matches the (context, query) -> answer interface.

    Args:
        model: Root LM model
        recursive_model: Sub-LM model for recursion
        enable_recursion: Whether to enable recursive calls
        max_iterations: Max iterations for RLM

    Returns:
        Function that takes (context, query) and returns answer
    """
    rlm = RLM_REPL(
        model=model,
        recursive_model=recursive_model if enable_recursion else None,
        enable_logging=False,  # Disable logging for benchmarks
        max_iterations=max_iterations,
        track_costs=True
    )

    def wrapper(context: str, query: str) -> str:
        # Reset RLM state between queries
        rlm.reset()
        return rlm.completion(context=context, query=query)

    # Attach cost tracking
    wrapper.get_costs = lambda: rlm.cost_summary()

    return wrapper


def run_oolong_benchmark(args):
    """Run OOLONG benchmark with selected models."""
    print("\n" + "="*80)
    print("OOLONG BENCHMARK")
    print("="*80 + "\n")

    # Create benchmark
    benchmark = OOLONGBenchmark(
        num_queries=args.num_queries,
        entries_per_query=args.entries_per_query,
        verbose=True
    )

    # Create models to compare
    models = {}

    if args.run_rlm:
        print("Creating RLM(GPT-4o-mini)...")
        models["RLM(GPT-4o-mini)"] = create_rlm_wrapper(
            model="gpt-4o-mini",
            recursive_model="gpt-4o-mini",
            enable_recursion=True
        )

    if args.run_rlm_gpt4:
        print("Creating RLM(GPT-4o)...")
        models["RLM(GPT-4o)"] = create_rlm_wrapper(
            model="gpt-4o",
            recursive_model="gpt-4o-mini",
            enable_recursion=True
        )

    if args.run_rlm_no_recursion:
        print("Creating RLM(GPT-4o) without recursion...")
        models["RLM(GPT-4o) No-Recursion"] = create_rlm_wrapper(
            model="gpt-4o",
            enable_recursion=False
        )

    if args.run_direct_gpt:
        print("Creating Direct GPT-4o...")
        models["Direct GPT-4o"] = DirectGPT(model="gpt-4o")

    if args.run_direct_gpt_mini:
        print("Creating Direct GPT-4o-mini...")
        models["Direct GPT-4o-mini"] = DirectGPTMini(model="gpt-4o-mini")

    if args.run_react:
        print("Creating ReAct + GPT-4o + BM25...")
        models["ReAct + GPT-4o + BM25"] = ReActAgent(model="gpt-4o")

    if not models:
        print("No models selected! Use --run-all or specific model flags.")
        return

    # Run comparison
    results = benchmark.compare_models(models, num_queries=args.num_queries)

    # Save results
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"benchmarks/results/oolong_{timestamp}.json"
        save_results(results, output_path)

    return results


def run_browsecomp_benchmark(args):
    """Run BrowseComp-Plus benchmark with selected models."""
    print("\n" + "="*80)
    print("BROWSECOMP-PLUS BENCHMARK")
    print("="*80 + "\n")

    # Create benchmark
    benchmark = BrowseCompPlusBenchmark(
        num_queries=args.num_queries,
        num_documents=args.num_documents,
        num_evidence_docs=3,
        verbose=True
    )

    # Create models to compare
    models = {}

    if args.run_rlm_gpt4:
        print("Creating RLM(GPT-4o)...")
        models["RLM(GPT-4o)"] = create_rlm_wrapper(
            model="gpt-4o",
            recursive_model="gpt-4o-mini",
            enable_recursion=True
        )

    if args.run_rlm_no_recursion:
        print("Creating RLM(GPT-4o) without recursion...")
        models["RLM(GPT-4o) No-Recursion"] = create_rlm_wrapper(
            model="gpt-4o",
            enable_recursion=False
        )

    if args.run_direct_gpt:
        print("Creating Direct GPT-4o...")
        models["Direct GPT-4o"] = DirectGPT(model="gpt-4o")

    if args.run_direct_gpt_truncated:
        print("Creating Direct GPT-4o (Truncated)...")
        models["Direct GPT-4o (Truncated)"] = DirectGPTTruncated(model="gpt-4o")

    if args.run_direct_gpt_bm25:
        print("Creating Direct GPT-4o + BM25...")
        models["Direct GPT-4o + BM25"] = DirectGPTWithBM25(
            model="gpt-4o",
            top_k_docs=40
        )

    if args.run_react:
        print("Creating ReAct + GPT-4o + BM25...")
        models["ReAct + GPT-4o + BM25"] = ReActAgent(model="gpt-4o")

    if not models:
        print("No models selected! Use --run-all or specific model flags.")
        return

    # Run comparison
    results = benchmark.compare_models(models, num_queries=args.num_queries)

    # Save results
    if args.save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"benchmarks/results/browsecomp_{timestamp}.json"
        save_results(results, output_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run long-context benchmarks for RLM evaluation"
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        choices=["oolong", "browsecomp", "both"],
        default="both",
        help="Which benchmark to run (default: both)"
    )

    # Model selection
    parser.add_argument("--run-all", action="store_true", help="Run all models")
    parser.add_argument("--run-rlm", action="store_true", help="Run RLM(GPT-4o-mini)")
    parser.add_argument("--run-rlm-gpt4", action="store_true", help="Run RLM(GPT-4o)")
    parser.add_argument("--run-rlm-no-recursion", action="store_true", help="Run RLM without recursion")
    parser.add_argument("--run-direct-gpt", action="store_true", help="Run Direct GPT-4o")
    parser.add_argument("--run-direct-gpt-mini", action="store_true", help="Run Direct GPT-4o-mini")
    parser.add_argument("--run-direct-gpt-truncated", action="store_true", help="Run Direct GPT-4o with truncation")
    parser.add_argument("--run-direct-gpt-bm25", action="store_true", help="Run Direct GPT-4o + BM25")
    parser.add_argument("--run-react", action="store_true", help="Run ReAct agent")

    # Benchmark parameters
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5,
        help="Number of queries to run (default: 5)"
    )
    parser.add_argument(
        "--entries-per-query",
        type=int,
        default=5000,
        help="Number of entries per OOLONG query (default: 5000)"
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=100,
        help="Number of documents per BrowseComp query (default: 100)"
    )

    # Output options
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # If --run-all is specified, enable all models
    if args.run_all:
        args.run_rlm = True
        args.run_rlm_gpt4 = True
        args.run_rlm_no_recursion = True
        args.run_direct_gpt = True
        args.run_direct_gpt_mini = True
        args.run_direct_gpt_truncated = True
        args.run_direct_gpt_bm25 = True
        args.run_react = True

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in a .env file or export it:")
        print("  export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # Run benchmarks
    results = {}

    if args.benchmark in ["oolong", "both"]:
        results['oolong'] = run_oolong_benchmark(args)

    if args.benchmark in ["browsecomp", "both"]:
        results['browsecomp'] = run_browsecomp_benchmark(args)

    # Print final summary
    print("\n" + "="*80)
    print("BENCHMARK RUN COMPLETE")
    print("="*80)

    if 'oolong' in results:
        print("\nOOLONG Results:")
        for model_name, model_results in results['oolong']['results'].items():
            print(f"  {model_name}: Avg Score = {model_results['avg_score']:.3f}")

    if 'browsecomp' in results:
        print("\nBrowseComp-Plus Results:")
        for model_name, model_results in results['browsecomp']['results'].items():
            print(f"  {model_name}: Avg Score = {model_results['avg_score']:.3f}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
