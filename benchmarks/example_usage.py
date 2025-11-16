"""
Example usage of the benchmark suite.

This script demonstrates how to:
1. Run a single benchmark query
2. Compare multiple models
3. Analyze results

Note: Requires OPENAI_API_KEY environment variable to be set.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_oolong_single_query():
    """Example: Run a single OOLONG query with RLM."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single OOLONG Query with RLM")
    print("="*80 + "\n")

    from benchmarks.oolong import OOLONGBenchmark
    from rlm.rlm_repl import RLM_REPL

    # Create benchmark (small for demo)
    benchmark = OOLONGBenchmark(
        num_queries=1,
        entries_per_query=1000,  # Smaller context for demo
        verbose=True
    )

    # Create RLM
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        enable_logging=False,  # Disable for cleaner output
        max_iterations=5
    )

    # Create wrapper function
    def rlm_fn(context: str, query: str) -> str:
        rlm.reset()  # Reset state between queries
        return rlm.completion(context=context, query=query)

    # Run evaluation
    results = benchmark.evaluate(rlm_fn, model_name="RLM(GPT-4o-mini)", num_queries=1)

    print(f"\n✓ Average Score: {results['avg_score']:.3f}")
    print(f"✓ Success Rate: {results['success_rate']:.1%}")


def example_browsecomp_comparison():
    """Example: Compare RLM vs Direct GPT on BrowseComp-Plus."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare Models on BrowseComp-Plus")
    print("="*80 + "\n")

    from benchmarks.browsecomp_plus import BrowseCompPlusBenchmark
    from benchmarks.baselines import DirectGPT
    from rlm.rlm_repl import RLM_REPL

    # Create benchmark (small corpus for demo)
    benchmark = BrowseCompPlusBenchmark(
        num_queries=3,
        num_documents=20,  # Small corpus for demo
        verbose=True
    )

    # Create RLM
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        enable_logging=False,
        max_iterations=5
    )

    def rlm_fn(context: str, query: str) -> str:
        rlm.reset()
        return rlm.completion(context=context, query=query)

    # Create Direct GPT baseline
    direct_gpt = DirectGPT(model="gpt-4o-mini")

    # Compare models
    models = {
        "RLM(GPT-4o-mini)": rlm_fn,
        "Direct GPT-4o-mini": direct_gpt
    }

    results = benchmark.compare_models(models, num_queries=3)

    print("\n✓ Comparison complete!")
    for model_name, metrics in results['results'].items():
        print(f"  {model_name}: Score = {metrics['avg_score']:.3f}")


def example_custom_model():
    """Example: Test a custom model function."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Model Function")
    print("="*80 + "\n")

    from benchmarks.oolong import OOLONGBenchmark

    # Create benchmark
    benchmark = OOLONGBenchmark(num_queries=1, entries_per_query=500)

    # Define a simple custom model (just for demo - won't work well!)
    def simple_model(context: str, query: str) -> str:
        """A very simple model that just counts words."""
        # Extract target label from query
        import re
        match = re.search(r"label '(\w+)'", query)
        if not match:
            return "Answer: 0"

        label = match.group(1)

        # Count occurrences (very naive!)
        count = context.lower().count(label.lower())

        return f"Answer: {count}"

    # Evaluate
    results = benchmark.evaluate(
        simple_model,
        model_name="Simple Word Counter",
        num_queries=1
    )

    print(f"\n✓ Simple model score: {results['avg_score']:.3f}")
    print("(Note: This won't work well - just a demo!)")


def example_analyze_saved_results():
    """Example: Analyze saved results (if any exist)."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Analyze Saved Results")
    print("="*80 + "\n")

    import glob
    from benchmarks.utils import load_results

    # Find result files
    result_files = glob.glob("benchmarks/results/*.json")

    if not result_files:
        print("No saved results found.")
        print("Run benchmarks with --save-results to create result files.")
        return

    print(f"Found {len(result_files)} result file(s):")
    for path in result_files:
        print(f"  - {path}")

    # Load and display first result
    results = load_results(result_files[0])
    print(f"\nLoaded: {result_files[0]}")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Num queries: {results.get('num_queries', 'N/A')}")


def main():
    """Run examples based on user choice."""
    print("\n" + "="*80)
    print("BENCHMARK USAGE EXAMPLES")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Single OOLONG query with RLM")
    print("  2. Compare models on BrowseComp-Plus")
    print("  3. Custom model function")
    print("  4. Analyze saved results")
    print("  5. Run all examples")

    choice = input("\nEnter your choice (1-5): ").strip()

    # Check for API key
    if choice in ["1", "2", "5"]:
        if not os.getenv("OPENAI_API_KEY"):
            print("\nERROR: OPENAI_API_KEY environment variable not set!")
            print("Please set it in a .env file or export it:")
            print("  export OPENAI_API_KEY='your-api-key'")
            return

    if choice == "1":
        example_oolong_single_query()
    elif choice == "2":
        example_browsecomp_comparison()
    elif choice == "3":
        example_custom_model()
    elif choice == "4":
        example_analyze_saved_results()
    elif choice == "5":
        example_oolong_single_query()
        example_browsecomp_comparison()
        example_custom_model()
        example_analyze_saved_results()
    else:
        print("Invalid choice. Running example 3 (no API key needed)...")
        example_custom_model()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
