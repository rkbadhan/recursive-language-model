"""
Test suite for async execution and depth > 1 recursion features.

This module demonstrates and tests:
1. Parallel LLM queries using llm_query_batch()
2. Async/await code execution in REPL
3. Depth > 1 recursion (nested RLM calls)
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rlm.rlm_repl import RLM_REPL


def test_batch_execution():
    """
    Test 1: Batch Execution for Parallel LLM Queries

    Demonstrates using llm_query_batch() to process multiple chunks in parallel,
    which should be significantly faster than sequential processing.
    """
    print("\n" + "="*80)
    print("TEST 1: Batch Execution (Parallel LLM Queries)")
    print("="*80)

    # Create context with multiple sections
    sections = [
        "Alice works in Engineering and has 5 years of experience.",
        "Bob works in Sales and has 3 years of experience.",
        "Charlie works in Marketing and has 7 years of experience.",
        "Diana works in Engineering and has 2 years of experience.",
        "Eve works in Sales and has 10 years of experience.",
    ]

    context = "\n---\n".join(sections)

    query = "For each person, extract their name, department, and years of experience."

    print(f"\nContext: {len(sections)} sections")
    print(f"Query: {query}")

    # Test with batch execution
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        enable_logging=True,
        max_iterations=5,
        track_costs=True
    )

    print("\nRunning RLM with batch processing enabled...\n")
    start_time = time.time()

    result = rlm.completion(context=context, query=query)

    elapsed = time.time() - start_time

    print("\n" + "="*80)
    print(f"Result: {result}")
    print(f"Time taken: {elapsed:.2f}s")

    if rlm.track_costs:
        cost_summary = rlm.cost_summary()
        print(f"\nCost Summary:")
        print(f"  Total calls: {cost_summary['total_calls']}")
        print(f"  Total tokens: {cost_summary['total_tokens']:,}")
        print(f"  Estimated cost: ${cost_summary['estimated_cost_usd']:.4f}")

    print("="*80)


def test_async_execution():
    """
    Test 2: Async/Await Code Execution

    Demonstrates using async/await syntax in REPL code for parallel processing.
    """
    print("\n" + "="*80)
    print("TEST 2: Async/Await Execution")
    print("="*80)

    # Create context with data to process
    context = """
    User 1: Completed tasks A, B, C
    User 2: Completed tasks D, E
    User 3: Completed tasks F, G, H, I
    User 4: Completed task J
    User 5: Completed tasks K, L, M, N, O
    """

    query = "Count the total number of tasks completed by all users."

    print(f"\nQuery: {query}")

    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        enable_logging=True,
        max_iterations=5
    )

    print("\nRunning RLM (may use async/await internally)...\n")

    result = rlm.completion(context=context, query=query)

    print("\n" + "="*80)
    print(f"Result: {result}")
    print(f"Expected: 15 tasks")
    print("="*80)


def test_depth_2_recursion():
    """
    Test 3: Depth > 1 Recursion

    Demonstrates nested RLM calls where sub-RLMs can spawn their own sub-RLMs.
    This requires max_depth=2 or higher.
    """
    print("\n" + "="*80)
    print("TEST 3: Depth > 1 Recursion (Nested RLM Calls)")
    print("="*80)

    # Create hierarchical context that benefits from nested processing
    context = """
    Department: Engineering
      Team: Backend
        - Alice: Senior Developer (Python, Go)
        - Bob: Junior Developer (Python)
      Team: Frontend
        - Charlie: Senior Developer (React, TypeScript)
        - Diana: Mid-level Developer (React)

    Department: Sales
      Team: Enterprise
        - Eve: Account Executive
        - Frank: Sales Engineer
      Team: SMB
        - Grace: Account Manager
    """

    query = (
        "Organize all employees by department and team, and count how many "
        "employees are in each team."
    )

    print(f"\nQuery: {query}")
    print(f"Context: Hierarchical structure (departments > teams > employees)")

    # Create RLM with depth=2 to allow nested recursion
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        max_iterations=10,
        depth=0,
        max_depth=2,  # Enable depth > 1!
        enable_logging=True,
        track_costs=True
    )

    print(f"\nRLM Configuration:")
    print(f"  Depth: {rlm.depth}")
    print(f"  Max Depth: {rlm.max_depth}")
    print(f"  -> Sub-RLMs can spawn their own RLMs!\n")

    result = rlm.completion(context=context, query=query)

    print("\n" + "="*80)
    print(f"Result:\n{result}")

    if rlm.track_costs:
        cost_summary = rlm.cost_summary()
        print(f"\nCost Summary:")
        print(f"  Total calls: {cost_summary['total_calls']}")
        print(f"  Total tokens: {cost_summary['total_tokens']:,}")
        print(f"  Estimated cost: ${cost_summary['estimated_cost_usd']:.4f}")

    print("="*80)


def test_depth_3_recursion():
    """
    Test 4: Depth = 3 Recursion

    Pushes the limits with 3 levels of recursion.
    """
    print("\n" + "="*80)
    print("TEST 4: Depth = 3 Recursion (Deep Nesting)")
    print("="*80)

    context = """
    Company: TechCorp
      Region: North America
        Country: USA
          State: California - Revenue: $50M
          State: Texas - Revenue: $30M
        Country: Canada
          Province: Ontario - Revenue: $20M
      Region: Europe
        Country: UK
          Region: London - Revenue: $40M
        Country: Germany
          City: Berlin - Revenue: $25M
    """

    query = "Calculate the total revenue for the entire company."

    print(f"\nQuery: {query}")

    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        max_iterations=15,
        max_depth=3,  # Allow 3 levels of recursion
        enable_logging=True
    )

    print(f"\nRLM with max_depth={rlm.max_depth}\n")

    result = rlm.completion(context=context, query=query)

    print("\n" + "="*80)
    print(f"Result: {result}")
    print(f"Expected: $165M")
    print("="*80)


def main():
    """Run all tests."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in a .env file or export it:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return

    print("\n" + "="*80)
    print("ASYNC EXECUTION & DEPTH > 1 RECURSION TESTS")
    print("="*80)
    print("\nThese tests demonstrate:")
    print("  1. Parallel LLM queries with llm_query_batch()")
    print("  2. Async/await code execution in REPL")
    print("  3. Depth > 1 recursion (nested RLM calls)")
    print()

    tests = [
        ("1", "Batch Execution", test_batch_execution),
        ("2", "Async Execution", test_async_execution),
        ("3", "Depth=2 Recursion", test_depth_2_recursion),
        ("4", "Depth=3 Recursion", test_depth_3_recursion),
        ("5", "Run All Tests", None),
    ]

    print("Available tests:")
    for num, name, _ in tests:
        print(f"  {num}. {name}")

    choice = input("\nEnter your choice (1-5): ").strip()

    if choice == "5":
        confirm = input("Running all tests will be expensive. Continue? (yes/no): ").strip().lower()
        if confirm == "yes":
            test_batch_execution()
            test_async_execution()
            test_depth_2_recursion()
            test_depth_3_recursion()
        else:
            print("Cancelled.")
    elif choice in ["1", "2", "3", "4"]:
        test_func = next((func for num, _, func in tests if num == choice), None)
        if test_func:
            test_func()
    else:
        print("Invalid choice. Running batch execution test...")
        test_batch_execution()

    print("\n" + "="*80)
    print("TESTS COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
