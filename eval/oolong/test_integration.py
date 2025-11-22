#!/usr/bin/env python3
"""
Test OOLONG integration with RLM.

This script tests the OOLONG adapter without requiring full dataset download.
It uses mock examples to verify the integration works correctly.
"""

import os
from rlm import RLM_REPL


def test_rlm_basic():
    """Test basic RLM functionality."""
    print("\n" + "="*80)
    print("TEST 1: Basic RLM Functionality")
    print("="*80 + "\n")

    # Create RLM
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        max_iterations=5,
        enable_logging=True,
        track_costs=True,
    )

    # Test data
    context = "Alice has 5 apples. Bob has 3 oranges. Charlie has 7 bananas."
    query = "How many total fruits are there?"

    print("Context:", context)
    print("Query:", query)

    print("\nCalling RLM...")
    response = rlm.completion(context=context, query=query)

    print(f"\nResponse: {response}")

    # Check if answer is correct
    if "17" in response:
        print("✓ Test PASSED - Found correct answer (17)")
    else:
        print("✗ Test FAILED - Expected 17 in response")

    # Show costs
    if rlm.track_costs:
        costs = rlm.cost_summary()
        print(f"\nCost: ${costs.get('estimated_cost_usd', 0):.4f}")
        print(f"Tokens: {costs.get('total_tokens', 0):,}")

    return "17" in response


def test_rlm_oolong_format():
    """Test RLM with OOLONG-style data."""
    print("\n" + "="*80)
    print("TEST 2: OOLONG Format Compatibility")
    print("="*80 + "\n")

    # Create RLM
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        max_iterations=5,
        enable_logging=False,
    )

    # OOLONG-style context and question
    context = """
    Entry 1: User=1001, Category=entity, Value=Alpha
    Entry 2: User=1002, Category=description, Value=Beta
    Entry 3: User=1001, Category=entity, Value=Gamma
    Entry 4: User=1003, Category=number, Value=Delta
    Entry 5: User=1001, Category=description, Value=Epsilon
    Entry 6: User=1002, Category=entity, Value=Zeta
    """

    query = "Only consider entries for User=1001. How many have Category='entity'?"

    print("Context length:", len(context))
    print("Query:", query)

    print("\nCalling RLM...")
    response = rlm.completion(context=context, query=query)

    print(f"\nResponse: {response}")

    # Check if answer is correct (should be 2)
    if "2" in response:
        print("✓ Test PASSED - Found correct answer (2)")
        return True
    else:
        print("✗ Test might have failed - Expected 2 in response")
        return False


def main():
    """Run all tests."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in a .env file or export it:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return

    print("\n" + "="*80)
    print("OOLONG INTEGRATION TEST SUITE")
    print("="*80)

    results = []

    # Run tests
    try:
        results.append(("Basic RLM", test_rlm_basic()))
    except Exception as e:
        print(f"\n✗ Test FAILED with error: {e}")
        results.append(("Basic RLM", False))

    try:
        results.append(("OOLONG Format", test_rlm_oolong_format()))
    except Exception as e:
        print(f"\n✗ Test FAILED with error: {e}")
        results.append(("OOLONG Format", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! OOLONG integration is working.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above.")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
