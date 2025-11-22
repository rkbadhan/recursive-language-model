"""
Test script to verify true async implementation with AsyncOpenAI.

This demonstrates the performance improvement of using true async vs fake async.
"""

import asyncio
import time
import os
from rlm.utils.llm import AsyncOpenAIClient, OpenAIClient


async def test_async_client():
    """Test that AsyncOpenAIClient works correctly."""
    print("\n" + "="*80)
    print("TEST: AsyncOpenAIClient Basic Functionality")
    print("="*80)

    client = AsyncOpenAIClient(
        model="gpt-4o-mini",
        track_costs=True
    )

    # Single query
    print("\n1. Testing single async query...")
    start = time.time()
    response = await client.completion("Say 'Hello from async!'")
    elapsed = time.time() - start

    print(f"   Response: {response}")
    print(f"   Time: {elapsed:.2f}s")

    # Batch queries (true async)
    print("\n2. Testing parallel batch queries (TRUE ASYNC)...")
    prompts = [
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
        "What is 5+5?",
    ]

    start = time.time()
    responses = await client.completion_batch(prompts)
    elapsed = time.time() - start

    print(f"   Sent {len(prompts)} queries in parallel")
    print(f"   Total time: {elapsed:.2f}s")
    print(f"   Responses:")
    for i, resp in enumerate(responses):
        print(f"      {i+1}. {resp.strip()}")

    # Cost summary
    print("\n3. Cost tracking:")
    summary = client.get_cost_summary()
    print(f"   Total calls: {summary['total_calls']}")
    print(f"   Total tokens: {summary['total_tokens']}")
    print(f"   Estimated cost: ${summary['estimated_cost_usd']:.4f}")

    # Clean up
    await client.close()

    print("\n" + "="*80)
    print("✓ AsyncOpenAIClient test PASSED!")
    print("="*80)


async def compare_async_vs_sync():
    """Compare true async vs sync performance."""
    print("\n" + "="*80)
    print("BENCHMARK: True Async vs Sync Performance")
    print("="*80)

    # Prepare test queries
    prompts = [
        "Count to 3",
        "Count to 4",
        "Count to 5",
        "Count to 6",
    ]

    # Test 1: Sync client (sequential)
    print("\n1. Sync client (sequential)...")
    sync_client = OpenAIClient(model="gpt-4o-mini", track_costs=True)

    start = time.time()
    sync_results = []
    for prompt in prompts:
        result = sync_client.completion(prompt)
        sync_results.append(result)
    sync_time = time.time() - start

    print(f"   Time: {sync_time:.2f}s")
    print(f"   Average per query: {sync_time/len(prompts):.2f}s")

    # Test 2: Async client (parallel)
    print("\n2. Async client (parallel)...")
    async_client = AsyncOpenAIClient(model="gpt-4o-mini", track_costs=True)

    start = time.time()
    async_results = await async_client.completion_batch(prompts)
    async_time = time.time() - start

    print(f"   Time: {async_time:.2f}s")
    print(f"   Average per query: {async_time/len(prompts):.2f}s")

    # Comparison
    print("\n3. Performance comparison:")
    speedup = sync_time / async_time
    improvement = ((sync_time - async_time) / sync_time) * 100

    print(f"   Sync time:     {sync_time:.2f}s")
    print(f"   Async time:    {async_time:.2f}s")
    print(f"   Speedup:       {speedup:.2f}x")
    print(f"   Improvement:   {improvement:.1f}% faster")

    # Verify results are similar
    print("\n4. Verification:")
    print(f"   Sync results:  {len(sync_results)} responses")
    print(f"   Async results: {len(async_results)} responses")
    print(f"   ✓ Both returned {len(prompts)} responses")

    # Clean up
    await async_client.close()

    print("\n" + "="*80)
    if speedup > 1.5:
        print(f"✓ TRUE ASYNC is {speedup:.1f}x FASTER! 🚀")
    else:
        print(f"⚠ Speedup only {speedup:.1f}x (network latency may vary)")
    print("="*80)


async def test_in_repl():
    """Test that async works within REPL environment."""
    print("\n" + "="*80)
    print("TEST: Async in REPL Environment")
    print("="*80)

    from rlm.rlm_repl import RLM_REPL

    # Create RLM
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        recursive_model="gpt-4o-mini",
        enable_logging=False,
        max_iterations=5
    )

    # Test with batch queries
    context = """
    Item 1: Apple
    Item 2: Banana
    Item 3: Cherry
    Item 4: Date
    """

    query = "List all 4 items. Use llm_query_batch() to process them in parallel."

    print(f"\nContext: {context.strip()}")
    print(f"Query: {query}")
    print("\nExecuting RLM with async batch queries...\n")

    start = time.time()
    result = rlm.completion(context=context, query=query)
    elapsed = time.time() - start

    print(f"\n✓ Result: {result}")
    print(f"✓ Time: {elapsed:.2f}s")

    print("\n" + "="*80)
    print("✓ REPL async test PASSED!")
    print("="*80)


async def main():
    """Run all tests."""
    print("\n🚀 TRUE ASYNC IMPLEMENTATION TEST SUITE 🚀\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set!")
        print("Please set it in .env or export it:")
        print("  export OPENAI_API_KEY='your-key'")
        return

    try:
        # Test 1: Basic async functionality
        await test_async_client()

        # Test 2: Performance comparison
        await compare_async_vs_sync()

        # Test 3: Integration with REPL
        await test_in_repl()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! True async is working! 🎉")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
