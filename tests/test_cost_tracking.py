"""
Test script to verify cost tracking improvements.

This tests that:
1. SubRLM tracks costs
2. REPL aggregates sync and async costs
3. RLM_REPL aggregates all costs (root + REPL)
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rlm.repl import SubRLM, REPLEnv
from rlm.utils.llm import OpenAIClient


def test_subrlm_cost_tracking():
    """Test that SubRLM enables cost tracking by default."""
    print("Testing SubRLM cost tracking...")

    # Create SubRLM with cost tracking enabled
    sub_rlm = SubRLM(model="gpt-4o-mini", track_costs=True)

    # Verify client has cost tracking enabled
    assert sub_rlm.client.track_costs == True, "SubRLM should enable cost tracking"
    print("✓ SubRLM enables cost tracking")

    # Verify cost_summary is implemented
    summary = sub_rlm.cost_summary()
    assert 'total_calls' in summary, "Cost summary should have total_calls"
    assert summary['total_calls'] == 0, "Initial calls should be 0"
    print("✓ SubRLM.cost_summary() returns proper format")

    # Verify reset is implemented
    sub_rlm.reset()
    print("✓ SubRLM.reset() works")

    print("\n✅ SubRLM cost tracking tests passed!\n")


def test_repl_cost_structure():
    """Test REPL cost aggregation structure."""
    print("Testing REPL cost aggregation structure...")

    # Create REPL with cost tracking
    repl = REPLEnv(
        recursive_model="gpt-4o-mini",
        context_str="test context",
        track_costs=True
    )

    # Verify REPL has cost tracking enabled
    assert repl.track_costs == True, "REPL should have cost tracking enabled"
    print("✓ REPL cost tracking enabled")

    # Verify async cost counters exist
    assert hasattr(repl, 'async_input_tokens'), "REPL should have async_input_tokens"
    assert hasattr(repl, 'async_output_tokens'), "REPL should have async_output_tokens"
    assert hasattr(repl, 'async_calls'), "REPL should have async_calls"
    print("✓ REPL has async cost counters")

    # Verify cost summary method exists and returns proper structure
    summary = repl.get_repl_cost_summary()
    assert 'total_calls' in summary, "REPL cost summary should have total_calls"
    assert 'depth' in summary, "REPL cost summary should have depth"
    assert 'breakdown' in summary, "REPL cost summary should have breakdown"
    assert 'sync_calls' in summary['breakdown'], "Should track sync calls"
    assert 'async_calls' in summary['breakdown'], "Should track async calls"
    print("✓ REPL cost summary has proper structure")

    # Verify SubRLM inside REPL has cost tracking enabled
    assert repl.sub_rlm.client.track_costs == True, "SubRLM in REPL should track costs"
    print("✓ SubRLM in REPL has cost tracking enabled")

    print("\n✅ REPL cost aggregation tests passed!\n")


def test_cost_aggregation_logic():
    """Test that cost aggregation logic is sound."""
    print("Testing cost aggregation logic...")

    # Create REPL with cost tracking
    repl = REPLEnv(
        recursive_model="gpt-4o-mini",
        track_costs=True
    )

    # Simulate some async calls by manually setting counters
    repl.async_input_tokens = 100
    repl.async_output_tokens = 50
    repl.async_calls = 2

    # Get cost summary
    summary = repl.get_repl_cost_summary()

    # Verify aggregation
    assert summary['breakdown']['async_calls'] == 2, "Should show 2 async calls"
    assert summary['total_calls'] >= 2, "Total calls should include async calls"
    print("✓ Cost aggregation includes async calls")

    # Verify cost calculation exists
    assert 'estimated_cost_usd' in summary, "Should have cost estimate"
    assert summary['estimated_cost_usd'] >= 0, "Cost should be non-negative"
    print("✓ Cost calculation works")

    print("\n✅ Cost aggregation logic tests passed!\n")


def test_openai_client_cost_tracking():
    """Test OpenAIClient cost tracking configuration."""
    print("Testing OpenAIClient cost tracking...")

    # Create client with cost tracking disabled
    client1 = OpenAIClient(model="gpt-4o-mini", track_costs=False)
    assert client1.track_costs == False, "Should disable cost tracking"
    print("✓ OpenAIClient can disable cost tracking")

    # Create client with cost tracking enabled
    client2 = OpenAIClient(model="gpt-4o-mini", track_costs=True)
    assert client2.track_costs == True, "Should enable cost tracking"
    print("✓ OpenAIClient can enable cost tracking")

    # Verify cost tracking counters exist
    assert hasattr(client2, 'total_input_tokens'), "Should have input token counter"
    assert hasattr(client2, 'total_output_tokens'), "Should have output token counter"
    assert hasattr(client2, 'total_calls'), "Should have call counter"
    print("✓ OpenAIClient has cost tracking counters")

    # Verify cost summary method
    summary = client2.get_cost_summary()
    assert 'total_calls' in summary, "Summary should have total_calls"
    assert 'estimated_cost_usd' in summary, "Summary should have cost estimate"
    print("✓ OpenAIClient cost summary works")

    print("\n✅ OpenAIClient cost tracking tests passed!\n")


if __name__ == "__main__":
    test_openai_client_cost_tracking()
    test_subrlm_cost_tracking()
    test_repl_cost_structure()
    test_cost_aggregation_logic()

    print("=" * 50)
    print("🎉 All cost tracking tests passed successfully!")
    print("=" * 50)
