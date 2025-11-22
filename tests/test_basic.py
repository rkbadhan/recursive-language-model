"""
Basic validation tests for RLM implementation.

Run this to verify the implementation works correctly.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from rlm import RLM
        from rlm.rlm_repl import RLM_REPL
        from rlm.repl import REPLEnv, SubRLM
        from rlm.utils.llm import OpenAIClient
        from rlm.utils.prompts import build_system_prompt, next_action_prompt
        from rlm.utils import utils
        from rlm.logger.root_logger import ColorfulLogger
        from rlm.logger.repl_logger import REPLEnvLogger
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_repl_basic():
    """Test basic REPL functionality without API calls."""
    print("\nTesting REPL basic execution...")

    try:
        from rlm.repl import REPLEnv

        # Create REPL with simple context
        repl = REPLEnv(
            context_str="Hello, World!",
            recursive_model="gpt-4o-mini"
        )

        # Test simple code execution
        result = repl.code_execution("x = 5 + 3\nprint(x)")

        assert "8" in result.stdout, f"Expected '8' in stdout, got: {result.stdout}"
        assert result.locals.get('x') == 8, f"Expected x=8, got: {result.locals.get('x')}"

        print("✓ REPL basic execution works")
        return True

    except Exception as e:
        print(f"✗ REPL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_loading():
    """Test context loading in REPL."""
    print("\nTesting context loading...")

    try:
        from rlm.repl import REPLEnv

        # Test string context
        repl = REPLEnv(context_str="Test context string")
        result = repl.code_execution("print(len(context))")
        assert "19" in result.stdout  # "Test context string" is 19 chars

        # Test JSON context
        repl2 = REPLEnv(context_json={"key": "value", "num": 42})
        result2 = repl2.code_execution("print(context['num'])")
        assert "42" in result2.stdout

        print("✓ Context loading works")
        return True

    except Exception as e:
        print(f"✗ Context loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")

    try:
        from rlm.utils import utils

        # Test find_code_blocks - Standard case
        text_with_code = """
Some text
```repl
x = 42
print(x)
```
More text
"""
        blocks = utils.find_code_blocks(text_with_code)
        assert blocks is not None
        assert len(blocks) == 1
        assert "x = 42" in blocks[0]

        # Test find_code_blocks - No trailing newline (edge case fix)
        text_no_newline = "```repl\nx = 5```"
        blocks_no_newline = utils.find_code_blocks(text_no_newline)
        assert blocks_no_newline is not None
        assert "x = 5" in blocks_no_newline[0]

        # Test find_final_answer - Simple case
        text_with_final = "The answer is FINAL(42)"
        final = utils.find_final_answer(text_with_final)
        assert final is not None
        assert final[0] == 'FINAL'
        assert '42' in final[1]

        # Test FINAL with nested parentheses (edge case fix)
        text_nested = "FINAL(calculate(5 + 3))"
        final_nested = utils.find_final_answer(text_nested)
        assert final_nested is not None
        assert final_nested[0] == 'FINAL'
        assert final_nested[1] == 'calculate(5 + 3)', f"Expected 'calculate(5 + 3)', got '{final_nested[1]}'"

        # Test FINAL_VAR
        text_with_var = "FINAL_VAR(my_answer)"
        final_var = utils.find_final_answer(text_with_var)
        assert final_var is not None
        assert final_var[0] == 'FINAL_VAR'
        assert 'my_answer' in final_var[1]

        print("✓ Utility functions work (including edge case fixes)")
        return True

    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompts():
    """Test prompt generation."""
    print("\nTesting prompt generation...")

    try:
        from rlm.utils.prompts import build_system_prompt, next_action_prompt

        # Test system prompt
        sys_prompt = build_system_prompt()
        assert isinstance(sys_prompt, list)
        assert len(sys_prompt) > 0
        assert sys_prompt[0]['role'] == 'system'
        assert 'REPL' in sys_prompt[0]['content']

        # Test next action prompt
        action_prompt = next_action_prompt("Test query", iteration=0)
        assert action_prompt['role'] == 'user'
        assert 'Test query' in action_prompt['content']

        print("✓ Prompt generation works")
        return True

    except Exception as e:
        print(f"✗ Prompt test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_client():
    """Test LLM client initialization (without API call)."""
    print("\nTesting LLM client initialization...")

    try:
        from rlm.utils.llm import OpenAIClient

        # This will fail if no API key, which is expected
        try:
            client = OpenAIClient(api_key="test-key", model="gpt-4o-mini")
            print("✓ LLM client initialization works")
            return True
        except Exception as e:
            # Expected to fail without real API key
            if "api" in str(e).lower() or "key" in str(e).lower():
                print("✓ LLM client properly validates API key")
                return True
            raise

    except Exception as e:
        print(f"✗ LLM client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rlm_initialization():
    """Test RLM initialization (without API call)."""
    print("\nTesting RLM initialization...")

    try:
        # Set a dummy API key for testing initialization
        os.environ["OPENAI_API_KEY"] = "sk-test-key-for-initialization-only"

        from rlm.rlm_repl import RLM_REPL

        rlm = RLM_REPL(
            model="gpt-4o-mini",
            recursive_model="gpt-4o-mini",
            enable_logging=False,
            max_iterations=5
        )

        assert rlm.model == "gpt-4o-mini"
        assert rlm.recursive_model == "gpt-4o-mini"
        assert rlm._max_iterations == 5

        print("✓ RLM initialization works")
        return True

    except Exception as e:
        print(f"✗ RLM initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cost_tracking():
    """Test cost tracking features."""
    print("\nTesting cost tracking...")

    try:
        # Set a dummy API key for testing initialization
        os.environ["OPENAI_API_KEY"] = "sk-test-key-for-initialization-only"

        from rlm.repl import SubRLM, REPLEnv
        from rlm.utils.llm import OpenAIClient

        # Test 1: OpenAIClient cost tracking
        client = OpenAIClient(api_key="sk-test", model="gpt-4o-mini", track_costs=True)
        assert client.track_costs == True, "OpenAIClient should enable cost tracking"
        assert hasattr(client, 'total_input_tokens'), "Should have input token counter"
        assert hasattr(client, 'total_output_tokens'), "Should have output token counter"

        # Test 2: SubRLM cost tracking
        sub_rlm = SubRLM(model="gpt-4o-mini", track_costs=True)
        assert sub_rlm.client.track_costs == True, "SubRLM should enable cost tracking"
        summary = sub_rlm.cost_summary()
        assert 'total_calls' in summary, "SubRLM should return cost summary"

        # Test 3: REPL cost tracking initialization
        repl = REPLEnv(
            recursive_model="gpt-4o-mini",
            context_str="test",
            track_costs=True
        )
        assert repl.track_costs == True, "REPL should have cost tracking enabled"
        assert hasattr(repl, 'async_input_tokens'), "REPL should track async input tokens"
        assert hasattr(repl, 'async_output_tokens'), "REPL should track async output tokens"
        assert hasattr(repl, 'async_calls'), "REPL should track async call count"

        # Test 4: REPL cost summary structure
        repl_summary = repl.get_repl_cost_summary()
        assert 'total_calls' in repl_summary, "REPL summary should have total_calls"
        assert 'depth' in repl_summary, "REPL summary should have depth"
        assert 'breakdown' in repl_summary, "REPL summary should have breakdown"
        assert 'sync_calls' in repl_summary['breakdown'], "Should track sync calls separately"
        assert 'async_calls' in repl_summary['breakdown'], "Should track async calls separately"

        print("✓ Cost tracking structure works correctly")
        return True

    except Exception as e:
        print(f"✗ Cost tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("RLM IMPLEMENTATION - BASIC VALIDATION TESTS")
    print("="*80)

    tests = [
        test_imports,
        test_repl_basic,
        test_context_loading,
        test_utils,
        test_prompts,
        test_llm_client,
        test_rlm_initialization,
        test_cost_tracking,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "="*80)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("="*80)

    if all(results):
        print("\n🎉 All tests passed! Implementation is working correctly.")
        print("\nNext steps:")
        print("1. Set your OPENAI_API_KEY in .env file")
        print("2. Run: python main.py")
        print("3. Choose option 4 for a quick test with the API")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
