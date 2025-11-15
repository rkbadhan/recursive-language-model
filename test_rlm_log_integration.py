"""
Integration test showing how RLM uses log analysis tools.

This demonstrates the full RLM flow where the LLM:
1. Receives logs in REPL context
2. Writes Python code to use parse_log(), correlate_logs(), etc.
3. Iteratively explores and analyzes logs
4. Provides final root cause analysis

NOTE: Requires OPENAI_API_KEY environment variable.
"""

import os
import sys

# Sample logs for testing
SAMPLE_JSTACK = '''
"Worker-1" #12 prio=5 os_prio=0 tid=0x00007f8a4c000800 nid=0x1a2b waiting on condition
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.Service.process(Service.java:45)
    - waiting to lock <0x00000000e1234560> (a java.lang.Object)

Found one Java-level deadlock:
"Worker-1": waiting to lock object 0x00000000e1234560
'''

SAMPLE_GC = '''
[2024-11-15T14:30:15.456+0000][gc] GC(101) Pause Young (Normal) 55M->15M(100M) 5234.567ms
[2024-11-15T14:31:00.000+0000][gc] GC(103) Pause Full (Ergonomics) 90M->30M(100M) 8000.123ms
'''

SAMPLE_SYSLOG = '''
Nov 15 14:30:15 hostname application[12346]: ERROR: Database connection failed
Nov 15 14:30:16 hostname application[12346]: WARNING: Retrying connection
'''


def test_rlm_log_analyzer_integration():
    """
    Test that demonstrates how RLM uses log analysis tools.

    This shows the full flow:
    1. User provides logs to RLMLogAnalyzer
    2. LLM explores logs using REPL with injected functions
    3. LLM uses parse_log(), correlate_logs(), detect_all_patterns()
    4. LLM provides comprehensive analysis
    """
    print("\n" + "="*80)
    print("RLM LOG ANALYSIS INTEGRATION TEST")
    print("="*80)
    print()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  SKIPPED: OPENAI_API_KEY not set")
        print()
        print("This test requires OpenAI API to demonstrate RLM integration.")
        print("Set OPENAI_API_KEY to run this test.")
        print()
        print("What this test would show:")
        print("  1. RLMLogAnalyzer receives multiple logs as context")
        print("  2. LLM writes Python code in REPL to:")
        print("     - Use parse_log() to parse each log")
        print("     - Use correlate_logs() to build timeline")
        print("     - Use detect_all_patterns() to find issues")
        print("  3. LLM iteratively explores the data")
        print("  4. LLM provides final root cause analysis")
        return False

    try:
        from rlm import RLMLogAnalyzer

        print("Setting up RLMLogAnalyzer...")
        analyzer = RLMLogAnalyzer(
            model="gpt-4o-mini",
            enable_logging=True,  # Show what the LLM is doing
            max_iterations=10
        )

        # Provide multiple logs as context
        logs = {
            'jstack': SAMPLE_JSTACK,
            'gc': SAMPLE_GC,
            'syslog': SAMPLE_SYSLOG
        }

        print()
        print("Query: Analyze these logs and identify any issues.")
        print()
        print("="*80)
        print("RLM EXECUTION (Watch the LLM use the tools)")
        print("="*80)
        print()

        # The LLM will:
        # 1. Peek at context to see it's a dict of logs
        # 2. Use parse_log() on each log
        # 3. Use correlate_logs() to build timeline
        # 4. Use detect_all_patterns() to find issues
        # 5. Provide comprehensive analysis

        result = analyzer.completion(
            context=logs,
            query="Analyze these logs and identify any issues. What patterns do you detect?"
        )

        print()
        print("="*80)
        print("FINAL ANALYSIS FROM RLM")
        print("="*80)
        print(result)
        print()

        # Verify the result contains expected content
        result_lower = result.lower()

        checks = {
            'deadlock': 'deadlock' in result_lower,
            'gc_pause': 'gc' in result_lower or 'pause' in result_lower,
            'error': 'error' in result_lower or 'database' in result_lower,
        }

        print("="*80)
        print("VERIFICATION")
        print("="*80)
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} Analysis mentions {check}: {passed}")

        all_passed = all(checks.values())
        print()
        if all_passed:
            print("✓ RLM successfully used log analysis tools!")
        else:
            print("⚠️  RLM completed but may have missed some issues")

        return all_passed

    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure 'openai' package is installed: pip install openai")
        return False
    except Exception as e:
        print(f"Error during RLM execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_repl_tools():
    """
    Show what tools are available in the RLM REPL environment.
    """
    print("\n" + "="*80)
    print("TOOLS AVAILABLE IN RLM REPL ENVIRONMENT")
    print("="*80)
    print()

    from rlm.repl_log import LogAnalysisREPLEnv

    # Create a REPL environment to show what's injected
    env = LogAnalysisREPLEnv(
        context_json={'test': 'data'},
        context_str="test",
        recursive_model="gpt-4o-mini",
        depth=0,
        max_depth=1,
        enable_logging=False
    )

    # Show injected functions
    log_functions = [
        'parse_log', 'parse_jstack', 'parse_strace', 'parse_gc_log',
        'parse_pstack', 'parse_syslog', 'parse_json_logs',
        'detect_log_format', 'correlate_logs', 'detect_all_patterns',
        'find_correlated_events', 'generate_correlation_summary'
    ]

    print("Log Analysis Functions:")
    for func_name in log_functions:
        if func_name in env.globals:
            print(f"  ✓ {func_name}()")
        else:
            print(f"  ✗ {func_name}() - NOT FOUND")

    print()
    print("RLM Functions:")
    rlm_functions = ['llm_query', 'llm_query_batch', 'FINAL', 'FINAL_VAR']
    for func_name in rlm_functions:
        if func_name in env.globals:
            print(f"  ✓ {func_name}()")
        else:
            print(f"  ✗ {func_name}() - NOT FOUND")

    print()
    print("The LLM can write code like:")
    print()
    print("```python")
    print("# 1. Parse each log")
    print("jstack_parsed = parse_log(context['jstack'])")
    print("gc_parsed = parse_log(context['gc'])")
    print()
    print("# 2. Correlate them")
    print("timeline = correlate_logs({")
    print("    'jstack': jstack_parsed,")
    print("    'gc': gc_parsed")
    print("})")
    print()
    print("# 3. Detect patterns")
    print("patterns = detect_all_patterns(timeline)")
    print()
    print("# 4. Return analysis")
    print("FINAL(f'Found {len(patterns)} issues: {patterns}')")
    print("```")
    print()


if __name__ == "__main__":
    # First show what tools are available
    demonstrate_repl_tools()

    # Then run integration test
    success = test_rlm_log_analyzer_integration()

    sys.exit(0 if success else 1)
