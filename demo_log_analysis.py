"""
Demo: RLM-Powered System Log Analysis

This demo shows how RLM automatically analyzes logs by:
1. Auto-detecting log formats (jstack, strace, GC, syslog)
2. Parsing logs into structured data
3. Correlating events across multiple logs
4. Detecting issue patterns (deadlocks, memory leaks, etc.)
5. Providing root cause analysis

The key: You just provide logs and ask a question.
RLM handles everything else automatically via LLM-generated code.

Requires: OPENAI_API_KEY environment variable
"""

import sys
import os

# Realistic scenario: Application freeze at 14:30
# ============================================================================

JSTACK_DUMP_1430 = """2024-11-15 14:30:15
Full thread dump OpenJDK 64-Bit Server VM (11.0.16+8 mixed mode):

"ConnectionPool-Worker-1" #45 prio=5 os_prio=0 tid=0x00007f8a4c000800 nid=0x1a2b waiting on condition
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.db.ConnectionPool.acquire(ConnectionPool.java:67)
    - waiting to lock <0x00000000e1234560> (a com.example.db.PoolLock)
    at com.example.service.UserService.getUser(UserService.java:34)

"ConnectionPool-Worker-2" #46 prio=5 os_prio=0 tid=0x00007f8a4c001000 nid=0x1a2c waiting on condition
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.db.ConnectionPool.release(ConnectionPool.java:89)
    - waiting to lock <0x00000000e1234000> (a com.example.db.PoolLock)
    - locked <0x00000000e1234560> (a com.example.db.PoolLock)
    at com.example.service.UserService.closeConnection(UserService.java:56)

"DB-Cleanup-Thread" #47 prio=5 os_prio=0 tid=0x00007f8a4c002000 nid=0x1a2d runnable
   java.lang.Thread.State: RUNNABLE
    at com.example.db.ConnectionPool.cleanup(ConnectionPool.java:120)
    - locked <0x00000000e1234000> (a com.example.db.PoolLock)
    at com.example.db.CleanupTask.run(CleanupTask.java:23)

Found one Java-level deadlock:
=============================
"ConnectionPool-Worker-1":
  waiting to lock monitor 0x00007f8a4c003000 (object 0x00000000e1234560),
  which is held by "ConnectionPool-Worker-2"
"ConnectionPool-Worker-2":
  waiting to lock monitor 0x00007f8a4c003100 (object 0x00000000e1234000),
  which is held by "DB-Cleanup-Thread"
"""

STRACE_LOG_1430 = """14:30:10.123456 accept(3, {sa_family=AF_INET, sin_port=htons(54321)}, [16]) = 5 <0.000123>
14:30:10.234567 read(5, "GET /api/users/123 HTTP/1.1\\r\\n", 8192) = 28 <0.000056>
14:30:10.345678 open("/var/log/app.log", O_WRONLY|O_APPEND) = 6 <0.000034>
14:30:10.456789 write(6, "[INFO] Processing user request 123\\n", 38) = 38 <0.000019>
14:30:12.000000 connect(7, {sa_family=AF_INET, sin_port=htons(5432)}, 16) = 8 <0.234567>
14:30:12.500000 write(8, "SELECT * FROM users WHERE id=123", 33) = 33 <0.000089>
14:30:14.000000 read(8, 0x7fff12345678, 8192) = -1 ETIMEDOUT (Connection timed out) <2.500000>
14:30:14.500100 close(8) = 0 <0.000012>
14:30:15.000000 write(6, "[ERROR] Database timeout\\n", 26) = 26 <0.000015>
14:30:16.000000 futex(0x7f8a4c000000, FUTEX_WAIT, 0, NULL) = 0 <10.123456>
14:30:26.123456 write(2, "Application hang detected\\n", 27) = 27 <0.000021>
"""

GC_LOG_1430 = """[2024-11-15T14:25:00.000+0000][gc] GC(95) Pause Young (Normal) 45M->8M(100M) 156.234ms
[2024-11-15T14:26:00.000+0000][gc] GC(96) Pause Young (Normal) 48M->9M(100M) 189.456ms
[2024-11-15T14:27:00.000+0000][gc] GC(97) Pause Young (Allocation Failure) 52M->12M(100M) 234.567ms
[2024-11-15T14:28:00.000+0000][gc] GC(98) Pause Young (Allocation Failure) 58M->15M(100M) 312.678ms
[2024-11-15T14:29:00.000+0000][gc] GC(99) Pause Young (Allocation Failure) 65M->18M(100M) 498.789ms
[2024-11-15T14:29:30.000+0000][gc] GC(100) Pause Young (Allocation Failure) 72M->22M(100M) 1234.890ms
[2024-11-15T14:30:00.000+0000][gc] GC(101) Pause Young (Allocation Failure) 80M->28M(100M) 3456.123ms
[2024-11-15T14:30:15.456+0000][gc] GC(102) Pause Full (Ergonomics) 95M->35M(100M) 8234.567ms
[2024-11-15T14:31:00.000+0000][gc] GC(103) Pause Full (Allocation Failure) 98M->40M(100M) 12345.678ms
"""

SYSLOG_1430 = """Nov 15 14:25:00 appserver kernel: TCP: request_sock_TCP: Possible SYN flooding on port 8080
Nov 15 14:28:00 appserver systemd[1]: application.service: Main process exited, code=killed, status=9/KILL
Nov 15 14:28:01 appserver systemd[1]: application.service: Failed with result 'signal'.
Nov 15 14:28:02 appserver systemd[1]: application.service: Service hold-off time over, scheduling restart.
Nov 15 14:28:03 appserver systemd[1]: application.service: Scheduled restart job, restart counter is at 1.
Nov 15 14:28:05 appserver application[23456]: INFO: Application starting...
Nov 15 14:30:00 appserver kernel: Out of memory: Kill process 23457 (java) score 850 or sacrifice child
Nov 15 14:30:01 appserver kernel: Killed process 23457 (java) total-vm:4194304kB, anon-rss:3145728kB, file-rss:0kB
Nov 15 14:30:15 appserver application[23456]: ERROR: Database connection pool exhausted
Nov 15 14:30:16 appserver application[23456]: WARNING: Thread deadlock detected in connection pool
Nov 15 14:30:20 appserver application[23456]: CRITICAL: Application unresponsive, initiating emergency shutdown
"""


def main():
    """Run RLM log analysis demo."""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "RLM LOG ANALYSIS DEMO" + " "*37 + "║")
    print("╚" + "="*78 + "╝")
    print()
    print("Scenario: Application froze at 14:30")
    print("Question: Why did it freeze? What's the root cause?")
    print()
    print("We have 4 log files:")
    print("  • jstack (thread dump)")
    print("  • strace (system call trace)")
    print("  • gc.log (garbage collection)")
    print("  • syslog (system messages)")
    print()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set")
        print()
        print("This demo requires OpenAI API to show RLM in action.")
        print("Set your API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        print()
        print("What RLM will do:")
        print("  1. LLM peeks at logs to understand structure")
        print("  2. LLM writes code: parse_log(context['jstack'])")
        print("  3. LLM writes code: correlate_logs(...)")
        print("  4. LLM writes code: detect_all_patterns(...)")
        print("  5. LLM provides root cause analysis")
        print()
        print("The LLM autonomously uses the log analysis tools!")
        print()
        return 1

    try:
        from log_analysis import RLMLogAnalyzer

        print("="*80)
        print("INITIALIZING RLM LOG ANALYZER")
        print("="*80)
        print()

        analyzer = RLMLogAnalyzer(
            model="gpt-4o-mini",
            enable_logging=True,  # Show execution steps
            track_costs=True,
            max_iterations=20
        )

        # Provide all logs to RLM
        logs = {
            'jstack_14:30': JSTACK_DUMP_1430,
            'strace': STRACE_LOG_1430,
            'gc': GC_LOG_1430,
            'syslog': SYSLOG_1430
        }

        query = """
        Analyze these logs from an application freeze at 14:30.

        Please:
        1. Identify all critical issues
        2. Build a timeline of events
        3. Correlate events across logs
        4. Determine the root cause
        5. Provide actionable recommendations

        Output should be structured and comprehensive.
        """

        print("Sending logs to RLM...")
        print("(Watch the LLM autonomously explore the logs)")
        print()

        # RLM automatically:
        # - Detects log formats
        # - Parses logs
        # - Correlates events
        # - Detects patterns
        # - Provides root cause analysis
        result = analyzer.completion(context=logs, query=query)

        print("\n" + "="*80)
        print("RLM ANALYSIS COMPLETE")
        print("="*80)
        print(result)

        # Show costs
        if analyzer.track_costs:
            costs = analyzer.cost_summary()
            print("\n" + "="*80)
            print("COST SUMMARY")
            print("="*80)
            print(f"Total API calls: {costs.get('total_calls', 0)}")
            print(f"Total tokens: {costs.get('total_tokens', 0):,}")
            print(f"Estimated cost: ${costs.get('estimated_cost_usd', 0):.4f}")

        print("\n" + "="*80)
        print("Demo complete! ✓")
        print("="*80)
        print()
        print("What just happened:")
        print("  • RLM received 4 raw log files")
        print("  • LLM autonomously explored them by writing Python code")
        print("  • LLM used parse_log(), correlate_logs(), detect_all_patterns()")
        print("  • LLM iteratively built understanding")
        print("  • LLM provided comprehensive root cause analysis")
        print()
        print("You just asked a question. RLM did everything else!")
        print()

        return 0

    except ImportError as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure 'openai' package is installed:")
        print("  pip install openai")
        return 1
    except Exception as e:
        print(f"❌ Error during RLM execution: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
