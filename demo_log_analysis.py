"""
Demo: System Log Analysis with Cross-Log Correlation

This demo shows how to use the RLM log analysis system to:
1. Parse different log formats (jstack, strace, GC, syslog)
2. Correlate events across multiple logs
3. Detect known issue patterns
4. Generate analysis summaries

Note: This demo can run in two modes:
- Parser-only mode (no API key needed) - shows parsing and correlation
- Full RLM mode (requires OpenAI API key) - includes AI analysis
"""

import sys
import os

# Realistic multi-log scenario: Application freeze at 14:30
# ============================================================================

# JStack dump taken during the freeze
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

# Strace output showing I/O blocking around the same time
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

# GC log showing memory pressure leading up to the freeze
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

# Syslog showing system-level issues
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


def demo_parser_only():
    """Demo showing parsing and correlation without RLM (no API key needed)."""
    print("="*80)
    print("DEMO 1: Log Parsing and Correlation (No API Key Required)")
    print("="*80)
    print()
    print("Scenario: Application froze at 14:30. We have logs from multiple sources.")
    print()

    from rlm.log_parsers import parse_log, detect_log_format
    from rlm.log_correlator import correlate_logs, detect_all_patterns, generate_correlation_summary

    # Step 1: Parse each log
    print("Step 1: Parsing logs...")
    print("-" * 40)

    jstack_format = detect_log_format(JSTACK_DUMP_1430)
    print(f"✓ jstack: detected as '{jstack_format}'")
    jstack_parsed = parse_log(JSTACK_DUMP_1430, jstack_format)
    print(f"  - Threads: {jstack_parsed['total_threads']}")
    print(f"  - Deadlock: {jstack_parsed['has_deadlock']}")
    print(f"  - Blocked threads: {len(jstack_parsed['blocked_threads'])}")

    strace_format = detect_log_format(STRACE_LOG_1430)
    print(f"✓ strace: detected as '{strace_format}'")
    strace_parsed = parse_log(STRACE_LOG_1430, strace_format)
    print(f"  - Total syscalls: {strace_parsed['total_syscalls']}")
    print(f"  - Slow calls (>1s): {len(strace_parsed['slow_calls'])}")
    print(f"  - Errors: {len(strace_parsed['errors'])}")

    gc_format = detect_log_format(GC_LOG_1430)
    print(f"✓ GC log: detected as '{gc_format}'")
    gc_parsed = parse_log(GC_LOG_1430, gc_format)
    print(f"  - Collections: {gc_parsed['total_collections']}")
    print(f"  - Max pause: {gc_parsed['max_pause_ms']:.1f}ms")
    print(f"  - Long pauses (>1s): {len(gc_parsed['long_pauses'])}")

    syslog_format = detect_log_format(SYSLOG_1430)
    print(f"✓ syslog: detected as '{syslog_format}'")
    syslog_parsed = parse_log(SYSLOG_1430, syslog_format)
    print(f"  - Entries: {syslog_parsed['total_entries']}")
    print(f"  - Errors: {syslog_parsed['error_count']}")
    print(f"  - Warnings: {syslog_parsed['warning_count']}")

    # Step 2: Correlate logs
    print()
    print("Step 2: Correlating events across logs...")
    print("-" * 40)

    parsed_logs = {
        'jstack_14:30': jstack_parsed,
        'strace': strace_parsed,
        'gc': gc_parsed,
        'syslog': syslog_parsed
    }

    timeline = correlate_logs(parsed_logs)
    print(f"✓ Timeline created:")
    print(f"  - Total events: {len(timeline.events)}")
    print(f"  - Sources: {list(timeline.by_source.keys())}")
    for source, events in timeline.by_source.items():
        print(f"  - {source}: {len(events)} events")

    # Step 3: Detect patterns
    print()
    print("Step 3: Detecting issue patterns...")
    print("-" * 40)

    patterns = detect_all_patterns(timeline)
    print(f"✓ Patterns detected: {len(patterns)}")
    print()

    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. [{pattern['severity']}] {pattern['pattern'].upper()}")
        print(f"   {pattern['description']}")
        print()

    # Step 4: Generate summary
    print("Step 4: Generating correlation summary...")
    print("-" * 40)
    print()

    summary = generate_correlation_summary(timeline, patterns)
    print(summary)

    # Step 5: Manual root cause analysis
    print()
    print("Step 5: Root Cause Analysis (Manual)")
    print("="*80)
    print()
    print("Based on the correlated logs, here's what happened:")
    print()
    print("1. MEMORY PRESSURE (14:25-14:30)")
    print("   - GC pause times increased from 156ms to 12,345ms (78x worse)")
    print("   - Heap usage grew from 45M to 98M")
    print("   - Indicates memory leak or excessive allocation")
    print()
    print("2. OOM KILL (14:30:00)")
    print("   - Kernel killed Java process due to memory exhaustion")
    print("   - Confirmed in syslog: 'Out of memory: Kill process 23457 (java)'")
    print()
    print("3. DATABASE TIMEOUT (14:30:14)")
    print("   - Application restarted but database connection timed out")
    print("   - Strace shows 2.5s timeout on database read")
    print()
    print("4. DEADLOCK (14:30:15)")
    print("   - Connection pool deadlock detected in jstack")
    print("   - Workers waiting on locks held by cleanup thread")
    print("   - Application completely frozen")
    print()
    print("ROOT CAUSE:")
    print("  Memory leak → OOM → Restart → Connection pool exhaustion → Deadlock")
    print()
    print("RECOMMENDATION:")
    print("  1. Fix memory leak (heap dump analysis needed)")
    print("  2. Fix connection pool lock ordering bug")
    print("  3. Add connection timeout and retry logic")
    print("  4. Increase heap size as temporary mitigation")
    print()


def demo_with_rlm():
    """Demo using RLM for AI-powered analysis (requires API key)."""
    print("="*80)
    print("DEMO 2: AI-Powered Log Analysis with RLM (Requires OpenAI API Key)")
    print("="*80)
    print()

    try:
        from rlm import RLMLogAnalyzer

        analyzer = RLMLogAnalyzer(
            model="gpt-4o-mini",
            enable_logging=True,
            track_costs=True
        )

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

        print("Sending logs to RLM for analysis...")
        print("(This may take 30-60 seconds)")
        print()

        result = analyzer.completion(context=logs, query=query)

        print("\n" + "="*80)
        print("RLM ANALYSIS RESULT")
        print("="*80)
        print(result)

        # Show costs
        if analyzer.track_costs:
            costs = analyzer.cost_summary()
            print("\n" + "="*80)
            print("COST SUMMARY")
            print("="*80)
            print(f"Total API calls: {costs.get('total_calls', 0)}")
            print(f"Total tokens: {costs.get('total_tokens', 0)}")
            print(f"Estimated cost: ${costs.get('estimated_cost_usd', 0):.4f}")

    except ImportError:
        print("RLMLogAnalyzer requires the 'openai' package.")
        print("Install it with: pip install openai")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have OPENAI_API_KEY set in your environment.")


def main():
    """Run demos."""
    print()
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "RLM LOG ANALYSIS DEMO" + " "*32 + "║")
    print("╚" + "="*78 + "╝")
    print()

    # Always run parser demo (no API key needed)
    demo_parser_only()

    # Ask if user wants to run RLM demo
    print("\n" + "="*80)
    print()
    response = input("Run AI-powered analysis demo? (requires OpenAI API key) [y/N]: ")

    if response.lower() in ('y', 'yes'):
        print()
        demo_with_rlm()
    else:
        print()
        print("Skipping AI demo. You can run it later by setting OPENAI_API_KEY")
        print("and running: python demo_log_analysis.py")

    print()
    print("="*80)
    print("Demo complete! ✓")
    print("="*80)


if __name__ == "__main__":
    main()
