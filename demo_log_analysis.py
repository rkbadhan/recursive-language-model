"""
Demo: RLM for System Log Analysis

This demonstrates the pure RLM approach - just provide raw logs and a prompt.
The LLM writes ALL the code to parse, correlate, and analyze.

No pre-written parsers. No helper functions. Just RLM.
"""

import os
from rlm.rlm_repl import RLM_REPL

# Sample logs from an application freeze incident
LOGS = {
    'jstack_14:30': """
2024-11-15 14:30:15
Full thread dump OpenJDK 64-Bit Server VM (11.0.16+8 mixed mode):

"ConnectionPool-Worker-1" #45 prio=5 os_prio=0 tid=0x00007f8a4c000800 nid=0x1a2b waiting on condition
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.db.ConnectionPool.acquire(ConnectionPool.java:67)
    - waiting to lock <0x00000000e1234560> (a com.example.db.PoolLock)
    at com.example.service.UserService.getUser(UserService.java:34)

"GC-Worker-Thread" #12 daemon prio=9 os_prio=0 tid=0x00007f8a50001000 nid=0x1a3c runnable
   java.lang.Thread.State: RUNNABLE
    at java.lang.Object.wait(Native Method)
    - locked <0x00000000e1234560> (a com.example.db.PoolLock)

Found one Java-level deadlock:
=============================
"ConnectionPool-Worker-1":
  waiting to lock monitor 0x00007f8a4c0008c8 (object 0x00000000e1234560, a com.example.db.PoolLock),
  which is held by "GC-Worker-Thread"
"GC-Worker-Thread":
  waiting to lock monitor 0x00007f8a500010b0 (object 0x00000000e1234560, a com.example.db.PoolLock),
  which is held by "ConnectionPool-Worker-1"
""",

    'gc_log': """
[2024-11-15T14:30:12.123+0000][gc] GC(100) Pause Young (Normal) (G1 Evacuation Pause) 45M->12M(100M) 234.567ms
[2024-11-15T14:30:15.456+0000][gc] GC(101) Pause Young (Normal) (G1 Evacuation Pause) 55M->15M(100M) 5234.567ms
[2024-11-15T14:30:20.789+0000][gc] GC(102) Pause Full (Allocation Failure) 95M->25M(100M) 15000.123ms
[2024-11-15T14:31:00.000+0000][gc] GC(103) Pause Full (Ergonomics) 90M->30M(100M) 8000.123ms
""",

    'strace': """
14:30:15.123456 read(3, "GET /api/users HTTP/1.1\\r\\n", 8192) = 24
14:30:15.234567 open("/var/log/app.log", O_WRONLY|O_APPEND) = 4
14:30:15.345678 write(4, "Processing request\\n", 19) = 19
14:30:15.456789 poll([{fd=5, events=POLLIN}], 1, 5000) = 0 (Timeout)
14:30:20.457890 poll([{fd=5, events=POLLIN}], 1, 5000) = 0 (Timeout)
14:30:25.458901 read(5, 0x7fff12345678, 1024) = -1 ETIMEDOUT (Connection timed out)
14:30:25.459012 write(2, "ERROR: Database timeout\\n", 24) = 24
""",

    'syslog': """
Nov 15 14:30:10 hostname application[12346]: INFO: Starting request processing
Nov 15 14:30:12 hostname kernel: [12345.678901] Out of memory: Kill process 12346 or sacrifice child
Nov 15 14:30:15 hostname application[12346]: ERROR: Database connection pool exhausted
Nov 15 14:30:16 hostname application[12346]: WARNING: Retrying database connection (attempt 1/3)
Nov 15 14:30:20 hostname application[12346]: ERROR: All database connections timed out
Nov 15 14:30:25 hostname application[12346]: CRITICAL: Application freeze detected
"""
}

QUERY = """
You are analyzing system logs from an application freeze incident at 14:30.

# Context Structure
The context contains 4 log files from the incident:
- 'jstack_14:30': Java thread dump (jstack format)
- 'gc_log': JVM garbage collection log
- 'strace': System call trace
- 'syslog': System log messages

# Log Format Guide

**jstack format:**
- Thread dumps showing thread states (BLOCKED, WAITING, RUNNABLE)
- Lock information: "waiting to lock <0xHEX>" or "locked <0xHEX>"
- Stack traces with line numbers
- Deadlock detection: "Found one Java-level deadlock:"

**GC log format:**
- Format: [timestamp][gc] GC(N) Pause Type (Reason) BeforeM->AfterM(TotalM) DurationMs
- Watch for: Long pause times (>1000ms), Full GC events, Allocation failures
- Correlate GC pauses with app behavior

**strace format:**
- Format: timestamp syscall(args) = return_value [errno]
- Key syscalls: read(), write(), open(), poll(), connect()
- Look for: Timeouts (ETIMEDOUT), slow syscalls (>1s between timestamps)

**syslog format:**
- Format: Month Day Time hostname process[pid]: LEVEL: message
- Levels: INFO, WARNING, ERROR, CRITICAL
- Look for: OOM killer, connection errors, timeouts

# Analysis Strategy

1. **Parse each log** - Write Python code using regex/string parsing to extract:
   - Timestamps (normalize to comparable format)
   - Key events (deadlocks, errors, slow operations)
   - Thread/process states

2. **Build timeline** - Correlate events by timestamp:
   - What happened first?
   - What cascade of events followed?
   - Time gaps between related events

3. **Pattern detection** - Look for:
   - Deadlocks: Circular lock dependencies in jstack
   - Memory pressure: Long GC pauses, OOM messages
   - I/O blocking: Slow syscalls, timeouts in strace
   - Resource exhaustion: Connection pool errors

4. **Root cause analysis** - Connect the dots:
   - What was the triggering event?
   - How did it propagate?
   - Why did the system freeze?

5. **Evidence-based answer** - Provide:
   - Specific log excerpts as evidence
   - Timeline of events
   - Root cause explanation
   - Actionable recommendations

Write Python code to systematically analyze the logs and determine the root cause.
"""


def main():
    print("\n" + "="*80)
    print("RLM DEMO: System Log Analysis")
    print("="*80)
    print()
    print("This demo shows the pure RLM approach:")
    print("  • Raw logs provided as context")
    print("  • A prompt asking for analysis")
    print("  • LLM writes ALL the code to parse and analyze")
    print("  • NO pre-written helper functions")
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        print("Set it to run this demo:")
        print("  export OPENAI_API_KEY='your-key'")
        return 1

    print("Initializing RLM...")
    rlm = RLM_REPL(
        model="gpt-4o-mini",
        enable_logging=True,
        max_iterations=15
    )

    print()
    print("="*80)
    print("RLM is now analyzing the logs...")
    print("(Watch it write code to parse, correlate, and find the root cause)")
    print("="*80)
    print()

    result = rlm.completion(context=LOGS, query=QUERY)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print(result)
    print()
    print("="*80)
    print()
    print("Notice: RLM wrote all the parsing and analysis code itself!")
    print("No pre-written helpers needed.")


if __name__ == "__main__":
    main()
