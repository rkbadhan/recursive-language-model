"""
Standalone tests for log parsing and correlation (no OpenAI dependencies).

Run this to test the core log analysis functionality without needing API keys.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest
from log_analysis.log_parsers import (
    parse_jstack, parse_strace, parse_gc_log, parse_pstack,
    parse_syslog, parse_json_logs, detect_log_format, parse_log
)
from log_analysis.log_correlator import (
    correlate_logs, detect_all_patterns,
    extract_events_from_jstack, extract_events_from_strace,
    extract_events_from_gc, detect_deadlock_pattern,
    detect_memory_pressure_pattern, detect_gc_caused_blocking
)


# Same test data as before
SAMPLE_JSTACK = """2024-11-15 14:30:15
Full thread dump OpenJDK 64-Bit Server VM (11.0.16+8 mixed mode):

"Thread-1" #12 prio=5 os_prio=0 tid=0x00007f8a4c000800 nid=0x1a2b waiting on condition [0x00007f8a3d6fe000]
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.Service.process(Service.java:45)
    - waiting to lock <0x00000000e1234560> (a java.lang.Object)
    at com.example.Worker.run(Worker.java:23)

"Thread-2" #13 prio=5 os_prio=0 tid=0x00007f8a4c001000 nid=0x1a2c waiting on condition [0x00007f8a3d7ff000]
   java.lang.Thread.State: BLOCKED (on object monitor)
    at com.example.Service.process(Service.java:50)
    - waiting to lock <0x00000000e1234000> (a java.lang.Object)
    - locked <0x00000000e1234560> (a java.lang.Object)
    at com.example.Worker.run(Worker.java:23)

"Thread-3" #14 prio=5 os_prio=0 tid=0x00007f8a4c002000 nid=0x1a2d runnable [0x00007f8a3d8ff000]
   java.lang.Thread.State: RUNNABLE
    at com.example.Service.compute(Service.java:100)
    - locked <0x00000000e1234000> (a java.lang.Object)
    at com.example.Worker.run(Worker.java:23)

Found one Java-level deadlock:
=============================
"Thread-1":
  waiting to lock monitor 0x00007f8a4c003000 (object 0x00000000e1234560, a java.lang.Object),
  which is held by "Thread-2"
"""

SAMPLE_STRACE = """14:30:12.123456 open("/etc/config.conf", O_RDONLY) = 3 <0.000025>
14:30:12.123500 read(3, "config_data\\n", 1024) = 12 <0.000018>
14:30:12.123550 close(3) = 0 <0.000010>
14:30:13.500000 connect(4, {sa_family=AF_INET, sin_port=htons(8080)}, 16) = -1 ECONNREFUSED (Connection refused) <0.000030>
14:30:20.000000 futex(0x7f8a4c000000, FUTEX_WAIT, 0, NULL) = 0 <5.234567>
"""

SAMPLE_GC_LOG = """[2024-11-15T14:29:58.123+0000][gc] GC(100) Pause Young (Normal) 50M->10M(100M) 200.123ms
[2024-11-15T14:30:15.456+0000][gc] GC(101) Pause Young (Normal) 55M->15M(100M) 5234.567ms
[2024-11-15T14:31:00.000+0000][gc] GC(103) Pause Full (Ergonomics) 90M->30M(100M) 8000.123ms
"""

SAMPLE_SYSLOG = """Nov 15 14:30:00 hostname systemd[1]: Starting application service...
Nov 15 14:30:15 hostname application[12346]: ERROR: Database connection failed
Nov 15 14:30:16 hostname application[12346]: WARNING: Retrying connection
"""


print("="*70)
print("LOG PARSER TESTS (Standalone - No OpenAI Required)")
print("="*70)
print()

# Test 1: jstack parser
print("Test 1: Parsing jstack...")
result = parse_jstack(SAMPLE_JSTACK)
assert result['total_threads'] == 3, f"Expected 3 threads, got {result['total_threads']}"
assert result['has_deadlock'] == True, "Should detect deadlock"
assert len(result['blocked_threads']) == 2, f"Expected 2 blocked threads, got {len(result['blocked_threads'])}"
print(f"✓ jstack: {result['total_threads']} threads, deadlock={result['has_deadlock']}, blocked={len(result['blocked_threads'])}")

# Test 2: strace parser
print("\nTest 2: Parsing strace...")
result = parse_strace(SAMPLE_STRACE)
assert result['total_syscalls'] >= 5, f"Expected >=5 syscalls, got {result['total_syscalls']}"
assert len(result['slow_calls']) > 0, "Should detect slow calls (>1s)"
print(f"✓ strace: {result['total_syscalls']} syscalls, {len(result['slow_calls'])} slow, {len(result['errors'])} errors")

# Test 3: GC parser
print("\nTest 3: Parsing GC logs...")
result = parse_gc_log(SAMPLE_GC_LOG)
assert result['total_collections'] == 3, f"Expected 3 GC events, got {result['total_collections']}"
assert result['major_collections'] == 1, f"Expected 1 major GC, got {result['major_collections']}"
assert result['max_pause_ms'] > 8000, f"Expected max pause >8000ms, got {result['max_pause_ms']}"
print(f"✓ GC: {result['total_collections']} collections, max_pause={result['max_pause_ms']:.1f}ms")

# Test 4: syslog parser
print("\nTest 4: Parsing syslog...")
result = parse_syslog(SAMPLE_SYSLOG)
assert result['total_entries'] == 3, f"Expected 3 entries, got {result['total_entries']}"
assert result['error_count'] == 1, f"Expected 1 error, got {result['error_count']}"
assert result['warning_count'] == 1, f"Expected 1 warning, got {result['warning_count']}"
print(f"✓ syslog: {result['total_entries']} entries, {result['error_count']} errors, {result['warning_count']} warnings")

# Test 5: Format detection
print("\nTest 5: Format auto-detection...")
assert detect_log_format(SAMPLE_JSTACK) == 'jstack'
assert detect_log_format(SAMPLE_STRACE) == 'strace'
assert detect_log_format(SAMPLE_GC_LOG) == 'gc'
assert detect_log_format(SAMPLE_SYSLOG) == 'syslog'
print("✓ Format detection: all formats correctly identified")

# Test 6: Universal parser
print("\nTest 6: Universal parser...")
result = parse_log(SAMPLE_JSTACK)
assert result['format'] == 'jstack'
assert 'threads' in result
result = parse_log(SAMPLE_GC_LOG)
assert result['format'] == 'gc'
assert 'gc_events' in result
print("✓ Universal parser: auto-detects and parses correctly")

# Test 7: Event extraction
print("\nTest 7: Event extraction...")
parsed_jstack = parse_jstack(SAMPLE_JSTACK)
events = extract_events_from_jstack(parsed_jstack, 'jstack')
assert len(events) > 0, "Should extract events from jstack"
deadlock_events = [e for e in events if e.data['type'] == 'deadlock']
assert len(deadlock_events) == 1, "Should have 1 deadlock event"
assert deadlock_events[0].data['severity'] == 'CRITICAL'
print(f"✓ Event extraction: {len(events)} events, including {len(deadlock_events)} deadlock")

# Test 8: Log correlation
print("\nTest 8: Multi-log correlation...")
parsed_logs = {
    'jstack': parse_log(SAMPLE_JSTACK),  # Use parse_log to add 'format' key
    'gc': parse_log(SAMPLE_GC_LOG),
    'strace': parse_log(SAMPLE_STRACE)
}
timeline = correlate_logs(parsed_logs)
assert len(timeline.by_source) == 3, f"Expected 3 sources, got {len(timeline.by_source)}"
assert len(timeline.events) > 0, "Timeline should have events"
print(f"✓ Correlation: {len(timeline.events)} total events from {len(timeline.by_source)} sources")

# Test 9: Pattern detection
print("\nTest 9: Pattern detection...")
patterns = detect_all_patterns(timeline)
assert len(patterns) > 0, "Should detect at least one pattern (deadlock)"
deadlock_patterns = [p for p in patterns if p['pattern'] == 'deadlock']
assert len(deadlock_patterns) == 1, "Should detect the deadlock pattern"
print(f"✓ Pattern detection: {len(patterns)} patterns detected")
for p in patterns:
    print(f"  - {p['severity']}: {p['pattern']}")

# Test 10: Timeline functionality
print("\nTest 10: Timeline operations...")
timeline_dict = timeline.to_dict()
assert 'total_events' in timeline_dict
assert 'sources' in timeline_dict
assert timeline_dict['sources'] == ['jstack', 'gc', 'strace']
print(f"✓ Timeline: {timeline_dict['total_events']} events, sources={timeline_dict['sources']}")

# Final summary
print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print("\nLog analysis parsers and correlators are working correctly.")
print("Core functionality validated without requiring OpenAI API.")
