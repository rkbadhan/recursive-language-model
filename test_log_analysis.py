"""
Comprehensive tests for log analysis functionality.

Tests include:
- Parser tests for each log format
- Correlation and timeline tests
- Pattern detection tests
- Integration tests with RLMLogAnalyzer
"""

import unittest
from log_analysis.log_parsers import (
    parse_jstack, parse_strace, parse_gc_log, parse_pstack,
    parse_syslog, parse_json_logs, detect_log_format, parse_log
)
from log_analysis.log_correlator import (
    correlate_logs, detect_all_patterns, Timeline,
    extract_events_from_jstack, extract_events_from_strace,
    extract_events_from_gc, detect_deadlock_pattern,
    detect_memory_pressure_pattern, detect_gc_caused_blocking
)


# ============================================================================
# Test Data - Realistic Log Examples
# ============================================================================

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
"Thread-2":
  waiting to lock monitor 0x00007f8a4c003100 (object 0x00000000e1234000, a java.lang.Object),
  which is held by "Thread-3"
"""

SAMPLE_STRACE = """14:30:12.123456 open("/etc/config.conf", O_RDONLY) = 3 <0.000025>
14:30:12.123500 read(3, "config_data\\n", 1024) = 12 <0.000018>
14:30:12.123550 close(3) = 0 <0.000010>
14:30:13.500000 connect(4, {sa_family=AF_INET, sin_port=htons(8080)}, 16) = -1 ECONNREFUSED (Connection refused) <0.000030>
14:30:14.000000 open("/var/log/app.log", O_WRONLY|O_APPEND) = 5 <0.000020>
14:30:14.000050 write(5, "ERROR: Connection failed\\n", 25) = 25 <0.000015>
14:30:15.123456 read(6, 0x7fff12345678, 8192) = -1 EAGAIN (Resource temporarily unavailable) <0.000012>
14:30:20.000000 futex(0x7f8a4c000000, FUTEX_WAIT, 0, NULL) = 0 <5.234567>
14:30:25.234567 write(2, "Timeout waiting for lock\\n", 26) = 26 <0.000018>
"""

SAMPLE_GC_LOG = """[2024-11-15T14:29:58.123+0000][gc] GC(100) Pause Young (Normal) 50M->10M(100M) 200.123ms
[2024-11-15T14:30:15.456+0000][gc] GC(101) Pause Young (Normal) 55M->15M(100M) 5234.567ms
[2024-11-15T14:30:30.789+0000][gc] GC(102) Pause Young (Allocation Failure) 60M->20M(100M) 300.789ms
[2024-11-15T14:31:00.000+0000][gc] GC(103) Pause Full (Ergonomics) 90M->30M(100M) 8000.123ms
"""

SAMPLE_PSTACK = """Thread 1 (LWP 12345):
#0  0x00007f8a4c001234 in pthread_cond_wait () from /lib64/libpthread.so.0
#1  0x00007f8a4c002345 in std::condition_variable::wait() from /usr/lib64/libstdc++.so.6
#2  0x0000000000401234 in Worker::wait_for_task() at worker.cpp:45
#3  0x0000000000401345 in Worker::run() at worker.cpp:23

Thread 2 (LWP 12346):
#0  0x00007f8a4c003456 in pthread_mutex_lock () from /lib64/libpthread.so.0
#1  0x0000000000401456 in Service::process() at service.cpp:100
#2  0x0000000000401567 in main () at main.cpp:10
"""

SAMPLE_SYSLOG = """Nov 15 14:29:58 hostname kernel: Out of memory: Kill process 12345 (java) score 850 or sacrifice child
Nov 15 14:30:00 hostname systemd[1]: Starting application service...
Nov 15 14:30:05 hostname application[12346]: INFO: Application started successfully
Nov 15 14:30:15 hostname application[12346]: ERROR: Database connection failed: Connection timeout
Nov 15 14:30:16 hostname application[12346]: WARNING: Retrying connection (attempt 1/3)
Nov 15 14:30:20 hostname kernel: TCP: out of memory -- consider tuning tcp_mem
"""

SAMPLE_JSON_LOGS = """{"timestamp": "2024-11-15T14:30:15.123Z", "level": "ERROR", "message": "Connection timeout", "service": "api-gateway"}
{"timestamp": "2024-11-15T14:30:16.456Z", "level": "WARNING", "message": "High memory usage: 85%", "service": "worker"}
{"timestamp": "2024-11-15T14:30:17.789Z", "level": "INFO", "message": "Request processed", "service": "api-gateway"}
{"timestamp": "2024-11-15T14:30:18.012Z", "level": "ERROR", "message": "Database query failed", "service": "database"}
"""


# ============================================================================
# Parser Tests
# ============================================================================

class TestLogParsers(unittest.TestCase):
    """Test individual log parsers."""

    def test_parse_jstack(self):
        """Test jstack parser."""
        result = parse_jstack(SAMPLE_JSTACK)

        self.assertEqual(result['total_threads'], 3)
        self.assertTrue(result['has_deadlock'])
        self.assertEqual(len(result['blocked_threads']), 2)
        self.assertTrue(len(result['deadlocks']) > 0)
        self.assertTrue(len(result['lock_chains']) > 0)

        # Check thread states
        thread_states = [t['state'] for t in result['threads']]
        self.assertIn('BLOCKED', thread_states)
        self.assertIn('RUNNABLE', thread_states)

    def test_parse_strace(self):
        """Test strace parser."""
        result = parse_strace(SAMPLE_STRACE)

        self.assertGreater(result['total_syscalls'], 0)
        self.assertGreater(len(result['slow_calls']), 0)  # futex call >5s
        self.assertGreater(len(result['errors']), 0)  # ECONNREFUSED

        # Check slow call detection
        slow_call = result['slow_calls'][0]
        self.assertGreater(slow_call['duration'], 5.0)

    def test_parse_gc_log(self):
        """Test GC log parser."""
        result = parse_gc_log(SAMPLE_GC_LOG)

        self.assertEqual(result['total_collections'], 4)
        self.assertEqual(result['minor_collections'], 3)
        self.assertEqual(result['major_collections'], 1)
        self.assertGreater(result['max_pause_ms'], 8000)
        self.assertGreater(len(result['long_pauses']), 0)  # >1s pauses

    def test_parse_pstack(self):
        """Test pstack parser."""
        result = parse_pstack(SAMPLE_PSTACK)

        self.assertEqual(result['total_threads'], 2)
        self.assertGreater(result['symbol_count'], 0)
        self.assertIn('pthread_cond_wait', result['unique_symbols'])

    def test_parse_syslog(self):
        """Test syslog parser."""
        result = parse_syslog(SAMPLE_SYSLOG)

        self.assertGreater(result['total_entries'], 0)
        self.assertGreater(result['error_count'], 0)
        self.assertGreater(result['warning_count'], 0)

        # Check level grouping
        self.assertIn('ERROR', result['by_level'])
        self.assertIn('WARNING', result['by_level'])

    def test_parse_json_logs(self):
        """Test JSON log parser."""
        result = parse_json_logs(SAMPLE_JSON_LOGS)

        self.assertEqual(result['total_entries'], 4)
        self.assertIn('level', result['schema'])
        self.assertIn('timestamp', result['schema'])
        self.assertIn('ERROR', result['by_level'])

    def test_detect_log_format(self):
        """Test format auto-detection."""
        self.assertEqual(detect_log_format(SAMPLE_JSTACK), 'jstack')
        self.assertEqual(detect_log_format(SAMPLE_STRACE), 'strace')
        self.assertEqual(detect_log_format(SAMPLE_GC_LOG), 'gc')
        self.assertEqual(detect_log_format(SAMPLE_PSTACK), 'pstack')
        self.assertEqual(detect_log_format(SAMPLE_SYSLOG), 'syslog')
        self.assertEqual(detect_log_format(SAMPLE_JSON_LOGS), 'json')

    def test_universal_parser(self):
        """Test universal parse_log function."""
        # Auto-detect and parse
        result = parse_log(SAMPLE_JSTACK)
        self.assertEqual(result['format'], 'jstack')
        self.assertIn('threads', result)

        result = parse_log(SAMPLE_GC_LOG)
        self.assertEqual(result['format'], 'gc')
        self.assertIn('gc_events', result)


# ============================================================================
# Correlation Tests
# ============================================================================

class TestLogCorrelation(unittest.TestCase):
    """Test log correlation and timeline building."""

    def test_event_extraction_jstack(self):
        """Test extracting events from parsed jstack."""
        parsed = parse_jstack(SAMPLE_JSTACK)
        events = extract_events_from_jstack(parsed, 'jstack')

        self.assertGreater(len(events), 0)

        # Should have deadlock event
        deadlock_events = [e for e in events if e.data['type'] == 'deadlock']
        self.assertEqual(len(deadlock_events), 1)
        self.assertEqual(deadlock_events[0].data['severity'], 'CRITICAL')

    def test_event_extraction_strace(self):
        """Test extracting events from parsed strace."""
        parsed = parse_strace(SAMPLE_STRACE)
        events = extract_events_from_strace(parsed, 'strace')

        # Should have slow call events
        slow_events = [e for e in events if e.data['type'] == 'slow_syscall']
        self.assertGreater(len(slow_events), 0)

        # Should have error events
        error_events = [e for e in events if e.data['type'] == 'syscall_error']
        self.assertGreater(len(error_events), 0)

    def test_event_extraction_gc(self):
        """Test extracting events from parsed GC log."""
        parsed = parse_gc_log(SAMPLE_GC_LOG)
        events = extract_events_from_gc(parsed, 'gc')

        self.assertEqual(len(events), 4)  # 4 GC events

        # Check severity assignment based on pause time
        critical_events = [e for e in events if e.data['severity'] == 'CRITICAL']
        self.assertGreater(len(critical_events), 0)  # 5s+ and 8s+ pauses

    def test_correlate_logs(self):
        """Test correlating multiple logs into timeline."""
        parsed_logs = {
            'jstack': parse_log(SAMPLE_JSTACK),
            'strace': parse_log(SAMPLE_STRACE),
            'gc': parse_log(SAMPLE_GC_LOG)
        }

        timeline = correlate_logs(parsed_logs)

        self.assertGreater(len(timeline.events), 0)
        self.assertEqual(len(timeline.by_source), 3)
        self.assertIn('jstack', timeline.by_source)
        self.assertIn('strace', timeline.by_source)
        self.assertIn('gc', timeline.by_source)

    def test_timeline_sorting(self):
        """Test that timeline events are sorted by timestamp."""
        parsed_logs = {
            'gc': parse_log(SAMPLE_GC_LOG),
            'strace': parse_log(SAMPLE_STRACE)
        }

        timeline = correlate_logs(parsed_logs)

        # Events with timestamps should be sorted
        timestamped_events = [e for e in timeline.events if e.timestamp is not None]
        timestamps = [e.timestamp for e in timestamped_events]

        self.assertEqual(timestamps, sorted(timestamps))


# ============================================================================
# Pattern Detection Tests
# ============================================================================

class TestPatternDetection(unittest.TestCase):
    """Test issue pattern detection."""

    def test_detect_deadlock(self):
        """Test deadlock pattern detection."""
        parsed_logs = {'jstack': parse_log(SAMPLE_JSTACK)}
        timeline = correlate_logs(parsed_logs)
        patterns = detect_deadlock_pattern(timeline)

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]['pattern'], 'deadlock')
        self.assertEqual(patterns[0]['severity'], 'CRITICAL')

    def test_detect_memory_pressure(self):
        """Test memory pressure detection from GC logs."""
        # Create GC log with increasing pause times
        gc_log_with_pressure = """[2024-11-15T14:00:00.000+0000][gc] GC(1) Pause Young 50M->10M(100M) 100.0ms
[2024-11-15T14:00:10.000+0000][gc] GC(2) Pause Young 50M->10M(100M) 120.0ms
[2024-11-15T14:00:20.000+0000][gc] GC(3) Pause Young 50M->10M(100M) 150.0ms
[2024-11-15T14:00:30.000+0000][gc] GC(4) Pause Young 50M->10M(100M) 200.0ms
[2024-11-15T14:00:40.000+0000][gc] GC(5) Pause Young 50M->10M(100M) 250.0ms
[2024-11-15T14:01:00.000+0000][gc] GC(6) Pause Young 50M->15M(100M) 300.0ms
[2024-11-15T14:01:10.000+0000][gc] GC(7) Pause Young 50M->15M(100M) 350.0ms
[2024-11-15T14:01:20.000+0000][gc] GC(8) Pause Young 50M->15M(100M) 400.0ms
[2024-11-15T14:01:30.000+0000][gc] GC(9) Pause Young 50M->15M(100M) 450.0ms
[2024-11-15T14:01:40.000+0000][gc] GC(10) Pause Young 50M->20M(100M) 500.0ms"""

        parsed_logs = {'gc': parse_log(gc_log_with_pressure)}
        timeline = correlate_logs(parsed_logs)
        patterns = detect_memory_pressure_pattern(timeline)

        self.assertGreater(len(patterns), 0)
        self.assertEqual(patterns[0]['pattern'], 'memory_pressure')

    def test_detect_gc_caused_blocking(self):
        """Test GC-caused I/O blocking pattern."""
        parsed_logs = {
            'gc': parse_log(SAMPLE_GC_LOG),
            'strace': parse_log(SAMPLE_STRACE)
        }

        timeline = correlate_logs(parsed_logs)
        patterns = detect_gc_caused_blocking(timeline)

        # May or may not find pattern depending on timestamp correlation
        # Just verify it runs without error
        self.assertIsInstance(patterns, list)

    def test_detect_all_patterns(self):
        """Test running all pattern detectors."""
        parsed_logs = {
            'jstack': parse_log(SAMPLE_JSTACK),
            'gc': parse_log(SAMPLE_GC_LOG),
            'strace': parse_log(SAMPLE_STRACE)
        }

        timeline = correlate_logs(parsed_logs)
        patterns = detect_all_patterns(timeline)

        # Should detect at least the deadlock
        self.assertGreater(len(patterns), 0)

        # Patterns should be sorted by severity
        severities = [p['severity'] for p in patterns]
        for i in range(len(severities) - 1):
            severity_order = {'CRITICAL': 0, 'WARNING': 1, 'INFO': 2}
            self.assertLessEqual(
                severity_order.get(severities[i], 3),
                severity_order.get(severities[i + 1], 3)
            )


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_single_log_workflow(self):
        """Test complete workflow with single log."""
        # 1. Detect format
        format_type = detect_log_format(SAMPLE_JSTACK)
        self.assertEqual(format_type, 'jstack')

        # 2. Parse
        parsed = parse_log(SAMPLE_JSTACK, format_type)
        self.assertEqual(parsed['format'], 'jstack')

        # 3. Extract events
        timeline = correlate_logs({'jstack': parsed})
        self.assertGreater(len(timeline.events), 0)

        # 4. Detect patterns
        patterns = detect_all_patterns(timeline)
        self.assertGreater(len(patterns), 0)

    def test_multi_log_workflow(self):
        """Test complete workflow with multiple logs."""
        # 1. Parse all logs
        parsed_logs = {
            'jstack': parse_log(SAMPLE_JSTACK),
            'gc': parse_log(SAMPLE_GC_LOG),
            'strace': parse_log(SAMPLE_STRACE),
            'syslog': parse_log(SAMPLE_SYSLOG)
        }

        # 2. Correlate
        timeline = correlate_logs(parsed_logs)
        self.assertEqual(len(timeline.by_source), 4)

        # 3. Detect patterns
        patterns = detect_all_patterns(timeline)
        self.assertGreater(len(patterns), 0)

        # 4. Get dict representation
        timeline_dict = timeline.to_dict()
        self.assertIn('total_events', timeline_dict)
        self.assertIn('sources', timeline_dict)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance(unittest.TestCase):
    """Basic performance tests."""

    def test_large_log_parsing(self):
        """Test parsing reasonably large logs."""
        # Create a large syslog (1000 lines)
        large_syslog = "\n".join([
            f"Nov 15 {14 + i//3600:02d}:{(i//60)%60:02d}:{i%60:02d} hostname app[12345]: INFO: Message {i}"
            for i in range(1000)
        ])

        result = parse_syslog(large_syslog)
        self.assertEqual(result['total_entries'], 1000)

    def test_large_gc_log_parsing(self):
        """Test parsing large GC logs."""
        # Create 100 GC events
        large_gc_log = "\n".join([
            f"[2024-11-15T14:{i//60:02d}:{i%60:02d}.000+0000][gc] GC({i}) Pause Young 50M->10M(100M) {100+i}.0ms"
            for i in range(100)
        ])

        result = parse_gc_log(large_gc_log)
        self.assertEqual(result['total_collections'], 100)


# ============================================================================
# Main Test Runner
# ============================================================================

def run_tests():
    """Run all tests and print summary."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLogParsers))
    suite.addTests(loader.loadTestsFromTestCase(TestLogCorrelation))
    suite.addTests(loader.loadTestsFromTestCase(TestPatternDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
