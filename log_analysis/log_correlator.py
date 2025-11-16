"""
Log Correlation Utilities

Tools for correlating events across multiple log files:
- Timestamp alignment and normalization
- Event sequencing and timeline construction
- Multi-log event correlation
- Pattern detection across logs
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from .log_parsers import extract_timestamp, normalize_timestamp


# ============================================================================
# Event and Timeline Data Structures
# ============================================================================

class LogEvent:
    """Represents a single log event with normalized timestamp."""

    def __init__(self, source: str, timestamp: Optional[float], data: Dict[str, Any]):
        """
        Args:
            source: Log source identifier (e.g., 'jstack', 'gc', 'strace')
            timestamp: Unix epoch timestamp (float)
            data: Event-specific data dictionary
        """
        self.source = source
        self.timestamp = timestamp
        self.data = data

    def __repr__(self):
        ts_str = datetime.fromtimestamp(self.timestamp).isoformat() if self.timestamp else 'unknown'
        return f"LogEvent({self.source}, {ts_str}, {self.data.get('type', 'unknown')})"

    def __lt__(self, other):
        """Sort by timestamp."""
        if self.timestamp is None:
            return False
        if other.timestamp is None:
            return True
        return self.timestamp < other.timestamp


class Timeline:
    """Timeline of events from multiple log sources."""

    def __init__(self):
        self.events: List[LogEvent] = []
        self.by_source: Dict[str, List[LogEvent]] = defaultdict(list)

    def add_event(self, event: LogEvent):
        """Add event to timeline."""
        self.events.append(event)
        self.by_source[event.source].append(event)

    def sort(self):
        """Sort events by timestamp."""
        self.events.sort()
        for source in self.by_source:
            self.by_source[source].sort()

    def get_events_in_window(self, start_ts: float, end_ts: float) -> List[LogEvent]:
        """Get events within time window."""
        return [e for e in self.events if e.timestamp and start_ts <= e.timestamp <= end_ts]

    def get_events_around(self, target_ts: float, window_seconds: float = 5.0) -> List[LogEvent]:
        """Get events within +/- window_seconds of target timestamp."""
        return self.get_events_in_window(
            target_ts - window_seconds,
            target_ts + window_seconds
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_events': len(self.events),
            'sources': list(self.by_source.keys()),
            'events': [
                {
                    'source': e.source,
                    'timestamp': e.timestamp,
                    'iso_time': datetime.fromtimestamp(e.timestamp).isoformat() if e.timestamp else None,
                    'data': e.data
                }
                for e in self.events
            ]
        }


# ============================================================================
# Event Extraction from Parsed Logs
# ============================================================================

def extract_events_from_jstack(parsed: Dict[str, Any], source_name: str = 'jstack') -> List[LogEvent]:
    """Extract events from parsed jstack data."""
    events = []

    # Deadlock events
    if parsed.get('has_deadlock'):
        events.append(LogEvent(
            source=source_name,
            timestamp=None,  # jstack typically doesn't have timestamps
            data={
                'type': 'deadlock',
                'severity': 'CRITICAL',
                'deadlocks': parsed['deadlocks'],
                'thread_count': parsed['total_threads']
            }
        ))

    # Blocked threads
    for thread in parsed.get('blocked_threads', []):
        events.append(LogEvent(
            source=source_name,
            timestamp=None,
            data={
                'type': 'thread_blocked',
                'severity': 'WARNING',
                'thread_name': thread['name'],
                'locks_waiting': thread['locks_waiting']
            }
        ))

    # Lock contention events
    for chain in parsed.get('lock_chains', []):
        events.append(LogEvent(
            source=source_name,
            timestamp=None,
            data={
                'type': 'lock_contention',
                'severity': 'WARNING',
                'waiting_thread': chain['waiting_thread'],
                'held_by': chain['held_by'],
                'lock': chain['lock']
            }
        ))

    return events


def extract_events_from_strace(parsed: Dict[str, Any], source_name: str = 'strace') -> List[LogEvent]:
    """Extract events from parsed strace data."""
    events = []

    # Slow syscalls
    for syscall in parsed.get('slow_calls', []):
        ts = normalize_timestamp(syscall['timestamp']) if syscall.get('timestamp') else None
        events.append(LogEvent(
            source=source_name,
            timestamp=ts,
            data={
                'type': 'slow_syscall',
                'severity': 'WARNING',
                'syscall': syscall['name'],
                'duration_sec': syscall['duration'],
                'args': syscall['args']
            }
        ))

    # Error syscalls
    for syscall in parsed.get('errors', []):
        ts = normalize_timestamp(syscall['timestamp']) if syscall.get('timestamp') else None
        events.append(LogEvent(
            source=source_name,
            timestamp=ts,
            data={
                'type': 'syscall_error',
                'severity': 'ERROR',
                'syscall': syscall['name'],
                'error': syscall.get('error'),
                'retval': syscall['retval']
            }
        ))

    return events


def extract_events_from_gc(parsed: Dict[str, Any], source_name: str = 'gc') -> List[LogEvent]:
    """Extract events from parsed GC log data."""
    events = []

    for gc_event in parsed.get('gc_events', []):
        ts_str = gc_event.get('timestamp')
        ts = normalize_timestamp(ts_str) if ts_str else None

        # Determine severity based on pause time
        pause_ms = gc_event['pause_ms']
        if pause_ms > 5000:  # > 5 seconds
            severity = 'CRITICAL'
        elif pause_ms > 1000:  # > 1 second
            severity = 'WARNING'
        else:
            severity = 'INFO'

        events.append(LogEvent(
            source=source_name,
            timestamp=ts,
            data={
                'type': 'gc_pause',
                'severity': severity,
                'gc_type': gc_event['type'],
                'pause_ms': pause_ms,
                'heap_before': gc_event.get('heap_before'),
                'heap_after': gc_event.get('heap_after'),
                'is_full': gc_event.get('is_full', False)
            }
        ))

    return events


def extract_events_from_syslog(parsed: Dict[str, Any], source_name: str = 'syslog') -> List[LogEvent]:
    """Extract events from parsed syslog data."""
    events = []

    for entry in parsed.get('entries', []):
        ts_str = entry.get('timestamp')
        ts = normalize_timestamp(ts_str) if ts_str else None

        # Only extract ERROR and WARNING level events
        if entry['level'] in ('ERROR', 'WARNING'):
            events.append(LogEvent(
                source=source_name,
                timestamp=ts,
                data={
                    'type': 'syslog_message',
                    'severity': entry['level'],
                    'process': entry['process'],
                    'message': entry['message']
                }
            ))

    return events


def extract_events_from_parsed_log(parsed: Dict[str, Any], source_name: str) -> List[LogEvent]:
    """
    Universal event extractor based on log format.

    Args:
        parsed: Parsed log data (output from parse_log)
        source_name: Identifier for this log source

    Returns:
        List of LogEvent objects
    """
    format_type = parsed.get('format', 'unknown')

    extractors = {
        'jstack': extract_events_from_jstack,
        'strace': extract_events_from_strace,
        'gc': extract_events_from_gc,
        'syslog': extract_events_from_syslog,
    }

    extractor = extractors.get(format_type)
    if extractor:
        return extractor(parsed, source_name)
    else:
        # Generic extraction for unknown formats
        return []


# ============================================================================
# Multi-Log Correlation
# ============================================================================

def correlate_logs(parsed_logs: Dict[str, Dict[str, Any]]) -> Timeline:
    """
    Correlate events from multiple parsed log files.

    Args:
        parsed_logs: Dict mapping source_name -> parsed_log_data

    Returns:
        Timeline object with all events sorted chronologically
    """
    timeline = Timeline()

    for source_name, parsed_data in parsed_logs.items():
        events = extract_events_from_parsed_log(parsed_data, source_name)
        for event in events:
            timeline.add_event(event)

    timeline.sort()
    return timeline


def find_correlated_events(timeline: Timeline,
                           source_a: str,
                           source_b: str,
                           max_delta_seconds: float = 5.0) -> List[Tuple[LogEvent, LogEvent]]:
    """
    Find events from two sources that occurred close in time.

    Args:
        timeline: Timeline with events from multiple sources
        source_a: First source name
        source_b: Second source name
        max_delta_seconds: Maximum time difference to consider correlated

    Returns:
        List of (event_a, event_b) tuples
    """
    events_a = timeline.by_source.get(source_a, [])
    events_b = timeline.by_source.get(source_b, [])

    correlated = []

    for event_a in events_a:
        if event_a.timestamp is None:
            continue

        for event_b in events_b:
            if event_b.timestamp is None:
                continue

            delta = abs(event_a.timestamp - event_b.timestamp)
            if delta <= max_delta_seconds:
                correlated.append((event_a, event_b))

    return correlated


# ============================================================================
# Pattern Detection
# ============================================================================

def detect_gc_caused_blocking(timeline: Timeline) -> List[Dict[str, Any]]:
    """
    Detect pattern: GC pause followed by I/O blocking (indicates swapping).

    Returns list of detected pattern instances.
    """
    patterns = []

    gc_events = [e for e in timeline.events if e.source == 'gc' and e.data.get('type') == 'gc_pause']
    strace_events = [e for e in timeline.events if e.source == 'strace' and e.data.get('type') == 'slow_syscall']

    for gc_event in gc_events:
        if gc_event.timestamp is None:
            continue

        # Look for slow I/O within 5 seconds after GC
        nearby_io = [
            s for s in strace_events
            if s.timestamp and gc_event.timestamp <= s.timestamp <= gc_event.timestamp + 5.0
        ]

        if nearby_io and gc_event.data['pause_ms'] > 1000:
            patterns.append({
                'pattern': 'gc_blocking_io',
                'severity': 'WARNING',
                'gc_event': gc_event.data,
                'gc_timestamp': gc_event.timestamp,
                'blocked_io': [s.data for s in nearby_io],
                'description': f"GC pause ({gc_event.data['pause_ms']}ms) followed by {len(nearby_io)} slow I/O operations"
            })

    return patterns


def detect_thread_blocking_pattern(timeline: Timeline) -> List[Dict[str, Any]]:
    """
    Detect pattern: Thread blocking in jstack correlated with syscall blocking.

    Returns list of detected pattern instances.
    """
    patterns = []

    # Find blocked threads from jstack
    blocked_events = [
        e for e in timeline.events
        if e.source == 'jstack' and e.data.get('type') == 'thread_blocked'
    ]

    # Find slow syscalls from strace
    slow_syscalls = [
        e for e in timeline.events
        if e.source == 'strace' and e.data.get('type') == 'slow_syscall'
    ]

    if blocked_events and slow_syscalls:
        patterns.append({
            'pattern': 'thread_syscall_blocking',
            'severity': 'WARNING',
            'blocked_threads': [e.data for e in blocked_events],
            'slow_syscalls': [e.data for e in slow_syscalls],
            'description': f"{len(blocked_events)} blocked threads with {len(slow_syscalls)} slow syscalls"
        })

    return patterns


def detect_memory_pressure_pattern(timeline: Timeline) -> List[Dict[str, Any]]:
    """
    Detect pattern: Increasing GC frequency and duration (memory pressure/leak).

    Returns list of detected pattern instances.
    """
    patterns = []

    gc_events = [
        e for e in timeline.events
        if e.source == 'gc' and e.data.get('type') == 'gc_pause' and e.timestamp
    ]

    if len(gc_events) < 5:
        return patterns

    # Sort by time
    gc_events.sort(key=lambda e: e.timestamp)

    # Check if pause times are increasing
    recent_pauses = [e.data['pause_ms'] for e in gc_events[-5:]]
    early_pauses = [e.data['pause_ms'] for e in gc_events[:5]]

    avg_recent = sum(recent_pauses) / len(recent_pauses)
    avg_early = sum(early_pauses) / len(early_pauses)

    if avg_recent > avg_early * 2:  # 2x increase
        patterns.append({
            'pattern': 'memory_pressure',
            'severity': 'WARNING',
            'avg_early_pause_ms': avg_early,
            'avg_recent_pause_ms': avg_recent,
            'increase_factor': avg_recent / avg_early,
            'total_gc_events': len(gc_events),
            'description': f"GC pause times increasing {avg_recent/avg_early:.1f}x (memory pressure or leak suspected)"
        })

    return patterns


def detect_deadlock_pattern(timeline: Timeline) -> List[Dict[str, Any]]:
    """
    Detect deadlock pattern from jstack data.

    Returns list of detected pattern instances.
    """
    patterns = []

    deadlock_events = [
        e for e in timeline.events
        if e.source == 'jstack' and e.data.get('type') == 'deadlock'
    ]

    for event in deadlock_events:
        patterns.append({
            'pattern': 'deadlock',
            'severity': 'CRITICAL',
            'deadlock_info': event.data.get('deadlocks', []),
            'thread_count': event.data.get('thread_count', 0),
            'description': f"Deadlock detected with {event.data.get('thread_count', 0)} total threads"
        })

    return patterns


def detect_all_patterns(timeline: Timeline) -> List[Dict[str, Any]]:
    """
    Run all pattern detectors and return combined results.

    Args:
        timeline: Timeline with correlated events

    Returns:
        List of all detected patterns
    """
    all_patterns = []

    all_patterns.extend(detect_gc_caused_blocking(timeline))
    all_patterns.extend(detect_thread_blocking_pattern(timeline))
    all_patterns.extend(detect_memory_pressure_pattern(timeline))
    all_patterns.extend(detect_deadlock_pattern(timeline))

    # Sort by severity: CRITICAL > WARNING > INFO
    severity_order = {'CRITICAL': 0, 'WARNING': 1, 'INFO': 2}
    all_patterns.sort(key=lambda p: severity_order.get(p.get('severity', 'INFO'), 3))

    return all_patterns


# ============================================================================
# Analysis Summary Generation
# ============================================================================

def generate_correlation_summary(timeline: Timeline, patterns: List[Dict[str, Any]]) -> str:
    """
    Generate human-readable summary of correlation analysis.

    Args:
        timeline: Timeline with all events
        patterns: List of detected patterns

    Returns:
        Markdown-formatted summary string
    """
    summary_parts = []

    summary_parts.append("# Log Correlation Analysis\n")

    # Overview
    summary_parts.append("## Overview\n")
    summary_parts.append(f"- **Total Events**: {len(timeline.events)}")
    summary_parts.append(f"- **Log Sources**: {', '.join(timeline.by_source.keys())}")
    summary_parts.append(f"- **Patterns Detected**: {len(patterns)}\n")

    # Events by source
    summary_parts.append("## Events by Source\n")
    for source, events in timeline.by_source.items():
        summary_parts.append(f"- **{source}**: {len(events)} events")
    summary_parts.append("")

    # Detected patterns
    if patterns:
        summary_parts.append("## Detected Issues\n")
        for i, pattern in enumerate(patterns, 1):
            severity_emoji = {
                'CRITICAL': '🔴',
                'WARNING': '⚠️',
                'INFO': 'ℹ️'
            }.get(pattern['severity'], '•')

            summary_parts.append(f"{i}. {severity_emoji} **{pattern['pattern'].upper()}** ({pattern['severity']})")
            summary_parts.append(f"   - {pattern['description']}\n")
    else:
        summary_parts.append("## Detected Issues\n")
        summary_parts.append("No significant issues detected.\n")

    # Timeline highlights
    if timeline.events:
        summary_parts.append("## Timeline Highlights\n")
        critical_events = [e for e in timeline.events if e.data.get('severity') == 'CRITICAL']
        warning_events = [e for e in timeline.events if e.data.get('severity') == 'WARNING']

        summary_parts.append(f"- **Critical Events**: {len(critical_events)}")
        summary_parts.append(f"- **Warning Events**: {len(warning_events)}")

    return '\n'.join(summary_parts)
