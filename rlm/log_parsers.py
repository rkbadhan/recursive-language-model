"""
System Log Parsers for RLM

Specialized parsers for various system log formats:
- jstack: Java thread dumps
- strace: System call traces
- GC logs: JVM garbage collection
- pstack/pmstack: Native stack traces
- syslog: Standard system logs
- Apache/Nginx access logs
- JSON structured logs
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict


# ============================================================================
# Timestamp Utilities
# ============================================================================

def extract_timestamp(line: str) -> Optional[str]:
    """
    Extract timestamp from log line (supports multiple formats).

    Returns ISO format timestamp string or None.
    """
    patterns = [
        # ISO 8601: 2024-11-15T14:30:15.123Z
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)',
        # Common syslog: Nov 15 14:30:15
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
        # Date with time: 2024-11-15 14:30:15.123
        r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)',
        # GC timestamp: 2024-11-15T14:30:15.123+0000
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}[+-]\d{4})',
        # Epoch with decimals: 1699876215.123
        r'(\d{10}\.\d+)',
        # Epoch seconds: 1699876215
        r'(\d{10})',
    ]

    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            return match.group(1)

    return None


def normalize_timestamp(ts: str) -> Optional[float]:
    """
    Convert timestamp string to Unix epoch (float).

    Returns epoch timestamp or None if parsing fails.
    """
    try:
        # Try epoch formats first
        if re.match(r'^\d{10}(?:\.\d+)?$', ts):
            return float(ts)

        # Try ISO 8601
        if 'T' in ts:
            # Remove timezone suffix for simplicity
            ts_clean = re.sub(r'[+-]\d{2}:?\d{2}$|Z$', '', ts)
            dt = datetime.fromisoformat(ts_clean)
            return dt.timestamp()

        # Try standard format
        if '-' in ts and ':' in ts:
            dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
            return dt.timestamp()

    except (ValueError, AttributeError):
        pass

    return None


# ============================================================================
# jstack Parser (Java Thread Dumps)
# ============================================================================

def parse_jstack(content: str) -> Dict[str, Any]:
    """
    Parse Java thread dump (jstack output).

    Returns dict with:
    - threads: List of thread info dicts
    - deadlocks: List of detected deadlocks
    - blocked_threads: Threads in BLOCKED state
    - waiting_threads: Threads in WAITING/TIMED_WAITING
    - lock_info: Lock acquisition chains
    """
    lines = content.split('\n')

    threads = []
    current_thread = None
    deadlocks = []
    in_deadlock_section = False

    for line in lines:
        # Thread header: "Thread-1" #123 prio=5 os_prio=0 tid=0x... nid=0x... waiting on condition
        thread_match = re.match(r'"([^"]+)".*tid=(0x[0-9a-f]+).*nid=(0x[0-9a-f]+)\s+(.+)$', line)
        if thread_match:
            if current_thread:
                threads.append(current_thread)

            current_thread = {
                'name': thread_match.group(1),
                'tid': thread_match.group(2),
                'nid': thread_match.group(3),
                'state_desc': thread_match.group(4),
                'state': None,
                'stack': [],
                'locks_held': [],
                'locks_waiting': []
            }
            continue

        # Thread state: java.lang.Thread.State: BLOCKED (on object monitor)
        state_match = re.search(r'java\.lang\.Thread\.State:\s+(\w+)', line)
        if state_match and current_thread:
            current_thread['state'] = state_match.group(1)
            continue

        # Stack trace line
        if line.strip().startswith('at ') and current_thread:
            current_thread['stack'].append(line.strip())
            continue

        # Lock info: - waiting to lock <0x00000000e1234560> (a java.lang.Object)
        lock_waiting_match = re.search(r'- waiting (?:to lock|on) <(0x[0-9a-f]+)>\s+\(a (.+)\)', line)
        if lock_waiting_match and current_thread:
            current_thread['locks_waiting'].append({
                'address': lock_waiting_match.group(1),
                'class': lock_waiting_match.group(2)
            })
            continue

        # Lock held: - locked <0x00000000e1234560> (a java.lang.Object)
        lock_held_match = re.search(r'- locked <(0x[0-9a-f]+)>\s+\(a (.+)\)', line)
        if lock_held_match and current_thread:
            current_thread['locks_held'].append({
                'address': lock_held_match.group(1),
                'class': lock_held_match.group(2)
            })
            continue

        # Deadlock detection section
        if 'Found one Java-level deadlock' in line or 'Found Java-level deadlocks' in line:
            in_deadlock_section = True
            continue

        if in_deadlock_section and line.strip():
            deadlocks.append(line.strip())

    # Add last thread
    if current_thread:
        threads.append(current_thread)

    # Categorize threads by state
    blocked_threads = [t for t in threads if t['state'] == 'BLOCKED']
    waiting_threads = [t for t in threads if t['state'] in ('WAITING', 'TIMED_WAITING')]

    # Build lock dependency graph
    lock_holders = {}  # lock_address -> thread_name
    for thread in threads:
        for lock in thread['locks_held']:
            lock_holders[lock['address']] = thread['name']

    lock_chains = []
    for thread in blocked_threads:
        for lock in thread['locks_waiting']:
            holder = lock_holders.get(lock['address'])
            if holder:
                lock_chains.append({
                    'waiting_thread': thread['name'],
                    'lock': lock['address'],
                    'held_by': holder
                })

    return {
        'total_threads': len(threads),
        'threads': threads,
        'deadlocks': deadlocks,
        'blocked_threads': blocked_threads,
        'waiting_threads': waiting_threads,
        'lock_chains': lock_chains,
        'has_deadlock': len(deadlocks) > 0
    }


# ============================================================================
# strace Parser (System Call Traces)
# ============================================================================

def parse_strace(content: str) -> Dict[str, Any]:
    """
    Parse strace output.

    Returns dict with:
    - syscalls: List of syscall dicts
    - slow_calls: Calls taking > 1 second
    - errors: Failed syscalls
    - io_operations: File I/O operations
    - timeline: Time-ordered events
    """
    lines = content.split('\n')

    syscalls = []
    slow_calls = []
    errors = []
    io_operations = []

    for line in lines:
        if not line.strip():
            continue

        # Parse syscall: 14:30:15.123456 open("/etc/passwd", O_RDONLY) = 3 <0.000012>
        # Or: open("/etc/passwd", O_RDONLY) = 3 <0.000012>
        syscall_match = re.search(
            r'(?:(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+)?'  # Optional timestamp
            r'(\w+)\((.*?)\)\s*=\s*(-?\d+|0x[0-9a-f]+|\?)'  # syscall(args) = retval
            r'(?:\s+([A-Z_]+)\s+\([^)]+\))?'  # Optional error (ENOENT, etc.)
            r'(?:\s+<([\d.]+)>)?',  # Optional duration
            line
        )

        if syscall_match:
            timestamp = syscall_match.group(1)
            syscall_name = syscall_match.group(2)
            args = syscall_match.group(3)
            retval = syscall_match.group(4)
            error = syscall_match.group(5)
            duration = syscall_match.group(6)

            syscall_info = {
                'timestamp': timestamp,
                'name': syscall_name,
                'args': args,
                'retval': retval,
                'error': error,
                'duration': float(duration) if duration else None
            }

            syscalls.append(syscall_info)

            # Track slow calls (> 1 second)
            if duration and float(duration) > 1.0:
                slow_calls.append(syscall_info)

            # Track errors (negative return value or explicit error)
            if error or (retval.startswith('-') and retval != '-1'):
                errors.append(syscall_info)

            # Track I/O operations
            if syscall_name in ('read', 'write', 'open', 'close', 'lseek', 'pread', 'pwrite'):
                io_operations.append(syscall_info)

    # Calculate statistics
    total_duration = sum(s['duration'] for s in syscalls if s['duration'])
    avg_duration = total_duration / len(syscalls) if syscalls else 0

    # Group by syscall name
    syscall_counts = defaultdict(int)
    for sc in syscalls:
        syscall_counts[sc['name']] += 1

    return {
        'total_syscalls': len(syscalls),
        'syscalls': syscalls,
        'slow_calls': slow_calls,
        'errors': errors,
        'io_operations': io_operations,
        'total_duration': total_duration,
        'avg_duration': avg_duration,
        'syscall_distribution': dict(syscall_counts),
        'most_common': sorted(syscall_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    }


# ============================================================================
# GC Log Parser (JVM Garbage Collection)
# ============================================================================

def parse_gc_log(content: str) -> Dict[str, Any]:
    """
    Parse JVM GC logs (supports multiple formats).

    Returns dict with:
    - gc_events: List of GC event dicts
    - pause_times: Pause duration statistics
    - heap_usage: Heap utilization over time
    - collections: Minor vs Major collection counts
    """
    lines = content.split('\n')

    gc_events = []
    minor_collections = 0
    major_collections = 0

    for line in lines:
        if not line.strip():
            continue

        # Parse unified logging format (JDK 9+):
        # [2024-11-15T14:30:15.123+0000][gc] GC(123) Pause Young (Normal) 50M->10M(100M) 5.123ms
        unified_match = re.search(
            r'\[([^\]]+)\]\[gc(?:,\w+)?\]\s+(?:GC\(\d+\)\s+)?'
            r'(Pause\s+\w+.*?|[\w\s]+)'
            r'(?:(\d+[KMG])->(\d+[KMG])\((\d+[KMG])\))?'
            r'\s+([\d.]+)ms',
            line
        )

        if unified_match:
            timestamp = unified_match.group(1)
            gc_type = unified_match.group(2).strip()
            heap_before = unified_match.group(3)
            heap_after = unified_match.group(4)
            heap_total = unified_match.group(5)
            pause_time = float(unified_match.group(6))

            event = {
                'timestamp': timestamp,
                'type': gc_type,
                'heap_before': heap_before,
                'heap_after': heap_after,
                'heap_total': heap_total,
                'pause_ms': pause_time,
                'is_young': 'Young' in gc_type or 'Minor' in gc_type,
                'is_full': 'Full' in gc_type or 'Major' in gc_type
            }

            gc_events.append(event)

            if event['is_young']:
                minor_collections += 1
            if event['is_full']:
                major_collections += 1

            continue

        # Parse old format (JDK 8):
        # 2024-11-15T14:30:15.123+0000: [GC (Allocation Failure) 50M->10M(100M), 0.0051234 secs]
        old_match = re.search(
            r'([^:]+):\s+\[(\w+)\s*(?:\([^)]+\))?\s+'
            r'(?:(\d+[KMG])->(\d+[KMG])\((\d+[KMG])\))?,?\s+'
            r'([\d.]+)\s+secs\]',
            line
        )

        if old_match:
            timestamp = old_match.group(1)
            gc_type = old_match.group(2)
            heap_before = old_match.group(3)
            heap_after = old_match.group(4)
            heap_total = old_match.group(5)
            pause_time = float(old_match.group(6)) * 1000  # Convert to ms

            event = {
                'timestamp': timestamp,
                'type': gc_type,
                'heap_before': heap_before,
                'heap_after': heap_after,
                'heap_total': heap_total,
                'pause_ms': pause_time,
                'is_young': gc_type == 'GC' or 'Young' in gc_type,
                'is_full': gc_type == 'Full GC' or 'Full' in gc_type
            }

            gc_events.append(event)

            if event['is_young']:
                minor_collections += 1
            if event['is_full']:
                major_collections += 1

    # Calculate statistics
    if gc_events:
        pause_times = [e['pause_ms'] for e in gc_events]
        max_pause = max(pause_times)
        avg_pause = sum(pause_times) / len(pause_times)
        total_pause = sum(pause_times)
    else:
        max_pause = avg_pause = total_pause = 0

    return {
        'total_collections': len(gc_events),
        'minor_collections': minor_collections,
        'major_collections': major_collections,
        'gc_events': gc_events,
        'max_pause_ms': max_pause,
        'avg_pause_ms': avg_pause,
        'total_pause_ms': total_pause,
        'long_pauses': [e for e in gc_events if e['pause_ms'] > 1000]  # > 1 second
    }


# ============================================================================
# pstack/pmstack Parser (Native Stack Traces)
# ============================================================================

def parse_pstack(content: str) -> Dict[str, Any]:
    """
    Parse pstack/pmstack output (native stack traces).

    Returns dict with:
    - threads: List of native thread stacks
    - processes: Process information
    - symbols: Unique symbols seen
    """
    lines = content.split('\n')

    threads = []
    current_thread = None
    symbols = set()

    for line in lines:
        # Thread header: Thread 1 (LWP 12345):
        thread_match = re.match(r'Thread\s+(\d+)\s+\((?:LWP|Thread)\s+(\d+)\)', line)
        if thread_match:
            if current_thread:
                threads.append(current_thread)

            current_thread = {
                'thread_id': thread_match.group(1),
                'lwp': thread_match.group(2),
                'stack': []
            }
            continue

        # Stack frame: #0  0x00007f8a4c001234 in pthread_cond_wait () from /lib64/libpthread.so.0
        frame_match = re.match(r'#(\d+)\s+(0x[0-9a-f]+)\s+in\s+([^\s]+)\s+(?:from\s+(.+))?', line)
        if frame_match and current_thread:
            frame = {
                'frame_num': int(frame_match.group(1)),
                'address': frame_match.group(2),
                'symbol': frame_match.group(3),
                'library': frame_match.group(4)
            }
            current_thread['stack'].append(frame)
            symbols.add(frame_match.group(3))
            continue

    # Add last thread
    if current_thread:
        threads.append(current_thread)

    return {
        'total_threads': len(threads),
        'threads': threads,
        'unique_symbols': list(symbols),
        'symbol_count': len(symbols)
    }


# ============================================================================
# Syslog Parser
# ============================================================================

def parse_syslog(content: str) -> Dict[str, Any]:
    """
    Parse standard syslog format.

    Returns dict with:
    - entries: List of log entry dicts
    - by_level: Grouped by severity level
    - by_process: Grouped by process name
    """
    lines = content.split('\n')
    entries = []

    for line in lines:
        if not line.strip():
            continue

        # Parse syslog line: Nov 15 14:30:15 hostname process[pid]: message
        match = re.match(
            r'(\w+\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+'  # timestamp
            r'(\S+)\s+'  # hostname
            r'([^\[:\s]+)(?:\[(\d+)\])?:\s*'  # process[pid]
            r'(.+)$',  # message
            line
        )

        if match:
            timestamp = match.group(1)
            hostname = match.group(2)
            process = match.group(3)
            pid = match.group(4)
            message = match.group(5)

            # Detect log level from message
            level = 'INFO'
            if re.search(r'\b(ERROR|FATAL|CRITICAL)\b', message, re.IGNORECASE):
                level = 'ERROR'
            elif re.search(r'\bWARN(?:ING)?\b', message, re.IGNORECASE):
                level = 'WARNING'
            elif re.search(r'\bDEBUG\b', message, re.IGNORECASE):
                level = 'DEBUG'

            entries.append({
                'timestamp': timestamp,
                'hostname': hostname,
                'process': process,
                'pid': pid,
                'level': level,
                'message': message
            })

    # Group by level
    by_level = defaultdict(list)
    for entry in entries:
        by_level[entry['level']].append(entry)

    # Group by process
    by_process = defaultdict(list)
    for entry in entries:
        by_process[entry['process']].append(entry)

    return {
        'total_entries': len(entries),
        'entries': entries,
        'by_level': dict(by_level),
        'by_process': dict(by_process),
        'error_count': len(by_level.get('ERROR', [])),
        'warning_count': len(by_level.get('WARNING', []))
    }


# ============================================================================
# JSON Log Parser
# ============================================================================

def parse_json_logs(content: str) -> Dict[str, Any]:
    """
    Parse JSON-formatted logs (one JSON object per line).

    Returns dict with:
    - entries: List of parsed JSON objects
    - schema: Detected field names
    - by_level: Grouped by level field (if present)
    """
    lines = content.split('\n')
    entries = []
    all_keys = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            entries.append(obj)
            all_keys.update(obj.keys())
        except json.JSONDecodeError:
            # Skip malformed lines
            continue

    # Group by level if present
    by_level = defaultdict(list)
    level_keys = {'level', 'severity', 'loglevel', 'log_level'}
    level_key = None

    for key in level_keys:
        if key in all_keys:
            level_key = key
            break

    if level_key:
        for entry in entries:
            level = entry.get(level_key, 'UNKNOWN')
            by_level[str(level)].append(entry)

    return {
        'total_entries': len(entries),
        'entries': entries,
        'schema': list(all_keys),
        'by_level': dict(by_level) if level_key else {},
        'level_key': level_key
    }


# ============================================================================
# Log Format Detection
# ============================================================================

def detect_log_format(content: str) -> str:
    """
    Auto-detect log format by analyzing content.

    Returns one of: 'jstack', 'strace', 'gc', 'pstack', 'syslog', 'json', 'unknown'
    """
    lines = [l.strip() for l in content.split('\n')[:50] if l.strip()]  # Sample first 50 lines

    # Check for jstack
    if any('java.lang.Thread.State' in l for l in lines):
        return 'jstack'
    if any(re.search(r'"[^"]+" #\d+ prio=', l) for l in lines):
        return 'jstack'

    # Check for strace
    if any(re.search(r'\w+\([^)]*\)\s*=\s*-?\d+', l) for l in lines):
        if any(re.search(r'<[\d.]+>', l) for l in lines):  # Duration markers
            return 'strace'

    # Check for GC logs
    if any('[gc' in l.lower() for l in lines):
        return 'gc'
    if any(re.search(r'\[GC\s+\(', l) for l in lines):
        return 'gc'
    if any('Pause Young' in l or 'Pause Full' in l for l in lines):
        return 'gc'

    # Check for pstack
    if any(re.match(r'Thread\s+\d+\s+\(LWP', l) for l in lines):
        return 'pstack'
    if any(re.match(r'#\d+\s+0x[0-9a-f]+\s+in', l) for l in lines):
        return 'pstack'

    # Check for JSON (majority of lines are valid JSON)
    json_count = 0
    for line in lines[:20]:
        try:
            json.loads(line)
            json_count += 1
        except:
            pass
    if json_count > len(lines) * 0.7:  # 70% JSON lines
        return 'json'

    # Check for syslog
    if any(re.match(r'\w+\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+\S+', l) for l in lines):
        return 'syslog'

    return 'unknown'


# ============================================================================
# Universal Parser Dispatcher
# ============================================================================

def parse_log(content: str, format_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Universal log parser that auto-detects format and parses accordingly.

    Args:
        content: Log content as string
        format_hint: Optional format hint ('jstack', 'strace', 'gc', etc.)

    Returns:
        Parsed log data dict with 'format' and format-specific fields
    """
    if format_hint is None:
        format_hint = detect_log_format(content)

    parsers = {
        'jstack': parse_jstack,
        'strace': parse_strace,
        'gc': parse_gc_log,
        'pstack': parse_pstack,
        'pmstack': parse_pstack,  # Same as pstack
        'syslog': parse_syslog,
        'json': parse_json_logs
    }

    parser = parsers.get(format_hint)
    if parser:
        result = parser(content)
        result['format'] = format_hint
        return result
    else:
        return {
            'format': 'unknown',
            'content': content,
            'line_count': len(content.split('\n'))
        }
