"""
Log Analysis Use Case for RLM

This module demonstrates how to use RLM for system log analysis.
It provides parsers, correlators, and a specialized RLM analyzer for logs.

This is a USE CASE of RLM, not part of the core RLM framework.
"""

from .log_parsers import (
    parse_log,
    detect_log_format,
    parse_jstack,
    parse_strace,
    parse_gc_log,
    parse_pstack,
    parse_syslog,
    parse_json_logs,
)

from .log_correlator import (
    LogEvent,
    Timeline,
    correlate_logs,
    detect_all_patterns,
    detect_gc_caused_blocking,
    detect_thread_blocking_pattern,
    detect_memory_pressure_pattern,
    detect_deadlock_pattern,
)

# Lazy import for RLMLogAnalyzer (requires openai package)
def __getattr__(name):
    if name == "RLMLogAnalyzer":
        from .rlm_log_analysis import RLMLogAnalyzer
        return RLMLogAnalyzer
    elif name == "analyze_logs":
        from .rlm_log_analysis import analyze_logs
        return analyze_logs
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Parsers
    'parse_log',
    'detect_log_format',
    'parse_jstack',
    'parse_strace',
    'parse_gc_log',
    'parse_pstack',
    'parse_syslog',
    'parse_json_logs',
    # Correlators
    'LogEvent',
    'Timeline',
    'correlate_logs',
    'detect_all_patterns',
    'detect_gc_caused_blocking',
    'detect_thread_blocking_pattern',
    'detect_memory_pressure_pattern',
    'detect_deadlock_pattern',
    # RLM Analyzer (lazy loaded)
    'RLMLogAnalyzer',
    'analyze_logs',
]
