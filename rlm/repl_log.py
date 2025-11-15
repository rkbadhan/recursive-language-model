"""
Log Analysis REPL Environment

Specialized REPL environment for system log analysis that injects
log parsing functions and correlation utilities into the environment.
"""

from typing import Optional, Dict, Any
from rlm.repl import REPLEnv
from rlm.log_parsers import (
    parse_jstack, parse_strace, parse_gc_log, parse_pstack,
    parse_syslog, parse_json_logs, parse_log, detect_log_format,
    extract_timestamp, normalize_timestamp
)
from rlm.log_correlator import (
    correlate_logs, find_correlated_events, detect_all_patterns,
    generate_correlation_summary, Timeline, LogEvent,
    extract_events_from_parsed_log
)


class LogAnalysisREPLEnv(REPLEnv):
    """
    Specialized REPL environment for log analysis.

    Extends the base REPLEnv to inject log parsing and correlation functions
    that the LLM can use for system log analysis.
    """

    def _inject_special_functions(self) -> None:
        """
        Inject log analysis functions into REPL globals.

        This is called during REPL initialization to make log parsing
        and correlation tools available to the LLM.
        """
        # Call parent to inject base functions (llm_query, etc.)
        super()._inject_special_functions()

        # Inject log parser functions
        self.globals['parse_jstack'] = parse_jstack
        self.globals['parse_strace'] = parse_strace
        self.globals['parse_gc_log'] = parse_gc_log
        self.globals['parse_pstack'] = parse_pstack
        self.globals['parse_syslog'] = parse_syslog
        self.globals['parse_json_logs'] = parse_json_logs
        self.globals['parse_log'] = parse_log
        self.globals['detect_log_format'] = detect_log_format

        # Inject timestamp utilities
        self.globals['extract_timestamp'] = extract_timestamp
        self.globals['normalize_timestamp'] = normalize_timestamp

        # Inject correlation functions
        self.globals['correlate_logs'] = correlate_logs
        self.globals['find_correlated_events'] = find_correlated_events
        self.globals['detect_all_patterns'] = detect_all_patterns
        self.globals['generate_correlation_summary'] = generate_correlation_summary

        # Inject data structures
        self.globals['Timeline'] = Timeline
        self.globals['LogEvent'] = LogEvent
        self.globals['extract_events_from_parsed_log'] = extract_events_from_parsed_log

        # Add helper documentation as a string (accessible via help_log_analysis)
        self.globals['help_log_analysis'] = self._get_log_analysis_help()

    def _get_log_analysis_help(self) -> str:
        """
        Return help text for log analysis functions.
        """
        return """
# Log Analysis Functions Available in REPL

## Parser Functions
- parse_log(content, format_hint=None) - Universal parser with auto-detection
- parse_jstack(content) - Parse Java thread dumps
- parse_strace(content) - Parse system call traces
- parse_gc_log(content) - Parse JVM GC logs
- parse_pstack(content) - Parse native stack traces
- parse_syslog(content) - Parse syslog format
- parse_json_logs(content) - Parse JSON logs
- detect_log_format(content) - Auto-detect log format

## Correlation Functions
- correlate_logs(parsed_logs_dict) - Build timeline from multiple logs
- find_correlated_events(timeline, source_a, source_b, max_delta_seconds=5.0)
- detect_all_patterns(timeline) - Detect known issue patterns
- generate_correlation_summary(timeline, patterns) - Generate summary

## Utility Functions
- extract_timestamp(line) - Extract timestamp from log line
- normalize_timestamp(ts_str) - Convert to Unix epoch
- extract_events_from_parsed_log(parsed, source_name) - Extract events

## Example Workflow
```python
# 1. Detect and parse logs
format = detect_log_format(context)
parsed = parse_log(context, format)

# 2. For multiple logs, correlate them
logs = {
    'jstack': parse_jstack(jstack_content),
    'gc': parse_gc_log(gc_content)
}
timeline = correlate_logs(logs)

# 3. Detect patterns
patterns = detect_all_patterns(timeline)

# 4. Generate summary
summary = generate_correlation_summary(timeline, patterns)
print(summary)
```

Use print(help_log_analysis) to see this help again.
"""
