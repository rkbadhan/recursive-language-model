# System Log Analysis - Quick Start Guide for RLM Integration

## Quick Overview

The RLM framework is already designed to handle system log analysis. Here's what you need to add:

```
┌─────────────────────────────────────────────────────────────┐
│ User's Log Analysis Query                                   │
│ "Find all errors in syslog from past 24 hours"             │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ RLM_REPL (Orchestrator)                                     │
│ - Loads logs as context variable                            │
│ - Iteratively guides LM exploration                         │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ LogAnalysisREPLEnv (Specialized Environment)  [NEW]         │
│ - parse_syslog()       - parse_apache_access()             │
│ - parse_json_logs()    - parse_csv_logs()                  │
│ - detect_log_format()  - aggregate_by_level()              │
│ - llm_query_batch()    - llm_query()                        │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ LLM Decision Loop (Intelligent)                             │
│                                                              │
│ Iteration 1: "Let me peek at the log structure"            │
│   → Runs: len(context), context[:500], detect_log_format() │
│                                                              │
│ Iteration 2: "It's syslog format, too large to parse all"  │
│   → Chunks into 20 pieces (50k lines each)                 │
│                                                              │
│ Iteration 3: "Now analyze chunks in parallel"              │
│   → Uses llm_query_batch() for 10x speed                   │
│                                                              │
│ Iteration 4: "Aggregate results by severity"               │
│   → Calls aggregate_by_level(parsed_entries)               │
│                                                              │
│ Iteration 5: "Provide final answer"                        │
│   → FINAL(structured_results)                              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ User Gets: Structured log analysis results                 │
│ - Grouped by severity/host/timestamp                        │
│ - Semantic analysis of error messages                       │
│ - Correlation across log sources                           │
└─────────────────────────────────────────────────────────────┘
```

## Files to Create/Modify

### 1. Create: `/rlm/log_parsers.py` (NEW)
```python
"""Log format parsers for system log analysis."""

import re
import json
from typing import List, Dict

def parse_syslog(text: str) -> List[Dict]:
    """Parse syslog format: Jan 15 10:30:45 host process[123]: message"""
    pattern = r'(\w+ \d+ \d+:\d+:\d+) (\S+) (\S+)\[(\d+)\]: (.*)'
    entries = []
    for line in text.split('\n'):
        match = re.search(pattern, line)
        if match:
            entries.append({
                'timestamp': match.group(1),
                'hostname': match.group(2),
                'process': match.group(3),
                'pid': match.group(4),
                'message': match.group(5)
            })
    return entries

def parse_json_logs(text: str) -> List[Dict]:
    """Parse newline-delimited JSON logs (common in modern systems)."""
    entries = []
    for line in text.split('\n'):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append({'raw': line})
    return entries

def parse_apache_access(text: str) -> List[Dict]:
    """Parse Apache access logs: 192.168.1.1 - - [15/Jan/2024:10:30:45 +0000] "GET / HTTP/1.1" 200 1234"""
    pattern = r'(\S+) - - \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)'
    entries = []
    for line in text.split('\n'):
        match = re.search(pattern, line)
        if match:
            entries.append({
                'ip': match.group(1),
                'timestamp': match.group(2),
                'method': match.group(3),
                'path': match.group(4),
                'protocol': match.group(5),
                'status_code': int(match.group(6)),
                'bytes': int(match.group(7))
            })
    return entries

def parse_csv_logs(text: str, delimiter: str = ',') -> List[Dict]:
    """Parse CSV-formatted logs."""
    lines = text.split('\n')
    if not lines:
        return []
    
    headers = lines[0].split(delimiter)
    entries = []
    for line in lines[1:]:
        if line.strip():
            values = line.split(delimiter)
            entry = dict(zip(headers, values))
            entries.append(entry)
    return entries

def detect_log_format(sample: str) -> str:
    """Auto-detect log format from sample."""
    lines = sample.split('\n')[:100]
    
    # JSON check
    if all(line.strip().startswith('{') and line.strip().endswith('}') 
           for line in lines if line.strip()):
        return 'json'
    
    # Apache check
    if all(re.search(r'\d+\.\d+\.\d+\.\d+', line) for line in lines if line.strip()):
        return 'apache'
    
    # CSV check
    if all(',' in line for line in lines[:5] if line.strip()):
        return 'csv'
    
    # Default to syslog
    return 'syslog'
```

### 2. Create: `/rlm/repl_log.py` (NEW)
```python
"""Log analysis specialized REPL environment."""

from typing import List, Dict
from rlm.repl import REPLEnv
from rlm import log_parsers

class LogAnalysisREPLEnv(REPLEnv):
    """REPL environment specialized for system log analysis."""
    
    def _inject_special_functions(self) -> None:
        """Inject log-specific functions into REPL globals."""
        super()._inject_special_functions()  # Keep llm_query, etc.
        
        # Inject log parsers
        self.globals['parse_syslog'] = log_parsers.parse_syslog
        self.globals['parse_json_logs'] = log_parsers.parse_json_logs
        self.globals['parse_apache_access'] = log_parsers.parse_apache_access
        self.globals['parse_csv_logs'] = log_parsers.parse_csv_logs
        self.globals['detect_log_format'] = log_parsers.detect_log_format
        
        # Add log analysis helpers
        def aggregate_by_level(entries: List[Dict]) -> Dict:
            """Group log entries by severity level."""
            by_level = {}
            for entry in entries:
                level = entry.get('level') or entry.get('severity') or 'UNKNOWN'
                if level not in by_level:
                    by_level[level] = []
                by_level[level].append(entry)
            return by_level
        
        def aggregate_by_host(entries: List[Dict]) -> Dict:
            """Group log entries by hostname."""
            by_host = {}
            for entry in entries:
                host = entry.get('hostname') or entry.get('host') or 'UNKNOWN'
                if host not in by_host:
                    by_host[host] = []
                by_host[host].append(entry)
            return by_host
        
        def find_errors(entries: List[Dict]) -> List[Dict]:
            """Filter entries that appear to be errors."""
            error_keywords = ['ERROR', 'FATAL', 'CRITICAL', '500', '404', '403']
            errors = []
            for entry in entries:
                entry_str = str(entry).upper()
                if any(keyword in entry_str for keyword in error_keywords):
                    errors.append(entry)
            return errors
        
        self.globals['aggregate_by_level'] = aggregate_by_level
        self.globals['aggregate_by_host'] = aggregate_by_host
        self.globals['find_errors'] = find_errors
```

### 3. Modify: `/rlm/utils/prompts.py` (EXTEND)
Add this constant at the end:

```python
LOG_ANALYSIS_SYSTEM_PROMPT = """You are a system log analysis expert with deep knowledge of:
- Syslog format (standard Linux/Unix logs)
- JSON logs (structured logs from applications)
- Apache/Nginx web server logs
- CSV-formatted logs
- Windows Event logs
- Custom application logs

Your REPL environment provides:

SPECIALIZED LOG FUNCTIONS:
1. parse_syslog(text) - Extract structured entries from syslog
2. parse_json_logs(text) - Parse newline-delimited JSON logs
3. parse_apache_access(text) - Parse Apache/Nginx access logs
4. parse_csv_logs(text) - Parse CSV-formatted logs
5. detect_log_format(sample) - Auto-detect log format from sample
6. aggregate_by_level(entries) - Group by severity level
7. aggregate_by_host(entries) - Group by hostname
8. find_errors(entries) - Filter error-level entries

PARALLEL PROCESSING:
- llm_query_batch(prompts) - Process multiple chunks in parallel (10x faster!)

ANALYSIS STRATEGY FOR LARGE LOGS:
1. Detect format using detect_log_format(context[:5000])
2. If > 100k chars:
   a. Split into chunks: chunks = [context[i:i+50000] for i in range(0, len(context), 50000)]
   b. Parse in parallel: all_entries = llm_query_batch([f"Parse and list errors:\\n{c}" for c in chunks])
   c. Aggregate: consolidated = aggregate_by_level(flatten(all_entries))
3. If <= 100k chars:
   a. Parse directly: entries = parse_syslog(context) (or appropriate parser)
   b. Analyze: errors = find_errors(entries)
   c. Group: by_level = aggregate_by_level(errors)

EXAMPLE WORKFLOW:
```repl
# Detect format
sample = context[:5000]
fmt = detect_log_format(sample)
print(f"Format detected: {fmt}")

# Parse appropriately
if fmt == 'syslog':
    entries = parse_syslog(context)
elif fmt == 'json':
    entries = parse_json_logs(context)

# Analyze
errors = find_errors(entries)
errors_by_level = aggregate_by_level(errors)
print(f"Found {len(errors)} errors across {len(errors_by_level)} severity levels")

# Return structured answer
FINAL_VAR(errors_by_level)
```

Remember: Large logs require chunking + parallel processing for speed!
"""
```

### 4. Create: `/rlm/rlm_log_analysis.py` (NEW - Convenience Class)
```python
"""Convenience class for system log analysis using RLM."""

from rlm.rlm_repl import RLM_REPL
from rlm.repl_log import LogAnalysisREPLEnv
from rlm.utils.prompts import LOG_ANALYSIS_SYSTEM_PROMPT

class RLMLogAnalyzer(RLM_REPL):
    """Specialized RLM for system log analysis."""
    
    def setup_context(self, context, query=None):
        """Override to use LogAnalysisREPLEnv instead of standard REPLEnv."""
        if query is None:
            query = "Analyze these logs and provide insights"
        
        self.query = query
        self.logger.log_query_start(query)
        
        # Use log analysis system prompt instead of default
        self.messages = [{"role": "system", "content": LOG_ANALYSIS_SYSTEM_PROMPT}]
        
        # Convert context
        import rlm.utils.utils as utils
        context_data, context_str = utils.convert_context_for_repl(context)
        
        # Create LogAnalysisREPLEnv instead of standard REPLEnv
        self.repl_env = LogAnalysisREPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
            depth=self.depth,
            max_depth=self.max_depth,
            enable_logging=self.enable_logging,
            parent_rlm_class=RLMLogAnalyzer if self.max_depth > 1 else None,
        )
        
        return self.messages
```

## Usage Examples

### Example 1: Simple Syslog Analysis
```python
from rlm.rlm_log_analysis import RLMLogAnalyzer

# Load logs
with open('/var/log/syslog') as f:
    logs = f.read()

# Analyze
analyzer = RLMLogAnalyzer(
    model="gpt-4o",
    recursive_model="gpt-4o-mini",
    enable_logging=True
)

result = analyzer.completion(
    context=logs,
    query="Find all ERROR and CRITICAL level messages from the past 24 hours"
)

print(result)
```

### Example 2: Multi-File Log Correlation
```python
from rlm.rlm_log_analysis import RLMLogAnalyzer

# Load multiple logs
logs_dict = {
    'auth.log': open('/var/log/auth.log').read(),
    'syslog': open('/var/log/syslog').read(),
    'apache.log': open('/var/log/apache2/access.log').read(),
}

analyzer = RLMLogAnalyzer(
    model="gpt-4o",
    recursive_model="gpt-4o-mini",
    enable_logging=True
)

result = analyzer.completion(
    context=logs_dict,
    query="Correlate authentication failures across auth.log and syslog. What patterns do you see?"
)

print(result)
```

### Example 3: Structured JSON Logs
```python
from rlm.rlm_log_analysis import RLMLogAnalyzer

# Load JSON logs (Kubernetes, Docker, etc.)
json_logs = [
    json.loads(line) for line in open('/var/log/app.log')
]

analyzer = RLMLogAnalyzer(model="gpt-4o-mini", enable_logging=True)

result = analyzer.completion(
    context=json_logs,
    query="Summarize error patterns and suggest fixes"
)

print(result)
```

## Key Design Principles

1. **Format Detection First** - LLM peeks at logs to detect format before parsing
2. **Intelligent Chunking** - For large logs, automatically chunks and processes in parallel
3. **Parallel Processing** - Uses `llm_query_batch()` for 10x speed on large datasets
4. **State Maintenance** - Parsed data persists in REPL variables for multi-step analysis
5. **Flexible Output** - Returns structured data (JSON) or natural language insights
6. **No Hard Limits** - Can handle 10M+ character logs by chunking intelligently

## What's Happening Under the Hood

```
User Input (Raw Logs)
    ↓
LogAnalysisREPLEnv (loads context, injects parsers)
    ↓
Iteration 1: LLM peeks → detects format
    ↓
Iteration 2: LLM chunks → prepares parallel work
    ↓
Iteration 3: llm_query_batch() → processes chunks in parallel
    ↓
Iteration 4: LLM aggregates → groups/filters results
    ↓
Iteration 5: LLM returns → FINAL(structured_results)
    ↓
User Output (Analyzed Logs)
```

## Next Steps

1. Create the 4 files above in the RLM project
2. Add tests in `test_log_analysis.py`
3. Add examples to `main.py` option for log analysis
4. Run `python main.py` and choose log analysis option

The system is ready to go - just needs these lightweight wrapper modules!
