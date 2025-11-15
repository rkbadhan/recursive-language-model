# Recursive Language Model (RLM) - Architecture Overview for System Log Analysis

## Executive Summary

The RLM framework is a sophisticated system for processing unbounded context through a **programmable REPL interface** rather than direct prompting. It enables the language model to act as a program orchestrator that explores large datasets through Python code execution and recursive sub-LLM calls.

**Key insight for log analysis:** The system's strength is letting the LLM decide how to break down complex problems. For logs, this means the LLM can decide whether to grep, chunk, filter, parse, or aggregate based on what it discovers about the log structure.

---

## 1. Main Entry Points and Input Processing Flow

### Primary Interface
```
User Code
    ↓
rlm = RLM_REPL(model="gpt-4o", recursive_model="gpt-4o-mini")
    ↓
answer = rlm.completion(context=logs, query="Find all errors")
```

### Execution Flow

**File:** `/home/user/recursive-language-model/rlm/rlm_repl.py`

```
RLM_REPL.completion(context, query)
    ↓
1. setup_context()
   └─ Normalize context format (str/dict/list)
   └─ create REPLEnv with loaded context
   └─ Build system prompt
    ↓
2. Main Loop (max_iterations times):
   ├─ Query root LM for next action
   ├─ Extract ```repl code blocks
   ├─ Execute in REPL environment
   ├─ Add results to message history
   ├─ Check for FINAL/FINAL_VAR patterns
   └─ Return if final answer found
    ↓
3. return final_answer
```

### Context Normalization

**File:** `/home/user/recursive-language-model/rlm/utils/utils.py` → `convert_context_for_repl()`

Accepts multiple formats:
- **String**: `"Large log text..."` → Saved to temp file, loaded as `context` variable
- **Dict**: `{"logs": [...], "metadata": {...}}` → Converted to JSON, accessible as `context`
- **List of dicts**: `[{msg}, {msg}, ...]` → Treated as structured data
- **List of strings**: `["line1", "line2", ...]` → Converted to structured data

Each format loads into the REPL as a `context` variable that the LLM can manipulate.

---

## 2. Current Parsing and Analysis Capabilities

### Text Extraction & Pattern Matching

**Regex-based parsing in utils:**

1. **Code Block Extraction** (utils.find_code_blocks)
   ```python
   Pattern: r'```repl\s*\n(.*?)\n```'
   Returns: List[str] of code blocks from LLM response
   ```

2. **Final Answer Detection** (utils.find_final_answer)
   ```python
   Patterns: 
   - FINAL_VAR(variable_name)
   - FINAL(answer_text)
   ```

### Python-Powered Analysis in REPL

The system leverages **full Python capabilities** within the REPL environment:

**Available builtins** (REPLEnv._create_safe_globals):
- String operations: `str`, `len`, `format`, regex via `import re`
- Collections: `list`, `dict`, `set`, iteration functions
- File I/O: `open` (safe, within temp directory)
- Data processing: `json`, manual parsing

**Example parsing patterns the LLM can use:**

```python
# Pattern 1: Regex extraction
import re
lines = context.split('\n')
errors = [line for line in lines if re.search(r'ERROR|FATAL', line)]

# Pattern 2: String manipulation
sections = context.split('---')
for section in sections:
    timestamp = section.split('|')[0]
    level = section.split('|')[1]

# Pattern 3: JSON parsing
import json
log_entries = [json.loads(line) for line in context.split('\n') if line.strip()]

# Pattern 4: File operations
with open('temp_logs.csv', 'w') as f:
    f.write(processed_logs)
```

### Analysis Strategies (Self-Discovered by LLM)

The system documentation mentions emergent strategies:
1. **Peeking**: Inspect structure first (`type()`, `len()`, slicing)
2. **Grepping**: Pattern search with regex
3. **Chunking**: Split large context, process each chunk
4. **Summarization**: Process sections separately, aggregate

For logs, the LLM typically discovers:
- Log format detection (syslog, JSON, CSV, custom)
- Timestamp parsing
- Level/severity extraction
- Message filtering
- Event aggregation

---

## 3. How the Model Processes Different Content Types

### Type-Aware Processing

The system's `convert_context_for_repl()` determines how to present data:

```
String Input:
  "2024-01-15 ERROR database connection failed"
  → Save to temp file → Load as context string
  → LLM sees: context = "2024-01-15 ERROR database..."
  
Dict Input:
  {"logs": [...], "hosts": ["server1", "server2"]}
  → Save to JSON file → Load as context dict
  → LLM sees: context = {"logs": [...], "hosts": [...]}
  
List Input:
  [{"timestamp": "...", "level": "ERROR", "msg": "..."}]
  → Treated as structured data
  → LLM sees: context = [{...}, {...}, ...]
```

### Content Processing Capabilities

**File:** `/home/user/recursive-language-model/rlm/repl.py` → `REPLEnv.load_context()`

1. **Large String Handling**
   - Loaded into REPL workspace
   - Slicing: `context[:1000000]`
   - Splitting: `context.split('\n')`, `context.split('---')`
   - Regex search: `re.findall()`, `re.search()`

2. **Structured Data Handling**
   - JSON loading: `json.load()`, `json.loads()`
   - Dictionary access: `context['key']`, `context.get()`
   - List operations: `len(context)`, filtering, mapping

3. **Interactive Transformation**
   - Variables persist across REPL executions
   - Can build buffers: `results = []` → append → aggregate
   - Can write temp files: Absolute paths in `self.temp_dir`

### Iterative Refinement

The system maintains **message history** across iterations:

```
Iteration 0: LLM → "Let me peek at structure"
             Code: print(len(context))
             Result: "5000000 characters"
             
Iteration 1: LLM → "It's too large, let me chunk it"
             Code: chunks = [context[i:i+50000] for i in range(...)]
             Result: "Created 100 chunks"
             
Iteration 2: LLM → "Now query each chunk in parallel"
             Code: results = llm_query_batch([f"Extract errors: {c}" for c in chunks])
             Result: [error_list_1, error_list_2, ...]
             
Iteration 3: LLM → "Aggregate results"
             Code: final = llm_query("Summarize: " + str(results))
             Result: final_answer
```

---

## 4. Extension Points for New Log Format Parsers

### Architecture for Custom Parsers

The RLM system provides several extension points perfect for log analysis:

#### A. Custom REPL Functions (Recommended)

**File to modify:** `/home/user/recursive-language-model/rlm/repl.py` → `_inject_special_functions()`

**Current pattern:**
```python
def _inject_special_functions(self) -> None:
    def llm_query(prompt: str) -> str:
        return self.sub_rlm.completion(prompt)
    
    def llm_query_batch(prompts: List[str]) -> List[str]:
        # Parallel queries
        
    self.globals['llm_query'] = llm_query
    self.globals['llm_query_batch'] = llm_query_batch
```

**Extension for log parsers:**
```python
# Add new functions before: self.globals['llm_query'] = llm_query

def parse_syslog(text: str) -> List[Dict]:
    """Parse syslog format: timestamp hostname process[pid]: message"""
    import re
    pattern = r'(\w+ \d+ \d+:\d+:\d+) (\w+) (\w+)\[(\d+)\]: (.*)'
    matches = re.findall(pattern, text)
    return [
        {
            "timestamp": m[0],
            "hostname": m[1],
            "process": m[2],
            "pid": m[3],
            "message": m[4]
        }
        for m in matches
    ]

def parse_json_logs(text: str) -> List[Dict]:
    """Parse newline-delimited JSON logs"""
    import json
    lines = text.split('\n')
    return [json.loads(line) for line in lines if line.strip()]

# Inject into globals
self.globals['parse_syslog'] = parse_syslog
self.globals['parse_json_logs'] = parse_json_logs
```

Then the LLM can use:
```python
entries = parse_syslog(context)
error_entries = [e for e in entries if 'ERROR' in e['message']]
```

#### B. Subclass REPLEnv for Specialized Environments

**File:** Create `/home/user/recursive-language-model/rlm/repl_log.py`

```python
from rlm.repl import REPLEnv

class LogAnalysisREPLEnv(REPLEnv):
    """REPL environment specialized for log analysis"""
    
    def _inject_special_functions(self) -> None:
        super()._inject_special_functions()  # Keep all standard functions
        
        # Add log-specific parsers
        def parse_apache_access(text: str) -> List[Dict]:
            # Implement Apache access log parser
            pass
        
        def parse_nginx_access(text: str) -> List[Dict]:
            # Implement Nginx access log parser
            pass
        
        def aggregate_by_level(entries: List[Dict]) -> Dict:
            # Aggregate by severity level
            by_level = {}
            for entry in entries:
                level = entry.get('level', 'UNKNOWN')
                if level not in by_level:
                    by_level[level] = []
                by_level[level].append(entry)
            return by_level
        
        self.globals['parse_apache_access'] = parse_apache_access
        self.globals['parse_nginx_access'] = parse_nginx_access
        self.globals['aggregate_by_level'] = aggregate_by_level
```

Then use in RLM_REPL:
```python
rlm = RLM_REPL(...)
rlm.repl_env = LogAnalysisREPLEnv(...)
```

#### C. Extend with Smart Detection

**File:** Create `/home/user/recursive-language-model/rlm/log_format_detector.py`

```python
class LogFormatDetector:
    """Auto-detect log format from samples"""
    
    PATTERNS = {
        'syslog': r'\w+ \d+ \d+:\d+:\d+ \w+ \w+\[\d+\]:',
        'apache': r'\d+\.\d+\.\d+\.\d+ .* "GET|POST',
        'json': r'^\{.*\}$',
        'csv': r'^[^,]+,[^,]+,[^,]+',
    }
    
    @staticmethod
    def detect_format(sample: str, num_lines: int = 100) -> str:
        """Detect log format from sample"""
        lines = sample.split('\n')[:num_lines]
        for format_name, pattern in LogFormatDetector.PATTERNS.items():
            if all(re.search(pattern, line) for line in lines):
                return format_name
        return 'unknown'
    
    @staticmethod
    def get_parser(format_name: str):
        """Get appropriate parser function"""
        parsers = {
            'syslog': parse_syslog,
            'apache': parse_apache_access,
            'json': parse_json_logs,
            'csv': parse_csv_logs,
        }
        return parsers.get(format_name, None)
```

Inject into system prompt to guide LLM:
```python
system_message = """
Available log parsers in REPL:
- parse_syslog(text): For syslog format
- parse_apache_access(text): For Apache logs
- parse_json_logs(text): For JSON logs
- parse_csv_logs(text): For CSV format
- detect_log_format(text): Auto-detect format

Use format detection first if unsure about log type.
"""
```

#### D. Custom Prompts for Log Analysis

**File:** `/home/user/recursive-language-model/rlm/utils/prompts.py` → Add new template

```python
LOG_ANALYSIS_SYSTEM_PROMPT = """
You are a system log analysis expert. Your REPL environment has:

1. context: The raw logs to analyze
2. Special log parsing functions:
   - parse_syslog(text): Extract structured syslog entries
   - parse_apache_access(text): Parse Apache access logs
   - parse_json_logs(text): Load JSON log streams
   - detect_log_format(sample): Identify log format
   - aggregate_by_level(entries): Group by severity
   - find_anomalies(entries): Detect unusual patterns

Analysis Strategy:
1. First, determine log format (type, size, structure)
2. Parse into structured data
3. Filter/aggregate as needed
4. Use llm_query_batch() for semantic analysis on large datasets
5. Return findings

Example approach for 1M lines:
- Detect format
- Split into 20 chunks of 50k lines each
- Parse each chunk in parallel with llm_query_batch()
- Aggregate results
"""
```

---

## 5. Existing File I/O and Multi-File Processing

### File I/O Infrastructure

**File:** `/home/user/recursive-language-model/rlm/repl.py` → `REPLEnv`

**Temporary Directory Management:**
```python
class REPLEnv:
    def __init__(self, ...):
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")  # Isolated temp dir
        self.original_cwd = os.getcwd()
```

**File Operations Available:**
1. **Read/Write files** (within temp_dir):
   ```python
   with open('logs_parsed.json', 'w') as f:
       json.dump(parsed_entries, f)
   
   with open('logs_parsed.json', 'r') as f:
       data = json.load(f)
   ```

2. **JSON serialization** (built-in):
   ```python
   results = [{"line": 1, "error": True}, ...]
   json_str = json.dumps(results, indent=2)
   ```

3. **File listing** (via Python):
   ```python
   import os
   files = os.listdir('.')  # Lists temp_dir contents
   ```

### Multi-File Processing Patterns

**Current capability (from main.py):**

Example 2 demonstrates multi-document reasoning:
```python
documents = [
    {"title": "...", "content": "..."},
    {"title": "...", "content": "..."},
    # 100 documents repeated
]

# Context: 100 documents, handled via:
# 1. String concatenation
# 2. LLM chunking in REPL
# 3. Individual sub-LLM queries per document
# 4. Aggregation of results
```

### Parallel Processing for Multiple Files

**Advanced capability (v2.0 feature):**

**File:** `/home/user/recursive-language-model/rlm/repl.py` → `llm_query_batch()`

```python
# In REPL environment:
log_files = ["auth.log", "syslog", "app.log"]
chunks = []
for log_file in log_files:
    with open(log_file) as f:
        content = f.read()
    chunks.append(content)

# Process all files in parallel!
prompts = [f"Extract errors from this log:\n{chunk}" for chunk in chunks]
results = llm_query_batch(prompts)  # All at once - 10x faster!

# Aggregate
all_errors = []
for result in results:
    all_errors.extend(parse_error_list(result))
```

**Performance characteristics:**
- Sequential: 40 seconds for 20 files
- Parallel: 4 seconds for 20 files (10x speedup)

### Multi-File Processing Design

**Recommended architecture for log analysis:**

```python
# Step 1: Load multiple log files
files = ["auth.log", "syslog", "audit.log"]
log_contents = {}
for filename in files:
    with open(filename) as f:
        log_contents[filename] = f.read()

# Step 2: Parallel analysis with llm_query_batch()
prompts = [
    f"Analyze {fname} for security events:\n{content}"
    for fname, content in log_contents.items()
]
results = llm_query_batch(prompts)

# Step 3: Aggregate and cross-reference
findings_by_file = dict(zip(files, results))
cross_file_events = llm_query(
    f"Find correlations across these analyses:\n{findings_by_file}"
)

# Step 4: Return final answer
FINAL(cross_file_events)
```

---

## System Log Analysis Integration Points

Based on the above architecture, here's where system log analysis features fit:

### 1. Parser Functions (REPL Injection)
**Location:** Extend `REPLEnv._inject_special_functions()`
- Syslog parser
- JSON log parser
- Apache/Nginx parsers
- CSV parser
- Windows Event Log parser

### 2. Format Detection
**Location:** Create `log_format_detector.py`
- Auto-detect log format from samples
- Guide parser selection

### 3. Analysis Functions
**Location:** Create log-specific aggregators
- Group by severity level
- Group by host
- Group by process
- Find anomalies
- Extract metrics

### 4. Specialized REPL Environment
**Location:** Create `LogAnalysisREPLEnv` subclass
- Inject all log-specific functions
- Pre-load format detector
- Provide log-aware system prompt

### 5. Custom System Prompt
**Location:** Extend `prompts.py`
- Guide LLM on log analysis strategy
- Suggest chunking strategies
- Recommend use of batch processing

### 6. Multi-File Handling
**Location:** Leverage existing `llm_query_batch()`
- Process multiple log files in parallel
- Correlate findings across files

---

## Data Flow Diagram for Log Analysis

```
User provides:
  - Log files (syslog, JSON, etc.)
  - Query (Find errors in past 24h across all hosts)

        ↓

RLM_REPL.completion(context=logs, query=query)

        ↓

REPLEnv loads logs as context variable

        ↓

LLM (iteration 1):
  - Peeks at context (format detection)
  - Determines: JSON logs, 1M lines, 50MB

        ↓

LLM (iteration 2):
  - Chunks: splits into 20 chunks
  - Uses llm_query_batch() with parallel processing

        ↓

Sub-LLMs (parallel):
  - Parse each chunk
  - Extract error entries
  - Return structured results

        ↓

LLM (iteration 3):
  - Aggregates results
  - Groups by host/timestamp
  - Identifies patterns

        ↓

LLM (iteration 4):
  - Provides FINAL answer

        ↓

User gets structured analysis of logs
```

---

## Key Design Decisions for Implementation

1. **Don't force a parser** - Let the LLM discover the format through peeking
2. **Provide building blocks** - Inject simple parsers, let LLM combine them
3. **Use batch processing** - For 1M+ lines, parallelize across chunks
4. **Maintain state** - Keep parsed data in REPL variables for reuse
5. **Flexible output** - Support JSON, CSV, or natural language results
6. **Multi-file support** - Use llm_query_batch() for parallel file analysis

---

## Summary

| Aspect | Location | Capability |
|--------|----------|-----------|
| **Entry Point** | `RLM_REPL.completion()` | Handle any size context |
| **Text Parsing** | REPL environment | Full Python, regex, custom parsers |
| **Type Handling** | `convert_context_for_repl()` | String, dict, list, JSON |
| **Extension Points** | `REPLEnv._inject_special_functions()` | Add custom parsers/functions |
| **Multi-file** | `llm_query_batch()` | Parallel processing (10x speedup) |
| **File I/O** | `tempfile.mkdtemp()` | Safe temp directory operations |
| **Log Parsing** | Custom injection or subclass | Detection → Parsing → Aggregation |

The framework is **architecture-ready for system log analysis** - it just needs the specialized parsers and prompts to be added.
