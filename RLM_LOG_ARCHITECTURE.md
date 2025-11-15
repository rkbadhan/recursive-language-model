# RLM Log Analysis Architecture

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                     │
│  analyzer = RLMLogAnalyzer()                                    │
│  result = analyzer.completion(context=logs, query="Find issues")│
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RLMLogAnalyzer (rlm_log_analysis.py)           │
│  - Specialized RLM for log analysis                             │
│  - Uses LOG_ANALYSIS_SYSTEM_PROMPT                              │
│  - Creates LogAnalysisREPLEnv                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RLM ITERATIVE LOOP (rlm_repl.py)                   │
│                                                                  │
│  for iteration in range(max_iterations):                        │
│      1. Query LLM for next action                               │
│      2. Extract code blocks from response                       │
│      3. Execute code in REPL                                    │
│      4. Add results to message history                          │
│      5. Check for FINAL answer                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          LogAnalysisREPLEnv (repl_log.py)                       │
│                                                                  │
│  Injected Functions:                                            │
│  ├─ parse_log()           ← from log_parsers.py                │
│  ├─ parse_jstack()        ← from log_parsers.py                │
│  ├─ parse_strace()        ← from log_parsers.py                │
│  ├─ parse_gc_log()        ← from log_parsers.py                │
│  ├─ correlate_logs()      ← from log_correlator.py             │
│  ├─ detect_all_patterns() ← from log_correlator.py             │
│  ├─ llm_query()           ← from repl.py (recursive calls)     │
│  └─ llm_query_batch()     ← from repl.py (parallel calls)      │
│                                                                  │
│  Context: 'context' variable = user's logs                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LLM GENERATED CODE                             │
│                                                                  │
│  Example iteration 1:                                           │
│  ```repl                                                        │
│  # Peek at structure                                            │
│  print(type(context))                                           │
│  print(list(context.keys()))                                    │
│  ```                                                            │
│                                                                  │
│  Example iteration 2:                                           │
│  ```repl                                                        │
│  # Parse each log                                               │
│  jstack = parse_log(context['jstack'])                          │
│  gc = parse_log(context['gc'])                                  │
│  print(f"Deadlock: {jstack.get('has_deadlock')}")               │
│  ```                                                            │
│                                                                  │
│  Example iteration 3:                                           │
│  ```repl                                                        │
│  # Correlate and detect patterns                                │
│  timeline = correlate_logs({'jstack': jstack, 'gc': gc})        │
│  patterns = detect_all_patterns(timeline)                       │
│  print(f"Patterns: {[p['pattern'] for p in patterns]}")         │
│  ```                                                            │
│                                                                  │
│  Example iteration 4:                                           │
│  FINAL("""                                                      │
│  Root Cause: Deadlock due to memory pressure                    │
│  Timeline: Memory leak → GC pauses → Deadlock                   │
│  """)                                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User Logs (jstack, gc, strace)
        ↓
[Loaded into REPL as 'context' variable]
        ↓
LLM sees: "You have logs in the context variable"
        ↓
LLM writes: parse_log(context['jstack'])
        ↓
[Code executed in REPL using injected parse_log()]
        ↓
[Result returned to LLM]
        ↓
LLM writes: correlate_logs({...})
        ↓
[Code executed in REPL using injected correlate_logs()]
        ↓
[Timeline returned to LLM]
        ↓
LLM writes: detect_all_patterns(timeline)
        ↓
[Code executed in REPL using injected detect_all_patterns()]
        ↓
[Patterns returned to LLM]
        ↓
LLM analyzes patterns and provides FINAL answer
        ↓
Root Cause Analysis returned to user
```

## Component Responsibilities

### 1. log_parsers.py (Standalone Tools)
```python
# Pure Python functions - NO RLM dependency
def parse_log(content, format_hint=None):
    """Parse any log format"""
    # Auto-detect format
    # Parse to structured data
    # Return dict with results

def parse_jstack(content):
    """Parse Java thread dumps"""
    # Extract threads, deadlocks, lock chains
    # Return structured data

# ... more parsers
```

**Role:** Provides the **tools** that RLM will use

### 2. log_correlator.py (Standalone Tools)
```python
# Pure Python functions - NO RLM dependency
def correlate_logs(parsed_logs):
    """Build timeline from multiple logs"""
    # Extract events from each log
    # Sort by timestamp
    # Return Timeline object

def detect_all_patterns(timeline):
    """Find known issue patterns"""
    # Look for deadlocks
    # Look for memory pressure
    # Look for I/O blocking
    # Return list of patterns

# ... more correlation tools
```

**Role:** Provides **correlation tools** that RLM will use

### 3. repl_log.py (RLM Integration)
```python
from rlm.repl import REPLEnv
from rlm.log_parsers import *
from rlm.log_correlator import *

class LogAnalysisREPLEnv(REPLEnv):
    """REPL with log analysis tools injected"""

    def _inject_special_functions(self):
        # Inject base RLM functions
        super()._inject_special_functions()

        # INJECT LOG ANALYSIS TOOLS
        self.globals['parse_log'] = parse_log
        self.globals['correlate_logs'] = correlate_logs
        self.globals['detect_all_patterns'] = detect_all_patterns
        # ... etc
```

**Role:** Makes tools **available to LLM-generated code**

### 4. rlm_log_analysis.py (RLM Orchestrator)
```python
class RLMLogAnalyzer(RLM):
    """RLM specialized for log analysis"""

    def setup_context(self, context, query):
        # Use specialized system prompt
        self.messages = build_log_analysis_system_prompt()

        # Use specialized REPL with tools
        self.repl_env = LogAnalysisREPLEnv(...)

    def completion(self, context, query):
        # Main RLM loop
        for iteration in range(max_iterations):
            response = self.llm.completion(messages)

            if code_blocks:
                # Execute in REPL (which has tools)
                execute_in_repl()

            if final_answer:
                return final_answer
```

**Role:** Orchestrates the **LLM exploration** of logs

### 5. LOG_ANALYSIS_SYSTEM_PROMPT (Guidance)
```python
LOG_ANALYSIS_SYSTEM_PROMPT = """
You are a log analyzer with these tools:
- parse_log(content) - Parse any log
- correlate_logs(dict) - Build timelines
- detect_all_patterns(timeline) - Find issues

Your process:
1. Peek at context
2. Parse logs
3. Correlate them
4. Detect patterns
5. Provide root cause
"""
```

**Role:** Tells LLM **HOW to use the tools**

## Example Execution Trace

```
[USER]
result = analyzer.completion(
    context={'jstack': jstack_log, 'gc': gc_log},
    query="Why did app freeze?"
)

[RLM] Setting up context...
[RLM] Loading logs into REPL as 'context' variable
[RLM] Injecting log analysis functions into REPL

[ITERATION 1]
[LLM] → Let me check what's in the context
[LLM] → Writing code:
```repl
print(type(context))
print(list(context.keys()))
```
[REPL] → Executing code...
[REPL] → Output: <class 'dict'>
                ['jstack', 'gc']

[ITERATION 2]
[LLM] → Let me parse each log
[LLM] → Writing code:
```repl
jstack_parsed = parse_log(context['jstack'])
gc_parsed = parse_log(context['gc'])
print(f"jstack format: {jstack_parsed['format']}")
print(f"deadlock: {jstack_parsed.get('has_deadlock')}")
print(f"gc collections: {gc_parsed['total_collections']}")
```
[REPL] → Executing code...
[REPL] → Output: jstack format: jstack
                deadlock: True
                gc collections: 5

[ITERATION 3]
[LLM] → Let me correlate and find patterns
[LLM] → Writing code:
```repl
timeline = correlate_logs({
    'jstack': jstack_parsed,
    'gc': gc_parsed
})
patterns = detect_all_patterns(timeline)
for p in patterns:
    print(f"{p['severity']}: {p['pattern']} - {p['description']}")
```
[REPL] → Executing code...
[REPL] → Output: CRITICAL: deadlock - Circular lock dependency
                WARNING: memory_pressure - GC pauses increasing

[ITERATION 4]
[LLM] → Now I can provide the final analysis
[LLM] → Writing:
FINAL("""
ROOT CAUSE ANALYSIS:

1. DEADLOCK (CRITICAL)
   - Found in jstack at Thread-1 and Thread-2
   - Circular lock dependency detected

2. MEMORY PRESSURE (WARNING)
   - GC pause times increased from 200ms to 8000ms
   - Indicates memory leak

3. TIMELINE:
   - Memory pressure started before deadlock
   - Likely trigger for resource contention

ROOT CAUSE: Memory leak caused resource exhaustion,
leading to thread contention and eventual deadlock.

RECOMMENDATIONS:
1. Analyze heap dump to find memory leak
2. Fix lock ordering in connection pool
3. Add connection timeout mechanisms
""")

[RLM] Final answer found!
[RLM] Returning to user...

[USER] Receives:
"ROOT CAUSE ANALYSIS: ..."
```

## Key Takeaways

1. **Parsers/Correlators** = Standalone tools (like a library)
2. **LogAnalysisREPLEnv** = Injects tools into REPL (makes them available)
3. **RLMLogAnalyzer** = Orchestrates LLM exploration (RLM loop)
4. **System Prompt** = Guides LLM on using tools (instructions)
5. **LLM** = Autonomously explores logs using the tools (intelligence)

The magic is: **LLM decides WHAT to do, tools provide HOW to do it.**
