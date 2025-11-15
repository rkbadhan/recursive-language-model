# How RLM Uses Log Analysis Tools

## The Problem You Identified

You're correct! The tests in `test_log_analysis.py` only test the **parsers and correlators directly**, but don't show how **RLM actually uses these tools**. Let me explain the full architecture.

## The RLM Concept for Log Analysis

### Without RLM (Traditional Approach)
```python
# User would manually:
1. Parse each log
2. Correlate them
3. Detect patterns
4. Analyze results

jstack_parsed = parse_log(jstack_log)
gc_parsed = parse_log(gc_log)
timeline = correlate_logs({'jstack': jstack_parsed, 'gc': gc_parsed})
patterns = detect_all_patterns(timeline)
# Manual analysis...
```

### With RLM (AI-Powered Approach)
```python
from rlm import RLMLogAnalyzer

analyzer = RLMLogAnalyzer(model="gpt-4o-mini")

# Just provide logs and ask a question!
result = analyzer.completion(
    context={
        'jstack': open('thread_dump.txt').read(),
        'gc': open('gc.log').read(),
        'strace': open('strace.log').read()
    },
    query="Why did the application freeze at 14:30?"
)

# LLM automatically:
# 1. Explores the logs
# 2. Uses parse_log() to parse each
# 3. Uses correlate_logs() to build timeline
# 4. Uses detect_all_patterns() to find issues
# 5. Provides comprehensive root cause analysis
```

## How It Works: The RLM Flow

### Step 1: Setup (RLMLogAnalyzer.__init__)
```
User creates RLMLogAnalyzer
    ↓
Initializes with log-specific system prompt
    ↓
Sets up LogAnalysisREPLEnv (extends REPLEnv)
```

### Step 2: Context Loading (completion method)
```
User calls: analyzer.completion(context=logs, query="...")
    ↓
Logs loaded into REPL as 'context' variable
    ↓
Special functions injected into REPL:
  - parse_log()
  - correlate_logs()
  - detect_all_patterns()
  - llm_query()
  - llm_query_batch()
```

### Step 3: Iterative Exploration (RLM loop)
```
Iteration 1:
  LLM: "Let me peek at the context structure"
  LLM writes: ```repl
              print(type(context))
              for name, log in context.items():
                  print(f"{name}: {len(log)} chars")
              ```
  Output: jstack: 1500 chars, gc: 2000 chars, strace: 3000 chars

Iteration 2:
  LLM: "Let me parse each log"
  LLM writes: ```repl
              from rlm.log_parsers import parse_log
              jstack_parsed = parse_log(context['jstack'])
              gc_parsed = parse_log(context['gc'])
              strace_parsed = parse_log(context['strace'])
              print(f"jstack: {jstack_parsed['format']}, deadlock={jstack_parsed.get('has_deadlock')}")
              print(f"gc: {gc_parsed['total_collections']} collections")
              ```
  Output: jstack: jstack, deadlock=True
          gc: 5 collections

Iteration 3:
  LLM: "Let me correlate the logs"
  LLM writes: ```repl
              from rlm.log_correlator import correlate_logs, detect_all_patterns
              timeline = correlate_logs({
                  'jstack': jstack_parsed,
                  'gc': gc_parsed,
                  'strace': strace_parsed
              })
              patterns = detect_all_patterns(timeline)
              print(f"Found {len(patterns)} patterns")
              for p in patterns:
                  print(f"  {p['severity']}: {p['pattern']}")
              ```
  Output: Found 2 patterns
            CRITICAL: deadlock
            WARNING: memory_pressure

Iteration 4:
  LLM: "Now I can provide the analysis"
  LLM writes: FINAL("""
              Root Cause Analysis:

              1. DEADLOCK detected in jstack (CRITICAL)
                 - Thread-1 waiting on Lock A
                 - Thread-2 waiting on Lock B

              2. MEMORY PRESSURE detected in GC logs (WARNING)
                 - GC pause times increased from 200ms to 8000ms
                 - Indicates memory leak

              3. Timeline correlation:
                 - Memory pressure started at 14:25
                 - Deadlock occurred at 14:30

              ROOT CAUSE: Memory leak caused resource exhaustion,
              leading to thread contention and eventual deadlock.

              RECOMMENDATION:
              1. Fix memory leak (heap dump analysis needed)
              2. Fix deadlock in connection pool
              """)
```

## The Key Difference

### Traditional Tests (test_log_analysis.py)
```python
# Direct function calls
parsed = parse_jstack(jstack_log)  # Calling function directly
timeline = correlate_logs(parsed_logs)  # Calling function directly
patterns = detect_all_patterns(timeline)  # Calling function directly
```

**These tests validate that the TOOLS work, but not that RLM uses them.**

### RLM Integration (what we need to show)
```python
# RLM uses the tools via code execution
analyzer = RLMLogAnalyzer()
result = analyzer.completion(
    context=logs,
    query="Analyze and find issues"
)
# Behind the scenes, the LLM:
# 1. Writes Python code: parsed = parse_log(context['jstack'])
# 2. Executes it in REPL
# 3. Sees the output
# 4. Decides what to do next
# 5. Iterates until it has the answer
```

## Files Showing RLM Integration

### 1. rlm/repl_log.py (LogAnalysisREPLEnv)
```python
class LogAnalysisREPLEnv(REPLEnv):
    """Injects log analysis tools into REPL"""

    def _inject_special_functions(self):
        # Make these available to LLM-generated code
        self.globals['parse_log'] = parse_log
        self.globals['correlate_logs'] = correlate_logs
        self.globals['detect_all_patterns'] = detect_all_patterns
        # ... etc
```

**This is where the tools become available to the LLM!**

### 2. rlm/rlm_log_analysis.py (RLMLogAnalyzer)
```python
class RLMLogAnalyzer(RLM):
    """RLM specialized for log analysis"""

    def setup_context(self, context, query):
        # Uses LogAnalysisREPLEnv instead of regular REPLEnv
        self.repl_env = LogAnalysisREPLEnv(...)

    def completion(self, context, query):
        # Main RLM loop - LLM iteratively explores
        for iteration in range(max_iterations):
            response = self.llm.completion(...)

            # Execute code blocks in REPL
            if code_blocks:
                execute_in_repl()

            # Check for final answer
            if final_answer:
                return final_answer
```

**This is the RLM loop that lets the LLM explore!**

### 3. LOG_ANALYSIS_SYSTEM_PROMPT
```python
LOG_ANALYSIS_SYSTEM_PROMPT = """
You have access to specialized log parsers:
- parse_log() - auto-detect and parse
- correlate_logs() - build timelines
- detect_all_patterns() - find issues

Your process:
1. PEEK at context structure
2. PARSE each log using parse_log()
3. CORRELATE using correlate_logs()
4. DETECT patterns using detect_all_patterns()
5. ANALYZE and provide root cause

Example:
```repl
parsed = parse_log(context['jstack'])
print(f"Deadlock: {parsed.get('has_deadlock')}")
```
"""
```

**This guides the LLM on HOW to use the tools!**

## Why We Need Both

### Parser/Correlator Tests (test_log_analysis.py)
- ✅ Validates tools work correctly
- ✅ Fast (no API calls)
- ✅ Deterministic
- ❌ Doesn't show RLM integration

### RLM Integration Demo (demo_log_analysis.py)
- ✅ Shows full RLM flow
- ✅ Demonstrates AI-powered analysis
- ✅ Shows how LLM uses tools
- ❌ Requires API key
- ❌ Non-deterministic (LLM decides strategy)

## Running the RLM Demo

To see RLM actually using these tools:

```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the full demo
python demo_log_analysis.py

# Choose option "y" for AI-powered analysis
```

You'll see output like:
```
================================================================================
RLM ANALYSIS (Watch the LLM explore)
================================================================================

[LLM] Iteration 1: Peeking at context...
```repl
print(type(context))
for name in context.keys():
    print(f"{name}: {len(context[name])} chars")
```

[REPL Output]
<class 'dict'>
jstack: 1500 chars
gc: 2000 chars
strace: 3000 chars

[LLM] Iteration 2: Parsing logs...
```repl
jstack_parsed = parse_log(context['jstack'])
gc_parsed = parse_log(context['gc'])
print(f"Deadlock: {jstack_parsed.get('has_deadlock')}")
```

[REPL Output]
Deadlock: True

... (continues iterating) ...

[LLM] Final answer:
FINAL("""
ROOT CAUSE: Memory leak → OOM → Connection pool deadlock
RECOMMENDATION: Fix memory leak and deadlock bug
""")
```

## Summary

**What we built:**
1. ✅ Log parsers (parse_log, parse_jstack, etc.)
2. ✅ Correlation tools (correlate_logs, detect_all_patterns)
3. ✅ LogAnalysisREPLEnv (injects tools into REPL)
4. ✅ RLMLogAnalyzer (RLM that uses the REPL)
5. ✅ System prompt (guides LLM on using tools)

**What tests show:**
- `test_log_analysis.py` - Tools work correctly ✅
- `demo_log_analysis.py` - RLM uses tools (needs API key) ✅
- `test_rlm_log_integration.py` - Integration test (needs API key) ✅

**The RLM concept in action:**
Instead of manually calling functions, the **LLM explores logs autonomously** by writing Python code that uses the injected tools, iteratively building up understanding and providing comprehensive analysis.
