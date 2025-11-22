# RLM Analysis Examples

This directory contains specialized prompts for analyzing different types of system logs and diagnostic data using the Recursive Language Model (RLM) framework.

## Available Examples

### System Logs & Debugging
- **`log_analysis.py`** - Multi-source log correlation (jstack, GC, strace, syslog)
- **`jstack_analysis.py`** - Java thread dump analysis (deadlocks, contention)
- **`gc_analysis.py`** - Garbage collection log analysis (memory pressure, tuning)

### Performance Analysis
- **`strace_analysis.py`** - System call tracing (bottlenecks, I/O issues)
- **`pstack_analysis.py`** - Native C/C++ stack traces (deadlocks, blocking)
- **`pmstack_analysis.py`** - Memory mapping analysis (leaks, fragmentation)

## Usage

### Basic Usage

```python
from rlm.rlm_repl import RLM_REPL
from examples.jstack_analysis import JSTACK_PROMPT

# Load your data
with open("thread_dump.txt") as f:
    dump = f.read()

# Create RLM with specialized prompt
rlm = RLM_REPL(custom_prompt=JSTACK_PROMPT)

# Run analysis
result = rlm.query(
    context=dump,
    query="Find deadlocks and thread contention issues"
)

print(result)
```

### Customizing Prompts

You can modify the prompts to fit your specific needs:

```python
from examples.jstack_analysis import JSTACK_PROMPT

# Extend the prompt with custom instructions
my_prompt = JSTACK_PROMPT + """

Additional instructions:
- Focus on pool-* threads only
- Highlight any threads waiting >10 seconds
"""

rlm = RLM_REPL(custom_prompt=my_prompt)
```

### Multi-Source Analysis

For correlating multiple log sources:

```python
from examples.log_analysis import LOG_ANALYSIS_PROMPT

logs = {
    "jstack": open("thread_dump.txt").read(),
    "gc_log": open("gc.log").read(),
    "strace": open("strace.out").read()
}

rlm = RLM_REPL(custom_prompt=LOG_ANALYSIS_PROMPT)
result = rlm.query(
    context=logs,
    query="What caused the system failure?"
)
```

## How It Works

These specialized prompts:

1. **Provide domain expertise** - Format specifications, error codes, common patterns
2. **Guide parsing strategies** - Regex patterns, extraction logic, correlation techniques
3. **Suggest analysis workflows** - Step-by-step methodology, best practices
4. **Leverage parallel processing** - Use `llm_query_batch()` for efficiency
5. **Offer actionable recommendations** - Concrete fixes, configuration examples

## Creating Your Own Prompts

To create a custom analysis prompt:

1. Start with `REPL_SYSTEM_PROMPT` from `rlm/utils/prompts.py`
2. Add your domain-specific knowledge:
   - Data formats and structures
   - Common patterns and anti-patterns
   - Parsing strategies with code examples
   - Analysis methodology
   - Recommendations and fixes

3. Save as `examples/my_analysis.py`:

```python
"""My custom analysis with RLM."""

MY_PROMPT = """You are an expert in [domain]...

[Your specialized instructions here]

When done, return FINAL(answer) or FINAL_VAR(variable_name).
"""

# Usage example
if __name__ == "__main__":
    from rlm.rlm_repl import RLM_REPL

    rlm = RLM_REPL(custom_prompt=MY_PROMPT)
    result = rlm.query(context=your_data)
    print(result)
```

## Tips

- **Peek first** - Always inspect `context` structure before analysis
- **Use batch processing** - `llm_query_batch()` is much faster than loops
- **Chunk wisely** - Sub-LLMs can handle ~500K chars
- **Build incrementally** - Use buffers to accumulate results
- **Provide evidence** - Quote specific log entries, timestamps, metrics

## Contributing

Have a useful analysis prompt? Consider sharing it by:
1. Adding it to this directory
2. Including clear documentation and usage examples
3. Submitting a pull request

## Framework Documentation

For core RLM framework documentation, see the main project README.
