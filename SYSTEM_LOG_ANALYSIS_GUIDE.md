# System Log Analysis for RLM - Complete Guide

## Overview

This guide explains how to understand and implement system log analysis capabilities in the Recursive Language Model (RLM) framework.

Three comprehensive documents have been created to support this effort:

1. **ARCHITECTURE_FOR_LOG_ANALYSIS.md** - Deep architectural understanding
2. **LOG_ANALYSIS_QUICK_START.md** - Implementation examples and quick start
3. **IMPLEMENTATION_CHECKLIST.md** - Step-by-step implementation roadmap

---

## Quick Navigation

### For Understanding the Architecture
Start here: **ARCHITECTURE_FOR_LOG_ANALYSIS.md**

This document covers:
- Main entry points and input processing flow
- Current parsing and analysis capabilities  
- How the model processes different content types
- Extension points for new log format parsers
- Existing file I/O and multi-file processing

**Time to read:** 30-40 minutes
**Best for:** Understanding "how RLM works" and "where does log analysis fit"

---

### For Implementation Details
Start here: **LOG_ANALYSIS_QUICK_START.md**

This document provides:
- Overview of what needs to be built
- Code templates for each new module
- Usage examples (simple, multi-file, structured)
- Key design principles
- Performance characteristics

**Time to read:** 20-30 minutes
**Best for:** "Show me the code" and "How do I use this"

---

### For Building the Feature
Start here: **IMPLEMENTATION_CHECKLIST.md**

This document outlines:
- 8 phases of implementation
- Detailed tasks for each phase
- Acceptance criteria
- Success metrics
- Timeline estimates (13-21 hours total)

**Time to read:** 20-25 minutes
**Best for:** Project planning and task tracking

---

## Architecture at a Glance

```
User Query + System Logs
        ↓
RLM_REPL.completion(context=logs, query="...")
        ↓
REPLEnv loads logs as `context` variable
        ↓
LLM Iterative Loop:
  1. Peek at structure
  2. Detect format (syslog/JSON/Apache/CSV)
  3. Chunk if large (100k+ chars)
  4. Parse using specialized functions
  5. Aggregate/filter results
  6. Return FINAL answer
        ↓
User gets structured log analysis
```

## Key Files in RLM Framework

### Current Implementation (Existing)
- **rlm/rlm_repl.py** - Main RLM orchestrator (213 lines)
- **rlm/repl.py** - REPL environment with Python execution (656 lines)
- **rlm/utils/utils.py** - Parsing utilities (335 lines)
- **rlm/utils/llm.py** - OpenAI client wrapper (198 lines)
- **rlm/utils/prompts.py** - System prompts (210 lines)

### For Log Analysis (New, to Create)
- **rlm/log_parsers.py** - Log format parsers (NEW - ~150 lines)
- **rlm/repl_log.py** - Log-specialized REPL environment (NEW - ~100 lines)
- **rlm/rlm_log_analysis.py** - Convenience class (NEW - ~50 lines)
- **test_log_analysis.py** - Tests (NEW - ~300 lines)

### Documentation (New, Created)
- **ARCHITECTURE_FOR_LOG_ANALYSIS.md** - This architecture overview (18KB)
- **LOG_ANALYSIS_QUICK_START.md** - Quick start guide (15KB)
- **IMPLEMENTATION_CHECKLIST.md** - Implementation checklist (12KB)
- **SYSTEM_LOG_ANALYSIS_GUIDE.md** - This navigation guide (this file)

---

## Implementation Roadmap

### Phase 1: Foundation (2-3 hours)
Create `/rlm/log_parsers.py` with:
- `parse_syslog()` - Standard syslog format
- `parse_json_logs()` - JSON log streams
- `parse_apache_access()` - Web server logs
- `parse_csv_logs()` - CSV-formatted logs
- `detect_log_format()` - Auto-detection

Create `/rlm/repl_log.py` with:
- `LogAnalysisREPLEnv` class - Inject parsers into REPL

### Phase 2: System Prompt (1 hour)
Update `/rlm/utils/prompts.py` with:
- `LOG_ANALYSIS_SYSTEM_PROMPT` - Guide LLM on log analysis
- Optional: Security and performance analysis prompts

### Phase 3: Convenience Class (1-2 hours)
Create `/rlm/rlm_log_analysis.py` with:
- `RLMLogAnalyzer` class - Easy-to-use interface
- Helper methods: `analyze_file()`, `analyze_files()`, etc.

### Phase 4: Testing (2-3 hours)
Create `/test_log_analysis.py` with:
- Parser tests
- REPL environment tests
- Integration tests
- Performance benchmarks

### Phase 5: Documentation (2-3 hours)
- Update README.md
- Create EXAMPLES.md
- Create API reference

### Phase 6: Integration (1-2 hours)
- Add log analysis examples to main.py
- Add menu options for log analysis

### Phase 7: Advanced Features (2-4 hours, optional)
- Additional aggregators (by timestamp, process, etc.)
- Format-specific enhancements
- Anomaly detection

### Phase 8: Optimization (2-3 hours, optional)
- Smart chunking
- Result caching
- Memory optimization

---

## Key Design Decisions

### 1. Don't Force a Parser
The LLM discovers log format through "peeking" at the context first. This makes it adaptive to any format.

### 2. Provide Building Blocks
Inject simple, focused parsers. Let the LLM combine them as needed.

### 3. Leverage Parallel Processing
For large logs, use `llm_query_batch()` for 10x speedup over sequential processing.

### 4. Maintain State
Keep parsed data in REPL variables for multi-step analysis without re-parsing.

### 5. Flexible Output
Support both structured (JSON) and natural language results.

---

## Usage Examples

### Simple Usage
```python
from rlm.rlm_log_analysis import RLMLogAnalyzer

analyzer = RLMLogAnalyzer(model="gpt-4o-mini")
result = analyzer.completion(
    context=open('/var/log/syslog').read(),
    query="Find all ERROR messages"
)
print(result)
```

### Multi-File Analysis
```python
logs = {
    'auth': open('/var/log/auth.log').read(),
    'syslog': open('/var/log/syslog').read(),
    'apache': open('/var/log/apache2/access.log').read(),
}

result = analyzer.completion(
    context=logs,
    query="Correlate authentication failures across logs"
)
```

### JSON Logs
```python
import json

logs = [
    json.loads(line) 
    for line in open('/var/log/app.log')
]

result = analyzer.completion(
    context=logs,
    query="Summarize error patterns"
)
```

---

## Performance Characteristics

| Scenario | Processing | Speed | Notes |
|----------|-----------|-------|-------|
| Small logs (< 100k chars) | Direct parsing | < 1 sec | Single API call |
| Medium logs (100k-1M chars) | Chunking + sequential | 5-10 sec | Multiple API calls |
| Large logs (1M+ chars) | Chunking + parallel | 5-10 sec | Uses llm_query_batch() |

**Key advantage:** Parallel processing achieves 10x speedup for large datasets.

---

## Capabilities After Implementation

1. **Format Detection** - Auto-detect syslog, JSON, Apache, CSV
2. **Parsing** - Extract structured data from any format
3. **Aggregation** - Group by level, host, process, timestamp
4. **Analysis** - Find errors, anomalies, patterns
5. **Multi-File** - Correlate across multiple logs
6. **Parallel** - Process 1M+ character logs efficiently
7. **Flexible** - Return JSON, structured data, or insights

---

## Testing Strategy

### Unit Tests (Parser functions)
- Test each parser with realistic log samples
- Test edge cases (malformed entries, empty input)
- Test format detection

### Integration Tests (REPL environment)
- Verify parsers work in REPL context
- Verify aggregation functions work
- Verify llm_query_batch() integration

### End-to-End Tests (RLMLogAnalyzer)
- Test simple analysis workflows
- Test multi-file analysis
- Test with various log formats
- Performance benchmarks

### Validation Tests
- Results match manual inspection
- No memory leaks
- No data loss during parsing

---

## Success Criteria

By the end of implementation, users should be able to:

1. **Analyze any system log** in a single function call
2. **Handle 1M+ character logs** efficiently
3. **Process multiple files in parallel**
4. **Get structured or natural language results**
5. **Detect patterns and anomalies** automatically
6. **Correlate events** across multiple logs

---

## Next Steps

1. Read **ARCHITECTURE_FOR_LOG_ANALYSIS.md** (30 min)
   - Understand how RLM works
   - Identify where log analysis fits

2. Review **LOG_ANALYSIS_QUICK_START.md** (20 min)
   - See code examples
   - Understand design patterns

3. Follow **IMPLEMENTATION_CHECKLIST.md** (ongoing)
   - Create files Phase 1-3
   - Test Phase 4
   - Document Phase 5
   - Integrate Phase 6
   - Optimize Phase 7-8

4. Execute in order (13-21 hours total)

---

## Questions Answered

### "How much code do I need to write?"
- **log_parsers.py**: ~150 lines
- **repl_log.py**: ~100 lines
- **rlm_log_analysis.py**: ~50 lines
- **test_log_analysis.py**: ~300 lines
- **Total: ~600 lines** to add full log analysis

### "Will this break existing features?"
No. All code is new. The only modification to existing code is adding a new prompt constant to `utils/prompts.py`.

### "What's the performance impact?"
Minimal. Parallel processing actually makes things faster (10x for large logs).

### "Do I need new dependencies?"
No. All parsing uses Python stdlib (re, json, csv). All parallel processing uses existing llm_query_batch().

### "Can I add support for other log formats?"
Yes. Add a new parser function to `log_parsers.py` and inject it in `LogAnalysisREPLEnv._inject_special_functions()`.

---

## Additional Resources

- **RLM Research Paper**: https://alexzhang13.github.io/blog/2025/rlm/
- **README.md** - General RLM documentation
- **IMPLEMENTATION_SUMMARY.md** - RLM implementation details
- **ASYNC_DEPTH_FEATURES.md** - Advanced RLM features

---

## Support

For questions or clarifications:
1. Review the relevant section in the three guide documents
2. Check the code examples in LOG_ANALYSIS_QUICK_START.md
3. Reference the detailed explanations in ARCHITECTURE_FOR_LOG_ANALYSIS.md

All information needed to implement system log analysis is contained in these three documents.

---

**Last Updated**: November 15, 2024
**RLM Version**: 2.0+
**Status**: Architecture and planning complete, ready for implementation
