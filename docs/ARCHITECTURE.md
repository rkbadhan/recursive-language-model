# RLM Architecture Guide

This document explains the architecture and organization of the Recursive Language Models codebase.

## Project Structure

```
recursive-language-model/
├── rlm/                      # Core implementation
├── eval/                     # Benchmarking & evaluation
├── docs/                     # Documentation
├── tests/                    # Unit tests
└── main.py                   # Example demonstrations
```

---

## Core Implementation (`rlm/`)

The core RLM implementation with three key components:

### 1. Core Classes

- **`rlm.py`** - Abstract base class defining the RLM interface
- **`rlm_repl.py`** - Main RLM implementation using REPL
- **`repl.py`** - Python REPL environment for code execution

### 2. Interfaces (`rlm/interfaces/`)

Different ways to interact with RLM:

**`chat_completion.py`** - Standard chat completion API
- Provides OpenAI/Anthropic-style message interface
- Converts messages → RLM's context/query format
- Makes RLM compatible with any chat-based system

**Usage:**
```python
from rlm.interfaces import RLMChatCompletionClient

client = RLMChatCompletionClient(
    model="gpt-4o",
    recursive_model="gpt-4o-mini"
)

response = client.completion(messages=[
    {"role": "system", "content": "context"},
    {"role": "user", "content": "question"}
])
```

**Future interfaces:**
- `streaming.py` - Streaming responses
- `function_calling.py` - Function calling support
- `async_client.py` - Pure async interface

### 3. Utils (`rlm/utils/`)

- **`llm.py`** - OpenAI API client wrapper
- **`prompts.py`** - Prompt templates
- **`utils.py`** - Helper functions

### 4. Logger (`rlm/logger/`)

- **`root_logger.py`** - Root LM execution logs
- **`repl_logger.py`** - REPL execution display

---

## Evaluation (`eval/`)

Benchmarking and evaluation scripts organized by benchmark:

### OOLONG (`eval/oolong/`)

Long-context aggregation benchmark evaluation.

**Files:**
- `eval.py` - Main evaluation script
- `test_integration.py` - Integration tests
- `README.md` - Full documentation

**Usage:**
```bash
python eval/oolong/eval.py --dataset synth --max-examples 10
```

### Adding New Benchmarks

Create a new subdirectory:

```
eval/
└── your_benchmark/
    ├── __init__.py
    ├── eval.py
    ├── test_integration.py
    └── README.md
```

Examples:
- `eval/ruler/` - RULER benchmark
- `eval/longbench/` - LongBench evaluation
- `eval/niah/` - Needle in a Haystack

---

## Design Principles

### 1. Separation of Concerns

- **Core** (`rlm/`) - Implementation details
- **Interfaces** (`rlm/interfaces/`) - How to use RLM
- **Evaluation** (`eval/`) - Testing and benchmarking

### 2. Interface Abstraction

The `interfaces/` package provides different ways to use the same core:

```
User → Interface → Core RLM
     (chat API)   (context/query)
```

Benefits:
- Core stays clean and focused
- Easy to add new interfaces
- Backward compatible

### 3. Pluggable Evaluation

Each benchmark is self-contained:

```
eval/
├── benchmark_a/  # Independent
├── benchmark_b/  # Independent
└── benchmark_c/  # Independent
```

Benefits:
- Easy to add new benchmarks
- No cross-dependencies
- Clear organization

---

## Import Patterns

### Core Usage

```python
from rlm import RLM_REPL

rlm = RLM_REPL(model="gpt-4o")
result = rlm.completion(context="...", query="...")
```

### Chat Completion Interface

```python
from rlm.interfaces import RLMChatCompletionClient

client = RLMChatCompletionClient(model="gpt-4o")
result = client.completion(messages=[...])
```

### Backward Compatibility

Old imports still work:

```python
# Old way (still works)
from rlm import RLMOolongAdapter

# New way (recommended)
from rlm.interfaces import RLMChatCompletionClient
```

---

## Adding New Features

### Adding a New Interface

1. Create `rlm/interfaces/your_interface.py`
2. Implement the interface class
3. Export from `rlm/interfaces/__init__.py`
4. Update `rlm/__init__.py` for top-level import (optional)

Example:
```python
# rlm/interfaces/streaming.py
class RLMStreamingClient:
    def __init__(self, ...):
        self.rlm = RLM_REPL(...)

    async def stream(self, context, query):
        # Stream responses as they're generated
        ...
```

### Adding a New Benchmark

1. Create `eval/your_benchmark/` directory
2. Add `eval.py`, `test_integration.py`, `README.md`
3. Add `__init__.py`
4. Update `eval/README.md`

Example structure:
```
eval/ruler/
├── __init__.py
├── eval.py           # Main evaluation
├── test_integration.py
└── README.md
```

---

## Key Concepts

### 1. Context vs Query

RLM's core interface separates:
- **Context** - The data to explore (can be huge)
- **Query** - The question to answer

```python
rlm.completion(
    context="<1M tokens of data>",
    query="What is X?"
)
```

### 2. Message-based Interfaces

Standard chat APIs use messages:
- **System** - Background context
- **User** - Question
- **Assistant** - Response

The chat completion interface converts this to context/query.

### 3. Evaluation Independence

Each benchmark is independent:
- Own dependencies (`requirements-*.txt`)
- Own documentation
- Own test suite

---

## Dependencies

### Core Dependencies (`requirements.txt`)

```
openai>=1.0.0
python-dotenv>=1.0.0
rich>=13.0.0
```

### Benchmark Dependencies

Each benchmark has its own requirements file:
- `requirements-oolong.txt` - OOLONG evaluation
- `requirements-ruler.txt` - RULER evaluation (future)

Install as needed:
```bash
pip install -r requirements.txt              # Core
pip install -r requirements-oolong.txt       # OOLONG
```

---

## Testing

### Unit Tests (`tests/`)

```bash
python tests/test_basic.py
python tests/test_async_depth.py
```

### Integration Tests

```bash
python eval/oolong/test_integration.py
```

### Example Demonstrations

```bash
python main.py
```

---

## Future Architecture

### Planned Additions

1. **More Interfaces**
   - `rlm/interfaces/streaming.py`
   - `rlm/interfaces/function_calling.py`
   - `rlm/interfaces/anthropic_messages.py`

2. **More Benchmarks**
   - `eval/ruler/` - RULER benchmark
   - `eval/longbench/` - LongBench
   - `eval/niah/` - Needle in a Haystack

3. **Performance**
   - Caching layer in `rlm/cache/`
   - Async optimizations
   - Prefix caching support

---

## Summary

The architecture is designed for:
- ✅ **Clarity** - Clear separation of concerns
- ✅ **Extensibility** - Easy to add interfaces and benchmarks
- ✅ **Compatibility** - Works with existing systems
- ✅ **Maintainability** - Organized and documented

**Key insight:** The chat completion interface is NOT specific to OOLONG—it's a general-purpose way to use RLM with any message-based system!
