# Recursive Language Models (RLM) - Implementation from Scratch

A clean-room implementation of Recursive Language Models based on the research paper by Alex Zhang and Omar Khattab (MIT CSAIL, Oct 2025).

## 🎯 What are Recursive Language Models?

RLMs are an inference strategy that enables language models to process **unbounded context** by treating input as programmable variables in a REPL environment, rather than direct prompts.

### The Problem: Context Rot
- Traditional LLMs degrade as context grows (even within their context window)
- Example: GPT-5 solves <10% of tasks with 75k+ token histories

### The RLM Solution
Instead of:
```python
LLM("Here's 1M tokens... answer this question")  # ❌ Fails
```

Do this:
```python
context = load_1M_tokens()  # Store as REPL variable
# LLM programmatically explores it:
# - Peek at structure
# - Grep for patterns
# - Chunk intelligently
# - Recursively query sub-LLMs
# - Build answer incrementally
```

## 🚀 Key Features

- **Unbounded Context**: Handle 1M+ token inputs by treating context as data
- **Recursive Exploration**: Root LM can spawn sub-LLM calls via `llm_query()`
- **Programmable**: Full Python REPL for complex context manipulation
- **Drop-in Replacement**: `rlm.completion(context, query)` replaces `llm.completion(prompt)`
- **Learnable Trajectories**: Exploration strategies are trainable via RL

## 📊 Performance Highlights (from Paper)

**OOLONG Benchmark (128k+ tokens):**
- RLM(GPT-4o-mini) **outperforms GPT-5** by +33% (2x performance)
- **Cheaper** than GPT-5 per query
- Even when context fits in window, RLM wins

**BrowseComp-Plus (1000 docs = 10M+ tokens):**
- RLM(GPT-5): **Perfect performance maintained**
- Base GPT-5: Significant degradation
- Handles contexts that don't fit in any model's window

## 🎯 OOLONG Benchmark Evaluation

You can now **evaluate RLM on the OOLONG benchmark** to validate these performance claims!

### Quick Start

```bash
# Install OOLONG dependencies
pip install -r requirements-oolong.txt

# Test the integration
python eval/oolong/test_integration.py

# Run evaluation on OOLONG-synth
python eval/oolong/eval.py --dataset synth --max-examples 10

# Run evaluation on OOLONG-real
python eval/oolong/eval.py --dataset real --max-examples 10
```

### Full Documentation

See [eval/oolong/README.md](eval/oolong/README.md) for:
- Detailed setup instructions
- Command-line options
- Cost optimization tips
- Result interpretation
- Programmatic usage examples

**Compare RLM vs baselines yourself!** 📊

## 📁 Project Structure

```
recursive-language-model/
├── rlm/                      # Core RLM implementation
│   ├── __init__.py           # Package initialization
│   ├── rlm.py                # Base RLM abstract class
│   ├── rlm_repl.py           # Main RLM implementation
│   ├── repl.py               # REPL environment
│   ├── interfaces/           # Different interfaces for RLM
│   │   ├── __init__.py
│   │   └── chat_completion.py # Chat completion API (messages-based)
│   ├── utils/
│   │   ├── llm.py            # OpenAI client wrapper
│   │   ├── prompts.py        # Prompt templates
│   │   └── utils.py          # Helper functions
│   └── logger/
│       ├── root_logger.py    # Root LM logger
│       └── repl_logger.py    # REPL execution logger (Jupyter-style)
├── eval/                     # Evaluation and benchmarking
│   ├── __init__.py
│   └── oolong/               # OOLONG benchmark evaluation
│       ├── __init__.py
│       ├── eval.py           # Main evaluation script
│       ├── test_integration.py # Integration tests
│       └── README.md         # OOLONG documentation
├── docs/                     # General documentation
│   ├── IMPLEMENTATION_SUMMARY.md
│   └── CRITICAL_REVIEW.md
├── main.py                   # Example demonstrations
├── requirements.txt          # Core dependencies
├── requirements-oolong.txt   # OOLONG benchmark dependencies
├── .env.example             # Environment variable template
└── README.md                # This file
```

## 🔧 Installation

### 1. Clone or Download

```bash
cd recursive-language-model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required:**
- `openai>=1.0.0` - OpenAI API client
- `python-dotenv>=1.0.0` - Environment variable management

**Optional:**
- `rich>=13.0.0` - Beautiful terminal output (highly recommended)

### 3. Setup API Key

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

## 📖 Quick Start

### Basic Usage

```python
from rlm.rlm_repl import RLM_REPL

# Create RLM instance
rlm = RLM_REPL(
    model="gpt-4o",              # Root LM
    recursive_model="gpt-4o-mini",  # Sub-LM for recursion
    enable_logging=True,          # Show execution logs
    max_iterations=10             # Max reasoning steps
)

# Use it like a normal LLM call
context = "Your huge context here..."  # Can be 1M+ tokens!
query = "What is the magic number?"

answer = rlm.completion(context=context, query=query)
print(answer)
```

### Run Examples

```bash
python main.py
```

Choose from:
1. **Needle-in-Haystack** - Find a number in 1M lines (intensive)
2. **Multi-Document Reasoning** - Answer from multiple sources
3. **Counting and Aggregation** - OOLONG-style task
4. **Simple Test** - Quick validation (recommended to start)

## 🧠 How It Works

### Architecture

```
User Query + Huge Context
         ↓
    [RLM_REPL]
         ↓
   Root LM (depth=0)
   - Sees: Query only
   - Has: REPL with `context` variable
   - Can: Write Python code
         ↓
    [REPL Env]
   - Executes code
   - Provides llm_query() for recursion
   - Stores intermediate results
         ↓
   Sub-LM calls (depth=1)
   - Process chunks semantically
   - Return results to REPL
         ↓
   Root LM builds final answer
   - Uses FINAL() or FINAL_VAR()
         ↓
      Answer
```

### Iterative Loop

```python
for iteration in range(max_iterations):
    # 1. Root LM decides next action
    response = root_llm.completion(messages)

    # 2. Extract and execute code blocks
    if "```repl" in response:
        execute_in_repl(code)
        add_results_to_messages()

    # 3. Check for final answer
    if "FINAL(" in response:
        return extract_answer()
```

### Special REPL Functions

**Available in REPL environment:**

```python
# 1. Context variable
context  # Your huge input, loaded as Python variable

# 2. Recursive LLM query
result = llm_query("Summarize this chunk: " + chunk)

# 3. Final answer
FINAL("The answer is 42")  # Direct answer
FINAL_VAR(my_answer)       # Return a variable
```

## 🎨 Emergent Strategies

The RLM autonomously discovers these patterns:

### 1. Peeking
```python
```repl
# Check structure first
print(type(context))
print(len(context))
print(context[:1000])  # Preview
```
```

### 2. Grepping
```python
```repl
import re
matches = re.findall(r'magic number is (\d+)', context)
print(matches)
```
```

### 3. Partition + Map
```python
```repl
# Chunk and query each
chunks = [context[i:i+50000] for i in range(0, len(context), 50000)]
results = []
for chunk in chunks:
    result = llm_query(f"Find X in: {chunk}")
    results.append(result)
```
```

### 4. Summarization
```python
```repl
sections = context.split("###")
summaries = [llm_query(f"Summarize: {s}") for s in sections]
final = llm_query(f"Answer based on: {summaries}")
```
```

## ⚙️ Configuration Options

```python
RLM_REPL(
    api_key=None,                    # OpenAI API key (or use env var)
    model="gpt-4o",                  # Root LM model
    recursive_model="gpt-4o-mini",   # Sub-LM model (cheaper)
    max_iterations=20,               # Max reasoning steps
    depth=0,                         # Recursion depth (future use)
    enable_logging=True,             # Colorful execution logs
    track_costs=True                 # Track API usage and costs
)
```

### Cost Tracking

```python
rlm = RLM_REPL(track_costs=True)
answer = rlm.completion(context, query)

# Get cost summary
summary = rlm.cost_summary()
print(f"Total cost: ${summary['estimated_cost_usd']}")
print(f"Total tokens: {summary['total_tokens']}")
print(f"API calls: {summary['total_calls']}")
```

## 🧪 Testing

### Simple Validation

```python
from rlm.rlm_repl import RLM_REPL

# Test basic functionality
context = "Alice has 5 apples. Bob has 3 oranges."
query = "How many fruits total?"

rlm = RLM_REPL(
    model="gpt-4o-mini",
    enable_logging=True
)

result = rlm.completion(context, query)
assert "8" in result
print("✓ Test passed!")
```

### Run Example Suite

```bash
python main.py
# Choose option 4 for quick test
# Choose option 5 for comprehensive suite (expensive!)
```

## 📚 Examples Explained

### Example 1: Needle-in-Haystack
- **Context:** 1M lines of random text
- **Task:** Find hidden magic number
- **Strategy:** Binary search, grepping, chunking
- **Demonstrates:** Handling massive contexts

### Example 2: Multi-Document
- **Context:** 100 documents
- **Task:** Multi-hop question across docs
- **Strategy:** Extract relevant docs, aggregate info
- **Demonstrates:** Information synthesis

### Example 3: Counting/Aggregation
- **Context:** 5000 entries with metadata
- **Task:** Count entries matching criteria
- **Strategy:** Filter, map, reduce pattern
- **Demonstrates:** Structured data processing

### Example 4: Simple Test
- **Context:** Small text snippet
- **Task:** Basic arithmetic
- **Strategy:** Direct computation
- **Demonstrates:** Basic functionality

## 🔬 Advanced Usage

### Custom Context Formats

```python
# String context
rlm.completion("Plain text...", "Query?")

# Structured data
rlm.completion({"key": "value", "data": [...]}, "Query?")

# List of messages
rlm.completion([
    {"role": "user", "content": "Message 1"},
    {"role": "assistant", "content": "Response 1"},
], "Query?")
```

### Reset Between Queries

```python
rlm = RLM_REPL()

# First query
answer1 = rlm.completion(context1, query1)

# Reset state
rlm.reset()

# Fresh query (no contamination)
answer2 = rlm.completion(context2, query2)
```

### Adjust Iteration Limit

```python
# Quick tasks
rlm = RLM_REPL(max_iterations=5)

# Complex tasks
rlm = RLM_REPL(max_iterations=30)
```

## ✨ New Features (v2.0)

### 🚀 Async Execution & Parallel Queries

RLMs now support **parallel LLM queries** for dramatic speed improvements:

```python
# OLD WAY - Sequential (slow)
results = []
for chunk in chunks:
    result = llm_query(f"Process: {chunk}")
    results.append(result)

# NEW WAY - Parallel (much faster!)
prompts = [f"Process: {chunk}" for chunk in chunks]
results = llm_query_batch(prompts)  # All at once!
```

**Available functions:**
- `llm_query(prompt)` - Synchronous single query
- `llm_query_batch(prompts)` - Parallel batch queries (recommended!)
- `llm_query_async(prompt)` - Async single query (for await)
- `llm_query_batch_async(prompts)` - Async batch queries

### 🔄 Depth > 1 Recursion

Sub-RLMs can now spawn their own RLMs for **nested recursive reasoning**:

```python
# Enable depth > 1
rlm = RLM_REPL(
    model="gpt-4o",
    max_depth=2,  # Allow nested RLM calls!
)

# Now sub-RLMs can recursively call other RLMs
# Useful for hierarchical data processing
```

**Depth levels:**
- `max_depth=1` (default): Sub-LLMs only, no recursion
- `max_depth=2`: Sub-RLMs can spawn their own sub-LLMs
- `max_depth=3+`: Deeper nesting for complex hierarchies

**Use cases:**
- Hierarchical document structures (company → dept → team)
- Multi-level summarization
- Recursive problem decomposition
- Tree-structured data processing

## 🧪 Testing New Features

Run the test suite:

```bash
python test_async_depth.py
```

Tests include:
1. **Batch Execution** - Parallel LLM query performance
2. **Async Execution** - Async/await syntax in REPL
3. **Depth=2 Recursion** - Nested RLM calls
4. **Depth=3 Recursion** - Deep nesting

## 🚧 Current Limitations

1. **No Prefix Caching**: Each call is independent (future optimization)
2. **OpenAI Only**: Other providers not yet supported
3. **Thread-based Async**: Uses executor, not true async LLM calls (future: native async)

## 🔮 Future Extensions

- **True Async LLM Calls**: Native async OpenAI client
- **Prefix Caching**: Reuse common context prefixes
- **Multi-Provider**: Anthropic, local models
- **Streaming**: Real-time execution feedback
- **RL Training**: Learn optimal exploration strategies
- **Visualization**: Interactive trajectory viewer

## 📖 Research Paper

**Recursive Language Models**
Alex Zhang, Omar Khattab
MIT CSAIL, October 2025
https://alexzhang13.github.io/blog/2025/rlm/

### Key Insights from Paper

> "RLMs are designed on the principle that fundamentally, LMs should decide how to break down problems to be digestible for an LM."

> "If tomorrow, the best frontier LM can handle 10M tokens, then an RLM can handle 100M tokens (maybe at half the cost)."

## 🤝 Contributing

This is a clean-room implementation for educational purposes. Feel free to:
- Report issues
- Suggest improvements
- Add new examples
- Extend functionality

## 📄 License

MIT License - See original research paper for citation.

## 🙏 Acknowledgments

- **Alex Zhang & Omar Khattab** - Original RLM research
- **MIT CSAIL** - Research institution
- **OpenAI** - API infrastructure

## 📞 Citation

```bibtex
@article{zhang2025rlm,
  title   = "Recursive Language Models",
  author  = "Zhang, Alex and Khattab, Omar",
  year    = "2025",
  month   = "October",
  url     = "https://alexzhang13.github.io/blog/2025/rlm/"
}
```

---

**Built with ❤️ to explore the future of long-context AI**
