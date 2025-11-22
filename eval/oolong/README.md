# OOLONG Benchmark Integration

This document explains how to evaluate RLM on the OOLONG benchmark for long-context aggregation tasks.

## What is OOLONG?

**OOLONG** (Bertsch et al., 2025) is a challenging benchmark designed to test how well large language models handle long-context aggregation and reasoning tasks.

### Key Features
- Tests contexts with **128k+ tokens**
- Evaluates **aggregation and reasoning** capabilities
- Exposes **"context rot"** - performance degradation with long contexts
- Two datasets: **Oolong-synth** (synthetic) and **Oolong-real** (natural data)

### Why RLMs Excel on OOLONG

Traditional LLMs struggle with long contexts even within their stated limits:
- GPT-5 solves **<10% of tasks** with 75k+ tokens
- Performance degrades as context grows

RLMs solve this by:
- **Storing context as data** (not in the prompt)
- **Programmatic exploration** instead of direct reading
- **Recursive sub-queries** for semantic tasks

**Reported Results:**
- RLM(GPT-4o-mini) **beats GPT-5** by +33% on OOLONG
- **Cheaper** per query despite recursive calls
- Maintains performance even at extreme context lengths

---

## Installation

### 1. Install Base Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install OOLONG Dependencies

```bash
pip install -r requirements-oolong.txt
```

This installs:
- `datasets` - HuggingFace datasets library
- `transformers` - For tokenization
- `tiktoken` - OpenAI tokenizer
- `jsonlines` - Result formatting

### 3. Set Up API Key

```bash
export OPENAI_API_KEY='sk-your-key-here'
```

Or add to `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

---

## Quick Start

### Run a Quick Test

Test the integration without downloading full datasets:

```bash
python eval/oolong/test_integration.py
```

This runs 2 tests:
1. Basic RLM functionality
2. OOLONG format compatibility

### Evaluate on OOLONG-Synth

```bash
python eval/oolong/eval.py --dataset synth --max-examples 10
```

### Evaluate on OOLONG-Real

```bash
python eval/oolong/eval.py --dataset real --max-examples 10
```

---

## Usage Guide

### Basic Evaluation

```bash
python eval/oolong/eval.py \
  --dataset synth \
  --model gpt-4o \
  --recursive-model gpt-4o-mini \
  --max-iterations 15 \
  --output results.jsonl
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset to use: `synth` or `real` | (required) |
| `--model` | Root LM model | `gpt-4o` |
| `--recursive-model` | Recursive LM model | `gpt-4o-mini` |
| `--max-iterations` | Max RLM iterations | `15` |
| `--max-examples` | Limit number of examples | All |
| `--output` | Output file (JSONL) | Auto-generated |
| `--enable-logging` | Show detailed RLM logs | False |

### Examples

**Quick test (10 examples):**
```bash
python eval/oolong/eval.py --dataset synth --max-examples 10
```

**Full evaluation with logging:**
```bash
python eval/oolong/eval.py \
  --dataset real \
  --enable-logging \
  --output oolong_real_full.jsonl
```

**Budget evaluation (cheaper model):**
```bash
python eval/oolong/eval.py \
  --dataset synth \
  --model gpt-4o-mini \
  --recursive-model gpt-4o-mini \
  --max-iterations 10
```

---

## Understanding Results

### Output Format

The evaluation script produces two files:

1. **Results file** (`*.jsonl`) - One JSON object per example:
   ```json
   {
     "id": "synth_001",
     "context_window_id": "cw_001",
     "dataset": "synth",
     "model": "rlm(gpt-4o,gpt-4o-mini)",
     "attempted_parse": "42",
     "parse_confidence": 1.0,
     "full_answer": "The answer is 42.",
     "score": 1,
     "context_len": 128000,
     "task_group": "aggregation",
     "task": "count",
     "answer_type": "number",
     "answer": "42"
   }
   ```

2. **Summary file** (`*_summary.json`) - Overall statistics:
   ```json
   {
     "dataset": "synth",
     "model": "rlm(gpt-4o,gpt-4o-mini)",
     "total_examples": 100,
     "correct": 85,
     "accuracy": 0.85,
     "total_cost_usd": 12.45,
     "total_tokens": 245000,
     "total_calls": 150,
     "timestamp": "2025-11-22T08:30:00"
   }
   ```

### Interpreting Accuracy

- **Score**: Binary (0 or 1) per example
- **Accuracy**: Percentage of correct answers
- **Context Length**: Average tokens per example

**Typical Performance Targets:**
- **OOLONG-synth**: 70-85% accuracy (RLM should excel here)
- **OOLONG-real**: 60-75% accuracy (more challenging)
- **Baseline (GPT-5)**: ~50% on long contexts
- **RLM advantage**: +20-30% over baselines

---

## Programmatic Usage

### Using RLM Directly

```python
from rlm import RLM_REPL

# Create RLM
rlm = RLM_REPL(
    model="gpt-4o",
    recursive_model="gpt-4o-mini",
    max_iterations=15,
    enable_logging=False,
    track_costs=True,
)

# Call RLM with context and query
context = "<long context here>"
query = "What is the answer?"

response = rlm.completion(context=context, query=query)
print(response)

# Check costs
costs = rlm.cost_summary()
print(f"Cost: ${costs['estimated_cost_usd']:.4f}")
```

### Processing Multiple Examples

```python
from rlm import RLM_REPL

# Create RLM
rlm = RLM_REPL(model="gpt-4o", recursive_model="gpt-4o-mini")

# Multiple examples from OOLONG dataset
examples = [
    {"context": "...", "query": "..."},
    {"context": "...", "query": "..."},
]

# Process each example
responses = []
for example in examples:
    rlm.reset()  # Reset state between examples
    response = rlm.completion(
        context=example["context"],
        query=example["query"]
    )
    responses.append(response)
```

---

## Architecture

### How the Integration Works

```
OOLONG Dataset
    ↓
Load examples from HuggingFace
    ↓
Extract context and query from examples
    ↓
RLM_REPL.completion(context, query)
    ↓
Root LM explores context programmatically
    ↓
Returns answer
    ↓
Score and save results
```

### Data Extraction

**OOLONG example format:**
```python
example = {
    "context_window_text": "<context>",
    "question": "<question>",
    "answer": "<expected answer>"
}
```

**RLM format:**
```python
rlm.completion(
    context=example["context_window_text"],
    query=example["question"]
)
```

---

## Cost Optimization

### Model Selection

**For accuracy:**
```bash
--model gpt-4o --recursive-model gpt-4o-mini
```

**For budget:**
```bash
--model gpt-4o-mini --recursive-model gpt-4o-mini
```

**For speed:**
```bash
--max-iterations 10  # Fewer iterations
```

### Estimated Costs

Approximate costs per example (varies by context length):

| Configuration | Cost/Example | Speed | Accuracy |
|--------------|--------------|-------|----------|
| gpt-4o + gpt-4o-mini | $0.10-0.30 | Slow | High |
| gpt-4o-mini + gpt-4o-mini | $0.01-0.05 | Fast | Good |

**Total costs for full dataset:**
- OOLONG-synth (1000 examples): ~$50-150
- OOLONG-real (500 examples): ~$30-100

---

## Troubleshooting

### Dataset Not Found

**Error:** `DatasetNotFoundError: oolongbench/oolong-synth`

**Solution:** The dataset is loaded from HuggingFace. Ensure you have internet connection and `datasets` library installed:
```bash
pip install datasets
```

### API Rate Limits

**Error:** `RateLimitError` from OpenAI

**Solution:** Add delays or reduce batch size:
```python
import time
time.sleep(1)  # Between examples
```

### Out of Memory

**Error:** Large contexts cause memory issues

**Solution:**
- Use `--max-examples` to limit evaluation
- Increase system memory
- Process smaller context windows

### Low Accuracy

**Possible causes:**
1. **Insufficient iterations:** Try `--max-iterations 20`
2. **Weak model:** Use `--model gpt-4o`
3. **Answer parsing issues:** Check `full_answer` in results

---

## Advanced Usage

### Custom Response Processing

Modify `eval_oolong.py` to customize answer extraction:

```python
def process_response_custom(response: str, answer: Any) -> Dict[str, Any]:
    # Custom parsing logic
    import re
    numbers = re.findall(r'\d+', response)
    if numbers and str(answer) in numbers:
        return {"score": 1, ...}
    return {"score": 0, ...}
```

### Integration with Other Benchmarks

RLM can be used with any benchmark by extracting context and query:

```python
from rlm import RLM_REPL

# Create RLM
rlm = RLM_REPL(model="gpt-4o", recursive_model="gpt-4o-mini")

# Use with custom datasets
my_context = "..."
my_question = "..."

response = rlm.completion(context=my_context, query=my_question)
```

---

## Comparing with Baselines

### Running Baseline Models

To compare RLM with baseline models, you can use OOLONG's original evaluation script with LiteLLM:

```bash
# From oolong repository
python src/eval/eval_script_batched.py \
  --model gpt-4o \
  --dataset synth
```

### Expected Comparisons

| Model | OOLONG-Synth | OOLONG-Real |
|-------|--------------|-------------|
| GPT-4o | ~50% | ~45% |
| GPT-5 | ~55% | ~50% |
| RLM(GPT-4o-mini) | **~75%** | **~65%** |
| RLM(GPT-4o) | **~85%** | **~75%** |

*(These are approximate based on research paper claims)*

---

## References

- **OOLONG Paper:** Bertsch et al. (2025), arXiv:2511.02817
- **RLM Blog Post:** https://alexzhang13.github.io/blog/2025/rlm/
- **OOLONG Repository:** https://github.com/abertsch72/oolong
- **Datasets:** https://huggingface.co/oolongbench

---

## Contributing

To improve OOLONG integration:

1. **Better answer parsing:** The current implementation uses simple string matching. More sophisticated parsing could improve accuracy.

2. **Async processing:** Implement async/await for parallel evaluation of multiple examples.

3. **Caching:** Add prompt caching to reduce costs on repeated contexts.

4. **Additional metrics:** Track reasoning steps, token usage per task, etc.

See `eval/oolong/eval.py` for implementation details.

---

## Summary

The OOLONG integration allows you to:
- ✅ Evaluate RLM on standardized long-context benchmarks
- ✅ Compare RLM performance against baseline models
- ✅ Validate the research claims about RLM effectiveness
- ✅ Track costs and optimize model selection
- ✅ Generate reproducible benchmark results

**Quick start:**
```bash
pip install -r requirements-oolong.txt
python eval/oolong/test_integration.py  # Verify setup
python eval/oolong/eval.py --dataset synth --max-examples 10  # Run evaluation
```

Happy benchmarking! 🎉
