# Long-Context Benchmarks Implementation

This document describes the implementation of OOLONG and BrowseComp-Plus benchmarks for evaluating RLMs.

## Overview

We've implemented two key benchmarks from the RLM paper:

1. **OOLONG**: Tests context rot and fine-grained reasoning
2. **BrowseComp-Plus**: Tests multi-document retrieval and reasoning

Both benchmarks come with:
- Synthetic data generators (for reproducibility without needing original datasets)
- Multiple baseline implementations
- Evaluation harnesses
- Result analysis tools

## Architecture

### Benchmark Structure

Each benchmark follows a consistent pattern:

```python
class Benchmark:
    def generate_query() -> Dict:
        """Generate a single test query with context and ground truth"""

    def evaluate_single(model_fn, example) -> Dict:
        """Evaluate model on one query"""

    def evaluate(model_fn, num_queries) -> Dict:
        """Evaluate model on multiple queries"""

    def compare_models(models, num_queries) -> Dict:
        """Compare multiple models on same queries"""
```

### Baseline Models

All baselines implement the same interface:

```python
def model_fn(context: str, query: str) -> str:
    """Take context and query, return answer"""
```

This allows fair comparison across:
- RLM variants
- Direct LLM calls
- Retrieval-augmented approaches
- Agent-based methods

## Implementation Details

### OOLONG Benchmark

**Data Generation:**
- Generates ~5000 synthetic TREC-style question entries
- Each entry has: Date, User ID, Question instance
- Questions are automatically classified into categories (entity, description, etc.)
- Queries ask to count entries matching specific criteria

**Example:**
```
Entry: Date: Jan 15, 2023 || User: 12345 || Instance: Who invented the telephone?
Query: Among user IDs [12345, 67890, ...], how many are classified as 'entity'?
```

**Scoring:**
- Uses continuous scoring for numerical answers
- Tolerance-based scoring (within 10% = full credit, gradual decay)
- Handles both exact numbers and approximate answers

**Key Features:**
- Configurable context length (via entries_per_query)
- Automatic ground truth tracking
- Supports context lengths up to 263k+ tokens

### BrowseComp-Plus Benchmark

**Data Generation:**
- Generates synthetic documents about companies, products, people
- Multi-hop queries require combining information across documents
- Evidence documents contain fragments of the answer
- Hard negative documents added for difficulty

**Example:**
```
Query: What product did TechCorp launch in 2024, and who was the CEO?
Answer: Requires finding (1) product launch doc and (2) CEO doc
```

**Scoring:**
- Primary metric: Contains-answer (lenient, checks if answer substring present)
- Secondary metric: Exact match (strict)
- Both case-insensitive

**Key Features:**
- Scalable from 10 to 1000+ documents
- Always includes evidence documents in corpus
- Configurable document corpus size

### Baseline Implementations

**1. DirectGPT / DirectGPTMini**
- Simple LLM call with full context
- Returns error if context exceeds window
- No special processing

**2. DirectGPTTruncated**
- Truncates context to fit window
- Keeps most recent tokens (may lose answer!)
- Simulates context window limitations

**3. DirectGPTWithBM25**
- Pre-retrieves top-K documents using BM25
- Passes only retrieved docs to LLM
- Simulates retrieval-first approach

**4. ReActAgent**
- Iterative agent with Search action
- Uses BM25 for retrieval
- Can search multiple times
- Requires more API calls

**5. RLM Variants**
- RLM(GPT-4o-mini): Uses mini model recursively
- RLM(GPT-4o): Uses GPT-4o + mini for sub-calls
- RLM without recursion: Ablation study

## Usage Examples

### Quick Start

```bash
# Run OOLONG with just RLM
python benchmarks/run_benchmarks.py \
    --benchmark oolong \
    --run-rlm \
    --num-queries 5

# Compare all models on BrowseComp-Plus
python benchmarks/run_benchmarks.py \
    --benchmark browsecomp \
    --run-all \
    --num-queries 10 \
    --num-documents 100 \
    --save-results
```

### Programmatic Usage

```python
from benchmarks.oolong import OOLONGBenchmark
from benchmarks.baselines import DirectGPT
from rlm.rlm_repl import RLM_REPL

# Create benchmark
benchmark = OOLONGBenchmark(num_queries=5, entries_per_query=5000)

# Create models
rlm = RLM_REPL(model="gpt-4o-mini", enable_logging=False)
direct = DirectGPT(model="gpt-4o")

# Compare
models = {
    "RLM(GPT-4o-mini)": lambda ctx, q: rlm.completion(ctx, q),
    "Direct GPT-4o": direct
}

results = benchmark.compare_models(models)
print(results)
```

### Analyzing Results

```bash
# View results
python benchmarks/analyze_results.py \
    benchmarks/results/oolong_20250116_*.json

# Export to CSV
python benchmarks/analyze_results.py \
    benchmarks/results/browsecomp_*.json \
    --export-csv comparison.csv
```

## Expected Results

Based on the RLM paper:

### OOLONG (128k tokens)
- **RLM(GPT-4o-mini)**: ~0.67 avg score (2x better than baseline)
- **GPT-4o**: ~0.50 avg score
- **GPT-4o-mini**: ~0.45 avg score (context rot)
- **ReAct + BM25**: ~0.30 avg score (retrieval difficult)

### BrowseComp-Plus (1000 docs)
- **RLM(GPT-4o)**: 100% accuracy (perfect!)
- **RLM(GPT-4o) No-Recursion**: ~90% accuracy
- **Direct GPT-4o + BM25**: ~60% accuracy
- **ReAct + BM25**: ~50% accuracy

## Cost Considerations

**Estimated costs per query:**

OOLONG (128k context):
- RLM(GPT-4o-mini): $0.05 - $0.10
- GPT-4o: $0.10 - $0.15
- RLM(GPT-4o): $0.08 - $0.12

BrowseComp-Plus (100 docs):
- RLM(GPT-4o): $0.20 - $0.40
- Direct GPT-4o: $0.15 - $0.25 (when it fits)
- ReAct: $0.30 - $0.60 (multiple searches)

**Tips to reduce costs:**
1. Use `--run-rlm` (GPT-4o-mini) for initial testing
2. Limit `--num-queries` during development
3. Start with fewer documents/entries
4. Save results to avoid re-running

## Extensibility

### Adding a New Benchmark

1. Create directory: `benchmarks/your_benchmark/`

2. Implement benchmark class:
```python
class YourBenchmark:
    def generate_query(self) -> Dict:
        # Return {'context': ..., 'query': ..., 'answer': ...}
        pass

    def evaluate_single(self, model_fn, example) -> Dict:
        # Call model and score
        pass

    def evaluate(self, model_fn, num_queries) -> Dict:
        # Run multiple queries
        pass
```

3. Add to `run_benchmarks.py`:
```python
from benchmarks.your_benchmark import YourBenchmark

def run_your_benchmark(args):
    benchmark = YourBenchmark()
    # ... rest of implementation
```

4. Update documentation

### Adding a New Baseline

1. Create model class in `benchmarks/baselines/`:
```python
class YourModel:
    def __call__(self, context: str, query: str) -> str:
        # Implement your approach
        pass
```

2. Export from `benchmarks/baselines/__init__.py`

3. Add to `run_benchmarks.py` argparse options

## Technical Notes

### BM25 Implementation
- Simple token-based (word splitting)
- Standard parameters: k1=1.5, b=0.75
- No stemming or stopwords (for simplicity)
- Could be enhanced with proper NLP preprocessing

### ReAct Agent
- Maximum 5 iterations by default
- Retrieves 5 documents per search
- Simple thought-action-observation loop
- Could be enhanced with better prompting

### RLM Integration
- Uses existing `RLM_REPL` implementation
- Disables logging during benchmarks (for speed)
- Resets state between queries
- Cost tracking enabled for all models

## Troubleshooting

**"Context too large" errors:**
- Reduce `--entries-per-query` or `--num-documents`
- Use truncated variants
- Use RLM (designed for large contexts)

**Low scores on OOLONG:**
- Check if model is extracting numbers correctly
- Verify continuous scoring tolerance
- Try more iterations for RLM

**Low scores on BrowseComp-Plus:**
- Synthetic data may have different patterns than real data
- Try adjusting scoring (exact vs contains)
- Check if evidence docs are being found

**High costs:**
- Start with fewer queries (--num-queries 3)
- Use smaller contexts initially
- Monitor with cost tracking

## Future Enhancements

Potential improvements:

1. **Real Datasets**: Integrate actual OOLONG/BrowseComp data
2. **Better Scoring**: More sophisticated metrics
3. **Caching**: Save generated queries for reproducibility
4. **Visualization**: Plot performance vs context length
5. **Streaming**: Real-time progress indicators
6. **Parallelization**: Run queries in parallel
7. **More Baselines**: Add Claude, Gemini, local models

## References

- **OOLONG Paper**: [Citation needed]
- **BrowseComp**: [Citation needed]
- **RLM Blog**: https://alexzhang13.github.io/blog/2025/rlm/
- **Original Implementation**: This is a clean-room re-implementation

---

**Questions or issues?** Open an issue on GitHub or check the main README.
