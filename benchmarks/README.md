# RLM Benchmarks

This directory contains benchmark implementations for evaluating Recursive Language Models (RLMs) on long-context tasks.

## 📊 Available Benchmarks

### 1. OOLONG Benchmark

**Purpose:** Evaluates long-context reasoning over fine-grained information.

The OOLONG benchmark (`trec_coarse` split) tests models on distributional queries over large contexts with thousands of entries. Queries require semantic understanding and counting/aggregation.

**Example Query:**
```
For the following question, only consider the subset of instances that are
associated with user IDs 1234, 5678, 9101. Among instances associated with
these users, how many data points should be classified as label 'entity'?
Give your final answer in the form 'Answer: number'.
```

**Key Characteristics:**
- Context: ~5000 entries (~128k-263k tokens)
- Tasks: Classification + counting
- Scoring: Continuous scoring metric for numerical answers
- Challenge: Context rot - performance degrades with context length

### 2. BrowseComp-Plus Benchmark

**Purpose:** Evaluates multi-document retrieval and reasoning.

This benchmark presents queries that require finding and combining information from multiple documents in a large corpus.

**Example Query:**
```
What product did TechCorp launch in 2024, and who was the CEO at that time?
```

**Key Characteristics:**
- Context: 10-1000 documents (~10k-10M+ tokens)
- Tasks: Multi-hop reasoning across documents
- Scoring: Contains-answer and exact-match metrics
- Challenge: Finding relevant information in massive corpuses

## 🚀 Quick Start

### Basic Usage

```bash
# Run OOLONG benchmark with RLM(GPT-4o-mini)
python benchmarks/run_benchmarks.py \
    --benchmark oolong \
    --run-rlm \
    --num-queries 5

# Run BrowseComp-Plus with multiple models
python benchmarks/run_benchmarks.py \
    --benchmark browsecomp \
    --run-rlm-gpt4 \
    --run-direct-gpt \
    --run-react \
    --num-queries 10 \
    --num-documents 100

# Run both benchmarks with all models (expensive!)
python benchmarks/run_benchmarks.py \
    --benchmark both \
    --run-all \
    --save-results
```

### Model Options

Available models for comparison:

- `--run-rlm`: RLM(GPT-4o-mini) with recursion
- `--run-rlm-gpt4`: RLM(GPT-4o) with recursion
- `--run-rlm-no-recursion`: RLM(GPT-4o) without recursive calls (ablation)
- `--run-direct-gpt`: Direct GPT-4o call with full context
- `--run-direct-gpt-mini`: Direct GPT-4o-mini call
- `--run-direct-gpt-truncated`: Direct GPT-4o with context truncation
- `--run-direct-gpt-bm25`: Direct GPT-4o with BM25 pre-retrieval
- `--run-react`: ReAct agent with GPT-4o + BM25 retrieval

## 📈 Results from Paper

### OOLONG (128k+ tokens)

| Model | Avg Score | Cost per Query |
|-------|-----------|---------------|
| **RLM(GPT-4o-mini)** | **0.667** | Similar to GPT-4o |
| GPT-4o | 0.500 | Baseline |
| GPT-4o-mini | 0.450 | Lower |
| ReAct + GPT-4o + BM25 | 0.300 | Higher |

**Key Finding:** RLM(GPT-4o-mini) outperforms GPT-4o by +33% while maintaining similar API costs!

### BrowseComp-Plus (1000 docs)

| Model | Accuracy (1000 docs) |
|-------|---------------------|
| **RLM(GPT-4o)** | **100%** |
| RLM(GPT-4o) No-Recursion | 90% |
| Direct GPT-4o + BM25 | 60% |
| ReAct + GPT-4o + BM25 | 50% |
| Direct GPT-4o (fits) | 100% (10 docs only) |

**Key Finding:** RLM maintains perfect performance at scale while other approaches degrade significantly!

## 🔧 Analyzing Results

```bash
# Analyze a single benchmark run
python benchmarks/analyze_results.py \
    benchmarks/results/oolong_20250116_143022.json

# Compare multiple runs
python benchmarks/analyze_results.py \
    benchmarks/results/oolong_*.json

# Export to CSV
python benchmarks/analyze_results.py \
    benchmarks/results/browsecomp_20250116_143022.json \
    --export-csv results.csv
```

## 📁 Directory Structure

```
benchmarks/
├── __init__.py                 # Package exports
├── utils.py                    # Common utilities
├── run_benchmarks.py           # Main runner script
├── analyze_results.py          # Results analysis
├── README.md                   # This file
│
├── oolong/                     # OOLONG benchmark
│   ├── __init__.py
│   ├── benchmark.py            # Benchmark implementation
│   └── data_generator.py      # Synthetic data generation
│
├── browsecomp_plus/            # BrowseComp-Plus benchmark
│   ├── __init__.py
│   ├── benchmark.py            # Benchmark implementation
│   └── data_generator.py      # Synthetic data generation
│
├── baselines/                  # Baseline models
│   ├── __init__.py
│   ├── direct_models.py        # Direct GPT models
│   ├── retrieval.py            # BM25 retriever
│   └── react_agent.py          # ReAct agent
│
└── results/                    # Saved results (JSON)
    ├── oolong_*.json
    └── browsecomp_*.json
```

## 🧪 Custom Benchmark Usage

### OOLONG

```python
from benchmarks.oolong import OOLONGBenchmark
from rlm.rlm_repl import RLM_REPL

# Create benchmark
benchmark = OOLONGBenchmark(
    num_queries=10,
    entries_per_query=5000
)

# Create model wrapper
rlm = RLM_REPL(model="gpt-4o-mini", enable_logging=False)

def model_fn(context, query):
    rlm.reset()
    return rlm.completion(context, query)

# Run evaluation
results = benchmark.evaluate(model_fn, model_name="RLM(GPT-4o-mini)")

print(f"Average Score: {results['avg_score']:.3f}")
```

### BrowseComp-Plus

```python
from benchmarks.browsecomp_plus import BrowseCompPlusBenchmark

# Create benchmark
benchmark = BrowseCompPlusBenchmark(
    num_queries=20,
    num_documents=100
)

# Run with same model function
results = benchmark.evaluate(model_fn, model_name="RLM(GPT-4o-mini)")

print(f"Average Score: {results['avg_score']:.3f}")
print(f"Exact Match: {results['avg_exact_match']:.3f}")
```

### Comparing Multiple Models

```python
from benchmarks.baselines import DirectGPT, ReActAgent

# Create model wrappers
models = {
    "RLM(GPT-4o-mini)": model_fn,
    "Direct GPT-4o": DirectGPT(model="gpt-4o"),
    "ReAct + BM25": ReActAgent(model="gpt-4o")
}

# Compare on same queries
comparison = benchmark.compare_models(models, num_queries=10)

# Results automatically printed and returned
```

## 💡 Tips

### For OOLONG:
- Start with 5-10 queries for quick testing
- Use 20+ queries for meaningful comparisons
- Context length scales with `entries_per_query`
- Watch for context rot effects at 128k+ tokens

### For BrowseComp-Plus:
- Start with 10-50 documents
- Scale up to 100-1000 documents to see RLM advantages
- Evidence documents are always included in corpus
- Multi-hop queries require combining multiple documents

### Cost Management:
- Use `--run-rlm` (GPT-4o-mini) for cost-effective evaluation
- Limit `--num-queries` during development
- Save results with `--save-results` to avoid re-running

## 🔬 Research Applications

These benchmarks are designed to evaluate:

1. **Context Rot**: How does performance degrade with context length?
2. **Scalability**: Can models handle 100k+ token contexts effectively?
3. **Cost Efficiency**: What's the cost/performance tradeoff?
4. **Reasoning Depth**: Do recursive approaches help on complex queries?

## 📚 References

1. **OOLONG Paper**: [Link to OOLONG paper]
2. **BrowseComp**: [Link to BrowseComp paper]
3. **RLM Blog Post**: https://alexzhang13.github.io/blog/2025/rlm/

## 🤝 Contributing

To add a new benchmark:

1. Create directory under `benchmarks/your_benchmark/`
2. Implement `Benchmark` class with `evaluate()` and `compare_models()`
3. Add data generator if needed
4. Update `run_benchmarks.py` to include new benchmark
5. Add documentation to this README

---

**Built to demonstrate RLM's advantages on long-context tasks!**
