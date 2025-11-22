# Evaluation and Benchmarking

This directory contains evaluation scripts and benchmarking tools for Recursive Language Models.

## Available Benchmarks

### OOLONG

Long-context aggregation and reasoning benchmark.

**Location:** `eval/oolong/`

**Quick start:**
```bash
# Install dependencies
pip install -r requirements-oolong.txt

# Run tests
python eval/oolong/test_integration.py

# Run evaluation
python eval/oolong/eval.py --dataset synth --max-examples 10
```

**Documentation:** [eval/oolong/README.md](oolong/README.md)

---

## Adding New Benchmarks

To add a new benchmark:

1. Create a subdirectory: `eval/your_benchmark/`
2. Add evaluation script: `eval.py`
3. Add tests: `test_integration.py`
4. Add documentation: `README.md`
5. Update this file with the new benchmark

## Structure

```
eval/
├── README.md (this file)
├── __init__.py
└── oolong/
    ├── __init__.py
    ├── eval.py
    ├── test_integration.py
    └── README.md
```
