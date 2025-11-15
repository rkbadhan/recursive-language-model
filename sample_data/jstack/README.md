# JStack Sample Data

This directory contains sample JSON Execution Stack (jstack) files for validating and testing the Recursive Language Model (RLM) execution tracing system.

## What is JStack?

**JStack** (JSON Execution Stack) is a structured representation of the complete execution trace of an RLM session. It captures:

- **Metadata**: Query, model configuration, and timing information
- **Execution Stack**: Ordered sequence of execution frames (LLM calls, code executions, recursive calls)
- **Summary**: Aggregated statistics and final results

## Directory Contents

### Sample Data Files

1. **`simple_execution.json`**
   - Basic example with no recursion
   - Shows straightforward code execution and final answer
   - Use case: Sum of squares calculation

2. **`recursive_execution.json`**
   - Complex example with recursive sub-calls
   - Demonstrates parallel processing of chunks
   - Use case: Processing large text context with multiple recursive calls

3. **`error_handling.json`**
   - Example showing error detection and recovery
   - Demonstrates how errors are logged in the execution trace
   - Use case: Handling invalid input (negative factorial)

### Schema and Validation

4. **`jstack_schema.py`**
   - Python dataclass definitions for all jstack structures
   - `JStackValidator` class for validating jstack JSON files
   - Run as script to validate all sample files

## JStack Structure

```json
{
  "metadata": {
    "query": "User's input query",
    "model": "gpt-4o",
    "recursive_model": "gpt-4o-mini",
    "max_depth": 2,
    "max_iterations": 20,
    "start_time": "ISO8601 timestamp",
    "end_time": "ISO8601 timestamp",
    "total_duration_seconds": 15.2
  },
  "execution_stack": [
    {
      "frame_id": "unique_frame_identifier",
      "depth": 0,
      "type": "root_llm | code_execution | recursive_call",
      "iteration": 1,
      "timestamp": "ISO8601 timestamp",
      "state": { ... },
      "llm_call": { ... },
      "code_block": "...",
      "result": { ... },
      "sub_call": { ... }
    }
  ],
  "summary": {
    "total_iterations": 2,
    "total_frames": 3,
    "total_llm_calls": 2,
    "total_code_executions": 1,
    "total_recursive_calls": 0,
    "max_depth_reached": 0,
    "final_answer": "Result",
    "success": true,
    "total_tokens_used": 1235,
    "estimated_cost_usd": 0.0062
  }
}
```

## Frame Types

### 1. Root LLM Frame
Represents a call to the root language model.

**Required fields:**
- `state`: Current REPL environment state
- `llm_call`: Model configuration and token usage
- `response`: Text response from the model
- `contains_code_blocks`: Boolean indicating presence of code

**Optional fields:**
- `final_answer`: Extracted final answer if present

### 2. Code Execution Frame
Represents execution of a code block in the REPL.

**Required fields:**
- `code_block`: The Python code that was executed
- `result`: Execution result containing stdout, stderr, timing, and variable changes

**Result structure:**
```json
{
  "stdout": "Output text",
  "stderr": "Error text if any",
  "execution_time": 0.003,
  "locals_created": ["var1", "var2"],
  "locals_modified": ["var3"],
  "error": false,
  "error_type": "ValueError",
  "error_message": "Description"
}
```

### 3. Recursive Call Frame
Represents a recursive call to a sub-LM.

**Required fields:**
- `sub_call`: Information about the recursive call
- `response`: Text response from the sub-model

**Sub-call structure:**
```json
{
  "prompt": "Prompt sent to sub-model",
  "model": "gpt-4o-mini",
  "depth": 1,
  "input_tokens": 2000,
  "output_tokens": 150
}
```

## Validation

### Using the Python Validator

```bash
# Validate all sample files
cd sample_data/jstack
python jstack_schema.py
```

### Programmatic Validation

```python
from jstack_schema import JStackValidator

# Validate a file
is_valid, errors = JStackValidator.validate_file('simple_execution.json')

if is_valid:
    print("Valid jstack file!")
else:
    for error in errors:
        print(f"Error: {error}")

# Validate a Python dict
import json
with open('simple_execution.json') as f:
    data = json.load(f)

is_valid, errors = JStackValidator.validate(data)
```

## Use Cases

### 1. Testing and Validation
- Validate RLM execution tracing implementation
- Test jstack serialization/deserialization
- Verify execution flow correctness

### 2. Debugging
- Analyze execution traces for debugging
- Identify performance bottlenecks
- Track token usage and costs

### 3. Analytics
- Aggregate execution statistics
- Calculate success rates
- Monitor model performance

### 4. Documentation
- Generate execution reports
- Create execution visualizations
- Document LLM behavior patterns

## Adding New Sample Data

When creating new jstack sample files:

1. Follow the structure defined in `jstack_schema.py`
2. Use descriptive `frame_id` values (e.g., `frame_0`, `frame_0_1`, `frame_1`)
3. Ensure timestamps are in ISO8601 format with 'Z' suffix
4. Include realistic token counts and timing data
5. Validate using `python jstack_schema.py`

## Integration with RLM

To integrate jstack logging into the RLM system:

```python
from rlm.rlm_repl import RLM_REPL
from sample_data.jstack.jstack_schema import JStack, ExecutionFrame

# During RLM execution, collect frames
frames = []

# Add frames as execution progresses
frame = ExecutionFrame(
    frame_id=f"frame_{iteration}",
    depth=current_depth,
    type="root_llm",
    iteration=iteration,
    timestamp=datetime.utcnow().isoformat() + 'Z',
    # ... other fields
)
frames.append(frame)

# At the end, create complete jstack
jstack = JStack(
    metadata=metadata,
    execution_stack=frames,
    summary=summary
)
```

## License

This sample data is part of the Recursive Language Model project.
