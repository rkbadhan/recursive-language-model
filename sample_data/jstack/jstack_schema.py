"""
JSON Execution Stack (jstack) Schema Definition and Validator

This module defines the data structures for RLM execution traces and provides
validation utilities for jstack JSON files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
import json


@dataclass
class LLMCall:
    """Information about a language model API call."""
    model: str
    input_tokens: int
    output_tokens: int
    temperature: float = 0.7


@dataclass
class SubCall:
    """Information about a recursive sub-call."""
    prompt: str
    model: str
    depth: int
    input_tokens: int
    output_tokens: int


@dataclass
class CodeExecutionResult:
    """Result of executing a code block in the REPL."""
    stdout: str
    stderr: str
    execution_time: float
    locals_created: List[str] = field(default_factory=list)
    locals_modified: List[str] = field(default_factory=list)
    error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ExecutionState:
    """State of the REPL environment at a given point."""
    conversation_step: int
    message_history_length: int
    repl_locals_count: int


@dataclass
class ExecutionFrame:
    """A single frame in the execution stack."""
    frame_id: str
    depth: int
    type: Literal["root_llm", "code_execution", "recursive_call"]
    iteration: int
    timestamp: str

    # Optional fields depending on frame type
    state: Optional[ExecutionState] = None
    llm_call: Optional[LLMCall] = None
    response: Optional[str] = None
    contains_code_blocks: Optional[bool] = None
    final_answer: Optional[str] = None
    code_block: Optional[str] = None
    result: Optional[CodeExecutionResult] = None
    sub_call: Optional[SubCall] = None


@dataclass
class ExecutionMetadata:
    """Metadata about the overall execution."""
    query: str
    model: str
    recursive_model: str
    max_depth: int
    max_iterations: int
    start_time: str
    end_time: str
    total_duration_seconds: float


@dataclass
class ExecutionSummary:
    """Summary statistics for the execution."""
    total_iterations: int
    total_frames: int
    total_llm_calls: int
    total_code_executions: int
    total_recursive_calls: int
    max_depth_reached: int
    final_answer: str
    success: bool
    total_tokens_used: int
    estimated_cost_usd: float
    errors_encountered: int = 0
    errors_recovered: int = 0


@dataclass
class JStack:
    """Complete JSON Execution Stack representation."""
    metadata: ExecutionMetadata
    execution_stack: List[ExecutionFrame]
    summary: ExecutionSummary


class JStackValidator:
    """Validator for jstack JSON files."""

    @staticmethod
    def validate_metadata(data: Dict[str, Any]) -> List[str]:
        """Validate metadata section."""
        errors = []
        required_fields = [
            "query", "model", "recursive_model", "max_depth",
            "max_iterations", "start_time", "end_time", "total_duration_seconds"
        ]

        for field_name in required_fields:
            if field_name not in data:
                errors.append(f"Missing required field in metadata: {field_name}")

        # Validate timestamps
        for ts_field in ["start_time", "end_time"]:
            if ts_field in data:
                try:
                    datetime.fromisoformat(data[ts_field].replace('Z', '+00:00'))
                except ValueError:
                    errors.append(f"Invalid timestamp format for {ts_field}: {data[ts_field]}")

        return errors

    @staticmethod
    def validate_frame(frame: Dict[str, Any], index: int) -> List[str]:
        """Validate a single execution frame."""
        errors = []
        required_fields = ["frame_id", "depth", "type", "iteration", "timestamp"]

        for field_name in required_fields:
            if field_name not in frame:
                errors.append(f"Frame {index}: Missing required field: {field_name}")

        # Validate frame type
        valid_types = ["root_llm", "code_execution", "recursive_call"]
        if frame.get("type") not in valid_types:
            errors.append(f"Frame {index}: Invalid type '{frame.get('type')}'. Must be one of {valid_types}")

        # Type-specific validation
        frame_type = frame.get("type")
        if frame_type == "root_llm":
            if "llm_call" not in frame:
                errors.append(f"Frame {index}: root_llm frame must have 'llm_call' field")
            if "response" not in frame:
                errors.append(f"Frame {index}: root_llm frame must have 'response' field")

        elif frame_type == "code_execution":
            if "code_block" not in frame:
                errors.append(f"Frame {index}: code_execution frame must have 'code_block' field")
            if "result" not in frame:
                errors.append(f"Frame {index}: code_execution frame must have 'result' field")
            elif "stdout" not in frame["result"] or "stderr" not in frame["result"]:
                errors.append(f"Frame {index}: result must have 'stdout' and 'stderr' fields")

        elif frame_type == "recursive_call":
            if "sub_call" not in frame:
                errors.append(f"Frame {index}: recursive_call frame must have 'sub_call' field")
            if "response" not in frame:
                errors.append(f"Frame {index}: recursive_call frame must have 'response' field")

        return errors

    @staticmethod
    def validate_summary(data: Dict[str, Any]) -> List[str]:
        """Validate summary section."""
        errors = []
        required_fields = [
            "total_iterations", "total_frames", "total_llm_calls",
            "total_code_executions", "total_recursive_calls",
            "max_depth_reached", "final_answer", "success",
            "total_tokens_used", "estimated_cost_usd"
        ]

        for field_name in required_fields:
            if field_name not in data:
                errors.append(f"Missing required field in summary: {field_name}")

        # Validate counts are non-negative
        count_fields = [
            "total_iterations", "total_frames", "total_llm_calls",
            "total_code_executions", "total_recursive_calls",
            "max_depth_reached", "total_tokens_used"
        ]
        for field_name in count_fields:
            if field_name in data and data[field_name] < 0:
                errors.append(f"Summary field '{field_name}' must be non-negative, got {data[field_name]}")

        return errors

    @staticmethod
    def validate(jstack_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a complete jstack JSON structure.

        Returns:
            (is_valid, errors): Tuple of validation result and list of error messages
        """
        errors = []

        # Check top-level structure
        if "metadata" not in jstack_data:
            errors.append("Missing 'metadata' section")
        else:
            errors.extend(JStackValidator.validate_metadata(jstack_data["metadata"]))

        if "execution_stack" not in jstack_data:
            errors.append("Missing 'execution_stack' section")
        elif not isinstance(jstack_data["execution_stack"], list):
            errors.append("'execution_stack' must be a list")
        else:
            for i, frame in enumerate(jstack_data["execution_stack"]):
                errors.extend(JStackValidator.validate_frame(frame, i))

        if "summary" not in jstack_data:
            errors.append("Missing 'summary' section")
        else:
            errors.extend(JStackValidator.validate_summary(jstack_data["summary"]))

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_file(file_path: str) -> tuple[bool, List[str]]:
        """
        Validate a jstack JSON file.

        Args:
            file_path: Path to the JSON file to validate

        Returns:
            (is_valid, errors): Tuple of validation result and list of error messages
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return JStackValidator.validate(data)
        except json.JSONDecodeError as e:
            return (False, [f"Invalid JSON: {str(e)}"])
        except FileNotFoundError:
            return (False, [f"File not found: {file_path}"])
        except Exception as e:
            return (False, [f"Error reading file: {str(e)}"])


def main():
    """Example usage and validation of sample jstack files."""
    import os
    import sys

    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all JSON files in the same directory
    json_files = [f for f in os.listdir(script_dir) if f.endswith('.json')]

    if not json_files:
        print("No JSON files found to validate.")
        return

    print("Validating jstack sample files...\n")
    all_valid = True

    for json_file in sorted(json_files):
        file_path = os.path.join(script_dir, json_file)
        print(f"Validating: {json_file}")

        is_valid, errors = JStackValidator.validate_file(file_path)

        if is_valid:
            print(f"  ✓ Valid\n")
        else:
            print(f"  ✗ Invalid - {len(errors)} error(s):")
            for error in errors:
                print(f"    - {error}")
            print()
            all_valid = False

    if all_valid:
        print("All jstack files are valid!")
        sys.exit(0)
    else:
        print("Some jstack files have validation errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
