"""
Common utilities for benchmark evaluation.
"""

import re
import json
from typing import Dict, Any, List, Union
from datetime import datetime


def extract_number_from_answer(answer: str) -> Union[float, None]:
    """
    Extract a number from a model's answer.

    Handles formats like:
    - "Answer: 42"
    - "The answer is 42"
    - "42"
    - "approximately 42.5"

    Args:
        answer: The model's response string

    Returns:
        Extracted number or None if not found
    """
    # Try to find "Answer: number" format first
    match = re.search(r'(?:Answer|answer):\s*(-?\d+(?:\.\d+)?)', answer)
    if match:
        return float(match.group(1))

    # Try to find any number in the text
    match = re.search(r'(-?\d+(?:\.\d+)?)', answer)
    if match:
        return float(match.group(1))

    return None


def continuous_score(predicted: float, actual: float, tolerance: float = 0.1) -> float:
    """
    Compute continuous scoring metric for numerical answers.

    Used in OOLONG benchmark for counting problems.

    Args:
        predicted: Model's predicted number
        actual: Ground truth number
        tolerance: Relative tolerance (default 10%)

    Returns:
        Score between 0 and 1
    """
    if actual == 0:
        return 1.0 if predicted == 0 else 0.0

    relative_error = abs(predicted - actual) / abs(actual)

    if relative_error <= tolerance:
        return 1.0
    elif relative_error <= 2 * tolerance:
        # Linear decay
        return 1.0 - (relative_error - tolerance) / tolerance
    else:
        return 0.0


def exact_match_score(predicted: str, actual: str, case_sensitive: bool = False) -> float:
    """
    Compute exact match score for string answers.

    Args:
        predicted: Model's answer
        actual: Ground truth answer
        case_sensitive: Whether to compare case-sensitively

    Returns:
        1.0 if match, 0.0 otherwise
    """
    if not case_sensitive:
        predicted = predicted.lower().strip()
        actual = actual.lower().strip()
    else:
        predicted = predicted.strip()
        actual = actual.strip()

    return 1.0 if predicted == actual else 0.0


def contains_answer_score(predicted: str, actual: str, case_sensitive: bool = False) -> float:
    """
    Check if the predicted answer contains the actual answer.

    Args:
        predicted: Model's answer
        actual: Ground truth answer
        case_sensitive: Whether to compare case-sensitively

    Returns:
        1.0 if actual is in predicted, 0.0 otherwise
    """
    if not case_sensitive:
        predicted = predicted.lower()
        actual = actual.lower()

    return 1.0 if actual in predicted else 0.0


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save benchmark results to JSON file.

    Args:
        results: Dictionary containing benchmark results
        output_path: Path to save results
    """
    results['timestamp'] = datetime.now().isoformat()

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def load_results(input_path: str) -> Dict[str, Any]:
    """
    Load benchmark results from JSON file.

    Args:
        input_path: Path to results file

    Returns:
        Dictionary containing results
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple benchmark runs.

    Args:
        results: List of result dictionaries

    Returns:
        Aggregated statistics
    """
    if not results:
        return {}

    total_queries = sum(r.get('total_queries', 0) for r in results)
    total_correct = sum(r.get('correct', 0) for r in results)
    total_cost = sum(r.get('total_cost', 0.0) for r in results)
    total_time = sum(r.get('total_time', 0.0) for r in results)

    return {
        'num_runs': len(results),
        'total_queries': total_queries,
        'total_correct': total_correct,
        'accuracy': total_correct / total_queries if total_queries > 0 else 0.0,
        'avg_cost_per_query': total_cost / total_queries if total_queries > 0 else 0.0,
        'avg_time_per_query': total_time / total_queries if total_queries > 0 else 0.0,
        'total_cost': total_cost,
        'total_time': total_time,
    }


class BenchmarkLogger:
    """Simple logger for benchmark execution."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.logs = []

    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)

        if self.verbose:
            print(log_entry)

    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")

    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")

    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")

    def get_logs(self) -> List[str]:
        """Get all logged messages."""
        return self.logs
