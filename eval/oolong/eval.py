#!/usr/bin/env python3
"""
OOLONG Benchmark Evaluation Script for RLM.

This script evaluates Recursive Language Models on the OOLONG benchmark,
which tests long-context aggregation and reasoning capabilities.

Usage:
    python eval/oolong/eval.py --dataset synth --output results.jsonl
    python eval/oolong/eval.py --dataset real --model gpt-4o --max-examples 100

Requirements:
    pip install datasets transformers tiktoken jsonlines
"""

import argparse
import json
import jsonlines
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from rlm import RLM_REPL


def load_oolong_dataset(
    dataset_name: str,
    split: str = "test",
    min_context_len: Optional[int] = None,
    max_context_len: Optional[int] = None,
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load OOLONG dataset from Hugging Face.

    Args:
        dataset_name: "synth" or "real"
        split: Dataset split (default: "test")
        min_context_len: Minimum context length filter
        max_context_len: Maximum context length filter
        max_examples: Limit number of examples

    Returns:
        List of dataset examples
    """
    print(f"\n{'='*80}")
    print(f"Loading OOLONG-{dataset_name} dataset...")
    print(f"{'='*80}\n")

    # Map dataset names to HuggingFace identifiers
    dataset_map = {
        "synth": ("oolongbench/oolong-synth", None),
        "real": ("oolongbench/oolong-real", "dnd"),
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'synth' or 'real'")

    hf_name, config = dataset_map[dataset_name]

    # Load dataset
    if config:
        dataset = load_dataset(hf_name, config, split=split)
    else:
        dataset = load_dataset(hf_name, split=split)

    examples = list(dataset)

    # Filter by context length
    if min_context_len is not None or max_context_len is not None:
        filtered = []
        for ex in examples:
            ctx_len = ex.get("context_len", 0)
            if min_context_len and ctx_len < min_context_len:
                continue
            if max_context_len and ctx_len > max_context_len:
                continue
            filtered.append(ex)
        examples = filtered

    # Limit number of examples
    if max_examples:
        examples = examples[:max_examples]

    print(f"Loaded {len(examples)} examples")
    if examples:
        avg_len = sum(ex.get("context_len", 0) for ex in examples) / len(examples)
        print(f"Average context length: {avg_len:,.0f} tokens")

    return examples


def process_response_synth(response: str, answer: Any) -> Dict[str, Any]:
    """
    Process response for synthetic dataset.

    Args:
        response: Model output
        answer: Expected answer

    Returns:
        Dict with parsing and scoring results
    """
    # Simple exact match for now
    # TODO: Implement more sophisticated parsing from oolong
    response_clean = response.strip().lower()
    answer_str = str(answer).strip().lower()

    # Try to find the answer in the response
    is_correct = answer_str in response_clean

    return {
        "attempted_parse": response_clean,
        "parse_confidence": 1.0 if is_correct else 0.0,
        "score": 1 if is_correct else 0,
        "full_answer": response,
    }


def process_response_real(response: str, answer: Any) -> Dict[str, Any]:
    """
    Process response for real dataset.

    Args:
        response: Model output
        answer: Expected answer

    Returns:
        Dict with parsing and scoring results
    """
    # For DND dataset, answer matching may be more complex
    # Using simple approach for now
    response_clean = response.strip().lower()
    answer_str = str(answer).strip().lower()

    is_correct = answer_str in response_clean

    return {
        "attempted_parse": response_clean,
        "parse_confidence": 1.0 if is_correct else 0.0,
        "score": 1 if is_correct else 0,
        "full_answer": response,
    }


def evaluate_rlm_on_oolong(
    dataset_name: str,
    model: str = "gpt-4o",
    recursive_model: str = "gpt-4o-mini",
    max_iterations: int = 15,
    max_examples: Optional[int] = None,
    output_file: Optional[str] = None,
    enable_logging: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate RLM on OOLONG benchmark.

    Args:
        dataset_name: "synth" or "real"
        model: Root LM model
        recursive_model: Recursive LM model
        max_iterations: Max RLM iterations
        max_examples: Limit number of examples
        output_file: Path to save results (JSONL)
        enable_logging: Show detailed RLM logs

    Returns:
        Evaluation results summary
    """
    # Load dataset
    examples = load_oolong_dataset(
        dataset_name=dataset_name,
        max_examples=max_examples
    )

    if not examples:
        print("No examples to evaluate!")
        return {}

    # Create RLM
    print(f"\n{'='*80}")
    print(f"Initializing RLM...")
    print(f"  Root model: {model}")
    print(f"  Recursive model: {recursive_model}")
    print(f"  Max iterations: {max_iterations}")
    print(f"{'='*80}\n")

    rlm = RLM_REPL(
        model=model,
        recursive_model=recursive_model,
        max_iterations=max_iterations,
        enable_logging=enable_logging,
        track_costs=True,
    )

    # Choose response processor
    response_processor = (
        process_response_synth if dataset_name == "synth"
        else process_response_real
    )

    # Evaluate each example
    results = []
    correct_count = 0

    print(f"\n{'='*80}")
    print(f"Evaluating {len(examples)} examples...")
    print(f"{'='*80}\n")

    for i, example in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Evaluating example {example.get('id', 'unknown')}")
        print(f"  Context length: {example.get('context_len', 0):,} tokens")
        print(f"  Task: {example.get('task', 'unknown')}")

        # Extract context and query directly from example
        context = example.get("context_window_text_with_labels") or \
                  example.get("context_window_text", "")
        query = example.get("question", "")

        # Get RLM response
        try:
            rlm.reset()  # Reset RLM state for fresh evaluation
            response = rlm.completion(context=context, query=query)
        except Exception as e:
            print(f"  ERROR: {e}")
            response = f"ERROR: {str(e)}"

        # Process and score response
        expected_answer = example.get("answer")
        result = response_processor(response, expected_answer)

        # Build result record
        result_record = {
            "id": example.get("id"),
            "context_window_id": example.get("context_window_id"),
            "dataset": dataset_name,
            "model": f"rlm({model},{recursive_model})",
            "attempted_parse": result["attempted_parse"],
            "parse_confidence": result["parse_confidence"],
            "full_answer": result["full_answer"],
            "score": result["score"],
            "context_len": example.get("context_len"),
            "task_group": example.get("task_group"),
            "task": example.get("task"),
            "answer_type": example.get("answer_type"),
            "answer": expected_answer,
        }

        results.append(result_record)
        correct_count += result["score"]

        print(f"  Expected: {expected_answer}")
        print(f"  Got: {result['attempted_parse'][:100]}...")
        print(f"  Score: {result['score']}")
        print(f"  Running accuracy: {correct_count}/{i} ({100*correct_count/i:.1f}%)")

    # Calculate summary statistics
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0

    summary = {
        "dataset": dataset_name,
        "model": f"rlm({model},{recursive_model})",
        "total_examples": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "timestamp": datetime.now().isoformat(),
    }

    # Add cost summary if available
    if rlm.track_costs:
        cost_info = rlm.cost_summary()
        summary["total_cost_usd"] = cost_info.get("estimated_cost_usd", 0)
        summary["total_tokens"] = cost_info.get("total_tokens", 0)
        summary["total_calls"] = cost_info.get("total_calls", 0)

    # Save results
    if output_file:
        print(f"\n{'='*80}")
        print(f"Saving results to {output_file}...")
        print(f"{'='*80}\n")

        with jsonlines.open(output_file, mode='w') as writer:
            for result in results:
                writer.write(result)

        # Also save summary
        summary_file = output_file.replace('.jsonl', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")

    # Print final summary
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nDataset: OOLONG-{dataset_name}")
    print(f"Model: RLM({model}, {recursive_model})")
    print(f"Total examples: {total}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.1%}")

    if "total_cost_usd" in summary:
        print(f"\nCost Summary:")
        print(f"  Total cost: ${summary['total_cost_usd']:.4f}")
        print(f"  Total tokens: {summary['total_tokens']:,}")
        print(f"  Total API calls: {summary['total_calls']}")
        print(f"  Cost per example: ${summary['total_cost_usd']/total:.4f}")

    print(f"\n{'='*80}\n")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate RLM on OOLONG benchmark"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synth", "real"],
        required=True,
        help="OOLONG dataset to evaluate on"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Root LM model (default: gpt-4o)"
    )

    parser.add_argument(
        "--recursive-model",
        type=str,
        default="gpt-4o-mini",
        help="Recursive LM model (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum RLM iterations (default: 15)"
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (default: all)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSONL format)"
    )

    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Enable detailed RLM execution logging"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in a .env file or export it:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return

    # Set default output file if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"oolong_{args.dataset}_rlm_{timestamp}.jsonl"

    # Run evaluation
    evaluate_rlm_on_oolong(
        dataset_name=args.dataset,
        model=args.model,
        recursive_model=args.recursive_model,
        max_iterations=args.max_iterations,
        max_examples=args.max_examples,
        output_file=args.output,
        enable_logging=args.enable_logging,
    )


if __name__ == "__main__":
    main()
