"""
BrowseComp-Plus Benchmark implementation.

Evaluates models on multi-document retrieval and reasoning tasks.
"""

import time
from typing import Dict, Any, List, Callable
from ..utils import contains_answer_score, exact_match_score, BenchmarkLogger
from .data_generator import generate_full_browsecomp_example


class BrowseCompPlusBenchmark:
    """
    BrowseComp-Plus benchmark for multi-document reasoning evaluation.

    This benchmark tests models' ability to answer multi-hop queries
    over large document corpora.
    """

    def __init__(
        self,
        num_queries: int = 20,
        num_documents: int = 100,
        num_evidence_docs: int = 3,
        verbose: bool = True
    ):
        """
        Initialize BrowseComp-Plus benchmark.

        Args:
            num_queries: Number of queries to evaluate
            num_documents: Number of documents per query
            num_evidence_docs: Number of documents containing evidence
            verbose: Whether to print progress
        """
        self.num_queries = num_queries
        self.num_documents = num_documents
        self.num_evidence_docs = num_evidence_docs
        self.logger = BenchmarkLogger(verbose=verbose)

    def generate_query(self) -> Dict[str, Any]:
        """
        Generate a single BrowseComp-Plus query.

        Returns:
            Dictionary with context, query, and ground truth
        """
        example = generate_full_browsecomp_example(
            num_documents=self.num_documents,
            num_evidence_docs=self.num_evidence_docs
        )

        self.logger.info(
            f"Generated query with {example['num_documents']} documents, "
            f"context length: {example['context_length']:,} chars"
        )

        return example

    def evaluate_single(
        self,
        model_fn: Callable[[str, str], str],
        example: Dict[str, Any],
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a single query with the given model.

        Args:
            model_fn: Function that takes (context, query) and returns answer
            example: Generated BrowseComp-Plus example
            model_name: Name of the model for logging

        Returns:
            Evaluation results dictionary
        """
        context = example['context']
        query = example['query']
        ground_truth = example['answer']

        self.logger.info(f"Evaluating {model_name} on query...")
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Ground truth: {ground_truth}")

        try:
            # Time the model
            start_time = time.time()
            answer = model_fn(context, query)
            elapsed_time = time.time() - start_time

            self.logger.info(f"Model answer: {answer}")

            # Score using contains_answer (more lenient for multi-hop)
            score = contains_answer_score(answer, ground_truth, case_sensitive=False)

            # Also compute exact match for comparison
            exact_score = exact_match_score(answer, ground_truth, case_sensitive=False)

            self.logger.info(f"Contains Score: {score:.2f}, Exact Match: {exact_score:.2f}")

            return {
                'context_length': example['context_length'],
                'num_documents': example['num_documents'],
                'ground_truth': ground_truth,
                'predicted': answer,
                'score': score,
                'exact_match': exact_score,
                'time': elapsed_time,
                'success': True
            }

        except Exception as e:
            self.logger.error(f"Error evaluating {model_name}: {str(e)}")
            return {
                'context_length': example['context_length'],
                'num_documents': example['num_documents'],
                'ground_truth': ground_truth,
                'predicted': None,
                'score': 0.0,
                'exact_match': 0.0,
                'time': 0.0,
                'success': False,
                'error': str(e)
            }

    def evaluate(
        self,
        model_fn: Callable[[str, str], str],
        model_name: str = "Model",
        num_queries: int = None
    ) -> Dict[str, Any]:
        """
        Evaluate a model on multiple BrowseComp-Plus queries.

        Args:
            model_fn: Function that takes (context, query) and returns answer
            model_name: Name of the model for logging
            num_queries: Number of queries to run (default: self.num_queries)

        Returns:
            Aggregated results dictionary
        """
        if num_queries is None:
            num_queries = self.num_queries

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting BrowseComp-Plus Benchmark: {model_name}")
        self.logger.info(f"Number of queries: {num_queries}")
        self.logger.info(f"Documents per query: {self.num_documents}")
        self.logger.info(f"{'='*80}\n")

        results = []
        total_score = 0.0
        total_exact = 0.0
        total_time = 0.0

        for i in range(num_queries):
            self.logger.info(f"\n--- Query {i+1}/{num_queries} ---")

            # Generate query
            example = self.generate_query()

            # Evaluate
            result = self.evaluate_single(model_fn, example, model_name)

            results.append(result)
            total_score += result['score']
            total_exact += result['exact_match']
            total_time += result['time']

            self.logger.info(
                f"Query {i+1} - Contains Score: {result['score']:.2f}, "
                f"Exact Match: {result['exact_match']:.2f}"
            )

        # Aggregate results
        avg_score = total_score / num_queries
        avg_exact = total_exact / num_queries
        successful_queries = sum(1 for r in results if r['success'])

        summary = {
            'model_name': model_name,
            'num_queries': num_queries,
            'num_documents_per_query': self.num_documents,
            'total_score': total_score,
            'avg_score': avg_score,
            'avg_exact_match': avg_exact,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / num_queries,
            'total_time': total_time,
            'avg_time_per_query': total_time / num_queries,
            'results': results
        }

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"BrowseComp-Plus Benchmark Results: {model_name}")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Average Contains Score: {avg_score:.3f}")
        self.logger.info(f"Average Exact Match: {avg_exact:.3f}")
        self.logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        self.logger.info(f"Avg Time per Query: {summary['avg_time_per_query']:.2f}s")
        self.logger.info(f"{'='*80}\n")

        return summary

    def compare_models(
        self,
        models: Dict[str, Callable[[str, str], str]],
        num_queries: int = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same BrowseComp-Plus queries.

        Args:
            models: Dictionary mapping model names to model functions
            num_queries: Number of queries to run

        Returns:
            Comparison results
        """
        if num_queries is None:
            num_queries = self.num_queries

        # Generate queries once
        self.logger.info("Generating queries for comparison...")
        queries = [self.generate_query() for _ in range(num_queries)]

        # Evaluate each model
        all_results = {}

        for model_name, model_fn in models.items():
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Evaluating: {model_name}")
            self.logger.info(f"{'='*80}")

            model_results = []
            total_score = 0.0
            total_exact = 0.0
            total_time = 0.0

            for i, example in enumerate(queries):
                self.logger.info(f"\nQuery {i+1}/{num_queries}")

                result = self.evaluate_single(model_fn, example, model_name)
                model_results.append(result)
                total_score += result['score']
                total_exact += result['exact_match']
                total_time += result['time']

            # Store aggregated results
            avg_score = total_score / num_queries
            avg_exact = total_exact / num_queries
            successful = sum(1 for r in model_results if r['success'])

            all_results[model_name] = {
                'avg_score': avg_score,
                'avg_exact_match': avg_exact,
                'total_score': total_score,
                'success_rate': successful / num_queries,
                'avg_time': total_time / num_queries,
                'total_time': total_time,
                'results': model_results
            }

        # Print comparison table
        self.logger.info(f"\n{'='*80}")
        self.logger.info("BrowseComp-Plus Benchmark Comparison")
        self.logger.info(f"{'='*80}")
        self.logger.info(
            f"{'Model':<30} {'Avg Score':>12} {'Exact Match':>12} {'Success %':>12} {'Avg Time':>12}"
        )
        self.logger.info(f"{'-'*80}")

        for model_name, results in all_results.items():
            self.logger.info(
                f"{model_name:<30} "
                f"{results['avg_score']:>12.3f} "
                f"{results['avg_exact_match']:>12.3f} "
                f"{results['success_rate']*100:>11.1f}% "
                f"{results['avg_time']:>11.2f}s"
            )

        self.logger.info(f"{'='*80}\n")

        return {
            'num_queries': num_queries,
            'num_documents': self.num_documents,
            'queries': queries,
            'results': all_results
        }
