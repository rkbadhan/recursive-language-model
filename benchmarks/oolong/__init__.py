"""
OOLONG Benchmark: Long-context reasoning over fine-grained information.

This module implements the OOLONG benchmark (trec_coarse split) which evaluates
long-context reasoning tasks over distributional queries.
"""

from .benchmark import OOLONGBenchmark
from .data_generator import generate_oolong_query

__all__ = ['OOLONGBenchmark', 'generate_oolong_query']
