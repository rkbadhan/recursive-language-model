"""
Benchmarks for evaluating Recursive Language Models.

This package contains implementations of various long-context benchmarks:
- OOLONG: Context rot benchmark with fine-grained reasoning tasks
- BrowseComp-Plus: Multi-document retrieval and reasoning benchmark
"""

from .oolong.benchmark import OOLONGBenchmark
from .browsecomp_plus.benchmark import BrowseCompPlusBenchmark
from .baselines import (
    DirectGPT,
    DirectGPTMini,
    DirectGPTTruncated,
    BM25Retriever,
    ReActAgent
)

__all__ = [
    'OOLONGBenchmark',
    'BrowseCompPlusBenchmark',
    'DirectGPT',
    'DirectGPTMini',
    'DirectGPTTruncated',
    'BM25Retriever',
    'ReActAgent'
]
