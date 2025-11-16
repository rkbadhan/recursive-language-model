"""
BrowseComp-Plus Benchmark: Multi-document retrieval and reasoning.

This module implements the BrowseComp-Plus benchmark which evaluates
models on multi-hop queries over large document corpora.
"""

from .benchmark import BrowseCompPlusBenchmark
from .data_generator import generate_document_corpus, generate_multi_hop_query

__all__ = ['BrowseCompPlusBenchmark', 'generate_document_corpus', 'generate_multi_hop_query']
