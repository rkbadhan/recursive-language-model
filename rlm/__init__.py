"""
Recursive Language Models (RLM) - A framework for processing unbounded context.

This package provides an implementation of Recursive Language Models that can
handle arbitrarily long contexts by treating them as programmable variables in
a REPL environment.
"""

from .rlm import RLM

__version__ = "0.1.0"
__all__ = ["RLM"]
