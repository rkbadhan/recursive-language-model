"""
Recursive Language Models (RLM) - A framework for processing unbounded context.

This package provides an implementation of Recursive Language Models that can
handle arbitrarily long contexts by treating them as programmable variables in
a REPL environment.
"""

from .rlm import RLM
from .rlm_repl import RLM_REPL

# OOLONG benchmark integration (optional import)
try:
    from .oolong_adapter import RLMOolongAdapter, create_oolong_compatible_model
    __all__ = ["RLM", "RLM_REPL", "RLMOolongAdapter", "create_oolong_compatible_model"]
except ImportError:
    # OOLONG dependencies not installed
    __all__ = ["RLM", "RLM_REPL"]

__version__ = "0.2.0"
