"""
Recursive Language Models (RLM) - A framework for processing unbounded context.

This package provides an implementation of Recursive Language Models that can
handle arbitrarily long contexts by treating them as programmable variables in
a REPL environment.
"""

from .rlm import RLM
from .rlm_repl import RLM_REPL

# Chat completion interface (optional - works without extra dependencies)
try:
    from .interfaces import (
        RLMChatCompletionClient,
        create_chat_completion_client,
        # Backward compatibility
        RLMOolongAdapter,
        create_oolong_compatible_model,
    )
    __all__ = [
        "RLM",
        "RLM_REPL",
        "RLMChatCompletionClient",
        "create_chat_completion_client",
        "RLMOolongAdapter",
        "create_oolong_compatible_model",
    ]
except ImportError:
    # Interfaces module not available
    __all__ = ["RLM", "RLM_REPL"]

__version__ = "0.2.0"
