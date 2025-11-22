"""
RLM Interfaces - Different ways to interact with Recursive Language Models.

This package provides various interfaces for RLM:
- chat_completion: Standard chat completion API (OpenAI/Anthropic-style messages)
"""

from .chat_completion import (
    RLMChatCompletionClient,
    create_chat_completion_client,
    # Backward compatibility
    RLMOolongAdapter,
    create_oolong_compatible_model,
)

__all__ = [
    "RLMChatCompletionClient",
    "create_chat_completion_client",
    "RLMOolongAdapter",
    "create_oolong_compatible_model",
]
