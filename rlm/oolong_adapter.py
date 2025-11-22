"""
OOLONG Benchmark Adapter for RLM.

This module provides an adapter to evaluate RLM on the OOLONG benchmark.
It converts between OOLONG's message format and RLM's context/query interface.
"""

from typing import List, Dict, Any, Optional
from rlm.rlm_repl import RLM_REPL


class RLMOolongAdapter:
    """
    Adapter to make RLM compatible with OOLONG benchmark evaluation.

    OOLONG expects models to work with message-based APIs (like LiteLLM),
    but RLM uses a context/query interface. This adapter bridges the gap.

    Usage:
        adapter = RLMOolongAdapter(
            model="gpt-4o",
            recursive_model="gpt-4o-mini"
        )
        response = adapter.completion(messages=[...])
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        recursive_model: str = "gpt-4o-mini",
        max_iterations: int = 15,
        max_depth: int = 1,
        enable_logging: bool = False,
        track_costs: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the RLM adapter for OOLONG.

        Args:
            model: Root LM model name (e.g., "gpt-4o")
            recursive_model: Model for recursive sub-calls (e.g., "gpt-4o-mini")
            max_iterations: Maximum reasoning iterations
            max_depth: Maximum recursion depth
            enable_logging: Enable detailed execution logging
            track_costs: Track API costs
            api_key: OpenAI API key (or use env var)
        """
        self.rlm = RLM_REPL(
            model=model,
            recursive_model=recursive_model,
            max_iterations=max_iterations,
            depth=0,
            max_depth=max_depth,
            enable_logging=enable_logging,
            track_costs=track_costs,
            api_key=api_key,
        )

        self.model_name = f"rlm({model},{recursive_model})"

    def completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate completion compatible with OOLONG evaluation.

        OOLONG format:
            messages = [
                {"role": "system", "content": "<context>"},
                {"role": "user", "content": "<question>"}
            ]

        RLM format:
            rlm.completion(context="<context>", query="<question>")

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (ignored for RLM)

        Returns:
            Answer string from RLM
        """
        # Extract context and query from messages
        context = ""
        query = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                # System message contains the context
                context = content
            elif role == "user":
                # User message contains the question
                query = content
            elif role == "assistant":
                # Some formats might include assistant messages
                # We'll append them to context as conversation history
                if context:
                    context += f"\n\nAssistant: {content}"
                else:
                    context = f"Assistant: {content}"

        # Handle edge cases
        if not query and context:
            # If no explicit user message, treat last message as query
            query = context
            context = ""

        # Reset RLM state for fresh evaluation
        self.rlm.reset()

        # Run RLM completion
        try:
            answer = self.rlm.completion(context=context, query=query)
            return answer
        except Exception as e:
            # Return error message if RLM fails
            return f"RLM_ERROR: {str(e)}"

    def batch_completion(
        self,
        message_list: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """
        Generate completions for multiple examples.

        Note: Currently processes sequentially. Could be parallelized
        using async RLM calls in the future.

        Args:
            message_list: List of message lists
            **kwargs: Additional parameters

        Returns:
            List of answer strings
        """
        results = []

        for messages in message_list:
            result = self.completion(messages, **kwargs)
            results.append(result)

        return results

    def cost_summary(self) -> Dict[str, Any]:
        """
        Get cost summary from RLM.

        Returns:
            Dict with cost statistics
        """
        if self.rlm.track_costs:
            return self.rlm.cost_summary()
        return {}

    def reset(self):
        """Reset RLM state between evaluations."""
        self.rlm.reset()


def create_oolong_compatible_model(
    model: str = "gpt-4o",
    recursive_model: str = "gpt-4o-mini",
    **kwargs
) -> RLMOolongAdapter:
    """
    Factory function to create an OOLONG-compatible RLM instance.

    This is the main entry point for OOLONG evaluation scripts.

    Example:
        model = create_oolong_compatible_model(
            model="gpt-4o",
            recursive_model="gpt-4o-mini",
            enable_logging=False
        )

        response = model.completion(messages=[
            {"role": "system", "content": "Context: ..."},
            {"role": "user", "content": "Question: ..."}
        ])

    Args:
        model: Root LM model
        recursive_model: Recursive LM model
        **kwargs: Additional RLM parameters

    Returns:
        RLMOolongAdapter instance
    """
    return RLMOolongAdapter(
        model=model,
        recursive_model=recursive_model,
        **kwargs
    )
