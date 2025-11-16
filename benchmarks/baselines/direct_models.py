"""
Direct LLM baseline models.

These models call the LLM directly with the full context or truncated context.
"""

from typing import Optional
from rlm.utils.llm import OpenAIClient


class DirectGPT:
    """
    Direct GPT model baseline.

    Calls the LLM with the full context. If context exceeds model's
    context window, returns an error or truncates.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_context_tokens: int = 128000,
        api_key: Optional[str] = None
    ):
        """
        Initialize DirectGPT baseline.

        Args:
            model: OpenAI model name
            max_context_tokens: Maximum context tokens allowed
            api_key: OpenAI API key
        """
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = OpenAIClient(api_key=api_key, model=model)

    def __call__(self, context: str, query: str) -> str:
        """
        Answer query with full context.

        Args:
            context: Full context string
            query: User query

        Returns:
            Model's answer
        """
        # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
        estimated_tokens = (len(context) + len(query)) / 4

        if estimated_tokens > self.max_context_tokens:
            return f"ERROR: Context too large ({estimated_tokens:.0f} tokens > {self.max_context_tokens})"

        # Create prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Call LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.completion(messages=messages)
            return response
        except Exception as e:
            return f"ERROR: {str(e)}"


class DirectGPTMini:
    """
    Direct GPT-mini model baseline.

    Same as DirectGPT but uses a smaller/cheaper model (e.g., gpt-4o-mini).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_context_tokens: int = 128000,
        api_key: Optional[str] = None
    ):
        """
        Initialize DirectGPTMini baseline.

        Args:
            model: OpenAI model name
            max_context_tokens: Maximum context tokens allowed
            api_key: OpenAI API key
        """
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = OpenAIClient(api_key=api_key, model=model)

    def __call__(self, context: str, query: str) -> str:
        """
        Answer query with full context using smaller model.

        Args:
            context: Full context string
            query: User query

        Returns:
            Model's answer
        """
        # Estimate tokens
        estimated_tokens = (len(context) + len(query)) / 4

        if estimated_tokens > self.max_context_tokens:
            return f"ERROR: Context too large ({estimated_tokens:.0f} tokens > {self.max_context_tokens})"

        # Create prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Call LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.completion(messages=messages)
            return response
        except Exception as e:
            return f"ERROR: {str(e)}"


class DirectGPTTruncated:
    """
    Direct GPT with context truncation.

    Truncates context to fit within model's context window by taking
    the most recent tokens (which may be random for unstructured data).
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_context_tokens: int = 128000,
        api_key: Optional[str] = None
    ):
        """
        Initialize DirectGPTTruncated baseline.

        Args:
            model: OpenAI model name
            max_context_tokens: Maximum context tokens to use
            api_key: OpenAI API key
        """
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.client = OpenAIClient(api_key=api_key, model=model)

    def __call__(self, context: str, query: str) -> str:
        """
        Answer query with truncated context.

        Args:
            context: Full context string
            query: User query

        Returns:
            Model's answer
        """
        # Estimate tokens for query
        query_tokens = len(query) / 4
        available_tokens = self.max_context_tokens - query_tokens - 100  # Buffer

        # Truncate context if needed
        max_context_chars = int(available_tokens * 4)

        if len(context) > max_context_chars:
            # Truncate from the end (keep most recent)
            context = context[-max_context_chars:]

        # Create prompt
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Call LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.completion(messages=messages)
            return response
        except Exception as e:
            return f"ERROR: {str(e)}"
