"""
LLM Client wrapper for OpenAI API.

This module provides a unified interface for interacting with language models,
abstracting away API-specific details and providing consistent error handling.
"""

import os
import asyncio
from typing import Optional, List, Dict, Union, Any
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class OpenAIClient:
    """
    Wrapper for OpenAI API that provides a simplified interface for completions.

    This client handles:
    - API authentication
    - Message format normalization
    - Error handling and retries
    - Optional cost tracking
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        track_costs: bool = False
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY env var
            model: Model identifier (e.g., 'gpt-4o', 'gpt-4o-mini')
            track_costs: Whether to track token usage and costs

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

        # Cost tracking
        self.track_costs = track_costs
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

        # Model pricing (per 1M tokens) - update as needed
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }

    def completion(
        self,
        messages: Union[str, Dict[str, str], List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        timeout: int = 300,
        **kwargs
    ) -> str:
        """
        Generate a completion from the language model.

        Args:
            messages: Input messages. Can be:
                - str: Single user message
                - Dict: Single message dict with 'role' and 'content'
                - List[Dict]: Full conversation history
            max_tokens: Maximum tokens to generate (None = model default)
            temperature: Sampling temperature (0.0 to 2.0)
            timeout: Request timeout in seconds
            **kwargs: Additional arguments to pass to the API

        Returns:
            str: The model's response content

        Raises:
            RuntimeError: If the API call fails
        """
        try:
            # Normalize message format
            normalized_messages = self._normalize_messages(messages)

            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=normalized_messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs
            )

            # Track usage if enabled
            if self.track_costs and hasattr(response, 'usage'):
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.total_calls += 1

            # Extract and return content
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")

    def _normalize_messages(
        self,
        messages: Union[str, Dict[str, str], List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Normalize various message formats to List[Dict[str, str]].

        Args:
            messages: Input in various formats

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if isinstance(messages, str):
            # Single string -> user message
            return [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            # Single dict -> wrap in list
            return [messages]
        elif isinstance(messages, list):
            # Already a list -> return as-is
            return messages
        else:
            raise ValueError(f"Unsupported message format: {type(messages)}")

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of API costs for this client.

        Returns:
            Dictionary with cost information:
            {
                'total_calls': int,
                'input_tokens': int,
                'output_tokens': int,
                'total_tokens': int,
                'estimated_cost': float (in USD),
                'model': str
            }
        """
        if not self.track_costs:
            return {
                "error": "Cost tracking is not enabled for this client",
                "model": self.model
            }

        # Calculate estimated cost
        input_cost = 0.0
        output_cost = 0.0

        if self.model in self.pricing:
            input_cost = (
                self.total_input_tokens / 1_000_000
            ) * self.pricing[self.model]["input"]
            output_cost = (
                self.total_output_tokens / 1_000_000
            ) * self.pricing[self.model]["output"]

        return {
            "total_calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
            "breakdown": {
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4),
            },
            "model": self.model,
        }

    def reset_costs(self) -> None:
        """Reset all cost tracking counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0


class AsyncOpenAIClient:
    """
    Async wrapper for OpenAI API that provides true async/await support.

    This client handles:
    - API authentication
    - Message format normalization
    - Error handling and retries
    - Optional cost tracking
    - True async I/O with AsyncOpenAI
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        track_costs: bool = False
    ):
        """
        Initialize the async OpenAI client.

        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY env var
            model: Model identifier (e.g., 'gpt-4o', 'gpt-4o-mini')
            track_costs: Whether to track token usage and costs

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Cost tracking (with lock for thread safety)
        self.track_costs = track_costs
        self._cost_lock = asyncio.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

        # Model pricing (per 1M tokens) - update as needed
        self.pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }

    async def completion(
        self,
        messages: Union[str, Dict[str, str], List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        timeout: int = 300,
        **kwargs
    ) -> str:
        """
        Generate a completion from the language model (async).

        Args:
            messages: Input messages. Can be:
                - str: Single user message
                - Dict: Single message dict with 'role' and 'content'
                - List[Dict]: Full conversation history
            max_tokens: Maximum tokens to generate (None = model default)
            temperature: Sampling temperature (0.0 to 2.0)
            timeout: Request timeout in seconds
            **kwargs: Additional arguments to pass to the API

        Returns:
            str: The model's response content

        Raises:
            RuntimeError: If the API call fails
        """
        try:
            # Normalize message format
            normalized_messages = self._normalize_messages(messages)

            # Make async API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=normalized_messages,
                max_completion_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs
            )

            # Track usage if enabled (thread-safe)
            if self.track_costs and hasattr(response, 'usage'):
                async with self._cost_lock:
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                    self.total_calls += 1

            # Extract and return content
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")

    async def completion_batch(
        self,
        messages_list: List[Union[str, Dict[str, str], List[Dict[str, str]]]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        timeout: int = 300,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple completions in parallel (true async).

        Args:
            messages_list: List of message inputs
            max_tokens: Maximum tokens to generate per completion
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            **kwargs: Additional arguments to pass to the API

        Returns:
            List of response strings in the same order as inputs
        """
        tasks = [
            self.completion(
                messages=msg,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs
            )
            for msg in messages_list
        ]

        # Run all requests concurrently
        results = await asyncio.gather(*tasks)
        return list(results)

    def _normalize_messages(
        self,
        messages: Union[str, Dict[str, str], List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Normalize various message formats to List[Dict[str, str]].

        Args:
            messages: Input in various formats

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if isinstance(messages, str):
            # Single string -> user message
            return [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            # Single dict -> wrap in list
            return [messages]
        elif isinstance(messages, list):
            # Already a list -> return as-is
            return messages
        else:
            raise ValueError(f"Unsupported message format: {type(messages)}")

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of API costs for this client.

        Returns:
            Dictionary with cost information
        """
        if not self.track_costs:
            return {
                "error": "Cost tracking is not enabled for this client",
                "model": self.model
            }

        # Calculate estimated cost
        input_cost = 0.0
        output_cost = 0.0

        if self.model in self.pricing:
            input_cost = (
                self.total_input_tokens / 1_000_000
            ) * self.pricing[self.model]["input"]
            output_cost = (
                self.total_output_tokens / 1_000_000
            ) * self.pricing[self.model]["output"]

        return {
            "total_calls": self.total_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
            "breakdown": {
                "input_cost_usd": round(input_cost, 4),
                "output_cost_usd": round(output_cost, 4),
            },
            "model": self.model,
        }

    def reset_costs(self) -> None:
        """Reset all cost tracking counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    async def close(self):
        """Close the async client connection."""
        await self.client.close()
