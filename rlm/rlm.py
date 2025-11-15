"""
Base abstraction for Recursive Language Models.

This module defines the interface that all RLM implementations must follow,
ensuring consistent API across different strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List


class RLM(ABC):
    """
    Abstract base class for Recursive Language Models.

    An RLM is a thin wrapper around a language model that can spawn recursive
    LM calls for intermediate computation. From the user's perspective, it
    provides the same interface as a standard LM call.
    """

    @abstractmethod
    def completion(
        self,
        context: Union[str, List[str], Dict[str, Any], List[Dict[str, str]]],
        query: str
    ) -> str:
        """
        Process a query over a potentially large context.

        This is the main interface for RLM. It should be a drop-in replacement
        for standard LM completion calls like: llm.completion(prompt).

        Args:
            context: The context to process. Can be:
                - str: Plain text context
                - List[str]: List of text chunks
                - Dict: Structured data (e.g., JSON)
                - List[Dict]: List of messages or structured data
            query: The user's question or instruction

        Returns:
            str: The final answer from the RLM

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        pass

    @abstractmethod
    def cost_summary(self) -> Dict[str, Any]:
        """
        Get a summary of API costs incurred during RLM execution.

        Returns:
            Dict containing cost information, e.g.:
            {
                'total_cost': float,
                'input_tokens': int,
                'output_tokens': int,
                'num_calls': int,
                'breakdown': {...}
            }

        Raises:
            NotImplementedError: If cost tracking is not implemented
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the RLM state, clearing conversation history and REPL environment.

        This should prepare the RLM for a fresh query, clearing:
        - Message history
        - REPL environment state
        - Cost tracking counters
        - Any cached data

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        pass
