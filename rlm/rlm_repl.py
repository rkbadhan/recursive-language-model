"""
Recursive Language Model with REPL Environment.

This module provides the main RLM implementation that uses a REPL environment
to handle unbounded context through iterative code execution and recursive
LLM calls.
"""

from typing import Dict, List, Optional, Any, Union
import os

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import (
    DEFAULT_QUERY,
    next_action_prompt,
    build_system_prompt
)
import rlm.utils.utils as utils
from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_REPL(RLM):
    """
    Recursive Language Model with REPL environment.

    This implementation allows language models to process arbitrarily long
    contexts by:
    1. Storing context in a REPL environment as variables
    2. Letting the root LM write code to explore the context
    3. Enabling recursive sub-LLM calls via llm_query()
    4. Building up the final answer iteratively
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        recursive_model: str = "gpt-4o-mini",
        max_iterations: int = 20,
        depth: int = 0,
        enable_logging: bool = False,
        track_costs: bool = False,
    ):
        """
        Initialize the RLM with REPL environment.

        Args:
            api_key: OpenAI API key (uses env var if None)
            model: Model for root LM (e.g., 'gpt-4o')
            recursive_model: Model for recursive sub-calls (e.g., 'gpt-4o-mini')
            max_iterations: Maximum number of root LM iterations
            depth: Recursion depth (0 = root, currently only supports depth=1)
            enable_logging: Whether to enable colorful logging
            track_costs: Whether to track API costs
        """
        # API configuration
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.recursive_model = recursive_model

        # Initialize root LLM client
        self.llm = OpenAIClient(
            api_key=self.api_key,
            model=model,
            track_costs=track_costs
        )

        # Configuration
        self.depth = depth
        self._max_iterations = max_iterations
        self.track_costs = track_costs

        # State
        self.repl_env: Optional[REPLEnv] = None
        self.messages: List[Dict[str, str]] = []
        self.query: Optional[str] = None

        # Logging
        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)

    def setup_context(
        self,
        context: Union[List[str], str, Dict, List[Dict[str, str]]],
        query: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Setup the context for the RLM.

        Args:
            context: The large context to analyze
            query: The user's question

        Returns:
            Initial message list with system prompt
        """
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)

        # Initialize conversation with system prompt
        self.messages = build_system_prompt()
        self.logger.log_initial_messages(self.messages)

        # Convert context for REPL
        context_data, context_str = utils.convert_context_for_repl(context)

        # Initialize REPL environment with context
        self.repl_env = REPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
        )

        return self.messages

    def completion(
        self,
        context: Union[List[str], str, Dict, List[Dict[str, str]]],
        query: Optional[str] = None
    ) -> str:
        """
        Process a query over potentially unbounded context.

        This is the main RLM interface - a drop-in replacement for LLM completion.

        Args:
            context: The context to process (can be very large)
            query: The user's question

        Returns:
            str: The final answer

        Example:
            >>> rlm = RLM_REPL(model="gpt-4o", recursive_model="gpt-4o-mini")
            >>> context = "..." * 1_000_000  # Huge context
            >>> answer = rlm.completion(context, "What is the magic number?")
        """
        # Setup context and initialize REPL
        self.messages = self.setup_context(context, query)

        # Main iterative loop
        for iteration in range(self._max_iterations):

            # Query root LM for next action
            prompt = self.messages + [next_action_prompt(query, iteration)]
            response = self.llm.completion(prompt)

            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(
                response,
                has_tool_calls=code_blocks is not None
            )

            # Process code execution or add assistant message
            if code_blocks is not None:
                # Execute code and add results to message history
                self.messages = utils.process_code_execution(
                    response,
                    self.messages,
                    self.repl_env,
                    self.repl_env_logger,
                    self.logger
                )
            else:
                # No code blocks - add as assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": "You responded with:\n" + response
                }
                self.messages.append(assistant_message)

            # Check for final answer
            final_answer = utils.check_for_final_answer(
                response,
                self.repl_env,
                self.logger
            )

            if final_answer:
                self.logger.log_final_response(final_answer)
                return final_answer

        # No final answer found - force one
        self.logger.log_tool_execution(
            "MAX_ITERATIONS_REACHED",
            f"Forcing final answer after {self._max_iterations} iterations"
        )

        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.logger.log_final_response(final_answer)

        return final_answer

    def cost_summary(self) -> Dict[str, Any]:
        """
        Get cost summary for this RLM execution.

        Returns:
            Dictionary with cost information

        Raises:
            NotImplementedError: If cost tracking is not enabled
        """
        if not self.track_costs:
            return {
                "error": "Cost tracking not enabled",
                "note": "Set track_costs=True when initializing RLM_REPL"
            }

        return self.llm.get_cost_summary()

    def reset(self) -> None:
        """
        Reset the RLM state for a fresh query.

        Clears:
        - REPL environment
        - Message history
        - Query state
        - Cost tracking (if enabled)
        """
        # Reset REPL environment
        if self.repl_env is not None:
            # Clean up old REPL
            try:
                del self.repl_env
            except:
                pass
            self.repl_env = None

        # Reset state
        self.messages = []
        self.query = None

        # Reset loggers
        self.repl_env_logger.clear()

        # Reset costs
        if self.track_costs:
            self.llm.reset_costs()


if __name__ == "__main__":
    # Example usage
    print("RLM_REPL implementation ready!")
    print("Usage:")
    print("  from rlm.rlm_repl import RLM_REPL")
    print("  rlm = RLM_REPL(model='gpt-4o', recursive_model='gpt-4o-mini')")
    print("  answer = rlm.completion(context='...', query='...')")
