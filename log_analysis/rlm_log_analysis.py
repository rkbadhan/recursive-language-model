"""
RLM Log Analysis - Specialized RLM for System Log Analysis

This module provides a specialized RLM implementation optimized for analyzing
system logs like jstack, strace, GC logs, pstack, and syslog with built-in
correlation and pattern detection capabilities.
"""

from typing import Dict, List, Optional, Any, Union
import os

from rlm import RLM
from .repl_log import LogAnalysisREPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import (
    DEFAULT_QUERY,
    next_action_prompt,
)
import rlm.utils.utils as utils
from rlm.logger.root_logger import ColorfulLogger
from rlm.logger.repl_logger import REPLEnvLogger


# Log analysis specific system prompt
LOG_ANALYSIS_SYSTEM_PROMPT = """You are a specialized system log analyzer tasked with analyzing logs to identify issues, correlate events, and provide root cause analysis. You have access to a REPL environment with powerful log parsing and correlation tools.

The REPL environment is initialized with:
1. A `context` variable containing log data (single log or dict of multiple logs)
2. Specialized log parser functions: parse_jstack(), parse_strace(), parse_gc_log(), parse_pstack(), parse_syslog(), parse_json_logs()
3. Auto-detection function: detect_log_format() and universal parser: parse_log()
4. Correlation utilities: correlate_logs(), find_correlated_events(), detect_all_patterns()
5. Timeline tools: Timeline, LogEvent, extract_events_from_parsed_log()
6. Recursive LLM functions: llm_query(prompt), llm_query_batch(prompts) for parallel processing
7. Standard Python libraries and print() for debugging

**Your Analysis Process:**

1. **PEEK at the context first** - Understand what you're working with:
   ```repl
   print(type(context))
   if isinstance(context, dict):
       print("Multiple logs:", list(context.keys()))
       for name, content in context.items():
           print(f"{name}: {len(content)} chars, format: {detect_log_format(content)}")
   else:
       print(f"Single log: {len(context)} chars")
       print(f"Format: {detect_log_format(context)}")
   ```

2. **PARSE the logs** - Extract structured data:
   ```repl
   # Single log
   format = detect_log_format(context)
   parsed = parse_log(context, format)
   print(f"Parsed: {parsed.keys()}")

   # Or multiple logs
   parsed_logs = {}
   for name, content in context.items():
       parsed_logs[name] = parse_log(content)
   ```

3. **CORRELATE across logs** (if multiple):
   ```repl
   timeline = correlate_logs(parsed_logs)
   print(f"Total events: {len(timeline.events)}")
   print(f"Sources: {list(timeline.by_source.keys())}")
   ```

4. **DETECT patterns** - Find known issues:
   ```repl
   patterns = detect_all_patterns(timeline)
   for p in patterns:
       print(f"{p['severity']}: {p['pattern']} - {p['description']}")
   ```

5. **ANALYZE specific issues**:
   - For jstack: Look for deadlocks, blocked threads, lock chains
   - For strace: Find slow syscalls (>1s), errors, I/O bottlenecks
   - For GC logs: Check pause times (>1s warning, >5s critical), heap usage trends
   - For correlation: Match timestamps across logs (+/- 5 seconds)

6. **Use parallel processing** for large logs:
   ```repl
   # For logs >100k chars, chunk and process in parallel
   if len(context) > 100000:
       chunks = [context[i:i+50000] for i in range(0, len(context), 50000)]
       prompts = [f"Extract all ERROR messages from: {chunk}" for chunk in chunks]
       results = llm_query_batch(prompts)  # Parallel, much faster!
   ```

**Output Requirements:**

Your final answer should include:
1. **Summary**: What logs were analyzed
2. **Issues Found**: Categorized by severity (CRITICAL, WARNING, INFO)
3. **Root Cause**: Explain WHY issues occurred (if determinable)
4. **Timeline**: Key events in chronological order (if timestamps available)
5. **Recommendations**: Actionable fixes or next steps

Use structured output when appropriate (JSON, markdown tables, etc.).

**Important Notes:**
- ALWAYS peek at context first before parsing
- Use detect_log_format() to auto-detect format
- For multiple logs, ALWAYS correlate them temporally
- Use llm_query_batch() for parallel processing of chunks
- Focus on finding root causes, not just listing errors
- Correlate events across logs (e.g., GC pause + I/O blocking = swapping)

**Common Patterns to Detect:**
- **Deadlock**: Circular lock dependencies in jstack
- **Memory Pressure**: Increasing GC pause times and frequency
- **I/O Bottleneck**: Slow syscalls in strace (>1s)
- **GC-Caused Blocking**: GC pause followed by I/O blocking (swapping)
- **Thread Contention**: Many BLOCKED threads waiting on same lock

When you have completed your analysis, provide your final answer using:
- FINAL(your analysis here) for direct text response
- FINAL_VAR(variable_name) to return a structured result

Think step-by-step, execute in REPL immediately, and provide comprehensive analysis. Do not just say "I will do this" - execute it now!
"""


def build_log_analysis_system_prompt() -> List[Dict[str, str]]:
    """
    Build the system message for log analysis RLM.

    Returns:
        List containing the system message with log analysis instructions
    """
    return [
        {
            "role": "system",
            "content": LOG_ANALYSIS_SYSTEM_PROMPT
        }
    ]


class RLMLogAnalyzer(RLM):
    """
    Specialized RLM for System Log Analysis.

    This class extends RLM with log-specific capabilities:
    - Auto-detection of log formats (jstack, strace, GC, pstack, syslog, JSON)
    - Temporal correlation across multiple log files
    - Pattern detection (deadlocks, memory leaks, I/O bottlenecks)
    - Root cause analysis

    Example:
        >>> analyzer = RLMLogAnalyzer(model="gpt-4o-mini")
        >>>
        >>> # Single log analysis
        >>> result = analyzer.completion(
        ...     context=open('/var/log/syslog').read(),
        ...     query="Find all ERROR messages and their causes"
        ... )
        >>>
        >>> # Multi-log correlation
        >>> logs = {
        ...     'jstack': open('thread_dump.txt').read(),
        ...     'gc': open('gc.log').read(),
        ...     'strace': open('strace.log').read()
        ... }
        >>> result = analyzer.completion(
        ...     context=logs,
        ...     query="Why did the application freeze at 14:30?"
        ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        recursive_model: str = "gpt-4o-mini",
        max_iterations: int = 20,
        depth: int = 0,
        max_depth: int = 1,
        enable_logging: bool = False,
        track_costs: bool = False,
    ):
        """
        Initialize the Log Analysis RLM.

        Args:
            api_key: OpenAI API key (uses env var if None)
            model: Model for root LM (e.g., 'gpt-4o')
            recursive_model: Model for recursive sub-calls (e.g., 'gpt-4o-mini')
            max_iterations: Maximum number of root LM iterations
            depth: Current recursion depth (0 = root)
            max_depth: Maximum recursion depth allowed
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
        self.max_depth = max_depth
        self._max_iterations = max_iterations
        self.track_costs = track_costs
        self.enable_logging = enable_logging

        # State
        self.repl_env: Optional[LogAnalysisREPLEnv] = None
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
        Setup the context for log analysis.

        Args:
            context: Log data (string, dict of logs, or list)
            query: The analysis query

        Returns:
            Initial message list with system prompt
        """
        if query is None:
            query = "Analyze these logs and identify any issues, errors, or anomalies."

        self.query = query
        self.logger.log_query_start(query)

        # Initialize conversation with log analysis system prompt
        self.messages = build_log_analysis_system_prompt()
        self.logger.log_initial_messages(self.messages)

        # Convert context for REPL
        context_data, context_str = utils.convert_context_for_repl(context)

        # Initialize specialized Log Analysis REPL environment
        self.repl_env = LogAnalysisREPLEnv(
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
            depth=self.depth,
            max_depth=self.max_depth,
            enable_logging=self.enable_logging,
            parent_rlm_class=RLMLogAnalyzer if self.max_depth > 1 else None,
        )

        return self.messages

    def completion(
        self,
        context: Union[List[str], str, Dict, List[Dict[str, str]]],
        query: Optional[str] = None
    ) -> str:
        """
        Analyze logs and answer query.

        Args:
            context: Log data (can be very large, single or multiple logs)
            query: The analysis question/task

        Returns:
            str: The analysis result

        Example:
            >>> analyzer = RLMLogAnalyzer(model="gpt-4o-mini")
            >>> result = analyzer.completion(
            ...     context={'gc': gc_log, 'strace': strace_log},
            ...     query="Find correlation between GC pauses and I/O blocking"
            ... )
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
        Get cost summary for this analysis.

        Returns:
            Dictionary with cost information
        """
        if not self.track_costs:
            return {
                "error": "Cost tracking not enabled",
                "note": "Set track_costs=True when initializing RLMLogAnalyzer"
            }

        return self.llm.get_cost_summary()

    def reset(self) -> None:
        """
        Reset the analyzer state for a fresh query.
        """
        # Reset REPL environment
        if self.repl_env is not None:
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


# Convenience function
def analyze_logs(
    context: Union[str, Dict[str, str]],
    query: str = "Analyze these logs and identify any issues.",
    model: str = "gpt-4o-mini",
    enable_logging: bool = True
) -> str:
    """
    Convenience function for quick log analysis.

    Args:
        context: Log content (string or dict of logs)
        query: Analysis question
        model: Model to use
        enable_logging: Whether to show execution logs

    Returns:
        Analysis result string

    Example:
        >>> result = analyze_logs(
        ...     context=open('gc.log').read(),
        ...     query="Are there any memory issues?"
        ... )
    """
    analyzer = RLMLogAnalyzer(model=model, enable_logging=enable_logging)
    return analyzer.completion(context, query)


if __name__ == "__main__":
    print("RLMLogAnalyzer ready!")
    print("Usage:")
    print("  from rlm.rlm_log_analysis import RLMLogAnalyzer, analyze_logs")
    print("  result = analyze_logs(context=log_content, query='Find errors')")
