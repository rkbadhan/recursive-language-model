"""
Utility functions for RLM processing.

This module provides helper functions for:
- Parsing code blocks and final answers from model responses
- Executing code in REPL environments
- Processing and formatting execution results
- Managing message histories
"""

import re
from typing import List, Dict, Optional, Tuple, Any, Union


def find_code_blocks(text: str) -> Optional[List[str]]:
    """
    Extract REPL code blocks from model response.

    Searches for code wrapped in triple backticks with 'repl' language identifier:
    ```repl
    code here
    ```

    Args:
        text: Model response text

    Returns:
        List of code blocks (as strings), or None if no blocks found
    """
    pattern = r'```repl\s*\n(.*?)\n```'
    matches = re.finditer(pattern, text, re.DOTALL)

    results = []
    for match in matches:
        code_content = match.group(1).strip()
        results.append(code_content)

    return results if results else None


def find_final_answer(text: str) -> Optional[Tuple[str, str]]:
    """
    Find FINAL(...) or FINAL_VAR(...) statements in model response.

    Args:
        text: Model response text

    Returns:
        Tuple of (type, content) where type is 'FINAL' or 'FINAL_VAR',
        or None if neither pattern is found
    """
    # Check for FINAL_VAR pattern first
    # Pattern matches FINAL_VAR anywhere in text, preferably at start of line
    final_var_pattern = r'FINAL_VAR\((.*?)\)'
    match = re.search(final_var_pattern, text, re.DOTALL)
    if match:
        return ('FINAL_VAR', match.group(1).strip())

    # Check for FINAL pattern
    # Pattern matches FINAL anywhere in text, preferably at start of line
    final_pattern = r'FINAL\((.*?)\)'
    match = re.search(final_pattern, text, re.DOTALL)
    if match:
        return ('FINAL', match.group(1).strip())

    return None


def format_execution_result(
    stdout: str,
    stderr: str,
    locals_dict: Dict[str, Any],
    truncate_length: int = 100
) -> str:
    """
    Format REPL execution result for model consumption.

    Args:
        stdout: Standard output from execution
        stderr: Standard error from execution
        locals_dict: Local variables after execution
        truncate_length: Max length for variable value display

    Returns:
        Formatted string describing execution result
    """
    result_parts = []

    # Add stdout if present
    if stdout:
        result_parts.append(f"\n{stdout}")

    # Add stderr if present
    if stderr:
        result_parts.append(f"\n{stderr}")

    # Show key variables (excluding internal ones)
    important_vars = {}
    for key, value in locals_dict.items():
        # Skip internal variables
        if key.startswith('_') or key in ['__builtins__', '__name__', '__doc__']:
            continue

        try:
            # Only show simple types or short representations
            if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                if isinstance(value, str) and len(value) > truncate_length:
                    important_vars[key] = f"'{value[:truncate_length]}...'"
                else:
                    important_vars[key] = repr(value)
        except:
            important_vars[key] = f"<{type(value).__name__}>"

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


def execute_code(
    repl_env,
    code: str,
    repl_env_logger,
    logger
) -> str:
    """
    Execute code in REPL environment and return formatted result.

    Args:
        repl_env: The REPL environment instance
        code: Python code to execute
        repl_env_logger: Logger for REPL execution
        logger: Main root logger

    Returns:
        Formatted execution result string
    """
    try:
        result = repl_env.code_execution(code)

        formatted_result = format_execution_result(
            result.stdout,
            result.stderr,
            result.locals
        )

        # Log execution
        repl_env_logger.log_execution(
            code,
            result.stdout,
            result.stderr,
            result.execution_time
        )
        repl_env_logger.display_last()

        # Log to root logger
        logger.log_tool_execution("CODE_EXECUTION", formatted_result)

        return formatted_result

    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        logger.log_tool_execution("CODE_EXECUTION", error_msg)
        return error_msg


def add_execution_result_to_messages(
    messages: List[Dict[str, str]],
    code: str,
    result: str,
    max_character_length: int = 100000
) -> List[Dict[str, str]]:
    """
    Add code execution result to conversation message history.

    Args:
        messages: Current conversation messages
        code: The code that was executed
        result: Result from code execution
        max_character_length: Maximum length of result to include

    Returns:
        Updated messages list
    """
    # Truncate result if too long
    if len(result) > max_character_length:
        result = result[:max_character_length] + "..."

    # Add execution result message
    execution_message = {
        "role": "user",
        "content": (
            f"Code executed:\n```python\n{code}\n```\n\n"
            f"REPL output:\n{result}"
        )
    }
    messages.append(execution_message)

    return messages


def process_code_execution(
    response: str,
    messages: List[Dict[str, str]],
    repl_env,
    repl_env_logger,
    logger
) -> List[Dict[str, str]]:
    """
    Process code execution from model response.

    Extracts code blocks, executes them in REPL, and adds results to messages.

    Args:
        response: The model response containing code
        messages: Current conversation messages
        repl_env: The REPL environment
        repl_env_logger: Logger for REPL execution
        logger: Main root logger

    Returns:
        Updated messages list with execution results
    """
    # Extract code blocks from response
    code_blocks = find_code_blocks(response)

    if code_blocks:
        # Execute each code block
        for code in code_blocks:
            execution_result = execute_code(
                repl_env,
                code,
                repl_env_logger,
                logger
            )

            # Add execution result to conversation
            messages = add_execution_result_to_messages(
                messages,
                code,
                execution_result
            )

    return messages


def check_for_final_answer(
    response: str,
    repl_env,
    logger
) -> Optional[str]:
    """
    Check if model response contains a final answer.

    Looks for FINAL() or FINAL_VAR() patterns and retrieves the answer.

    Args:
        response: Model response text
        repl_env: REPL environment (for FINAL_VAR lookups)
        logger: Logger instance

    Returns:
        Final answer string, or None if no final answer found
    """
    result = find_final_answer(response)
    if result is None:
        return None

    answer_type, content = result

    if answer_type == 'FINAL':
        # Direct answer
        return content

    elif answer_type == 'FINAL_VAR':
        # Get variable from REPL environment
        try:
            # Clean variable name
            variable_name = content.strip().strip('"').strip("'").strip()

            # Check if variable exists
            if variable_name in repl_env.locals:
                variable_value = repl_env.locals[variable_name]
                return str(variable_value)
            else:
                error_msg = f"Variable '{variable_name}' not found in REPL"
                logger.log_tool_execution("FINAL_VAR", error_msg)
                return None

        except Exception as e:
            error_msg = f"Error retrieving variable '{content}': {str(e)}"
            logger.log_tool_execution("FINAL_VAR", error_msg)
            return None

    return None


def convert_context_for_repl(
    context: Union[str, List, Dict]
) -> Tuple[Optional[Union[Dict, List]], Optional[str]]:
    """
    Convert various context formats to REPL-compatible format.

    Args:
        context: Context in various formats (str, list, dict)

    Returns:
        Tuple of (context_json, context_str) where one is None
    """
    if isinstance(context, dict):
        # Structured data -> JSON
        return context, None

    elif isinstance(context, str):
        # Plain text -> string
        return None, context

    elif isinstance(context, list):
        if len(context) > 0 and isinstance(context[0], dict):
            # List of dicts (e.g., messages)
            if "content" in context[0]:
                # Extract content from messages
                context_data = [msg.get("content", "") for msg in context]
            else:
                # Use as-is
                context_data = context
            return context_data, None
        else:
            # List of other types
            return context, None

    else:
        # Unknown type -> try as JSON
        return context, None
