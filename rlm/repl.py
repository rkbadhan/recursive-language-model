"""
REPL Environment for Recursive Language Models.

This module provides a sandboxed Python execution environment where language
models can programmatically interact with large contexts through code execution
and recursive LLM calls.
"""

import sys
import io
import os
import json
import time
import tempfile
import threading
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List

from rlm import RLM
from rlm.utils.llm import OpenAIClient
from openai import AsyncOpenAI


@dataclass
class REPLResult:
    """
    Result of executing code in the REPL environment.

    Attributes:
        stdout: Standard output captured during execution
        stderr: Standard error captured during execution
        locals: Local variables after execution
        execution_time: Time taken to execute (in seconds)
    """
    stdout: str
    stderr: str
    locals: Dict[str, Any]
    execution_time: float

    def __str__(self) -> str:
        return (
            f"REPLResult(\n"
            f"  stdout={repr(self.stdout[:100])},\n"
            f"  stderr={repr(self.stderr[:100])},\n"
            f"  num_locals={len(self.locals)},\n"
            f"  execution_time={self.execution_time:.4f}s\n"
            f")"
        )


class SubRLM(RLM):
    """
    Simple sub-RLM for depth=1 recursion.

    This is a lightweight LLM wrapper used within the REPL environment.
    For deeper recursion (depth > 1), this could be replaced with a full RLM_REPL.
    """

    def __init__(self, model: str = "gpt-4o-mini", track_costs: bool = True):
        """
        Initialize the sub-RLM.

        Args:
            model: Model identifier to use for recursive calls
            track_costs: Whether to track API costs
        """
        self.model = model
        self.client = OpenAIClient(model=model, track_costs=track_costs)

    def completion(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        context: Optional[str] = None
    ) -> str:
        """
        Simple LLM query for sub-calls.

        Args:
            prompt: The prompt to send to the LLM
            context: Optional context (unused, for API compatibility)

        Returns:
            str: The LLM's response
        """
        return self.client.completion(messages=prompt, timeout=300)

    def cost_summary(self) -> Dict[str, Any]:
        """Get cost summary from the client."""
        return self.client.get_cost_summary()

    def reset(self) -> None:
        """Reset cost tracking."""
        self.client.reset_costs()


class REPLEnv:
    """
    Python REPL environment for RLM execution.

    Provides a sandboxed execution environment where the LLM can:
    - Execute Python code
    - Access context as variables
    - Make recursive LLM calls via llm_query()
    - Return final answers via FINAL_VAR()
    """

    def __init__(
        self,
        recursive_model: str = "gpt-4o-mini",
        context_json: Optional[Union[Dict, List]] = None,
        context_str: Optional[str] = None,
        setup_code: Optional[str] = None,
        depth: int = 0,
        max_depth: int = 1,
        enable_logging: bool = False,
        parent_rlm_class=None,
        track_costs: bool = False,
    ):
        """
        Initialize the REPL environment.

        Args:
            recursive_model: Model to use for recursive llm_query() calls
            context_json: Context data as JSON (dict or list)
            context_str: Context data as string
            setup_code: Optional Python code to run during initialization
            depth: Current recursion depth (0 = root)
            max_depth: Maximum recursion depth allowed
            enable_logging: Whether to enable logging for recursive RLMs
            parent_rlm_class: Class to use for recursive RLM creation (for depth > 1)
            track_costs: Whether to track API costs
        """
        # Store original working directory
        self.original_cwd = os.getcwd()

        # Create temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")

        # Store depth configuration
        self.depth = depth
        self.max_depth = max_depth
        self.recursive_model = recursive_model
        self.enable_logging = enable_logging
        self.parent_rlm_class = parent_rlm_class
        self.track_costs = track_costs

        # Cost tracking for async calls
        self.async_input_tokens = 0
        self.async_output_tokens = 0
        self.async_calls = 0

        # Initialize async client for true async support (using OpenAI's native AsyncOpenAI)
        # Let AsyncOpenAI handle validation - it will fail with clear error if no API key
        self.async_client = AsyncOpenAI()  # Uses OPENAI_API_KEY env var by default
        self.async_model = recursive_model

        # Initialize sub-RLM for recursive calls
        # Use full RLM if depth > 1 is allowed, otherwise use simple SubRLM
        if depth < max_depth and parent_rlm_class is not None:
            # Create a full recursive RLM for depth > 1
            self.sub_rlm: RLM = parent_rlm_class(
                model=recursive_model,
                recursive_model=recursive_model,
                max_iterations=10,  # Reasonable default for sub-RLMs
                depth=depth + 1,
                max_depth=max_depth,
                enable_logging=enable_logging,
                track_costs=track_costs,
            )
        else:
            # Use simple SubRLM for depth=1 or when max_depth reached
            self.sub_rlm: RLM = SubRLM(model=recursive_model, track_costs=track_costs)

        # Setup safe execution environment
        self.globals = self._create_safe_globals()
        self.locals = {}
        self._lock = threading.Lock()

        # Load context into REPL
        self.load_context(context_json, context_str)

        # Add special functions to globals
        self._inject_special_functions()

        # Run setup code if provided
        if setup_code:
            self.code_execution(setup_code)

    def _create_safe_globals(self) -> Dict[str, Any]:
        """
        Create a safe globals dictionary with whitelisted built-ins.

        Returns:
            Dict with safe built-in functions and classes
        """
        safe_builtins = {
            # Basic types
            'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
            'bytes': bytes, 'bytearray': bytearray, 'complex': complex,

            # Type checking
            'type': type, 'isinstance': isinstance, 'issubclass': issubclass,
            'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
            'delattr': delattr, 'callable': callable,

            # Iteration
            'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
            'sorted': sorted, 'reversed': reversed, 'range': range,
            'iter': iter, 'next': next, 'slice': slice,

            # Math
            'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'pow': pow, 'divmod': divmod,

            # String/encoding
            'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct,
            'repr': repr, 'ascii': ascii, 'format': format, 'hash': hash,

            # Introspection
            'dir': dir, 'vars': vars, 'id': id,

            # Object-oriented
            'object': object, 'super': super, 'property': property,
            'staticmethod': staticmethod, 'classmethod': classmethod,

            # Utilities
            'any': any, 'all': all,

            # Exceptions
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            'KeyError': KeyError, 'IndexError': IndexError,
            'AttributeError': AttributeError, 'RuntimeError': RuntimeError,
            'FileNotFoundError': FileNotFoundError, 'OSError': OSError,
            'IOError': IOError, 'NameError': NameError, 'ImportError': ImportError,
            'StopIteration': StopIteration, 'AssertionError': AssertionError,
            'NotImplementedError': NotImplementedError,

            # Allow imports and file access (needed for context loading)
            '__import__': __import__,
            'open': open,

            # Blocked for security
            'input': None, 'eval': None, 'exec': None, 'compile': None,
            'globals': None, 'locals': None,
        }

        return {'__builtins__': safe_builtins}

    def _inject_special_functions(self) -> None:
        """Inject special RLM functions into the REPL globals."""

        def llm_query(prompt: str) -> str:
            """
            Query a sub-LLM from within the REPL environment (synchronous).

            Args:
                prompt: The prompt to send to the LLM

            Returns:
                str: The LLM's response
            """
            return self.sub_rlm.completion(prompt)

        async def llm_query_async(prompt: str) -> str:
            """
            Query a sub-LLM from within the REPL environment (async version).

            This uses OpenAI's native AsyncOpenAI client for true async I/O.

            Args:
                prompt: The prompt to send to the LLM

            Returns:
                str: The LLM's response
            """
            # Use OpenAI's native async client directly
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            response = await self.async_client.chat.completions.create(
                model=self.async_model,
                messages=messages,
                temperature=0.7,
            )

            # Track costs if enabled
            if self.track_costs and hasattr(response, 'usage'):
                self.async_input_tokens += response.usage.prompt_tokens
                self.async_output_tokens += response.usage.completion_tokens
                self.async_calls += 1

            return response.choices[0].message.content

        def llm_query_batch(prompts: List[str]) -> List[str]:
            """
            Query multiple sub-LLMs in parallel (synchronous interface).

            This function runs multiple LLM queries concurrently, which is much
            faster than calling llm_query() in a loop.

            Args:
                prompts: List of prompts to send to the LLM

            Returns:
                List[str]: List of LLM responses in the same order

            Example:
                chunks = [context[i:i+1000] for i in range(0, len(context), 1000)]
                prompts = [f"Summarize: {chunk}" for chunk in chunks]
                summaries = llm_query_batch(prompts)
            """
            # Run async version and return results
            return asyncio.run(self._batch_query_async(prompts))

        async def llm_query_batch_async(prompts: List[str]) -> List[str]:
            """
            Query multiple sub-LLMs in parallel (async version).

            Args:
                prompts: List of prompts to send to the LLM

            Returns:
                List[str]: List of LLM responses in the same order
            """
            return await self._batch_query_async(prompts)

        def FINAL_VAR(variable_name: str) -> str:
            """
            Return a variable as the final answer.

            Args:
                variable_name: Name of the variable to return

            Returns:
                str: String representation of the variable's value
            """
            # Clean the variable name
            var_name = variable_name.strip().strip('"').strip("'").strip()

            try:
                if var_name in self.locals:
                    value = self.locals[var_name]
                    return str(value)
                else:
                    return f"Error: Variable '{var_name}' not found in REPL"
            except Exception as e:
                return f"Error retrieving variable '{var_name}': {str(e)}"

        # Add functions to globals
        self.globals['llm_query'] = llm_query
        self.globals['llm_query_async'] = llm_query_async
        self.globals['llm_query_batch'] = llm_query_batch
        self.globals['llm_query_batch_async'] = llm_query_batch_async
        self.globals['FINAL_VAR'] = FINAL_VAR

        # Also add asyncio for async code execution
        self.globals['asyncio'] = asyncio

    def load_context(
        self,
        context_json: Optional[Union[Dict, List]] = None,
        context_str: Optional[str] = None
    ) -> None:
        """
        Load context data into the REPL environment as variables.

        Args:
            context_json: Structured data to load as JSON
            context_str: Text data to load as string
        """
        # Load JSON context
        if context_json is not None:
            context_path = os.path.join(self.temp_dir, "context.json")
            with open(context_path, "w") as f:
                json.dump(context_json, f, indent=2)

            context_code = (
                f"import json\n"
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = json.load(f)\n"
            )
            self.code_execution(context_code)

        # Load string context
        if context_str is not None:
            context_path = os.path.join(self.temp_dir, "context.txt")
            with open(context_path, "w") as f:
                f.write(context_str)

            context_code = (
                f"with open(r'{context_path}', 'r') as f:\n"
                f"    context = f.read()\n"
            )
            self.code_execution(context_code)

    async def _batch_query_async(self, prompts: List[str]) -> List[str]:
        """
        Internal async method to run multiple LLM queries in parallel.

        Uses OpenAI's native AsyncOpenAI client with asyncio.gather for true async I/O.

        Args:
            prompts: List of prompts to process

        Returns:
            List of responses in the same order as prompts
        """
        # Use OpenAI's native async client with asyncio.gather for parallel execution
        async def single_query(prompt: str) -> str:
            messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            response = await self.async_client.chat.completions.create(
                model=self.async_model,
                messages=messages,
                temperature=0.7,
            )

            # Track costs if enabled
            if self.track_costs and hasattr(response, 'usage'):
                self.async_input_tokens += response.usage.prompt_tokens
                self.async_output_tokens += response.usage.completion_tokens
                self.async_calls += 1

            return response.choices[0].message.content

        tasks = [single_query(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return list(results)

    @contextmanager
    def _capture_output(self):
        """Context manager to capture stdout/stderr in thread-safe manner."""
        with self._lock:
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                yield stdout_buffer, stderr_buffer
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    @contextmanager
    def _temp_working_directory(self):
        """Context manager to temporarily change working directory."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def _detect_async_code(self, code: str) -> bool:
        """
        Detect if code contains async/await syntax.

        Args:
            code: Python code to check

        Returns:
            True if code appears to use async/await
        """
        # Simple heuristic: check for async def, await, or asyncio
        async_keywords = ['async def', 'await ', 'asyncio.run', 'asyncio.gather']
        return any(keyword in code for keyword in async_keywords)

    def code_execution(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment (Jupyter-style).

        Supports both synchronous and asynchronous code execution.

        Args:
            code: Python code to execute

        Returns:
            REPLResult: Execution results with stdout, stderr, locals, time
        """
        start_time = time.time()

        # Check if code uses async/await
        is_async = self._detect_async_code(code)

        if is_async:
            # Execute as async code
            return self._code_execution_async(code, start_time)

        with self._capture_output() as (stdout_buffer, stderr_buffer):
            with self._temp_working_directory():
                try:
                    # Split code into imports and other code
                    lines = code.split('\n')
                    import_lines = []
                    other_lines = []

                    for line in lines:
                        if line.strip().startswith(('import ', 'from ')):
                            import_lines.append(line)
                        else:
                            other_lines.append(line)

                    # Execute imports in globals first
                    if import_lines:
                        import_code = '\n'.join(import_lines)
                        exec(import_code, self.globals, self.globals)

                    # Execute remaining code
                    if other_lines:
                        other_code = '\n'.join(other_lines)
                        combined_namespace = {**self.globals, **self.locals}

                        # Check if last line is an expression (auto-print)
                        non_comment_lines = [
                            line for line in other_lines
                            if line.strip() and not line.strip().startswith('#')
                        ]

                        if non_comment_lines:
                            last_line = non_comment_lines[-1].strip()

                            # Heuristic: is it an expression?
                            is_expression = (
                                not last_line.startswith((
                                    'import ', 'from ', 'def ', 'class ',
                                    'if ', 'for ', 'while ', 'try:', 'with ',
                                    'return ', 'yield ', 'break', 'continue', 'pass'
                                )) and
                                '=' not in last_line.split('#')[0] and
                                not last_line.endswith(':') and
                                not last_line.startswith('print(')
                            )

                            if is_expression:
                                try:
                                    # Execute all but last line
                                    if len(non_comment_lines) > 1:
                                        last_line_idx = -1
                                        for i, line in enumerate(other_lines):
                                            if line.strip() == last_line:
                                                last_line_idx = i
                                                break

                                        if last_line_idx > 0:
                                            statements = '\n'.join(
                                                other_lines[:last_line_idx]
                                            )
                                            exec(statements, combined_namespace,
                                                 combined_namespace)

                                    # Evaluate last line and print
                                    result = eval(last_line, combined_namespace,
                                                 combined_namespace)
                                    if result is not None:
                                        print(repr(result))

                                except:
                                    # Fall back to normal execution
                                    exec(other_code, combined_namespace,
                                         combined_namespace)
                            else:
                                # Execute as statements
                                exec(other_code, combined_namespace,
                                     combined_namespace)
                        else:
                            # Only comments
                            exec(other_code, combined_namespace,
                                 combined_namespace)

                        # Update locals with new variables
                        for key, value in combined_namespace.items():
                            if key not in self.globals:
                                self.locals[key] = value

                    stdout_content = stdout_buffer.getvalue()
                    stderr_content = stderr_buffer.getvalue()

                except Exception as e:
                    stderr_content = stderr_buffer.getvalue() + str(e)
                    stdout_content = stdout_buffer.getvalue()

        end_time = time.time()
        execution_time = end_time - start_time

        # Store outputs in locals for access
        self.locals['_stdout'] = stdout_content
        self.locals['_stderr'] = stderr_content

        return REPLResult(
            stdout=stdout_content,
            stderr=stderr_content,
            locals=self.locals.copy(),
            execution_time=execution_time
        )

    def _code_execution_async(self, code: str, start_time: float) -> REPLResult:
        """
        Execute async Python code in the REPL environment.

        Args:
            code: Async Python code to execute
            start_time: Start time for execution tracking

        Returns:
            REPLResult: Execution results
        """
        with self._capture_output() as (stdout_buffer, stderr_buffer):
            with self._temp_working_directory():
                try:
                    # Split code into imports and other code
                    lines = code.split('\n')
                    import_lines = []
                    other_lines = []

                    for line in lines:
                        if line.strip().startswith(('import ', 'from ')):
                            import_lines.append(line)
                        else:
                            other_lines.append(line)

                    # Execute imports in globals first
                    if import_lines:
                        import_code = '\n'.join(import_lines)
                        exec(import_code, self.globals, self.globals)

                    # Execute remaining code
                    if other_lines:
                        other_code = '\n'.join(other_lines)
                        combined_namespace = {**self.globals, **self.locals}

                        # Wrap code in async function and run
                        async_wrapper = f"""
async def __async_exec():
{chr(10).join('    ' + line for line in other_lines)}
"""
                        # Execute the wrapper to define the function
                        exec(async_wrapper, combined_namespace, combined_namespace)

                        # Run the async function
                        asyncio.run(combined_namespace['__async_exec']())

                        # Update locals with new variables
                        for key, value in combined_namespace.items():
                            if key not in self.globals and key != '__async_exec':
                                self.locals[key] = value

                    stdout_content = stdout_buffer.getvalue()
                    stderr_content = stderr_buffer.getvalue()

                except Exception as e:
                    stderr_content = stderr_buffer.getvalue() + str(e)
                    stdout_content = stdout_buffer.getvalue()

        end_time = time.time()
        execution_time = end_time - start_time

        # Store outputs in locals for access
        self.locals['_stdout'] = stdout_content
        self.locals['_stderr'] = stderr_content

        return REPLResult(
            stdout=stdout_content,
            stderr=stderr_content,
            locals=self.locals.copy(),
            execution_time=execution_time
        )

    def get_repl_cost_summary(self) -> Dict[str, Any]:
        """
        Get cost summary for all REPL-initiated LLM calls.

        Returns:
            Dictionary with aggregated cost information from sync and async calls
        """
        if not self.track_costs:
            return {"error": "Cost tracking not enabled"}

        # Get sync call costs from sub_rlm
        sync_costs = self.sub_rlm.cost_summary()

        # Calculate async call costs
        async_input_cost = 0.0
        async_output_cost = 0.0

        # Model pricing (per 1M tokens)
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.150, "output": 0.600},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }

        if self.async_model in pricing:
            async_input_cost = (self.async_input_tokens / 1_000_000) * pricing[self.async_model]["input"]
            async_output_cost = (self.async_output_tokens / 1_000_000) * pricing[self.async_model]["output"]

        # Aggregate costs
        total_calls = sync_costs.get('total_calls', 0) + self.async_calls
        total_input_tokens = sync_costs.get('input_tokens', 0) + self.async_input_tokens
        total_output_tokens = sync_costs.get('output_tokens', 0) + self.async_output_tokens
        sync_cost = sync_costs.get('estimated_cost_usd', 0.0)
        async_cost = async_input_cost + async_output_cost

        return {
            "depth": self.depth,
            "total_calls": total_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "estimated_cost_usd": round(sync_cost + async_cost, 4),
            "breakdown": {
                "sync_calls": sync_costs.get('total_calls', 0),
                "async_calls": self.async_calls,
                "sync_cost_usd": round(sync_cost, 4),
                "async_cost_usd": round(async_cost, 4),
            },
            "model": self.recursive_model,
        }

    def __del__(self):
        """Clean up temporary directory and async client on deletion."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

        # Close AsyncOpenAI client (it handles its own cleanup gracefully)
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(self.async_client.close())
        except:
            pass
