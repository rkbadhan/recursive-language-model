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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List

from rlm import RLM
from rlm.utils.llm import OpenAIClient


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

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the sub-RLM.

        Args:
            model: Model identifier to use for recursive calls
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.model = model
        self.client = OpenAIClient(api_key=api_key, model=model)

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
        try:
            response = self.client.completion(
                messages=prompt,
                timeout=300
            )
            return response

        except Exception as e:
            error_msg = f"Error in sub-RLM call: {str(e)}"
            return error_msg

    def cost_summary(self) -> Dict[str, Any]:
        """Not implemented for SubRLM."""
        raise NotImplementedError("Cost tracking not implemented for SubRLM")

    def reset(self) -> None:
        """Not implemented for SubRLM."""
        raise NotImplementedError("Reset not implemented for SubRLM")


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
    ):
        """
        Initialize the REPL environment.

        Args:
            recursive_model: Model to use for recursive llm_query() calls
            context_json: Context data as JSON (dict or list)
            context_str: Context data as string
            setup_code: Optional Python code to run during initialization
        """
        # Store original working directory
        self.original_cwd = os.getcwd()

        # Create temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp(prefix="rlm_repl_")

        # Initialize sub-RLM for recursive calls
        self.sub_rlm: RLM = SubRLM(model=recursive_model)

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
            Query a sub-LLM from within the REPL environment.

            Args:
                prompt: The prompt to send to the LLM

            Returns:
                str: The LLM's response
            """
            return self.sub_rlm.completion(prompt)

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
        self.globals['FINAL_VAR'] = FINAL_VAR

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

    def code_execution(self, code: str) -> REPLResult:
        """
        Execute Python code in the REPL environment (Jupyter-style).

        Args:
            code: Python code to execute

        Returns:
            REPLResult: Execution results with stdout, stderr, locals, time
        """
        start_time = time.time()

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

    def __del__(self):
        """Clean up temporary directory on deletion."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass
