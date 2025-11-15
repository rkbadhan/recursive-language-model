"""
REPL execution logger with Jupyter-style formatting.

This module provides beautiful, Jupyter-like logging for code executions
in the REPL environment using the rich library.
"""

from dataclasses import dataclass
from typing import List, Optional

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.rule import Rule
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class CodeExecution:
    """
    Record of a single code execution in the REPL.

    Attributes:
        code: The code that was executed
        stdout: Standard output from execution
        stderr: Standard error from execution
        execution_number: Sequential execution number
        execution_time: Time taken to execute (in seconds)
    """
    code: str
    stdout: str
    stderr: str
    execution_number: int
    execution_time: Optional[float] = None


class REPLEnvLogger:
    """
    Logger for REPL environment code executions.

    Displays executions in a Jupyter-style format with syntax highlighting,
    colored panels, and execution timing.
    """

    def __init__(self, max_output_length: int = 2000, enabled: bool = True):
        """
        Initialize the REPL logger.

        Args:
            max_output_length: Maximum length of output to display
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.max_output_length = max_output_length
        self.executions: List[CodeExecution] = []
        self.execution_count = 0

        if RICH_AVAILABLE and enabled:
            self.console = Console()
        else:
            self.console = None

    def _truncate_output(self, text: str) -> str:
        """
        Truncate text output to prevent overwhelming console output.

        Args:
            text: Text to potentially truncate

        Returns:
            Original text or truncated version with ellipsis
        """
        if len(text) <= self.max_output_length:
            return text

        # Show first half, then ellipsis, then last half
        half_length = self.max_output_length // 2
        first_part = text[:half_length]
        last_part = text[-half_length:]
        truncated_chars = len(text) - self.max_output_length

        return (
            f"{first_part}\n\n"
            f"... [TRUNCATED {truncated_chars} characters] ...\n\n"
            f"{last_part}"
        )

    def log_execution(
        self,
        code: str,
        stdout: str,
        stderr: str = "",
        execution_time: Optional[float] = None
    ) -> None:
        """
        Log a code execution with its output.

        Args:
            code: The code that was executed
            stdout: Standard output from execution
            stderr: Standard error from execution
            execution_time: Time taken to execute (in seconds)
        """
        self.execution_count += 1
        execution = CodeExecution(
            code=code,
            stdout=stdout,
            stderr=stderr,
            execution_number=self.execution_count,
            execution_time=execution_time
        )
        self.executions.append(execution)

    def display_last(self) -> None:
        """Display the last logged execution."""
        if not self.enabled:
            return
        if self.executions:
            self._display_single_execution(self.executions[-1])

    def display_all(self) -> None:
        """Display all logged executions in Jupyter-like format."""
        if not self.enabled:
            return

        for i, execution in enumerate(self.executions):
            self._display_single_execution(execution)
            # Add divider between cells (but not after the last one)
            if i < len(self.executions) - 1:
                if self.console:
                    self.console.print(Rule(style="dim", characters="─"))
                    self.console.print()
                else:
                    print("─" * 80)

    def _display_single_execution(self, execution: CodeExecution) -> None:
        """
        Display a single code execution like a Jupyter cell.

        Args:
            execution: The CodeExecution to display
        """
        if not self.enabled:
            return

        if self.console and RICH_AVAILABLE:
            self._display_with_rich(execution)
        else:
            self._display_plain(execution)

    def _display_with_rich(self, execution: CodeExecution) -> None:
        """Display execution with rich formatting."""
        # Truncate code if too long
        display_code = self._truncate_output(execution.code)

        # Input panel (code)
        input_panel = Panel(
            Syntax(display_code, "python", theme="monokai", line_numbers=True),
            title=f"[bold blue]In [{execution.execution_number}]:[/bold blue]",
            border_style="blue",
            box=box.ROUNDED
        )
        self.console.print(input_panel)

        # Output panel
        if execution.stderr:
            # Error output
            display_stderr = self._truncate_output(execution.stderr)
            error_text = Text(display_stderr, style="bold red")
            output_panel = Panel(
                error_text,
                title=(
                    f"[bold red]Error in [{execution.execution_number}]:"
                    f"[/bold red]"
                ),
                border_style="red",
                box=box.ROUNDED
            )
            self.console.print(output_panel)

        elif execution.stdout:
            # Normal output
            display_stdout = self._truncate_output(execution.stdout)
            output_text = Text(display_stdout, style="white")

            output_panel = Panel(
                output_text,
                title=(
                    f"[bold green]Out [{execution.execution_number}]:"
                    f"[/bold green]"
                ),
                border_style="green",
                box=box.ROUNDED
            )
            self.console.print(output_panel)

            # Show timing as separate panel if available
            if execution.execution_time is not None:
                timing_panel = Panel(
                    Text(
                        f"Execution time: {execution.execution_time:.4f}s",
                        style="bright_black"
                    ),
                    border_style="grey37",
                    box=box.ROUNDED,
                    title=(
                        f"[bold grey37]Timing [{execution.execution_number}]:"
                        f"[/bold grey37]"
                    )
                )
                self.console.print(timing_panel)

        else:
            # No output
            if execution.execution_time is not None:
                timing_text = Text(
                    f"Execution time: {execution.execution_time:.4f}s",
                    style="dim"
                )
                output_panel = Panel(
                    timing_text,
                    title=(
                        f"[bold dim]Out [{execution.execution_number}]:"
                        f"[/bold dim]"
                    ),
                    border_style="dim",
                    box=box.ROUNDED
                )
            else:
                output_panel = Panel(
                    Text("No output", style="dim"),
                    title=(
                        f"[bold dim]Out [{execution.execution_number}]:"
                        f"[/bold dim]"
                    ),
                    border_style="dim",
                    box=box.ROUNDED
                )
            self.console.print(output_panel)

    def _display_plain(self, execution: CodeExecution) -> None:
        """Display execution with plain text formatting (fallback)."""
        print(f"\n{'='*80}")
        print(f"In [{execution.execution_number}]:")
        print(f"{'-'*80}")
        print(execution.code)
        print(f"{'='*80}")

        if execution.stderr:
            print(f"Error [{execution.execution_number}]:")
            print(execution.stderr)
        elif execution.stdout:
            print(f"Out [{execution.execution_number}]:")
            print(execution.stdout)
        else:
            print(f"Out [{execution.execution_number}]: (no output)")

        if execution.execution_time is not None:
            print(f"Execution time: {execution.execution_time:.4f}s")

        print(f"{'='*80}\n")

    def clear(self) -> None:
        """Clear all logged executions."""
        self.executions.clear()
        self.execution_count = 0
