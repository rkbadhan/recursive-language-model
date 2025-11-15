"""
Recursive Language Models (RLM) - A framework for processing unbounded context.

This package provides an implementation of Recursive Language Models that can
handle arbitrarily long contexts by treating them as programmable variables in
a REPL environment.
"""

from .rlm import RLM

# Lazy import for log analysis (requires openai package)
def __getattr__(name):
    if name == "RLMLogAnalyzer":
        from .rlm_log_analysis import RLMLogAnalyzer
        return RLMLogAnalyzer
    elif name == "analyze_logs":
        from .rlm_log_analysis import analyze_logs
        return analyze_logs
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "0.1.0"
__all__ = ["RLM", "RLMLogAnalyzer", "analyze_logs"]
