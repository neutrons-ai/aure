"""
LangSmith tracing support for the reflectivity analysis workflow.

This module provides utilities for tracing workflow execution with LangSmith,
enabling visibility into:
- Each node's inputs and outputs
- Timing and performance metrics
- Error tracking
- Chi-squared progression through refinement iterations

Tracing is enabled when LANGCHAIN_TRACING_V2=true is set in the environment.
"""

import os
from typing import Callable, Dict, Any, Optional


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled via environment variable."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"


def get_trace_context():
    """
    Get the LangSmith trace context manager if tracing is enabled.
    
    Returns:
        The langsmith.trace context manager, or None if tracing is disabled
        or langsmith is not installed.
    """
    if not is_tracing_enabled():
        return None
    try:
        from langsmith import trace
        return trace
    except ImportError:
        return None


def run_with_tracing(
    func: Callable,
    state: Dict[str, Any],
    name: str,
    trace_ctx,
    run_type: str = "chain",
) -> Dict[str, Any]:
    """
    Run a function with optional LangSmith tracing.
    
    Args:
        func: The function to run (takes state, returns updates dict)
        state: Current workflow state
        name: Name for the trace span
        trace_ctx: Trace context manager (from get_trace_context), or None
        run_type: LangSmith run type (default: "chain")
        
    Returns:
        The function's return value (updates dict)
    """
    if trace_ctx is None:
        return func(state)
    
    iteration = state.get("iteration", 0)
    with trace_ctx(
        name=name,
        run_type=run_type,
        metadata={
            "node": name.replace("node_", ""),
            "iteration": iteration,
            "data_file": state.get("data_file", ""),
        },
        inputs={
            "current_node": name.replace("node_", ""),
            "iteration": iteration,
            "current_chi2": state.get("current_chi2"),
            "has_model": state.get("current_model") is not None,
            "n_fit_results": len(state.get("fit_results", [])),
        },
    ) as run:
        updates = func(state)
        run.end(outputs={
            "error": updates.get("error"),
            "chi2": updates.get("current_chi2") or state.get("current_chi2"),
            "has_model": (
                updates.get("current_model") is not None 
                or state.get("current_model") is not None
            ),
            "n_messages": len(updates.get("messages", [])),
        })
        return updates


class TracedWorkflow:
    """
    Context manager for tracing an entire workflow run.
    
    Usage:
        with TracedWorkflow(data_file, sample_description, max_iterations) as tw:
            result = run_workflow(...)
            tw.set_result(result)
    """
    
    def __init__(
        self,
        data_file: str,
        sample_description: str,
        hypothesis: Optional[str] = None,
        max_iterations: int = 5,
    ):
        self.data_file = data_file
        self.sample_description = sample_description
        self.hypothesis = hypothesis
        self.max_iterations = max_iterations
        self._trace_ctx = get_trace_context()
        self._trace_cm = None  # The context manager object
        self._run = None       # The RunTree from __enter__
        self._result = None
    
    def __enter__(self):
        if self._trace_ctx is not None:
            # Create the context manager
            self._trace_cm = self._trace_ctx(
                name="reflectivity_analysis",
                run_type="chain",
                metadata={
                    "data_file": self.data_file,
                    "max_iterations": self.max_iterations,
                },
                inputs={
                    "data_file": self.data_file,
                    "sample_description": self.sample_description,
                    "hypothesis": self.hypothesis,
                    "max_iterations": self.max_iterations,
                },
            )
            # Enter it and store the RunTree
            self._run = self._trace_cm.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._trace_cm is not None:
            if self._result is not None and self._run is not None:
                self._run.end(outputs={
                    "success": self._result.get("error") is None,
                    "final_chi2": self._result.get("current_chi2"),
                    "iterations": self._result.get("iteration", 0),
                    "n_fit_results": len(self._result.get("fit_results", [])),
                })
            # Exit the context manager (not the RunTree)
            self._trace_cm.__exit__(exc_type, exc_val, exc_tb)
        return False  # Don't suppress exceptions
    
    def set_result(self, result: Dict[str, Any]):
        """Set the workflow result for trace output."""
        self._result = result
    
    @property
    def is_tracing(self) -> bool:
        """Check if this workflow run is being traced."""
        return self._trace_ctx is not None
