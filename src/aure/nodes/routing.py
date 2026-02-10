"""Routing functions for workflow transitions.

These functions determine which node to execute next based on the current state.
All routes lead either to the next processing node, back to a previous node
for refinement, or to completion (END).
"""

from typing import Literal
from ..state import ReflectivityState


def route_after_intake(state: ReflectivityState) -> Literal["analysis", "error"]:
    """
    Route after intake node.

    Proceeds to analysis if data was loaded successfully.
    """
    if state.get("error"):
        return "error"

    if not state.get("Q") or len(state["Q"]) == 0:
        return "error"

    return "analysis"


def route_after_analysis(
    state: ReflectivityState
) -> Literal["modeling", "error"]:
    """
    Route after analysis node.

    Always proceeds to modeling if no error occurred.
    The modeling node will work with whatever information is available.
    """
    if state.get("error"):
        return "error"

    return "modeling"


def route_after_modeling(
    state: ReflectivityState
) -> Literal["fitting", "error"]:
    """
    Route after modeling node.
    """
    if state.get("error"):
        return "error"

    if not state.get("current_model"):
        return "error"

    return "fitting"


def route_after_fitting(
    state: ReflectivityState
) -> Literal["evaluation", "error"]:
    """
    Route after fitting node.
    """
    if state.get("error"):
        return "error"

    return "evaluation"


def route_after_evaluation(
    state: ReflectivityState
) -> Literal["modeling", "complete", "error"]:
    """
    Route after evaluation node.

    Options:
    - modeling: Fit quality is poor, loop back to regenerate model with LLM
    - complete: Fit is acceptable or max iterations reached
    - error: Evaluation failed
    """
    if state.get("error"):
        return "error"

    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 5)

    # Check if we've reached max iterations
    if iteration >= max_iter:
        return "complete"

    # Get suggestions from evaluation
    fit_results = state.get("fit_results", [])
    if fit_results:
        latest = fit_results[-1]
        if latest.get("issues"):
            return "modeling"

    return "complete"
