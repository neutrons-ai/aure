"""
LangGraph workflow graph definition for reflectivity analysis.

This module defines the state machine that orchestrates the analysis:
INTAKE → ANALYSIS → MODELING → FITTING → EVALUATION → (MODELING → FITTING) → COMPLETE

When evaluation identifies issues, the workflow loops back to MODELING where
the LLM regenerates the complete model script based on evaluation feedback.

The graph defines:
- Nodes: Processing steps that transform state
- Edges: Transitions between nodes based on routing logic
"""

from langgraph.graph import StateGraph, START, END

from ..state import ReflectivityState
from ..nodes import (
    intake,
    analysis,
    modeling,
    fitting,
    evaluation,
    finalize,
    routing,
)


def create_workflow(include_fitting: bool = True) -> StateGraph:
    """
    Create the reflectivity analysis workflow graph.

    Args:
        include_fitting: If True (default), include fitting/evaluation nodes.
                        If False, stop at modeling (useful for quick initialization).

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph with our state type
    workflow = StateGraph(ReflectivityState)

    # ========== Add Nodes ==========
    workflow.add_node("intake", intake.intake_node)
    workflow.add_node("analysis", analysis.analysis_node)
    workflow.add_node("modeling", modeling.modeling_node)

    if include_fitting:
        workflow.add_node("fitting", fitting.fitting_node)
        workflow.add_node("evaluation", evaluation.evaluation_node)
        # Terminal packaging step: pick the best fit of the run and promote it
        # to current_model / current_chi2. Every exit from the fitting portion
        # of the graph goes through it — including the error edges, because a
        # late LLM failure still leaves a run's worth of good fits to choose
        # from. It no-ops when there is nothing to select.
        workflow.add_node("finalize", finalize.finalize_node)

    # ========== Add Edges ==========
    # Start → Intake
    workflow.add_edge(START, "intake")

    # Intake → Analysis (or error)
    workflow.add_conditional_edges(
        "intake",
        routing.route_after_intake,
        {
            "analysis": "analysis",
            "error": END,
        },
    )

    # Analysis → Modeling
    workflow.add_conditional_edges(
        "analysis",
        routing.route_after_analysis,
        {
            "modeling": "modeling",
            "error": END,
        },
    )

    if include_fitting:
        # Modeling → Fitting (a refinement-iteration error can still have
        # earlier successful fits to package, so route it through finalize)
        workflow.add_conditional_edges(
            "modeling",
            routing.route_after_modeling,
            {
                "fitting": "fitting",
                "error": "finalize",
            },
        )

        # Fitting → Evaluation
        workflow.add_conditional_edges(
            "fitting",
            routing.route_after_fitting,
            {
                "evaluation": "evaluation",
                "error": "finalize",
            },
        )

        # Evaluation → Modeling (loop back for refinement) or Finalize
        workflow.add_conditional_edges(
            "evaluation",
            routing.route_after_evaluation,
            {
                "modeling": "modeling",
                "fitting": "fitting",
                "complete": "finalize",
                "error": "finalize",
            },
        )

        # Finalize is terminal
        workflow.add_edge("finalize", END)
    else:
        # Without fitting, modeling is the end
        workflow.add_edge("modeling", END)

    # Compile and return
    return workflow.compile()


def create_workflow_app():
    """Create workflow app for streaming/interactive execution."""
    return create_workflow()
