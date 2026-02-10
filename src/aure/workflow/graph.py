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
from ..nodes import intake, analysis, modeling, fitting, evaluation, routing


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
        # Modeling → Fitting
        workflow.add_conditional_edges(
            "modeling",
            routing.route_after_modeling,
            {
                "fitting": "fitting",
                "error": END,
            },
        )

        # Fitting → Evaluation
        workflow.add_conditional_edges(
            "fitting",
            routing.route_after_fitting,
            {
                "evaluation": "evaluation",
                "error": END,
            },
        )

        # Evaluation → Modeling (loop back for refinement) or Complete
        workflow.add_conditional_edges(
            "evaluation",
            routing.route_after_evaluation,
            {
                "modeling": "modeling",
                "complete": END,
                "error": END,
            },
        )
    else:
        # Without fitting, modeling is the end
        workflow.add_edge("modeling", END)

    # Compile and return
    return workflow.compile()


def create_workflow_app():
    """Create workflow app for streaming/interactive execution."""
    return create_workflow()
