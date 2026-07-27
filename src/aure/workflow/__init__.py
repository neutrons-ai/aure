"""
Workflow package for the reflectivity analysis agent.

This package provides the workflow that analyzes neutron reflectivity data
through an iterative refinement process. Execution is a hand-written state
machine in ``runner.py`` (``run_workflow_with_checkpoints``); there is no
external graph framework.

Key functions:
- run_analysis: Main entry point for running a full analysis
- run_prepare: Intake → analysis → modeling only (no fitting)
- run_workflow_with_checkpoints: Run with checkpoint saving
- run_from_checkpoint: Resume from a saved checkpoint

Example usage:
    from aure.workflow import run_analysis

    result = run_analysis(
        data_file="data/sample.ort",
        sample_description="Ni thin film on silicon substrate",
        hypothesis="Single layer model",
        max_iterations=5,
    )
"""

from .runner import (
    run_analysis,
    run_prepare,
    run_workflow_with_checkpoints,
    run_from_checkpoint,
    NODE_ORDER,
    NODE_FUNCTIONS,
    ROUTING_FUNCTIONS,
)
from .checkpoints import (
    CheckpointManager,
    get_restart_state,
    get_node_after,
)
from .tracing import (
    is_tracing_enabled,
    get_trace_context,
    run_with_tracing,
    TracedWorkflow,
)


__all__ = [
    # Workflow execution
    "run_analysis",
    "run_prepare",
    "run_workflow_with_checkpoints",
    "run_from_checkpoint",
    "NODE_ORDER",
    "NODE_FUNCTIONS",
    "ROUTING_FUNCTIONS",
    # Checkpoints
    "CheckpointManager",
    "get_restart_state",
    "get_node_after",
    # Tracing
    "is_tracing_enabled",
    "get_trace_context",
    "run_with_tracing",
    "TracedWorkflow",
]
