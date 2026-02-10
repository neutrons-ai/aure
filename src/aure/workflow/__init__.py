"""
Workflow package for the reflectivity analysis agent.

This package provides the LangGraph-based workflow for analyzing
neutron reflectivity data through an iterative refinement process.

Key functions:
- create_workflow: Create the LangGraph workflow definition
- run_analysis: Main entry point for running a full analysis
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

from .graph import create_workflow, create_workflow_app
from .runner import (
    run_analysis,
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
    # Graph creation
    "create_workflow",
    "create_workflow_app",
    # Workflow execution
    "run_analysis",
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
