"""
Workflow execution and orchestration.

This module provides functions for running the reflectivity analysis workflow:
- run_analysis: Main entry point for full analysis
- run_workflow_with_checkpoints: Step-by-step execution with checkpoint support
- run_from_checkpoint: Resume workflow from a saved checkpoint
"""

from typing import Optional, Callable, Dict, Any
from pathlib import Path

from ..state import ReflectivityState, create_initial_state
from ..nodes import intake, analysis, modeling, fitting, evaluation, routing
from .checkpoints import CheckpointManager, get_node_after
from .tracing import get_trace_context, run_with_tracing, TracedWorkflow


# Node execution order (evaluation routes back to modeling for refinement)
NODE_ORDER = ["intake", "analysis", "modeling", "fitting", "evaluation"]

# Node function registry
NODE_FUNCTIONS = {
    "intake": intake.intake_node,
    "analysis": analysis.analysis_node,
    "modeling": modeling.modeling_node,
    "fitting": fitting.fitting_node,
    "evaluation": evaluation.evaluation_node,
}

# Routing function registry
ROUTING_FUNCTIONS = {
    "intake": routing.route_after_intake,
    "analysis": routing.route_after_analysis,
    "modeling": routing.route_after_modeling,
    "fitting": routing.route_after_fitting,
    "evaluation": routing.route_after_evaluation,
}


def run_analysis(
    data_file: str,
    sample_description: str,
    hypothesis: str = None,
    max_iterations: int = 5,
    output_dir: Optional[str] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> ReflectivityState:
    """
    Run the reflectivity analysis workflow.

    Args:
        data_file: Path to reflectivity data file
        sample_description: User's description of the sample
        hypothesis: Optional hypothesis to test
        max_iterations: Maximum refinement iterations (default: 5)
        output_dir: Optional directory for checkpoints and results
        checkpoint_callback: Optional callback(state, node_name) for custom checkpoint handling

    Returns:
        Final workflow state with results
    """
    # Create initial state
    initial_state = create_initial_state(
        data_file=data_file,
        sample_description=sample_description,
        hypothesis=hypothesis,
        max_iterations=max_iterations,
    )

    # Run with optional tracing
    with TracedWorkflow(data_file, sample_description, hypothesis, max_iterations) as tw:
        result = run_workflow_with_checkpoints(
            initial_state=initial_state,
            output_dir=output_dir,
            checkpoint_callback=checkpoint_callback,
        )
        tw.set_result(result)
        return result


def run_workflow_with_checkpoints(
    initial_state: ReflectivityState,
    output_dir: Optional[str] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any], str], None]] = None,
    start_node: Optional[str] = None,
) -> ReflectivityState:
    """
    Run workflow with checkpoint support.
    
    This function runs the workflow step by step, saving checkpoints
    after each node completes.
    
    Args:
        initial_state: Starting state (from create_initial_state or loaded checkpoint)
        output_dir: Directory for saving checkpoints
        checkpoint_callback: Optional callback for custom handling
        start_node: Optional node to start from (for restart scenarios)
        
    Returns:
        Final workflow state
    """
    # Setup checkpoint manager if output_dir provided
    checkpoint_mgr = None
    if output_dir:
        checkpoint_mgr = CheckpointManager(output_dir)
        if not start_node:  # New run, initialize
            checkpoint_mgr.initialize(
                initial_state,
                initial_state.get("data_file", ""),
                initial_state.get("sample_description", ""),
            )
        else:
            # Resuming from checkpoint - initialize for continuation
            checkpoint_mgr.initialize_for_resume(
                initial_state,
                start_node,
            )
    
    # Determine starting point
    if start_node and start_node in NODE_ORDER:
        start_idx = NODE_ORDER.index(start_node)
    else:
        start_idx = 0
    
    # Run workflow manually with checkpoints
    state = dict(initial_state)
    if output_dir:
        state["output_dir"] = output_dir
    current_node = NODE_ORDER[start_idx] if start_idx < len(NODE_ORDER) else None
    
    max_total_iterations = 20  # Safety limit
    iteration_count = 0
    
    # Get trace context once (None if tracing disabled)
    trace_ctx = get_trace_context()
    
    while current_node and iteration_count < max_total_iterations:
        iteration_count += 1
        
        # Execute node
        node_fn = NODE_FUNCTIONS.get(current_node)
        if not node_fn:
            break
        
        # Run the node with optional tracing
        updates = run_with_tracing(
            node_fn, state, f"node_{current_node}", trace_ctx
        )
        
        # Merge updates into state
        _merge_state_updates(state, updates)
        
        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(state, current_node)
        
        if checkpoint_callback:
            checkpoint_callback(state, current_node)
        
        # Check for error or completion
        if state.get("error"):
            break
        
        if state.get("workflow_complete"):
            break
        
        # Route to next node
        route_fn = ROUTING_FUNCTIONS.get(current_node)
        if not route_fn:
            break
        
        next_route = route_fn(state)
        
        # Map route to node
        if next_route == "error":
            break
        elif next_route == "complete":
            break
        elif next_route in NODE_ORDER:
            current_node = next_route
        else:
            # Try to find matching node
            current_node = next_route if next_route in NODE_FUNCTIONS else None
    
    # Save final state
    if checkpoint_mgr:
        checkpoint_mgr.save_final_state(state)
    
    return state


def run_from_checkpoint(
    checkpoint_path: str,
    output_dir: Optional[str] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> ReflectivityState:
    """
    Restart workflow from a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint JSON file
        output_dir: Directory for new checkpoints (if different from original)
        checkpoint_callback: Optional callback for checkpoint handling
        
    Returns:
        Final workflow state
    """
    # Load checkpoint
    checkpoint_data = CheckpointManager.load_checkpoint(checkpoint_path)
    state = checkpoint_data["state"]
    completed_node = checkpoint_data["node"]
    
    # Clear any error from previous run
    state["error"] = None
    
    # Determine the next node to run
    next_node = get_node_after(completed_node)
    
    if not next_node:
        # Already at end, return loaded state
        return state
    
    # Use output_dir from checkpoint path if not specified
    if not output_dir:
        output_dir = str(Path(checkpoint_path).parent.parent)
    
    return run_workflow_with_checkpoints(
        initial_state=state,
        output_dir=output_dir,
        checkpoint_callback=checkpoint_callback,
        start_node=next_node,
    )


def _merge_state_updates(state: dict, updates: dict) -> None:
    """
    Merge node updates into the current state.
    
    Some fields accumulate (messages, model_history, fit_results),
    while others are overwritten.
    
    Args:
        state: Current state dict (modified in place)
        updates: Updates from node execution
    """
    for key, value in updates.items():
        if key == "messages" and isinstance(value, list):
            # Accumulate messages
            state["messages"] = state.get("messages", []) + value
        elif key == "model_history" and isinstance(value, list):
            # Accumulate model history
            state["model_history"] = state.get("model_history", []) + value
        elif key == "fit_results" and isinstance(value, list):
            # Accumulate fit results
            state["fit_results"] = state.get("fit_results", []) + value
        else:
            state[key] = value
