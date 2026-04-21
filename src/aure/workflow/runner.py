"""
Workflow execution and orchestration.

This module provides functions for running the reflectivity analysis workflow:
- run_analysis: Main entry point for full analysis
- run_workflow_with_checkpoints: Step-by-step execution with checkpoint support
- run_from_checkpoint: Resume workflow from a saved checkpoint
"""

import json as _json
import logging
from typing import Optional, Callable, Dict, Any
from pathlib import Path

from ..state import ReflectivityState, Message, create_initial_state
from ..nodes import intake, analysis, modeling, fitting, evaluation, routing
from .checkpoints import CheckpointManager, get_node_after
from .tracing import get_trace_context, run_with_tracing, TracedWorkflow


def _rejoin_messages(data: object) -> None:
    """Rejoin message content arrays back to strings (in-place)."""
    if isinstance(data, dict):
        if "role" in data and "content" in data and isinstance(data["content"], list):
            data["content"] = "\n".join(data["content"])
        for v in data.values():
            _rejoin_messages(v)
    elif isinstance(data, list):
        for item in data:
            _rejoin_messages(item)


logger = logging.getLogger(__name__)


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


def _load_checkpoint_by_iteration(
    output_dir: Optional[str], iteration: int
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint state by matching its iteration number.

    Returns the checkpoint state dict, or *None* if not found.
    """
    if not output_dir:
        return None
    cp_dir = Path(output_dir) / "checkpoints"
    if not cp_dir.exists():
        return None
    for cp_file in sorted(cp_dir.glob("*.json")):
        try:
            cp_data = _json.loads(cp_file.read_text(encoding="utf-8"))
            cp_state = cp_data.get("state", cp_data)
            # Rejoin message content arrays that were split for readability
            _rejoin_messages(cp_state)
            if cp_state.get("iteration") == iteration:
                return cp_state
        except Exception:
            continue
    return None


def run_analysis(
    data_file: str,
    sample_description: str,
    hypothesis: str = None,
    max_iterations: int = 5,
    output_dir: Optional[str] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any], str], None]] = None,
    user_config: Optional[dict] = None,
    interactive: bool = False,
    pause_callback: Optional[Callable[[Dict[str, Any], str], Optional[str]]] = None,
    data_files: Optional[list[dict]] = None,
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
        user_config: Optional user-supplied YAML configuration dict
        interactive: Enable interactive mode (pause after evaluation for user feedback)
        pause_callback: Blocking callback(state, node_name) -> Optional[str] that
            returns user feedback text (or None). Only called when interactive=True.

    Returns:
        Final workflow state with results
    """
    # Create initial state
    initial_state = create_initial_state(
        data_file=data_file,
        sample_description=sample_description,
        hypothesis=hypothesis,
        max_iterations=max_iterations,
        user_config=user_config,
        data_files=data_files,
    )
    if interactive:
        initial_state["interactive"] = True

    # Run with optional tracing
    with TracedWorkflow(
        data_file, sample_description, hypothesis, max_iterations
    ) as tw:
        result = run_workflow_with_checkpoints(
            initial_state=initial_state,
            output_dir=output_dir,
            checkpoint_callback=checkpoint_callback,
            pause_callback=pause_callback if interactive else None,
        )
        tw.set_result(result)
        return result


def run_prepare(
    data_file: str,
    sample_description: str,
    hypothesis: str = None,
    output_dir: Optional[str] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any], str], None]] = None,
    user_config: Optional[dict] = None,
    data_files: Optional[list[dict]] = None,
) -> ReflectivityState:
    """
    Run only intake → analysis → modeling.

    Produces a ModelDefinition ready to be serialized as a refl1d/bumps
    ``problem.json``.  No fitting or evaluation is performed.

    Args:
        data_file: Path to reflectivity data file
        sample_description: User's description of the sample
        hypothesis: Optional hypothesis to test
        output_dir: Optional directory for checkpoints and results
        checkpoint_callback: Optional callback(state, node_name)
        user_config: Optional user-supplied YAML configuration dict
        data_files: Optional list of DatasetInfo dicts for multi-file co-refinement

    Returns:
        Final workflow state (stopped after modeling)
    """
    initial_state = create_initial_state(
        data_file=data_file,
        sample_description=sample_description,
        hypothesis=hypothesis,
        max_iterations=0,
        user_config=user_config,
        data_files=data_files,
    )

    with TracedWorkflow(data_file, sample_description, hypothesis, 0) as tw:
        result = run_workflow_with_checkpoints(
            initial_state=initial_state,
            output_dir=output_dir,
            checkpoint_callback=checkpoint_callback,
            stop_after="modeling",
        )
        tw.set_result(result)
        return result


def run_workflow_with_checkpoints(
    initial_state: ReflectivityState,
    output_dir: Optional[str] = None,
    checkpoint_callback: Optional[Callable[[Dict[str, Any], str], None]] = None,
    start_node: Optional[str] = None,
    pause_callback: Optional[Callable[[Dict[str, Any], str], Optional[str]]] = None,
    stop_after: Optional[str] = None,
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
        pause_callback: Optional blocking callback for interactive mode.
            Called after evaluation nodes; returns user feedback string or None.

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
        updates = run_with_tracing(node_fn, state, f"node_{current_node}", trace_ctx)

        # Merge updates into state
        _merge_state_updates(state, updates)

        # Save checkpoint
        if checkpoint_mgr:
            checkpoint_mgr.save_checkpoint(state, current_node)

        if checkpoint_callback:
            checkpoint_callback(state, current_node)

        # ---- Interactive pause after evaluation -------------------
        if (
            pause_callback
            and state.get("interactive")
            and current_node == "evaluation"
            and not state.get("workflow_complete")
            and not state.get("error")
        ):
            logger.info("[RUNNER] Interactive mode — waiting for user feedback")
            feedback = pause_callback(state, current_node)
            if feedback == "__STOP__":
                logger.info("[RUNNER] User requested stop")
                state["workflow_complete"] = True
                break
            elif feedback:
                # Handle structured feedback (dict with advanced options)
                feedback_text = feedback
                if isinstance(feedback, dict):
                    feedback_text = feedback.get("feedback") or None
                    if feedback.get("dream_steps"):
                        state["fit_steps"] = int(feedback["dream_steps"])
                        state["fit_burn"] = int(feedback["dream_steps"])
                        logger.info(
                            "[RUNNER] User set DREAM steps to %d", state["fit_steps"]
                        )
                    if feedback.get("restart_checkpoint"):
                        # Load checkpoint state and merge relevant fields
                        cp_iter = int(feedback["restart_checkpoint"])
                        logger.info(
                            "[RUNNER] User requested restart from checkpoint iteration %d",
                            cp_iter,
                        )
                        cp_state = _load_checkpoint_by_iteration(
                            state.get("output_dir"), cp_iter
                        )
                        if cp_state:
                            # Restore model and fit state from checkpoint
                            for key in (
                                "current_model",
                                "best_model",
                                "best_chi2",
                                "current_chi2",
                                "iteration",
                                "fit_results",
                            ):
                                if key in cp_state:
                                    state[key] = cp_state[key]
                            state["workflow_complete"] = False
                            state["error"] = None
                            logger.info(
                                "[RUNNER] Restored state from checkpoint iteration %d",
                                cp_iter,
                            )

                if feedback_text:
                    state["pending_user_feedback"] = feedback_text
                    state["messages"] = state.get("messages", []) + [
                        Message(
                            role="user",
                            content=feedback_text,
                            timestamp=None,
                        )
                    ]
                    logger.info(
                        "[RUNNER] Received user feedback: %s", feedback_text[:100]
                    )
                else:
                    state["pending_user_feedback"] = None
            else:
                state["pending_user_feedback"] = None

        # Check for error or completion
        if state.get("error"):
            break

        if state.get("workflow_complete"):
            break

        # Stop after a specific node (e.g. "modeling") if requested
        if stop_after and current_node == stop_after:
            state["workflow_complete"] = True
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


def prepare_state_for_restart(
    state: Dict[str, Any],
    user_insight: str,
    restart_from: str = "modeling",
    extra_iterations: int = 1,
) -> Dict[str, Any]:
    """
    Prepare a completed workflow state for restart with new user insight.

    This resets completion/error flags, injects the user's new guidance
    into the conversation, and grants additional iterations so the
    refinement loop can run again.

    Args:
        state: A completed workflow state (e.g. loaded from final_state.json)
        user_insight: Free-text guidance from the user describing what to
            change or try differently.
        restart_from: Node to restart from. ``"modeling"`` (default) re-builds
            and re-fits the model.  ``"analysis"`` re-analyses features and
            re-builds from scratch.  ``"fitting"`` re-fits directly with the
            current model parameters (skips LLM refinement).
        extra_iterations: Number of additional iterations to allow beyond
            those already consumed (default: 3).

    Returns:
        A **new** state dict ready to be passed to
        ``run_workflow_with_checkpoints(start_node=restart_from)``.
    """
    restart_from = restart_from if restart_from in NODE_ORDER else "modeling"
    new_state = dict(state)

    # ---- Clear completion / error flags ----------------------------
    new_state["workflow_complete"] = False
    new_state["error"] = None

    # ---- Grant more iteration budget -------------------------------
    used = new_state.get("iteration", 0)
    new_state["max_iterations"] = used + extra_iterations

    # ---- Inject insight as user feedback ---------------------------
    new_state["pending_user_feedback"] = user_insight
    new_state["messages"] = new_state.get("messages", []) + [
        Message(
            role="user",
            content=f"[Restart with new insight] {user_insight}",
            timestamp=None,
        )
    ]

    # If restarting from analysis, also clear parsed sample / features
    # so they are regenerated with the new insight in mind.
    if restart_from == "analysis":
        new_state["parsed_sample"] = None
        new_state["extracted_features"] = None

    logger.info(
        "[RUNNER] State prepared for restart from '%s' with %d extra iterations",
        restart_from,
        extra_iterations,
    )
    return new_state


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

    Some fields accumulate (messages, model_history, fit_results,
    llm_calls), while others are overwritten.

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
        elif key == "llm_calls" and isinstance(value, list):
            # Accumulate LLM call records across nodes
            state["llm_calls"] = state.get("llm_calls", []) + value
        else:
            state[key] = value
