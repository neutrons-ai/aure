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
from ..nodes import (
    intake,
    analysis,
    modeling,
    fitting,
    evaluation,
    finalize,
    final_fit,
    routing,
)
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


# Node execution order (evaluation routes back to modeling for refinement;
# finalize is the terminal packaging step and is listed so it is a legal
# ``start_node`` — re-running selection on an old run needs no refitting)
NODE_ORDER = ["intake", "analysis", "modeling", "fitting", "evaluation", "finalize"]

# Node function registry. ``final_fit`` is intentionally absent from NODE_ORDER
# and ROUTING_FUNCTIONS: it is not a loop node and can never be routed to. It is
# invoked once, explicitly, in the terminal block after finalize.
NODE_FUNCTIONS = {
    "intake": intake.intake_node,
    "analysis": analysis.analysis_node,
    "modeling": modeling.modeling_node,
    "fitting": fitting.fitting_node,
    "evaluation": evaluation.evaluation_node,
    "finalize": finalize.finalize_node,
    "final_fit": final_fit.final_fit_node,
}

# Routing function registry. ``finalize`` deliberately has no entry: the loop
# breaks when a node has no router, which is exactly what makes it terminal.
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
    states: Optional[list[dict]] = None,
    chi2_max: Optional[float] = None,
    chi2_min: Optional[float] = None,
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
        chi2_max: χ² acceptance ceiling for this run; falls back to ``CHI2_MAX``.
        chi2_min: χ² acceptance floor for this run; falls back to ``CHI2_MIN``.

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
        states=states,
    )
    if interactive:
        initial_state["interactive"] = True
    # Seeding the window rather than exporting CHI2_MAX/CHI2_MIN: the web UI drives
    # this from a background thread, where mutating the process environment would
    # race any other request. The runner pins the window only when it is unset, so
    # a seeded value is what the run keeps — and what its checkpoints record, so a
    # resume inherits it too.
    seeded_chi2_max = None
    if chi2_max is not None:
        seeded_chi2_max = evaluation._validated_chi2_max(chi2_max, "chi2_max (run_analysis)")
        if seeded_chi2_max is not None:
            initial_state["chi2_max"] = seeded_chi2_max

    if chi2_min is not None:
        seeded_chi2_min = evaluation._validated_chi2_min(chi2_min, "chi2_min (run_analysis)")
        effective_chi2_max = seeded_chi2_max if seeded_chi2_max is not None else evaluation._get_chi2_max()
        if seeded_chi2_min is not None and seeded_chi2_min < effective_chi2_max:
            initial_state["chi2_min"] = seeded_chi2_min

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
    states: Optional[list[dict]] = None,
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
        states=states,
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
    # Pin the χ² acceptance threshold into the state on the first pass only. It
    # deterministically terminates the refinement loop, so a resumed or
    # restarted run must keep the threshold it was launched with rather than
    # inherit whatever CHI2_MAX the resuming process happens to carry. The
    # runner is the env → state bridge because ``state.py`` must not import from
    # the nodes package.
    if state.get("chi2_max") is None:
        state["chi2_max"] = evaluation._get_chi2_max()
        logger.info(
            "[RUNNER] χ² acceptance threshold for this run: %.3f",
            state["chi2_max"],
        )
    # The floor is pinned for the same reason and against the *pinned* ceiling,
    # so the "floor below ceiling" rule is checked against this run's window
    # rather than the resuming shell's. ``0.0`` is a legal pinned value (no
    # floor), which is why absence is tested with ``is None``.
    if state.get("chi2_min") is None:
        state["chi2_min"] = evaluation._get_chi2_min(chi2_max=state.get("chi2_max"))
        logger.info(
            "[RUNNER] χ² acceptance floor for this run: %.3f%s",
            state["chi2_min"],
            " (disabled)" if not state["chi2_min"] else "",
        )
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
        # An accepting verdict normally needs no confirmation: the evaluator
        # agreed the fit is done. A *clamped* accept is different — the
        # deterministic χ² threshold overrode an evaluator that objected, and
        # those objections go into the report unaddressed. That is exactly the
        # decision an interactive run exists to put in front of the user, so it
        # still gets its review pause even though `workflow_complete` is set.
        if (
            pause_callback
            and state.get("interactive")
            and current_node == "evaluation"
            and (not state.get("workflow_complete") or state.get("chi2_clamp_accepted"))
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
                                "current_chi2",
                                "iteration",
                                "fit_results",
                            ):
                                if key in cp_state:
                                    state[key] = cp_state[key]
                            # model_history is append-only, so rewinding
                            # fit_results without it would leave abandoned
                            # branches on record — and the terminal model
                            # selection resolves a fit's structure through
                            # model_history. Truncate the two together.
                            if "model_history" in cp_state:
                                state["model_history"] = cp_state["model_history"]
                            # best_* is the run's regression baseline and the
                            # input to final model selection — rewinding to an
                            # earlier checkpoint must not make it worse, or the
                            # true best found is lost for good.
                            cp_best = cp_state.get("best_chi2")
                            incumbent = state.get("best_chi2")
                            if cp_best is not None and (
                                incumbent is None or cp_best < incumbent
                            ):
                                state["best_chi2"] = cp_best
                                if "best_model" in cp_state:
                                    state["best_model"] = cp_state["best_model"]
                            state["workflow_complete"] = False
                            state["finalized"] = False
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
                    # Withdraw a clamped accept. The pause exists here precisely
                    # because the χ² threshold overrode an objecting evaluator, so
                    # answering it with guidance has to be able to reopen the loop —
                    # otherwise the run breaks on `workflow_complete` below and the
                    # feedback is recorded but never acted on. `chi2_clamp_accepted`
                    # is cleared with it so the next evaluation does not pause again
                    # on a stale flag.
                    if state.get("workflow_complete") and state.get(
                        "chi2_clamp_accepted"
                    ):
                        state["workflow_complete"] = False
                        state["chi2_clamp_accepted"] = False
                        logger.info(
                            "[RUNNER] User gave guidance at a clamped accept — "
                            "reopening the refinement loop"
                        )
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

    # ---- Terminal packaging -----------------------------------------
    # The loop has six exits and only two of them consult a routing function:
    # the `complete` / `error` routes, the interactive __STOP__ break, the
    # `stop_after` break, the max_total_iterations cap on the `while` itself,
    # and the missing-router break. On top of that, `workflow_complete` breaks
    # out immediately after evaluation, before routing runs at all. Wiring
    # finalize as an edge would therefore miss most of them, so run it here
    # where every exit converges. `finalize_node` is idempotent via the
    # `finalized` flag, so this is a no-op if it already ran as a node.
    if not stop_after:
        if not state.get("finalized"):
            updates = run_with_tracing(
                NODE_FUNCTIONS["finalize"], state, "node_finalize", trace_ctx
            )
            _merge_state_updates(state, updates)
            if checkpoint_mgr:
                checkpoint_mgr.save_checkpoint(state, "finalize")
            if checkpoint_callback:
                checkpoint_callback(state, "finalize")

        # ---- Optional final uncertainty fit ---------------------------
        # After finalize has selected the winning model, optionally run one
        # more fit (``FIT_METHOD_FINAL``, typically dream) on it to attach the
        # uncertainties a fast exploration optimizer does not produce. The node
        # self-gates on the env var, so this is an inert, un-checkpointed no-op
        # ({} update) unless the feature is requested. Runs here — the single
        # path both the CLI and the web UI take — so it needs no graph edge. A
        # failure inside is never fatal (it returns a skip record, not a raise),
        # so the run still reports the finalize-selected model.
        #
        # Looked up leniently (unlike the required ``finalize``): it is an
        # optional terminal step, so a caller that swaps in a partial
        # NODE_FUNCTIONS registry simply skips it rather than crashing.
        final_fit_fn = NODE_FUNCTIONS.get("final_fit")
        if final_fit_fn is not None:
            updates = run_with_tracing(final_fit_fn, state, "node_final_fit", trace_ctx)
            if updates:
                _merge_state_updates(state, updates)
                if checkpoint_mgr:
                    checkpoint_mgr.save_checkpoint(state, "final_fit")
                if checkpoint_callback:
                    checkpoint_callback(state, "final_fit")

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
    # The restarted run must pick its own winner over the extended set of
    # iterations, so drop the previous run's terminal selection.
    new_state["finalized"] = False

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

    # Clear any error from previous run, and let the resumed run re-select its
    # final model over whatever iterations it ends up with.
    state["error"] = None
    state["finalized"] = False

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
