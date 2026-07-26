"""
Checkpoint system for the reflectivity analysis workflow.

This module provides functionality to save and load workflow state checkpoints,
enabling:
- Saving state after each node for debugging and analysis
- Restarting workflows from a specific checkpoint
- Reviewing intermediate results
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

# Node execution order for reference
NODE_ORDER = ["intake", "analysis", "modeling", "fitting", "evaluation", "finalize"]


class CheckpointManager:
    """
    Manages workflow checkpoints.

    Checkpoints are saved as JSON files in a structured directory:

        output_dir/
        ├── run_info.json           # Metadata about the run
        ├── checkpoints/
        │   ├── 001_intake.json
        │   ├── 002_analysis.json
        │   ├── 003_modeling.json
        │   ├── 004_fitting.json
        │   ├── 005_evaluation.json
        │   └── 006_refinement_iter1.json
        ├── refl1d_output/          # refl1d fitting output (problem.json, etc.)
        └── final_state.json
    """

    def __init__(self, output_dir: str, run_id: Optional[str] = None):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Directory for checkpoints and results
            run_id: Optional run identifier. If not provided, uses timestamp.
        """
        self.output_dir = Path(output_dir)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory structure
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.refl1d_output_dir = self.output_dir / "refl1d_output"

        self._checkpoint_counter = 0
        self._message_offset = 0
        self._initialized = False

    def initialize(
        self, initial_state: Dict[str, Any], data_file: str, sample_description: str
    ):
        """
        Initialize the checkpoint directory for a new run.

        Args:
            initial_state: Initial workflow state
            data_file: Path to data file
            sample_description: User's sample description
        """
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.refl1d_output_dir.mkdir(exist_ok=True)

        # Save run info
        run_info = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "data_file": str(data_file),
            "sample_description": sample_description,
            "hypothesis": initial_state.get("hypothesis"),
            "checkpoints": [],
        }
        # Record an explicit model_name up front when available; the fitting
        # node fills in a resolved one (via the state) once it runs.
        explicit_name = initial_state.get("model_name") or (
            initial_state.get("user_config") or {}
        ).get("model_name")
        if explicit_name:
            run_info["model_name"] = explicit_name
        # Persist co-refinement file list (paths + labels only, no data arrays)
        raw_df = initial_state.get("data_files") or []
        if raw_df:
            run_info["data_files"] = [
                {"file": str(df.get("file", "")), "label": df.get("label", "")}
                for df in raw_df
            ]
        # Persist explicit, user-named states so the data-assembler can group
        # runs per measurement state (→ one AI-ready record per state) without
        # parsing file names. Each state carries its angles + its conditions.
        raw_states = initial_state.get("states") or []
        if raw_states:
            run_info["states"] = [
                {
                    "name": s.get("name"),
                    "extra_description": s.get("extra_description"),
                    "data_files": [
                        {"file": str(df.get("file", "")), "label": df.get("label", "")}
                        for df in (s.get("data_files") or [])
                    ],
                }
                for s in raw_states
            ]
            # Identity: do the co-refined states denote distinct physical
            # samples (distinct sample_id per state downstream) or one sample
            # measured under several conditions (shared sample_id, the
            # default)? Sourced from the user config; orthogonal to ties and to
            # per-state structure.
            run_info["distinct_sample"] = bool(
                (initial_state.get("user_config") or {}).get("distinct_sample", False)
            )
        self._save_json(self.output_dir / "run_info.json", run_info)

        self._initialized = True
        logger.info(f"[CHECKPOINT] Initialized checkpoint directory: {self.output_dir}")

    def initialize_for_resume(self, state: Dict[str, Any], start_node: str):
        """
        Initialize the checkpoint directory for resuming from a checkpoint.

        Args:
            state: State loaded from checkpoint
            start_node: Node to start from
        """
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.refl1d_output_dir.mkdir(exist_ok=True)

        # Load existing run_info or create new one
        run_info_path = self.output_dir / "run_info.json"
        if run_info_path.exists():
            with open(run_info_path) as f:
                run_info = json.load(f)
            # Update checkpoint counter based on existing checkpoints
            self._checkpoint_counter = len(run_info.get("checkpoints", []))
        else:
            run_info = {
                "run_id": self.run_id,
                "started_at": datetime.now().isoformat(),
                "resumed_at": datetime.now().isoformat(),
                "data_file": str(state.get("data_file", "")),
                "sample_description": state.get("sample_description", ""),
                "hypothesis": state.get("hypothesis"),
                "checkpoints": [],
                "resumed_from_node": start_node,
            }
            self._save_json(run_info_path, run_info)

        self._initialized = True
        # Skip messages already written in prior checkpoints
        self._message_offset = len(state.get("messages") or [])
        logger.info(
            f"[CHECKPOINT] Initialized for resume from {start_node}: {self.output_dir}"
        )

    def save_checkpoint(self, state: Dict[str, Any], node_name: str) -> str:
        """
        Save a checkpoint after a node completes.

        Args:
            state: Current workflow state
            node_name: Name of the node that just completed

        Returns:
            Path to saved checkpoint file
        """
        if not self._initialized:
            raise RuntimeError(
                "CheckpointManager not initialized. Call initialize() first."
            )

        self._checkpoint_counter += 1
        iteration = state.get("iteration", 0)

        # Create checkpoint filename
        if node_name in ("fitting", "evaluation", "refinement") and iteration > 0:
            filename = (
                f"{self._checkpoint_counter:03d}_{node_name}_iter{iteration}.json"
            )
        else:
            filename = f"{self._checkpoint_counter:03d}_{node_name}.json"

        checkpoint_path = self.checkpoints_dir / filename

        # Prepare checkpoint data
        checkpoint_data = {
            "checkpoint_id": self._checkpoint_counter,
            "node": node_name,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "state": self._serialize_state(state),
        }

        # Save checkpoint
        self._save_json(checkpoint_path, checkpoint_data)

        # Update run info
        self._update_run_info(filename, node_name, iteration, state)

        # Write companion markdown log with readable messages
        self._save_checkpoint_log(checkpoint_path, state, node_name, iteration)

        logger.info(f"[CHECKPOINT] Saved: {filename}")
        return str(checkpoint_path)

    def save_final_state(self, state: Dict[str, Any]):
        """Save the final workflow state."""
        if not self._initialized:
            return

        final_path = self.output_dir / "final_state.json"
        final_data = {
            "completed_at": datetime.now().isoformat(),
            "success": not state.get("error"),
            "error": state.get("error"),
            "iterations": state.get("iteration", 0),
            "final_chi2": state.get("current_chi2"),
            "state": self._serialize_state(state),
        }
        self._save_json(final_path, final_data)
        logger.info(f"[CHECKPOINT] Saved final state: {final_path}")

        # Copy the best-fit problem.json to the top-level output directory
        try:
            self._copy_best_problem_json(state)
        except Exception as exc:
            logger.warning("[CHECKPOINT] Could not copy best problem.json: %s", exc)

    def _copy_best_problem_json(self, state: Dict[str, Any]):
        """Copy the best fit's serialized FitProblem JSON to the top-level
        output directory as ``problem.json``.

        Prefers the iteration the ``finalize`` node selected, so this artifact
        can never disagree with ``current_model`` in ``final_state.json``.
        Falls back to the best-chi-squared iteration for states written before
        that node existed (e.g. a resumed legacy run). Then finds the exported
        FitProblem JSON in
        ``refl1d_output/fit_iter{N}_{method}/``. bumps names that file after
        the problem (``<model_name>.json``, or ``None.json`` when unnamed), not
        literally ``problem.json`` — so we resolve it by name and fall back to
        the largest non-``*-expt.json`` / non-``*_definition.json`` JSON.
        """
        import shutil

        fit_results = state.get("fit_results") or []
        if not fit_results:
            return

        best_fit = None

        # Preferred: whichever iteration the finalize node selected.
        selection = state.get("final_selection") or {}
        if selection.get("selected"):
            sel_iter = selection.get("iteration")
            for fr in fit_results:
                if fr.get("iteration") == sel_iter:
                    best_fit = fr
                    break

        if best_fit is None:
            best_chi2 = state.get("best_chi2")
            if best_chi2 is None:
                return
            # Find the fit result that matches best_chi2
            for fr in fit_results:
                if fr.get("chi_squared") == best_chi2:
                    best_fit = fr
                    break

        # Fallback: pick the fit with the lowest chi_squared
        if best_fit is None:
            best_fit = min(
                fit_results, key=lambda f: f.get("chi_squared", float("inf"))
            )

        iteration = best_fit.get("iteration", 0)
        method = best_fit.get("method", "dream")
        fit_dir = self.refl1d_output_dir / f"fit_iter{iteration}_{method}"

        src = self._find_problem_json(fit_dir, state)
        if src is None:
            logger.warning(
                "[CHECKPOINT] Best-fit FitProblem JSON not found in %s", fit_dir
            )
            return

        dst = self.output_dir / "problem.json"
        shutil.copy2(src, dst)
        logger.info(
            "[CHECKPOINT] Copied best-fit problem.json (%s) → %s", src.name, dst
        )

    @staticmethod
    def _find_problem_json(fit_dir, state: Dict[str, Any]):
        """Locate the serialized FitProblem JSON bumps exported into *fit_dir*.

        Returns the path, or ``None`` if the directory is missing / has no
        candidate. Prefers the model-named file, then ``problem.json``, then
        the largest JSON that is not a per-experiment (``*-expt.json``) or
        model-definition (``*_definition.json``) sidecar.
        """
        if not fit_dir.is_dir():
            return None

        model_name = (state.get("user_config") or {}).get("model_name")
        for preferred in (f"{model_name}.json" if model_name else None, "problem.json"):
            if preferred:
                candidate = fit_dir / preferred
                if candidate.is_file():
                    return candidate

        sidecar = ("-expt.json", "_definition.json")
        candidates = [
            p
            for p in fit_dir.glob("*.json")
            if not any(p.name.endswith(s) for s in sidecar)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_size)

    def _save_checkpoint_log(
        self,
        checkpoint_path: Path,
        state: Dict[str, Any],
        node_name: str,
        iteration: int,
    ):
        """Write a companion ``.md`` file with human-readable messages.

        Only messages added since the previous checkpoint are written.
        The file lives next to the checkpoint JSON and shares its name,
        e.g. ``003_modeling.md``.
        """
        messages = state.get("messages") or []
        if not messages:
            return

        new_messages = messages[self._message_offset :]
        self._message_offset = len(messages)

        if not new_messages:
            return

        md_path = checkpoint_path.with_suffix(".md")
        lines = [
            f"# Checkpoint: {node_name} (iteration {iteration})",
            f"_Saved at {datetime.now().isoformat()}_",
            "",
        ]

        active_skills = state.get("active_skills")
        if active_skills:
            lines.append(f"**Active skills:** {', '.join(active_skills)}")
            lines.append("")

        # χ² progression summary for evaluation/fitting checkpoints
        if node_name in ("evaluation", "fitting"):
            chi2_line = self._format_chi2_progression(state)
            if chi2_line:
                lines.append(chi2_line)
                lines.append("")

        for msg in new_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            lines.append(f"## [{role}]")
            lines.append("")
            lines.append(content)
            lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _format_chi2_progression(state: Dict[str, Any]) -> str:
        """Return a one-line χ² progression summary from fit_results."""
        fit_results = state.get("fit_results") or []
        if not fit_results:
            return ""
        values = []
        for i, fr in enumerate(fit_results, 1):
            chi2 = fr.get("chi_squared")
            iteration = fr.get("iteration", i)
            if chi2 is not None:
                values.append(f"iter {iteration}: {chi2:.2f}")
        if not values:
            return ""
        return "**χ² progression:** " + " → ".join(values)

    def _update_run_info(
        self,
        checkpoint_file: str,
        node_name: str,
        iteration: int,
        state: Optional[Dict[str, Any]] = None,
    ):
        """Update run_info.json with new checkpoint."""
        run_info_path = self.output_dir / "run_info.json"
        run_info = json.loads(run_info_path.read_text())

        run_info["checkpoints"].append(
            {
                "file": checkpoint_file,
                "node": node_name,
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
            }
        )
        run_info["last_updated"] = datetime.now().isoformat()

        # Capture the model name once the fitting node resolves it onto state.
        if state and state.get("model_name") and not run_info.get("model_name"):
            run_info["model_name"] = state["model_name"]

        self._save_json(run_info_path, run_info)

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert state to JSON-serializable format."""
        serialized = {}

        for key, value in state.items():
            if value is None:
                serialized[key] = None
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                serialized[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v for v in value
                ]
            elif isinstance(value, dict):
                serialized[key] = self._serialize_dict(value)
            elif isinstance(value, (int, float, str, bool)):
                serialized[key] = value
            else:
                # Try to convert, fall back to string representation
                try:
                    serialized[key] = json.loads(json.dumps(value))
                except (TypeError, ValueError):
                    serialized[key] = str(value)

        return serialized

    def _serialize_dict(self, d: Dict) -> Dict:
        """Recursively serialize a dictionary."""
        result = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, dict):
                result[k] = self._serialize_dict(v)
            elif isinstance(v, (list, tuple)):
                result[k] = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
            else:
                result[k] = v
        return result

    def _save_json(self, path: Path, data: Dict):
        """Save data as JSON with pretty formatting.

        Uses ``ensure_ascii=False`` so that Unicode characters (Å, χ², etc.)
        are written directly rather than as ``\\uXXXX`` escapes.

        Message ``content`` strings are split into arrays of lines so that
        multi-line messages are easy to read in the raw JSON.
        """
        serializable = _format_messages_for_json(data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str, ensure_ascii=False)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint JSON file

        Returns:
            Checkpoint data including state
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(path, encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        # Rejoin message content that was split into line arrays for readability
        _rejoin_message_content(checkpoint_data)

        logger.info(f"[CHECKPOINT] Loaded: {checkpoint_path}")
        return checkpoint_data

    @classmethod
    def list_checkpoints(cls, output_dir: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints in a run directory.

        Args:
            output_dir: Path to output directory

        Returns:
            List of checkpoint info dictionaries
        """
        run_info_path = Path(output_dir) / "run_info.json"
        if not run_info_path.exists():
            return []

        with open(run_info_path) as f:
            run_info = json.load(f)

        return run_info.get("checkpoints", [])

    @classmethod
    def get_checkpoint_for_node(
        cls, output_dir: str, node_name: str, iteration: int = 0
    ) -> Optional[str]:
        """
        Find the checkpoint file for a specific node.

        Args:
            output_dir: Path to output directory
            node_name: Name of the node (e.g., "fitting", "evaluation")
            iteration: Iteration number (for nodes that run multiple times)

        Returns:
            Path to checkpoint file, or None if not found
        """
        checkpoints = cls.list_checkpoints(output_dir)

        for cp in checkpoints:
            if cp["node"] == node_name and cp.get("iteration", 0) == iteration:
                return str(Path(output_dir) / "checkpoints" / cp["file"])

        return None


def get_restart_state(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get state suitable for restarting workflow from a checkpoint.

    This loads the checkpoint and prepares the state for continuing
    from the next node in the workflow.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        State dictionary ready for workflow restart
    """
    checkpoint_data = CheckpointManager.load_checkpoint(checkpoint_path)
    state = checkpoint_data["state"]

    # Clear any error state
    state["error"] = None

    # The state is ready to continue from where it left off
    # The current_node field indicates where we are

    logger.info(
        f"[CHECKPOINT] Prepared restart state from node: {checkpoint_data['node']}"
    )
    return state


def get_node_after(node_name: str) -> Optional[str]:
    """
    Get the next node in the workflow after the given node.

    Args:
        node_name: Current node name

    Returns:
        Next node name, or None if at end
    """
    try:
        idx = NODE_ORDER.index(node_name)
        if idx < len(NODE_ORDER) - 1:
            return NODE_ORDER[idx + 1]
    except ValueError:
        pass
    return None


# ======================================================================
# JSON formatting helpers
# ======================================================================


def _format_messages_for_json(data: object) -> object:
    """Split message ``content`` strings into arrays of lines.

    Walks the data tree looking for message dicts (containing ``role``
    and ``content`` keys).  When ``content`` is a multi-line string it
    is replaced with a list of lines so the JSON file is human-readable.

    Returns a *new* object — the original is not mutated.
    """
    if isinstance(data, dict):
        out: dict = {}
        for k, v in data.items():
            out[k] = _format_messages_for_json(v)
        # Detect a message dict and split its content
        if "role" in out and "content" in out and isinstance(out["content"], str):
            if "\n" in out["content"]:
                out["content"] = out["content"].split("\n")
        return out
    if isinstance(data, list):
        return [_format_messages_for_json(item) for item in data]
    return data


def _rejoin_message_content(data: object) -> None:
    """Rejoin message ``content`` arrays back into strings (in-place).

    Reverses the transformation done by :func:`_format_messages_for_json`
    so that loaded checkpoint state has regular string content.
    """
    if isinstance(data, dict):
        if "role" in data and "content" in data and isinstance(data["content"], list):
            data["content"] = "\n".join(data["content"])
        for v in data.values():
            _rejoin_message_content(v)
    elif isinstance(data, list):
        for item in data:
            _rejoin_message_content(item)
