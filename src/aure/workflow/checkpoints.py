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
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

logger = logging.getLogger(__name__)

# Node execution order for reference
NODE_ORDER = ["intake", "analysis", "modeling", "fitting", "evaluation"]


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
        ├── models/
        │   ├── model_initial.py
        │   └── model_refined_iter1.py
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
        self.models_dir = self.output_dir / "models"
        self.refl1d_output_dir = self.output_dir / "refl1d_output"

        self._checkpoint_counter = 0
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
        self.models_dir.mkdir(exist_ok=True)
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
        self.models_dir.mkdir(exist_ok=True)
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
        self._update_run_info(filename, node_name, iteration)

        # Save model if present
        if state.get("current_model"):
            self._save_model(
                state["current_model"],
                node_name,
                iteration,
                checkpoint_number=self._checkpoint_counter,
            )

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

        # Write model_final.py with best-fit parameters baked in
        try:
            self._save_final_model(state)
        except Exception as exc:
            logger.warning("[CHECKPOINT] Could not write model_final.py: %s", exc)

        # Copy the best-fit problem.json to the top-level output directory
        try:
            self._copy_best_problem_json(state)
        except Exception as exc:
            logger.warning("[CHECKPOINT] Could not copy best problem.json: %s", exc)

    # ------------------------------------------------------------------
    # Final model with fitted parameters
    # ------------------------------------------------------------------

    def _save_final_model(self, state: Dict[str, Any]):
        """Write ``models/model_final.py`` with fitted parameter values.

        For JSON ``ModelDefinition`` dicts, uses ``export_model_script``.
        For legacy script strings, patches values via regex and strips
        ``.range()`` calls.
        """
        model = state.get("best_model") or state.get("current_model")
        if not model:
            return

        fit_results = state.get("fit_results", [])
        if not fit_results:
            return

        # Use the fit result that matches best_chi2 (not necessarily the last)
        best_chi2_val = state.get("best_chi2")
        best_fit = None
        if best_chi2_val is not None:
            for fr in fit_results:
                if fr.get("chi_squared") == best_chi2_val:
                    best_fit = fr
                    break
        if best_fit is None:
            best_fit = min(
                fit_results, key=lambda f: f.get("chi_squared", float("inf"))
            )

        params: dict = best_fit.get("parameters", {})
        uncertainties: dict = best_fit.get("uncertainties") or {}
        chi2 = best_fit.get("chi_squared")
        method = best_fit.get("method", "unknown")

        if not params:
            return

        out_path = self.models_dir / "model_final.py"

        if isinstance(model, dict):
            # New JSON path
            from aure.nodes.model_builder import export_model_script

            script = export_model_script(
                model,
                fitted_params=params,
                fitted_uncertainties=uncertainties,
                include_ranges=False,
            )
            # Also save the JSON definition with fitted values applied
            json_path = self.models_dir / "model_final.json"
            final_def = dict(model)
            final_def["fitted_parameters"] = params
            final_def["fitted_uncertainties"] = uncertainties
            self._save_json(json_path, final_def)
        else:
            # Legacy script path
            script = self._patch_model_parameters(str(model), params)
            script = self._strip_range_calls(script)

        # Prepend a header with fit metadata
        header_lines = [
            "# " + "=" * 68,
            f"# model_final.py — best-fit result (chi2 = {chi2:.4f}, method = {method})",
            "#",
            "# Parameter values below are the optimised values from the fit.",
            "# .range() constraints have been removed; each line shows the",
            "# original range as a comment for reference.",
        ]
        if uncertainties:
            header_lines.append("#")
            header_lines.append("# Uncertainties (1-sigma):")
            for pname, unc in uncertainties.items():
                header_lines.append(f"#   {pname}: \u00b1{unc:.4f}")
        header_lines.append("# " + "=" * 68)
        header_lines.append("")

        script = "\n".join(header_lines) + "\n" + script

        out_path.write_text(script)
        logger.info(f"[CHECKPOINT] Saved final model: {out_path}")

    def _copy_best_problem_json(self, state: Dict[str, Any]):
        """Copy ``problem.json`` from the best fit's refl1d output to the
        top-level output directory.

        Locates the fit iteration that produced the best chi-squared, finds
        the corresponding ``refl1d_output/fit_iter{N}_{method}/problem.json``,
        and copies it to ``<output_dir>/problem.json``.
        """
        import shutil

        fit_results = state.get("fit_results") or []
        if not fit_results:
            return

        best_chi2 = state.get("best_chi2")
        if best_chi2 is None:
            return

        # Find the fit result that matches best_chi2
        best_fit = None
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

        src = self.refl1d_output_dir / f"fit_iter{iteration}_{method}" / "problem.json"
        if not src.exists():
            logger.warning("[CHECKPOINT] Best-fit problem.json not found: %s", src)
            return

        dst = self.output_dir / "problem.json"
        shutil.copy2(src, dst)
        logger.info(f"[CHECKPOINT] Copied best-fit problem.json → {dst}")

    @staticmethod
    def _patch_model_parameters(script: str, params: dict) -> str:
        """Substitute fitted values into ``SLD(name=..., rho=...)`` and
        sample stack ``material(thickness, interface)`` definitions.

        Refl1d names parameters like ``<material> <attribute>``
        (e.g. ``copper thickness``, ``SiO2 rho``).  We rebuild a lookup
        keyed on ``(material_name, attribute)`` and patch matching lines.
        """
        # Build {(material, attr): value} lookup
        lookup: Dict[tuple, float] = {}
        for pname, value in params.items():
            # Handle 'intensity <probe_name>' specially
            if pname.startswith("intensity "):
                lookup[("probe", "intensity")] = value
                continue
            parts = pname.rsplit(" ", 1)
            if len(parts) == 2:
                lookup[(parts[0], parts[1])] = value

        lines = script.split("\n")
        new_lines: list[str] = []

        for line in lines:
            new_line = line

            # --- SLD(name="<mat>", rho=<val>) ---------------------------------
            m = re.match(
                r'^(\s*\w+\s*=\s*SLD\(\s*name\s*=\s*["\'])(\w+)(["\']\s*,\s*rho\s*=\s*)'
                r"([\d.eE+-]+)(.*)",
                line,
            )
            if m:
                mat_name = m.group(2)
                key = (mat_name, "rho")
                if key in lookup:
                    new_line = (
                        f"{m.group(1)}{mat_name}{m.group(3)}{lookup[key]}{m.group(5)}"
                    )

            # --- sample stack: material(thickness, interface) -----------------
            m = re.match(
                r"^(\s*[|]?\s*)(\w+)\(\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\)(.*)",
                line,
            )
            if m:
                var_name = m.group(2)
                # Resolve variable name → material name via earlier assignment
                mat_name = _resolve_material_name(script, var_name)
                thick_key = (mat_name, "thickness")
                iface_key = (mat_name, "interface")
                thickness = lookup.get(thick_key, m.group(3))
                interface = lookup.get(iface_key, m.group(4))
                new_line = (
                    f"{m.group(1)}{var_name}({thickness}, {interface}){m.group(5)}"
                )

            new_lines.append(new_line)

        return "\n".join(new_lines)

    @staticmethod
    def _strip_range_calls(script: str) -> str:
        """Replace ``sample[i].attr.range(lo, hi)`` lines with comments."""

        def _replace(m: re.Match) -> str:
            indent = m.group(1)
            target = m.group(2)
            args = m.group(3)
            comment = m.group(4) or ""
            return f"{indent}# {target}.range({args}){comment}"

        return re.sub(
            r"^(\s*)(sample\[\d+\]\.[\w.]+|probe\.[\w.]+)\.range\(([^)]+)\)(.*)",
            _replace,
            script,
            flags=re.MULTILINE,
        )

    def _save_model(
        self,
        model: object,
        node_name: str,
        iteration: int,
        checkpoint_number: int | None = None,
    ):
        """Save model to models directory.

        For new JSON ``ModelDefinition`` dicts, saves both a ``.json`` and a
        ``.py`` (via ``export_model_script``).  For legacy script strings,
        saves only the ``.py`` file.

        File names mirror the checkpoint numbering so that checkpoint
        ``003_modeling.json`` produces ``003_model_modeling.json``.
        """
        prefix = f"{checkpoint_number:03d}_" if checkpoint_number else ""

        if node_name == "modeling" and iteration == 0:
            base = f"{prefix}model_initial"
        elif node_name == "refinement":
            base = f"{prefix}model_refinement_iter{iteration}"
        elif iteration > 0:
            base = f"{prefix}model_{node_name}_iter{iteration}"
        else:
            base = f"{prefix}model_{node_name}"

        if isinstance(model, dict):
            # JSON path — save definition as JSON
            json_path = self.models_dir / f"{base}.json"
            self._save_json(json_path, model)
            # Also export a readable .py for inspection
            try:
                from aure.nodes.model_builder import export_model_script

                script = export_model_script(model)
                py_path = self.models_dir / f"{base}.py"
                py_path.write_text(script)
            except Exception as exc:
                logger.debug("Could not export model script for %s: %s", base, exc)
        else:
            # Legacy script string
            model_path = self.models_dir / f"{base}.py"
            model_path.write_text(str(model))

    def _save_checkpoint_log(
        self,
        checkpoint_path: Path,
        state: Dict[str, Any],
        node_name: str,
        iteration: int,
    ):
        """Write a companion ``.md`` file with human-readable messages.

        The file lives next to the checkpoint JSON and shares its name,
        e.g. ``003_modeling.md``.
        """
        messages = state.get("messages") or []
        if not messages:
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

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            lines.append(f"## [{role}]")
            lines.append("")
            lines.append(content)
            lines.append("")

        md_path.write_text("\n".join(lines), encoding="utf-8")

    def _update_run_info(self, checkpoint_file: str, node_name: str, iteration: int):
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


def _resolve_material_name(script: str, var_name: str) -> str:
    """Find the ``name=`` argument in the SLD assignment for *var_name*."""
    m = re.search(
        rf'^\s*{re.escape(var_name)}\s*=\s*SLD\(\s*name\s*=\s*["\']([\w]+)["\']',
        script,
        re.MULTILINE,
    )
    return m.group(1) if m else var_name


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
