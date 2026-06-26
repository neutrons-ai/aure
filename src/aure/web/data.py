"""
Data access layer for the web visualisation app.

Reads checkpoint files, run metadata, and model outputs produced by
``aure analyze -o <output_dir>``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RunData:
    """
    Read-only accessor for a single workflow output directory.

    Typical layout on disk::

        output_dir/
        ├── run_info.json
        ├── final_state.json
        ├── checkpoints/
        │   ├── 001_intake.json
        │   └── ...
        └── refl1d_output/
            └── fit_iter0_dream/
                └── problem.json
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self._run_info: Optional[dict] = None
        self._final_state: Optional[dict] = None
        self._checkpoints_cache: Optional[List[dict]] = None
        self._sld_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------

    def get_run_info(self) -> dict:
        """Return contents of ``run_info.json``."""
        if self._run_info is None:
            path = self.output_dir / "run_info.json"
            if path.exists():
                self._run_info = json.loads(path.read_text())
            else:
                self._run_info = {}
        return self._run_info

    def get_final_state(self) -> dict:
        """Return the final workflow state (from ``final_state.json``)."""
        if self._final_state is None:
            path = self.output_dir / "final_state.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                state = data.get("state", data)
                # Rejoin message content arrays written for readability
                _rejoin_message_content(state)
                self._final_state = state
            else:
                # Fall back to the latest checkpoint
                self._final_state = self._load_latest_checkpoint_state()
        return self._final_state

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_all_checkpoints(self) -> List[dict]:
        """Load every checkpoint file and return a list of full checkpoint dicts."""
        if self._checkpoints_cache is not None:
            return self._checkpoints_cache

        cp_dir = self.output_dir / "checkpoints"
        infos = self.get_run_info().get("checkpoints", [])
        result: List[dict] = []

        for info in infos:
            cp_path = cp_dir / info["file"]
            if cp_path.exists():
                cp_data = json.loads(cp_path.read_text())
                cp_data["_info"] = info  # attach the run_info entry
                result.append(cp_data)

        self._checkpoints_cache = result
        return result

    def _load_latest_checkpoint_state(self) -> dict:
        """Return the state dict from the last checkpoint on disk."""
        cps = self._load_all_checkpoints()
        if cps:
            return cps[-1].get("state", {})
        return {}

    # ------------------------------------------------------------------
    # Chi-squared progression
    # ------------------------------------------------------------------

    def get_chi2_progression(self) -> List[dict]:
        """
        Walk checkpoints and return one entry per step.

        Each entry: ``{step, node, iteration, chi2, timestamp, error, llm_calls}``.
        ``chi2`` is ``None`` for nodes that don't produce a fit.
        ``llm_calls`` lists LLM records added during that step.
        """
        cps = self._load_all_checkpoints()
        result: List[dict] = []
        prev_llm_count = 0

        for i, cp in enumerate(cps):
            state = cp.get("state", {})
            info = cp.get("_info", {})
            all_llm = state.get("llm_calls", [])
            step_llm = all_llm[prev_llm_count:]
            prev_llm_count = len(all_llm)
            result.append(
                {
                    "step": i + 1,
                    "node": info.get("node", cp.get("node", "")),
                    "iteration": info.get("iteration", cp.get("iteration", 0)),
                    "chi2": state.get("current_chi2"),
                    "timestamp": info.get("timestamp", cp.get("timestamp")),
                    "error": state.get("error"),
                    "llm_calls": step_llm,
                }
            )

        return result

    # ------------------------------------------------------------------
    # LLM call summary
    # ------------------------------------------------------------------

    def get_llm_summary(self) -> dict:
        """
        Return aggregate statistics and individual records for all LLM
        calls made during the workflow.

        Returns::

            {
                "total": int,
                "succeeded": int,
                "failed": int,
                "used_fallback": int,
                "all_ok": bool,
                "calls": [<LLMCallRecord>, ...],
            }
        """
        state = self.get_final_state()
        calls: List[dict] = state.get("llm_calls", [])
        succeeded = sum(1 for c in calls if c.get("success"))
        failed = len(calls) - succeeded
        fallback = sum(1 for c in calls if c.get("used_fallback"))
        return {
            "total": len(calls),
            "succeeded": succeeded,
            "failed": failed,
            "used_fallback": fallback,
            "all_ok": failed == 0 and fallback == 0,
            "calls": calls,
        }

    # ------------------------------------------------------------------
    # Reflectivity data  (experimental + model curves)
    # ------------------------------------------------------------------

    def get_reflectivity_data(self) -> dict:
        """
        Return experimental Q/R/dR and per-iteration model curves.

        Returns::

            {
                "Q": [...], "R": [...], "dR": [...],
                "models": [
                    {"label": "...", "Q": [...], "R": [...], "chi2": float},
                    ...
                ]
            }
        """
        state = self.get_final_state()
        Q = state.get("Q", [])
        R = state.get("R", [])
        dR = state.get("dR", [])

        # Multi-file co-refinement: per-file experimental data
        data_files = state.get("data_files", [])
        has_multi = len(data_files) > 1 and any(ds.get("Q") for ds in data_files)

        # Build a {dataset label -> state name} map for multi-state runs.
        # Each StateDefinition carries its own data_files list, and PerFileFitResult
        # entries from Ticket 07 carry an explicit ``state`` field.
        state_for_label: Dict[str, str] = {}
        states_list = state.get("states") or []
        for st in states_list:
            st_name = st.get("name", "")
            for ds in st.get("data_files") or []:
                if ds.get("label"):
                    state_for_label[ds["label"]] = st_name

        models: List[dict] = []
        for fr in state.get("fit_results", []):
            iteration = fr.get("iteration", len(models))
            chi2 = fr.get("chi_squared")
            per_file = fr.get("per_file_results") or []

            if has_multi and per_file:
                # Emit one model curve per file per iteration
                for pf in per_file:
                    label = f"{pf.get('label', '?')} – iter {iteration}"
                    pf_chi2 = pf.get("chi_squared")
                    if pf_chi2 is not None:
                        label += f" (χ²={pf_chi2:.2f})"
                    pf_state = pf.get("state") or state_for_label.get(
                        pf.get("label", "")
                    )
                    entry: Dict[str, Any] = {
                        "label": label,
                        "Q": pf.get("Q_fit", []),
                        "R": pf.get("R_fit", []),
                        "chi2": pf_chi2,
                        "file_label": pf.get("label", ""),
                        "iteration": iteration,
                    }
                    if pf_state:
                        entry["state"] = pf_state
                    models.append(entry)
            else:
                label = f"Iteration {iteration}"
                if chi2 is not None:
                    label += f" (χ²={chi2:.2f})"
                models.append(
                    {
                        "label": label,
                        "Q": fr.get("Q_fit", []),
                        "R": fr.get("R_fit", []),
                        "chi2": chi2,
                        "iteration": iteration,
                    }
                )

        # Identify best-chi2 iteration using aggregate chi_squared from fit_results
        # (avoids picking based on a single segment's chi2 in multi-file mode)
        best_iteration = None
        best_chi2 = float("inf")
        for fr in state.get("fit_results", []):
            c = fr.get("chi_squared")
            if c is not None and c < best_chi2:
                best_chi2 = c
                best_iteration = fr.get("iteration")

        # Build unique iteration list for the selector
        seen = set()
        iterations = []
        for m in models:
            it = m.get("iteration")
            if it is not None and it not in seen:
                seen.add(it)
                # Aggregate chi2 from fit_results for this iteration
                fr_match = next(
                    (
                        fr
                        for fr in state.get("fit_results", [])
                        if fr.get("iteration") == it
                    ),
                    None,
                )
                agg_chi2 = fr_match.get("chi_squared") if fr_match else None
                label = f"Iteration {it}"
                if agg_chi2 is not None:
                    label += f" (χ²={agg_chi2:.2f})"
                iterations.append({"iteration": it, "label": label})

        result = {
            "Q": Q,
            "R": R,
            "dR": dR,
            "models": models,
            "best_iteration": best_iteration,
            "iterations": iterations,
        }

        # Include per-file experimental data for the frontend
        if has_multi:
            result["data_files"] = [
                {
                    "label": ds.get("label", ""),
                    "Q": ds.get("Q", []),
                    "R": ds.get("R", []),
                    "dR": ds.get("dR", []),
                    **(
                        {"state": state_for_label[ds.get("label", "")]}
                        if ds.get("label") in state_for_label
                        else {}
                    ),
                }
                for ds in data_files
            ]

        return result

    # ------------------------------------------------------------------
    # Model-per-iteration lookup
    # ------------------------------------------------------------------

    def _get_model_for_iteration(self, iteration: int) -> object | None:
        """Return the model definition that was used for a given fit iteration.

        Looks up ``model_history`` first (keyed by iteration number).
        Falls back to ``current_model`` when history is unavailable.
        """
        state = self.get_final_state()
        model_history = state.get("model_history") or []
        for entry in model_history:
            if entry.get("iteration") == iteration:
                defn = entry.get("definition")
                if defn and isinstance(defn, dict):
                    return defn
                # Legacy script path
                script = entry.get("script")
                if script:
                    return script
        # Fallback: current model (best we can do)
        return state.get("current_model")

    # ------------------------------------------------------------------
    # SLD profiles  (requires refl1d model execution)
    # ------------------------------------------------------------------

    def get_sld_profiles(self) -> dict:
        """
        Compute SLD(z) profiles for each fitting iteration.

        The profiles correspond 1-to-1 with the model curves returned by
        :meth:`get_reflectivity_data` and use the same labels/ordering so
        that colours match in the UI.

        Returns ``{"profiles": [{"label": ..., "z": [...], "sld": [...]}]}``.
        Gracefully returns an empty list when model execution fails.
        """
        if self._sld_cache is not None:
            return self._sld_cache

        state = self.get_final_state()
        fit_results = state.get("fit_results", [])

        profiles: List[dict] = []

        for idx, fr in enumerate(fit_results):
            iteration = fr.get("iteration", idx)
            chi2 = fr.get("chi_squared")

            label = f"Iteration {iteration}"
            if chi2 is not None:
                label += f" (χ²={chi2:.2f})"

            fitted_params = fr.get("parameters", {})
            model = self._get_model_for_iteration(iteration)

            # Multi-state (or single-state with per-state nuisance): emit one
            # profile per state per iteration via the states-problem path.
            from aure.nodes.model_builder import needs_states_problem as _needs_states

            if isinstance(model, dict) and _needs_states(model):
                try:
                    state_results = _compute_states_sld(model, fitted_params)
                except Exception as exc:
                    logger.debug(
                        "Could not compute multi-state SLD for iter %d: %s",
                        iteration,
                        exc,
                    )
                    state_results = []
                for sr in state_results:
                    profiles.append(
                        {
                            "label": f"{label} – {sr['state']}",
                            "z": sr["z"],
                            "sld": sr["sld"],
                            "state": sr["state"],
                            "iteration": iteration,
                        }
                    )
                continue

            try:
                result = _compute_sld_from_model(
                    model,
                    fitted_params,
                    output_dir=self.output_dir,
                    iteration=iteration,
                )
                if result and result.get("z") is not None:
                    profiles.append(
                        {"label": label, "z": result["z"], "sld": result["sld"]}
                    )
            except Exception as exc:
                logger.debug(
                    "Could not compute SLD for iteration %d: %s", iteration, exc
                )

        self._sld_cache = {"profiles": profiles}
        return self._sld_cache

    # ------------------------------------------------------------------
    # Fit parameters
    # ------------------------------------------------------------------

    def get_fit_parameters(self, iteration: int | None = None) -> dict:
        """Return parameters for a specific fit iteration.

        Parameters
        ----------
        iteration
            0-based index into ``fit_results``.  When *None* (default),
            the iteration with the lowest chi-squared is used.

        Returns::

            {
                "chi_squared": float,
                "method": str,
                "converged": bool,
                "iteration": int,
                "best_iteration": int,
                "parameters": [{"name": ..., "value": ..., "uncertainty": ...,
                                "bounds": [lo, hi] | null}],
            }
        """
        state = self.get_final_state()
        fit_results = state.get("fit_results", [])
        if not fit_results:
            return {
                "parameters": [],
                "chi_squared": None,
                "method": None,
                "converged": None,
                "iteration": None,
                "best_iteration": None,
            }

        # Find best-chi2 iteration
        best_idx = 0
        best_chi2 = float("inf")
        for i, fr in enumerate(fit_results):
            c = fr.get("chi_squared")
            if c is not None and c < best_chi2:
                best_chi2 = c
                best_idx = i

        idx = iteration if iteration is not None else best_idx
        idx = max(0, min(idx, len(fit_results) - 1))

        selected = fit_results[idx]
        params = selected.get("parameters", {})
        uncertainties = selected.get("uncertainties") or {}
        bounds = selected.get("bounds") or {}

        # Fallback: read bounds from model definition or problem.json
        if not bounds:
            iter_num = selected.get("iteration", idx)
            iter_model = self._get_model_for_iteration(iter_num)
            bounds = self._read_bounds_from_model_definition(iter_model)
        if not bounds:
            bounds = self._read_bounds_from_problem_json()

        rows = []
        for name, value in params.items():
            rows.append(
                {
                    "name": name,
                    "value": value,
                    "uncertainty": uncertainties.get(name),
                    "bounds": bounds.get(name),
                }
            )

        return {
            "chi_squared": selected.get("chi_squared"),
            "method": selected.get("method"),
            "converged": selected.get("converged"),
            "iteration": idx,
            "best_iteration": best_idx,
            "parameters": rows,
        }

    # ------------------------------------------------------------------
    # Intake / data-loading report
    # ------------------------------------------------------------------

    def get_intake_report(self) -> dict:
        """Summarise what the intake stage concluded about the input data.

        Returns a dict with per-file findings (theta, dQ convention,
        probe type that will be built) and high-level warnings — e.g.
        when some files would be loaded as ``QProbe`` and therefore
        cannot participate in shared ``sample_broadening`` /
        ``theta_offset`` parameters during co-refinement.

        Shape::

            {
                "primary_file": {"path": str, "dq_is_fwhm": bool},
                "files": [
                    {"label": str, "path": str, "theta": float,
                     "dq_is_fwhm": bool, "probe_type": "NeutronProbe"|"QProbe"},
                    ...
                ],
                "warnings": [str, ...],
                "messages": [str, ...],   # intake-stage messages from state
            }
        """
        state = self.get_final_state()

        def _probe_type(theta: float | None) -> str:
            return "NeutronProbe" if (theta is not None and theta > 0) else "QProbe"

        data_files = state.get("data_files") or []
        files_report: list[dict] = []
        for ds in data_files:
            theta = ds.get("theta")
            files_report.append(
                {
                    "label": ds.get("label", ""),
                    "path": ds.get("file", ""),
                    "theta": theta,
                    "dq_is_fwhm": ds.get("dq_is_fwhm"),
                    "probe_type": _probe_type(theta),
                }
            )

        warnings: list[str] = []
        if len(files_report) > 1:
            qprobes = [f for f in files_report if f["probe_type"] == "QProbe"]
            nprobes = [f for f in files_report if f["probe_type"] == "NeutronProbe"]
            if qprobes and nprobes:
                q_labels = ", ".join(f["label"] for f in qprobes)
                warnings.append(
                    "Mixed probe types in co-refinement: "
                    f"{q_labels} lack an incident angle and will be loaded "
                    "as QProbe. sample_broadening and theta_offset "
                    "cannot be applied to those files."
                )
            elif qprobes and not nprobes:
                warnings.append(
                    "All files loaded as QProbe (no incident angle "
                    "detected). sample_broadening and theta_offset are "
                    "unavailable — the files will be fit independently "
                    "aside from structural parameters."
                )

        # Intake-stage messages (anything written during the intake node).
        # We cannot filter perfectly without a node tag, but intake
        # messages are always the first ones and tend to carry the
        # "data validation" / "multi-file co-refinement" prefixes.
        messages: list[str] = []
        for m in state.get("messages", []) or []:
            if not isinstance(m, dict):
                continue
            content = m.get("content", "")
            if not content:
                continue
            if any(
                tag in content
                for tag in (
                    "Data validation",
                    "Multi-file co-refinement",
                    "Understood sample structure",
                    "could not load",
                    "Could not parse sample",
                )
            ):
                messages.append(content)

        primary = {
            "path": state.get("data_file", ""),
            "dq_is_fwhm": state.get("dq_is_fwhm"),
        }

        return {
            "primary_file": primary,
            "files": files_report,
            "warnings": warnings,
            "messages": messages,
        }

    def _read_bounds_from_model_definition(
        self,
        model: object | None = None,
    ) -> dict:
        """Extract parameter bounds from a ModelDefinition dict.

        Parameters
        ----------
        model
            A ``ModelDefinition`` dict.  When *None*, falls back to
            ``current_model`` from the final state.

        Returns ``{param_name: [lo, hi]}`` for parameters with defined ranges.
        """
        if model is None:
            state = self.get_final_state()
            model = state.get("current_model")
        if not isinstance(model, dict):
            return {}

        bounds: dict = {}
        for layer in model.get("layers", []):
            name = layer.get("name", "unknown")
            for prop in ("sld", "thickness", "roughness"):
                lo_key = f"{prop}_min"
                hi_key = f"{prop}_max"
                lo = layer.get(lo_key)
                hi = layer.get(hi_key)
                val = layer.get(prop)
                if lo is not None and hi is not None:
                    bounds[f"{name} {prop}"] = [lo, hi]
                elif val is not None:
                    # fixed parameter – no bounds
                    pass
        # Substrate roughness
        sub = model.get("substrate", {})
        sub_name = sub.get("name", "substrate")
        r_max = sub.get("roughness_max")
        # r_val = sub.get("roughness", 0)
        if r_max is not None:
            bounds[f"{sub_name} interface"] = [0, r_max]
        return bounds

    def _read_bounds_from_problem_json(self) -> dict:
        """Extract parameter bounds from the persisted ``problem.json``.

        Returns ``{param_name: [lo, hi]}`` for free parameters with finite
        bounds, or an empty dict if the file is unavailable.
        """
        path = self.output_dir / "problem.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            refs = data.get("references", {})
            bounds: dict = {}
            for ref in refs.values():
                if ref.get("fixed"):
                    continue
                name = ref.get("name", "")
                limits = ref.get("limits")
                if not name or not limits or len(limits) < 2:
                    continue
                try:
                    lo = float(limits[0])
                    hi = float(limits[1])
                except (ValueError, TypeError):
                    continue
                if lo != float("-inf") and hi != float("inf"):
                    bounds[name] = [lo, hi]
            return bounds
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        parameters: Dict[str, float],
        *,
        bounds: Optional[Dict[str, list]] = None,
        iteration: int | None = None,
    ) -> dict:
        """Compute reflectivity, SLD, and chi² for user-specified parameters.

        Builds a model from the ModelDefinition (or legacy script) and applies
        the given parameter values, then computes curves via refl1d.

        For multi-file co-refinement, returns per-file curves in a
        ``per_file`` list alongside the SLD profile.

        Parameters
        ----------
        bounds
            Optional ``{name: [lo, hi]}`` overrides coming from the UI.
        iteration
            Fit iteration whose model structure to use.  When *None*,
            falls back to the current (latest) model.

        Returns ``{"Q_fit", "R_fit", "sld_z", "sld_rho", "chi_squared"}``,
        or ``{"per_file": [...], "sld_z", "sld_rho", "chi_squared"}`` for
        multi-file.
        """
        if iteration is not None:
            model = self._get_model_for_iteration(iteration)
        else:
            state = self.get_final_state()
            model = state.get("current_model")

        if model is None:
            return {"error": "No model available"}

        # Multi-state (or single state with per-state nuisance):
        # build_states_problem is the only path that wires up shared
        # theta_offset / sample_broadening correctly. Take precedence
        # over the legacy multi-file path.
        from aure.nodes.model_builder import needs_states_problem as _needs_states

        if isinstance(model, dict) and _needs_states(model):
            try:
                return _compute_states_simulation(
                    model,
                    parameters,
                    bounds=bounds,
                )
            except Exception as exc:
                return {"error": str(exc)}

        # Check for multi-file co-refinement
        state = self.get_final_state()
        data_files = state.get("data_files", [])
        has_multi = len(data_files) > 1

        if has_multi and isinstance(model, dict):
            try:
                return _compute_multi_file_simulation(
                    model,
                    data_files,
                    parameters,
                    bounds=bounds,
                )
            except Exception as exc:
                return {"error": str(exc)}

        try:
            result = _compute_from_model(
                model,
                parameters,
                bounds=bounds,
                output_dir=self.output_dir,
                compute_reflectivity=True,
            )
        except Exception as exc:
            return {"error": str(exc)}

        if result is None:
            return {"error": "Model computation returned no result"}

        return {
            "Q_fit": result.get("Q_fit") or [],
            "R_fit": result.get("R_fit") or [],
            "sld_z": result.get("z") or [],
            "sld_rho": result.get("sld") or [],
            "chi_squared": result.get("chi_squared"),
        }


# ======================================================================
# JSON helpers
# ======================================================================


def _rejoin_message_content(data: object) -> None:
    """Rejoin message ``content`` line-arrays back into strings (in-place).

    Checkpoint files split multi-line content into JSON arrays for
    readability.  This reverses that transformation on load.
    """
    if isinstance(data, dict):
        if "role" in data and "content" in data and isinstance(data["content"], list):
            data["content"] = "\n".join(data["content"])
        for v in data.values():
            _rejoin_message_content(v)
    elif isinstance(data, list):
        for item in data:
            _rejoin_message_content(item)


# ======================================================================
# Model computation helpers (JSON ModelDefinition path)
# ======================================================================


def _compute_sld_from_model(
    model: object,
    fitted_params: Dict[str, float],
    output_dir: Path,
    iteration: int,
) -> Optional[Dict[str, Any]]:
    """Compute SLD profile from a model (dict or legacy script).

    Thin convenience wrapper: delegates to ``_compute_from_model`` with
    ``compute_reflectivity=False``.
    """
    return _compute_from_model(
        model, fitted_params, output_dir=output_dir, compute_reflectivity=False
    )


def _compute_from_model(
    model: object,
    parameters: Dict[str, float],
    *,
    bounds: Optional[Dict[str, list]] = None,
    output_dir: Optional[Path] = None,
    compute_reflectivity: bool = False,
) -> Optional[Dict[str, Any]]:
    """Build a model, apply parameters, and extract curves.

    Returns a dict with keys ``z``, ``sld``, and optionally ``Q_fit``,
    ``R_fit``, ``chi_squared``.
    """
    from aure.nodes.model_builder import (
        apply_bounds,
        apply_parameters,
        build_problem,
    )

    if not isinstance(model, dict):
        return None

    # New JSON ModelDefinition path
    definition = dict(model)  # type: ignore[arg-type]
    problem = build_problem(definition)

    if bounds:
        apply_bounds(problem, bounds)
    if parameters:
        apply_parameters(problem, parameters)

    experiment = problem.fitness
    if hasattr(experiment, "_models"):
        experiment = experiment._models[0]

    result: Dict[str, Any] = {}

    # SLD profile
    try:
        z_arr, sld_arr, _ = experiment.smooth_profile(dz=1.0)
        result["z"] = np.array(z_arr).tolist()
        result["sld"] = np.array(sld_arr).tolist()
    except Exception:
        result["z"] = None
        result["sld"] = None

    # Reflectivity + chi² (optional)
    if compute_reflectivity:
        try:
            experiment.update()
            Q_arr, R_arr = experiment.reflectivity()
            result["Q_fit"] = np.array(Q_arr).tolist()
            result["R_fit"] = np.array(R_arr).tolist()
        except Exception:
            result["Q_fit"] = None
            result["R_fit"] = None
        try:
            chi2 = float(problem.chisq())
            result["chi_squared"] = chi2 if math.isfinite(chi2) else None
        except Exception:
            result["chi_squared"] = None

    return result


def _compute_multi_file_simulation(
    model: dict,
    data_files: list,
    parameters: Dict[str, float],
    *,
    bounds: Optional[Dict[str, list]] = None,
) -> dict:
    """Simulate reflectivity for a multi-file co-refinement model.

    Builds a joint ``FitProblem`` via ``build_multi_problem``, applies
    the user's parameter values, then extracts per-file R(Q) curves
    and the shared SLD profile.
    """
    from aure.nodes.model_builder import (
        apply_bounds,
        apply_parameters,
        build_multi_problem,
    )

    definition = dict(model)
    problem, experiments, sorted_data_files = build_multi_problem(
        definition, data_files
    )

    if bounds:
        apply_bounds(problem, bounds)
    if parameters:
        apply_parameters(problem, parameters)

    result: Dict[str, Any] = {}

    # SLD profile (shared across all experiments)
    try:
        z_arr, sld_arr, _ = experiments[0].smooth_profile(dz=1.0)
        result["sld_z"] = np.array(z_arr).tolist()
        result["sld_rho"] = np.array(sld_arr).tolist()
    except Exception:
        result["sld_z"] = []
        result["sld_rho"] = []

    # Per-file reflectivity
    per_file = []
    for exp, ds in zip(experiments, sorted_data_files):
        pf: Dict[str, Any] = {"label": ds.get("label", "")}
        try:
            exp.update()
            Q_arr, R_arr = exp.reflectivity()
            pf["Q_fit"] = np.array(Q_arr).tolist()
            pf["R_fit"] = np.array(R_arr).tolist()
        except Exception:
            pf["Q_fit"] = []
            pf["R_fit"] = []
        per_file.append(pf)
    result["per_file"] = per_file

    # Aggregate chi²
    try:
        chi2 = float(problem.chisq())
        result["chi_squared"] = chi2 if math.isfinite(chi2) else None
    except Exception:
        result["chi_squared"] = None

    return result


def _with_angle_theta(model: dict) -> dict:
    """Deep-copy *model* and back-fill each state dataset's incident angle.

    ``theta_offset`` / ``sample_broadening`` only take effect on an angle-based
    refl1d NeutronProbe, which :func:`build_states_problem` builds only when a
    dataset carries ``theta > 0``. Freshly-run models are enriched at intake,
    but imported / older models can carry an *enabled* nuisance block next to
    theta-less datasets — in which case the parameter is silently dropped and
    the Results-tab slider does nothing. Re-derive theta from the file header
    so the recompute matches what the fit actually optimised.
    """
    import copy

    from aure.nodes.model_builder import _nuisance_enabled

    definition = copy.deepcopy(model)
    model_wants_angle = _nuisance_enabled(
        definition.get("theta_offset")
    ) or _nuisance_enabled(definition.get("sample_broadening"))
    for st in definition.get("states") or []:
        wants_angle = (
            model_wants_angle
            or _nuisance_enabled(st.get("theta_offset"))
            or _nuisance_enabled(st.get("sample_broadening"))
        )
        if not wants_angle:
            continue
        for ds in st.get("data_files") or []:
            if ds.get("theta"):
                continue
            try:
                from aure.nodes.intake import _parse_theta_from_header

                t = _parse_theta_from_header(ds.get("file", ""))
            except Exception:
                t = 0.0
            if t and t > 0:
                ds["theta"] = t
    return definition


def _compute_states_simulation(
    model: dict,
    parameters: Dict[str, float],
    *,
    bounds: Optional[Dict[str, list]] = None,
) -> dict:
    """Simulate per-state R(Q) and SLD profiles for a multi-state model.

    Builds the joint ``FitProblem`` via :func:`build_states_problem`,
    applies user parameters, then extracts one R(Q) curve per state file
    and one SLD profile per state.
    """
    from aure.nodes.model_builder import (
        apply_bounds,
        apply_parameters,
        build_states_problem,
    )

    definition = _with_angle_theta(model)
    problem, experiments_by_state, sorted_files_by_state = build_states_problem(
        definition
    )

    if bounds:
        apply_bounds(problem, bounds)
    if parameters:
        apply_parameters(problem, parameters)

    per_file: List[Dict[str, Any]] = []
    sld_profiles: List[Dict[str, Any]] = []

    for state_name, experiments in experiments_by_state.items():
        files = sorted_files_by_state.get(state_name, [])

        # SLD profile: one per state, taken from the first experiment
        if experiments:
            try:
                z_arr, sld_arr, _ = experiments[0].smooth_profile(dz=1.0)
                sld_profiles.append(
                    {
                        "state": state_name,
                        "z": np.array(z_arr).tolist(),
                        "sld": np.array(sld_arr).tolist(),
                    }
                )
            except Exception:
                sld_profiles.append({"state": state_name, "z": [], "sld": []})

        for exp, ds in zip(experiments, files):
            pf: Dict[str, Any] = {
                "label": ds.get("label", ""),
                "state": state_name,
            }
            try:
                exp.update()
                Q_arr, R_arr = exp.reflectivity()
                pf["Q_fit"] = np.array(Q_arr).tolist()
                pf["R_fit"] = np.array(R_arr).tolist()
                # theta_offset re-assigns each measured point's Q (Q is derived
                # from the incident angle), so the experimental data must be
                # replotted at the corrected Q. ``probe.Q`` already carries the
                # offset-corrected Q; ``probe.R`` is the unchanged measured R.
                probe = exp.probe
                pf["Q_data"] = np.asarray(probe.Q, dtype=float).tolist()
                pf["R_data"] = np.asarray(probe.R, dtype=float).tolist()
                _dr = getattr(probe, "dR", None)
                if _dr is not None:
                    pf["dR_data"] = np.asarray(_dr, dtype=float).tolist()
            except Exception:
                pf["Q_fit"] = []
                pf["R_fit"] = []
            per_file.append(pf)

    try:
        chi2 = float(problem.chisq())
        chi_squared = chi2 if math.isfinite(chi2) else None
    except Exception:
        chi_squared = None

    return {
        "per_file": per_file,
        "sld_profiles": sld_profiles,
        "chi_squared": chi_squared,
    }


def _compute_states_sld(
    model: dict,
    parameters: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Return ``[{state, z, sld}]`` for each state in a multi-state model."""
    sim = _compute_states_simulation(model, parameters)
    return sim.get("sld_profiles") or []
