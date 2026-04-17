"""
Data access layer for the web visualisation app.

Reads checkpoint files, run metadata, and model outputs produced by
``aure analyze -o <output_dir>``.
"""

from __future__ import annotations

import json
import logging
import math
import os
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
        └── models/
            ├── model_initial.py
            └── ...
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
                    models.append(
                        {
                            "label": label,
                            "Q": pf.get("Q_fit", []),
                            "R": pf.get("R_fit", []),
                            "chi2": pf_chi2,
                            "file_label": pf.get("label", ""),
                            "iteration": iteration,
                        }
                    )
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

        # Identify best-chi2 iteration (by iteration number, not array index)
        best_iteration = None
        best_chi2 = float("inf")
        for m in models:
            c = m.get("chi2")
            if c is not None and c < best_chi2:
                best_chi2 = c
                best_iteration = m.get("iteration")

        # Build unique iteration list for the selector
        seen = set()
        iterations = []
        for m in models:
            it = m.get("iteration")
            if it is not None and it not in seen:
                seen.add(it)
                # Aggregate chi2 from fit_results for this iteration
                fr_match = next(
                    (fr for fr in state.get("fit_results", []) if fr.get("iteration") == it),
                    None,
                )
                agg_chi2 = fr_match.get("chi_squared") if fr_match else None
                label = f"Iteration {it}"
                if agg_chi2 is not None:
                    label += f" (χ²={agg_chi2:.2f})"
                iterations.append({"iteration": it, "label": label})

        result = {
            "Q": Q, "R": R, "dR": dR,
            "models": models,
            "best_iteration": best_iteration,
            "iterations": iterations,
        }

        # Include per-file experimental data for the frontend
        if has_multi:
            result["data_files"] = [
                {"label": ds.get("label", ""), "Q": ds.get("Q", []), "R": ds.get("R", []), "dR": ds.get("dR", [])}
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
                logger.debug("Could not compute SLD for iteration %d: %s", iteration, exc)

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

    def _read_bounds_from_model_definition(
        self, model: object | None = None,
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
        r_val = sub.get("roughness", 0)
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

        # Check for multi-file co-refinement
        state = self.get_final_state()
        data_files = state.get("data_files", [])
        has_multi = len(data_files) > 1

        if has_multi and isinstance(model, dict):
            try:
                return _compute_multi_file_simulation(
                    model, data_files, parameters, bounds=bounds,
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

    Handles both new JSON ``ModelDefinition`` dicts and legacy script
    strings.  For dicts the model is built via ``model_builder``; for
    strings the old ``_execute_model_file`` code-path is used.

    Returns a dict with keys ``z``, ``sld``, and optionally ``Q_fit``,
    ``R_fit``, ``chi_squared``.
    """
    from aure.nodes.model_builder import (
        apply_bounds,
        apply_parameters,
        build_problem,
        is_legacy_script,
    )

    if is_legacy_script(model):
        # Legacy: find a model file on disk and exec() it
        if output_dir is None:
            return None
        models_dir = output_dir / "models"
        model_file = models_dir / "model_final.py"
        if not model_file.exists():
            return None
        Q_data = np.array([])  # not needed for SLD-only
        return _execute_model_file(
            model_file,
            Q_data,
            working_dir=output_dir.parent,
            fitted_parameters=parameters or None,
            parameter_bounds=bounds,
            compute_reflectivity=compute_reflectivity,
        )

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
    problem, experiments, sorted_data_files = build_multi_problem(definition, data_files)

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


# ======================================================================
# Model-file execution helper  (adapted from cli.py)
# ======================================================================


def _execute_model_file(
    model_file: Path,
    Q_data: np.ndarray,
    working_dir: Optional[Path] = None,
    fitted_parameters: Optional[Dict[str, float]] = None,
    parameter_bounds: Optional[Dict[str, list]] = None,
    compute_reflectivity: bool = False,
) -> Optional[dict]:
    """Execute a refl1d model script and extract SLD profile.

    Parameters
    ----------
    fitted_parameters
        If provided, a ``{name: value}`` mapping of best-fit parameter
        values.  After the script is executed the parameters of the
        resulting ``FitProblem`` are updated to these values so that the
        SLD profile reflects the actual fit result rather than the
        (possibly arbitrary) defaults in the script.
    parameter_bounds
        Optional ``{name: [lo, hi]}`` overrides for parameter bounds.
        Applied *before* values so that the new value falls within the
        updated range and chi² reflects only data misfit.
    compute_reflectivity
        If *True*, also compute the reflectivity curve and chi² from the
        model with the current parameter values.
    """
    original_cwd = os.getcwd()
    try:
        script = model_file.read_text()

        if working_dir and working_dir.exists():
            os.chdir(working_dir)

        globs: Dict[str, Any] = {"__file__": str(model_file)}
        exec(compile(script, str(model_file), "exec"), globs)

        experiment = globs.get("experiment")
        problem = globs.get("problem")

        if experiment is None and problem is not None:
            fitness = getattr(problem, "fitness", problem)
            if hasattr(fitness, "_models"):
                experiment = fitness._models[0]
            elif hasattr(fitness, "reflectivity"):
                experiment = fitness

        if experiment is None:
            return None

        # ---- Apply fitted parameter values --------------------------
        # Values are applied FIRST so that the original model bounds can
        # disambiguate duplicate parameter names (e.g. two parameters
        # both named "silicon interface" with different bound ranges).
        if fitted_parameters and problem is not None:
            _apply_fitted_parameters(problem, fitted_parameters)
        elif fitted_parameters and experiment is not None:
            # problem may not exist; try wrapping experiment
            try:
                from bumps.fitproblem import FitProblem

                tmp_problem = FitProblem(experiment)
                _apply_fitted_parameters(tmp_problem, fitted_parameters)
                problem = tmp_problem
            except Exception:
                pass

        # ---- Widen bounds to UI ranges (after values) ---------------
        # Done after value assignment so that chi² does not penalise
        # values the user intentionally moved outside the original range.
        if parameter_bounds and problem is not None:
            _apply_parameter_bounds(problem, parameter_bounds)

        result: Dict[str, Any] = {}

        # ---- SLD profile --------------------------------------------
        z, sld = None, None
        try:
            z_arr, sld_arr, _ = experiment.smooth_profile(dz=1.0)
            z = np.array(z_arr).tolist()
            sld = np.array(sld_arr).tolist()
        except Exception:
            pass
        result["z"] = z
        result["sld"] = sld

        # ---- Reflectivity + chi² (optional) -------------------------
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
                if problem is not None:
                    chi2 = float(problem.chisq())
                    result["chi_squared"] = chi2 if math.isfinite(chi2) else None
                else:
                    result["chi_squared"] = None
            except Exception:
                result["chi_squared"] = None

        return result
    except Exception as exc:
        raise RuntimeError(f"Model execution failed: {exc}") from exc
    finally:
        os.chdir(original_cwd)


def _apply_parameter_bounds(problem: Any, bounds: Dict[str, list]) -> None:
    """Widen parameter bounds on a bumps ``FitProblem`` to user-specified ranges.

    For each parameter whose name appears in *bounds*, the bounds and
    prior are set to the union of the model's current range and the UI
    range so that the requested value is always within bounds and chi²
    reflects only data misfit.
    """
    from bumps.bounds import init_bounds

    params = getattr(problem, "_parameters", None)
    if params is None:
        return
    for par in params:
        name = str(par.name)
        if name not in bounds:
            continue
        lo_new, hi_new = bounds[name]
        cur_bounds = getattr(par, "bounds", None)
        if isinstance(cur_bounds, tuple) and len(cur_bounds) == 2:
            lo = min(cur_bounds[0], lo_new)
            hi = max(cur_bounds[1], hi_new)
        else:
            lo, hi = lo_new, hi_new
        par.range(lo, hi)
        par.prior = init_bounds((lo, hi))


def _apply_fitted_parameters(problem: Any, fitted_parameters: Dict[str, float]) -> None:
    """Set parameter values on a bumps ``FitProblem`` from a name→value dict.

    When multiple model parameters share the same name (e.g. substrate and
    a layer with identical material produce two "silicon interface" entries),
    assign the value only to the parameter whose current bounds contain it.
    If *all* or *none* of the duplicates contain the value, set them all to
    avoid silently dropping updates.
    """
    params = getattr(problem, "_parameters", None)
    if params is None:
        return

    # Group parameters by name so we can detect duplicates.
    from collections import defaultdict

    by_name: dict[str, list] = defaultdict(list)
    for par in params:
        by_name[str(par.name)].append(par)

    for name, value in fitted_parameters.items():
        group = by_name.get(name)
        if not group:
            continue
        if len(group) == 1:
            group[0].value = float(value)
        else:
            # Multiple params share this name – prefer those whose bounds
            # contain the requested value.
            in_bounds = []
            for par in group:
                cur_bounds = getattr(par, "bounds", None)
                if isinstance(cur_bounds, tuple) and len(cur_bounds) == 2:
                    lo, hi = cur_bounds
                else:
                    lo, hi = -float("inf"), float("inf")
                if lo <= value <= hi:
                    in_bounds.append(par)
            targets = in_bounds if in_bounds else group
            for par in targets:
                par.value = float(value)
