"""
Data access layer for the web visualisation app.

Reads checkpoint files, run metadata, and model outputs produced by
``aure analyze -o <output_dir>``.
"""

from __future__ import annotations

import json
import logging
import os
import re
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
                data = json.loads(path.read_text())
                self._final_state = data.get("state", data)
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

        Each entry: ``{step, node, iteration, chi2, timestamp}``.
        ``chi2`` is ``None`` for nodes that don't produce a fit.
        """
        cps = self._load_all_checkpoints()
        result: List[dict] = []

        for i, cp in enumerate(cps):
            state = cp.get("state", {})
            info = cp.get("_info", {})
            result.append(
                {
                    "step": i + 1,
                    "node": info.get("node", cp.get("node", "")),
                    "iteration": info.get("iteration", cp.get("iteration", 0)),
                    "chi2": state.get("current_chi2"),
                    "timestamp": info.get("timestamp", cp.get("timestamp")),
                    "error": state.get("error"),
                }
            )

        return result

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

        models: List[dict] = []
        for fr in state.get("fit_results", []):
            iteration = fr.get("iteration", len(models))
            chi2 = fr.get("chi_squared")
            label = f"Iteration {iteration}"
            if chi2 is not None:
                label += f" (χ²={chi2:.2f})"
            models.append(
                {
                    "label": label,
                    "Q": fr.get("Q_fit", []),
                    "R": fr.get("R_fit", []),
                    "chi2": chi2,
                }
            )

        return {"Q": Q, "R": R, "dR": dR, "models": models}

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

        models_dir = self.output_dir / "models"
        if not models_dir.exists():
            self._sld_cache = {"profiles": []}
            return self._sld_cache

        state = self.get_final_state()
        Q_data = np.array(state.get("Q", []))
        fit_results = state.get("fit_results", [])

        profiles: List[dict] = []

        for idx, fr in enumerate(fit_results):
            iteration = fr.get("iteration", idx)
            chi2 = fr.get("chi_squared")

            # Build the same label used by get_reflectivity_data()
            label = f"Iteration {iteration}"
            if chi2 is not None:
                label += f" (χ²={chi2:.2f})"

            # Find the corresponding model file
            model_file = models_dir / f"model_fitting_iter{iteration}.py"
            if not model_file.exists():
                logger.debug("No model file for iteration %d", iteration)
                continue

            try:
                fitted_params = fr.get("parameters", {})
                result = _execute_model_file(
                    model_file, Q_data,
                    working_dir=self.output_dir.parent,
                    fitted_parameters=fitted_params,
                )
                if result and result.get("z") is not None:
                    profiles.append(
                        {"label": label, "z": result["z"], "sld": result["sld"]}
                    )
            except Exception as exc:
                logger.debug("Could not execute model %s: %s", model_file.name, exc)

        self._sld_cache = {"profiles": profiles}
        return self._sld_cache

    # ------------------------------------------------------------------
    # Fit parameters
    # ------------------------------------------------------------------

    def get_fit_parameters(self) -> dict:
        """
        Return parameters from the latest fit result.

        Returns::

            {
                "chi_squared": float,
                "method": str,
                "converged": bool,
                "parameters": [{"name": ..., "value": ..., "uncertainty": ...}],
            }
        """
        state = self.get_final_state()
        fit_results = state.get("fit_results", [])
        if not fit_results:
            return {"parameters": [], "chi_squared": None, "method": None, "converged": None}

        latest = fit_results[-1]
        params = latest.get("parameters", {})
        uncertainties = latest.get("uncertainties") or {}

        rows = []
        for name, value in params.items():
            rows.append(
                {
                    "name": name,
                    "value": value,
                    "uncertainty": uncertainties.get(name),
                }
            )

        return {
            "chi_squared": latest.get("chi_squared"),
            "method": latest.get("method"),
            "converged": latest.get("converged"),
            "parameters": rows,
        }


# ======================================================================
# Model-file execution helper  (adapted from cli.py)
# ======================================================================

def _execute_model_file(
    model_file: Path,
    Q_data: np.ndarray,
    working_dir: Optional[Path] = None,
    fitted_parameters: Optional[Dict[str, float]] = None,
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
        if fitted_parameters and problem is not None:
            _apply_fitted_parameters(problem, fitted_parameters)
        elif fitted_parameters and experiment is not None:
            # problem may not exist; try wrapping experiment
            try:
                from bumps.fitproblem import FitProblem
                tmp_problem = FitProblem(experiment)
                _apply_fitted_parameters(tmp_problem, fitted_parameters)
            except Exception:
                pass

        z, sld = None, None
        try:
            z_arr, sld_arr, _ = experiment.smooth_profile(dz=1.0)
            z = np.array(z_arr).tolist()
            sld = np.array(sld_arr).tolist()
        except Exception:
            pass

        return {"z": z, "sld": sld}
    except Exception as exc:
        raise RuntimeError(f"Model execution failed: {exc}") from exc
    finally:
        os.chdir(original_cwd)


def _apply_fitted_parameters(
    problem: Any, fitted_parameters: Dict[str, float]
) -> None:
    """Set parameter values on a bumps ``FitProblem`` from a name→value dict."""
    params = getattr(problem, "_parameters", None)
    if params is None:
        return
    for par in params:
        name = str(par.name)
        if name in fitted_parameters:
            par.value = float(fitted_parameters[name])



