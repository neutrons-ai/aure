"""
FITTING node: Run refl1d optimization.

This node builds a FitProblem from the ModelDefinition JSON and fits
the data using bumps.  Supports multiple fitting methods:
- 'lm': Levenberg-Marquardt (fast, local optimizer)
- 'de': Differential Evolution (global optimizer)
- 'dream': MCMC for uncertainty quantification
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np

from ..state import ReflectivityState, FitResult, Message
from .model_builder import build_problem, is_legacy_script
from .evaluation import _count_free_params, _compute_bic

logger = logging.getLogger(__name__)


def fitting_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Run refl1d fit on current model.

    Args:
        state: Current workflow state

    Returns:
        State updates including fit results
    """
    updates = {
        "current_node": "fitting",
        "messages": [],
        "fit_results": [],
    }

    model = state.get("current_model")
    if not model:
        updates["error"] = "No model to fit"
        return updates

    iteration = state.get("iteration", 0)
    method = os.environ.get("FIT_METHOD", "dream").lower()
    # Allow UI / state override for DREAM steps, then env var, then default
    steps = int(state.get("fit_steps") or os.environ.get("FIT_STEPS", "1000"))
    burn = int(state.get("fit_burn") or os.environ.get("FIT_BURN", str(steps)))
    logger.info(
        f"[FITTING] Starting iteration {iteration} (steps={steps}, burn={burn})"
    )

    # Build export directory for refl1d output
    export_dir: Optional[str] = None
    base_output = state.get("output_dir")
    if base_output:
        export_dir = str(
            Path(base_output) / "refl1d_output" / f"fit_iter{iteration}_{method}"
        )
        Path(export_dir).mkdir(parents=True, exist_ok=True)

    # ========== Run Fit ==========
    try:
        logger.info(f"[FITTING] Running {method.upper()} optimization...")

        if is_legacy_script(model):
            # Backward compatibility: exec-based path for old script models
            result = _run_refl1d_fit_legacy(
                model_script=model,
                method=method,
                iteration=iteration,
                steps=steps,
                burn=burn,
                export_dir=export_dir,
            )
        else:
            result = run_refl1d_fit(
                model_definition=model,
                method=method,
                iteration=iteration,
                steps=steps,
                burn=burn,
                export_dir=export_dir,
            )

        updates["fit_results"] = [result]
        updates["current_chi2"] = result["chi_squared"]
        logger.info(f"[FITTING] Completed with χ² = {result['chi_squared']:.3f}")

        # Update best chi2 and save best model
        best = state.get("best_chi2")
        if best is None or result["chi_squared"] < best:
            updates["best_chi2"] = result["chi_squared"]
            updates["best_model"] = model
            logger.info(f"[FITTING] New best χ² = {result['chi_squared']:.3f}")

        # Update best BIC (complexity-penalized score)
        if isinstance(model, dict):
            n_data = len(state.get("Q", []))
            n_params = _count_free_params(model)
            bic = _compute_bic(result["chi_squared"], n_data, n_params)
            result["bic"] = bic
            logger.info(f"[FITTING] BIC = {bic:.1f} (k={n_params}, n={n_data})")
            best_bic = state.get("best_bic")
            if best_bic is None or bic < best_bic:
                updates["best_bic"] = bic
                updates["best_bic_model"] = model
                logger.info(f"[FITTING] New best BIC = {bic:.1f}")

        # Format message
        updates["messages"] = [
            Message(
                role="assistant", content=_format_fit_result(result), timestamp=None
            )
        ]

    except Exception as e:
        updates["error"] = f"Fitting failed: {str(e)}"
        updates["messages"] = [
            Message(
                role="system", content=f"Error during fitting: {str(e)}", timestamp=None
            )
        ]

    return updates


def run_refl1d_fit(
    model_definition: dict,
    method: str = "lm",
    iteration: int = 0,
    steps: int = 1000,
    burn: int = 1000,
    export_dir: Optional[str] = None,
) -> FitResult:
    """
    Execute refl1d fit by building a FitProblem from a ModelDefinition.

    Args:
        model_definition: ModelDefinition dict describing the model
        method: Fitting method ('lm', 'de', 'dream')
        iteration: Current iteration number
        steps: Number of steps for MCMC methods
        burn: Number of burn-in steps for MCMC
        export_dir: Optional directory for bumps/refl1d export output

    Returns:
        FitResult dictionary
    """
    from bumps.fitters import fit as bumps_fit

    problem = build_problem(model_definition)

    # Configure fit options based on method
    fit_options = _build_fit_options(method, steps, burn, export_dir)

    # Run the fit
    logger.info(f"[FITTING] Running {method.upper()} with bumps.fit...")
    result = bumps_fit(problem, **fit_options)

    # Extract results
    return _extract_bumps_results(
        problem=problem,
        fit_result=result,
        method=method,
        iteration=iteration,
        export_dir=export_dir,
    )


def _run_refl1d_fit_legacy(
    model_script: str,
    method: str = "lm",
    iteration: int = 0,
    steps: int = 1000,
    burn: int = 1000,
    export_dir: Optional[str] = None,
) -> FitResult:
    """Legacy exec-based fitting for old script-string models."""
    from bumps.fitters import fit as bumps_fit

    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "model.py"
        model_file.write_text(model_script)

        model_globals = {"__file__": str(model_file)}
        exec(compile(model_script, model_file, "exec"), model_globals)

        problem = model_globals.get("problem")
        if problem is None:
            raise ValueError("Model script must define a 'problem' variable")

        fit_options = _build_fit_options(method, steps, burn, export_dir)

        logger.info(f"[FITTING] Running {method.upper()} with bumps.fit (legacy)...")
        result = bumps_fit(problem, **fit_options)

        return _extract_bumps_results(
            problem=problem,
            fit_result=result,
            method=method,
            iteration=iteration,
            export_dir=export_dir,
        )


def _build_fit_options(
    method: str, steps: int, burn: int, export_dir: Optional[str]
) -> dict:
    """Build the options dict for bumps.fitters.fit."""
    fit_options = {
        "method": method,
        "parallel": 0,
    }

    if method == "dream":
        fit_options["samples"] = steps
        fit_options["burn"] = burn
        fit_options["pop"] = 10
        fit_options["thin"] = 1
        fit_options["alpha"] = 0.0
        fit_options["trim"] = False
        fit_options["steps"] = 0
    elif method == "de":
        fit_options["steps"] = steps
        fit_options["pop"] = 10
    elif method == "lm":
        fit_options["steps"] = steps

    if export_dir:
        fit_options["export"] = export_dir
        logger.info(f"[FITTING] Exporting refl1d output to {export_dir}")

    return fit_options


def _extract_bumps_results(
    problem,
    fit_result,
    method: str,
    iteration: int,
    export_dir: Optional[str] = None,
) -> FitResult:
    """Extract fit results from bumps problem and fit result."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="bumps")

    # Get chi-squared (bumps uses chisq method on problem)
    chi_squared = problem.chisq()

    # Compute theory curve at fitted parameter values
    Q_fit = []
    R_fit = []
    try:
        experiment = problem.fitness
        experiment.update()
        Q_arr, R_arr = experiment.reflectivity()
        Q_fit = Q_arr.tolist()
        R_fit = R_arr.tolist()
    except Exception as e:
        logger.warning(f"[FITTING] Could not compute theory curve: {e}")

    # Get parameter values and names from problem._parameters
    parameters = {}
    uncertainties = {}

    param_bounds = {}
    for i, par in enumerate(problem._parameters):
        name = str(par.name)
        parameters[name] = par.value

        # Get uncertainties from fit result if available
        if hasattr(fit_result, "dx") and fit_result.dx is not None:
            try:
                if i < len(fit_result.dx):
                    uncertainties[name] = fit_result.dx[i]
            except (IndexError, TypeError):
                pass

        # Get parameter bounds
        if par.bounds is not None:
            lo, hi = par.bounds
            param_bounds[name] = [float(lo), float(hi)]

    # Check convergence
    converged = chi_squared < 100  # Simple heuristic
    if hasattr(fit_result, "success"):
        converged = fit_result.success

    logger.info(f"[FITTING] Fit complete: χ² = {chi_squared:.3f}")
    for name, value in parameters.items():
        unc_str = (
            f" ± {uncertainties.get(name, 0):.3f}" if name in uncertainties else ""
        )
        logger.info(f"[FITTING]   {name}: {value:.3f}{unc_str}")

    # Read SLD profile from refl1d export if available
    sld_z, sld_rho = _read_profile_dat(export_dir)

    # Compute residuals and residual ratio for fringe analysis
    residuals = []
    residual_ratio = []
    if Q_fit and R_fit:
        try:
            R_data = problem.fitness.probe.R
            dR_data = problem.fitness.probe.dR
            R_fit_arr = np.array(R_fit)
            if R_data is not None and len(R_data) == len(R_fit_arr):
                # Normalized residuals: (data - model) / error
                if dR_data is not None and len(dR_data) == len(R_data):
                    safe_dR = np.maximum(np.abs(dR_data), 1e-20)
                    residuals = ((R_data - R_fit_arr) / safe_dR).tolist()
                # Ratio for fringe analysis: data / model
                safe_R_fit = np.maximum(R_fit_arr, 1e-20)
                residual_ratio = (R_data / safe_R_fit).tolist()
        except Exception as e:
            logger.debug(f"[FITTING] Could not compute residuals: {e}")

    return FitResult(
        iteration=iteration,
        method=method,
        chi_squared=chi_squared,
        converged=converged,
        parameters=parameters,
        uncertainties=uncertainties if uncertainties else None,
        bounds=param_bounds if param_bounds else None,
        Q_fit=Q_fit,
        R_fit=R_fit,
        residuals=residuals,
        residual_ratio=residual_ratio,
        sld_z=sld_z,
        sld_rho=sld_rho,
        issues=[],
        suggestions=[],
    )


def _format_fit_result(result: FitResult) -> str:
    """Format fit result for display."""
    lines = ["**Fit Results:**"]
    lines.append("")

    chi2 = result["chi_squared"]
    if chi2 < 2:
        quality = "✓ Excellent"
    elif chi2 < 5:
        quality = "○ Good"
    elif chi2 < 10:
        quality = "△ Acceptable"
    else:
        quality = "✗ Poor"

    lines.append(f"- **χ² = {chi2:.2f}** ({quality})")
    lines.append(f"- Method: {result['method'].upper()}")
    lines.append(f"- Converged: {'Yes' if result['converged'] else 'No'}")

    if result["parameters"]:
        lines.append("")
        lines.append("**Best-fit parameters:**")
        for param, value in list(result["parameters"].items())[:10]:
            lines.append(f"- {param}: {value:.3f}")

    if result["uncertainties"]:
        lines.append("")
        lines.append("**Uncertainties (1σ):**")
        for param, unc in list(result["uncertainties"].items())[:10]:
            lines.append(f"- {param}: ±{unc:.3f}")

    return "\n".join(lines)


def _read_profile_dat(
    export_dir: Optional[str],
) -> tuple:
    """Read SLD profile from refl1d ``problem-1-profile.dat``.

    Returns (z_list, rho_list) or (None, None) if unavailable.
    """
    if not export_dir:
        return None, None
    profile_file = Path(export_dir) / "problem-1-profile.dat"
    if not profile_file.exists():
        return None, None
    try:
        z_vals = []
        rho_vals = []
        with open(profile_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    z_vals.append(float(parts[0]))
                    rho_vals.append(float(parts[1]))
        # Downsample if very dense (0.1 Å step → keep every 10th point)
        if len(z_vals) > 1000:
            step = max(1, len(z_vals) // 500)
            z_vals = z_vals[::step]
            rho_vals = rho_vals[::step]
        logger.info(f"[FITTING] Read SLD profile: {len(z_vals)} points")
        return z_vals, rho_vals
    except Exception as exc:
        logger.warning(f"[FITTING] Could not read profile.dat: {exc}")
        return None, None
