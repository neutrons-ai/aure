"""
FITTING node: Run refl1d optimization.

This node builds a FitProblem from the ModelDefinition JSON and fits
the data using bumps.  Supports multiple fitting methods:
- 'lm': Levenberg-Marquardt (fast, local optimizer)
- 'de': Differential Evolution (global optimizer)
- 'dream': MCMC for uncertainty quantification
"""

import copy
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np

from ..state import ReflectivityState, FitResult, PerFileFitResult, Message
from .model_builder import build_problem, build_multi_problem, build_states_problem
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

        data_files = state.get("data_files", [])
        is_multi_state = isinstance(model, dict) and len(model.get("states") or []) >= 2
        is_multi = (
            len(data_files) > 1 and isinstance(model, dict) and not is_multi_state
        )

        if is_multi_state:
            result = run_states_refl1d_fit(
                model_definition=model,
                method=method,
                iteration=iteration,
                steps=steps,
                burn=burn,
                export_dir=export_dir,
            )
        elif is_multi:
            result = run_multi_refl1d_fit(
                model_definition=model,
                data_files=data_files,
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

        # Update best chi2 and save best model. Deepcopy so downstream
        # mutations of current_model (refine carry-over, state-metadata
        # attachment) can't silently corrupt the regression snapshot the
        # evaluation guardrail will restore on χ² worsening.
        best = state.get("best_chi2")
        if best is None or result["chi_squared"] < best:
            updates["best_chi2"] = result["chi_squared"]
            updates["best_model"] = copy.deepcopy(model)
            logger.info(f"[FITTING] New best χ² = {result['chi_squared']:.3f}")

        # Update best BIC (complexity-penalized score)
        if isinstance(model, dict):
            if is_multi_state:
                # Multi-state: count unique free Parameters across all states
                # (avoid double-counting tied params); count data points
                # across every dataset of every state.
                n_params = result.get("_n_free_params") or _count_free_params(model)
                n_data = 0
                for st in model.get("states") or []:
                    for ds in st.get("data_files") or []:
                        q = ds.get("Q") or []
                        n_data += len(q)
                # Fall back to flattened state['Q'] when per-state Q not loaded
                if n_data == 0:
                    n_data = len(state.get("Q", []))
            else:
                n_data = len(state.get("Q", []))
                n_params = _count_free_params(model)
            bic = _compute_bic(result["chi_squared"], n_data, n_params)
            result["bic"] = bic
            logger.info(f"[FITTING] BIC = {bic:.1f} (k={n_params}, n={n_data})")
            best_bic = state.get("best_bic")
            if best_bic is None or bic < best_bic:
                updates["best_bic"] = bic
                updates["best_bic_model"] = copy.deepcopy(model)
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


def run_multi_refl1d_fit(
    model_definition: dict,
    data_files: list[dict],
    method: str = "lm",
    iteration: int = 0,
    steps: int = 1000,
    burn: int = 1000,
    export_dir: Optional[str] = None,
) -> FitResult:
    """Execute a joint fit across multiple data files (co-refinement).

    A single sample is shared across all experiments so that all
    structural parameters are automatically tied.  Each data file gets
    its own probe with an independent intensity parameter.

    Parameters
    ----------
    model_definition
        A ``ModelDefinition`` dict.
    data_files
        List of ``DatasetInfo`` dicts (``file``, ``label``).
    method, iteration, steps, burn, export_dir
        Same as :func:`run_refl1d_fit`.

    Returns
    -------
    FitResult
        Aggregate results with per-file breakdowns in ``per_file_results``.
    """
    from bumps.fitters import fit as bumps_fit

    problem, experiments, sorted_data_files = build_multi_problem(
        model_definition, data_files
    )
    n_files = len(sorted_data_files)
    logger.info(
        f"[FITTING] Running {method.upper()} co-refinement across {n_files} files..."
    )

    fit_options = _build_fit_options(method, steps, burn, export_dir)
    result = bumps_fit(problem, **fit_options)

    return _extract_multi_bumps_results(
        problem=problem,
        experiments=experiments,
        data_files=sorted_data_files,
        fit_result=result,
        method=method,
        iteration=iteration,
        export_dir=export_dir,
    )


def run_states_refl1d_fit(
    model_definition: dict,
    method: str = "lm",
    iteration: int = 0,
    steps: int = 1000,
    burn: int = 1000,
    export_dir: Optional[str] = None,
) -> FitResult:
    """Execute a multi-state co-refinement fit.

    Builds a single :func:`build_states_problem` ``FitProblem`` with
    cross-state parameter aliasing and emits a per-file
    :class:`PerFileFitResult` (with the ``state`` field populated) for
    every dataset across every state.

    Per-state ``profile.dat`` files are written under
    ``{export_dir}/state_{name}/`` for downstream consumption.
    """
    from bumps.fitters import fit as bumps_fit

    problem, experiments_by_state, sorted_files_by_state = build_states_problem(
        model_definition
    )

    n_states = len(experiments_by_state)
    n_files = sum(len(v) for v in experiments_by_state.values())
    logger.info(
        f"[FITTING] Running {method.upper()} multi-state co-refinement: "
        f"{n_states} states, {n_files} files total"
    )

    fit_options = _build_fit_options(method, steps, burn, export_dir)
    result = bumps_fit(problem, **fit_options)

    return _extract_states_bumps_results(
        problem=problem,
        experiments_by_state=experiments_by_state,
        sorted_files_by_state=sorted_files_by_state,
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
        per_file_results=None,
        issues=[],
        suggestions=[],
    )


def _extract_multi_bumps_results(
    problem,
    experiments: list,
    data_files: list[dict],
    fit_result,
    method: str,
    iteration: int,
    export_dir: Optional[str] = None,
) -> FitResult:
    """Extract fit results from a multi-experiment FitProblem.

    Returns an aggregate ``FitResult`` with per-file breakdowns stored
    in ``per_file_results``.
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="bumps")

    # Aggregate chi-squared
    chi_squared = problem.chisq()

    # Extract parameters.  Structural params are shared (single Parameter
    # object across experiments) but per-probe params (intensity,
    # sample_broadening, theta_offset) exist once per experiment and often
    # carry the same ``.name`` — which would collide in a dict.  Build a
    # map of id(par) -> file-label so we can disambiguate such names.
    per_probe_attrs = ("intensity", "sample_broadening", "theta_offset")
    probe_param_labels: dict[int, str] = {}
    for exp, ds in zip(experiments, data_files):
        probe = getattr(exp, "probe", None)
        if probe is None:
            continue
        for attr in per_probe_attrs:
            par = getattr(probe, attr, None)
            if par is not None:
                probe_param_labels[id(par)] = ds.get("label", "")

    parameters: dict = {}
    uncertainties: dict = {}
    param_bounds: dict = {}
    for i, par in enumerate(problem._parameters):
        name = str(par.name)
        # Disambiguate per-probe parameter names by appending the file label
        # when the base name would collide with an already-seen entry.
        if name in parameters and id(par) in probe_param_labels:
            label = probe_param_labels[id(par)]
            if label:
                name = f"{name} {label}"
        elif id(par) in probe_param_labels:
            # Even on first sight, if this is a per-probe param and the
            # same base name will appear again for another probe, pre-empt
            # the collision by suffixing now.
            label = probe_param_labels[id(par)]
            same_name_probes = sum(
                1
                for p in problem._parameters
                if str(p.name) == name and id(p) in probe_param_labels
            )
            if label and same_name_probes > 1:
                name = f"{name} {label}"
        parameters[name] = par.value
        if hasattr(fit_result, "dx") and fit_result.dx is not None:
            try:
                if i < len(fit_result.dx):
                    uncertainties[name] = fit_result.dx[i]
            except (IndexError, TypeError):
                pass
        if par.bounds is not None:
            lo, hi = par.bounds
            param_bounds[name] = [float(lo), float(hi)]

    converged = chi_squared < 100
    if hasattr(fit_result, "success"):
        converged = fit_result.success

    logger.info(f"[FITTING] Multi-file fit complete: aggregate χ² = {chi_squared:.3f}")
    for name, value in parameters.items():
        unc_str = (
            f" ± {uncertainties.get(name, 0):.3f}" if name in uncertainties else ""
        )
        logger.info(f"[FITTING]   {name}: {value:.3f}{unc_str}")

    # Per-file theory curves, chi2, and residuals
    per_file: list[PerFileFitResult] = []
    all_Q: list[float] = []
    all_R: list[float] = []
    all_residuals: list[float] = []
    all_residual_ratio: list[float] = []

    for idx, (exp, ds) in enumerate(zip(experiments, data_files)):
        pf: dict[str, Any] = {"file": ds["file"], "label": ds["label"]}
        try:
            exp.update()
            Q_arr, R_arr = exp.reflectivity()
            pf["Q_fit"] = Q_arr.tolist()
            pf["R_fit"] = R_arr.tolist()
            all_Q.extend(pf["Q_fit"])
            all_R.extend(pf["R_fit"])

            # Per-file chi2 (Experiment has no .chisq(); compute from residuals)
            resid = exp.residuals()
            n_pts = len(resid)
            pf["chi_squared"] = (
                float(np.sum(resid**2) / n_pts) if n_pts > 0 else float("inf")
            )
            logger.info(f"[FITTING]   {ds['label']}: χ² = {pf['chi_squared']:.3f}")

            # Residuals
            R_data = exp.probe.R
            dR_data = exp.probe.dR
            R_fit_arr = np.array(pf["R_fit"])
            res = []
            ratio = []
            if R_data is not None and len(R_data) == len(R_fit_arr):
                if dR_data is not None and len(dR_data) == len(R_data):
                    safe_dR = np.maximum(np.abs(dR_data), 1e-20)
                    res = ((R_data - R_fit_arr) / safe_dR).tolist()
                safe_R_fit = np.maximum(R_fit_arr, 1e-20)
                ratio = (R_data / safe_R_fit).tolist()
            pf["residuals"] = res
            pf["residual_ratio"] = ratio
            all_residuals.extend(res)
            all_residual_ratio.extend(ratio)
        except Exception as e:
            logger.warning(
                f"[FITTING] Could not extract results for {ds['label']}: {e}"
            )
            pf["Q_fit"] = []
            pf["R_fit"] = []
            pf["chi_squared"] = float("inf")
            pf["residuals"] = []
            pf["residual_ratio"] = []

        per_file.append(PerFileFitResult(**pf))

    # Read SLD profile
    sld_z, sld_rho = _read_profile_dat(export_dir)

    return FitResult(
        iteration=iteration,
        method=method,
        chi_squared=chi_squared,
        converged=converged,
        parameters=parameters,
        uncertainties=uncertainties if uncertainties else None,
        bounds=param_bounds if param_bounds else None,
        Q_fit=all_Q,
        R_fit=all_R,
        residuals=all_residuals,
        residual_ratio=all_residual_ratio,
        sld_z=sld_z,
        sld_rho=sld_rho,
        per_file_results=per_file,
        issues=[],
        suggestions=[],
    )


def _extract_states_bumps_results(
    problem,
    experiments_by_state: dict,
    sorted_files_by_state: dict,
    fit_result,
    method: str,
    iteration: int,
    export_dir: Optional[str] = None,
) -> FitResult:
    """Extract fit results from a multi-state co-refinement FitProblem.

    Emits one :class:`PerFileFitResult` per dataset (across all states),
    each tagged with its ``state`` name. The aggregate ``chi_squared`` is
    the bumps total. Per-state ``profile.dat`` files are written under
    ``export_dir/state_{name}/`` when ``export_dir`` is provided.
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="bumps")

    chi_squared = problem.chisq()

    # Deduplicated free parameter count: bumps usually deduplicates by
    # identity already, but a defensive id()-keyed dedup keeps us safe
    # from any future change.
    unique_params: dict[int, Any] = {}
    for par in problem._parameters:
        unique_params.setdefault(id(par), par)
    n_free_params = len(unique_params)

    # Build name → uncertainty / bounds mapping over deduplicated params.
    parameters: dict = {}
    uncertainties: dict = {}
    param_bounds: dict = {}
    for i, par in enumerate(problem._parameters):
        name = str(par.name)
        if name in parameters:
            # Same Parameter referenced twice (tied across states): keep first.
            continue
        parameters[name] = par.value
        if hasattr(fit_result, "dx") and fit_result.dx is not None:
            try:
                if i < len(fit_result.dx):
                    uncertainties[name] = fit_result.dx[i]
            except (IndexError, TypeError):
                pass
        if par.bounds is not None:
            lo, hi = par.bounds
            param_bounds[name] = [float(lo), float(hi)]

    converged = chi_squared < 100
    if hasattr(fit_result, "success"):
        converged = fit_result.success

    logger.info(
        f"[FITTING] Multi-state fit complete: aggregate χ² = {chi_squared:.3f}, "
        f"k={n_free_params}"
    )

    # Per-state, per-file extraction.
    per_file: list[PerFileFitResult] = []
    all_Q: list[float] = []
    all_R: list[float] = []
    all_residuals: list[float] = []
    all_residual_ratio: list[float] = []

    for state_name, experiments in experiments_by_state.items():
        data_files = sorted_files_by_state[state_name]
        state_chi2_sum = 0.0
        state_npts = 0
        for exp, ds in zip(experiments, data_files):
            pf: dict[str, Any] = {
                "file": ds["file"],
                "label": ds.get("label", ""),
                "state": state_name,
            }
            try:
                exp.update()
                Q_arr, R_arr = exp.reflectivity()
                pf["Q_fit"] = Q_arr.tolist()
                pf["R_fit"] = R_arr.tolist()
                all_Q.extend(pf["Q_fit"])
                all_R.extend(pf["R_fit"])

                resid = exp.residuals()
                n_pts = len(resid)
                pf["chi_squared"] = (
                    float(np.sum(resid**2) / n_pts) if n_pts > 0 else float("inf")
                )
                state_chi2_sum += float(np.sum(resid**2))
                state_npts += n_pts
                logger.info(
                    f"[FITTING]   [{state_name}] {pf['label']}: "
                    f"χ² = {pf['chi_squared']:.3f}"
                )

                R_data = exp.probe.R
                dR_data = exp.probe.dR
                R_fit_arr = np.array(pf["R_fit"])
                res = []
                ratio = []
                if R_data is not None and len(R_data) == len(R_fit_arr):
                    if dR_data is not None and len(dR_data) == len(R_data):
                        safe_dR = np.maximum(np.abs(dR_data), 1e-20)
                        res = ((R_data - R_fit_arr) / safe_dR).tolist()
                    safe_R_fit = np.maximum(R_fit_arr, 1e-20)
                    ratio = (R_data / safe_R_fit).tolist()
                pf["residuals"] = res
                pf["residual_ratio"] = ratio
                all_residuals.extend(res)
                all_residual_ratio.extend(ratio)
            except Exception as e:
                logger.warning(
                    f"[FITTING] Could not extract results for "
                    f"[{state_name}] {pf.get('label', '')}: {e}"
                )
                pf["Q_fit"] = []
                pf["R_fit"] = []
                pf["chi_squared"] = float("inf")
                pf["residuals"] = []
                pf["residual_ratio"] = []

            per_file.append(PerFileFitResult(**pf))

        if state_npts > 0:
            logger.info(
                f"[FITTING]   [{state_name}] aggregate χ² = "
                f"{state_chi2_sum / state_npts:.3f}"
            )

        # Write per-state profile.dat under export_dir/state_<name>/.
        if export_dir and experiments:
            try:
                _write_state_profile(experiments[0], export_dir, state_name)
            except Exception as exc:
                logger.warning(
                    f"[FITTING] Could not write profile for state {state_name}: {exc}"
                )

    sld_z, sld_rho = _read_profile_dat(export_dir)

    fr = FitResult(
        iteration=iteration,
        method=method,
        chi_squared=chi_squared,
        converged=converged,
        parameters=parameters,
        uncertainties=uncertainties if uncertainties else None,
        bounds=param_bounds if param_bounds else None,
        Q_fit=all_Q,
        R_fit=all_R,
        residuals=all_residuals,
        residual_ratio=all_residual_ratio,
        sld_z=sld_z,
        sld_rho=sld_rho,
        per_file_results=per_file,
        issues=[],
        suggestions=[],
    )
    # Pass dedup'd free-param count through to fitting_node for BIC.
    fr["_n_free_params"] = n_free_params  # type: ignore[typeddict-unknown-key]
    return fr


def _write_state_profile(experiment, export_dir: str, state_name: str) -> None:
    """Write a 2-column ``# z rho`` profile.dat for one state's sample."""
    state_dir = Path(export_dir) / f"state_{state_name}"
    state_dir.mkdir(parents=True, exist_ok=True)

    sample = getattr(experiment, "sample", None)
    if sample is None:
        return

    # refl1d Stack.render(probe) returns layered slabs; use sample.render
    # via the experiment to get the smoothed profile (z, rho, irho).
    try:
        z, rho, _irho = experiment.smooth_profile()
    except Exception:
        # Fallback for older refl1d: compute through Stack
        z, rho, _irho = sample.render(experiment.probe)

    profile_file = state_dir / "profile.dat"
    with open(profile_file, "w") as f:
        f.write("# z rho\n")
        for zi, ri in zip(z, rho):
            f.write(f"{float(zi):.6f}  {float(ri):.6e}\n")
    logger.info(f"[FITTING] Wrote per-state profile: {profile_file}")


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

    # Per-file chi2 for multi-file co-refinement
    per_file = result.get("per_file_results")
    if per_file:
        lines.append("")
        lines.append("**Per-file χ²:**")
        for pf in per_file:
            lines.append(f"- {pf['label']}: χ² = {pf['chi_squared']:.2f}")

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
