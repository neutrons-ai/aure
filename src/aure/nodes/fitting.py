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
import re
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np

from ..state import ReflectivityState, FitResult, PerFileFitResult, Message
from .model_builder import (
    build_problem,
    build_multi_problem,
    build_states_problem,
    data_chisq,
    needs_states_problem,
)
from .evaluation import _count_free_params, _compute_bic, _get_chi2_min

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

    # Name the exported FitProblem so bumps never writes None-*.dat / None.json.
    # Falls back through output-dir basename / data-file stem when no explicit
    # model_name was supplied (see _resolve_model_name). Persist it onto the
    # state so it's recorded in checkpoints / run_info.json and stays stable
    # across refinement iterations.
    model_name = _resolve_model_name(state, model)
    updates["model_name"] = model_name

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
        is_multi_state = isinstance(model, dict) and needs_states_problem(model)
        is_multi = (
            len(data_files) > 1 and isinstance(model, dict) and not is_multi_state
        )

        # ========== Thin-layer SLD mode enumeration (gated, single-file) ==========
        # Thin layers sit on a Δρ·t degeneracy ridge with distinct SLD modes a
        # single optimizer run will not hop between. When enabled, re-seed each
        # thin layer's SLD across discrete levels, cheaply polish each, and
        # start the main fit from the best basin. Off by default; never fatal.
        if (
            _mode_enumeration_enabled()
            and not is_multi_state
            and not is_multi
            and isinstance(model, dict)
        ):
            try:
                model = _enumerate_thin_layer_modes(model, state)
            except Exception as e:  # pragma: no cover - safety net
                logger.warning(f"[FITTING] Mode enumeration skipped: {e}")

        result = run_fit_for_model(
            model=model,
            data_files=data_files,
            method=method,
            iteration=iteration,
            steps=steps,
            burn=burn,
            export_dir=export_dir,
            model_name=model_name,
        )

        updates["fit_results"] = [result]
        updates["current_chi2"] = result["chi_squared"]
        logger.info(f"[FITTING] Completed with χ² = {result['chi_squared']:.3f}")

        # Update best chi2 and save best model. Deepcopy so downstream
        # mutations of current_model (refine carry-over, state-metadata
        # attachment) can't silently corrupt the regression snapshot the
        # evaluation guardrail will restore on χ² worsening.
        best = state.get("best_chi2")
        if _wins_baseline(result["chi_squared"], best, state):
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
            # BIC is monotone in χ², so a sub-floor fit wins here for the same
            # untrustworthy reason — and this guardrail has no slack and marks the
            # tried hypothesis `rejected`, so it does more damage than the χ² one.
            best_bic = state.get("best_bic")
            if _wins_baseline(
                result["chi_squared"],
                state.get("best_bic_chi2"),
                state,
                candidate_score=bic,
                incumbent_score=best_bic,
            ):
                updates["best_bic"] = bic
                updates["best_bic_model"] = copy.deepcopy(model)
                updates["best_bic_chi2"] = result["chi_squared"]
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


def _wins_baseline(
    candidate_chi2: float,
    incumbent_chi2: Optional[float],
    state: ReflectivityState,
    *,
    candidate_score: Optional[float] = None,
    incumbent_score: Optional[float] = None,
) -> bool:
    """Whether a fit should become the refinement loop's regression baseline.

    The baseline is what ``evaluation``'s χ² and BIC guardrails revert *to*, so a
    fit below the acceptance floor must not claim it: a reduced χ² far under 1 is
    evidence about the ``dR`` column rather than the structure, and one such
    noise-absorbing iteration would make every later honest fit read as a
    regression and get reverted — pinning the run to a model it should have moved
    away from.

    An in-window fit therefore always displaces a sub-floor incumbent, whatever the
    scores say. A sub-floor fit is still recorded when nothing in-window exists yet,
    because leaving the guardrails with *no* baseline would disable the check that
    stops the LLM refining an already-degraded model — a worse failure than a
    questionable baseline.

    *candidate_score* / *incumbent_score* rank by something other than χ² (BIC);
    the floor test always reads χ², since that is what the floor is about.
    """
    floor = _get_chi2_min(state)
    cand_sub = floor > 0 and candidate_chi2 < floor
    inc_sub = floor > 0 and incumbent_chi2 is not None and incumbent_chi2 < floor

    if candidate_score is None:
        candidate_score, incumbent_score = candidate_chi2, incumbent_chi2

    if incumbent_score is None:
        return True
    # Trustworthy beats untrustworthy regardless of score; the reverse never holds.
    if inc_sub and not cand_sub:
        logger.info(
            "[FITTING] Replacing the sub-floor regression baseline (χ²=%.3f, below "
            "the χ² ≥ %.3f floor) with this fit (χ²=%.3f) — a noise-absorbing fit "
            "must not be what later fits are judged against",
            incumbent_chi2,
            floor,
            candidate_chi2,
        )
        return True
    if cand_sub and not inc_sub:
        return False
    return candidate_score < incumbent_score


def run_fit_for_model(
    model: dict,
    data_files: list,
    method: str,
    iteration: int,
    steps: int,
    burn: int,
    export_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> FitResult:
    """Dispatch to the fitter that matches a model's shape.

    Chooses among the single-experiment, multi-file co-refinement, and
    multi-state co-refinement fitters using exactly the same predicates as
    :func:`fitting_node`, so the exploration loop and the terminal
    ``final_fit`` polish build an identical ``FitProblem`` for the same model.

    Deliberately does NOT run thin-layer SLD mode enumeration: that is an
    exploration-seeding step for escaping local minima, not part of
    characterising an already-selected model.
    """
    is_multi_state = isinstance(model, dict) and needs_states_problem(model)
    is_multi = len(data_files) > 1 and isinstance(model, dict) and not is_multi_state

    if is_multi_state:
        return run_states_refl1d_fit(
            model_definition=model,
            method=method,
            iteration=iteration,
            steps=steps,
            burn=burn,
            export_dir=export_dir,
            model_name=model_name,
        )
    if is_multi:
        return run_multi_refl1d_fit(
            model_definition=model,
            data_files=data_files,
            method=method,
            iteration=iteration,
            steps=steps,
            burn=burn,
            export_dir=export_dir,
            model_name=model_name,
        )
    return run_refl1d_fit(
        model_definition=model,
        method=method,
        iteration=iteration,
        steps=steps,
        burn=burn,
        export_dir=export_dir,
        model_name=model_name,
    )


def _mode_enumeration_enabled() -> bool:
    """Thin-layer SLD mode enumeration is opt-in via the MODE_ENUMERATION env
    var (default off), so baseline behavior and existing tests are unchanged."""
    return os.environ.get("MODE_ENUMERATION", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _resolution_limit(state: ReflectivityState) -> Optional[float]:
    """Real-space resolution limit d ≈ 2π / Q_max, or None if Q is unavailable."""
    q = state.get("Q") or []
    if len(q) == 0:
        return None
    q_max = float(np.max(np.asarray(q, dtype=float)))
    if q_max <= 0:
        return None
    return 2.0 * np.pi / q_max


def _enumerate_thin_layer_modes(model: dict, state: ReflectivityState) -> dict:
    """Re-seed thin layers to the best SLD basin before the main fit.

    For each layer thinner than ``THIN_LAYER_MODE_K × (2π/Q_max)`` (default
    K=1), try a few discrete SLD seeds spanning the layer's allowed range,
    cheaply polish each with a local optimizer, and adopt the lowest-χ² basin.
    Layers are visited greedily, carrying each improvement forward. Returns a
    (possibly re-seeded) copy of ``model``; on any trouble it returns the input
    unchanged. All choices are logged — nothing is silently capped.
    """
    d_res = _resolution_limit(state)
    if d_res is None:
        logger.info("[FITTING] Mode enumeration: no Q data; skipping")
        return model
    k = float(os.environ.get("THIN_LAYER_MODE_K", "1.0"))
    n_seeds = int(os.environ.get("THIN_LAYER_MODE_SEEDS", "3"))
    thin_cutoff = k * d_res

    layers = model.get("layers") or []
    thin_idx = [
        i
        for i, ly in enumerate(layers)
        if float(ly.get("thickness", 0.0)) < thin_cutoff
    ]
    if not thin_idx:
        logger.info(
            f"[FITTING] Mode enumeration: no layers thinner than "
            f"{thin_cutoff:.0f} Å; skipping"
        )
        return model

    working = copy.deepcopy(model)
    # Baseline χ² for the current seeds.
    try:
        best_chi2 = data_chisq(build_problem(working))
    except Exception as e:
        logger.warning(f"[FITTING] Mode enumeration: baseline build failed: {e}")
        return model
    logger.info(
        f"[FITTING] Mode enumeration: {len(thin_idx)} thin layer(s) "
        f"(< {thin_cutoff:.0f} Å), baseline χ²={best_chi2:.3f}"
    )

    from bumps.fitters import fit as bumps_fit

    for i in thin_idx:
        layer = working["layers"][i]
        name = layer.get("name", f"layer_{i}")
        sld0 = float(layer.get("sld", 0.0))
        lo = float(layer.get("sld_min", sld0 - 2.5))
        hi = float(layer.get("sld_max", sld0 + 2.5))
        seeds = list(np.linspace(lo, hi, max(2, n_seeds)))

        layer_best_chi2 = best_chi2
        layer_best_sld = sld0
        for seed in seeds:
            trial = copy.deepcopy(working)
            trial["layers"][i]["sld"] = float(seed)
            try:
                problem = build_problem(trial)
                bumps_fit(problem, method="amoeba", steps=1000)
                c2 = float(data_chisq(problem))
            except Exception as e:
                logger.debug(f"[FITTING]   {name} seed ρ={seed:.2f} failed: {e}")
                continue
            logger.info(f"[FITTING]   {name} seed ρ={seed:.2f} → χ²={c2:.3f}")
            if c2 < layer_best_chi2:
                layer_best_chi2 = c2
                layer_best_sld = float(seed)

        if layer_best_sld != sld0 and layer_best_chi2 < best_chi2:
            logger.info(
                f"[FITTING] Mode enumeration: {name} re-seeded ρ {sld0:.2f} → "
                f"{layer_best_sld:.2f} (χ² {best_chi2:.3f} → {layer_best_chi2:.3f})"
            )
            working["layers"][i]["sld"] = layer_best_sld
            best_chi2 = layer_best_chi2

    return working


def run_refl1d_fit(
    model_definition: dict,
    method: str = "lm",
    iteration: int = 0,
    steps: int = 1000,
    burn: int = 1000,
    export_dir: Optional[str] = None,
    model_name: Optional[str] = None,
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
        model_name: Optional name for the FitProblem (drives export filenames)

    Returns:
        FitResult dictionary
    """
    problem = build_problem(model_definition)
    _name_problem(problem, model_name)

    # Configure fit options based on method
    fit_options = _build_fit_options(method, steps, burn)

    # Run the fit
    logger.info(f"[FITTING] Running {method.upper()} with bumps.fit...")
    result = _run_bumps_fit(problem, fit_options, export_dir, model_name)

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
    model_name: Optional[str] = None,
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
    problem, experiments, sorted_data_files = build_multi_problem(
        model_definition, data_files
    )
    _name_problem(problem, model_name)
    n_files = len(sorted_data_files)
    logger.info(
        f"[FITTING] Running {method.upper()} co-refinement across {n_files} files..."
    )

    fit_options = _build_fit_options(method, steps, burn)
    result = _run_bumps_fit(problem, fit_options, export_dir, model_name)

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
    model_name: Optional[str] = None,
) -> FitResult:
    """Execute a multi-state co-refinement fit.

    Builds a single :func:`build_states_problem` ``FitProblem`` with
    cross-state parameter aliasing and emits a per-file
    :class:`PerFileFitResult` (with the ``state`` field populated) for
    every dataset across every state.

    Per-state ``profile.dat`` files are written under
    ``{export_dir}/state_{name}/`` for downstream consumption.
    """
    problem, experiments_by_state, sorted_files_by_state = build_states_problem(
        model_definition
    )
    _name_problem(problem, model_name)

    n_states = len(experiments_by_state)
    n_files = sum(len(v) for v in experiments_by_state.values())
    logger.info(
        f"[FITTING] Running {method.upper()} multi-state co-refinement: "
        f"{n_states} states, {n_files} files total"
    )

    fit_options = _build_fit_options(method, steps, burn)
    result = _run_bumps_fit(problem, fit_options, export_dir, model_name)

    return _extract_states_bumps_results(
        problem=problem,
        experiments_by_state=experiments_by_state,
        sorted_files_by_state=sorted_files_by_state,
        fit_result=result,
        method=method,
        iteration=iteration,
        export_dir=export_dir,
    )


def _sanitize_model_name(value) -> str:
    """Normalise a candidate model name; return ``""`` if unusable.

    Rejects empty / ``None``-like values and replaces filesystem-hostile
    characters (spaces, slashes, …) so the result is a safe filename stem.
    """
    if not value:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    return re.sub(r"[^\w.\-]+", "_", text)


def _resolve_model_name(state: dict, model) -> str:
    """Resolve a stable, non-empty basename for refl1d/bumps export files.

    Bumps names every exported artifact after ``FitProblem.name`` (and the
    ``export_fit`` basename). When that is unset, every file becomes
    ``None-*`` / ``None.json`` — the regression this guards against. We always
    return a real name, trying in order:

    1. an explicit ``user_config['model_name']`` (e.g. from a setup YAML),
    2. ``state['model_name']`` (set by some callers),
    3. the model definition's own ``model_name`` / ``name``,
    4. the run's output-directory basename (e.g. ``230536``),
    5. the primary data file's stem,
    6. a literal ``"model"`` as the last resort.
    """
    uc = state.get("user_config") or {}
    mdl = model if isinstance(model, dict) else {}
    candidates = [
        uc.get("model_name"),
        state.get("model_name"),
        mdl.get("model_name"),
        mdl.get("name"),
    ]

    out = state.get("output_dir")
    if out:
        candidates.append(Path(out).name)

    data_file = state.get("data_file")
    if not data_file:
        dfs = state.get("data_files") or []
        if dfs and isinstance(dfs[0], dict):
            data_file = dfs[0].get("file")
    if data_file:
        candidates.append(Path(str(data_file)).stem)

    for cand in candidates:
        name = _sanitize_model_name(cand)
        if name:
            return name
    return "model"


def _name_problem(problem, model_name: Optional[str]) -> None:
    """Name the FitProblem so bumps' export uses it as the filename stem.

    Without a name, bumps writes the refl1d output as ``None-*.dat`` /
    ``None.json``; setting it yields ``<model_name>-*`` / ``<model_name>.json``.
    """
    if model_name:
        try:
            problem.name = model_name
        except Exception:  # pragma: no cover - defensive
            logger.warning("[FITTING] Could not set FitProblem.name=%r", model_name)


def _build_fit_options(method: str, steps: int, burn: int) -> dict:
    """Build the options dict for bumps.fitters.fit.

    The export directory is intentionally NOT passed here; we export the
    output ourselves after the fit returns (see :func:`_run_bumps_fit`).
    """
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

    return fit_options


def _run_bumps_fit(
    problem,
    fit_options: dict,
    export_dir: Optional[str],
    model_name: Optional[str],
):
    """Run a bumps fit and export the full refl1d output directory.

    We deliberately do NOT pass ``export=`` to ``bumps.fitters.fit``. In
    bumps 1.0.4 that code path calls
    ``export_fit(export, problem, result.state, ...)`` — forwarding the bare
    ``MCMCDraw`` as the ``fit`` argument. ``export_fit`` then does
    ``getattr(fit, "fit_state", getattr(fit, "state", None))``, which an
    ``MCMCDraw`` does not satisfy, so it resolves to ``None`` and the entire
    uncertainty/MCMC block is silently skipped. The result is that dream runs
    only emit the model files (``-profile.dat``, ``-refl.dat``, ``-expt.json``,
    ``-model*.png``, ``.out``, ``.par``) and lose the uncertainty summary
    (``-err.json``), the MCMC chain (``-chain.mc.gz`` / ``-point.mc.gz`` /
    ``-stats.mc.gz``), and the dream diagnostic plots (corr/trace/logp/vars).

    Instead we call ``export_fit`` ourselves with the ``OptimizeResult`` — the
    documented usage (see ``bumps.fitters`` help: ``bp.export_fit(dir, problem,
    fitresult)``) — which correctly recovers ``result.state`` and writes the
    complete output, including uncertainties and chains for dream.
    """
    from bumps.fitters import fit as bumps_fit

    result = bumps_fit(problem, **fit_options)

    if export_dir:
        try:
            from bumps.webview.server.api import export_fit

            # basename=None lets export_fit fall back to problem.name (set by
            # _name_problem), preserving the existing "<model_name>-*" naming.
            export_fit(export_dir, problem, result, basename=model_name)
            logger.info(f"[FITTING] Exported full refl1d output to {export_dir}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"[FITTING] Output export to {export_dir} failed: {e}")

    return result


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
    chi_squared = data_chisq(problem)

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
    sld_z, sld_rho = _read_profile_dat(export_dir, getattr(problem, "name", None))

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
    chi_squared = data_chisq(problem)

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
    sld_z, sld_rho = _read_profile_dat(export_dir, getattr(problem, "name", None))

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

    chi_squared = data_chisq(problem)

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

        # Write per-state profile.dat under export_dir/state_<name>/, then read it
        # straight back onto this state's entries. refl1d exports one profile per
        # model, so the FitResult's top-level pair is states[0]'s alone — without
        # this the evaluator can only ever check one state of a co-refinement.
        if export_dir and experiments:
            try:
                _write_state_profile(experiments[0], export_dir, state_name)
                z, rho = _read_state_profile(export_dir, state_name)
                if z and rho:
                    for pf in per_file:
                        if pf.get("state") == state_name:
                            pf["sld_z"] = z
                            pf["sld_rho"] = rho
            except Exception as exc:
                logger.warning(
                    f"[FITTING] Could not write profile for state {state_name}: {exc}"
                )

    sld_z, sld_rho = _read_profile_dat(export_dir, getattr(problem, "name", None))

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


def _find_profile_dat(export_dir: str, problem_name: Optional[str] = None):
    """Locate refl1d's ``*-1-profile.dat`` in an export directory.

    The basename is the FitProblem's name, which :func:`_name_problem` sets from
    ``model_name`` — so it is only ``problem-1-profile.dat`` when the problem was
    left unnamed. Looking for that one fixed name silently found nothing on every
    named run, which left ``sld_z``/``sld_rho`` empty and thereby disabled the
    SLD-profile artifact detector in evaluation (it returns early without a
    profile).

    *problem_name* identifies **this** fit's export. Without it the choice was the
    fixed default and then ``sorted(glob)[0]``, so a stale export left in the
    directory — anything alphabetically earlier, and ``problem-1-profile.dat``
    always — shadowed the file the current fit had just written, and the artifact
    detector silently judged the wrong profile. Order: this fit's own export, then
    the documented default when the problem was unnamed (in which case that *is*
    this fit's export), then the freshest by mtime, with the name as a tie-break so
    the result is deterministic.
    """
    d = Path(export_dir)
    hits = sorted(d.glob("*-1-profile.dat"))
    if not hits:
        return None

    if problem_name:
        want = d / f"{problem_name}-1-profile.dat"
        if want in hits:
            return want
    else:
        default = d / "problem-1-profile.dat"
        if default in hits:
            return default

    return max(hits, key=lambda p: (p.stat().st_mtime, p.name))


def _read_profile_dat(
    export_dir: Optional[str],
    problem_name: Optional[str] = None,
) -> tuple:
    """Read SLD profile from refl1d's ``*-1-profile.dat``.

    *problem_name* pins the read to this fit's own export; see
    :func:`_find_profile_dat`. Returns (z_list, rho_list) or (None, None).
    """
    if not export_dir:
        return None, None
    profile_file = _find_profile_dat(export_dir, problem_name)
    if profile_file is None:
        return None, None
    return _parse_profile_dat(profile_file)


def _read_state_profile(export_dir: Optional[str], state_name: str) -> tuple:
    """Read one state's profile from ``export_dir/state_<name>/profile.dat``.

    Returns (z_list, rho_list) or (None, None) if unavailable. The evaluator marks
    a co-refinement verified only when *every* state reports one, so a miss here
    leaves the whole fit unverified rather than judged on the states that did.
    """
    if not export_dir:
        return None, None
    profile_file = Path(export_dir) / f"state_{state_name}" / "profile.dat"
    if not profile_file.is_file():
        return None, None
    return _parse_profile_dat(profile_file)


def _parse_profile_dat(profile_file) -> tuple:
    """Parse a 2-column ``z rho`` profile file, downsampling a dense one."""
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
