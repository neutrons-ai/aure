"""
FINAL_FIT node: optional MCMC (dream) polish of the finalize-selected model.

Exploration is fast local/global optimization (``FIT_METHOD``, e.g. ``amoeba``
or ``de``): it finds the best model structure cheaply but reports no — or only
crude — parameter uncertainties. When uncertainties are wanted, this step runs
one more fit with ``FIT_METHOD_FINAL`` (typically ``dream``) on the single model
the ``finalize`` node already selected, seeded from its fitted values. amoeba
finds the basin; dream then characterises it — which is exactly the regime dream
performs well in (a cold dream from a poor start is what performs badly).

Where it runs
-------------
The runner's terminal block calls this **after** ``finalize`` (so it operates on
the selected winner in ``current_model``) and **before** ``save_final_state`` (so
its refined values + uncertainties land in ``final_state.json`` and drive
``problem.json``). It is the only production path — both the CLI and the Flask
web UI drive fitting through ``run_workflow_with_checkpoints``, never the
compiled graph.

Self-gating (a no-op unless explicitly requested)
-------------------------------------------------
Returns an empty update — zero state churn, no checkpoint — unless ALL hold:

* ``FIT_METHOD_FINAL`` is set and differs from ``FIT_METHOD`` (nothing to gain
  if exploration already used the final method);
* there is a dict ``current_model`` to polish (``finalize`` produced one);
* the SLD-profile check did not veto the selected fit — precise uncertainties on
  a physically impossible model only lend it authority, and χ² cannot catch this
  one because the excursion is *what buys* the low χ², so a vetoed selection
  passes the quality gate below by construction;
* the selected χ² is finite and at or below the quality gate
  (``FINAL_FIT_CHI2_MAX``, default ``CHI2_MAX``) — there is no point spending a
  long MCMC characterising a fit that is already known to be poor.

Correctness of the reported model
---------------------------------
The polish fits the *same structure* ``finalize`` chose, so adopting its refined
values does not conflict with the parsimony selection. When dream holds or
improves the fit (χ² within ``FINAL_SELECTION_TOL`` of the selected χ²) the run
adopts it: ``current_model`` gets the dream values written back (via the same
``finalize._apply_fitted_values`` used elsewhere), ``current_chi2`` becomes the
dream χ², and ``final_fit['adopted']`` tells ``_copy_best_problem_json`` to point
``problem.json`` at the dream export instead of the exploration export. If dream
comes back worse (degenerate — rare, since dream evaluates the seed point), the
better finalize selection is kept untouched and only recorded as not adopted.

``best_model`` / ``best_chi2`` are never written here: they are the refinement
loop's regression baseline that ``aure resume`` compares against. Uncertainties
live in the appended dream ``FitResult`` and its ``-err.json`` export, not in the
``ModelDefinition`` (which has no field for them) — matching how a plain
``FIT_METHOD=dream`` run already reports them.
"""

import copy
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..state import Message, ReflectivityState
from .evaluation import _get_chi2_max
from .finalize import (
    _apply_fitted_values,
    _ARTIFACT_MARKER,
    _get_selection_tolerance,
    has_profile_artifact,
)
from .fitting import _resolve_model_name, run_fit_for_model

logger = logging.getLogger(__name__)

_DEFAULT_FINAL_STEPS = 10000


def _final_method() -> str:
    """The requested final-fit method (``FIT_METHOD_FINAL``), lower-cased."""
    return os.environ.get("FIT_METHOD_FINAL", "").strip().lower()


def _explore_method() -> str:
    """The exploration method the refinement loop used (``FIT_METHOD``)."""
    return os.environ.get("FIT_METHOD", "dream").strip().lower()


def _final_steps() -> int:
    """Step/sample budget for the final fit.

    Defaults to 10x the typical exploration budget because a dream fit needs
    far more generations than a fast optimizer to produce usable 1σ error
    bars — reusing a small ``FIT_STEPS`` is the classic way to get
    plausible-looking but meaningless uncertainties.
    """
    return _positive_int(os.environ.get("FIT_STEPS_FINAL"), _DEFAULT_FINAL_STEPS)


def _final_burn(steps: int) -> int:
    """Burn-in for the final fit (``FIT_BURN_FINAL``); defaults to ``steps``."""
    return _positive_int(os.environ.get("FIT_BURN_FINAL"), steps)


def _positive_int(raw: Optional[str], default: int) -> int:
    """Parse a positive int from an env value, falling back to *default*."""
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _quality_gate() -> float:
    """χ² ceiling above which the (expensive) final MCMC is skipped.

    ``FINAL_FIT_CHI2_MAX`` overrides; otherwise the evaluation node's
    ``CHI2_MAX`` acceptance threshold is reused so "good enough to characterise"
    means the same thing as "an acceptable fit".
    """
    raw = os.environ.get("FINAL_FIT_CHI2_MAX")
    if raw is not None:
        try:
            value = float(raw)
            if math.isfinite(value) and value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return _get_chi2_max()


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _profile_veto_reason(state: ReflectivityState) -> Optional[str]:
    """Why the selected fit failed the SLD-profile check, or None if it passed.

    Two sources, because ``final_selection`` only carries the verdict for runs
    finalized after it started recording one: an ``aure import-refl1d`` workspace
    or a checkpoint written earlier has no such field, and those must not be
    silently exempt. Falls back to the selected ``FitResult`` itself, resolved the
    way the report resolves it — ``final_selection["index"]``, else the last fit.
    """
    selection = state.get("final_selection") or {}
    fit_results = state.get("fit_results") or []

    index = selection.get("index") if selection.get("selected") else None
    if isinstance(index, int) and 0 <= index < len(fit_results):
        fit = fit_results[index]
    else:
        fit = fit_results[-1] if fit_results else None

    if selection.get("selected_has_profile_artifact") or has_profile_artifact(fit):
        return (
            selection.get("profile_veto_reason")
            or _artifact_issue(fit)
            or "non-physical SLD-profile excursion"
        )
    return None


def _artifact_issue(fit: Any) -> Optional[str]:
    """The evaluator's own sentence about the excursion, when it recorded one."""
    if not isinstance(fit, dict):
        return None
    for issue in fit.get("issues") or []:
        if _ARTIFACT_MARKER in str(issue).lower():
            return str(issue)
    return None


def final_fit_node(state: ReflectivityState) -> Dict[str, Any]:
    """Optionally run an MCMC polish of the finalize-selected model.

    Returns ``{}`` (no-op, no checkpoint) when the feature is disabled, so a run
    that does not request a final method is completely unaffected.
    """
    final_method = _final_method()
    explore_method = _explore_method()

    # Gate 1: feature off, or exploration already used the final method.
    if not final_method or final_method == explore_method:
        return {}

    updates: Dict[str, Any] = {"current_node": "final_fit", "messages": []}

    # Gate 2: need a concrete model to polish.
    model = state.get("current_model")
    if not isinstance(model, dict) or not model:
        logger.info(
            "[FINAL_FIT] No dict model selected; skipping %s polish", final_method
        )
        updates["final_fit"] = {
            "ran": False,
            "adopted": False,
            "method": final_method,
            "reason": "no dict model to polish",
        }
        return updates

    # Gate 3: never characterise a model the SLD-profile check rejected. Checked
    # ahead of the χ² gate because a vetoed selection has a low χ² by construction
    # and would otherwise be skipped — if at all — for the less useful reason.
    veto = _profile_veto_reason(state)
    if veto:
        logger.warning(
            "[FINAL_FIT] Selected model was vetoed by the SLD-profile check — "
            "skipping %s polish: %s",
            final_method,
            veto,
        )
        updates["final_fit"] = {
            "ran": False,
            "adopted": False,
            "method": final_method,
            "reason": "selected model failed the SLD-profile check",
            "profile_veto": veto,
        }
        return updates

    # Gate 4: only characterise a fit that is already acceptable.
    chi2_before = state.get("current_chi2")
    gate = _quality_gate()
    if not _is_finite_number(chi2_before) or chi2_before > gate:
        logger.info(
            "[FINAL_FIT] Selected χ²=%s exceeds gate %.3f — skipping %s polish",
            _fmt_chi2(chi2_before),
            gate,
            final_method,
        )
        updates["final_fit"] = {
            "ran": False,
            "adopted": False,
            "method": final_method,
            "reason": f"chi2 {_fmt_chi2(chi2_before)} above gate {gate:g}",
            "chi2_gate": gate,
            "chi2_before": chi2_before if _is_finite_number(chi2_before) else None,
        }
        return updates

    steps = _final_steps()
    burn = _final_burn(steps)

    selection = state.get("final_selection") or {}
    sel_iter = selection.get("iteration")
    if not isinstance(sel_iter, int):
        sel_iter = state.get("iteration", 0)

    export_dir: Optional[str] = None
    base_output = state.get("output_dir")
    if base_output:
        export_dir = str(Path(base_output) / "refl1d_output" / f"final_{final_method}")
        Path(export_dir).mkdir(parents=True, exist_ok=True)

    model_name = state.get("model_name") or _resolve_model_name(state, model)
    data_files = state.get("data_files", []) or []

    logger.info(
        "[FINAL_FIT] %s polish on selected model (iter %s, χ²=%.3f, steps=%d, burn=%d)",
        final_method.upper(),
        sel_iter,
        chi2_before,
        steps,
        burn,
    )

    try:
        result = run_fit_for_model(
            model=model,
            data_files=data_files,
            method=final_method,
            iteration=sel_iter,
            steps=steps,
            burn=burn,
            export_dir=export_dir,
            model_name=model_name,
        )
    except Exception as exc:  # never fatal: keep the finalize selection
        logger.warning(
            "[FINAL_FIT] %s polish failed (%s) — keeping the selected model",
            final_method,
            exc,
        )
        updates["final_fit"] = {
            "ran": False,
            "adopted": False,
            "method": final_method,
            "reason": f"fit error: {exc}",
            "chi2_before": chi2_before,
        }
        updates["messages"] = [
            Message(
                role="system",
                content=(
                    f"Final {final_method.upper()} uncertainty fit failed "
                    f"({exc}); reporting the selected model without it."
                ),
                timestamp=None,
            )
        ]
        return updates

    chi2_after = result.get("chi_squared")
    uncertainties = result.get("uncertainties") or {}
    n_unc = len(uncertainties)
    tol = _get_selection_tolerance()
    adopted = _is_finite_number(chi2_after) and chi2_after <= chi2_before * (1.0 + tol)

    # Always persist the dream fit so its uncertainties + chains are on record,
    # whether or not we adopt its point estimate as the reported model.
    updates["fit_results"] = [result]

    final_fit_record = {
        "ran": True,
        "adopted": adopted,
        "method": final_method,
        "steps": steps,
        "burn": burn,
        "export_dir": export_dir,
        "chi2_before": chi2_before,
        "chi2_after": chi2_after if _is_finite_number(chi2_after) else None,
        "n_uncertainties": n_unc,
        "selected_iteration": sel_iter,
    }

    if adopted:
        # Same structure finalize chose, refined by MCMC: write its values back
        # and report them. problem.json follows via final_fit['export_dir'].
        final_model = copy.deepcopy(model)
        n_applied, unapplied = _apply_fitted_values(
            final_model, result.get("parameters") or {}
        )
        updates["current_model"] = final_model
        updates["current_chi2"] = chi2_after
        final_fit_record["values_applied"] = n_applied
        final_fit_record["values_unapplied"] = unapplied[:20]
        logger.info(
            "[FINAL_FIT] Adopted %s polish: χ² %.3f → %.3f, %d parameter uncertaint%s",
            final_method,
            chi2_before,
            chi2_after,
            n_unc,
            "y" if n_unc == 1 else "ies",
        )
    else:
        # Degenerate: MCMC did not hold the seed's fit. Keep the better finalize
        # selection; problem.json stays on the exploration export.
        logger.warning(
            "[FINAL_FIT] %s polish χ²=%s did not improve on the selected χ²=%.3f "
            "(tol %.0f%%) — keeping the selected model; uncertainties recorded only",
            final_method,
            _fmt_chi2(chi2_after),
            chi2_before,
            tol * 100,
        )

    updates["final_fit"] = final_fit_record
    updates["messages"] = [
        Message(
            role="assistant",
            content=_format_final_fit(final_fit_record),
            timestamp=None,
        )
    ]
    return updates


def _fmt_chi2(value: Any) -> str:
    """Format a χ² that may be None, inf or nan."""
    if _is_finite_number(value):
        return f"{value:.4f}"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "diverged"
    return "n/a"


def _format_final_fit(record: dict) -> str:
    """Human-readable summary of the final-fit outcome (chat/log panel)."""
    method = str(record.get("method", "")).upper()
    if record.get("adopted"):
        return (
            f"**Final uncertainty fit ({method}):** adopted. "
            f"χ² {_fmt_chi2(record.get('chi2_before'))} → "
            f"{_fmt_chi2(record.get('chi2_after'))}, "
            f"{record.get('n_uncertainties', 0)} parameter uncertainties from the "
            f"{method} chain now reported."
        )
    return (
        f"**Final uncertainty fit ({method}):** not adopted — "
        f"χ² {_fmt_chi2(record.get('chi2_after'))} did not improve on the selected "
        f"χ² {_fmt_chi2(record.get('chi2_before'))}. Reporting the selected model."
    )
