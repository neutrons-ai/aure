"""
Terminal packaging node: pick the best fit the run found and make it the answer.

The refinement loop is a search, and searches wander — the last iteration to
run is not necessarily the best one. Before this node existed, a run that
stopped because it exhausted ``max_iterations`` reported whatever model
happened to be current when the loop broke, while ``best_model`` / ``best_chi2``
were tracked but never consumed at the end.

This node runs exactly once, after the loop has stopped for *any* reason, and
makes the choice explicit:

* selects the winning entry of ``fit_results`` (see below),
* promotes that iteration's ``ModelDefinition`` — with its fitted parameter
  values written in — to ``current_model``,
* sets ``current_chi2`` to the winning χ², so every existing consumer of the
  "current" keys (CLI summary, ``final_state.json``'s ``final_chi2``, the ISAAC
  exporter, the web UI) reports the winner instead of the last try,
* records the decision in ``final_selection`` so it is auditable from a
  checkpoint rather than being an invisible reshuffle.

Selection rule (deterministic — no LLM):

1. Lowest χ² wins.
2. **Parsimony tie-break.** Among iterations whose χ² is within
   ``FINAL_SELECTION_TOL`` (default 2%) of the lowest, prefer the one with the
   fewest free parameters, then the earliest iteration. This is the BIC idea —
   don't pay for complexity that buys nothing — without depending on the stored
   ``bic``, which is computed on inconsistent bases between the fitting and
   evaluation nodes for multi-state runs.

``best_model`` / ``best_chi2`` are deliberately **not** written here: they are
the loop's own regression baseline, and a later ``aure resume`` compares
against them.

Idempotent: sets ``finalized`` and no-ops when it is already set, so the runner
can call it defensively on loop-exit paths that never route through a node.
"""

import copy
import logging
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from ..state import Message, ReflectivityState
from .evaluation import _count_free_params

logger = logging.getLogger(__name__)

# refl1d names a slab's fitted parameters "<material> <suffix>"; the matching
# ModelDefinition keys are spelled differently. ``rough_frac`` is the tied
# variant (σ = fraction × thickness) installed by ``roughness_tie`` — the
# interface is then a derived expression, so the fitted quantity is the
# fraction and the definition records the resulting numeric roughness.
_LAYER_SUFFIX_TO_KEY = {
    "thickness": "thickness",
    "rho": "sld",
    "interface": "roughness",
}

_DEFAULT_TOL = 0.02


def _get_selection_tolerance() -> float:
    """Relative χ² band within which the simpler model wins (``FINAL_SELECTION_TOL``)."""
    try:
        tol = float(os.environ.get("FINAL_SELECTION_TOL", str(_DEFAULT_TOL)))
    except (TypeError, ValueError):
        return _DEFAULT_TOL
    if not math.isfinite(tol) or tol < 0:
        return _DEFAULT_TOL
    return tol


# ======================================================================
# Fitted-value write-back
# ======================================================================


def _apply_layers(
    layers: Optional[List[dict]], fitted: dict, prefix: str, consumed: Set[str]
) -> None:
    for layer in layers or []:
        name = layer.get("name")
        if not name:
            continue
        for suffix, key in _LAYER_SUFFIX_TO_KEY.items():
            pname = f"{prefix}{name} {suffix}"
            if pname in fitted:
                layer[key] = fitted[pname]
                consumed.add(pname)
        # Tied roughness: the free parameter is the fraction, so convert it
        # back to a numeric σ exactly as model_builder.extract_definition does.
        # Only when `interface` itself wasn't fitted (they are mutually
        # exclusive — build_sample installs one or the other).
        frac_name = f"{prefix}{name} rough_frac"
        if frac_name in fitted and f"{prefix}{name} interface" not in fitted:
            thickness = layer.get("thickness")
            if isinstance(thickness, (int, float)):
                layer["roughness"] = fitted[frac_name] * thickness
                consumed.add(frac_name)


def _apply_medium(
    medium: Optional[dict],
    fitted: dict,
    prefix: str,
    consumed: Set[str],
    *,
    accepts_interface: bool,
) -> None:
    """Write a bounding medium's fitted SLD (and roughness, where it has a home).

    ``accepts_interface`` encodes an asymmetry in ``_build_sample``: the field
    that seeds the outermost interface depends on the stack orientation. In
    front reflection ``sample[0].interface`` comes from
    ``substrate["roughness"]``, so ``"<substrate> interface"`` round-trips. In
    back reflection ``sample[0]`` is the *ambient* and its interface is seeded
    from ``layers[-1]["roughness"]`` — a field already claimed by
    ``"<last layer> interface"`` — so there is no distinct home for
    ``"<ambient> interface"``. Writing it to ``ambient["roughness"]`` would be
    silently discarded on the next rebuild, so we leave it unconsumed instead
    and let it surface in ``values_unapplied``.
    """
    name = (medium or {}).get("name")
    if not name:
        return
    rho_name = f"{prefix}{name} rho"
    if rho_name in fitted:
        medium["sld"] = fitted[rho_name]
        consumed.add(rho_name)
    if accepts_interface:
        iface_name = f"{prefix}{name} interface"
        if iface_name in fitted:
            medium["roughness"] = fitted[iface_name]
            consumed.add(iface_name)


def _intensity_matches(
    fitted: dict, prefix: str, exclude_prefixes: Tuple[str, ...] = ()
) -> List[str]:
    """Fitted beam-intensity parameter names belonging to one scope.

    refl1d never names this parameter bare ``"intensity"``: the builders spell
    it ``"intensity <probe label>"`` (single- and multi-file) and
    ``"<state> intensity"`` / ``"<state> <label> intensity"`` for the states
    path. Match on the token within this scope's prefix rather than on an exact
    key, with ``exclude_prefixes`` keeping the model-level scope from claiming a
    state's intensity (the empty prefix matches everything).
    """
    matches = []
    for name in fitted:
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix) :]
        if any(tail.startswith(ex) for ex in exclude_prefixes):
            continue
        if "intensity" in tail.split():
            matches.append(name)
    return matches


def _apply_intensity(
    block: Optional[dict],
    fitted: dict,
    prefix: str,
    consumed: Set[str],
    exclude_prefixes: Tuple[str, ...] = (),
) -> None:
    """Write the fitted intensity into *block*, when it is unambiguous.

    When several probes each contributed their own intensity there is no single
    definition field to write, so leave them unconsumed and visible.
    """
    if not isinstance(block, dict):
        return
    matches = _intensity_matches(fitted, prefix, exclude_prefixes)
    if len(matches) == 1:
        block["value"] = fitted[matches[0]]
        consumed.add(matches[0])


def _has_medium_params(medium: Optional[dict], fitted: dict, prefix: str) -> bool:
    name = (medium or {}).get("name")
    if not name:
        return False
    return any(f"{prefix}{name} {suffix}" in fitted for suffix in ("rho", "interface"))


def _has_state_prefixed_layer_params(
    layers: Optional[List[dict]], fitted: dict, prefix: str
) -> bool:
    for layer in layers or []:
        name = layer.get("name")
        if not name:
            continue
        for suffix in list(_LAYER_SUFFIX_TO_KEY) + ["rough_frac"]:
            if f"{prefix}{name} {suffix}" in fitted:
                return True
    return False


def _apply_fitted_values(definition: dict, fitted: dict) -> Tuple[int, List[str]]:
    """Write *fitted* parameter values into *definition* in place.

    Handles both the single-state shape (unprefixed ``"<material> <suffix>"``
    names) and the multi-state shape, where ``build_states_problem`` prefixes
    every *untied* parameter with ``"<state> "``.

    A state that inherits the model-level ``layers`` has no per-state structure
    to write untied values into, so when state-prefixed layer parameters exist
    the inherited layers are materialized onto that state first. Tying is
    driven by ``shared_parameters`` / ``unshared_parameters`` and the rename
    loop reads the *model-level* ``layers``, never the per-state ones, so this
    does not change how a re-fit ties parameters.

    Returns ``(n_applied, unconsumed_names)``. Anything unconsumed is reported
    rather than force-fitted into a field the builder does not read back.
    """
    consumed: Set[str] = set()
    back = bool(definition.get("back_reflection"))
    states = definition.get("states") or []
    state_prefixes = tuple(
        f"{st['name']} " for st in states if isinstance(st, dict) and st.get("name")
    )

    _apply_layers(definition.get("layers"), fitted, "", consumed)
    _apply_medium(
        definition.get("substrate"),
        fitted,
        "",
        consumed,
        accepts_interface=not back,
    )
    _apply_medium(
        definition.get("ambient"), fitted, "", consumed, accepts_interface=False
    )
    _apply_intensity(definition.get("intensity"), fitted, "", consumed, state_prefixes)

    for st in states:
        st_name = st.get("name")
        if not st_name:
            continue
        prefix = f"{st_name} "
        st_back = bool(st.get("back_reflection", back))
        # A state that inherits a model-level block has nowhere to put an
        # untied fitted value, so materialize the block first. The builders key
        # parameter names off the *model-level* structure, and the copy keeps
        # the same names, so a subsequent re-fit builds an identical problem.
        if not st.get("layers") and _has_state_prefixed_layer_params(
            definition.get("layers"), fitted, prefix
        ):
            st["layers"] = copy.deepcopy(definition.get("layers") or [])
        if not st.get("substrate") and _has_medium_params(
            definition.get("substrate"), fitted, prefix
        ):
            st["substrate"] = copy.deepcopy(definition.get("substrate") or {})
        if not st.get("intensity") and _intensity_matches(fitted, prefix):
            st["intensity"] = copy.deepcopy(definition.get("intensity") or {})
        _apply_layers(st.get("layers"), fitted, prefix, consumed)
        _apply_medium(
            st.get("substrate"),
            fitted,
            prefix,
            consumed,
            accepts_interface=not st_back,
        )
        _apply_medium(
            st.get("ambient"), fitted, prefix, consumed, accepts_interface=False
        )
        _apply_intensity(st.get("intensity"), fitted, prefix, consumed)

    unconsumed = sorted(n for n in fitted if n not in consumed)
    return len(consumed), unconsumed


# ======================================================================
# Selection
# ======================================================================


def _scored_fits(fit_results: List[dict]) -> List[Tuple[int, dict]]:
    """Indexed fit results that carry a usable χ²."""
    scored = []
    for i, fr in enumerate(fit_results):
        chi2 = fr.get("chi_squared")
        if isinstance(chi2, bool) or not isinstance(chi2, (int, float)):
            continue
        if math.isfinite(chi2) and chi2 > 0:
            scored.append((i, fr))
    return scored


def _definition_for_iteration(
    state: ReflectivityState, iteration: Optional[int], is_best_chi2: bool
) -> Any:
    """The ModelDefinition that produced a given fit iteration.

    ``model_history`` is the authoritative per-iteration record, but it is
    append-only and iteration numbers are not guaranteed unique:

    * the interactive rewind (``restart_checkpoint``) replays an iteration, so
      the *newest* entry for a number is the live one — take the last match,
      not the first, or a discarded branch wins;
    * the bounds-only re-fit route (``route_after_evaluation`` -> ``"fitting"``)
      and a ``restart_from="fitting"`` re-fit both produce a ``fit_results``
      entry with *no* ``model_history`` entry, because only ``modeling`` writes
      history. Those re-fit the previous structure, so resolve to the nearest
      *preceding* entry rather than silently promoting whatever is current.

    Falling back to ``best_model`` / ``current_model`` is the last resort, for
    imported or legacy states that carry no history at all.
    """
    best_preceding: Any = None
    best_iter: Optional[int] = None
    for entry in state.get("model_history") or []:
        defn = entry.get("definition")
        if not (isinstance(defn, dict) and defn):
            defn = entry.get("script") or None
        if not defn:
            continue
        entry_iter = entry.get("iteration")
        if entry_iter == iteration:
            # Keep scanning: a later entry with the same number supersedes.
            best_preceding, best_iter = defn, entry_iter
        elif (
            isinstance(entry_iter, int)
            and isinstance(iteration, int)
            and entry_iter < iteration
            and (best_iter is None or entry_iter >= best_iter)
        ):
            best_preceding, best_iter = defn, entry_iter
    if best_preceding is not None:
        return best_preceding
    if is_best_chi2 and state.get("best_model"):
        return state["best_model"]
    return state.get("current_model")


def _free_params(definition: Any, fit: dict) -> int:
    """Free-parameter count for a fit, preferring the value bumps reported."""
    n = fit.get("_n_free_params")
    if isinstance(n, int) and n > 0:
        return n
    if isinstance(definition, dict):
        try:
            return _count_free_params(definition)
        except Exception:  # pragma: no cover - defensive
            pass
    return 0


def _select(
    state: ReflectivityState, scored: List[Tuple[int, dict]]
) -> Tuple[int, dict, Any, dict]:
    """Pick the winning fit.

    Returns ``(index, fit, definition, decision_metadata)``. The definition is
    resolved here — not later — so the parsimony ranking counts parameters on
    exactly the model that will be promoted.
    """
    tol = _get_selection_tolerance()
    best_chi2 = min(fr["chi_squared"] for _, fr in scored)
    band = [(i, fr) for i, fr in scored if fr["chi_squared"] <= best_chi2 * (1.0 + tol)]

    resolved = [
        (
            i,
            fr,
            _definition_for_iteration(
                state, fr.get("iteration"), fr["chi_squared"] <= best_chi2
            ),
        )
        for i, fr in band
    ]
    ranked = sorted(
        resolved,
        key=lambda t: (_free_params(t[2], t[1]), t[1].get("iteration", t[0])),
    )
    index, fit, definition = ranked[0]

    return (
        index,
        fit,
        definition,
        {
            "criterion": "lowest chi2, parsimony tie-break",
            "tolerance": tol,
            "parsimony_tiebreak": fit["chi_squared"] > best_chi2,
            "lowest_chi2": best_chi2,
            "candidates_considered": len(scored),
            "candidates_in_band": len(band),
        },
    )


# ======================================================================
# Node
# ======================================================================


def finalize_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Select the best fit of the run and promote it to the current model.

    Args:
        state: Current workflow state

    Returns:
        State updates. Empty when already finalized (idempotent).
    """
    if state.get("finalized"):
        logger.debug("[FINALIZE] Already finalized — nothing to do")
        return {}

    updates: Dict[str, Any] = {
        "current_node": "finalize",
        "messages": [],
        "finalized": True,
    }

    fit_results = state.get("fit_results") or []
    scored = _scored_fits(fit_results)
    if not scored:
        logger.info("[FINALIZE] No usable fit results — nothing to select")
        updates["final_selection"] = {
            "selected": False,
            "reason": "no fit results with a finite, positive chi-squared",
        }
        return updates

    index, fit, definition, decision = _select(state, scored)
    iteration = fit.get("iteration")
    chi2 = fit["chi_squared"]

    # "The last iteration" means the last one actually fitted, whether or not
    # its χ² was usable — a diverged final fit is exactly the case a user needs
    # told about.
    last_chi2 = fit_results[-1].get("chi_squared")
    superseded = index != len(fit_results) - 1

    n_applied = 0
    unapplied: List[str] = []
    if isinstance(definition, dict):
        final_model = copy.deepcopy(definition)
        n_applied, unapplied = _apply_fitted_values(
            final_model, fit.get("parameters") or {}
        )
    else:
        # Legacy script-string model — nothing to write values into.
        final_model = definition

    if final_model:
        updates["current_model"] = final_model
    updates["current_chi2"] = chi2

    updates["final_selection"] = {
        "selected": True,
        "index": index,
        "iteration": iteration,
        "chi_squared": chi2,
        "bic": fit.get("bic"),
        "method": fit.get("method"),
        "n_free_params": _free_params(final_model, fit),
        "superseded_last_iteration": superseded,
        "last_iteration_chi2": last_chi2,
        "values_applied": n_applied,
        "values_unapplied": unapplied[:20],
        **decision,
    }

    if superseded:
        logger.info(
            "[FINALIZE] Selected iteration %s (χ²=%.3f) over the last iteration "
            "fitted (χ²=%s)",
            iteration,
            chi2,
            _fmt_chi2(last_chi2),
        )
    else:
        logger.info(
            "[FINALIZE] Last iteration %s (χ²=%.3f) is also the best — keeping it",
            iteration,
            chi2,
        )
    if unapplied:
        logger.info(
            "[FINALIZE] %d fitted parameter(s) had no ModelDefinition field: %s",
            len(unapplied),
            ", ".join(unapplied[:8]),
        )

    updates["messages"] = [
        Message(
            role="assistant",
            content=_format_selection(updates["final_selection"]),
            timestamp=None,
        )
    ]
    return updates


def _fmt_chi2(value: Any) -> str:
    """Format a χ² that may be None, inf or nan."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return f"{value:.4f}"
        return "diverged"
    return "n/a"


def _format_selection(sel: dict) -> str:
    """Human-readable summary of the final selection."""
    if not sel.get("selected"):
        return f"**Final model:** not selected — {sel.get('reason', 'unknown')}."

    lines = [
        f"**Final model:** iteration {sel['iteration']} "
        f"(χ² = {sel['chi_squared']:.4f}, {sel['n_free_params']} free parameters)."
    ]
    if sel.get("superseded_last_iteration"):
        lines.append(
            f"This supersedes the last iteration fitted "
            f"(χ² = {_fmt_chi2(sel.get('last_iteration_chi2'))}) — the refinement "
            f"loop's final attempt was not its best."
        )
    if sel.get("parsimony_tiebreak"):
        lines.append(
            f"Chosen over the lowest-χ² fit (χ² = {sel['lowest_chi2']:.4f}) "
            f"because it fits equally well within "
            f"{sel['tolerance'] * 100:.0f}% with fewer free parameters."
        )
    if sel.get("values_unapplied"):
        lines.append(
            "Fitted values with no model field (left as-is): "
            + ", ".join(sel["values_unapplied"][:8])
            + "."
        )
    return " ".join(lines)
