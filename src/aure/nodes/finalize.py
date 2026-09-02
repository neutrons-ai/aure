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

1. **Profile-vetoed fits are set aside.** The SLD-profile check's excursion is
   invisible to χ² and *buys* χ², so a vetoed fit is routinely the run's
   lowest-scoring one; ranking on χ² alone reported exactly the model
   ``evaluation`` had refused to accept. A vetoed fit is reported only when it is
   the whole field, and then the message says so.
2. Lowest χ² wins.
3. **Parsimony tie-break.** Among iterations whose χ² is within
   ``FINAL_SELECTION_TOL`` (default 2%) of the lowest, prefer the one with the
   fewest free parameters, then the earliest iteration. This is the BIC idea —
   don't pay for complexity that buys nothing — without depending on the stored
   ``bic``, which is computed on inconsistent bases between the fitting and
   evaluation nodes for multi-state runs.

``best_model`` / ``best_chi2`` are deliberately **not** written here: they are
the loop's own regression baseline, and a later ``aure resume`` compares
against them.

The node has a second, reporting-only responsibility: it lists the run's
**untried improvement ideas**. A run now stops as soon as χ² meets the
acceptance threshold, so the ranked ``structural_hypotheses`` backlog is
normally not exhausted — the leftovers, plus the outcomes of everything that
*was* attempted, are emitted as their own message so the reader can see what to
try next (and what was already ruled out) without opening a checkpoint.
Statuses are reported exactly as they stand; this node never re-derives them.
The bucket labels and the pending selector are exported (``pending_hypotheses``,
``hypothesis_label``, ``format_attempted_counts``) because ``cli`` renders the
same backlog for the terminal report and for ``aure batch`` — one definition of
the buckets is the only thing that keeps those surfaces from contradicting
each other.

Idempotent: sets ``finalized`` and no-ops when it is already set, so the runner
can call it defensively on loop-exit paths that never route through a node.
"""

import copy
import logging
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from ..state import Message, ReflectivityState
from .evaluation import _count_free_params, _get_chi2_min

logger = logging.getLogger(__name__)

# Legacy fallback only. Fits judged before the verdict was persisted onto the
# FitResult carry the veto as prose and nothing else, so a checkpoint written by
# an older run can still be ranked correctly. The structured flag wins whenever
# it is present, in both directions — an evaluator that merely quotes this phrase
# back on a fit the detector cleared must not be treated as vetoed.
_ARTIFACT_MARKER = "non-physical sld-profile excursion"


def has_profile_artifact(fit: Any) -> bool:
    """True when the SLD-profile check vetoed this fit.

    The excursion is invisible to χ² — worse, it *buys* χ² — so a vetoed fit is
    routinely the run's lowest-scoring one and must not win on that alone.
    """
    if not isinstance(fit, dict):
        return False
    flag = fit.get("profile_artifact")
    if isinstance(flag, bool):
        return flag
    return any(_ARTIFACT_MARKER in str(i).lower() for i in fit.get("issues") or [])


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

#: How much worse a preferred tier's χ² may be before the tier order stops
#: applying, as a multiple of the best χ² across every scored fit. See
#: :func:`_get_tier_chi2_factor`.
_DEFAULT_TIER_CHI2_FACTOR = 3.0


def _get_selection_tolerance() -> float:
    """Relative χ² band within which the simpler model wins (``FINAL_SELECTION_TOL``)."""
    try:
        tol = float(os.environ.get("FINAL_SELECTION_TOL", str(_DEFAULT_TOL)))
    except (TypeError, ValueError):
        return _DEFAULT_TOL
    if not math.isfinite(tol) or tol < 0:
        return _DEFAULT_TOL
    return tol


def _get_tier_chi2_factor() -> float:
    """How far the tier order may override χ² (``FINAL_TIER_CHI2_FACTOR``).

    The tier order — profile-clean, then sub-floor, then vetoed — decides which
    fits are eligible at all, and it does so without looking at χ². That is
    right when the candidates are comparable and wrong when they are not: on the
    2026-08-17 cu_film sweep the artifact detector vetoed every candidate in 38
    of 51 runs, so a single surviving iteration won its tier outright no matter
    how badly it fitted. Case 201152 reported χ² = 130.55 because the one
    profile-clean iteration was that fit; the loop had already found χ² = 3.82.

    A model that misses the data by that margin is not the better answer for
    being drawn with clean interfaces — both are wrong, and the one that also
    contradicts the measurement is the worse report. So a preferred tier only
    keeps its precedence while its best χ² stays within this factor of the best
    χ² overall; past that the next tier is admitted alongside it and χ² decides.

    ``0`` or a non-finite value disables the guard, restoring the strict tier
    order for anyone who wants it.
    """
    try:
        factor = float(
            os.environ.get("FINAL_TIER_CHI2_FACTOR", str(_DEFAULT_TIER_CHI2_FACTOR))
        )
    except (TypeError, ValueError):
        return _DEFAULT_TIER_CHI2_FACTOR
    if not math.isfinite(factor) or factor < 0:
        return _DEFAULT_TIER_CHI2_FACTOR
    return factor


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
    floor = _get_chi2_min(state)

    # Rank within the best available tier. Reporting nothing is worse than
    # reporting a flawed model that says so, so each tier falls back to the next:
    #
    #   1. profile-clean and inside the acceptance window — the real answer;
    #   2. profile-clean but below the floor — physically plausible, but its χ² is
    #      evidence about the `dR` column rather than the structure, and an
    #      overfitted iteration is exactly the kind that scores lowest, so it must
    #      not beat an honest fit on χ² alone;
    #   3. profile-vetoed — physically impossible, so genuinely last.
    def _tier(fr: dict) -> int:
        if has_profile_artifact(fr):
            return 2
        if floor > 0 and fr["chi_squared"] < floor:
            return 1
        return 0

    tiers = {i: _tier(fr) for i, fr in scored}
    best_tier = min(tiers.values())
    pool = [(i, fr) for i, fr in scored if tiers[i] == best_tier]

    # ...but only while the preferred tier is still fitting the data. A tier
    # that survives on physical plausibility alone, at many times the χ² of a
    # fit it outranks, is not the better answer to report; see
    # _get_tier_chi2_factor.
    #
    # The comparison deliberately ignores sub-floor fits. Their χ² is a
    # statement about the `dR` column, not about the structure, so it is not a
    # yardstick anything else can be "far worse" than — measuring against one
    # would trip this guard on every run that produced an overfitted iteration,
    # which inverts the reason that tier exists.
    factor = _get_tier_chi2_factor()
    lowest_any = min(fr["chi_squared"] for _, fr in scored)
    tier_override = False
    if factor > 0 and best_tier < 2:
        comparable = [(i, fr) for i, fr in scored if tiers[i] != 1]
        if comparable:
            lowest_comparable = min(fr["chi_squared"] for _, fr in comparable)
            tier_best = min(fr["chi_squared"] for _, fr in pool)
            if lowest_comparable > 0 and tier_best > lowest_comparable * factor:
                tier_override = True
                pool = comparable

    vetoed = [
        {"index": i, "iteration": fr.get("iteration"), "chi_squared": fr["chi_squared"]}
        for i, fr in scored
        if tiers[i] == 2
    ]
    sub_floor = [
        {"index": i, "iteration": fr.get("iteration"), "chi_squared": fr["chi_squared"]}
        for i, fr in scored
        if tiers[i] == 1
    ]

    best_chi2 = min(fr["chi_squared"] for _, fr in pool)
    band = [(i, fr) for i, fr in pool if fr["chi_squared"] <= best_chi2 * (1.0 + tol)]

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

    # "Demoted" means a filter changed the answer, not merely that it matched: a
    # set-aside fit that would have lost on χ² anyway is not worth reporting as one.
    lowest_overall = lowest_any
    changed_answer = best_chi2 > lowest_overall
    selected_tier = tiers[index]

    return (
        index,
        fit,
        definition,
        {
            "criterion": (
                "lowest chi2 across every scored fit, parsimony tie-break — the "
                "tier order was set aside because the preferred tier fitted the "
                "data far worse"
                if tier_override
                else "lowest chi2 among profile-clean fits inside the acceptance "
                "window, parsimony tie-break"
            ),
            "tier_chi2_override": tier_override,
            "tier_chi2_factor": factor,
            "tolerance": tol,
            "parsimony_tiebreak": fit["chi_squared"] > best_chi2,
            "lowest_chi2": best_chi2,
            "candidates_considered": len(scored),
            "candidates_in_band": len(band),
            "candidates_profile_clean": sum(1 for t in tiers.values() if t != 2),
            "vetoed_iterations": vetoed,
            "demoted_for_profile_artifact": bool(vetoed) and changed_answer,
            "selected_has_profile_artifact": selected_tier == 2,
            "sub_floor_iterations": sub_floor,
            "demoted_for_sub_floor_chi2": bool(sub_floor) and changed_answer,
            "selected_is_sub_floor": selected_tier == 1,
            "chi2_min": floor,
        },
    )


# ======================================================================
# Node
# ======================================================================


def finalize_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Select the best fit of the run, promote it, and report the untried ideas.

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

    improvements = _format_remaining_improvements(state.get("structural_hypotheses"))
    if improvements and _already_in_transcript(state, improvements):
        # `aure resume` clears `finalized`, so this node runs again on a state
        # whose transcript already holds the previous run's report. Suppress
        # only a byte-identical repeat — a backlog that moved since then is news.
        improvements = ""

    fit_results = state.get("fit_results") or []
    scored = _scored_fits(fit_results)
    if not scored:
        logger.info("[FINALIZE] No usable fit results — nothing to select")
        updates["final_selection"] = {
            "selected": False,
            "reason": "no fit results with a finite, positive chi-squared",
        }
        if improvements:
            updates["messages"] = [
                Message(role="assistant", content=improvements, timestamp=None)
            ]
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

    messages = [
        Message(
            role="assistant",
            content=_format_selection(updates["final_selection"]),
            timestamp=None,
        )
    ]
    if improvements:
        messages.append(Message(role="assistant", content=improvements, timestamp=None))
    updates["messages"] = messages
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

    lines = []
    if sel.get("selected_is_sub_floor"):
        lines.append(
            f"**The reported χ² is below the acceptance floor** "
            f"(χ² ≥ {sel.get('chi2_min', 0):.2f}), and no iteration landed inside "
            f"the window. Residuals that far under the quoted uncertainties are "
            f"evidence the `dR` column is overestimated, or that the model has "
            f"enough free parameters to absorb the noise — not that the structure "
            f"is right. Check the uncertainties and the parameter count."
        )
    if sel.get("selected_has_profile_artifact"):
        lines.append(
            "**The reported model is not physically valid.** Every iteration of "
            "this run was vetoed by the SLD-profile check, so there was no "
            "artifact-free alternative to report. Do not report these slabs as "
            "distinct layers — constrain the offending interface roughness and "
            "re-run, or treat the stack as a profile parametrization."
        )
    if sel.get("tier_chi2_override"):
        lines.append(
            f"**A physically cleaner fit was available and was not reported.** It "
            f"missed the data by more than {sel.get('tier_chi2_factor', 0):g}× the "
            f"best χ² of the run, so it was ranked on χ² alongside the rest rather "
            f"than promoted on plausibility alone: a model that contradicts the "
            f"measurement is not the safer answer for having clean interfaces. "
            f"Both problems are real — treat this run as unresolved."
        )
    lines.append(
        f"**Final model:** iteration {sel['iteration']} "
        f"(χ² = {sel['chi_squared']:.4f}, {sel['n_free_params']} free parameters)."
    )
    if sel.get("demoted_for_profile_artifact"):
        vetoed = ", ".join(
            f"iteration {v.get('iteration')} (χ² = {_fmt_chi2(v.get('chi_squared'))})"
            for v in sel.get("vetoed_iterations") or []
        )
        lines.append(
            f"**Demoted by the SLD-profile check:** {vetoed} fitted better but was "
            f"not reported — the profile check vetoed it as non-physical, a defect "
            f"χ² cannot see."
        )
    if sel.get("demoted_for_sub_floor_chi2"):
        under = ", ".join(
            f"iteration {v.get('iteration')} (χ² = {_fmt_chi2(v.get('chi_squared'))})"
            for v in sel.get("sub_floor_iterations") or []
        )
        lines.append(
            f"**Set aside as below the acceptance floor:** {under} scored lower but "
            f"was not reported — under χ² ≥ {sel.get('chi2_min', 0):.2f} the number "
            f"describes the error bars rather than the structure."
        )
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


# ======================================================================
# Untried-improvements report
# ======================================================================


def _already_in_transcript(state: ReflectivityState, content: str) -> bool:
    """Whether *content* was already said verbatim earlier in the run."""
    return any(
        str(m.get("content", "")) == content for m in state.get("messages") or []
    )


# Every status other than "pending" is an attempted outcome
# (``hypotheses.ALLOWED_STATUSES``). "tried" is not a failure: it means the
# change was realized but neither confirmed nor reverted, because the χ² verdict
# was ambiguous — ``evaluation`` records an outcome on both the refining and the
# accepting branch, so "tried" no longer just means "the run ended before the
# bookkeeping ran".
_ATTEMPTED_LABELS = (
    ("confirmed", "confirmed"),
    ("rejected", "rejected"),
    ("tried", "tried, inconclusive"),
)


def pending_hypotheses(hypotheses: Optional[List[dict]]) -> List[dict]:
    """The untried backlog, in rank order (rank is list position)."""
    return [h for h in hypotheses or [] if h.get("status") == "pending"]


def hypothesis_title(hypothesis: dict) -> str:
    """One spelling of a backlog entry's title, for every surface that shows it."""
    return str(hypothesis.get("title") or "untitled")


def hypothesis_label(hypothesis: dict) -> str:
    """``"[3] Split Cu into two slabs"`` — id + title, for terminal listings.

    A hypothesis scoped to a subset of states carries the scope, because
    "add an oxide" and "add an oxide in the air state only" are different
    claims and the reader of the report cannot tell them apart from the title.
    """
    scope = hypothesis.get("states") or []
    suffix = f" (states: {', '.join(str(n) for n in scope)})" if scope else ""
    return f"[{hypothesis.get('id', '?')}] {hypothesis_title(hypothesis)}{suffix}"


def _attempted_groups(hypotheses: Optional[List[dict]]) -> List[Tuple[str, List[dict]]]:
    """Non-empty attempted buckets as ``(label, entries)``, in report order."""
    groups = []
    for status, label in _ATTEMPTED_LABELS:
        group = [h for h in hypotheses or [] if h.get("status") == status]
        if group:
            groups.append((label, group))
    return groups


def format_attempted_counts(hypotheses: Optional[List[dict]]) -> str:
    """One-line tally of the backlog, e.g.
    ``"2 of 5 attempted — confirmed (1); tried, inconclusive (1)"``.

    Shared with the CLI report so the two renderings of one run's backlog cannot
    disagree. The attempted total is the sum of the buckets below it rather than
    "everything not pending", so the arithmetic on the line always closes even
    for an entry whose status is missing or unrecognized.

    Returns "" when there is no backlog at all.
    """
    if not hypotheses:
        return ""
    groups = _attempted_groups(hypotheses)
    attempted = sum(len(g) for _, g in groups)
    line = f"{attempted} of {len(hypotheses)} attempted"
    if groups:
        line += " — " + "; ".join(f"{label} ({len(g)})" for label, g in groups)
    return line


def _format_remaining_improvements(hypotheses: Optional[List[dict]]) -> str:
    """Human-readable state of the structural-hypothesis backlog at the end.

    The ``pending`` entries are the ideas the refinement loop never got to, in
    rank order (which is list position). The attempted ones are summarized in a
    single line so a reader does not re-propose something the run already ruled
    out. Statuses are reported as they stand — nothing is re-derived here.

    Returns "" when there is nothing worth saying.
    """
    if not hypotheses:
        return ""

    lines: List[str] = []
    pending = pending_hypotheses(hypotheses)
    if pending:
        lines.append("**Possible further improvements (not tried):**")
        for h in pending:
            lines.append(
                f"{h.get('id', '?')}. **{hypothesis_title(h)}** — "
                f"{h.get('change') or 'no concrete change recorded'}"
            )
            if h.get("rationale"):
                source = h.get("skill_source") or h.get("origin") or "?"
                lines.append(f"   _Rationale ({source}):_ {h['rationale']}")

    attempted = []
    for label, group in _attempted_groups(hypotheses):
        # An attempted entry is named without its id, so an untitled one falls
        # back to the id rather than to a row of indistinguishable "untitled"s.
        titles = ", ".join(str(h.get("title") or f"#{h.get('id', '?')}") for h in group)
        attempted.append(f"{label} ({len(group)}): {titles}")
    if attempted:
        # Blank line, or markdown absorbs this into the numbered list above.
        if lines:
            lines.append("")
        lines.append("**Already attempted:** " + "; ".join(attempted) + ".")

    return "\n".join(lines)
