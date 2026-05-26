"""Import a refl1d ``problem.json`` into an AuRE workflow output directory.

The public entry point is :func:`import_refl1d`, used by the
``aure import-refl1d`` CLI. It inverts a serialised bumps ``FitProblem``
back into AuRE's :class:`~aure.state.ModelDefinition` / :class:`~aure.state.ReflectivityState`
shape and writes the directory layout the web UI and ``aure resume``
expect:

    <output_dir>/
    ├── run_info.json
    ├── final_state.json
    ├── problem.json                    (copy of the source)
    ├── data/                           (data files extracted from probes)
    ├── checkpoints/
    │   ├── 001_intake.json
    │   ├── 002_analysis.json
    │   ├── 003_modeling.json
    │   ├── 004_fitting.json
    │   └── 005_evaluation.json
    └── refl1d_output/
        └── fit_iter0_<method>/
            └── (problem.json + companion files)

The two key helpers are also reused by other commands:

- :func:`extract_fit_result_from_problem` — used by ``aure evaluate``
  to ingest a refl1d output dir without running the workflow.
- :func:`definition_from_problem` — recovers a ``ModelDefinition`` (with
  ``states`` when the problem is multi-experiment).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .nodes.evaluation import _compute_bic, _count_free_params
from .state import (
    AmbientInfo,
    DatasetInfo,
    FitResult,
    LayerInfo,
    Message,
    ModelDefinition,
    ParsedSample,
    PerFileFitResult,
    ReflectivityState,
    StateDefinition,
    SubstrateInfo,
    create_initial_state,
    flatten_data_files,
)
from .workflow.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Substrate / ambient classification heuristics
# --------------------------------------------------------------------------

# Names treated as substrates when guessing back_reflection orientation.
_SUBSTRATE_NAMES = {
    "silicon", "si", "sapphire", "al2o3", "alumina",
    "quartz", "sio2", "glass", "germanium", "ge", "mica",
    "gold", "au",   # gold can be a substrate in some setups
}

# Names treated as ambient/fronting media.
_AMBIENT_NAMES = {
    "air", "vacuum", "d2o", "h2o", "water", "h2o-air", "d2o-air",
    "ethanol", "methanol", "acetone", "dthf", "hthf", "cmsi", "cmsi4",
    "cd3od", "cdcl3", "thf", "toluene", "buffer",
}


def _classify_orientation(stack_materials: list[str]) -> bool:
    """Guess ``back_reflection`` from a sample stack's material names.

    Returns True if the *last* slab looks more substrate-like than the
    first. Falls back to False when neither end is recognised — the
    caller can override via ``--back-reflection``.
    """
    if not stack_materials:
        return False
    first = stack_materials[0].lower()
    last = stack_materials[-1].lower()
    first_sub = first in _SUBSTRATE_NAMES
    last_sub = last in _SUBSTRATE_NAMES
    if last_sub and not first_sub:
        return True
    if first_sub and not last_sub:
        return False
    # Both or neither look substrate-like — try ambient detection.
    first_amb = first in _AMBIENT_NAMES
    last_amb = last in _AMBIENT_NAMES
    if first_amb and not last_amb:
        return True
    if last_amb and not first_amb:
        return False
    return False


# --------------------------------------------------------------------------
# State kind heuristic (combined vs partials), borrowed from config
# --------------------------------------------------------------------------

_PARTIAL_RE = re.compile(r"_(\d+)_(\d+)_partial\.txt$", re.IGNORECASE)
_COMBINED_RE = re.compile(r"_combined_data_auto\.txt$", re.IGNORECASE)
_PARTIAL_SETID_RE = re.compile(r"REFL_(\d+)_\d+_\d+_partial\.txt$", re.IGNORECASE)
_COMBINED_SETID_RE = re.compile(r"REFL_(\d+)_combined_data_auto\.txt$", re.IGNORECASE)


def _detect_state_kind(file_paths: list[str]) -> str:
    """Filename-based ``combined`` vs ``partials`` classifier for a state."""
    combined, partials = 0, 0
    for p in file_paths:
        name = os.path.basename(p)
        if _PARTIAL_RE.search(name):
            partials += 1
        else:
            combined += 1
    return "partials" if partials and not combined else "combined"


def _extract_set_id(file_path: str) -> Optional[str]:
    """Return the REF_L set_id encoded in *file_path* or None."""
    name = os.path.basename(file_path)
    for pattern in (_COMBINED_SETID_RE, _PARTIAL_SETID_RE):
        m = pattern.search(name)
        if m:
            return m.group(1)
    return None


# --------------------------------------------------------------------------
# Probe → data file writer
# --------------------------------------------------------------------------


def _write_probe_to_dat(probe, out_path: Path) -> None:
    """Dump a deserialised probe back to the 4-column ``Q R dR dQ`` format.

    Used so the imported workflow output is self-contained — subsequent
    calls to :func:`aure.nodes.model_builder.load_probe` use these files
    rather than relying on the user keeping the original around.

    Handles both ``QProbe`` (Q-based) and ``NeutronProbe`` (angle-based)
    by deriving Q from ``T`` and ``L`` in the latter case.
    """
    Q = np.asarray(getattr(probe, "Q", []), dtype=float)
    R = np.asarray(getattr(probe, "R", []), dtype=float)
    dR = getattr(probe, "dR", None)
    if dR is None:
        dR = np.zeros_like(R)
    else:
        dR = np.asarray(dR, dtype=float)
    dQ = getattr(probe, "dQ", None)
    if dQ is None:
        # No resolution: write zeros so the column structure is preserved.
        dQ = np.zeros_like(Q)
    else:
        dQ = np.asarray(dQ, dtype=float)

    if len(Q) == 0:
        raise ValueError(
            f"Cannot export probe to {out_path}: probe carries no Q values."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("# Q (1/A)  R  dR  dQ (FWHM, 1/A)\n")
        for q, r, dr, dq in zip(Q, R, dR, dQ):
            fh.write(f"{q:.8e}  {r:.8e}  {dr:.8e}  {dq:.8e}\n")


# --------------------------------------------------------------------------
# Inversion: refl1d objects → ModelDefinition
# --------------------------------------------------------------------------


def _slab_to_layer(slab) -> LayerInfo:
    """Build a :class:`LayerInfo` from a refl1d ``Slab``."""
    thickness = float(slab.thickness.value)
    sld = float(slab.material.rho.value)
    roughness = float(slab.interface.value)

    t_bounds = _bounds(slab.thickness)
    sld_bounds = _bounds(slab.material.rho)
    r_bounds = _bounds(slab.interface)

    layer: LayerInfo = {
        "name": str(slab.material.name),
        "sld": sld,
        "sld_min": sld_bounds[0],
        "sld_max": sld_bounds[1],
        "thickness": thickness,
        "thickness_min": t_bounds[0],
        "thickness_max": t_bounds[1],
        "roughness": roughness,
        "roughness_max": r_bounds[1],
    }
    return layer


def _slab_to_substrate(slab) -> SubstrateInfo:
    return {
        "name": str(slab.material.name),
        "sld": float(slab.material.rho.value),
        "roughness": float(slab.interface.value),
        "roughness_max": _bounds(slab.interface)[1],
    }


def _slab_to_ambient(slab) -> AmbientInfo:
    return {
        "name": str(slab.material.name),
        "sld": float(slab.material.rho.value),
    }


def _bounds(parameter) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(low, high)`` bounds for a bumps Parameter, or (None, None)."""
    b = getattr(parameter, "bounds", None)
    if b is None:
        return (None, None)
    try:
        lo, hi = b.limits  # bumps Bounds-like objects expose .limits
    except AttributeError:
        try:
            lo, hi = b
        except Exception:
            return (None, None)
    try:
        lo_f = float(lo)
    except (TypeError, ValueError):
        lo_f = None
    try:
        hi_f = float(hi)
    except (TypeError, ValueError):
        hi_f = None
    return (lo_f, hi_f)


def _structure_from_sample(sample, *, back_reflection: bool) -> Tuple[
    SubstrateInfo, List[LayerInfo], AmbientInfo
]:
    """Pull substrate/layers/ambient out of a refl1d ``Sample`` stack.

    Stack ordering (see ``_build_sample``):

    - normal:        ``[substrate, L1, L2, ..., Ln, ambient]``
    - back_reflection: ``[ambient, Ln, ..., L1, substrate]``
    """
    slabs = list(sample)
    if len(slabs) < 2:
        raise ValueError(
            f"Sample stack has only {len(slabs)} slab(s); need at least substrate "
            "and ambient."
        )

    if back_reflection:
        ambient_slab = slabs[0]
        substrate_slab = slabs[-1]
        layer_slabs = list(reversed(slabs[1:-1]))
    else:
        substrate_slab = slabs[0]
        ambient_slab = slabs[-1]
        layer_slabs = slabs[1:-1]

    return (
        _slab_to_substrate(substrate_slab),
        [_slab_to_layer(s) for s in layer_slabs],
        _slab_to_ambient(ambient_slab),
    )


def _probe_intensity(probe) -> dict:
    """Recover intensity normalisation settings from a probe."""
    intensity = getattr(probe, "intensity", None)
    if intensity is None:
        return {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False}
    lo, hi = _bounds(intensity)
    fixed = lo is None and hi is None
    return {
        "value": float(intensity.value),
        "min": float(lo) if lo is not None else 0.7,
        "max": float(hi) if hi is not None else 1.1,
        "fixed": fixed,
    }


def _group_experiments_by_sample(experiments: list) -> List[List[int]]:
    """Return groups of experiment indices that share a refl1d ``Sample``.

    Uses Python ``id()`` because refl1d's multi-file co-refinement shares
    one ``Sample`` object across experiments. Preserves first-seen order.
    """
    groups: "OrderedDict[int, list[int]]" = OrderedDict()
    for idx, exp in enumerate(experiments):
        key = id(exp.sample)
        groups.setdefault(key, []).append(idx)
    return list(groups.values())


def _recover_tied_set(
    samples: list,
    *,
    back_reflection: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Identify which structural parameters are aliased across samples.

    Returns ``(shared, unshared, all_default_pairs)``:

    - ``shared`` / ``unshared`` follow the :class:`ModelDefinition`
      convention — exactly one is non-empty (whichever is shorter),
      the other is ``[]``. When the default tied set matches exactly
      both are empty so the consumer falls back to the built-in defaults.
    - ``all_default_pairs`` is the full list of ``"<layer>.<attr>"``
      candidates the importer inspected. Callers use it to decide
      whether *zero* defaults were recovered as tied (a strong signal
      that the user expressed ties via bumps constraint expressions
      rather than object sharing — which the importer cannot recover).

    The candidate set mirrors :data:`aure.nodes.model_builder._DEFAULT_TIED_LAYER_ATTRS`
    plus the substrate's ``interface``. The substrate's index in the
    stack depends on ``back_reflection`` — the ambient end is *not* a
    default tied attribute, so it's excluded.

    Cross-state aliasing is detected by Python object identity on the
    ``bumps.Parameter`` instances — which is how
    :func:`~aure.nodes.model_builder.build_states_problem` produces them.
    """
    if len(samples) < 2:
        return ([], [], [])

    from .nodes.model_builder import (
        _DEFAULT_TIED_LAYER_ATTRS,
        _DEFAULT_TIED_SUBSTRATE_ATTRS,
        _get_layer_param,
    )

    first = list(samples[0])
    n = len(first)
    substrate_idx = n - 1 if back_reflection else 0
    ambient_idx = 0 if back_reflection else n - 1

    candidates: list[tuple[int, str, str]] = []  # (index, layer_name, attr)
    for idx, slab in enumerate(first):
        if idx == ambient_idx:
            continue  # ambient is per-state by design — not in default tied set
        layer_name = str(slab.material.name)
        if idx == substrate_idx:
            for attr in _DEFAULT_TIED_SUBSTRATE_ATTRS:
                candidates.append((idx, layer_name, attr))
        else:
            for attr in _DEFAULT_TIED_LAYER_ATTRS:
                candidates.append((idx, layer_name, attr))

    tied: list[str] = []
    untied: list[str] = []
    for idx, name, attr in candidates:
        try:
            ref_param = _get_layer_param(first[idx], attr)
        except AttributeError:
            continue
        is_tied = True
        for other in samples[1:]:
            other_slabs = list(other)
            if idx >= len(other_slabs):
                is_tied = False
                break
            try:
                other_param = _get_layer_param(other_slabs[idx], attr)
            except AttributeError:
                is_tied = False
                break
            if other_param is not ref_param:
                is_tied = False
                break
        spec = f"{name}.{attr}"
        (tied if is_tied else untied).append(spec)

    all_pairs = tied + untied

    if not untied:
        return ([], [], all_pairs)
    if not tied:
        return ([], untied, all_pairs)
    return (
        (tied, [], all_pairs)
        if len(tied) <= len(untied)
        else ([], untied, all_pairs)
    )


def _state_data_files(
    experiments: list,
    indices: List[int],
    *,
    out_dir: Path,
    state_name: str,
    multiple_states: bool,
) -> List[DatasetInfo]:
    """Dump each probe to disk and return a list of ``DatasetInfo``.

    The label format depends on context:

    - single state, single run     → ``run0``
    - single state, N runs         → ``run0``, ``run1``, …
    - multiple states, single run  → ``<state_name>``
    - multiple states, N runs each → ``<state_name>_run0``, ``<state_name>_run1``, …

    Globally unique labels matter for the web UI's
    ``state_for_label`` map, hence the state-name prefix when more than
    one state is present.
    """
    files: List[DatasetInfo] = []
    multi_run = len(indices) > 1
    for k, exp_idx in enumerate(indices):
        probe = experiments[exp_idx].probe
        if multiple_states and multi_run:
            label = f"{state_name}_run{k}"
        elif multiple_states:
            label = state_name
        elif multi_run:
            label = f"run{k}"
        else:
            label = "run0"
        out_path = out_dir / f"{label}.txt"
        _write_probe_to_dat(probe, out_path)
        ds: DatasetInfo = {
            "file": str(out_path.resolve()),
            "label": label,
            "dq_is_fwhm": True,  # we always write dQ as FWHM
        }
        files.append(ds)
    return files


def _summarise_ties(
    samples: list,
    *,
    back_reflection: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """Return ``(tied_specs, untied_specs, warnings)`` for *samples*.

    *tied_specs* and *untied_specs* fully partition the default tied set
    so users can read the CLI summary at a glance. *warnings* is a list
    of human-readable strings to surface verbatim.
    """
    shared, unshared, all_pairs = _recover_tied_set(
        samples, back_reflection=back_reflection
    )
    if not all_pairs:
        return ([], [], [])

    # Reconstruct the partition. _recover_tied_set always returns
    # exactly one populated list (or both empty for the default case);
    # the *other* is the complement w.r.t. all_pairs.
    if not shared and not unshared:
        tied = list(all_pairs)
        untied = []
    elif shared:
        tied = list(shared)
        untied = [p for p in all_pairs if p not in shared]
    else:
        untied = list(unshared)
        tied = [p for p in all_pairs if p not in unshared]

    warnings: list[str] = []
    if len(samples) >= 2 and not tied:
        warnings.append(
            "Detected {n} distinct samples but no shared parameter objects. "
            "If your refl1d script used bumps constraint expressions (e.g. "
            "Parameter.equals) to tie parameters across samples, those ties "
            "were NOT recovered — the imported model treats every structural "
            "parameter as independent per state. Edit final_state.json's "
            "shared_parameters list or re-fit through `aure analyze` to "
            "re-establish the intended ties.".format(n=len(samples))
        )
    return (tied, untied, warnings)


def _looks_like_single_state(
    samples: list,
    state_substrates: list,
    state_ambients: list,
    *,
    back_reflection: bool,
) -> bool:
    """Detect a single physical state spread across multiple experiments.

    bumps serialisation drops Python identity, so a problem built with
    :func:`~aure.nodes.model_builder.build_multi_problem` (one shared
    ``Sample`` across N probes — i.e. one state with N runs) comes back
    as N distinct ``Sample`` objects, identical to a true multi-state
    layout post-deserialisation.

    Heuristic: the two look identical *unless* the per-state ambient or
    substrate materials differ, or at least one default-tied structural
    parameter is untied across samples. When all of those match, treat
    it as one state with N runs (Q-segments, repeats, …) rather than
    inventing fake states.

    The caller can still force multi-state interpretation by passing
    ``state_names`` to :func:`definition_from_problem`.
    """
    if len(samples) < 2:
        return True

    # Ambient and substrate must match name + SLD across all samples.
    first_amb = state_ambients[0]
    for amb in state_ambients[1:]:
        if amb.get("name") != first_amb.get("name"):
            return False
        if abs(float(amb.get("sld", 0.0)) - float(first_amb.get("sld", 0.0))) > 1e-9:
            return False

    first_sub = state_substrates[0]
    for sub in state_substrates[1:]:
        if sub.get("name") != first_sub.get("name"):
            return False
        if abs(float(sub.get("sld", 0.0)) - float(first_sub.get("sld", 0.0))) > 1e-9:
            return False

    # Every default-tied parameter must actually be aliased.
    _shared, unshared, _all = _recover_tied_set(
        samples, back_reflection=back_reflection
    )
    return not unshared


def definition_from_problem(
    problem,
    *,
    data_dir: Path,
    back_reflection: Optional[bool] = None,
    state_names: Optional[List[str]] = None,
) -> ModelDefinition:
    """Invert a (possibly multi-experiment) ``FitProblem`` into a ``ModelDefinition``.

    The returned dict always carries an explicit ``states`` list so
    downstream code that branches on multi-state behaves uniformly,
    even for single-experiment imports.

    When the deserialised problem looks like a single physical state
    spread across N experiments (Q-segment co-refinement, repeat runs)
    — i.e. identical ambient + substrate and full default tying — the
    experiments are collapsed into one state with N runs. Pass
    ``state_names`` to force the multi-state interpretation.

    Parameters
    ----------
    problem
        A deserialised bumps ``FitProblem``.
    data_dir
        Directory where extracted ``.txt`` probe dumps will be written.
        Created if it does not exist.
    back_reflection
        Override the auto-detected stack orientation. When ``None``
        (the default), the orientation is guessed per-sample from
        material names; supply ``True``/``False`` to force one.
    state_names
        Optional override for state names (one per distinct sample, in
        order). When provided, the single-state collapse heuristic is
        bypassed.
    """
    experiments = list(problem.models)
    if not experiments:
        raise ValueError("problem.models is empty — nothing to import.")

    groups = _group_experiments_by_sample(experiments)
    samples = [experiments[g[0]].sample for g in groups]

    # Resolve back_reflection per sample (typically identical across states).
    resolved_back: list[bool] = []
    for sample in samples:
        if back_reflection is not None:
            resolved_back.append(bool(back_reflection))
        else:
            names = [str(s.material.name) for s in sample]
            resolved_back.append(_classify_orientation(names))

    # ── Detect single-state-multi-file ────────────────────────────
    # When all "states" share ambient + substrate + full default tying,
    # collapse them into one state with N runs (Q-segments, repeats, …).
    # The user can override by passing ``state_names`` explicitly.
    if state_names is None and len(groups) > 1:
        state_subs = [
            _structure_from_sample(s, back_reflection=resolved_back[i])[0]
            for i, s in enumerate(samples)
        ]
        state_ambs = [
            _structure_from_sample(s, back_reflection=resolved_back[i])[2]
            for i, s in enumerate(samples)
        ]
        if _looks_like_single_state(
            samples, state_subs, state_ambs, back_reflection=resolved_back[0]
        ):
            # Flatten all experiment indices into one group.
            all_indices = [idx for g in groups for idx in g]
            groups = [all_indices]
            samples = [samples[0]]
            resolved_back = [resolved_back[0]]

    multi_state = len(groups) > 1

    # Build per-state structures.
    state_defs: List[StateDefinition] = []
    for state_idx, (sample, group) in enumerate(zip(samples, groups)):
        name = (
            state_names[state_idx]
            if state_names and state_idx < len(state_names)
            else f"state{state_idx}"
        )
        substrate, layers, ambient = _structure_from_sample(
            sample, back_reflection=resolved_back[state_idx]
        )
        data_files = _state_data_files(
            experiments,
            group,
            out_dir=data_dir,
            state_name=name,
            multiple_states=multi_state,
        )
        intensity = _probe_intensity(experiments[group[0]].probe)
        state_def: StateDefinition = {
            "name": name,
            "data_files": data_files,
            "back_reflection": resolved_back[state_idx],
            "ambient": ambient,
            "intensity": intensity,
        }
        # Stash structure under the state too; build_states_problem reads
        # from the top-level keys but the web UI's parameter editor walks
        # the per-state layers when present.
        state_def["_substrate"] = substrate  # internal hint for callers
        state_def["_layers"] = layers  # not used by build path; debug only
        state_def["_kind"] = _detect_state_kind([d["file"] for d in data_files])
        state_defs.append(state_def)

    # Use state 0 as the canonical structure (cross-state ties make this
    # unambiguous for the default tied set).
    base_substrate = state_defs[0].pop("_substrate")
    base_layers = state_defs[0].pop("_layers")
    base_back = state_defs[0]["back_reflection"]
    base_ambient = state_defs[0]["ambient"]
    base_intensity = state_defs[0]["intensity"]
    # Strip the helper keys off the remaining states too.
    for sd in state_defs[1:]:
        sd.pop("_substrate", None)
        sd.pop("_layers", None)

    # build_states_problem rejects mixed orientations, so the first
    # sample's orientation applies to every state.
    shared_params, unshared_params, _all_pairs = _recover_tied_set(
        samples, back_reflection=resolved_back[0]
    )

    definition: ModelDefinition = {
        "substrate": base_substrate,
        "layers": base_layers,
        "ambient": base_ambient,
        "back_reflection": base_back,
        "constraints": [],
        "data_file": state_defs[0]["data_files"][0]["file"],
        "intensity": base_intensity,
        "dq_is_fwhm": True,
        "states": state_defs,
    }
    if shared_params:
        definition["shared_parameters"] = shared_params
    if unshared_params:
        definition["unshared_parameters"] = unshared_params
    return definition


# --------------------------------------------------------------------------
# Fit result extraction (moved from cli.py for reuse)
# --------------------------------------------------------------------------


def extract_fit_result_from_problem(
    problem,
    *,
    method: str,
    iteration: int,
    export_dir: str,
) -> FitResult:
    """Build a :class:`FitResult` from a deserialised, fitted ``FitProblem``.

    Mirrors the in-process logic in :mod:`aure.nodes.fitting` but works
    on a problem that has already been optimised and exported (no
    bumps ``fit_result`` object is available).

    Used by both ``aure evaluate`` and ``aure import-refl1d``.
    """
    from .nodes.fitting import _read_profile_dat

    chi_squared = float(problem.chisq())

    experiments = list(problem.models)
    is_multi = len(experiments) > 1

    parameters: Dict[str, float] = {}
    uncertainties: Dict[str, float] = {}
    param_bounds: Dict[str, list] = {}
    for par in problem._parameters:
        name = str(par.name)
        parameters[name] = par.value
        lo, hi = _bounds(par)
        if lo is not None and hi is not None:
            param_bounds[name] = [lo, hi]

    Q_fit: List[float] = []
    R_fit: List[float] = []
    residuals: List[float] = []
    residual_ratio: List[float] = []
    per_file_results: Optional[List[PerFileFitResult]] = None

    if is_multi:
        per_file_results = []
        for idx, exp in enumerate(experiments):
            pf: PerFileFitResult = {"file": "", "label": f"dataset {idx + 1}"}
            try:
                exp.update()
                Q_arr, R_arr = exp.reflectivity()
                pf["Q_fit"] = Q_arr.tolist()
                pf["R_fit"] = R_arr.tolist()
                Q_fit.extend(pf["Q_fit"])
                R_fit.extend(pf["R_fit"])

                resid = exp.residuals()
                n_pts = len(resid)
                pf["chi_squared"] = (
                    float(np.sum(resid**2) / n_pts) if n_pts > 0 else float("inf")
                )

                R_data = exp.probe.R
                dR_data = exp.probe.dR
                R_fit_arr = np.array(pf["R_fit"])
                res: list[float] = []
                ratio: list[float] = []
                if R_data is not None and len(R_data) == len(R_fit_arr):
                    if dR_data is not None and len(dR_data) == len(R_data):
                        safe_dR = np.maximum(np.abs(dR_data), 1e-20)
                        res = ((R_data - R_fit_arr) / safe_dR).tolist()
                    safe_R_fit = np.maximum(R_fit_arr, 1e-20)
                    ratio = (R_data / safe_R_fit).tolist()
                pf["residuals"] = res
                pf["residual_ratio"] = ratio
                residuals.extend(res)
                residual_ratio.extend(ratio)
            except Exception:
                pf.setdefault("Q_fit", [])
                pf.setdefault("R_fit", [])
                pf.setdefault("chi_squared", float("inf"))
                pf.setdefault("residuals", [])
                pf.setdefault("residual_ratio", [])
            per_file_results.append(pf)
    else:
        exp = experiments[0]
        try:
            exp.update()
            Q_arr, R_arr = exp.reflectivity()
            Q_fit = Q_arr.tolist()
            R_fit = R_arr.tolist()

            R_data = exp.probe.R
            dR_data = exp.probe.dR
            R_fit_arr = np.array(R_fit)
            if R_data is not None and len(R_data) == len(R_fit_arr):
                if dR_data is not None and len(dR_data) == len(R_data):
                    safe_dR = np.maximum(np.abs(dR_data), 1e-20)
                    residuals = ((R_data - R_fit_arr) / safe_dR).tolist()
                safe_R_fit = np.maximum(R_fit_arr, 1e-20)
                residual_ratio = (R_data / safe_R_fit).tolist()
        except Exception:
            pass

    sld_z, sld_rho = _read_profile_dat(export_dir)

    return FitResult(
        iteration=iteration,
        method=method,
        chi_squared=chi_squared,
        converged=True,
        parameters=parameters,
        uncertainties=uncertainties if uncertainties else None,
        bounds=param_bounds if param_bounds else None,
        Q_fit=Q_fit,
        R_fit=R_fit,
        residuals=residuals,
        residual_ratio=residual_ratio,
        sld_z=sld_z,
        sld_rho=sld_rho,
        per_file_results=per_file_results,
        issues=[],
        suggestions=[],
    )


def _attach_states_to_per_file(
    per_file_results: Optional[List[PerFileFitResult]],
    states: List[StateDefinition],
) -> None:
    """Fill in ``PerFileFitResult.state``/``file``/``label`` from *states*.

    The per-file results come out of :func:`extract_fit_result_from_problem`
    in the same order as ``problem.models`` (state-by-state, then within
    each state by file order), so we can walk both lists in lock-step.
    """
    if not per_file_results:
        return
    idx = 0
    for state in states:
        for ds in state.get("data_files", []) or []:
            if idx >= len(per_file_results):
                return
            pf = per_file_results[idx]
            pf["state"] = state.get("name", "")
            pf["file"] = ds.get("file", "")
            pf["label"] = ds.get("label", "")
            idx += 1


# --------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------


def resolve_refl1d_dir(refl1d_dir: str) -> Tuple[Path, Path, int, str]:
    """Locate ``problem.json`` from a refl1d output directory.

    Accepts either a specific ``fit_iter*_*`` directory or its parent.

    Returns
    -------
    (refl1d_path, problem_file, iteration, method)
    """
    path = Path(refl1d_dir)
    problem_file = path / "problem.json"
    if not problem_file.exists():
        candidates = sorted(path.glob("fit_iter*_*/problem.json"))
        if not candidates:
            # Maybe the user pointed at an AuRE output dir with refl1d_output/
            sub = path / "refl1d_output"
            if sub.exists():
                candidates = sorted(sub.glob("fit_iter*_*/problem.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No problem.json found in {refl1d_dir} or its fit_iter* subdirs."
            )
        problem_file = candidates[-1]

    refl1d_path = problem_file.parent
    match = re.search(r"fit_iter(\d+)_(\w+)", refl1d_path.name)
    iteration = int(match.group(1)) if match else 0
    method = match.group(2) if match else "imported"
    return refl1d_path, problem_file, iteration, method


# --------------------------------------------------------------------------
# Setup-driven inversion
# --------------------------------------------------------------------------


def _setup_file_q_min(file_path: str) -> float:
    """Return ``Q.min()`` for a 4-column reflectivity data file.

    Used to align user-listed setup files with refl1d experiments,
    which both :func:`~aure.nodes.model_builder.build_states_problem`
    and :func:`~aure.nodes.model_builder.build_multi_problem` sort by
    ``Q.min()`` before assembling.
    """
    arr = np.loadtxt(file_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return float(arr[:, 0].min())


def _sort_setup_files_by_q(files: list[dict]) -> list[dict]:
    """Return *files* reordered to match refl1d's internal Q-ascending sort.

    Each element is a :class:`DatasetInfo`-shaped dict from a parsed
    setup state. The original labels are preserved.
    """
    annotated = [(ds, _setup_file_q_min(ds["file"])) for ds in files]
    annotated.sort(key=lambda x: x[1])
    return [ds for ds, _ in annotated]


def _validate_setup_against_problem(
    experiments: list, setup_states: list
) -> None:
    """Check that the setup's file count lines up with the problem's experiments."""
    n_files = sum(len(st.get("data_files") or []) for st in setup_states)
    n_exps = len(experiments)
    if n_files != n_exps:
        raise ValueError(
            f"Setup file declares {n_files} data file(s) across "
            f"{len(setup_states)} state(s), but the refl1d problem.json has "
            f"{n_exps} experiment(s). The setup must describe the same problem."
        )


def _definition_from_setup_and_problem(
    problem,
    setup: dict,
    *,
    back_reflection: Optional[bool] = None,
) -> ModelDefinition:
    """Invert a refl1d problem using a setup YAML as the source of truth.

    The setup YAML drives:

    - state grouping (no auto-detection, no single-state collapse)
    - state names and per-state nuisance / ambient / intensity overrides
    - data file paths (the originals are referenced directly — no probe
      dump is written to ``data/``)

    The refl1d problem provides:

    - layer structure (substrate, layers, ambient names + SLDs)
    - fitted parameter values + bounds
    - χ², theory curves, residuals (via :func:`extract_fit_result_from_problem`)

    Experiments are matched to setup files by per-state ``Q.min()``
    ordering: within each state, setup files are sorted in ascending
    ``Q.min()`` to align with refl1d's internal experiment order. The
    experiment list is sliced state-by-state, matching the layout that
    :func:`~aure.nodes.model_builder.build_states_problem` (and
    :func:`~aure.nodes.model_builder.build_multi_problem`) produce.
    """
    experiments = list(problem.models)
    setup_states = setup.get("states") or []
    if not setup_states:
        raise ValueError("setup file has no states: cannot drive the import.")

    _validate_setup_against_problem(experiments, setup_states)

    # ── Slice experiments per state (state-by-state in setup order) ─────
    exp_by_state: list[list[int]] = []
    cursor = 0
    for st in setup_states:
        n = len(st.get("data_files") or [])
        exp_by_state.append(list(range(cursor, cursor + n)))
        cursor += n

    # ── Orientation (back_reflection): CLI override > setup state > auto ─
    samples = [experiments[idxs[0]].sample for idxs in exp_by_state]
    resolved_back: list[bool] = []
    for state_idx, sample in enumerate(samples):
        if back_reflection is not None:
            resolved_back.append(bool(back_reflection))
        elif "back_reflection" in setup_states[state_idx]:
            resolved_back.append(bool(setup_states[state_idx]["back_reflection"]))
        else:
            names = [str(s.material.name) for s in sample]
            resolved_back.append(_classify_orientation(names))

    # ── Canonical structure (from state 0's refl1d sample) ──────────────
    base_substrate, base_layers, base_ambient = _structure_from_sample(
        samples[0], back_reflection=resolved_back[0]
    )

    # ── Build per-state definitions ─────────────────────────────────────
    state_defs: List[StateDefinition] = []
    for state_idx, setup_state in enumerate(setup_states):
        idxs = exp_by_state[state_idx]
        sample = samples[state_idx]
        st_back = resolved_back[state_idx]

        # Use setup's data files, reordered to match refl1d's Q-ascending
        # experiment order within this state.
        sorted_files = _sort_setup_files_by_q(setup_state["data_files"])
        data_files = [
            {"file": ds["file"], "label": ds.get("label") or Path(ds["file"]).stem}
            for ds in sorted_files
        ]

        # Ambient: setup override wins (it carries the scientist's
        # canonical labelling, e.g. "D2O" instead of refl1d's serialized
        # short name). Otherwise extract from the per-state refl1d sample.
        _, _, refl1d_ambient = _structure_from_sample(sample, back_reflection=st_back)
        ambient = (
            dict(setup_state["ambient"])
            if setup_state.get("ambient")
            else refl1d_ambient
        )

        # Intensity: setup override wins; else read from the first probe.
        intensity = (
            dict(setup_state["intensity"])
            if setup_state.get("intensity")
            else _probe_intensity(experiments[idxs[0]].probe)
        )

        state_def: StateDefinition = {
            "name": setup_state["name"],
            "data_files": data_files,
            "back_reflection": st_back,
            "ambient": ambient,
            "intensity": intensity,
        }
        for opt in ("extra_description", "theta_offset", "sample_broadening"):
            if setup_state.get(opt):
                state_def[opt] = setup_state[opt]
        state_def["_kind"] = (
            setup_state.get("_kind")
            or _detect_state_kind([ds["file"] for ds in data_files])
        )
        state_defs.append(state_def)

    # ── Cross-state tied set (still recovered from refl1d via id()) ─────
    shared_params, unshared_params, _ = _recover_tied_set(
        samples, back_reflection=resolved_back[0]
    )

    definition: ModelDefinition = {
        "substrate": base_substrate,
        "layers": base_layers,
        "ambient": base_ambient,
        "back_reflection": resolved_back[0],
        "constraints": [],
        "data_file": state_defs[0]["data_files"][0]["file"],
        "intensity": state_defs[0]["intensity"],
        "dq_is_fwhm": True,
        "states": state_defs,
    }
    if shared_params:
        definition["shared_parameters"] = shared_params
    if unshared_params:
        definition["unshared_parameters"] = unshared_params
    return definition


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def import_refl1d(
    refl1d_dir: str,
    output_dir: str,
    *,
    setup_path: Optional[str] = None,
    setup_data_dir: Optional[str] = None,
    sample_description: Optional[str] = None,
    hypothesis: Optional[str] = None,
    state_names: Optional[List[str]] = None,
    back_reflection: Optional[bool] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Materialise an AuRE output directory from a refl1d ``problem.json``.

    When ``setup_path`` points at a setup YAML describing the same
    problem (e.g. the file an analyzer ``plan-data`` step emitted), the
    setup drives state grouping, state names, sample description, and
    data file paths — the refl1d output supplies the fitted numbers.
    Without it, the importer auto-detects everything from the deserialised
    problem (heuristics + collapse logic).

    Returns a summary dict describing what was imported (state count,
    χ², output paths). Does not call any LLM — purely deterministic.

    Raises
    ------
    FileExistsError
        If *output_dir* exists and ``force`` is False.
    FileNotFoundError
        If a ``problem.json`` cannot be located under *refl1d_dir*.
    ValueError
        If *setup_path* is given but its state/file count doesn't match
        the problem's experiments, or if *output_dir* would land inside
        the source refl1d tree.
    """
    if setup_data_dir is not None and not setup_path:
        raise ValueError("setup_data_dir is meaningful only together with setup_path")

    refl1d_path, problem_file, iteration, method = resolve_refl1d_dir(refl1d_dir)

    # ── Load setup (if any) ──────────────────────────────────────────
    setup: Optional[dict] = None
    if setup_path:
        from .setup import load_setup

        setup = load_setup(setup_path, data_dir=setup_data_dir)

    out = Path(output_dir).resolve()

    # Refuse to write inside the source tree: ``shutil.copytree(refl1d_path,
    # …/out/refl1d_output/…)`` would otherwise copy the output back into
    # itself, blowing up with an infinite-recursion path explosion when
    # ``out`` already exists as a child of the source.
    try:
        out.relative_to(refl1d_path)
        is_inside = True
    except ValueError:
        is_inside = False
    if is_inside:
        raise ValueError(
            f"Refusing to write the AuRE workspace inside the source refl1d "
            f"directory.\n  source: {refl1d_path}\n  output: {out}\n"
            "Pick an --output-dir outside the source tree (the default is a "
            "sibling directory)."
        )

    if out.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {out}. Use --force to overwrite."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # --- Load the bumps problem ----------------------------------------------
    from bumps.serialize import deserialize  # type: ignore[import-not-found]

    with open(problem_file) as fh:
        problem = deserialize(json.load(fh))

    # --- Reconstruct ModelDefinition (with states) ---------------------------
    data_dir = out / "data"
    if setup is not None:
        if state_names:
            # --setup is authoritative for state names; let the user know
            # their CLI override is being dropped rather than silently
            # accepting and ignoring it.
            raise ValueError(
                "--state-name cannot be combined with --setup; state names "
                "come from the setup file."
            )
        definition = _definition_from_setup_and_problem(
            problem,
            setup,
            back_reflection=back_reflection,
        )
    else:
        definition = definition_from_problem(
            problem,
            data_dir=data_dir,
            back_reflection=back_reflection,
            state_names=state_names,
        )
    states_list: List[StateDefinition] = definition["states"]  # type: ignore[assignment]

    # Tie summary + warnings (only meaningful for ≥ 2 distinct samples)
    experiments_list = list(problem.models)
    sample_groups = _group_experiments_by_sample(experiments_list)
    distinct_samples = [experiments_list[g[0]].sample for g in sample_groups]
    tied_specs, untied_specs, tie_warnings = _summarise_ties(
        distinct_samples, back_reflection=bool(definition["back_reflection"])
    )

    # --- Extract fit result + attach state metadata --------------------------
    fit_result = extract_fit_result_from_problem(
        problem,
        method=method,
        iteration=0,  # imported runs always anchor at iteration 0
        export_dir=str(refl1d_path),
    )
    _attach_states_to_per_file(fit_result.get("per_file_results"), states_list)

    # Update per-state data_files with Q/R/dR + header metadata so the
    # web UI plots them and a downstream refinement loads them with the
    # right probe type. Two paths:
    #
    # - **setup mode**: enrich from the user's original files on disk.
    #   The headers carry the real ``theta`` and resolution convention,
    #   so a subsequent refine call loads them via ``load_probe_from_angle``
    #   (NeutronProbe), which is what makes per-probe
    #   ``sample_broadening`` and ``theta_offset`` parameters available.
    #
    # - **auto-detect mode**: the data files referenced by the workspace
    #   are headerless probe dumps under ``<output>/data/``. We extract
    #   Q/R/dR straight from the deserialised refl1d probe (since the
    #   dumps contain nothing else useful) and default the rest.
    experiments = list(problem.models)
    flat_data_files = flatten_data_files(states_list)
    if setup is not None:
        from .nodes.intake import _parse_theta_from_header
        from .tools.data_tools import load_reflectivity_data

        for ds in flat_data_files:
            file_path = ds["file"]
            data = load_reflectivity_data(file_path)
            ds["Q"] = data["Q"].tolist()
            ds["R"] = data["R"].tolist()
            dR = data.get("dR")
            ds["dR"] = (
                dR.tolist() if dR is not None else [0.0] * len(ds["Q"])
            )
            # Deterministic theta extraction from the header; falls back
            # to 0.0 for combined / multi-segment files (intake-style
            # behaviour). dq_is_fwhm and num_segments use the same
            # defaults the intake node falls back to when the LLM is
            # unavailable.
            ds["theta"] = _parse_theta_from_header(file_path)
            ds.setdefault("dq_is_fwhm", True)
            ds.setdefault("num_segments", 0)
    else:
        for ds, exp in zip(flat_data_files, experiments):
            probe = exp.probe
            ds["Q"] = np.asarray(probe.Q, dtype=float).tolist()
            ds["R"] = np.asarray(probe.R, dtype=float).tolist()
            if probe.dR is not None:
                ds["dR"] = np.asarray(probe.dR, dtype=float).tolist()
            else:
                ds["dR"] = [0.0] * len(ds["Q"])
            ds["dq_is_fwhm"] = True
            ds["theta"] = 0.0
            ds["num_segments"] = 0

    # --- Build a synthetic ReflectivityState -----------------------------------
    primary_ds = states_list[0]["data_files"][0]
    Q0 = primary_ds["Q"]
    R0 = primary_ds["R"]
    dR0 = primary_ds["dR"]

    # Setup metadata wins over auto-synthesised values; explicit CLI
    # arguments (``sample_description``, ``hypothesis``) override the setup.
    setup_mode = setup is not None
    if setup_mode:
        if hypothesis is None and setup.get("hypothesis"):
            hypothesis = setup["hypothesis"]
        if not sample_description and setup.get("sample_description"):
            sample_description = setup["sample_description"]

    parsed_sample = _synth_parsed_sample(definition, hypothesis=hypothesis)
    extracted_features = _synth_features(Q0, R0, dR0)
    sample_desc = sample_description or _default_sample_description(
        parsed_sample, len(states_list)
    )

    state = create_initial_state(
        data_file=primary_ds["file"],
        sample_description=sample_desc,
        hypothesis=hypothesis,
        max_iterations=0,
        states=[dict(s) for s in states_list],
    )
    state["Q"] = list(Q0)
    state["R"] = list(R0)
    state["dR"] = list(dR0)
    state["dq_is_fwhm"] = True
    state["data_files"] = flat_data_files
    state["parsed_sample"] = parsed_sample
    state["extracted_features"] = extracted_features
    state["current_model"] = dict(definition)
    state["model_history"] = [dict(definition)]
    state["best_model"] = dict(definition)
    state["best_bic_model"] = dict(definition)
    state["fit_results"] = [fit_result]
    state["current_chi2"] = fit_result["chi_squared"]
    state["best_chi2"] = fit_result["chi_squared"]
    n_data = sum(len(ds.get("Q") or []) for ds in flat_data_files) or len(Q0)
    n_params = _count_free_params(definition)
    state["best_bic"] = _compute_bic(fit_result["chi_squared"], n_data, n_params)
    fit_result["bic"] = state["best_bic"]
    state["current_node"] = "evaluation"
    state["iteration"] = 1
    state["workflow_complete"] = True
    state["output_dir"] = str(out)
    state["active_skills"] = (
        ["multi-state-corefinement", "neutron-reflectometry"]
        if len(states_list) > 1
        else ["neutron-reflectometry"]
    )
    state["messages"] = [
        Message(
            role="system",
            content=(
                f"Imported from refl1d output: {refl1d_path}\n"
                f"states={len(states_list)} files={len(flat_data_files)} "
                f"chi2={fit_result['chi_squared']:.4f}"
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    ]

    # --- Write the directory layout -----------------------------------------
    mgr = CheckpointManager(str(out))
    mgr.initialize(
        state,
        data_file=primary_ds["file"],
        sample_description=sample_desc,
    )

    # Save checkpoints. Mirror the per-node iteration counts a real run
    # produces: intake/analysis/modeling at iteration 0; fitting at 0;
    # evaluation increments to 1. The checkpoint manager appends
    # ``_iter{n}`` to fitting/evaluation filenames when iteration > 0,
    # so the resulting names match a normal run's layout.
    pre_fit = ("intake", "analysis", "modeling")
    for node in pre_fit:
        state["current_node"] = node
        state["iteration"] = 0
        mgr.save_checkpoint(state, node)
    state["current_node"] = "fitting"
    state["iteration"] = 0
    mgr.save_checkpoint(state, "fitting")
    state["current_node"] = "evaluation"
    state["iteration"] = 1
    mgr.save_checkpoint(state, "evaluation")

    # We don't duplicate the refl1d output tree — the SLD profile and
    # theory curves are already baked into ``final_state.json`` via
    # ``extract_fit_result_from_problem``. Only the top-level
    # ``problem.json`` is needed downstream (``aure evaluate`` and the
    # web UI's parameter editor both read it).
    #
    # ``CheckpointManager.save_final_state`` looks for problem.json at
    # ``refl1d_output/fit_iter0_<method>/`` and warns if missing. Stage
    # exactly one file there so the lookup succeeds, then strip the
    # staging directory once save_final_state has copied it to the top
    # level.
    canonical_dir = out / "refl1d_output" / f"fit_iter0_{method}"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(problem_file, canonical_dir / "problem.json")

    mgr.save_final_state(state)

    # Tear down the staging directory — we never advertised it as part
    # of the workspace.
    shutil.rmtree(out / "refl1d_output", ignore_errors=True)

    # Defensive: if save_final_state's copy somehow didn't happen, drop
    # one in directly.
    top_problem = out / "problem.json"
    if not top_problem.exists():
        shutil.copy2(problem_file, top_problem)

    logger.info("[IMPORT] Wrote AuRE workspace at %s", out)

    return {
        "output_dir": str(out),
        "states": [s["name"] for s in states_list],
        "n_files": len(flat_data_files),
        "chi_squared": fit_result["chi_squared"],
        "method": method,
        "iteration": iteration,
        "back_reflection": definition["back_reflection"],
        "tied_parameters": tied_specs,
        "untied_parameters": untied_specs,
        "warnings": tie_warnings,
    }


# --------------------------------------------------------------------------
# Synthesis helpers (no LLM)
# --------------------------------------------------------------------------


def _synth_parsed_sample(
    definition: ModelDefinition, *, hypothesis: Optional[str]
) -> ParsedSample:
    """Build a :class:`ParsedSample` from a recovered model definition."""
    return ParsedSample(
        substrate=dict(definition["substrate"]),  # type: ignore[typeddict-item]
        layers=[dict(layer) for layer in definition.get("layers", [])],  # type: ignore[arg-type]
        ambient=dict(definition["ambient"]),  # type: ignore[typeddict-item]
        constraints=list(definition.get("constraints", []) or []),
        hypothesis=hypothesis,
        back_reflection=bool(definition.get("back_reflection", False)),
    )


def _synth_features(
    Q: list, R: list, dR: Optional[list]
) -> dict:
    """Run deterministic feature extraction on the primary dataset."""
    from .tools.feature_tools import extract_all_features

    Q_arr = np.asarray(Q, dtype=float)
    R_arr = np.asarray(R, dtype=float)
    dR_arr = np.asarray(dR, dtype=float) if dR else None
    try:
        return extract_all_features(Q_arr, R_arr, dR_arr)
    except Exception as exc:
        logger.warning("[IMPORT] Feature extraction failed: %s", exc)
        return {
            "critical_edges": [],
            "oscillation_periods": [],
            "estimated_total_thickness": None,
            "n_fringes": 0,
            "estimated_roughness": 0.0,
            "roughness_confidence": "low",
            "estimated_n_layers": len(R) // 50 if R else 0,
            "layer_count_confidence": "low",
            "q_min": float(Q_arr.min()) if len(Q_arr) else 0.0,
            "q_max": float(Q_arr.max()) if len(Q_arr) else 0.0,
            "n_points": len(Q_arr),
            "has_error_bars": dR is not None,
            "normalization_ok": True,
        }


def _default_sample_description(
    parsed: ParsedSample, n_states: int
) -> str:
    """Synthesise a sample description from the recovered structure.

    Layer order in :class:`ParsedSample` runs *bottom-up* (the first
    entry is adjacent to the substrate, the last entry is adjacent to
    the ambient). The conventional English reading runs top-down
    ("Cu on Ti on Si"), so we reverse before joining.

    When the run is back-reflecting, we append a sentence that names
    the substrate explicitly — the LLM relies on this to keep the
    neutron-entry side straight during refinement.
    """
    layers = parsed.get("layers") or []
    sub = parsed["substrate"]["name"]
    amb = parsed["ambient"]["name"]
    back_reflection = bool(parsed.get("back_reflection", False))

    if layers:
        top_down = list(reversed(layers))
        layer_str = " on ".join(
            f"{lay['thickness']:.1f} Å {lay['name']}" for lay in top_down
        )
        base = f"{layer_str} on {sub} in {amb}"
    else:
        base = f"bare {sub} substrate in {amb}"

    if back_reflection:
        base += f". Neutrons enter from the {sub} substrate side"

    if n_states > 1:
        base += f" (imported {n_states}-state co-refinement)"
    else:
        base += " (imported from refl1d)"
    return base
