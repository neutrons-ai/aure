"""
Model builder: construct refl1d Experiment / FitProblem from a ModelDefinition.

This module replaces the old approach of storing models as Python scripts.
Instead, models are stored as structured JSON (ModelDefinition dicts) and
refl1d objects are built on-the-fly when needed for fitting or visualisation.

Key functions:
- build_experiment()  — JSON → refl1d Experiment
- build_problem()     — JSON → bumps FitProblem
- extract_definition() — fitted FitProblem → updated ModelDefinition
"""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ======================================================================
# Data loading
# ======================================================================


def load_probe(file_path: str, *, dq_is_fwhm: bool = True):
    """Load a reflectivity data file into a refl1d ``Probe`` object.

    This is the single entry-point for data loading into refl1d.
    Different loading strategies can be dispatched here in the future
    (e.g., ORSO, polarised, event-mode).

    Parameters
    ----------
    file_path
        Path to the reflectivity data file (ASCII 4-column, .refl, .ort).
    dq_is_fwhm
        Whether the dQ column is FWHM (True) or 1-sigma (False).
        Defaults to True, which is the convention for most instrument
        reduction software.

    Returns
    -------
    probe
        A refl1d ``Probe`` (or ``QProbe``) object.
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="refl1d")
    from refl1d.names import load4

    abs_path = os.path.abspath(file_path)
    return load4(abs_path, FWHM=dq_is_fwhm)


def load_probe_from_angle(file_path: str, theta: float, *, dq_is_fwhm: bool = True):
    """Load a reflectivity data file and create an angle-based ``NeutronProbe``.

    Unlike :func:`load_probe` (which creates a Q-based ``QProbe``), this
    builds a ``NeutronProbe`` using the incident angle *theta*.  This
    enables ``sample_broadening`` and ``theta_offset`` as fittable
    parameters — important for multi-segment co-refinement where each
    segment may need independent resolution corrections.

    Parameters
    ----------
    file_path
        Path to the reflectivity data file (ASCII 4-column).
    theta
        Incident angle in degrees (half of TwoTheta from the header).
    dq_is_fwhm
        Whether the dQ column is FWHM (True) or 1-sigma (False).
    """
    import numpy as np
    from refl1d.probe import make_probe

    abs_path = os.path.abspath(file_path)
    q, r, dr, dq = np.loadtxt(abs_path).T

    if not dq_is_fwhm:
        dq = dq * (2 * np.sqrt(2 * np.log(2)))

    theta_rad = np.deg2rad(theta)
    wl = 4 * np.pi * np.sin(theta_rad) / q
    dT = dq / q * np.tan(theta_rad) * 180.0 / np.pi
    dL = np.zeros_like(q)

    return make_probe(
        T=theta,
        dT=dT,
        L=wl,
        dL=dL,
        data=(r, dr),
        radiation="neutron",
        resolution="uniform",
    )


# ======================================================================
# Build refl1d objects from ModelDefinition
# ======================================================================

# Default flat-background range used when a ``background`` block omits values.
_BACKGROUND_DEFAULTS = {"init": 1e-6, "min": 0.0, "max": 1e-5}


def _nuisance_enabled(spec) -> bool:
    """True when a nuisance block (background / theta_offset / sample_broadening)
    requests a parameter.

    A nuisance spec reaches the builder in one of two shapes and this reconciles
    both:

    * model-level (from the modeling LLM): ``{"enabled": bool, "min", "max"}`` —
      always a dict even when off, so it must be read via the ``enabled`` flag
      (a plain truthiness test would treat ``{"enabled": False}`` as on);
    * per-state (from config ``_normalise_nuisance``): ``{"init", "min", "max"}``
      when on, ``None`` when off — no ``enabled`` key, so presence of a
      range/init means on (a ``.get("enabled")`` test would treat it as off).

    Using one predicate everywhere keeps the three nuisance parameters
    consistent. ``background`` in particular cannot rely on ``hasattr(probe,
    ...)`` to gate it the way the angle-only ``theta_offset`` /
    ``sample_broadening`` do, because it exists on every refl1d probe type.
    """
    if not isinstance(spec, dict) or not spec:
        return False
    if "enabled" in spec:
        return bool(spec["enabled"])
    return any(k in spec for k in ("init", "value", "min", "max", "fixed"))


def _configure_background(probe, spec) -> bool:
    """Set ``probe.background`` from a background block. Returns True if applied.

    ``fixed: true`` sets only the value (constant background); otherwise the
    background is made fittable over ``[min, max]``. No-ops when the spec is
    disabled or the probe has no ``background`` parameter. Unlike
    ``theta_offset`` / ``sample_broadening``, ``background`` exists on every
    refl1d probe type, so it works for combined and partial data alike.
    """
    if not _nuisance_enabled(spec) or not hasattr(probe, "background"):
        return False
    probe.background.value = spec.get(
        "init", spec.get("value", _BACKGROUND_DEFAULTS["init"])
    )
    if not spec.get("fixed", False):
        probe.background.range(
            spec.get("min", _BACKGROUND_DEFAULTS["min"]),
            spec.get("max", _BACKGROUND_DEFAULTS["max"]),
        )
    return True


def build_experiment(definition: dict):
    """Construct a refl1d ``Experiment`` from a ``ModelDefinition`` dict.

    Parameters
    ----------
    definition
        A ``ModelDefinition`` dict with keys: substrate, layers, ambient,
        data_file, back_reflection, and optionally intensity.

    Returns
    -------
    experiment : refl1d.experiment.Experiment
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="refl1d")
    from refl1d.names import Experiment

    data_file = definition["data_file"]
    dq_is_fwhm = definition.get("dq_is_fwhm", True)
    probe = load_probe(data_file, dq_is_fwhm=dq_is_fwhm)
    intensity = definition.get("intensity", {})

    sample = _build_sample(definition)

    # Probe intensity
    if not intensity.get("fixed", False):
        int_val = intensity.get("value", 1.0)
        int_min = intensity.get("min", 0.7)
        int_max = intensity.get("max", 1.1)
        probe.intensity.value = int_val
        probe.intensity.range(int_min, int_max)

    # Optional fittable flat background.
    _configure_background(probe, definition.get("background", {}))

    experiment = Experiment(probe=probe, sample=sample)
    return experiment


_OUTER_ROUGHNESS_MAX_DEFAULT = 30.0


def _outer_roughness_max(layers_info: list) -> float:
    """Upper bound for the outermost (fronting-side) interface roughness.

    In back reflection the stack is assembled ambient-first, so
    ``sample[0].interface`` is the boundary between the ambient and the
    *topmost* layer — that layer's own outer interface. Its seed value already
    comes from ``layers_info[-1]["roughness"]``, so its ceiling is taken from the
    same layer's ``roughness_max`` for consistency, and stays overridable with
    ``ROUGHNESS_MAX_OUTER`` for batch/validation runs.

    This used to be hardcoded at 30 A, which silently capped genuinely diffuse
    interfaces: a solvent-swollen SEI is not a sharp slab, and against expert
    reference fits of Cu-in-THF electrodes the outer roughness exceeds 30 A in
    40 of 51 measured runs, reaching 209 A. No amount of adjustment elsewhere in
    the model can absorb that, so the fit lands in a systematically wrong basin.

    The default is unchanged, so behaviour only differs when a model (or the
    environment) explicitly asks for a wider bound.
    """
    override = os.environ.get("ROUGHNESS_MAX_OUTER")
    if override:
        try:
            return float(override)
        except ValueError:
            logger.warning(
                "[MODEL] Ignoring non-numeric ROUGHNESS_MAX_OUTER=%r", override
            )
    if layers_info:
        try:
            return float(
                layers_info[-1].get("roughness_max", _OUTER_ROUGHNESS_MAX_DEFAULT)
            )
        except (TypeError, ValueError):
            pass
    return _OUTER_ROUGHNESS_MAX_DEFAULT


def _build_sample(definition: dict):
    """Build a refl1d sample stack with parameter ranges from a ModelDefinition.

    The returned ``sample`` is a refl1d stack object suitable for passing
    to ``Experiment(probe=..., sample=sample)``.  All parameter ranges
    (thickness, SLD, roughness) are applied.

    This is factored out of :func:`build_experiment` so that
    :func:`build_multi_problem` can share a single sample across
    multiple experiments for multi-file co-refinement.
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="refl1d")
    from refl1d.names import SLD

    substrate_info = definition["substrate"]
    ambient_info = definition["ambient"]
    layers_info = definition.get("layers", [])
    back_reflection = definition.get("back_reflection", False)

    # --- Materials ---
    substrate = SLD(name=substrate_info["name"], rho=substrate_info["sld"])
    ambient = SLD(name=ambient_info["name"], rho=ambient_info["sld"])

    materials = []
    for layer in layers_info:
        materials.append(SLD(name=layer["name"], rho=layer["sld"]))

    # --- Sample stack ---
    if back_reflection:
        roughness_first = layers_info[-1]["roughness"] if layers_info else 3.0
        stack_parts = [ambient(0, roughness_first)]
        for i in reversed(range(len(layers_info))):
            layer = layers_info[i]
            stack_parts.append(materials[i](layer["thickness"], layer["roughness"]))
        stack_parts.append(substrate)
    else:
        stack_parts = [substrate(0, substrate_info.get("roughness", 3.0))]
        for i, layer in enumerate(layers_info):
            stack_parts.append(materials[i](layer["thickness"], layer["roughness"]))
        stack_parts.append(ambient)

    sample = stack_parts[0]
    for part in stack_parts[1:]:
        sample = sample | part

    # --- Parameter ranges ---
    if (
        ambient_info.get("name", "").lower() != "air"
        and ambient_info.get("sld", 0) != 0
    ):
        amb_sld = ambient_info["sld"]
        amb_min = ambient_info.get("sld_min", max(amb_sld * 0.8, -1.0))
        amb_max = ambient_info.get("sld_max", amb_sld * 1.2)
        amb_idx = 0 if back_reflection else len(layers_info) + 1
        sample[amb_idx].material.rho.range(amb_min, amb_max)

    for i, layer in enumerate(layers_info):
        if back_reflection:
            idx = len(layers_info) - i
        else:
            idx = i + 1

        t_min = layer.get("thickness_min", layer["thickness"] * 0.5)
        t_max = layer.get("thickness_max", layer["thickness"] * 2.0)
        sample[idx].thickness.range(t_min, t_max)

        sld_min = layer.get("sld_min", layer["sld"] - 2.5)
        sld_max = layer.get("sld_max", layer["sld"] + 2.5)
        sample[idx].material.rho.range(sld_min, sld_max)

        tie = layer.get("roughness_tie")
        if tie:
            # Tie this layer's interface roughness to its own thickness:
            # sigma = fraction * thickness, with `fraction` the fitted free
            # parameter. Prevents a thin layer's interface from outgrowing it
            # (erf-tail artifacts) while the thickness stays free.
            from bumps.parameter import Parameter

            f_init = float(tie.get("fraction_init", 0.3))
            f_min = float(tie.get("fraction_min", 0.05))
            f_max = float(tie.get("fraction_max", 0.5))
            frac = Parameter(f_init, name=f"{layer['name']} rough_frac")
            frac.range(f_min, f_max)
            sample[idx].interface = frac * sample[idx].thickness
        else:
            r_min = layer.get("roughness_min", 5.0)
            r_max = layer.get("roughness_max", 30.0)
            sample[idx].interface.range(r_min, r_max)

    if back_reflection:
        sample[0].interface.range(0, _outer_roughness_max(layers_info))
    else:
        sub_rough_max = substrate_info.get("roughness_max", 15.0)
        sample[0].interface.range(0, sub_rough_max)

    return sample


# ======================================================================
# Derived parameters (reparametrization)
# ======================================================================


_NAMESPACE_ATTRS = ("thickness", "interface", "material.rho")


def _parameter_namespace(
    definition: dict, sample, *, extra: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Map ``"<layer>.<attr>"`` to the live ``bumps`` parameter on *sample*.

    Built fresh at each use rather than cached, so an expression evaluated
    after an assignment sees the assigned value. That ordering is what lets a
    ``keep_physical`` guard constrain the DERIVED quantity (``SEI.rho > 0``)
    rather than the free parameter that replaced it.

    ``substrate`` and ``ambient`` are registered under both their material name
    and the literal alias, matching the tie-spec vocabulary.
    """
    ns: Dict[str, Any] = {}
    back = definition.get("back_reflection", False)

    names: list[tuple[str, str]] = []
    for layer in definition.get("layers") or []:
        if isinstance(layer, dict) and layer.get("name"):
            names.append((layer["name"], layer["name"]))
    sub_name = (definition.get("substrate") or {}).get("name")
    if sub_name:
        names.append((sub_name, sub_name))
        names.append((sub_name, "substrate"))
    amb_name = (definition.get("ambient") or {}).get("name")
    if amb_name:
        names.append((amb_name, amb_name))
        names.append((amb_name, "ambient"))

    for real_name, alias in names:
        idx = _layer_index(definition, real_name, back_reflection=back)
        if idx is None:
            continue
        for attr in _NAMESPACE_ATTRS:
            try:
                ns[f"{alias}.{attr}"] = _get_layer_param(sample[idx], attr)
            except AttributeError:
                continue  # e.g. the semi-infinite media have no thickness
    ns.update(extra or {})
    return ns


def _derived_specs(definition: dict, state_name: str) -> list[dict]:
    """The declarations that apply to this state (all of them when unscoped)."""
    out = []
    for spec in definition.get("derived_parameters") or []:
        if not isinstance(spec, dict) or not spec.get("name"):
            continue
        scope = spec.get("states") or []
        if scope and state_name and state_name not in scope:
            continue
        out.append(spec)
    return out


def _derived_assigned_slots(definition: dict, state_name: str) -> set:
    """``(sample index, attr path)`` pairs a reparametrization takes over here.

    A raw parameter that a declaration assigns is no longer free, so a
    cross-state tie must not alias it: doing so would replace the per-state
    expression with state 0's — reintroducing exactly the shared coordinate the
    reparametrization exists to remove, and silently resolving each state's
    derived SLD against the WRONG state's ambient.
    """
    from .expressions import canonical_name

    slots: set = set()
    back = definition.get("back_reflection", False)
    for spec in _derived_specs(definition, state_name):
        for target in (spec.get("assign") or {}):
            canon = canonical_name(str(target))
            layer_name, _, attr_path = canon.partition(".")
            if not attr_path:
                continue
            idx = _layer_index(definition, layer_name, back_reflection=back)
            if idx is not None:
                slots.add((idx, attr_path))
    return slots


def apply_derived_parameters(
    definition: dict,
    sample,
    *,
    state_name: str = "",
    shared: Dict[str, Any] | None = None,
) -> list:
    """Reparametrize *sample* in place; return the physicality constraints.

    For each declaration a new free ``Parameter`` is created and every raw
    parameter named in ``assign`` is replaced by an expression over it — so the
    fit explores the combination the data (or an independent measurement)
    actually constrains, instead of coordinates it does not resolve. The
    replaced parameters leave the free set automatically: ``bumps`` discovers
    parameters by traversing the model, and an expression is not one.

    *shared* is the cross-state cache. A declaration with ``tied`` (the default)
    resolves to ONE parameter object across every state, while ``assign`` is
    re-evaluated per state against that state's own namespace — which is
    precisely solvent-contrast variation: one invariant excess, a different
    derived SLD in each contrast. That relationship cannot be written as a
    ``shared_parameters`` entry, because the invariant is not a layer attribute.

    Raises ``ValueError`` on a declaration that does not describe a buildable
    reparametrization; the caller surfaces it rather than fitting something
    other than what was asked for.
    """
    from bumps.parameter import Parameter

    from .expressions import ExpressionError, canonical_name, evaluate
    from .expressions import evaluate_constraint

    specs = _derived_specs(definition, state_name)
    if not specs:
        return []

    # Every declared parameter exists before any assignment is evaluated, so
    # one reparametrization may be written in terms of another.
    created: Dict[str, Any] = {}
    for spec in specs:
        name = str(spec["name"])
        tied = bool(spec.get("tied", True))
        key = name if (tied or not state_name) else f"{state_name} {name}"
        par = (shared or {}).get(key)
        if par is None:
            free = spec.get("free") or {}
            init = float(free.get("init", free.get("value", 0.0)))
            par = Parameter(init, name=key)
            lo, hi = free.get("min"), free.get("max")
            if lo is not None and hi is not None:
                par.range(float(lo), float(hi))
            if shared is not None:
                shared[key] = par
        created[name] = par

    for spec in specs:
        for target, expr in (spec.get("assign") or {}).items():
            ns = _parameter_namespace(definition, sample, extra=created)
            try:
                value = evaluate(str(expr), ns)
            except ExpressionError as exc:
                raise ValueError(
                    f"derived parameter {spec['name']!r}: assignment to "
                    f"{target!r} is invalid — {exc}"
                ) from exc
            layer_name, _, _rest = canonical_name(str(target)).partition(".")
            attr_path = canonical_name(str(target)).split(".", 1)[1] if _rest else ""
            if not attr_path:
                raise ValueError(
                    f"derived parameter {spec['name']!r}: assignment target "
                    f"{target!r} must be '<layer>.<attr>'"
                )
            idx = _layer_index(
                definition,
                layer_name,
                back_reflection=definition.get("back_reflection", False),
            )
            if idx is None:
                raise ValueError(
                    f"derived parameter {spec['name']!r}: assignment target "
                    f"{target!r} names no layer in this "
                    + (f"state ({state_name})" if state_name else "model")
                )
            _set_layer_param(sample[idx], attr_path, value)

    constraints = []
    for spec in specs:
        for guard in spec.get("keep_physical") or []:
            ns = _parameter_namespace(definition, sample, extra=created)
            try:
                constraints.append(evaluate_constraint(str(guard), ns))
            except ExpressionError as exc:
                raise ValueError(
                    f"derived parameter {spec['name']!r}: keep_physical entry "
                    f"{guard!r} is invalid — {exc}"
                ) from exc
    return constraints


def validate_derived_parameters(definition: dict) -> None:
    """Check declarations against the model's structure, without building a fit.

    Everything here would otherwise surface as a mid-fit crash or, worse, as a
    silently different model. Called from the config layer so a typo is a
    startup error.
    """
    from .expressions import ExpressionError, referenced_names

    specs = definition.get("derived_parameters") or []
    if not specs:
        return
    if not isinstance(specs, list):
        raise ValueError("derived_parameters must be a list")

    structural = _valid_layer_names(definition)
    seen: set[str] = set()
    declared = {
        str(s.get("name")) for s in specs if isinstance(s, dict) and s.get("name")
    }
    state_names = {
        st.get("name") for st in (definition.get("states") or []) if st.get("name")
    }

    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("each derived_parameters entry must be a mapping")
        name = str(spec.get("name") or "").strip()
        if not name:
            raise ValueError("a derived parameter is missing `name`")
        if name in seen:
            raise ValueError(f"duplicate derived parameter name {name!r}")
        seen.add(name)
        if name in structural:
            raise ValueError(
                f"derived parameter {name!r} collides with a layer/material name"
            )
        free = spec.get("free") or {}
        lo, hi = free.get("min"), free.get("max")
        if lo is None or hi is None:
            raise ValueError(
                f"derived parameter {name!r}: `free` needs `min` and `max` "
                f"(a derived parameter has no bounds of its own to fall back on)"
            )
        if float(lo) >= float(hi):
            raise ValueError(f"derived parameter {name!r}: free.min must be < free.max")
        assign = spec.get("assign") or {}
        for scope_name in spec.get("states") or []:
            if state_names and scope_name not in state_names:
                raise ValueError(
                    f"derived parameter {name!r}: unknown state {scope_name!r}; "
                    f"known: {sorted(state_names)}"
                )
        known = structural | declared

        def _check(kind: str, text: str) -> None:
            try:
                refs = referenced_names(str(text))
            except ExpressionError as exc:
                raise ValueError(
                    f"derived parameter {name!r}: {kind} {text!r} — {exc}"
                ) from exc
            for ref in refs:
                if ref.split(".", 1)[0] not in known:
                    raise ValueError(
                        f"derived parameter {name!r}: {kind} {text!r} references "
                        f"unknown parameter {ref!r}; known: {sorted(known)}"
                    )

        for target, expr in assign.items():
            if str(target).split(".", 1)[0] not in structural:
                raise ValueError(
                    f"derived parameter {name!r}: assignment target {target!r} "
                    f"names no layer; known: {sorted(structural)}"
                )
            _check("assign", expr)
        for guard in spec.get("keep_physical") or []:
            _check("keep_physical", guard)

    # A declaration with no `assign` is an AUXILIARY parameter: free, but
    # reaching the model only through another declaration's expression. That is
    # how a two-parameter reparametrization is written — e.g. solvation, where
    # the volume fraction and the dry SLD are both free and only the layer's
    # SLD is derived from them. Legitimate, but only if something references
    # it; otherwise it is a free parameter the data cannot see, which would
    # wander the whole range and cost a parameter for nothing.
    referenced: set[str] = set()
    for spec in specs:
        for expr in (spec.get("assign") or {}).values():
            referenced |= referenced_names(str(expr))
        for guard in spec.get("keep_physical") or []:
            referenced |= referenced_names(str(guard))
    for spec in specs:
        name = str(spec.get("name"))
        if not (spec.get("assign") or {}) and name not in referenced:
            raise ValueError(
                f"derived parameter {name!r} has no `assign` and is referenced "
                f"by no other declaration — it would be a free parameter that "
                f"reaches nothing in the model"
            )


def data_chisq(problem) -> float:
    """Reduced χ² from the DATA term alone.

    ``FitProblem.chisq()`` scales the *total* nllf, which sums the model's
    misfit with the parameter-prior and constraint penalties
    (``_nllf_components`` → ``pmodel + pparameter + pconstraints``). With plain
    box bounds and no constraints those extra terms are identically zero and
    the two agree, which is why nothing noticed. The moment a reparametrization
    adds a ``keep_physical`` guard they do not: a violated constraint pushed χ²
    to ~10¹⁰ in testing, and that number drives the acceptance window
    (``chi2_min ≤ χ² ≤ chi2_max``), the regression guardrail and BIC.

    Those are all judgements about how well the model describes the
    measurement, so they must see the data term. The penalty is not discarded —
    the optimizer still minimizes the total — it is just not reported as
    goodness of fit.

    Falls back to ``problem.chisq()`` if bumps' internals move; the two are
    equal for every model that declares no priors or constraints.
    """
    try:
        _pparameter, _pconstraints, pmodel, failing = problem._nllf_components()
    except Exception:  # pragma: no cover - bumps internal rename
        logger.debug("[BUILDER] data_chisq fell back to problem.chisq()")
        return float(problem.chisq())
    if failing:
        # bumps short-circuits and returns pmodel = 0.0 without evaluating the
        # model when a prior or constraint is violated. Scaling that would
        # report a PERFECT fit for an infeasible one — and a χ² of 0 does not
        # merely look good, it lands under the acceptance floor and would be
        # filed as "the error bars are wrong" rather than "this model is not
        # allowed". Infeasible is the fit-failed sentinel.
        logger.info(
            "[BUILDER] infeasible parameters — constraint(s) violated: %s",
            "; ".join(str(f) for f in failing),
        )
        return float("inf")
    return float(problem.chisq(nllf=pmodel))


def penalty_nllf(problem) -> float:
    """The prior/constraint part of the nllf — what ``data_chisq`` leaves out.

    Reported alongside χ² so a fit dragged around by a ``keep_physical`` guard
    is visible rather than merely worse.
    """
    try:
        pparameter, pconstraints, _pmodel, _failing = problem._nllf_components()
        return float(pparameter) + float(pconstraints)
    except Exception:  # pragma: no cover - bumps internal rename
        return 0.0


def build_problem(definition: dict):
    """Construct a bumps ``FitProblem`` from a ``ModelDefinition`` dict.

    Parameters
    ----------
    definition
        A ``ModelDefinition`` dict.

    Returns
    -------
    problem : bumps.fitproblem.FitProblem
    """
    from bumps.fitproblem import FitProblem

    experiment = build_experiment(definition)
    constraints = apply_derived_parameters(definition, experiment.sample)
    return FitProblem(experiment, constraints=constraints or None)


def build_multi_problem(definition: dict, data_files: list[dict]):
    """Build a joint ``FitProblem`` for multi-file co-refinement.

    A *single* refl1d ``Sample`` is shared across multiple
    ``Experiment`` objects, each with its own ``Probe`` (loaded from a
    separate data file).  Because the ``Sample`` is shared, all
    structural parameters (thickness, SLD, roughness) are automatically
    tied — they are the same ``bumps.parameter.Parameter`` objects.
    Each probe gets its own independent intensity parameter.

    Parameters
    ----------
    definition
        A ``ModelDefinition`` dict (the model structure).
    data_files
        List of ``DatasetInfo`` dicts, each with ``"file"`` (path) and
        ``"label"`` (human-readable tag).

    Returns
    -------
    problem : bumps.fitproblem.FitProblem
        A ``FitProblem`` wrapping a list of experiments.
    experiments : list
        The individual ``Experiment`` objects (sorted by increasing Q).
    sorted_data_files : list
        The *data_files* list reordered to match *experiments*.
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="refl1d")
    from refl1d.names import Experiment
    from bumps.fitproblem import FitProblem

    sample = _build_sample(definition)
    intensity = definition.get("intensity", {})
    default_fwhm = definition.get("dq_is_fwhm", True)

    # Sort data files by minimum Q so experiments are in increasing Q order
    indexed = list(enumerate(data_files))
    probes = []
    for _i, ds in indexed:
        fwhm = ds.get("dq_is_fwhm", default_fwhm)
        theta = ds.get("theta", 0.0)
        if theta and theta > 0:
            probes.append(load_probe_from_angle(ds["file"], theta, dq_is_fwhm=fwhm))
        else:
            probes.append(load_probe(ds["file"], dq_is_fwhm=fwhm))

    sort_order = sorted(
        range(len(probes)),
        key=lambda k: float(probes[k].Q.min()) if len(probes[k].Q) else 0.0,
    )

    sorted_data_files = [data_files[k] for k in sort_order]
    sorted_probes = [probes[k] for k in sort_order]

    broadening = definition.get("sample_broadening", {})
    offset = definition.get("theta_offset", {})
    background = definition.get("background", {})

    # Shared sample_broadening / theta_offset / background parameters.  These
    # describe the *sample* / *measurement*, not the individual segment, so
    # they must be tied across all probes in a co-refinement.  We configure
    # the range on the first probe that exposes the attribute and then alias
    # the same bumps Parameter onto subsequent probes.
    shared_broadening = None
    shared_offset = None
    shared_background = None

    experiments = []
    for probe, ds in zip(sorted_probes, sorted_data_files):
        # Each probe gets its own independent intensity parameter
        if not intensity.get("fixed", False):
            int_val = intensity.get("value", 1.0)
            int_min = intensity.get("min", 0.7)
            int_max = intensity.get("max", 1.1)
            probe.intensity.value = int_val
            probe.intensity.range(int_min, int_max)

        # Give each probe's intensity a unique, stable name keyed by the
        # file label so it round-trips with the fit-result parameter names
        # produced by ``_extract_multi_bumps_results`` ("intensity <label>").
        # Without this, every probe's intensity is named "intensity", so
        # ``apply_parameters`` cannot match the per-file values sent by the
        # Results-page sliders and intensity overrides are silently dropped.
        label = ds.get("label") or ""
        if label and hasattr(probe, "intensity"):
            probe.intensity.name = f"intensity {label}"

        # sample_broadening / theta_offset only exist on NeutronProbe
        # (angle-based), not on QProbe (Q-based from load4).  Tie them
        # across probes so the fit uses a single shared parameter.
        if _nuisance_enabled(broadening) and hasattr(probe, "sample_broadening"):
            if shared_broadening is None:
                probe.sample_broadening.range(
                    broadening.get("min", 0.0), broadening.get("max", 0.5)
                )
                shared_broadening = probe.sample_broadening
            else:
                probe.sample_broadening = shared_broadening
        if _nuisance_enabled(offset) and hasattr(probe, "theta_offset"):
            if shared_offset is None:
                probe.theta_offset.range(
                    offset.get("min", -0.02), offset.get("max", 0.02)
                )
                shared_offset = probe.theta_offset
            else:
                probe.theta_offset = shared_offset

        # Single fittable flat background tied across all segments.
        if _nuisance_enabled(background) and hasattr(probe, "background"):
            if shared_background is None:
                _configure_background(probe, background)
                probe.background.name = "background"
                shared_background = probe.background
            else:
                probe.background = shared_background

        experiments.append(Experiment(probe=probe, sample=sample))

    constraints = apply_derived_parameters(definition, sample)
    problem = FitProblem(experiments, constraints=constraints or None)
    return problem, experiments, sorted_data_files


# ======================================================================
# Multi-state co-refinement (cross-state parameter aliasing)
# ======================================================================


# Default tied attributes when neither shared_parameters nor
# unshared_parameters is supplied. Each entry is a dotted suffix that is
# matched against the per-layer attribute path.
_DEFAULT_TIED_LAYER_ATTRS = ("thickness", "material.rho", "interface")
_DEFAULT_TIED_SUBSTRATE_ATTRS = ("interface",)

# refl1d's auto-generated parameter names use these short attr labels.
_ATTR_DISPLAY = {
    "thickness": "thickness",
    "material.rho": "rho",
    "interface": "interface",
}


def needs_states_problem(definition: dict) -> bool:
    """Return True when the model must be built via :func:`build_states_problem`.

    The multi-file path (:func:`build_multi_problem`) doesn't honor per-state
    ``theta_offset`` / ``sample_broadening`` / ``background`` blocks, so a
    single-state model with those nuisance parameters still needs the states
    route to wire them correctly. Multi-state models always need it.
    """
    states = definition.get("states") or []
    if len(states) > 1:
        return True
    return any(
        st.get("theta_offset") or st.get("sample_broadening") or st.get("background")
        for st in states
    )


def parameter_key(
    state_name: str, layer_name: str, attr_path: str, *, tied: bool
) -> str:
    """Canonical parameter name for a (state, layer, attr) triple.

    Tied parameters keep refl1d's default ``"<layer> <attr>"`` spelling
    so they appear once in the problem. Untied parameters are prefixed
    with the state name to ensure uniqueness across states.
    """
    display = _ATTR_DISPLAY.get(attr_path, attr_path.replace(".", " "))
    if tied:
        return f"{layer_name} {display}"
    return f"{state_name} {layer_name} {display}"


def _layer_index(
    definition: dict, layer_name: str, *, back_reflection: bool
) -> int | None:
    """Return the refl1d sample index for *layer_name*.

    Mirrors the indexing used in :func:`_build_sample`. Returns None
    when the name does not match any layer / substrate / ambient.
    """
    layers = definition.get("layers", []) or []
    n = len(layers)
    sub_name = (definition.get("substrate") or {}).get("name", "")
    amb_name = (definition.get("ambient") or {}).get("name", "")

    if layer_name == sub_name or layer_name.lower() == "substrate":
        return n + 1 if back_reflection else 0
    if layer_name == amb_name or layer_name.lower() == "ambient":
        return 0 if back_reflection else n + 1

    for i, layer in enumerate(layers):
        if layer["name"] == layer_name:
            return (n - i) if back_reflection else (i + 1)
    return None


def _valid_layer_names(definition: dict) -> set[str]:
    """Every layer name a tie spec may legitimately reference.

    The model-level layers plus each state's own layers (when a state overrides
    the structure), the substrate and ambient material names, and the literal
    aliases ``substrate``/``ambient``. Considering per-state layers makes this
    the UNION across states, so a layer present in only some states is still a
    valid tie target (the tie simply does not apply where the layer is absent).
    """
    names: set[str] = set()

    def _collect(d: dict) -> None:
        for layer in d.get("layers") or []:
            if isinstance(layer, dict) and layer.get("name"):
                names.add(layer["name"])
        sub = (d.get("substrate") or {}).get("name")
        if sub:
            names.add(sub)
        amb = (d.get("ambient") or {}).get("name")
        if amb:
            names.add(amb)

    _collect(definition)
    for st in definition.get("states") or []:
        _collect(st)  # per-state structure override (may add/remove layers)
    names.add("substrate")
    names.add("ambient")
    return names


def prune_tie_specs(definition: dict) -> list[str]:
    """Drop ``shared_/unshared_parameters`` entries whose layer is absent from
    the model's structure (model-level plus any per-state override), in place.

    A tie that references a layer no longer present — e.g. after a structural
    edit removed it — is meaningless and must not reach the fit, where
    :func:`_resolve_tied_set` would raise. Returns the dropped specs (for logging).
    """
    valid = _valid_layer_names(definition)
    dropped: list[str] = []
    for key in ("shared_parameters", "unshared_parameters"):
        specs = definition.get(key)
        if not specs:
            continue
        kept: list[str] = []
        for spec in specs:
            layer_name = spec.split(".", 1)[0] if "." in spec else spec
            (kept if layer_name in valid else dropped).append(spec)
        definition[key] = kept
    return dropped


def _resolve_tied_set(definition: dict) -> list[tuple[str, str]]:
    """Resolve the (layer_name, attr_path) pairs to alias across states.

    Reads ``shared_parameters`` (whitelist) or ``unshared_parameters``
    (blacklist) from *definition*. Either may be a list of strings of
    the form ``"<layer>.<attr_path>"`` (e.g. ``"Cu.thickness"``,
    ``"substrate.interface"``).
    """
    shared = definition.get("shared_parameters") or []
    unshared = definition.get("unshared_parameters") or []
    if shared and unshared:
        raise ValueError(
            "shared_parameters and unshared_parameters are mutually exclusive"
        )

    sub_name = (definition.get("substrate") or {}).get("name", "substrate")
    amb_name = (definition.get("ambient") or {}).get("name")

    # Normalize the literal aliases "substrate"/"ambient" to the actual
    # material names so tuple comparisons (block lookups, tied_lookup)
    # are consistent regardless of which spelling the user supplied.
    def _canon(layer_name: str) -> str:
        if layer_name.lower() == "substrate":
            return sub_name
        if layer_name.lower() == "ambient" and amb_name:
            return amb_name
        return layer_name

    # Build the default tied set over EVERY layer name across the model-level
    # stack and any per-state structure (the union). A layer present in >=2
    # states is tied across them; one present in a single state has nothing to
    # alias to and stays free (the build-time aliasing skips absent layers).
    union_names: list[str] = []
    seen_union: set[str] = set()
    for d in [definition, *(definition.get("states") or [])]:
        for layer in d.get("layers") or []:
            nm = layer.get("name") if isinstance(layer, dict) else None
            if nm and nm not in seen_union:
                seen_union.add(nm)
                union_names.append(nm)
    default: list[tuple[str, str]] = []
    for nm in union_names:
        for attr in _DEFAULT_TIED_LAYER_ATTRS:
            default.append((nm, attr))
    for attr in _DEFAULT_TIED_SUBSTRATE_ATTRS:
        default.append((sub_name, attr))

    def _split(spec: str) -> tuple[str, str]:
        if "." not in spec:
            raise ValueError(
                f"Parameter spec {spec!r} must be of the form '<layer>.<attr>'"
            )
        layer_name, attr = spec.split(".", 1)
        return layer_name, attr

    valid_layers = _valid_layer_names(definition)

    if shared:
        out: list[tuple[str, str]] = []
        for spec in shared:
            layer_name, attr = _split(spec)
            if layer_name not in valid_layers:
                raise ValueError(
                    f"shared_parameters references unknown layer {layer_name!r}; "
                    f"known: {sorted(valid_layers)}"
                )
            out.append((_canon(layer_name), attr))
        return out

    if unshared:
        block: set[tuple[str, str]] = set()
        for spec in unshared:
            layer_name, attr = _split(spec)
            if layer_name not in valid_layers:
                raise ValueError(
                    f"unshared_parameters references unknown layer {layer_name!r}; "
                    f"known: {sorted(valid_layers)}"
                )
            block.add((_canon(layer_name), attr))
        return [pair for pair in default if pair not in block]

    return default


def _get_layer_param(layer_obj, attr_path: str):
    """Resolve dotted *attr_path* (e.g. ``material.rho``) on a layer."""
    obj = layer_obj
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_layer_param(layer_obj, attr_path: str, value) -> None:
    """Assign *value* to the dotted *attr_path* on a layer."""
    parts = attr_path.split(".")
    target = layer_obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _state_overrides(definition: dict, state: dict) -> dict:
    """Return a per-state copy of *definition* with state-local overrides applied.

    Supports overrides for ``ambient``, ``back_reflection``, ``intensity``,
    and — when the state declares them — its own ``layers``/``substrate``
    (a full per-state structure; sample ≠ structure). A state without
    ``layers``/``substrate`` inherits the model-level template.

    The state's ``ambient`` is **merged** into the model-level ambient
    rather than replacing it, so a partial override (e.g. just the SLD)
    inherits ``name`` from the model. The UI uses the refl1d-side spelling
    ``rho`` for SLD; translate to the ModelDefinition's ``sld`` here so
    downstream code sees a uniform schema.
    """
    import copy

    eff = copy.deepcopy(definition)
    if state.get("ambient"):
        merged = dict(eff.get("ambient") or {})
        override = dict(state["ambient"])
        if "rho" in override and "sld" not in override:
            override["sld"] = override.pop("rho")
        merged.update(override)
        eff["ambient"] = merged
    if "back_reflection" in state:
        eff["back_reflection"] = bool(state["back_reflection"])
    if state.get("intensity"):
        eff["intensity"] = state["intensity"]
    # Per-state structure override: when a state declares its own layers (and
    # optionally substrate), use them as that state's complete stack so a layer
    # can be present in some states and absent in others.
    if state.get("layers") is not None:
        eff["layers"] = state["layers"]
    if state.get("substrate"):
        eff["substrate"] = state["substrate"]
    return eff


def build_states_problem(definition: dict):
    """Build a multi-state co-refinement ``FitProblem``.

    Each state in ``definition['states']`` becomes its own refl1d
    ``Sample`` and a set of ``Experiment`` objects (one per data file).
    Structural parameters listed in the resolved tied set are aliased
    across states by replacing the corresponding ``bumps.Parameter``
    objects with the ones from state 0.

    Returns
    -------
    problem : bumps.fitproblem.FitProblem
    experiments_by_state : dict[str, list]
        Mapping from state name to its list of refl1d ``Experiment``
        objects (sorted by increasing Q).
    sorted_data_files_by_state : dict[str, list]
        The per-state ``data_files`` lists reordered to match the
        experiments.
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="refl1d")
    from refl1d.names import Experiment
    from bumps.fitproblem import FitProblem

    states = definition.get("states") or []
    if not states:
        raise ValueError(
            "build_states_problem requires definition['states'] to be non-empty"
        )

    # Mixed back_reflection orientations across states cannot share
    # structural parameters cleanly: refl1d applies the substrate
    # roughness range on sample[0] in the normal stack and on
    # sample[n+1] in the back-reflection stack, so cross-state aliasing
    # of substrate.interface (and any same-attr layer pair whose range
    # is asymmetric) silently drops the range on one side.
    base_back = bool(definition.get("back_reflection", False))
    orientations = {bool(st.get("back_reflection", base_back)) for st in states}
    if len(orientations) > 1:
        raise ValueError(
            "build_states_problem: all states must share the same "
            "back_reflection orientation (mixed orientations would alias "
            "ranged and unranged parameters across states)"
        )

    tied_set = _resolve_tied_set(definition)

    # Cross-state cache for derived parameters. A tied declaration resolves to
    # one Parameter object for the whole problem while its `assign` is
    # re-evaluated per state — so the invariant (a surface excess, a volume
    # fraction) is shared and each state's derived SLD follows from its OWN
    # ambient. The `shared_parameters` mechanism cannot express that: it ties
    # layer attributes to each other, and the invariant is not one.
    shared_derived: Dict[str, Any] = {}
    derived_constraints: list = []
    # Per-state record of which slots a reparametrization owns. Consulted for
    # BOTH ends of every tie below: a parameter that is derived in *either*
    # state cannot be aliased to the other, in either direction.
    derived_slots_by_state: list[set] = []
    # Tie pairs that were NOT applied to a state because one end is derived.
    # The renaming pass below has to know: it keys off the tie SET, so without
    # this a parameter that is untied in fact keeps the tied spelling — and two
    # states that both fall back to a free parameter would then carry the same
    # name, silently colliding in the fitted-parameter dict.
    untied_by_derivation: Dict[int, set] = {}

    samples: list = []
    effective_defs: list[dict] = []
    experiments_by_state: dict[str, list] = {}
    sorted_files_by_state: dict[str, list] = {}
    all_experiments: list = []
    default_fwhm = definition.get("dq_is_fwhm", True)

    for state_idx, state in enumerate(states):
        eff = _state_overrides(definition, state)
        effective_defs.append(eff)
        sample = _build_sample(eff)
        samples.append(sample)
        st_name_for_derived = state.get("name", f"state{state_idx}")
        derived_slots_by_state.append(
            _derived_assigned_slots(eff, st_name_for_derived)
        )

        # Cross-state parameter aliasing.
        if state_idx > 0:
            ref_def = effective_defs[0]
            ref_sample = samples[0]
            ref_back = ref_def.get("back_reflection", False)
            cur_back = eff.get("back_reflection", False)
            for layer_name, attr_path in tied_set:
                ref_idx = _layer_index(ref_def, layer_name, back_reflection=ref_back)
                cur_idx = _layer_index(eff, layer_name, back_reflection=cur_back)
                derived_here = (
                    cur_idx is not None
                    and (cur_idx, attr_path) in derived_slots_by_state[state_idx]
                )
                # Aliasing the reference end is the subtler half: when a
                # declaration is scoped to state 0 only, tying the others to it
                # would hand them state 0's expression — resolving every
                # state's SLD against state 0's ambient, which is the opposite
                # of what a scoped reparametrization asked for.
                derived_at_ref = (
                    ref_idx is not None
                    and (ref_idx, attr_path) in derived_slots_by_state[0]
                )
                if derived_here or derived_at_ref:
                    if not derived_here:
                        untied_by_derivation.setdefault(state_idx, set()).add(
                            (layer_name, attr_path)
                        )
                    logger.debug(
                        "[STATES] tie %s.%s not applied to state %r — the "
                        "parameter is derived in %s",
                        layer_name,
                        attr_path,
                        state.get("name"),
                        "this state" if derived_here else "the reference state",
                    )
                    continue
                if ref_idx is None or cur_idx is None:
                    # The layer is absent from this state's stack (per-state
                    # structure, or a pruned spec) — the tie does not apply here.
                    # This is expected, not an error.
                    logger.debug(
                        "[STATES] tie %s.%s not applied to state %r — layer absent",
                        layer_name,
                        attr_path,
                        state.get("name"),
                    )
                    continue
                ref_param = _get_layer_param(ref_sample[ref_idx], attr_path)
                _set_layer_param(sample[cur_idx], attr_path, ref_param)

        # Applied after the tie aliasing above so the reparametrization wins
        # over any tie that survived the derived-slot filter.
        derived_constraints.extend(
            apply_derived_parameters(
                eff,
                sample,
                state_name=st_name_for_derived,
                shared=shared_derived,
            )
        )

        # Build probes for this state.
        data_files = state.get("data_files") or []
        if not data_files:
            raise ValueError(f"State {state.get('name')!r} has no data_files")

        intensity = eff.get("intensity", {}) or {}
        probes = []
        for ds in data_files:
            fwhm = ds.get("dq_is_fwhm", default_fwhm)
            theta = ds.get("theta", 0.0)
            if theta and theta > 0:
                probes.append(load_probe_from_angle(ds["file"], theta, dq_is_fwhm=fwhm))
            else:
                probes.append(load_probe(ds["file"], dq_is_fwhm=fwhm))

        order = sorted(
            range(len(probes)),
            key=lambda k: float(probes[k].Q.min()) if len(probes[k].Q) else 0.0,
        )
        sorted_files = [data_files[k] for k in order]
        sorted_probes = [probes[k] for k in order]

        # Per-state shared theta_offset / sample_broadening (partials) and
        # background (any data kind). Each is tied across the state's probes.
        broadening = state.get("sample_broadening") or definition.get(
            "sample_broadening", {}
        )
        offset = state.get("theta_offset") or definition.get("theta_offset", {})
        background = state.get("background") or definition.get("background", {})
        shared_broadening = None
        shared_offset = None
        shared_background = None

        state_experiments: list = []
        for ds, probe in zip(sorted_files, sorted_probes):
            # Per-file intensity override on the DatasetInfo takes
            # precedence over the state-level intensity. This is how
            # ``aure import-refl1d`` preserves the post-fit intensity
            # values for each segment in a single-state co-refinement,
            # where each probe carries its own intensity Parameter.
            ds_intensity = ds.get("intensity") or {}
            effective_intensity = {**intensity, **ds_intensity}
            if not effective_intensity.get("fixed", False):
                int_val = effective_intensity.get(
                    "value", effective_intensity.get("init", 1.0)
                )
                int_min = effective_intensity.get("min", 0.7)
                int_max = effective_intensity.get("max", 1.1)
                probe.intensity.value = int_val
                probe.intensity.range(int_min, int_max)

            if _nuisance_enabled(broadening) and hasattr(probe, "sample_broadening"):
                if shared_broadening is None:
                    init = broadening.get("init", broadening.get("value", 0.0))
                    probe.sample_broadening.value = init
                    probe.sample_broadening.range(
                        broadening.get("min", 0.0), broadening.get("max", 0.5)
                    )
                    shared_broadening = probe.sample_broadening
                else:
                    probe.sample_broadening = shared_broadening
            if _nuisance_enabled(offset) and hasattr(probe, "theta_offset"):
                if shared_offset is None:
                    init = offset.get("init", offset.get("value", 0.0))
                    probe.theta_offset.value = init
                    probe.theta_offset.range(
                        offset.get("min", -0.02), offset.get("max", 0.02)
                    )
                    shared_offset = probe.theta_offset
                else:
                    probe.theta_offset = shared_offset

            # Single fittable flat background tied across the state's probes.
            if _nuisance_enabled(background) and hasattr(probe, "background"):
                if shared_background is None:
                    _configure_background(probe, background)
                    shared_background = probe.background
                else:
                    probe.background = shared_background

            exp = Experiment(probe=probe, sample=sample)
            state_experiments.append(exp)

        name = state.get("name", f"state{state_idx}")
        experiments_by_state[name] = state_experiments
        sorted_files_by_state[name] = sorted_files
        all_experiments.extend(state_experiments)

    # ------------------------------------------------------------------
    # Rename untied parameters with a state prefix so every name in the
    # FitProblem is unique. Tied parameters keep their refl1d-default
    # "<layer> <attr>" spelling because the same Parameter object is
    # shared across states.
    # ------------------------------------------------------------------
    tied_lookup = set(tied_set)
    layers = definition.get("layers", []) or []
    sub_name = (definition.get("substrate") or {}).get("name", "substrate")

    for state_idx, state in enumerate(states):
        eff = effective_defs[state_idx]
        sample = samples[state_idx]
        st_name = state.get("name", f"state{state_idx}")
        back = eff.get("back_reflection", False)

        # Layer & substrate & ambient attributes.
        targets: list[tuple[str, str]] = []
        for layer in layers:
            for attr in _DEFAULT_TIED_LAYER_ATTRS:
                targets.append((layer["name"], attr))
        for attr in _DEFAULT_TIED_SUBSTRATE_ATTRS:
            targets.append((sub_name, attr))
        # Ambient SLD: always untied by default but rename for uniqueness.
        amb_name = (eff.get("ambient") or {}).get("name")
        if amb_name:
            targets.append((amb_name, "material.rho"))

        freed = untied_by_derivation.get(state_idx, set())
        derived_slots = (
            derived_slots_by_state[state_idx]
            if state_idx < len(derived_slots_by_state)
            else set()
        )
        for layer_name, attr_path in targets:
            if (layer_name, attr_path) in tied_lookup and (
                layer_name,
                attr_path,
            ) not in freed:
                continue  # shared with state 0 — keep default name
            idx = _layer_index(eff, layer_name, back_reflection=back)
            if idx is None:
                continue
            if (idx, attr_path) in derived_slots:
                # Derived here: the slot holds an expression, not a fittable
                # parameter, and has no name of its own to set.
                continue
            try:
                param = _get_layer_param(sample[idx], attr_path)
            except AttributeError:
                continue
            param.name = parameter_key(st_name, layer_name, attr_path, tied=False)

        # Per-probe intensity & per-state nuisance parameters.
        state_files = sorted_files_by_state[st_name]
        state_exps = experiments_by_state[st_name]
        multi_in_state = len(state_exps) > 1
        # `background` exists on every probe, so only rename it when this state
        # actually enabled one (theta_offset / sample_broadening only exist on
        # angle-based NeutronProbe, so their hasattr check already gates them).
        bg_enabled = _nuisance_enabled(
            state.get("background") or definition.get("background", {})
        )
        seen_nuisance: set[int] = set()
        for ds, exp in zip(state_files, state_exps):
            if hasattr(exp.probe, "intensity"):
                label = ds.get("label") or ""
                if multi_in_state and label:
                    exp.probe.intensity.name = f"{st_name} {label} intensity"
                else:
                    exp.probe.intensity.name = f"{st_name} intensity"
            for nuisance_attr in ("theta_offset", "sample_broadening", "background"):
                if nuisance_attr == "background" and not bg_enabled:
                    continue
                if not hasattr(exp.probe, nuisance_attr):
                    continue
                par = getattr(exp.probe, nuisance_attr)
                if id(par) in seen_nuisance:
                    continue
                seen_nuisance.add(id(par))
                par.name = f"{st_name} {nuisance_attr}"

    problem = FitProblem(
        all_experiments, constraints=derived_constraints or None
    )
    return problem, experiments_by_state, sorted_files_by_state


def save_problem_json(
    definition: dict,
    path,
    data_files: list[dict] | None = None,
) -> str:
    """Serialize a ``ModelDefinition`` to a bumps-compatible ``problem.json``.

    Builds a ``FitProblem`` via :func:`build_problem` (single experiment)
    or :func:`build_multi_problem` (multi-file co-refinement) and writes a
    JSON representation using ``bumps.serialize.save_file``. The resulting
    file can be loaded directly by refl1d / bumps (e.g. ``refl1d problem.json``)
    or submitted to a remote fit service.

    Parameters
    ----------
    definition
        A ``ModelDefinition`` dict.
    path
        Output file path (str or Path).
    data_files
        Optional list of ``DatasetInfo`` dicts for multi-file co-refinement.
        When provided with more than one entry, a multi-experiment
        ``FitProblem`` is built so all datasets share structural parameters
        while each probe gets its own intensity normalisation.

    Returns
    -------
    str
        The absolute path to the written file.
    """
    from bumps.serialize import save_file

    if definition.get("derived_parameters"):
        # bumps' serializer does not round-trip expression parameters (the same
        # limitation `roughness_tie` works around by re-applying the tie from
        # the ModelDefinition on every rebuild). A problem.json written here
        # would load as a DIFFERENT model — the derived parameters back to
        # free, the reparametrization gone, the constraints gone — and nothing
        # downstream would say so. Refuse rather than hand over a file that
        # quietly fits something else.
        raise ValueError(
            "cannot export problem.json: this model uses derived_parameters "
            f"({', '.join(str(d.get('name')) for d in definition['derived_parameters'])}), "
            "and bumps serialization does not preserve expression parameters. "
            "The exported problem would silently drop the reparametrization "
            "and its constraints. Run the fit through AuRE, or remove the "
            "derived parameters to export."
        )

    if needs_states_problem(definition):
        problem, _exps, _sorted = build_states_problem(definition)
    elif data_files and len(data_files) > 1:
        problem, _experiments, _sorted = build_multi_problem(definition, data_files)
    else:
        problem = build_problem(definition)

    out = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_file(out, problem)
    return out


def apply_parameters(problem, params: Dict[str, float]) -> None:
    """Apply fitted parameter values to a ``FitProblem`` by name.

    Accepts either the canonical name from :func:`parameter_key` or, for
    untied multi-state parameters, the legacy ``"<layer> <attr>"``
    spelling — in which case the value is broadcast to every parameter
    whose name ends with that suffix.
    """
    model_params = getattr(problem, "_parameters", None)
    if model_params is None:
        return

    from collections import defaultdict

    by_name: dict[str, list] = defaultdict(list)
    for par in model_params:
        by_name[str(par.name)].append(par)

    def _resolve(name: str) -> list:
        group = by_name.get(name)
        if group:
            return group
        # Fallback: legacy short name like "Cu thickness" matches every
        # state-prefixed name "<state> Cu thickness".
        suffix = " " + name
        return [
            par
            for key, plist in by_name.items()
            if key.endswith(suffix)
            for par in plist
        ]

    for name, value in params.items():
        group = _resolve(name)
        if not group:
            continue
        if len(group) == 1:
            group[0].value = float(value)
        else:
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


def apply_bounds(problem, bounds: Dict[str, list]) -> None:
    """Widen parameter bounds on a ``FitProblem`` to user-specified ranges."""
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


# ======================================================================
# Extract ModelDefinition from a fitted FitProblem
# ======================================================================


def extract_definition(
    problem,
    base_definition: dict,
    include_fitted: bool = True,
) -> dict:
    """Extract an updated ``ModelDefinition`` from a fitted ``FitProblem``.

    The returned dict is a copy of *base_definition* with layer SLD values,
    thicknesses, and roughnesses updated to the current best-fit values.

    Parameters
    ----------
    problem
        A bumps ``FitProblem`` that has been fitted.
    base_definition
        The ``ModelDefinition`` that was used to build the problem.
    include_fitted
        If *True*, also populate ``fitted_parameters`` and
        ``fitted_uncertainties`` keys.
    """
    import copy

    defn = copy.deepcopy(base_definition)

    params = getattr(problem, "_parameters", None)
    if params is None:
        return defn

    # Build name→value lookup
    fitted: Dict[str, float] = {}
    for par in params:
        fitted[str(par.name)] = par.value

    if include_fitted:
        defn["fitted_parameters"] = fitted

    states = defn.get("states") or []
    if len(states) > 1:
        # Multi-state: populate each state's layers view from the
        # per-state parameter names, falling back to the tied baseline.
        try:
            tied_set = set(_resolve_tied_set(defn))
        except ValueError:
            tied_set = set()

        attr_to_field = {
            "thickness": "thickness",
            "material.rho": "sld",
            "interface": "roughness",
        }

        for state in states:
            st_name = state.get("name", "")
            new_layers: list[dict] = []
            for layer in defn.get("layers", []):
                layer_name = layer["name"]
                merged = dict(layer)
                for attr_path, field in attr_to_field.items():
                    tied = (layer_name, attr_path) in tied_set or not tied_set
                    key = parameter_key(st_name, layer_name, attr_path, tied=tied)
                    if key in fitted:
                        merged[field] = fitted[key]
                    else:
                        # Try the alternate spelling.
                        alt = parameter_key(
                            st_name, layer_name, attr_path, tied=not tied
                        )
                        if alt in fitted:
                            merged[field] = fitted[alt]
                new_layers.append(merged)
            state["layers"] = new_layers
        # Also update the top-level layers from the tied baseline.
        for layer in defn.get("layers", []):
            layer_name = layer["name"]
            for attr_path, field in attr_to_field.items():
                key = parameter_key("", layer_name, attr_path, tied=True)
                if key in fitted:
                    layer[field] = fitted[key]
        return defn

    # Single-state legacy path
    for layer in defn.get("layers", []):
        layer_name = layer["name"]
        if f"{layer_name} thickness" in fitted:
            layer["thickness"] = fitted[f"{layer_name} thickness"]
        if f"{layer_name} rho" in fitted:
            layer["sld"] = fitted[f"{layer_name} rho"]
        if f"{layer_name} interface" in fitted:
            layer["roughness"] = fitted[f"{layer_name} interface"]
        elif f"{layer_name} rough_frac" in fitted:
            # Tied roughness (sigma = fraction * thickness): the interface is a
            # derived parameter, not fitted directly. Record the resulting
            # numeric roughness; the `roughness_tie` metadata (preserved by the
            # deepcopy above) re-applies the tie on the next rebuild, since
            # bumps serialization does not round-trip expression ties.
            layer["roughness"] = fitted[f"{layer_name} rough_frac"] * layer["thickness"]

    return defn


# ======================================================================
# Helpers
# ======================================================================


def definition_from_parsed_sample(
    parsed_sample: dict,
    data_file: str,
) -> dict:
    """Convert a ``ParsedSample`` (from intake) to a ``ModelDefinition``.

    This bridges the intake output to the model representation.
    """
    intensity_raw = parsed_sample.get("intensity", {})
    return {
        "substrate": parsed_sample["substrate"],
        "layers": parsed_sample.get("layers", []),
        "ambient": parsed_sample["ambient"],
        "constraints": parsed_sample.get("constraints", []),
        "back_reflection": parsed_sample.get("back_reflection", False),
        "data_file": os.path.abspath(data_file),
        "intensity": {
            "value": intensity_raw.get("value", 1.0),
            "min": intensity_raw.get("min", 0.7),
            "max": intensity_raw.get("max", 1.1),
            "fixed": intensity_raw.get("fixed", False),
        },
    }
