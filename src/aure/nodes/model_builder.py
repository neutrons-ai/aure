"""
Model builder: construct refl1d Experiment / FitProblem from a ModelDefinition.

This module replaces the old approach of storing models as Python scripts.
Instead, models are stored as structured JSON (ModelDefinition dicts) and
refl1d objects are built on-the-fly when needed for fitting or visualisation.

Key functions:
- build_states_problem() — JSON → bumps FitProblem. The one builder: one state
  or several, with the per-state experiments it returns alongside.
- build_problem() / build_multi_problem() / build_experiment() — views onto
  it for callers that want just the problem, one state's experiments, or a
  single curve's experiment. A definition with no ``states`` block is given
  the one state it implies (with_states()).
- extract_definition() — fitted FitProblem → updated ModelDefinition
"""

import logging
import os
from typing import Dict

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

    Notes
    -----
    ``.ort`` support is refl1d's, and in 1.0.1 its loader accepts only files
    that carry the incident angle *and* the wavelength as tagged data columns,
    each with a matching ``ErrorColumn``. A file stating them once in the
    header — the natural form for a reduced curve — raises ``AttributeError``
    from inside ``load4``. See ``tests/test_orso_probe.py``, whose strict
    xfails will fail once that is fixed upstream.
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


# Fallbacks when a theta_offset / sample_broadening block omits a value.
_ANGLE_NUISANCE_DEFAULTS = {
    "theta_offset": {"init": 0.0, "min": -0.02, "max": 0.02},
    "sample_broadening": {"init": 0.0, "min": 0.0, "max": 0.5},
}


def _configure_angle_nuisance(probe, attr: str, spec: dict) -> None:
    """Set ``probe.<attr>`` (``theta_offset`` / ``sample_broadening``).

    The same contract as :func:`_configure_background`: the value is always
    applied, and ``fixed: true`` withholds the range. These two used to range
    unconditionally, so a parameter declared fixed was fitted anyway. The
    caller gates on :func:`_nuisance_enabled` and on the probe having *attr*
    (only the angle-based ``NeutronProbe`` does).
    """
    defaults = _ANGLE_NUISANCE_DEFAULTS[attr]
    par = getattr(probe, attr)
    par.value = spec.get("init", spec.get("value", defaults["init"]))
    if not spec.get("fixed", False):
        par.range(spec.get("min", defaults["min"]), spec.get("max", defaults["max"]))


def _configure_intensity(probe, spec) -> None:
    """Set ``probe.intensity`` from an intensity block.

    The declared value (``value``, else ``init``) is applied whether or not the
    intensity is fitted; ``fixed: true`` only withholds the range. Applying it
    in the fitted branch alone left every fixed intensity at the probe's 1.0,
    so an imported fit held at 0.95 was rebuilt at 1.0.
    """
    spec = spec or {}
    value = spec.get("value", spec.get("init"))
    if spec.get("fixed", False):
        if value is not None:
            probe.intensity.value = value
        return
    probe.intensity.value = 1.0 if value is None else value
    probe.intensity.range(spec.get("min", 0.7), spec.get("max", 1.1))


def with_states(definition: dict, data_files: list | None = None) -> dict:
    """*definition*, carrying the one state it implies when it has none.

    :func:`build_states_problem` is the only builder. A model with no
    ``states`` block — a checkpoint written before every model carried its
    states, or a definition assembled by hand — describes exactly one state,
    and this is where it is given one: *data_files* when the caller passes the
    files being co-refined, otherwise the model's own ``data_file``. The
    synthesized state carries no overrides, so every model-level field
    (``ambient``, ``intensity``, nuisance blocks, ``dq_is_fwhm``) applies as
    written.

    A definition that already has states is returned untouched, and
    *data_files* is ignored: its states own their files. Never mutates
    *definition*.
    """
    if definition.get("states"):
        return definition
    files = list(data_files) if data_files else [{"file": definition["data_file"]}]
    return {**definition, "states": [{"name": _SYNTHESIZED_STATE, "data_files": files}]}


#: Name of the state :func:`with_states` synthesizes; the same name the CLI
#: gives the single state it builds from a positional data file.
_SYNTHESIZED_STATE = "state0"


def build_experiment(definition: dict):
    """Construct the refl1d ``Experiment`` of a single-curve model.

    A view onto :func:`build_states_problem` for the one shape that has a
    single experiment: one state with one file (a states-less definition is
    that, by :func:`with_states`).

    Raises
    ------
    ValueError
        When the model has more than one curve — ask
        :func:`build_states_problem` for its experiments instead.
    """
    _problem, by_state, _files = build_states_problem(with_states(definition))
    experiments = [exp for group in by_state.values() for exp in group]
    if len(experiments) != 1:
        raise ValueError(
            f"build_experiment needs a single-curve model; this one has "
            f"{len(experiments)} curves — use build_states_problem"
        )
    return experiments[0]


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
    declared = None
    if layers_info:
        try:
            declared = float(
                layers_info[-1].get("roughness_max", _OUTER_ROUGHNESS_MAX_DEFAULT)
            )
        except (TypeError, ValueError):
            declared = None

    override = os.environ.get("ROUGHNESS_MAX_OUTER")
    if override:
        try:
            forced = float(override)
        except ValueError:
            logger.warning(
                "[MODEL] Ignoring non-numeric ROUGHNESS_MAX_OUTER=%r", override
            )
        else:
            # The override REPLACES the model's ceiling rather than widening it,
            # and that is deliberate: on the reference corpus the outer
            # roughness runs to the cap in 108 of 835 fits where the expert
            # value is 46-200 Å, so the cap is a regularizer the fits lean on.
            # Say when it displaces a declaration, though — otherwise a model
            # asking for 400 Å gets 250 with nothing anywhere admitting it.
            if declared is not None and abs(declared - forced) > 1e-9:
                logger.info(
                    "[MODEL] outer interface ceiling %g Å from "
                    "ROUGHNESS_MAX_OUTER, displacing the %g Å declared on %r. "
                    "To set it from the model, name the interface explicitly "
                    "(see ModelDefinition.interfaces).",
                    forced,
                    declared,
                    (layers_info[-1] or {}).get("name", "the outermost layer"),
                )
            return forced

    return declared if declared is not None else _OUTER_ROUGHNESS_MAX_DEFAULT


def outer_ceiling_displacement(definition: dict) -> tuple | None:
    """``(forced, declared, layer_name)`` when the env displaces a declared
    outer ceiling; ``None`` otherwise.

    The displacement itself is deliberate — see :func:`_outer_roughness_max` —
    but it is invisible from inside a run: the model declares one ceiling and
    the fit uses another. `evaluation` reports this as an issue so the refiner
    is told its declaration had no effect, instead of re-raising it every
    iteration against a bound it cannot move.
    """
    override = os.environ.get("ROUGHNESS_MAX_OUTER")
    if not override:
        return None
    layers = definition.get("layers") or []
    if not layers:
        return None
    try:
        forced = float(override)
        declared = float(layers[-1]["roughness_max"])
    except (KeyError, TypeError, ValueError):
        return None
    if abs(forced - declared) <= 1e-9:
        return None
    return forced, declared, str(layers[-1].get("name", "the outermost layer"))


#: Lower bound applied to a layer interface when the model declares none.
#: Never allowed to exceed the layer's own declared roughness — see
#: :func:`_ranged`.
_ROUGHNESS_MIN_DEFAULT = 5.0


#: The parameters a model block can hold constant, the same way for each:
#: ``key -> (min key, max key, flag key)``. A layer has all three; the ambient
#: has ``sld``; the substrate has ``roughness`` (flag only — it declares no
#: ``roughness_min``, so it has no window to collapse).
PINNABLE = {
    "sld": ("sld_min", "sld_max", "sld_fixed"),
    "thickness": ("thickness_min", "thickness_max", "thickness_fixed"),
    "roughness": ("roughness_min", "roughness_max", "roughness_fixed"),
}


def is_pinned(block: dict, key: str) -> bool:
    """Whether *block* (a layer, the ambient or the substrate) holds *key*.

    Two spellings mean the same thing. ``<key>_fixed: true`` is the stated
    one. A zero-width window, ``<key>_min == <key>_max``, is the one models
    already wrote before the flag existed, and it stays honoured. Either way
    the parameter is set and not fitted: it never enters the problem's free
    parameters, so it is neither counted against the model in BIC nor
    reported as sitting on its bounds for something to widen.
    """
    lo_key, hi_key, flag = PINNABLE[key]
    if block.get(flag):
        return True
    lo, hi = block.get(lo_key), block.get(hi_key)
    return lo is not None and hi is not None and lo == hi


def _ranged(
    param,
    value,
    lo,
    hi,
    *,
    label: str,
    declared_lo: bool = True,
    fixed: bool = False,
) -> None:
    """Set a parameter's value and range, keeping the value inside the range.

    A parameter that is *held* gets its value and no range, which is what
    keeps it out of the fit: ``fixed=True`` (the ``<key>_fixed`` flag — the
    declared value stands, whatever window sits next to it), or a window that
    collapses to a point. A zero-width range handed to the optimizer is still
    a free parameter — one more in BIC's count, pinned against both bounds.

    A parameter built outside its own bounds makes the whole problem
    infeasible: bumps short-circuits, the model is never evaluated, and the fit
    reports `inf` before it starts. Nothing in the model JSON looks wrong, so
    the cause is invisible from the outside.

    Two ways that happens, handled differently because the intent differs:

    * A **built-in default** conflicts with a value the model explicitly
      declared — a layer given ``roughness: 3.0`` against the 5 Å interface
      floor. The declared value is the statement of intent, so the default
      yields to it silently; a default that overrides what the user wrote is
      not a default.
    * A **declared bound** excludes the declared value. That is the model
      contradicting itself, so the value is clamped into the range and the
      contradiction is logged rather than passed to the optimizer.
    """
    if fixed:
        param.value = value
        param.fixed = True  # even if an earlier pass gave it a range
        return
    if lo > hi:
        logger.warning("[BUILDER] %s: min %g > max %g — swapping", label, lo, hi)
        lo, hi = hi, lo
    if value < lo:
        if declared_lo:
            logger.warning(
                "[BUILDER] %s: starting value %g is below its declared minimum "
                "%g — clamped to the bound",
                label,
                value,
                lo,
            )
            value = lo
        else:
            lo = value  # a default must not contradict a declared value
    elif value > hi:
        logger.warning(
            "[BUILDER] %s: starting value %g is above its maximum %g — clamped",
            label,
            value,
            hi,
        )
        value = hi
    param.value = value
    if lo == hi:
        param.fixed = True  # a window of one point is a pin, not a range
        return
    param.range(lo, hi)


def _interface_owner_index(
    position: int, n_layers: int, *, back_reflection: bool
) -> int:
    """Slab index carrying the boundary above *position* in SAMPLE order.

    Sample order is substrate-first: ``-1`` is the substrate, ``0..n-1`` the
    layers, ``n`` the ambient. The boundary "above position p" is the one
    between p and p+1.

    A slab's ``interface`` is its boundary with whatever sits above it *in the
    stack*, and the stack is assembled ambient-first in back reflection — so
    the slab that carries a given physical boundary is the lower of the two in
    STACK order, which is the substrate-side material normally and the
    ambient-side material in back reflection.
    """
    if back_reflection:
        # stack = [ambient, L(n-1), ..., L0, substrate]; the ambient-side
        # material of the pair sits lower.
        upper = position + 1
        return 0 if upper >= n_layers else n_layers - upper
    # stack = [substrate, L0, ..., L(n-1), ambient]
    return position + 1


def _log_interface_map(
    layers_info: list, substrate_info: dict, ambient_info: dict, back_reflection: bool
) -> None:
    """Record which declared roughness feeds which physical boundary.

    The mapping is not what a reader of the model JSON would assume. A layer's
    ``roughness`` is its boundary with whatever sits above it in the *stack*,
    and back reflection assembles the stack ambient-first — so the same
    declaration describes a different interface depending on the geometry, the
    substrate's is discarded, and the outermost layer's is used twice (once for
    its own buried boundary, once for the borrowed outer surface). Working that
    out from the code takes a while; this makes it readable off a run.

    Logged at INFO for back reflection, where the mapping is surprising and
    lossy, and at DEBUG for a normal stack, where it is the obvious one.
    """
    if not layers_info:
        return
    names = [layer.get("name", f"layer{i}") for i, layer in enumerate(layers_info)]
    sub = substrate_info.get("name", "substrate")
    amb = ambient_info.get("name", "ambient")
    chain = [sub, *names, amb]

    pieces = []
    for position in range(-1, len(names)):
        lower, upper = chain[position + 1], chain[position + 2]
        if back_reflection:
            # The ambient-side material's slab carries the boundary, and the
            # ambient has no roughness of its own — the OUTER boundary borrows
            # the outermost layer's declaration.
            if position + 1 >= len(names):
                pieces.append(
                    f"{names[-1]}.roughness -> {lower}/{upper} (outer, borrowed)"
                )
                continue
            owner = names[position + 1]
        else:
            owner = sub if position < 0 else names[position]
        pieces.append(f"{owner}.roughness -> {lower}/{upper}")
    if back_reflection:
        pieces.append(f"{sub}.roughness -> (unused)")
        logger.info("[BUILDER] back-reflection interface map: %s", "; ".join(pieces))
    else:
        pieces.append(f"{amb} interface -> (unused)")
        logger.debug("[BUILDER] interface map: %s", "; ".join(pieces))


def _apply_interface_declarations(definition: dict, sample) -> None:
    """Apply an optional ``interfaces`` block, overriding the positional map.

    A layer's ``roughness`` describes its boundary with whatever sits above it
    in the *stack*, and back reflection assembles the stack ambient-first — so
    which physical interface a declaration describes depends on the geometry
    (see :func:`_log_interface_map`). That is self-consistent, and it matches
    the convention the reference refl1d fits were built with, so it is not
    changed here.

    What it cannot do is let someone *name* an interface. "The interface
    between the copper and the titanium is 3 nm" has no unambiguous home: the
    layer that owns it differs between geometries, and the substrate's own
    declaration is discarded in back reflection, so one boundary is unreachable
    altogether.

    An ``interfaces`` entry names the two materials the boundary separates, in
    sample order (substrate side first), and is therefore geometry-independent:

        interfaces:
          - {below: silicon, above: SiO2, roughness: 3.0, roughness_max: 20.0}

    Absent — which is every model today — this is a no-op and the positional
    mapping stands untouched.
    """
    declared = definition.get("interfaces") or []
    if not declared:
        return

    layers_info = definition.get("layers") or []
    back_reflection = definition.get("back_reflection", False)
    names = [layer.get("name", f"layer{i}") for i, layer in enumerate(layers_info)]
    sub = (definition.get("substrate") or {}).get("name", "substrate")
    amb = (definition.get("ambient") or {}).get("name", "ambient")
    chain = [sub, *names, amb]  # sample order: substrate first

    seen: dict[int, str] = {}
    for entry in declared:
        if not isinstance(entry, dict):
            raise ValueError("each `interfaces` entry must be a mapping")
        below, above = entry.get("below"), entry.get("above")
        if not below or not above:
            raise ValueError(
                f"interface entry {entry!r} needs both `below` and `above` "
                f"(the two materials it separates, substrate side first)"
            )
        try:
            i_below, i_above = chain.index(str(below)), chain.index(str(above))
        except ValueError:
            raise ValueError(
                f"interface {below!r}/{above!r} names a material not in this "
                f"model; known, in sample order: {chain}"
            ) from None
        if i_above != i_below + 1:
            raise ValueError(
                f"interface {below!r}/{above!r} is not a boundary: those "
                f"materials are not adjacent. Sample order is {chain}, and "
                f"`below` must be the substrate-side one."
            )

        position = i_below - 1  # sample-order position: -1 is the substrate
        idx = _interface_owner_index(
            position, len(names), back_reflection=back_reflection
        )
        if idx in seen:
            raise ValueError(
                f"interfaces declares {below!r}/{above!r} and {seen[idx]} for "
                f"the same boundary"
            )
        seen[idx] = f"{below!r}/{above!r}"

        # `roughness_tie` replaces the interface with an expression, which has
        # no value to set. It is the profile-artifact remedy and more specific
        # than a declared number, so it wins — loudly.
        if not hasattr(sample[idx].interface, "range"):
            logger.warning(
                "[BUILDER] interface %s/%s ignored: that interface is tied to "
                "its layer thickness (roughness_tie)",
                below,
                above,
            )
            continue

        value = entry.get("roughness")
        if value is None:
            raise ValueError(f"interface {below!r}/{above!r} declares no `roughness`")
        declared_lo = "roughness_min" in entry
        _ranged(
            sample[idx].interface,
            float(value),
            float(entry.get("roughness_min", _ROUGHNESS_MIN_DEFAULT)),
            float(entry.get("roughness_max", 30.0)),
            label=f"{below}/{above} interface",
            declared_lo=declared_lo,
            fixed=bool(entry.get("roughness_fixed")),
        )
        logger.info(
            "[BUILDER] interface %s/%s set explicitly: roughness=%g (slab %d)",
            below,
            above,
            float(value),
            idx,
        )


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

    # A roughness on the AMBIENT is ignored, in both geometries, so say so. An
    # interface belongs to the slab below it, and the outer surface is
    # therefore the outermost layer's: in a normal stack that layer's own slab
    # carries it, and in back reflection the ambient slab borrows that same
    # declared value (see the stack assembly below). Either way nothing reads
    # `ambient["roughness"]`, and a declaration that vanishes without comment
    # is worse than one that is refused.
    _rough_keys = ("roughness", "roughness_min", "roughness_max")
    _stray = [k for k in _rough_keys if k in ambient_info]
    if _stray:
        owner = layers_info[-1].get("name") if layers_info else None
        logger.warning(
            "[BUILDER] ambient %r declares %s — ignored. The outer surface is "
            "owned by %s; set its roughness instead.",
            ambient_info.get("name", "ambient"),
            ", ".join(_stray),
            f"the outermost layer ({owner})" if owner else "the outermost layer",
        )

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
        # TODO(bug): the multiplicative defaults invert for a NEGATIVE ambient
        # SLD — H2O (-0.56) yields min=-0.448 > max=-0.672. Fixing the defaults
        # is a behaviour change for every run with a negative ambient (H2O is
        # one of the two canonical contrast-variation media), so it is left for
        # its own change; `_ranged` swaps the bounds so the fit is at least not
        # built backwards.
        amb_min = ambient_info.get("sld_min", max(amb_sld * 0.8, -1.0))
        amb_max = ambient_info.get("sld_max", amb_sld * 1.2)
        amb_idx = 0 if back_reflection else len(layers_info) + 1
        _ranged(
            sample[amb_idx].material.rho,
            amb_sld,
            amb_min,
            amb_max,
            label=f"{ambient_info.get('name', 'ambient')} rho",
            fixed=bool(ambient_info.get("sld_fixed")),
        )

    for i, layer in enumerate(layers_info):
        if back_reflection:
            idx = len(layers_info) - i
        else:
            idx = i + 1

        t_min = layer.get("thickness_min", layer["thickness"] * 0.5)
        t_max = layer.get("thickness_max", layer["thickness"] * 2.0)
        _ranged(
            sample[idx].thickness,
            layer["thickness"],
            t_min,
            t_max,
            label=f"{layer['name']} thickness",
            fixed=bool(layer.get("thickness_fixed")),
        )

        sld_min = layer.get("sld_min", layer["sld"] - 2.5)
        sld_max = layer.get("sld_max", layer["sld"] + 2.5)
        _ranged(
            sample[idx].material.rho,
            layer["sld"],
            sld_min,
            sld_max,
            label=f"{layer['name']} rho",
            fixed=bool(layer.get("sld_fixed")),
        )

        tie = layer.get("roughness_tie")
        if tie and is_pinned(layer, "roughness"):
            # Held and tied contradict each other: one says the roughness is a
            # known number, the other that it follows the thickness. The
            # declared constant wins, as it does over a cross-state tie.
            logger.warning(
                "[BUILDER] %s: roughness is held at %g, so its roughness_tie "
                "is ignored",
                layer["name"],
                layer["roughness"],
            )
            tie = None
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
            declared_r_min = "roughness_min" in layer
            r_min = layer.get("roughness_min", _ROUGHNESS_MIN_DEFAULT)
            r_max = layer.get("roughness_max", 30.0)
            _ranged(
                sample[idx].interface,
                layer["roughness"],
                r_min,
                r_max,
                label=f"{layer['name']} interface",
                declared_lo=declared_r_min,
                fixed=bool(layer.get("roughness_fixed")),
            )

    # The bottom slab's interface. In a normal stack that is the substrate's
    # own roughness. In back reflection it is the outer surface, on the
    # ambient slab, which borrows the outermost layer's declared roughness —
    # one declared value behind two parameters, so holding that value holds
    # both.
    if back_reflection:
        bottom_held = bool(layers_info) and is_pinned(layers_info[-1], "roughness")
        bottom_max = _outer_roughness_max(layers_info)
    else:
        bottom_held = False  # the substrate's own roughness: held just below
        bottom_max = substrate_info.get("roughness_max", 15.0)
    if bottom_held:
        sample[0].interface.fixed = True
    else:
        sample[0].interface.range(0, bottom_max)

    # A held substrate roughness names a physical boundary — substrate / first
    # layer — and is applied to the slab that carries it. In a normal stack
    # that is the substrate slab. In back reflection it is the bottom layer's
    # slab: left on the substrate slab the declaration would hold nothing,
    # because that slab's interface is unused there.
    if is_pinned(substrate_info, "roughness"):
        owner = _interface_owner_index(
            -1, len(layers_info), back_reflection=back_reflection
        )
        value = float(substrate_info.get("roughness", 3.0))
        if hasattr(sample[owner].interface, "range"):
            sample[owner].interface.value = value
            sample[owner].interface.fixed = True
        else:
            # That layer's roughness_tie made the interface an expression.
            # Held wins over tied.
            from bumps.parameter import Parameter

            sample[owner].interface = Parameter(
                value, name=f"{sample[owner].material.name} interface"
            )
        if owner != 0:
            logger.info(
                "[BUILDER] substrate roughness held at %g: applied to the %s "
                "slab, which carries the substrate boundary in back reflection",
                value,
                sample[owner].material.name,
            )

    _apply_interface_declarations(definition, sample)
    _log_interface_map(layers_info, substrate_info, ambient_info, back_reflection)

    return sample


#: ``X.rho`` and ``X.sld`` are accepted as spellings of refl1d's
#: ``X.material.rho``. Tie specs are written by hand, and all three spellings
#: appear in the wild, so they have to canonicalize to one parameter.
_ATTR_ALIASES = {"rho": "material.rho", "sld": "material.rho"}


def canonical_name(name: str) -> str:
    """Canonicalize a dotted parameter name (``X.rho`` -> ``X.material.rho``)."""
    layer, _, attr = name.partition(".")
    if not attr:
        return name
    return f"{layer}.{_ATTR_ALIASES.get(attr, attr)}"


def data_chisq(problem) -> float:
    """Reduced χ² from the DATA term alone.

    ``FitProblem.chisq()`` scales the *total* nllf, which sums the model's
    misfit with the parameter-prior and constraint penalties
    (``_nllf_components`` → ``pmodel + pparameter + pconstraints``). Every
    model AuRE builds today declares plain box bounds and no constraints, so
    those extra terms are identically zero and the two agree.

    This is kept separate anyway, because the distinction is real and the
    conflation is silent. A single ``bumps`` ``Constraint`` anywhere in the
    problem makes ``chisq()`` a number about the penalty rather than about the
    data — a violated one measured ~10¹⁰ in testing — and that number drives
    the acceptance window (``chi2_min ≤ χ² ≤ chi2_max``), the regression
    guardrail and BIC. Those are all judgements about how well the model
    describes the measurement, so they must see the data term, and nothing
    downstream would notice if they stopped. Reporting the data term costs one
    call and cannot be wrong.

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

    Reported alongside χ² so a fit dragged around by a prior or a bumps
    constraint is visible rather than merely worse.
    """
    try:
        pparameter, pconstraints, _pmodel, _failing = problem._nllf_components()
        return float(pparameter) + float(pconstraints)
    except Exception:  # pragma: no cover - bumps internal rename
        return 0.0


def data_chisq_total(problem) -> float:
    """The **un-normalized** χ² of the data term: ``Σ((R − R_model)/dR)²``.

    This is what BIC needs. ``data_chisq`` reports the *reduced* χ² — divided by
    the degrees of freedom — which is the right number to judge fit quality
    against 1, and the wrong number to put in an information criterion, where
    the likelihood term must scale with the amount of data.

    bumps defines ``pmodel = ½·Σresiduals²`` for Gaussian independent
    uncertainties (see ``FitProblem.chisq_str``), so the total is exactly
    ``2·pmodel``. Deriving it this way deliberately avoids reconstructing it as
    ``χ²_reduced × dof``: bumps' ``dof`` also folds in prior degrees of freedom
    (``model_points() + Σ prior.dof − n_params``) and silently degrades to an
    un-normalized scale when it is non-positive, so the round trip is not an
    identity.

    Infeasible parameters return ``inf``, matching ``data_chisq`` — see the
    reasoning there.
    """
    try:
        _pparameter, _pconstraints, pmodel, failing = problem._nllf_components()
    except Exception:  # pragma: no cover - bumps internal rename
        # chisq(norm=False) scales the nllf by exactly 2, giving the total χ²
        # including the penalty terms. Only reachable if bumps renames its
        # internals, in which case an over-inclusive total beats no BIC at all.
        logger.debug("[BUILDER] data_chisq_total fell back to chisq(norm=False)")
        try:
            return float(problem.chisq(norm=False))
        except Exception:
            return float("inf")
    if failing:
        return float("inf")
    return 2.0 * float(pmodel)


def bic_inputs(problem) -> dict:
    """The three quantities BIC needs, read off the problem itself.

    Returns ``{_chi2_total, _n_data, _n_free_params}``. Every fit path stamps
    these onto its ``FitResult`` so that ``evaluation`` scores a fit from the
    problem that produced it rather than re-deriving ``n`` and ``k`` from the
    model dict. Re-deriving them is what let ``fitting`` and ``evaluation``
    disagree: ``state["Q"]`` holds only the primary data file, so a
    co-refinement was scored against one file's point count.

    ``model_points()`` and ``getp()`` are the same accessors bumps uses to build
    its own ``dof``, so ``n`` counts every point of every dataset of every state
    and ``k`` counts unique free parameters — tied-across-state parameters once,
    expression-derived ones not at all.
    """
    try:
        return {
            "_chi2_total": data_chisq_total(problem),
            "_n_data": int(problem.model_points()),
            "_n_free_params": len(problem.getp()),
        }
    except Exception:  # pragma: no cover - bumps internal rename
        logger.debug("[BUILDER] bic_inputs unavailable", exc_info=True)
        return {}


def build_problem(definition: dict):
    """Construct the bumps ``FitProblem`` for a ``ModelDefinition`` dict.

    Whatever the model's shape — one curve, several files of one state,
    several states — this is :func:`build_states_problem` without the
    per-state bookkeeping. A definition with no ``states`` block is built as
    the single state it describes (:func:`with_states`).
    """
    return build_states_problem(with_states(definition))[0]


def build_multi_problem(definition: dict, data_files: list[dict]):
    """Build the joint ``FitProblem`` for several files of one sample.

    A view onto :func:`build_states_problem`: *data_files* become the one
    state of a states-less definition (:func:`with_states`), which shares a
    single refl1d ``Sample`` across its experiments — so every structural
    parameter is tied — and gives each probe its own intensity.

    Parameters
    ----------
    definition
        A ``ModelDefinition`` dict with a single state, or none. One that
        carries its own state is built from that state's files; *data_files*
        is then ignored.
    data_files
        List of ``DatasetInfo`` dicts, each with ``"file"`` (path) and
        ``"label"`` (human-readable tag).

    Returns
    -------
    problem : bumps.fitproblem.FitProblem
    experiments : list
        The individual ``Experiment`` objects (sorted by increasing Q).
    sorted_data_files : list
        The files reordered to match *experiments*.

    Raises
    ------
    ValueError
        When the definition carries several states — that is
        :func:`build_states_problem`'s return shape, not this one's.
    """
    problem, by_state, files_by_state = build_states_problem(
        with_states(definition, data_files)
    )
    if len(by_state) != 1:
        raise ValueError(
            f"build_multi_problem needs a single-state model; this one has "
            f"{len(by_state)} states — use build_states_problem"
        )
    (name,) = by_state
    return problem, by_state[name], files_by_state[name]


# ======================================================================
# Multi-state co-refinement (cross-state parameter aliasing)
# ======================================================================


# Default tied attributes when neither shared_parameters nor
# unshared_parameters is supplied. Each entry is a dotted suffix that is
# matched against the per-layer attribute path.
_DEFAULT_TIED_LAYER_ATTRS = ("thickness", "material.rho", "interface")
_DEFAULT_TIED_SUBSTRATE_ATTRS = ("interface",)

#: Every attribute a tie spec may name, after canonicalization. Anything else
#: would reach refl1d as a bare ``getattr`` on a Slab.
_TIEABLE_ATTRS = frozenset(_DEFAULT_TIED_LAYER_ATTRS) | frozenset(
    _DEFAULT_TIED_SUBSTRATE_ATTRS
)

# refl1d's auto-generated parameter names use these short attr labels.
_ATTR_DISPLAY = {
    "thickness": "thickness",
    "material.rho": "rho",
    "interface": "interface",
}


def intensity_parameter_name(
    state_name: str, label: str, *, single_state: bool, multi_file: bool
) -> str:
    """Name of one probe's intensity parameter in a states problem.

    One state is spelled ``"intensity <label>"`` — a state prefix
    distinguishes nothing when there is one state, and this is the name every
    single-curve run has always reported. Several states prefix
    it, with the file label added when a state has more than one file.
    """
    if single_state:
        return f"intensity {label}" if label else "intensity"
    if multi_file and label:
        return f"{state_name} {label} intensity"
    return f"{state_name} intensity"


def nuisance_parameter_name(state_name: str, attr: str, *, single_state: bool) -> str:
    """Name of a state's shared ``theta_offset`` / ``sample_broadening`` /
    ``background``: bare for one state, state-prefixed for several."""
    return attr if single_state else f"{state_name} {attr}"


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
        """Split ``"<layer>.<attr>"``, canonicalizing and checking the attribute.

        The attribute is canonicalized by :func:`canonical_name`, so
        ``Cu.rho`` / ``Cu.sld`` / ``Cu.material.rho`` all name one parameter.
        Tie specs are hand-written and all three spellings appear in the wild;
        ``Cu.rho`` used to pass the layer-name check and then reach refl1d as
        ``getattr(slab, "rho")`` — `'Slab' object has no attribute 'rho'`,
        thrown from inside the builder long after the mistake.

        An attribute that is not tie-able at all is rejected here rather than
        deep in the build, so the modeling node's fallback (carry the previous
        tie spec) can catch it instead of the run ending.
        """
        if "." not in spec:
            raise ValueError(
                f"Parameter spec {spec!r} must be of the form '<layer>.<attr>'"
            )
        layer_name, attr = canonical_name(spec).split(".", 1)
        if attr not in _TIEABLE_ATTRS:
            raise ValueError(
                f"Parameter spec {spec!r} names an attribute that cannot be tied; "
                f"expected one of {sorted(_TIEABLE_ATTRS)} "
                f"(`rho`/`sld` are accepted as synonyms for `material.rho`)"
            )
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


def _is_held(param) -> bool:
    """A built parameter that is set and not fitted (not an expression)."""
    return bool(getattr(param, "fixed", False)) and hasattr(param, "range")


def _tie_across_states(
    states: list, effective_defs: list, samples: list, tied_set: list
) -> None:
    """Make every tied parameter one object across the states that have it.

    A tie says a parameter is the same quantity in every state; a held
    parameter says that quantity is known. Both are declarations, and they
    combine instead of one silently replacing the other:

    * if **any** tied state holds the parameter, it is held in all of them, at
      that value — not only when the holding state happens to be the first;
    * if two tied states hold it at **different** values, they are not one
      quantity and the declaration contradicts itself: refused, naming both.

    Otherwise the first state's parameter is shared, as before. A layer absent
    from the first state is not tied (and one absent from a later state is
    simply skipped there) — per-state structure, not an error.
    """
    import math

    ref_def, ref_sample = effective_defs[0], samples[0]
    ref_back = ref_def.get("back_reflection", False)

    for layer_name, attr_path in tied_set:
        ref_idx = _layer_index(ref_def, layer_name, back_reflection=ref_back)
        if ref_idx is None:
            logger.debug(
                "[STATES] tie %s.%s not applied — layer absent from state %r",
                layer_name,
                attr_path,
                states[0].get("name"),
            )
            continue

        members = [(0, ref_sample[ref_idx])]
        for i in range(1, len(states)):
            eff = effective_defs[i]
            idx = _layer_index(
                eff, layer_name, back_reflection=eff.get("back_reflection", False)
            )
            if idx is None:
                logger.debug(
                    "[STATES] tie %s.%s not applied to state %r — layer absent",
                    layer_name,
                    attr_path,
                    states[i].get("name"),
                )
                continue
            members.append((i, samples[i][idx]))

        params = [(i, _get_layer_param(slab, attr_path)) for i, slab in members]
        held = [(i, par) for i, par in params if _is_held(par)]
        shared = held[0][1] if held else params[0][1]

        for i, par in held[1:]:
            if not math.isclose(par.value, shared.value, rel_tol=1e-9, abs_tol=1e-12):
                where = ", ".join(
                    f"{states[j].get('name', f'state{j}')}={p.value:g}" for j, p in held
                )
                raise ValueError(
                    f"{layer_name}.{attr_path} is tied across states but held at "
                    f"different values ({where}). A tie says it is one quantity: "
                    f"untie it (unshared_parameters) or make the held values agree."
                )
        if held and len(held) < len(params):
            logger.info(
                "[STATES] %s.%s is held at %g in state %r and tied, so it is "
                "held in every tied state",
                layer_name,
                attr_path,
                shared.value,
                states[held[0][0]].get("name"),
            )

        for _i, slab in members:
            _set_layer_param(slab, attr_path, shared)


def check_state_ties(definition: dict) -> None:
    """Raise ``ValueError`` if *definition*'s ties and held values contradict.

    Builds each state's sample (no data file is read) and runs the same tie
    pass :func:`build_states_problem` does, so a setup that holds one tied
    parameter at two values is refused when the model is made rather than when
    the first fit is built. No-op for fewer than two states.
    """
    states = definition.get("states") or []
    if len(states) < 2:
        return
    effective_defs = [_state_overrides(definition, st) for st in states]
    samples = [_build_sample(eff) for eff in effective_defs]
    _tie_across_states(states, effective_defs, samples, _resolve_tied_set(definition))


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

    # One state has nothing to be tied to, and nothing to be told apart from:
    # tie specs are not consulted and no parameter takes a state prefix.
    single_state = len(states) == 1
    tied_set = [] if single_state else _resolve_tied_set(definition)

    samples: list = []
    effective_defs: list[dict] = []
    experiments_by_state: dict[str, list] = {}
    sorted_files_by_state: dict[str, list] = {}
    all_experiments: list = []
    default_fwhm = definition.get("dq_is_fwhm", True)

    for state in states:
        eff = _state_overrides(definition, state)
        effective_defs.append(eff)
        samples.append(_build_sample(eff))

    # Cross-state parameter aliasing, before any experiment sees a sample.
    _tie_across_states(states, effective_defs, samples, tied_set)

    for state_idx, state in enumerate(states):
        eff = effective_defs[state_idx]
        sample = samples[state_idx]

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
            _configure_intensity(probe, {**intensity, **ds_intensity})

            if _nuisance_enabled(broadening) and hasattr(probe, "sample_broadening"):
                if shared_broadening is None:
                    _configure_angle_nuisance(probe, "sample_broadening", broadening)
                    shared_broadening = probe.sample_broadening
                else:
                    probe.sample_broadening = shared_broadening
            if _nuisance_enabled(offset) and hasattr(probe, "theta_offset"):
                if shared_offset is None:
                    _configure_angle_nuisance(probe, "theta_offset", offset)
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
    # shared across states. With a single state nothing can collide, so no
    # name takes a prefix and the problem reads like a flat single-sample one.
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
        # Ambient SLD and its interface: always untied by default, so both need
        # a state prefix. The interface is easy to forget because it is a fitted
        # parameter only in BACK REFLECTION — there the ambient is sample[0] and
        # carries the outer interface — but when it is one, leaving it unnamed
        # gives every state a parameter called "<ambient> interface". Same name,
        # different objects: they collide in the fitted-parameter dict and in
        # every other place keyed by name.
        amb_name = (eff.get("ambient") or {}).get("name")
        if amb_name:
            targets.append((amb_name, "material.rho"))
            targets.append((amb_name, "interface"))

        for layer_name, attr_path in targets:
            if single_state or (layer_name, attr_path) in tied_lookup:
                continue  # one state, or shared with state 0 — keep default name
            idx = _layer_index(eff, layer_name, back_reflection=back)
            if idx is None:
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
                if single_state and not label:
                    label = os.path.splitext(os.path.basename(ds.get("file", "")))[0]
                exp.probe.intensity.name = intensity_parameter_name(
                    st_name,
                    label,
                    single_state=single_state,
                    multi_file=multi_in_state,
                )
            for nuisance_attr in ("theta_offset", "sample_broadening", "background"):
                if nuisance_attr == "background" and not bg_enabled:
                    continue
                if not hasattr(exp.probe, nuisance_attr):
                    continue
                par = getattr(exp.probe, nuisance_attr)
                if id(par) in seen_nuisance:
                    continue
                seen_nuisance.add(id(par))
                par.name = nuisance_parameter_name(
                    st_name, nuisance_attr, single_state=single_state
                )

    problem = FitProblem(all_experiments)
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

    multi = data_files if data_files and len(data_files) > 1 else None
    problem = build_problem(with_states(definition, multi))

    out = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_file(out, problem)
    return out


def _resolve_parameters(by_name: dict, name: str) -> list:
    """Parameters of a problem that the stored *name* refers to.

    Exact match first. Two spellings from other shapes of the same model are
    then accepted, so values stored under one still land on a rebuild:

    * a short ``"<layer> <attr>"`` name, broadcast to every state-prefixed
      parameter ending in it (a flat result applied to a multi-state problem);
    * a state-prefixed name on a problem that carries no prefix — what
      single-state runs with a nuisance key, and ``import-refl1d`` workspaces,
      stored before a single state stopped prefixing its parameters. Accepted
      only when it identifies exactly one parameter;
    * ``"background <probe>"``, refl1d's default, which the flat single-file
      builder used to leave in place where every other path says
      ``"background"``.
    """
    group = by_name.get(name)
    if group:
        return group

    suffix = " " + name
    group = [
        par for key, plist in by_name.items() if key.endswith(suffix) for par in plist
    ]
    if group:
        return group

    if name.startswith("background ") and "background" in by_name:
        # The flat single-file builder left refl1d's "background <probe>" name.
        return by_name["background"]

    if name.split()[-1:] == ["intensity"]:
        # "<state> intensity" / "<state> <label> intensity" -> "intensity <label>"
        head = name[: -len("intensity")].strip()
        keys = [k for k in by_name if k == "intensity" or k.startswith("intensity ")]
        if len(keys) > 1:
            keys = [
                k
                for k in keys
                if head == k[len("intensity ") :]
                or head.endswith(" " + k[len("intensity ") :])
            ]
    else:
        keys = [k for k in by_name if name.endswith(" " + k)]
        if len(keys) > 1:
            keys = [max(keys, key=len)]
    if len(keys) == 1 and len(by_name[keys[0]]) == 1:
        return by_name[keys[0]]
    return []


def _parameters_by_name(problem) -> dict:
    from collections import defaultdict

    by_name: dict[str, list] = defaultdict(list)
    for par in getattr(problem, "_parameters", None) or []:
        by_name[str(par.name)].append(par)
    return by_name


def apply_parameters(problem, params: Dict[str, float]) -> None:
    """Apply fitted parameter values to a ``FitProblem`` by name.

    Names are matched by :func:`_resolve_parameters`: exactly, or through the
    spellings another shape of the same model used. A name matching several
    parameters is applied to those whose bounds admit the value.
    """
    by_name = _parameters_by_name(problem)
    if not by_name:
        return

    for name, value in params.items():
        group = _resolve_parameters(by_name, name)
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
    """Widen parameter bounds on a ``FitProblem`` to user-specified ranges.

    Every parameter carrying the exact name is widened. Failing that, the name
    is resolved as :func:`_resolve_parameters` does — e.g. the state-prefixed
    spelling an older single-state run stored — and used only when it
    identifies exactly one parameter.
    """
    from bumps.bounds import init_bounds

    by_name = _parameters_by_name(problem)
    for name, (lo_new, hi_new) in bounds.items():
        group = by_name.get(name)
        if not group:
            group = _resolve_parameters(by_name, name)
            if len(group) != 1:
                continue
        for par in group:
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
