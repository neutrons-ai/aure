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

    experiment = Experiment(probe=probe, sample=sample)
    return experiment


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

        r_min = layer.get("roughness_min", 5.0)
        r_max = layer.get("roughness_max", 30.0)
        sample[idx].interface.range(r_min, r_max)

    if back_reflection:
        sample[0].interface.range(0, 30.0)
    else:
        sub_rough_max = substrate_info.get("roughness_max", 15.0)
        sample[0].interface.range(0, sub_rough_max)

    return sample


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
    return FitProblem(experiment)


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

    experiments = []
    for probe in sorted_probes:
        # Each probe gets its own independent intensity parameter
        if not intensity.get("fixed", False):
            int_val = intensity.get("value", 1.0)
            int_min = intensity.get("min", 0.7)
            int_max = intensity.get("max", 1.1)
            probe.intensity.value = int_val
            probe.intensity.range(int_min, int_max)

        # sample_broadening / theta_offset only exist on NeutronProbe
        # (angle-based), not on QProbe (Q-based from load4).
        if broadening.get("enabled") and hasattr(probe, "sample_broadening"):
            probe.sample_broadening.range(
                broadening.get("min", 0.0), broadening.get("max", 0.5)
            )
        if offset.get("enabled") and hasattr(probe, "theta_offset"):
            probe.theta_offset.range(offset.get("min", -0.02), offset.get("max", 0.02))

        experiments.append(Experiment(probe=probe, sample=sample))

    problem = FitProblem(experiments)
    return problem, experiments, sorted_data_files


def save_problem_json(definition: dict, path) -> str:
    """Serialize a ``ModelDefinition`` to a bumps-compatible ``problem.json``.

    Builds a ``FitProblem`` via :func:`build_problem` and writes a JSON
    representation using ``bumps.serialize.save_file``.  The resulting file
    can be loaded directly by refl1d / bumps (e.g. ``refl1d problem.json``)
    or submitted to a remote fit service.

    Parameters
    ----------
    definition
        A ``ModelDefinition`` dict.
    path
        Output file path (str or Path).

    Returns
    -------
    str
        The absolute path to the written file.
    """
    from bumps.serialize import save_file

    problem = build_problem(definition)
    out = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_file(out, problem)
    return out


def apply_parameters(problem, params: Dict[str, float]) -> None:
    """Apply fitted parameter values to a ``FitProblem`` by name.

    Uses the same name-matching logic as the old ``_apply_fitted_parameters``
    but built to work with problems constructed from ``build_problem``.
    """
    model_params = getattr(problem, "_parameters", None)
    if model_params is None:
        return

    from collections import defaultdict

    by_name: dict[str, list] = defaultdict(list)
    for par in model_params:
        by_name[str(par.name)].append(par)

    for name, value in params.items():
        group = by_name.get(name)
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

    # Update layer values from fitted parameters
    for i, layer in enumerate(defn.get("layers", [])):
        layer_name = layer["name"]
        if f"{layer_name} thickness" in fitted:
            layer["thickness"] = fitted[f"{layer_name} thickness"]
        if f"{layer_name} rho" in fitted:
            layer["sld"] = fitted[f"{layer_name} rho"]
        if f"{layer_name} interface" in fitted:
            layer["roughness"] = fitted[f"{layer_name} interface"]

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
