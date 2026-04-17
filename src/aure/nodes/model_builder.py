"""
Model builder: construct refl1d Experiment / FitProblem from a ModelDefinition.

This module replaces the old approach of storing models as Python scripts.
Instead, models are stored as structured JSON (ModelDefinition dicts) and
refl1d objects are built on-the-fly when needed for fitting or visualisation.

Key functions:
- build_experiment()  — JSON → refl1d Experiment
- build_problem()     — JSON → bumps FitProblem
- extract_definition() — fitted FitProblem → updated ModelDefinition
- export_model_script() — JSON → human-readable Python script
"""

import logging
import os
from typing import Dict, List, Optional

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
        int_min = intensity.get("min", 0.7)
        int_max = intensity.get("max", 1.1)
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
        probes.append(load_probe(ds["file"], dq_is_fwhm=fwhm))

    sort_order = sorted(
        range(len(probes)),
        key=lambda k: float(probes[k].Q.min()) if len(probes[k].Q) else 0.0,
    )

    sorted_data_files = [data_files[k] for k in sort_order]
    sorted_probes = [probes[k] for k in sort_order]

    experiments = []
    for probe in sorted_probes:
        # Each probe gets its own independent intensity parameter
        if not intensity.get("fixed", False):
            int_min = intensity.get("min", 0.7)
            int_max = intensity.get("max", 1.1)
            probe.intensity.range(int_min, int_max)

        experiments.append(Experiment(probe=probe, sample=sample))

    problem = FitProblem(experiments)
    return problem, experiments, sorted_data_files


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
# Export a Python script from ModelDefinition
# ======================================================================


def export_model_script(
    definition: dict,
    fitted_params: Optional[Dict[str, float]] = None,
    fitted_uncertainties: Optional[Dict[str, float]] = None,
    chi_squared: Optional[float] = None,
    method: Optional[str] = None,
    include_ranges: bool = True,
) -> str:
    """Generate a human-readable refl1d Python script from a ModelDefinition.

    Parameters
    ----------
    definition
        A ``ModelDefinition`` dict.
    fitted_params
        If provided, substitute these values into the script.
    fitted_uncertainties
        If provided, add uncertainty comments to the header.
    chi_squared
        If provided, add to the header.
    method
        Fitting method name for header.
    include_ranges
        If *True*, include ``.range()`` calls.  If *False*, comment them
        out (for ``model_final.py`` style output).

    Returns
    -------
    script : str
        A complete, executable refl1d Python script.
    """
    substrate = definition["substrate"]
    ambient = definition["ambient"]
    layers = definition.get("layers", [])
    data_file = definition.get("data_file", "")
    back_reflection = definition.get("back_reflection", False)
    intensity = definition.get("intensity", {})
    dq_is_fwhm = definition.get("dq_is_fwhm", True)

    abs_data_file = os.path.abspath(data_file) if data_file else data_file

    # Use fitted values if provided, falling back to definition values
    params = fitted_params or definition.get("fitted_parameters", {})

    lines: List[str] = []

    # Header
    if chi_squared is not None:
        lines.extend(
            [
                "# " + "=" * 68,
                f"# Best-fit result (chi2 = {chi_squared:.4f}, method = {method or 'unknown'})",
                "#",
                "# Parameter values below are the optimised values from the fit.",
            ]
        )
        if not include_ranges:
            lines.append(
                "# .range() constraints have been removed; each line shows the"
            )
            lines.append("# original range as a comment for reference.")
        if fitted_uncertainties:
            lines.append("#")
            lines.append("# Uncertainties (1-sigma):")
            for pname, unc in fitted_uncertainties.items():
                lines.append(f"#   {pname}: \u00b1{unc:.4f}")
        lines.append("# " + "=" * 68)
        lines.append("")

    lines.extend(
        [
            '"""',
            "Auto-generated refl1d model.",
            '"""',
            "",
            "import warnings",
            "from refl1d.names import *",
            "",
            'warnings.filterwarnings("ignore", category=UserWarning, module="refl1d")',
            "",
            "# ========== Load Data ==========",
            f'probe = load4("{abs_data_file}", FWHM={dq_is_fwhm})',
            "",
            "# ========== Materials ==========",
        ]
    )

    # Substrate SLD (use fitted value if available)
    sub_sld = params.get(f"{substrate['name']} rho", substrate["sld"])
    lines.append(f'substrate = SLD(name="{substrate["name"]}", rho={sub_sld:.4f})')

    # Ambient SLD
    amb_sld = params.get(f"{ambient['name']} rho", ambient["sld"])
    lines.append(f'ambient = SLD(name="{ambient["name"]}", rho={amb_sld:.4f})')

    for i, layer in enumerate(layers):
        mat_sld = params.get(f"{layer['name']} rho", layer["sld"])
        lines.append(
            f'material{i + 1} = SLD(name="{layer["name"]}", rho={mat_sld:.4f})'
        )

    lines.extend(["", "# ========== Sample Structure =========="])

    # Build sample stack
    if back_reflection:
        lines.append("# Neutrons come from substrate side (back reflection)")
        lines.append(
            "# Stack ordered in beam direction: ambient -> layers -> substrate"
        )
        roughness_first = layers[-1]["roughness"] if layers else 3.0
        r_first = (
            params.get(f"{layers[-1]['name']} interface", roughness_first)
            if layers
            else 3.0
        )
        stack_parts = [f"ambient(0, {r_first:.1f})"]
        for i in reversed(range(len(layers))):
            layer = layers[i]
            t = params.get(f"{layer['name']} thickness", layer["thickness"])
            r = params.get(f"{layer['name']} interface", layer["roughness"])
            stack_parts.append(f"material{i + 1}({t:.1f}, {r:.1f})")
        stack_parts.append("substrate")
    else:
        lines.append("# Built from substrate (bottom) to ambient (top)")
        sub_rough = params.get(
            f"{substrate['name']} interface", substrate.get("roughness", 3.0)
        )
        stack_parts = [f"substrate(0, {sub_rough:.1f})"]
        for i, layer in enumerate(layers):
            t = params.get(f"{layer['name']} thickness", layer["thickness"])
            r = params.get(f"{layer['name']} interface", layer["roughness"])
            stack_parts.append(f"material{i + 1}({t:.1f}, {r:.1f})")
        stack_parts.append("ambient")

    lines.append(f"sample = {' | '.join(stack_parts)}")
    lines.extend(["", "# ========== Fit Parameters =========="])

    def _range_line(target: str, lo: float, hi: float) -> str:
        if include_ranges:
            return f"{target}.range({lo:.2f}, {hi:.2f})"
        return f"# {target}.range({lo:.2f}, {hi:.2f})"

    # Ambient SLD range
    if ambient.get("name", "").lower() != "air" and ambient.get("sld", 0) != 0:
        amb_sld_val = ambient["sld"]
        amb_min = max(amb_sld_val * 0.8, -1.0)
        amb_max = amb_sld_val * 1.2
        amb_idx = 0 if back_reflection else len(layers) + 1
        lines.append(_range_line(f"sample[{amb_idx}].material.rho", amb_min, amb_max))

    for i, layer in enumerate(layers):
        idx = (len(layers) - i) if back_reflection else (i + 1)

        t_min = layer.get("thickness_min", layer["thickness"] * 0.5)
        t_max = layer.get("thickness_max", layer["thickness"] * 2.0)
        lines.append(_range_line(f"sample[{idx}].thickness", t_min, t_max))

        sld_min = layer.get("sld_min", layer["sld"] - 2.5)
        sld_max = layer.get("sld_max", layer["sld"] + 2.5)
        lines.append(_range_line(f"sample[{idx}].material.rho", sld_min, sld_max))

        r_max = layer.get("roughness_max", 30.0)
        lines.append(_range_line(f"sample[{idx}].interface", 0, r_max))

    # First-element roughness
    if back_reflection:
        lines.append(_range_line("sample[0].interface", 0, 30.0))
    else:
        sub_rough_max = substrate.get("roughness_max", 15.0)
        lines.append(_range_line("sample[0].interface", 0, sub_rough_max))

    # Probe intensity
    if not intensity.get("fixed", False):
        int_min = intensity.get("min", 0.7)
        int_max = intensity.get("max", 1.1)
        lines.extend(
            [
                "",
                "# ========== Probe Intensity ===========",
                "# Allow intensity to vary to account for normalization uncertainty",
                _range_line("probe.intensity", int_min, int_max),
            ]
        )

    lines.extend(
        [
            "",
            "# ========== Experiment ==========",
            "experiment = Experiment(probe=probe, sample=sample)",
            "problem = FitProblem(experiment)",
        ]
    )

    return "\n".join(lines)


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


def is_legacy_script(model: object) -> bool:
    """Return *True* if *model* looks like a legacy Python-script string."""
    return isinstance(model, str)
