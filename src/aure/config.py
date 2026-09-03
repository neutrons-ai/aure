"""
User-supplied YAML configuration for analysis constraints and criteria.

An optional YAML file (``--config config.yaml``) lets the user inject:

* **evaluation_criteria** – extra rules the LLM should apply when judging
  whether a fit is acceptable (e.g. "titanium thickness must be 45–55 Å").
* **model_constraints** – hard constraints the LLM must respect when
  building or refining a model (e.g. "do not add extra layers").
* **states** – list of measurement states for multi-state co-refinement
  (each grouping data files that share one physical sample).
* **shared_parameters** / **unshared_parameters** – whitelist or
  blacklist of layer attributes tied across states. Mutually exclusive.
* **derived_parameters** – reparametrization: fit a combination of raw
  parameters (a surface excess, a volume fraction) and derive a raw one
  from it. See ``aure.state.DerivedParameter``. Gated by
  **allow_derived_parameters** (default off) / ``ALLOW_DERIVED_PARAMETERS``.

See ``aure_config.example.yaml`` in the repository root for the full schema.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class UserConfig(TypedDict, total=False):
    """User-supplied analysis configuration."""

    evaluation_criteria: List[str]
    model_constraints: List[str]
    sample_description: Optional[str]
    states: List[dict]  # StateDefinition-shaped dicts (resolved & validated)
    shared_parameters: List[str]
    unshared_parameters: List[str]
    distinct_sample: bool  # co-refined states are distinct physical samples
    derived_parameters: List[dict]  # reparametrization (DerivedParameter-shaped)
    allow_derived_parameters: bool  # opt-in gate for the above (default False)


_EMPTY: UserConfig = {
    "evaluation_criteria": [],
    "model_constraints": [],
    "sample_description": None,
    "states": [],
    "shared_parameters": [],
    "unshared_parameters": [],
    "distinct_sample": False,
    "derived_parameters": [],
    "allow_derived_parameters": False,
}


class ConfigError(ValueError):
    """Raised on invalid user-config YAML."""


def load_user_config(path: Optional[str | Path] = None) -> UserConfig:
    """Load a YAML configuration file.

    Parameters
    ----------
    path
        Path to the YAML file.  If *None* or the file does not exist an
        empty config is returned.

    Returns
    -------
    UserConfig
        Validated configuration dictionary.

    Raises
    ------
    ConfigError
        If the YAML defines a ``states:`` block that fails validation
        (missing files, duplicate names, mixed file kinds, mutually
        exclusive whitelist/blacklist, etc.).  Malformed
        ``evaluation_criteria`` / ``model_constraints`` are silently
        coerced — only the new multi-state schema is strictly validated.
    """
    if path is None:
        return _empty_config()

    p = Path(path)
    if not p.exists():
        logger.warning("[CONFIG] Config file not found: %s", p)
        return _empty_config()

    import yaml

    raw: Dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    logger.info("[CONFIG] Loaded user config from %s", p)

    cfg: UserConfig = _empty_config()
    cfg["evaluation_criteria"] = _as_str_list(raw.get("evaluation_criteria"))
    cfg["model_constraints"] = _as_str_list(raw.get("model_constraints"))

    desc = (
        raw.get("sample_description") or raw.get("description") or raw.get("describe")
    )
    cfg["sample_description"] = str(desc) if desc else None

    cfg["shared_parameters"] = _as_str_list(raw.get("shared_parameters"))
    cfg["unshared_parameters"] = _as_str_list(raw.get("unshared_parameters"))
    cfg["derived_parameters"] = _parse_derived_parameters(
        raw.get("derived_parameters")
    )
    cfg["allow_derived_parameters"] = derived_parameters_enabled(
        raw.get("allow_derived_parameters")
    )
    check_derived_parameters_allowed(cfg, source=str(path))
    if cfg["shared_parameters"] and cfg["unshared_parameters"]:
        raise ConfigError(
            "shared_parameters and unshared_parameters are mutually exclusive; "
            "provide at most one."
        )

    cfg["distinct_sample"] = bool(raw.get("distinct_sample", False))

    cfg["states"] = _parse_states(raw.get("states"), base_dir=p.parent)

    if cfg["evaluation_criteria"]:
        logger.info(
            "[CONFIG]   %d evaluation criteria", len(cfg["evaluation_criteria"])
        )
    if cfg["model_constraints"]:
        logger.info("[CONFIG]   %d model constraints", len(cfg["model_constraints"]))
    if cfg["states"]:
        kinds = ", ".join(f"{s['name']}({s['_kind']})" for s in cfg["states"])
        logger.info("[CONFIG]   %d states: %s", len(cfg["states"]), kinds)

    return cfg


def _empty_config() -> UserConfig:
    return {  # type: ignore[return-value]
        "evaluation_criteria": [],
        "model_constraints": [],
        "sample_description": None,
        "states": [],
        "shared_parameters": [],
        "unshared_parameters": [],
        "distinct_sample": False,
    }


# ------------------------------------------------------------------
# states: parsing & validation
# ------------------------------------------------------------------


# Filename heuristics (REF_L convention) — keep loose; intake re-validates
# from the actual file headers.
_COMBINED_RE = re.compile(r"_combined_data_auto\.txt$", re.IGNORECASE)
_PARTIAL_RE = re.compile(r"_(\d+)_(\d+)_partial\.txt$", re.IGNORECASE)
_PARTIAL_SETID_RE = re.compile(r"REFL_(\d+)_\d+_\d+_partial\.txt$", re.IGNORECASE)

_NUISANCE_KEYS = ("theta_offset", "sample_broadening")


def _parse_states(
    raw: Any, *, base_dir: Path, data_dir: Optional[Path] = None
) -> List[dict]:
    """Parse and validate the ``states:`` block.

    Returns an empty list when *raw* is falsy.  Each returned state has a
    private ``_kind`` field set to ``"combined"`` or ``"partials"`` for
    downstream consumers (the kind heuristic is filename-based; intake
    re-validates from headers).

    Path resolution: relative paths inside ``data_files`` are resolved
    against a prioritized list of candidate directories, taking the first
    match that exists:

    1. ``data_dir`` (the explicit override — CLI ``--data-dir`` flag or a
       top-level ``data_dir:`` YAML key), when given.
    2. ``base_dir`` (the directory holding the YAML / manifest file).
    3. the current working directory.

    Use ``data_dir`` when the YAML references files by name only but the
    actual data sits in a different directory (e.g. analyzer's
    ``plan-data`` output stored in ``plan/`` while the data lives in the
    parent directory). The cwd fallback lets a run launched from the data
    directory resolve bare filenames without an explicit override. The
    directory portion of a relative path is always honored (no
    same-basename fallback), and a file missing from every candidate root
    still raises a hard error listing the directories searched.
    """
    candidate_roots: List[Path] = []
    for root in (data_dir, base_dir, Path.cwd()):
        if root is None:
            continue
        resolved_root = Path(root).resolve()
        if resolved_root not in candidate_roots:
            candidate_roots.append(resolved_root)
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ConfigError("`states:` must be a list of state mappings.")

    seen_names: set[str] = set()
    parsed: List[dict] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"states[{i}] must be a mapping, got {type(entry).__name__}."
            )

        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ConfigError(f"states[{i}] is missing a non-empty `name`.")
        if name in seen_names:
            raise ConfigError(f"Duplicate state name: {name!r}.")
        seen_names.add(name)

        files_raw = entry.get("data_files") or entry.get("data") or []
        if isinstance(files_raw, str):
            files_raw = [files_raw]
        if not isinstance(files_raw, list) or not files_raw:
            raise ConfigError(
                f"State {name!r} must list at least one file under `data_files`."
            )

        data_files: List[dict] = []
        for j, item in enumerate(files_raw):
            if isinstance(item, str):
                file_path = item
                label = None
            elif isinstance(item, dict):
                file_path = item.get("file") or item.get("path")
                label = item.get("label")
                if not file_path:
                    raise ConfigError(
                        f"State {name!r} data_files[{j}] is missing `file`."
                    )
            else:
                raise ConfigError(
                    f"State {name!r} data_files[{j}] must be a path or mapping."
                )

            candidate = Path(file_path)
            if candidate.is_absolute():
                resolved = candidate.resolve()
                if not resolved.exists():
                    raise ConfigError(
                        f"State {name!r}: data file not found: {resolved}"
                    )
            else:
                resolved = None
                for root in candidate_roots:
                    trial = (root / candidate).resolve()
                    if trial.exists():
                        resolved = trial
                        break
                if resolved is None:
                    searched = ", ".join(str(r) for r in candidate_roots)
                    raise ConfigError(
                        f"State {name!r}: data file not found: {file_path!r}. "
                        f"Searched (in priority order): {searched}"
                    )

            data_files.append(
                {
                    "file": str(resolved),
                    "label": label or resolved.stem,
                }
            )

        kind = _detect_kind(name, data_files)

        # Per-state nuisance parameters are partials-only.
        for key in _NUISANCE_KEYS:
            if key in entry and entry[key] not in (False, None):
                if kind != "partials":
                    raise ConfigError(
                        f"State {name!r}: `{key}` is only valid for partials states "
                        f"(detected kind: {kind})."
                    )

        state: dict = {
            "name": name,
            "data_files": data_files,
            "_kind": kind,
        }
        for opt in (
            "extra_description",
            "back_reflection",
            "theta_offset",
            "sample_broadening",
            "background",
            "ambient",
            "intensity",
        ):
            if opt in entry:
                state[opt] = _normalise_nuisance(opt, entry[opt])
        # Per-state structure overrides (sample != structure): a state may carry
        # its own complete layers/substrate, passed through verbatim.
        for opt in ("layers", "substrate"):
            if entry.get(opt):
                state[opt] = entry[opt]
        parsed.append(state)

    if not parsed:
        raise ConfigError("`states:` must contain at least one state.")

    return parsed


def _detect_kind(state_name: str, data_files: List[dict]) -> str:
    """Heuristically classify a state's files as combined or partials.

    Mixed kinds within one state raise ConfigError.  Multiple partial
    files must share a single set_id.  Unknown filenames are tolerated
    (treated as combined) since intake re-validates from headers.
    """
    combined = []
    partials = []
    for ds in data_files:
        name = Path(ds["file"]).name
        if _PARTIAL_RE.search(name):
            partials.append(name)
        elif _COMBINED_RE.search(name):
            combined.append(name)
        else:
            combined.append(name)  # tolerate unknown — let intake decide

    if combined and partials:
        raise ConfigError(
            f"State {state_name!r}: cannot mix combined and partial files in one state."
        )

    if partials:
        set_ids = set()
        for name in partials:
            m = _PARTIAL_SETID_RE.search(name)
            if m:
                set_ids.add(m.group(1))
        if len(set_ids) > 1:
            raise ConfigError(
                f"State {state_name!r}: partial files must share one set_id "
                f"(found: {sorted(set_ids)})."
            )
        return "partials"

    if len(combined) > 1:
        # Multiple combined files in one state is suspicious — but the
        # state model genuinely allows it (e.g. spliced Q segments). Keep
        # tolerating it; intake will warn if headers conflict.
        pass
    return "combined"


# Per-state nuisance parameters expressed as a fittable {init, min, max}
# triplet. ``true`` expands to the default range below; ``false`` / null
# disables. ``theta_offset`` / ``sample_broadening`` are partials-only (see
# ``_NUISANCE_KEYS``); ``background`` applies to any state (combined or
# partial) — a single tied, fittable flat background per state.
_TRIPLET_DEFAULTS: dict = {
    "theta_offset": {"init": 0.0, "min": -0.02, "max": 0.02},
    "sample_broadening": {"init": 0.0, "min": 0.0, "max": 0.05},
    "background": {"init": 1e-6, "min": 0.0, "max": 1e-5},
}


def _normalise_nuisance(key: str, value: Any) -> Any:
    """Expand ``true`` for triplet nuisance params into their default dict."""
    if key not in _TRIPLET_DEFAULTS:
        return value
    if value is True:
        return dict(_TRIPLET_DEFAULTS[key])
    if value in (False, None):
        return None
    if isinstance(value, dict):
        return value
    raise ConfigError(f"`{key}` must be true/false or a {{init, min, max}} mapping.")


def format_user_criteria(cfg: Optional[UserConfig]) -> str:
    """Render ``evaluation_criteria`` as bullet points for a prompt.

    Returns an empty string when there are no criteria.
    """
    items = (cfg or {}).get("evaluation_criteria", [])
    if not items:
        return ""
    lines = ["## User-Defined Evaluation Criteria"]
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def format_user_constraints(cfg: Optional[UserConfig]) -> str:
    """Render ``model_constraints`` as bullet points for a prompt.

    Returns an empty string when there are no constraints.
    """
    items = (cfg or {}).get("model_constraints", [])
    if not items:
        return ""
    lines = ["## User-Defined Model Constraints"]
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def states_from_config(cfg: Optional[UserConfig]) -> List[dict]:
    """Return a runner-ready list of states from a loaded config.

    The private ``_kind`` field added during parsing is preserved so the
    intake node can short-circuit header inspection when desired.
    Returns ``[]`` when the config has no ``states:`` block.
    """
    if not cfg:
        return []
    states = cfg.get("states") or []
    return [dict(s) for s in states]


_DERIVED_FLAG_ENV = "ALLOW_DERIVED_PARAMETERS"
_TRUTHY = ("1", "true", "yes", "on")


def derived_parameters_enabled(explicit: Any = None) -> bool:
    """Is reparametrization (``derived_parameters``) enabled for this run?

    **Off by default.** A reparametrized model asks more of the LLM than a
    plain layer stack: derived layer attributes are not fit parameters, their
    numbers are computed rather than fitted, and the layers they reference must
    not be removed. A model that does not hold all that in mind will "fix" a
    derived SLD or refine the layer away, and the run degrades in a way that is
    hard to read from the outside. So the whole feature — the config key, the
    prompt rule, the skill — stays out of the way unless it is asked for.

    An explicit ``allow_derived_parameters`` in the config wins; otherwise the
    ``ALLOW_DERIVED_PARAMETERS`` environment variable; otherwise False.
    """
    if explicit is not None:
        return bool(explicit)
    import os

    return os.environ.get(_DERIVED_FLAG_ENV, "").strip().lower() in _TRUTHY


def check_derived_parameters_allowed(cfg: Any, *, source: str = "config") -> None:
    """Refuse a config that declares reparametrizations while the gate is off.

    Ignoring the block would be worse than refusing it: the run would fit a
    model measurably different from the one that was described, and nothing in
    the report would say why the excess it was supposed to constrain came out
    unconstrained.
    """
    if not (cfg or {}).get("derived_parameters"):
        return
    if (cfg or {}).get("allow_derived_parameters"):
        return
    names = ", ".join(
        str(d.get("name")) for d in cfg["derived_parameters"]  # type: ignore[index]
    )
    raise ConfigError(
        f"{source}: derived_parameters ({names}) are declared, but "
        f"reparametrization is off by default because it asks more of the "
        f"model than a plain layer stack does. Enable it with "
        f"`allow_derived_parameters: true` in this file, or "
        f"{_DERIVED_FLAG_ENV}=1 in the environment."
    )


def _parse_derived_parameters(raw: Any) -> List[dict]:
    """Shape-check the ``derived_parameters:`` block.

    Only the shape is checked here — whether the names and expressions resolve
    against the model is checked in the modeling node, which is the first point
    that knows the layer stack. What is caught here is what would otherwise
    produce a confusing failure much later: a scalar where a list belongs, a
    missing ``name``, a range that is not a range.
    """
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ConfigError("`derived_parameters:` must be a list of mappings.")
    out: List[dict] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"derived_parameters[{i}] must be a mapping, got "
                f"{type(entry).__name__}."
            )
        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ConfigError(f"derived_parameters[{i}] is missing a non-empty `name`.")
        if name in seen:
            raise ConfigError(f"Duplicate derived parameter name: {name!r}.")
        seen.add(name)
        free = entry.get("free") or {}
        if not isinstance(free, dict):
            raise ConfigError(f"derived_parameters[{name}].free must be a mapping.")
        for key in ("min", "max"):
            if key not in free:
                raise ConfigError(
                    f"derived_parameters[{name}].free is missing `{key}` — a "
                    f"derived parameter has no bounds of its own to fall back on."
                )
        try:
            lo, hi = float(free["min"]), float(free["max"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"derived_parameters[{name}].free min/max must be numbers."
            ) from exc
        if lo >= hi:
            raise ConfigError(f"derived_parameters[{name}]: free.min must be < max.")
        assign = entry.get("assign") or {}
        if not isinstance(assign, dict):
            raise ConfigError(
                f"derived_parameters[{name}].assign must be a mapping of "
                f'"<layer>.<attr>" to an expression.'
            )
        guards = entry.get("keep_physical") or []
        if isinstance(guards, str):
            guards = [guards]
        if not isinstance(guards, list):
            raise ConfigError(
                f"derived_parameters[{name}].keep_physical must be a list."
            )
        spec: dict = {"name": name, "free": dict(free), "assign": dict(assign)}
        if guards:
            spec["keep_physical"] = [str(g) for g in guards]
        for opt in ("source", "tied", "states"):
            if opt in entry:
                spec[opt] = entry[opt]
        out.append(spec)
    return out


def _as_str_list(value: Any) -> List[str]:
    """Coerce *value* to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []
