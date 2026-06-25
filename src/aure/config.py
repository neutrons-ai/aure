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


_EMPTY: UserConfig = {
    "evaluation_criteria": [],
    "model_constraints": [],
    "sample_description": None,
    "states": [],
    "shared_parameters": [],
    "unshared_parameters": [],
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

    try:
        import yaml  # pyyaml is already a dependency
    except ImportError:
        logger.warning("[CONFIG] pyyaml not installed – ignoring config file")
        return _empty_config()

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
    if cfg["shared_parameters"] and cfg["unshared_parameters"]:
        raise ConfigError(
            "shared_parameters and unshared_parameters are mutually exclusive; "
            "provide at most one."
        )

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


def _as_str_list(value: Any) -> List[str]:
    """Coerce *value* to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []
