"""Canonical per-run setup file format (load + dump).

A "setup" YAML describes everything needed to launch ONE analysis run.
It is the file format used by:

- ``aure analyze -c setup.yaml`` — full run from a single file
- ``aure batch manifest.yaml`` — each job entry is a setup (with optional
  ``defaults:`` merging)
- ``aure batch setup.yaml`` — bare setup file treated as a 1-job manifest
- The web UI's ``Load Setup`` / ``Save Setup`` buttons

The schema is **states-only**: the legacy top-level ``data_file`` /
``data_files`` fields are gone. Every analysis (even a single data file)
declares its files inside a ``states:`` block. See the worked example in
[aure_config.example.yaml](../../aure_config.example.yaml).

Synonyms (kept for ``analyzer plan-data`` interop):

- ``describe:`` / ``description:`` are accepted in place of ``sample_description:``
- Inside a state, ``data:`` is accepted in place of ``data_files:``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from .config import (
    ConfigError,
    _as_str_list,
    _parse_states,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class SetupConfig(TypedDict, total=False):
    """A single analysis run, parsed from a setup YAML file.

    Field names match the YAML keys verbatim so :func:`dump_setup` can
    round-trip the file. ``states`` carries the
    :class:`~aure.state.StateDefinition`-shaped entries produced by
    :func:`aure.config._parse_states` (paths absolute and validated).
    """

    # Identification (used to name the output subdir; optional)
    name: str

    # Sample / model
    sample_description: str
    hypothesis: str
    model_name: str

    # Data — always via states
    states: List[dict]
    shared_parameters: List[str]
    unshared_parameters: List[str]
    distinct_sample: bool  # co-refined states are distinct physical samples

    # LLM-side rules
    evaluation_criteria: List[str]
    model_constraints: List[str]

    # Run controls (mirroring `aure analyze` flags)
    command: str  # "analyze" | "prepare"
    output_root: str  # parent for the output directory
    max_refinements: int
    fit_method: str  # "lm" | "de" | "dream"
    fit_steps: int
    fit_burn: int

    # Job-runner switches (batch only — ignored by analyze)
    verbose: bool
    json: bool

    # Per-job environment overrides (batch only; forwarded as env vars).
    llm_provider: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str
    llm_temperature: float
    llm_timeout: int
    alcf_cluster: str
    alcf_access_token: str

    # Free-form pass-through (analyzer compat: e.g. plan-data emits
    # `metadata.perform_assembly` / `metadata.notes`). AuRE ignores
    # everything inside but preserves it on round-trip.
    metadata: dict


# Top-level keys we recognise. Anything else is rejected to catch typos
# — except for ``metadata`` which is a free-form passthrough.
_KNOWN_TOP_LEVEL = {
    "name",
    "sample_description",
    "describe",  # synonym
    "description",  # synonym
    "hypothesis",
    "model_name",
    "data_dir",
    "states",
    "shared_parameters",
    "unshared_parameters",
    "distinct_sample",
    "evaluation_criteria",
    "model_constraints",
    "command",
    "output_root",
    "max_refinements",
    "fit_method",
    "fit_steps",
    "fit_burn",
    "verbose",
    "json",
    "llm_provider",
    "llm_model",
    "llm_api_key",
    "llm_base_url",
    "llm_temperature",
    "llm_timeout",
    "alcf_cluster",
    "alcf_access_token",
    "metadata",
}

# Keys that the legacy schema used at the top level (or in a job entry)
# but which are no longer supported. Loaders raise a migration error
# when they see one.
_LEGACY_DATA_KEYS = ("data_file", "data_files")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_setup(
    path: str | Path, *, data_dir: Optional[str | Path] = None
) -> SetupConfig:
    """Read and validate a setup YAML file.

    Parameters
    ----------
    path
        Path to the setup YAML file.
    data_dir
        Optional override for the directory used to resolve relative
        ``data_files`` entries. Defaults to the YAML's parent directory.
        Useful when the YAML lives in (say) ``plan/job.yaml`` but the
        data sits in the parent directory, or in a sibling ``data/``.

    Raises :class:`~aure.config.ConfigError` on:
    - missing or unreadable file
    - presence of legacy ``data_file:`` / ``data_files:`` at the top
      level (the schema is now states-only — see the migration message
      attached to the error)
    - an unrecognised top-level key (typo guard)
    - any validation error from :func:`~aure.config._parse_states`
    """
    p = Path(path).resolve()
    if not p.exists():
        raise ConfigError(f"setup file not found: {p}")

    import yaml

    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML: {exc}") from exc

    if raw is None:
        raise ConfigError(f"{p}: setup file is empty")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{p}: setup file must be a YAML mapping, got {type(raw).__name__}"
        )
    if "jobs" in raw:
        raise ConfigError(
            f"{p}: this looks like a batch manifest (it has a `jobs:` block). "
            "Use `aure batch` for manifests, or extract a single job into a "
            "flat setup file."
        )

    resolved_data_dir: Optional[Path] = None
    if data_dir is not None:
        resolved_data_dir = Path(data_dir).resolve()
        if not resolved_data_dir.is_dir():
            raise ConfigError(
                f"data_dir override is not a directory: {resolved_data_dir}"
            )

    return _setup_from_dict(
        raw, base_dir=p.parent, source=str(p), data_dir=resolved_data_dir
    )


def _setup_from_dict(
    raw: Dict[str, Any],
    *,
    base_dir: Path,
    source: str,
    data_dir: Optional[Path] = None,
) -> SetupConfig:
    """Convert a raw YAML dict into a validated SetupConfig.

    Shared by :func:`load_setup` (single file) and the batch manifest
    loader (one call per job entry).
    """
    # Reject legacy single-state shortcuts. Hard break — analyzer
    # plan-data output never used them, current users get a clear
    # migration path.
    for key in _LEGACY_DATA_KEYS:
        if key in raw:
            raise ConfigError(
                f"{source}: `{key}` at the top level is no longer supported. "
                "Move data files into a `states:` block:\n"
                "  states:\n"
                "    - name: state0\n"
                "      data_files:\n"
                "        - file: <path>\n"
            )

    unknown = set(raw.keys()) - _KNOWN_TOP_LEVEL
    if unknown:
        raise ConfigError(
            f"{source}: unknown top-level key(s): {sorted(unknown)}. "
            f"Allowed: {sorted(_KNOWN_TOP_LEVEL)}."
        )

    out: SetupConfig = {}  # type: ignore[assignment]

    if "name" in raw:
        out["name"] = str(raw["name"])

    desc = (
        raw.get("sample_description") or raw.get("describe") or raw.get("description")
    )
    if desc:
        out["sample_description"] = str(desc)

    for opt_str in ("hypothesis", "model_name", "command", "output_root", "fit_method"):
        if raw.get(opt_str):
            out[opt_str] = str(raw[opt_str])  # type: ignore[literal-required]

    for opt_int in ("max_refinements", "fit_steps", "fit_burn", "llm_timeout"):
        if opt_int in raw and raw[opt_int] is not None:
            try:
                out[opt_int] = int(raw[opt_int])  # type: ignore[literal-required]
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{source}: `{opt_int}` must be an integer") from exc

    if "llm_temperature" in raw and raw["llm_temperature"] is not None:
        try:
            out["llm_temperature"] = float(raw["llm_temperature"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{source}: `llm_temperature` must be a number") from exc

    for opt_str in (
        "llm_provider",
        "llm_model",
        "llm_api_key",
        "llm_base_url",
        "alcf_cluster",
        "alcf_access_token",
    ):
        if raw.get(opt_str):
            out[opt_str] = str(raw[opt_str])  # type: ignore[literal-required]

    for opt_bool in ("verbose", "json"):
        if opt_bool in raw and raw[opt_bool] is not None:
            out[opt_bool] = bool(raw[opt_bool])  # type: ignore[literal-required]

    out["evaluation_criteria"] = _as_str_list(raw.get("evaluation_criteria"))
    out["model_constraints"] = _as_str_list(raw.get("model_constraints"))

    shared = _as_str_list(raw.get("shared_parameters"))
    unshared = _as_str_list(raw.get("unshared_parameters"))
    if shared and unshared:
        raise ConfigError(
            f"{source}: shared_parameters and unshared_parameters are "
            "mutually exclusive; provide at most one."
        )
    out["shared_parameters"] = shared
    out["unshared_parameters"] = unshared

    if raw.get("distinct_sample") is not None:
        out["distinct_sample"] = bool(raw["distinct_sample"])

    # Resolve the effective data-file resolution override. A programmatic
    # / CLI ``data_dir`` argument wins over a top-level ``data_dir:`` YAML
    # key. A relative YAML key is resolved against the YAML's own
    # directory (``base_dir``), mirroring how relative ``data_files`` are
    # resolved. Validation lives here (not only in ``load_setup``) so the
    # manifest loader — which calls this function directly — is guarded too.
    effective_data_dir = data_dir
    if effective_data_dir is None and raw.get("data_dir"):
        dd = Path(str(raw["data_dir"]))
        if not dd.is_absolute():
            dd = base_dir / dd
        dd = dd.resolve()
        if not dd.is_dir():
            raise ConfigError(
                f"{source}: `data_dir` is not an existing directory: {dd}"
            )
        effective_data_dir = dd

    out["states"] = _parse_states(
        raw.get("states"), base_dir=base_dir, data_dir=effective_data_dir
    )
    if not out["states"]:
        raise ConfigError(
            f"{source}: at least one state must be declared under `states:`."
        )

    if "metadata" in raw and raw["metadata"] is not None:
        if not isinstance(raw["metadata"], dict):
            raise ConfigError(f"{source}: `metadata` must be a mapping if present.")
        out["metadata"] = dict(raw["metadata"])

    return out


# ---------------------------------------------------------------------------
# Dumping
# ---------------------------------------------------------------------------


# Field render order — drives the YAML output for readability.
# Note: ``data_dir`` is deliberately NOT dumped. It is a load-time-only
# resolution hint; dumped ``data_files`` already carry absolute paths, so a
# re-emitted ``data_dir:`` would be inert and potentially misleading.
_DUMP_ORDER: tuple[str, ...] = (
    "name",
    "sample_description",
    "hypothesis",
    "model_name",
    "command",
    "output_root",
    "max_refinements",
    "fit_method",
    "fit_steps",
    "fit_burn",
    "shared_parameters",
    "unshared_parameters",
    "distinct_sample",
    "evaluation_criteria",
    "model_constraints",
    "states",
    "metadata",
)


# State-internal field order (matches plan-data + AuRE conventions).
_STATE_DUMP_ORDER: tuple[str, ...] = (
    "name",
    "extra_description",
    "back_reflection",
    "theta_offset",
    "sample_broadening",
    "background",
    "ambient",
    "intensity",
    "data_files",
)


def dump_setup(setup: SetupConfig) -> str:
    """Render a :class:`SetupConfig` back to canonical setup-YAML text.

    Empty / None fields are dropped. Private ``_kind`` fields added by
    :func:`~aure.config._parse_states` are stripped. The output is a
    self-contained file that :func:`load_setup` will accept verbatim,
    so the web UI can round-trip a form → YAML → form.
    """
    body: dict = {}
    for key in _DUMP_ORDER:
        if key not in setup:
            continue
        value = setup[key]  # type: ignore[literal-required]
        if value is None:
            continue
        if isinstance(value, (list, dict, str)) and not value:
            continue
        if isinstance(value, bool) and not value:
            continue  # drop default/false flags (e.g. distinct_sample) for cleaner YAML
        if key == "states":
            body[key] = [_state_for_dump(st) for st in value]
        else:
            body[key] = value

    import yaml

    return yaml.safe_dump(
        body,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def _state_for_dump(state: dict) -> dict:
    """Strip internal markers and reorder a state for YAML output."""
    out: dict = {}
    for key in _STATE_DUMP_ORDER:
        if key not in state:
            continue
        value = state[key]
        if value is None:
            continue
        if isinstance(value, (list, dict, str)) and not value:
            continue
        if key == "data_files":
            out[key] = [_dataset_for_dump(ds) for ds in value]
        else:
            out[key] = value
    # Preserve any caller-attached fields we don't know about (forward-compat).
    for key, value in state.items():
        if key in out or key in _STATE_DUMP_ORDER or key.startswith("_"):
            continue
        out[key] = value
    return out


def _dataset_for_dump(ds: dict) -> dict:
    """Render one DatasetInfo: keep ``file`` and ``label`` only, drop Q/R/dR."""
    out: dict = {}
    if "file" in ds:
        out["file"] = ds["file"]
    if ds.get("label"):
        out["label"] = ds["label"]
    return out


# ---------------------------------------------------------------------------
# Convenience: derive the primary data_file from a setup
# ---------------------------------------------------------------------------


def setup_to_user_config(setup: SetupConfig) -> dict:
    """Extract the subset of fields the runner expects as ``user_config``.

    The runner / modeling node reads ``evaluation_criteria``,
    ``model_constraints``, ``shared_parameters``, and ``unshared_parameters``
    from a plain dict (see :mod:`aure.nodes.modeling`). ``model_name`` is
    carried so the fitting node can name the exported FitProblem (otherwise
    bumps writes ``None-*`` / ``None.json``). All other setup fields drive the
    CLI/web caller directly.
    """
    cfg: dict = {}
    for key in (
        "evaluation_criteria",
        "model_constraints",
        "shared_parameters",
        "unshared_parameters",
        "distinct_sample",
        "sample_description",
        "model_name",
    ):
        value = setup.get(key)  # type: ignore[literal-required]
        if value:
            cfg[key] = value
    return cfg


def primary_data_file(setup: SetupConfig) -> str:
    """Return the first dataset's file path — used to satisfy the
    runner's positional ``data_file`` argument when launching from
    a states-only setup.
    """
    states = setup.get("states") or []
    if not states:
        raise ConfigError("setup has no states; cannot derive a primary data file.")
    files = states[0].get("data_files") or []
    if not files:
        raise ConfigError(
            f"state {states[0].get('name')!r} has no data files; "
            "cannot derive a primary data file."
        )
    return files[0]["file"]


# ---------------------------------------------------------------------------
# Manifests (multi-job)
# ---------------------------------------------------------------------------


class Manifest(TypedDict):
    """A loaded batch manifest: ``defaults`` + a list of :class:`SetupConfig`."""

    defaults: Dict[str, Any]
    jobs: List[SetupConfig]


def load_manifest(
    path: str | Path, *, data_dir: Optional[str | Path] = None
) -> Manifest:
    """Load a batch manifest.

    Two shapes are accepted:

    1. **Manifest** — top-level ``jobs:`` (list) and optional ``defaults:``.
       Each job is merged with ``defaults`` and validated as a SetupConfig.
    2. **Flat setup** — no ``jobs:`` key. The file is treated as a
       single-job manifest (one entry, no defaults).

    File paths inside each state are resolved relative to the manifest
    file's directory.

    ``data_dir`` is an optional override (``aure batch --data-dir``) applied
    to **every** job, taking priority over any per-job or ``defaults:``
    ``data_dir:`` key. Relative ``data_files`` then resolve against this
    directory first, then the manifest directory, then the cwd.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise ConfigError(f"manifest file not found: {p}")

    resolved_data_dir: Optional[Path] = None
    if data_dir is not None:
        resolved_data_dir = Path(data_dir).resolve()
        if not resolved_data_dir.is_dir():
            raise ConfigError(
                f"data_dir override is not a directory: {resolved_data_dir}"
            )

    import yaml

    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: invalid YAML: {exc}") from exc

    if raw is None:
        raise ConfigError(f"{p}: manifest is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: manifest must be a YAML mapping")

    base_dir = p.parent

    if "jobs" in raw:
        defaults = raw.get("defaults") or {}
        jobs_raw = raw["jobs"]
        if not isinstance(defaults, dict):
            raise ConfigError(f"{p}: `defaults` must be a mapping if present.")
        if not isinstance(jobs_raw, list) or not jobs_raw:
            raise ConfigError(f"{p}: `jobs:` must be a non-empty list of job mappings.")
        jobs: List[SetupConfig] = []
        for i, job_raw in enumerate(jobs_raw):
            if not isinstance(job_raw, dict):
                raise ConfigError(
                    f"{p}: jobs[{i}] must be a mapping, got {type(job_raw).__name__}."
                )
            # Defaults merge first; per-job keys override.
            merged = {**defaults, **job_raw}
            try:
                jobs.append(
                    _setup_from_dict(
                        merged,
                        base_dir=base_dir,
                        source=f"{p}#jobs[{i}]",
                        data_dir=resolved_data_dir,
                    )
                )
            except ConfigError:
                raise
        return {"defaults": defaults, "jobs": jobs}

    # Flat shape — treat the whole file as a single-job manifest.
    single = _setup_from_dict(
        raw, base_dir=base_dir, source=str(p), data_dir=resolved_data_dir
    )
    return {"defaults": {}, "jobs": [single]}
