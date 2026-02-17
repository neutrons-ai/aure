"""
User-supplied YAML configuration for analysis constraints and criteria.

An optional YAML file (``--config config.yaml``) lets the user inject:

* **evaluation_criteria** – extra rules the LLM should apply when judging
  whether a fit is acceptable (e.g. "titanium thickness must be 45–55 Å").
* **model_constraints** – hard constraints the LLM must respect when
  building or refining a model (e.g. "do not add extra layers").

See ``aure_config.example.yaml`` in the repository root for the full schema.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)


class UserConfig(TypedDict, total=False):
    """User-supplied analysis configuration."""

    evaluation_criteria: List[str]
    model_constraints: List[str]


_EMPTY: UserConfig = {
    "evaluation_criteria": [],
    "model_constraints": [],
}


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
    """
    if path is None:
        return dict(_EMPTY)  # type: ignore[return-value]

    p = Path(path)
    if not p.exists():
        logger.warning("[CONFIG] Config file not found: %s", p)
        return dict(_EMPTY)  # type: ignore[return-value]

    try:
        import yaml  # pyyaml is already a dependency
    except ImportError:
        logger.warning("[CONFIG] pyyaml not installed – ignoring config file")
        return dict(_EMPTY)  # type: ignore[return-value]

    raw: Dict[str, Any] = yaml.safe_load(p.read_text()) or {}
    logger.info("[CONFIG] Loaded user config from %s", p)

    cfg: UserConfig = {
        "evaluation_criteria": _as_str_list(raw.get("evaluation_criteria")),
        "model_constraints": _as_str_list(raw.get("model_constraints")),
    }

    if cfg["evaluation_criteria"]:
        logger.info(
            "[CONFIG]   %d evaluation criteria", len(cfg["evaluation_criteria"])
        )
    if cfg["model_constraints"]:
        logger.info(
            "[CONFIG]   %d model constraints", len(cfg["model_constraints"])
        )

    return cfg


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

def _as_str_list(value: Any) -> List[str]:
    """Coerce *value* to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []
