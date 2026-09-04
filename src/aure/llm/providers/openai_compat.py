"""
OpenAI-compatible provider factories (openai, local).

Both use :class:`langchain_openai.ChatOpenAI` with different base URLs and
credential handling. The shared construction logic lives in
:func:`create_openai_compatible`.

Any OpenAI-compatible endpoint is reachable through the *local* provider by
setting ``LLM_BASE_URL`` and ``LLM_API_KEY`` — a self-hosted vLLM or Ollama
server, or a remote facility endpoint. It needs no provider of its own.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..config import get_llm_timeout

logger = logging.getLogger(__name__)

# Transient connection failures are routine against a local server: Ollama
# evicts an idle model after ~5 min, and AuRE's refl1d fits run far longer than
# that, so the call that follows a fit often arrives while the model is being
# reloaded from disk. With no retry a single such blip aborts the whole analysis
# — in one 23-case validation sweep it cost 12 cases, and one of those still
# emitted a degenerate fallback model that looked like a (terrible) answer.
_DEFAULT_MAX_RETRIES = 2


def _max_retries() -> int:
    """How many times to retry a failed LLM call (``LLM_MAX_RETRIES``).

    Set to 0 to restore the previous fail-fast behaviour. Each retry is bounded
    by the same per-call timeout, so the worst case stays predictable.
    """
    raw = os.environ.get("LLM_MAX_RETRIES")
    if raw is None or raw == "":
        return _DEFAULT_MAX_RETRIES
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("[LLM] Ignoring non-numeric LLM_MAX_RETRIES=%r", raw)
        return _DEFAULT_MAX_RETRIES


def create_openai_compatible(
    config: dict,
    temperature: float,
    *,
    api_key: str,
    base_url: Optional[str] = None,
):
    """Create a ``ChatOpenAI`` instance.

    Shared code-path for the *openai* and *local* providers.
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict = dict(
        model=config["model"],
        temperature=temperature,
        api_key=api_key,
        max_retries=_max_retries(),
        timeout=float(get_llm_timeout()),
    )
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


# ── Thin per-provider wrappers ──────────────────────────────────────────


def create_openai(config: dict, temperature: float):
    if not config["api_key"]:
        raise ValueError(
            "LLM_API_KEY or OPENAI_API_KEY must be set for OpenAI provider"
        )
    return create_openai_compatible(
        config, temperature, api_key=config["api_key"], base_url=config.get("base_url")
    )


def create_local(config: dict, temperature: float):
    if not config["base_url"]:
        raise ValueError(
            "LLM_BASE_URL must be set when using the local provider. "
            "Example: http://localhost:11434/v1 for Ollama"
        )
    api_key = config["api_key"] or "not-needed"
    return create_openai_compatible(
        config, temperature, api_key=api_key, base_url=config["base_url"]
    )
