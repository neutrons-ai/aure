"""
OpenAI-compatible provider factories (openai, alcf, local).

All three use :class:`langchain_openai.ChatOpenAI` with different
base URLs and credential handling.  The shared construction logic lives
in :func:`create_openai_compatible`.
"""

from __future__ import annotations

from typing import Optional

from ..config import get_llm_timeout
from .alcf_auth import get_token

ALCF_CLUSTER_PATHS: dict[str, str] = {
    "sophia": "/resource_server/sophia/vllm/v1",
    "metis": "/resource_server/metis/api/v1",
}


def create_openai_compatible(
    config: dict,
    temperature: float,
    *,
    api_key: str,
    base_url: Optional[str] = None,
):
    """Create a ``ChatOpenAI`` instance.

    Shared code-path for the *openai*, *alcf*, and *local* providers.
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict = dict(
        model=config["model"],
        temperature=temperature,
        api_key=api_key,
        max_retries=0,
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


def create_alcf(config: dict, temperature: float):
    cluster = config.get("alcf_cluster") or "sophia"
    path = ALCF_CLUSTER_PATHS.get(cluster)
    if path is None:
        raise ValueError(
            f"Unknown ALCF cluster '{cluster}'. "
            f"Supported: {', '.join(sorted(ALCF_CLUSTER_PATHS))}"
        )
    base_url = f"https://inference-api.alcf.anl.gov{path}"
    api_key = get_token()
    return create_openai_compatible(
        config, temperature, api_key=api_key, base_url=base_url
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
