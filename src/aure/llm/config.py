"""
LLM configuration read from environment variables.

Every provider module receives a *config dict* produced by
:func:`get_llm_config` so that provider-specific code never has to touch
``os.environ`` directly.
"""

import os
from typing import Dict, Any

# Default models per provider
DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash-lite",
    "alcf": "gpt-oss-120b",
    "local": "llama3",
}


def get_llm_timeout() -> int:
    """Return the LLM call timeout in seconds (``LLM_TIMEOUT``, default 120)."""
    return int(os.environ.get("LLM_TIMEOUT", "120"))


def get_llm_config() -> Dict[str, Any]:
    """
    Build an LLM configuration dict from environment variables.

    Returns a dict with keys: provider, api_key, model, base_url,
    temperature, alcf_cluster.
    """
    provider = os.environ.get("LLM_PROVIDER", "").lower()

    # Auto-detect provider from legacy API keys if not explicitly set
    if not provider:
        if os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "openai"

    # Resolve API key
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

    # ALCF cluster
    alcf_cluster = None
    if provider == "alcf":
        alcf_cluster = os.environ.get("ALCF_CLUSTER", "sophia").lower()

    default_model = DEFAULT_MODELS.get(provider, "gpt-4o-mini")

    return {
        "provider": provider,
        "api_key": api_key,
        "model": os.environ.get("LLM_MODEL", default_model),
        "base_url": os.environ.get("LLM_BASE_URL"),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.0")),
        "alcf_cluster": alcf_cluster,
    }


def llm_available() -> bool:
    """Return ``True`` if the active provider is usable."""
    config = get_llm_config()
    provider = config["provider"]

    if provider == "local":
        return bool(config["base_url"])
    if provider == "alcf":
        return True  # token obtained lazily
    # Cloud providers need an API key
    return bool(config["api_key"])


def get_llm_info() -> dict:
    """Return a small info dict useful for logging / debugging."""
    config = get_llm_config()
    info = {
        "provider": config["provider"],
        "model": config["model"],
        "available": llm_available(),
    }
    if config["provider"] == "local":
        info["base_url"] = config["base_url"]
    elif config["provider"] == "alcf":
        info["alcf_cluster"] = config["alcf_cluster"]
    return info
