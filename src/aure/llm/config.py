"""
LLM configuration read from environment variables.

Every provider module receives a *config dict* produced by
:func:`get_llm_config` so that provider-specific code never has to touch
``os.environ`` directly.
"""

import os
from typing import Dict, Any

# Default models per provider. ``claude_code`` defaults to empty: the CLI
# resolves its own model, and pinning one here would break an account whose
# backend (Bedrock, Vertex, Foundry) has not deployed that alias.
DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash-lite",
    "local": "llama3",
    "claude_code": "",
}


def get_llm_timeout() -> int:
    """Return the LLM call timeout in seconds (``LLM_TIMEOUT``, default 120)."""
    return int(os.environ.get("LLM_TIMEOUT", "120"))


def get_llm_config() -> Dict[str, Any]:
    """
    Build an LLM configuration dict from environment variables.

    Returns a dict with keys: provider, api_key, model, base_url,
    temperature.
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

    default_model = DEFAULT_MODELS.get(provider, "gpt-4o-mini")

    return {
        "provider": provider,
        "api_key": api_key,
        "model": os.environ.get("LLM_MODEL", default_model),
        "base_url": os.environ.get("LLM_BASE_URL"),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.0")),
    }


def llm_available() -> bool:
    """Return ``True`` if the active provider is usable."""
    config = get_llm_config()
    provider = config["provider"]

    if provider == "claude_code":
        # No key of its own: the CLI holds whatever credential is in play.
        # Imported lazily because providers/ imports this module.
        from .providers.claude_code import available

        return available()
    if provider == "local":
        return bool(config["base_url"])
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
    elif config["provider"] == "claude_code":
        from .providers.claude_code import _binary

        try:
            info["binary"] = _binary()
        except ValueError:
            info["binary"] = None
        info["model"] = config["model"] or "(the CLI's default)"
    return info
