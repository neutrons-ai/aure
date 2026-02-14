"""
LLM provider registry.

Each provider module exposes two functions:

* ``is_available(config) -> bool``  (optional – used for future probing)
* ``create_llm(config, temperature) -> BaseChatModel``

:func:`get_llm` reads the active provider from *config["provider"]* and
dispatches to the right module.  Adding a new provider is as simple as
dropping a new ``<name>.py`` in this package and registering it in
``_PROVIDERS`` below.
"""

from typing import Optional

from ..config import get_llm_config, get_llm_timeout

# Lazy registry: provider name → module path (relative to this package)
_PROVIDERS: dict[str, str] = {
    "openai": ".openai",
    "gemini": ".gemini",
    "alcf":   ".alcf",
    "local":  ".local",
}


def get_llm(temperature: Optional[float] = None):
    """
    Return a configured LangChain chat model for the active provider.

    Args:
        temperature: Override the configured temperature.

    Returns:
        A LangChain ``BaseChatModel`` instance.

    Raises:
        ValueError: If the provider is unknown or misconfigured.
    """
    config = get_llm_config()
    provider = config["provider"]

    module_path = _PROVIDERS.get(provider)
    if module_path is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Supported: {', '.join(sorted(_PROVIDERS))}"
        )

    # Import the provider module lazily so we don't pull in heavy SDKs
    # until they are actually needed.
    import importlib
    mod = importlib.import_module(module_path, package=__package__)

    temp = temperature if temperature is not None else config["temperature"]
    return mod.create_llm(config, temp)
