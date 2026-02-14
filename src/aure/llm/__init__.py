"""
LLM configuration and provider dispatch.

Submodules
----------
config      – environment-variable-driven configuration
timeout     – signal-based call timeout wrapper
providers/  – one module per LLM backend (openai, gemini, alcf, local, …)

Public API (re-exported here for convenience)
---------------------------------------------
get_llm_config, get_llm_timeout, llm_available, get_llm_info,
get_llm, invoke_with_timeout, LLMTimeoutError
"""

from .config import get_llm_config, get_llm_timeout, llm_available, get_llm_info
from .timeout import LLMTimeoutError, invoke_with_timeout
from .providers import get_llm

__all__ = [
    "get_llm_config",
    "get_llm_timeout",
    "llm_available",
    "get_llm_info",
    "get_llm",
    "invoke_with_timeout",
    "LLMTimeoutError",
]
