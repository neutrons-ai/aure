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


# Models that refused the temperature we sent, keyed by (base_url, model). A
# model in here is created with no temperature at all, so it runs at its own
# default: some deployments accept only that (gpt-6.1-sol refuses anything but
# 1), and sending 1 to a model that takes 0 would silently change the analysis.
_TEMPERATURE_REFUSED: set = set()


def _refuses_temperature(exc: BaseException) -> bool:
    """True if a provider error is a refusal of the ``temperature`` parameter."""
    if getattr(exc, "status_code", None) != 400:
        return False
    if getattr(exc, "param", None) == "temperature":
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        if error.get("param") == "temperature":
            return True
    message = str(exc).lower()
    return "temperature" in message and "unsupported" in message


_CHAT_CLASS = None


def _chat_class():
    """``ChatOpenAI`` with one retry, without a temperature, if the model refuses it.

    Built on first use so that importing AuRE does not import LangChain.
    """
    global _CHAT_CLASS
    if _CHAT_CLASS is None:
        from langchain_openai import ChatOpenAI

        class TemperatureFallbackChatOpenAI(ChatOpenAI):
            def _generate(self, *args, **kwargs):
                try:
                    return super()._generate(*args, **kwargs)
                except Exception as exc:
                    if self.temperature is None or not _refuses_temperature(exc):
                        raise
                    self._drop_temperature()
                    return super()._generate(*args, **kwargs)

            async def _agenerate(self, *args, **kwargs):
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as exc:
                    if self.temperature is None or not _refuses_temperature(exc):
                        raise
                    self._drop_temperature()
                    return await super()._agenerate(*args, **kwargs)

            def _drop_temperature(self) -> None:
                logger.warning(
                    "[LLM] %s refused temperature=%s; using its default from now on",
                    self.model_name,
                    self.temperature,
                )
                _TEMPERATURE_REFUSED.add((self.openai_api_base, self.model_name))
                self.temperature = None

        _CHAT_CLASS = TemperatureFallbackChatOpenAI
    return _CHAT_CLASS


def create_openai_compatible(
    config: dict,
    temperature: float,
    *,
    api_key: str,
    base_url: Optional[str] = None,
):
    """Create a ``ChatOpenAI`` instance.

    Shared code-path for the *openai* and *local* providers. The requested
    temperature is sent as asked unless this model has already refused it, in
    which case the call goes out with none and the model uses its default.
    """
    kwargs: dict = dict(
        model=config["model"],
        temperature=temperature,
        api_key=api_key,
        max_retries=_max_retries(),
        timeout=float(get_llm_timeout()),
    )
    if base_url:
        kwargs["base_url"] = base_url
    llm = _chat_class()(**kwargs)
    if (llm.openai_api_base, llm.model_name) in _TEMPERATURE_REFUSED:
        llm.temperature = None
    elif temperature is not None and llm.temperature != temperature:
        # langchain-openai drops any temperature but 1 for "gpt-5*" names, a
        # rule written for OpenAI's own reasoning models. Deployments that
        # accept 0 (Azure's gpt-5.4) then ran at their default without a word.
        # Send what was asked; a model that refuses it falls back above.
        llm.temperature = temperature
    return llm


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
