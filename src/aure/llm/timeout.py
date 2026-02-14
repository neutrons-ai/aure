"""
Signal-based timeout wrapper for LLM invocations.
"""

import signal
from typing import Any

from .config import get_llm_timeout


class LLMTimeoutError(Exception):
    """Raised when an LLM call exceeds the configured timeout."""


def _timeout_handler(signum, frame):
    raise LLMTimeoutError("LLM call timed out - possible quota/rate limit issue")


def invoke_with_timeout(llm, prompt, timeout_seconds: int = None) -> Any:
    """
    Invoke *llm* with a SIGALRM-based timeout (Unix only).

    Args:
        llm: LangChain chat model instance.
        prompt: A string or list of messages.
        timeout_seconds: Max wait in seconds (default: ``LLM_TIMEOUT``).

    Returns:
        The LLM response.

    Raises:
        LLMTimeoutError: If the call exceeds the timeout.
    """
    if timeout_seconds is None:
        timeout_seconds = get_llm_timeout()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        if isinstance(prompt, str):
            from langchain_core.messages import HumanMessage
            response = llm.invoke([HumanMessage(content=prompt)])
        else:
            response = llm.invoke(prompt)
        return response
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
