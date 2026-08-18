"""
Timeout wrapper for LLM invocations.

Uses SIGALRM when running on the main thread (fast, no extra thread) and
falls back to ``concurrent.futures`` when called from a background thread
(e.g. the Flask web-UI analysis runner).
"""

import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

from .config import get_llm_timeout


class LLMTimeoutError(Exception):
    """Raised when an LLM call exceeds the configured timeout."""


def _timeout_handler(signum, frame):
    raise LLMTimeoutError("LLM call timed out - possible quota/rate limit issue")


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def invoke_with_timeout(llm, prompt, timeout_seconds: int = None) -> Any:
    """
    Invoke *llm* with a timeout.

    When running on the **main** thread a SIGALRM-based approach is used
    (Unix only, zero overhead).  From any other thread a
    ``ThreadPoolExecutor`` future is used instead, which avoids the
    ``signal only works in main thread`` error.

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

    if _is_main_thread():
        return _invoke_signal(llm, prompt, timeout_seconds)
    return _invoke_futures(llm, prompt, timeout_seconds)


def _do_invoke(llm, prompt):
    """Perform the actual LLM invocation, recording it in the call ledger.

    Every node reaches the provider through here, which is why the ledger hooks
    this function and not each call site. The record is written whether the call
    succeeds or raises, so a timed-out or refused call still appears with its
    cost — a failed call is billed and must not vanish from the total.
    """
    import time

    from . import ledger

    if isinstance(prompt, str):
        from langchain_core.messages import HumanMessage

        payload = [HumanMessage(content=prompt)]
    else:
        payload = prompt

    started = time.monotonic()
    try:
        response = llm.invoke(payload)
    except Exception as exc:
        ledger.record(
            None, duration_s=time.monotonic() - started,
            model=getattr(llm, "model_name", None), prompt=prompt,
            ok=False, error=f"{type(exc).__name__}: {exc}",
        )
        raise
    ledger.record(
        response, duration_s=time.monotonic() - started,
        model=getattr(llm, "model_name", None), prompt=prompt,
    )
    return response


def _invoke_signal(llm, prompt, timeout_seconds: int) -> Any:
    """SIGALRM-based timeout (main thread only)."""
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return _do_invoke(llm, prompt)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _invoke_futures(llm, prompt, timeout_seconds: int) -> Any:
    """Thread-pool-based timeout (safe from any thread)."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_do_invoke, llm, prompt)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeout:
            future.cancel()
            raise LLMTimeoutError(
                "LLM call timed out - possible quota/rate limit issue"
            )
