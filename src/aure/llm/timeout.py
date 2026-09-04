"""
Timeout wrapper for LLM invocations.

Uses SIGALRM when running on the main thread of a platform that has it (fast,
no extra thread) and falls back to a watched daemon thread otherwise — from a
background thread (e.g. the Flask web-UI analysis runner), or on Windows, where
``signal.SIGALRM`` does not exist at all.

**What "timeout" means on the thread path.** A Python thread cannot be killed,
so the deadline bounds *how long the caller waits*, not how long the request
runs. On expiry the worker is abandoned: it keeps going until the provider's
socket timeout or TCP teardown ends it, and it records itself in the call ledger
whenever that happens (so a timed-out call still appears with its cost, unless
the process exits first). Only the caller is released.

That is also why this uses a bare ``threading.Thread(daemon=True)`` rather than
a ``concurrent.futures`` executor. Two properties of ``ThreadPoolExecutor`` make
it the wrong tool here:

* ``with ThreadPoolExecutor(...)`` calls ``shutdown(wait=True)`` on exit, which
  blocks until the abandoned call finishes — so the timeout reported late and
  bounded nothing. That was the original defect.
* Even ``shutdown(wait=False)`` does not fix it: ``concurrent.futures.thread``
  registers ``_python_exit`` through ``threading._register_atexit``, and that
  hook ``join()``s every worker thread at interpreter shutdown. A hung request
  would therefore block process exit. A daemon thread is not joined.
"""

import logging
import signal
import threading
from typing import Any

from .config import get_llm_timeout

logger = logging.getLogger(__name__)

#: SIGALRM is Unix-only. On Windows ``signal.SIGALRM`` does not exist, so the
#: signal path would raise ``AttributeError`` instead of timing out.
_HAVE_SIGALRM = hasattr(signal, "SIGALRM")


class LLMTimeoutError(Exception):
    """Raised when an LLM call exceeds the configured timeout."""


def _timeout_handler(signum, frame):
    raise LLMTimeoutError("LLM call timed out - possible quota/rate limit issue")


def _is_main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


def invoke_with_timeout(llm, prompt, timeout_seconds: int = None) -> Any:
    """
    Invoke *llm* with a timeout.

    On the **main thread of a platform that has SIGALRM** the signal-based path
    is used (zero overhead, and it interrupts the request itself). Otherwise a
    watched daemon thread is used: from a background thread, because
    ``signal.signal`` raises ``signal only works in main thread`` there, and on
    Windows, because ``signal.SIGALRM`` does not exist.

    The two paths differ in what they interrupt — see the module docstring. The
    signal path raises *inside* the request, so the request stops. The thread
    path releases the caller and abandons the request.

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

    if _is_main_thread() and _HAVE_SIGALRM:
        return _invoke_signal(llm, prompt, timeout_seconds)
    return _invoke_thread(llm, prompt, timeout_seconds)


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
            None,
            duration_s=time.monotonic() - started,
            model=getattr(llm, "model_name", None),
            prompt=prompt,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    ledger.record(
        response,
        duration_s=time.monotonic() - started,
        model=getattr(llm, "model_name", None),
        prompt=prompt,
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


def _invoke_thread(llm, prompt, timeout_seconds: int) -> Any:
    """Daemon-thread timeout — safe from any thread, and actually bounded.

    Returns when the call completes, re-raising whatever it raised, or raises
    :class:`LLMTimeoutError` the moment the deadline passes. It never waits on
    the worker past the deadline; see the module docstring for what that costs.
    """
    outcome: dict = {}

    def _run() -> None:
        try:
            outcome["value"] = _do_invoke(llm, prompt)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc

    worker = threading.Thread(
        target=_run,
        name="aure-llm-invoke",
        daemon=True,
    )
    worker.start()
    worker.join(timeout_seconds)

    if worker.is_alive():
        # Deliberately not joined. `outcome` stays owned by the abandoned
        # worker, which may still write into it; nothing reads it after this.
        logger.warning(
            "[LLM] Call exceeded %ss and was abandoned; the request may still "
            "be in flight and will be recorded in the ledger if it returns",
            timeout_seconds,
        )
        raise LLMTimeoutError("LLM call timed out - possible quota/rate limit issue")

    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
