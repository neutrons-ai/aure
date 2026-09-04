"""``invoke_with_timeout`` — that the deadline actually bounds the wait.

The defect this file pins down: the thread path used
``with ThreadPoolExecutor(...)``, whose ``__exit__`` calls
``shutdown(wait=True)``. On timeout it therefore blocked until the abandoned
call finished and only *then* raised ``LLMTimeoutError`` — reporting a timeout
late while bounding nothing. Since every LLM call from the Flask web UI runs on
a background thread, ``LLM_TIMEOUT`` had no effect there at all.
"""

import threading
import time

import pytest

from aure.llm.timeout import (
    LLMTimeoutError,
    _HAVE_SIGALRM,
    _invoke_signal,
    _invoke_thread,
    invoke_with_timeout,
)


class _Slow:
    """A provider whose call takes far longer than the deadline."""

    model_name = "slow-model"

    def __init__(self, seconds=5.0):
        self.seconds = seconds
        self.started = threading.Event()

    def invoke(self, payload):
        self.started.set()
        time.sleep(self.seconds)
        return "a reply nobody is waiting for any more"


class _Fast:
    model_name = "fast-model"

    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.reply


class _Broken:
    model_name = "broken-model"

    def invoke(self, payload):
        raise RuntimeError("provider said no")


def _elapsed(fn, *a, **kw):
    """Run *fn*, returning (elapsed_seconds, raised_exception_or_None)."""
    t0 = time.monotonic()
    try:
        fn(*a, **kw)
        return time.monotonic() - t0, None
    except BaseException as exc:  # noqa: BLE001 - the test inspects it
        return time.monotonic() - t0, exc


# ----------------------------------------------------------------------
# The regression
# ----------------------------------------------------------------------


def test_the_thread_path_returns_at_the_deadline_not_at_call_completion():
    """The bug: this waited the full 5 s before raising.

    A 1 s deadline against a 5 s call must release the caller at ~1 s. The
    assertion is deliberately loose (< 3 s) so it fails on the old blocking
    behaviour without being flaky on a loaded machine.
    """
    llm = _Slow(seconds=5.0)
    elapsed, exc = _elapsed(_invoke_thread, llm, [("human", "hi")], 1)

    assert isinstance(exc, LLMTimeoutError)
    assert llm.started.is_set(), "the provider was never actually called"
    assert elapsed < 3.0, (
        f"waited {elapsed:.2f}s for a 1s deadline — the timeout is not bounding "
        "the wait (the executor's shutdown(wait=True) regression)"
    )


def test_a_background_thread_caller_is_bounded_too():
    """The path the Flask web UI actually takes.

    ``invoke_with_timeout`` dispatches on thread identity, so this exercises the
    dispatch as well as the mechanism.
    """
    llm = _Slow(seconds=5.0)
    result = {}

    def caller():
        result["elapsed"], result["exc"] = _elapsed(
            invoke_with_timeout, llm, [("human", "hi")], 1
        )

    t = threading.Thread(target=caller)
    t.start()
    t.join(15)
    assert not t.is_alive(), "the calling thread never came back"

    assert isinstance(result["exc"], LLMTimeoutError)
    assert result["elapsed"] < 3.0


def test_the_abandoned_worker_does_not_block_interpreter_exit():
    """Why this is a daemon thread and not a futures executor.

    ``concurrent.futures.thread`` registers ``_python_exit`` via
    ``threading._register_atexit``, and that hook ``join()``s every worker — so
    even ``shutdown(wait=False)`` would let a hung request block process exit.
    """
    llm = _Slow(seconds=30.0)
    with pytest.raises(LLMTimeoutError):
        _invoke_thread(llm, [("human", "hi")], 1)

    workers = [
        t for t in threading.enumerate() if t.name == "aure-llm-invoke" and t.is_alive()
    ]
    assert workers, "expected the abandoned worker to still be running"
    assert all(t.daemon for t in workers), (
        "the abandoned worker must be a daemon thread, or a hung LLM call "
        "blocks interpreter shutdown"
    )


# ----------------------------------------------------------------------
# The non-timeout paths still behave
# ----------------------------------------------------------------------


def test_a_call_that_finishes_in_time_returns_its_result():
    llm = _Fast(reply="the answer")
    assert _invoke_thread(llm, [("human", "hi")], 30) == "the answer"
    assert len(llm.calls) == 1


def test_a_provider_error_propagates_unchanged():
    """The caller must see the provider's own exception, not a timeout."""
    with pytest.raises(RuntimeError, match="provider said no"):
        _invoke_thread(_Broken(), [("human", "hi")], 30)


def test_a_string_prompt_is_wrapped_as_a_human_message():
    llm = _Fast()
    _invoke_thread(llm, "plain text prompt", 30)
    (payload,) = llm.calls
    assert len(payload) == 1
    assert payload[0].content == "plain text prompt"


# ----------------------------------------------------------------------
# Path selection
# ----------------------------------------------------------------------


def test_the_main_thread_uses_signals_only_where_sigalrm_exists():
    """On Windows ``signal.SIGALRM`` is absent, so the signal path would raise
    AttributeError rather than time out; dispatch must fall through instead."""
    import aure.llm.timeout as mod

    calls = []
    orig_signal, orig_thread = mod._invoke_signal, mod._invoke_thread
    mod._invoke_signal = lambda *a: calls.append("signal")
    mod._invoke_thread = lambda *a: calls.append("thread")
    try:
        mod._HAVE_SIGALRM = True
        invoke_with_timeout(_Fast(), "hi", 30)
        assert calls == ["signal"]

        calls.clear()
        mod._HAVE_SIGALRM = False  # simulate Windows
        invoke_with_timeout(_Fast(), "hi", 30)
        assert calls == ["thread"], "no SIGALRM must fall through to the thread path"
    finally:
        mod._HAVE_SIGALRM = _HAVE_SIGALRM
        mod._invoke_signal, mod._invoke_thread = orig_signal, orig_thread


@pytest.mark.skipif(not _HAVE_SIGALRM, reason="SIGALRM is Unix-only")
def test_the_signal_path_restores_the_previous_handler_and_cancels_the_alarm():
    """A leaked alarm would fire during unrelated later work."""
    import signal

    sentinel = signal.getsignal(signal.SIGALRM)

    _invoke_signal(_Fast(reply="done"), [("human", "hi")], 30)

    assert signal.getsignal(signal.SIGALRM) is sentinel
    assert signal.alarm(0) == 0, "an alarm was left pending"
