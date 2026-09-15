"""Per-call ledger of LLM invocations, written as JSONL.

``state.llm_calls`` records that a call happened, at which node, and whether it
fell back. It does not record what the call cost. That gap is why the
nr-workbench comparison could report ~178 model calls and ~$11 per analysed
curve for the agentic arm and nothing at all for this one: nr-workbench keeps a
full session record per sample, and AuRE kept only counts.

This module closes it at the single chokepoint every node goes through
(:func:`aure.llm.timeout.invoke_with_timeout`), writing two files side by side.

**The ledger** — ``<output_dir>/llm_calls.jsonl``, one object per call,
always written once a sink is set::

    {"seq", "timestamp", "node", "model", "provider", "duration_s",
     "input_tokens", "output_tokens", "total_tokens", "cached_tokens",
     "cost_usd", "ok", "error"}

**The trace** — ``<output_dir>/llm_trace.jsonl``, one object per call, written
only when ``AURE_LLM_LOG_TEXT`` is set::

    {"seq", "timestamp", "node", "model", "provider", "messages",
     "completion", "ok", "error"}

``seq`` is the join key between the two, allocated once under one lock so the
two files cannot drift apart when one of them is disabled or fails to write.

Design notes
------------
* **Measurement only.** Nothing here alters a prompt, a retry, or a decision.
  A failure to record is swallowed: instrumentation must never be able to fail
  an analysis. The trace is written after the ledger row and in its own guard,
  so a bad prompt object can never cost the ledger a row.
* **Off unless a sink is set.** Set automatically when the CLI is given ``-o``;
  override the destination with ``AURE_LLM_LOG`` (useful for a batch harness
  that wants every case in one file, or for a run without an output directory).
  The trace is written beside whatever the ledger resolves to.
* **Token counts come from the provider's own response.** LangChain normalises
  these onto ``usage_metadata``; older providers only populate
  ``response_metadata["token_usage"]``, so both are read. A provider that
  reports neither leaves the fields ``None`` rather than zero — absent and free
  are different, and a zero would quietly corrupt a cost total. ``cost_usd``
  follows the same rule.
* **Why the exchange is a separate file, and opt-in.** A prompt carries the
  sample description and the user's hypothesis, which is their data and should
  not land on disk unless asked for. Keeping it out of the ledger also keeps
  the ledger cheap to read: a cost total should not have to stream megabytes of
  prompt text to sum a column.

  ``AURE_LLM_LOG_TEXT`` previously added *truncated* (4000-character) copies of
  the prompt and response to each ledger row. It now writes the full exchange
  to the trace file instead, and the ledger row carries no text at all.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_lock = threading.Lock()
_sink: Optional[Path] = None
_trace: Optional[Path] = None
_node: Optional[str] = None
_seq = 0

#: The trace is always written beside the ledger, under this name.
_TRACE_NAME = "llm_trace.jsonl"

#: LangChain message types, mapped onto the role names the rest of the world
#: uses. Recorded normalised so a trace can be replayed against any provider.
_ROLE_ALIASES = {"human": "user", "ai": "assistant"}


def set_sink(path: Optional[os.PathLike | str]) -> None:
    """Point the ledger at a JSONL file (``None`` disables it).

    The trace file is derived from the result, so ``AURE_LLM_LOG`` moves both.
    """
    global _sink, _trace
    override = os.environ.get("AURE_LLM_LOG")
    target = override or path
    _sink = Path(target) if target else None
    if _sink is not None:
        try:
            _sink.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            _sink = None
    _trace = (_sink.parent / _TRACE_NAME) if _sink is not None else None


def sink() -> Optional[Path]:
    return _sink


def trace_sink() -> Optional[Path]:
    """Where the exchange is written, or ``None`` when the ledger is off."""
    return _trace


def trace_enabled() -> bool:
    """Whether the full exchange is being recorded (``AURE_LLM_LOG_TEXT``)."""
    return bool(os.environ.get("AURE_LLM_LOG_TEXT")) and _trace is not None


def set_node(name: Optional[str]) -> None:
    """Label subsequent calls with the workflow node making them."""
    global _node
    _node = name


def _usage(response: Any) -> dict:
    """Token counts from a LangChain response, or Nones when unreported."""
    out = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cached_tokens": None,
    }
    if response is None:
        return out
    um = getattr(response, "usage_metadata", None)
    if isinstance(um, dict) and um:
        out["input_tokens"] = um.get("input_tokens")
        out["output_tokens"] = um.get("output_tokens")
        out["total_tokens"] = um.get("total_tokens")
        details = um.get("input_token_details") or {}
        if isinstance(details, dict):
            out["cached_tokens"] = details.get("cache_read")
        return out
    meta = getattr(response, "response_metadata", None) or {}
    tu = meta.get("token_usage") or meta.get("usage") or {}
    if isinstance(tu, dict) and tu:
        out["input_tokens"] = tu.get("prompt_tokens") or tu.get("input_tokens")
        out["output_tokens"] = tu.get("completion_tokens") or tu.get("output_tokens")
        out["total_tokens"] = tu.get("total_tokens")
    return out


def _provider() -> Optional[str]:
    """The active provider name, for rows the caller did not label.

    The column is what makes a mixed archive summable: a ``cost_usd`` of
    ``None`` means something different for a provider that never prices a call
    than for one that usually does and did not this time.
    """
    try:
        from .config import get_llm_config

        return get_llm_config().get("provider") or None
    except Exception:
        return None


def _cost(response: Any) -> Optional[float]:
    """What the provider says this call cost, in USD, or ``None``.

    Only a provider that prices its own call reports this — the Claude Code
    CLI does, and it is the authority, since the price depends on cache state
    and on which backend the account is configured for. Everything else leaves
    it ``None``: unknown, which is not the same as free.
    """
    if response is None:
        return None
    meta = getattr(response, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    raw = meta.get("total_cost_usd")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _answering_model(response: Any) -> Optional[str]:
    """The model that actually answered, when the response names one.

    The caller passes the model it *asked* for, which is not always a name. The
    ``claude_code`` provider lets the CLI resolve its own model and reports the
    result here; ``ChatOpenAI`` reports the served model the same way. Either
    beats the request for the purposes of a row someone will later sum by model.
    """
    meta = getattr(response, "response_metadata", None) or {}
    if isinstance(meta, dict):
        name = meta.get("model") or meta.get("model_name")
        if name:
            return str(name)
    return None


def _text(content: Any) -> str:
    """Flatten message content to text, including provider block lists."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _one_message(item: Any) -> tuple:
    """Normalise one message to ``(role, text)``."""
    content = getattr(item, "content", None)
    if content is not None:
        role = str(getattr(item, "type", None) or "unknown")
        return _ROLE_ALIASES.get(role, role), _text(content)
    if isinstance(item, dict):
        role = str(item.get("role") or item.get("type") or "unknown")
        return _ROLE_ALIASES.get(role, role), _text(item.get("content"))
    if isinstance(item, (list, tuple)) and len(item) == 2:
        role, content = item
        role = str(role)
        return _ROLE_ALIASES.get(role, role), _text(content)
    return "unknown", _text(item)


def _messages(prompt: Any) -> list:
    """Normalise whatever was sent into ``[{"role", "content"}, …]``.

    Three shapes reach this chokepoint: a bare string, a list of LangChain
    message objects, and a list of ``(role, content)`` tuples — nr-workbench's
    ``aure_adapter.complete`` sends the last. Anything unrecognised is kept as
    text under role ``unknown`` rather than dropped: a trace that silently
    omits part of the prompt is worse than one that admits it did not
    understand it.
    """
    if prompt is None:
        return []
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, (list, tuple)):
        prompt = [prompt]
    out = []
    for item in prompt:
        role, content = _one_message(item)
        out.append({"role": role, "content": content})
    return out


def _completion(response: Any) -> Optional[str]:
    """The reply exactly as received, untruncated, or ``None`` on a failure.

    A provider that post-processes its own reply before handing it on — the
    ``claude_code`` shim strips fences and lifts JSON out of prose — exposes
    the untouched text as ``raw_content``, and that is what gets recorded. The
    trace exists to show what the model actually said; recording the shim's
    improved version of it would hide exactly the failure the trace is there
    to catch.
    """
    if response is None:
        return None
    raw = getattr(response, "raw_content", None)
    if raw is not None:
        return _text(raw)
    content = getattr(response, "content", None)
    if content is None:
        return _text(response)
    return _text(content)


def _trace_row(
    response: Any, *, model, provider, prompt, ok, error, timestamp: str
) -> Optional[dict]:
    """Build the trace object, or ``None`` if it cannot be built.

    Guarded separately from the ledger row so that an exotic prompt object
    costs the trace and not the ledger.
    """
    try:
        return {
            "seq": None,
            "timestamp": timestamp,
            "node": _node,
            "model": model,
            "provider": provider,
            "messages": _messages(prompt),
            "completion": _completion(response),
            "ok": ok,
            "error": error,
        }
    except Exception:
        return None


def record(
    response: Any,
    *,
    duration_s: float,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    prompt: Any = None,
    ok: bool = True,
    error: Optional[str] = None,
) -> None:
    """Append one call to the ledger, and the exchange to the trace.

    Never raises.
    """
    global _seq
    if _sink is None:
        return
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        provider = provider or _provider()
        model = _answering_model(response) or model
        entry = {
            "seq": None,
            "timestamp": timestamp,
            "node": _node,
            "model": model,
            "provider": provider,
            "duration_s": round(duration_s, 3),
            "ok": ok,
            "error": error,
        }
        # Always emit the usage keys, even on a failure, so every row has the
        # same schema and a consumer can sum a column without guarding each
        # cell. A failed call reports None (unknown), never 0 (free).
        entry.update(_usage(response))
        entry["cost_usd"] = _cost(response)

        trace = None
        if trace_enabled():
            trace = _trace_row(
                response,
                model=model,
                provider=provider,
                prompt=prompt,
                ok=ok,
                error=error,
                timestamp=timestamp,
            )

        with _lock:
            _seq += 1
            entry["seq"] = _seq
            with _sink.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
            # After the ledger row, and in its own guard: the trace is the
            # expendable half of the pair, and losing it must not lose the
            # row that joins to it.
            if trace is not None:
                trace["seq"] = _seq
                try:
                    with _trace.open("a") as fh:
                        fh.write(json.dumps(trace) + "\n")
                except Exception:
                    pass
    except Exception:  # instrumentation must never fail an analysis
        pass
