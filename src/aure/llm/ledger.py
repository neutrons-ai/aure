"""Per-call ledger of LLM invocations, written as JSONL.

``state.llm_calls`` records that a call happened, at which node, and whether it
fell back. It does not record what the call cost. That gap is why the
nr-workbench comparison could report ~178 model calls and ~$11 per analysed
curve for the agentic arm and nothing at all for this one: nr-workbench keeps a
full session record per sample, and AuRE kept only counts.

This module closes it at the single chokepoint every node goes through
(:func:`aure.llm.timeout.invoke_with_timeout`), writing one JSON object per call
to ``<output_dir>/llm_calls.jsonl``:

    {"seq", "timestamp", "node", "model", "provider", "duration_s",
     "input_tokens", "output_tokens", "total_tokens", "cached_tokens",
     "ok", "error"}

Design notes
------------
* **Measurement only.** Nothing here alters a prompt, a retry, or a decision.
  A failure to record is swallowed: instrumentation must never be able to fail
  an analysis.
* **Off unless a sink is set.** Set automatically when the CLI is given ``-o``;
  override the destination with ``AURE_LLM_LOG`` (useful for a batch harness
  that wants every case in one file, or for a run without an output directory).
* **Token counts come from the provider's own response.** LangChain normalises
  these onto ``usage_metadata``; older providers only populate
  ``response_metadata["token_usage"]``, so both are read. A provider that
  reports neither leaves the fields ``None`` rather than zero — absent and free
  are different, and a zero would quietly corrupt a cost total.
* **Prompt/response text is NOT recorded by default.** Set
  ``AURE_LLM_LOG_TEXT=1`` to include truncated copies for provenance.
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
_node: Optional[str] = None
_seq = 0

#: Characters of prompt/response kept when AURE_LLM_LOG_TEXT is set.
_TEXT_LIMIT = 4000


def set_sink(path: Optional[os.PathLike | str]) -> None:
    """Point the ledger at a JSONL file (``None`` disables it)."""
    global _sink
    override = os.environ.get("AURE_LLM_LOG")
    target = override or path
    _sink = Path(target) if target else None
    if _sink is not None:
        try:
            _sink.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            _sink = None


def sink() -> Optional[Path]:
    return _sink


def set_node(name: Optional[str]) -> None:
    """Label subsequent calls with the workflow node making them."""
    global _node
    _node = name


def _usage(response: Any) -> dict:
    """Token counts from a LangChain response, or Nones when unreported."""
    out = {"input_tokens": None, "output_tokens": None,
           "total_tokens": None, "cached_tokens": None}
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


def record(response: Any, *, duration_s: float, model: Optional[str] = None,
           provider: Optional[str] = None, prompt: Any = None,
           ok: bool = True, error: Optional[str] = None) -> None:
    """Append one call to the ledger. Never raises."""
    global _seq
    if _sink is None:
        return
    try:
        entry = {
            "seq": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
        if os.environ.get("AURE_LLM_LOG_TEXT"):
            entry["prompt"] = str(prompt)[:_TEXT_LIMIT] if prompt is not None else None
            content = getattr(response, "content", None)
            entry["response"] = str(content)[:_TEXT_LIMIT] if content is not None else None
        with _lock:
            _seq += 1
            entry["seq"] = _seq
            with _sink.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
    except Exception:  # instrumentation must never fail an analysis
        pass
