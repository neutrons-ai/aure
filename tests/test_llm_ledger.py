"""The call ledger and the exchange trace.

Two properties are load-bearing and easy to lose:

* ``cost_usd`` must stay tri-state. ``None`` means the provider did not price
  the call; ``0.0`` would mean it was free. Collapsing the two silently
  corrupts any total summed over a mixed-provider archive.
* ``seq`` must be the same number in both files. It is the only join between a
  cost row and the exchange that produced it, and it is allocated while the
  ledger row is written — so a trace that counts for itself drifts the moment
  one of the two is disabled or fails.
"""

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from aure.llm import ledger


class _Reply:
    """A provider response, in the shape LangChain hands back."""

    def __init__(self, content="a reply", cost=None, usage=None):
        self.content = content
        self.response_metadata = {}
        if cost is not None:
            self.response_metadata["total_cost_usd"] = cost
        self.usage_metadata = usage or {}


@pytest.fixture
def ledger_dir(tmp_path, monkeypatch):
    """A fresh ledger pointed at a temp directory, with globals reset."""
    monkeypatch.delenv("AURE_LLM_LOG", raising=False)
    monkeypatch.delenv("AURE_LLM_LOG_TEXT", raising=False)
    monkeypatch.setattr(ledger, "_seq", 0)
    monkeypatch.setattr(ledger, "_node", None)
    ledger.set_sink(str(tmp_path / "llm_calls.jsonl"))
    yield tmp_path
    ledger.set_sink(None)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


# ── The ledger row ──────────────────────────────────────────────────────


def test_cost_recorded_when_the_provider_prices_the_call(ledger_dir):
    ledger.record(_Reply(cost=0.0165), duration_s=1.5)
    (row,) = _rows(ledger_dir / "llm_calls.jsonl")
    assert row["cost_usd"] == 0.0165


def test_cost_is_none_not_zero_when_unreported(ledger_dir):
    """Absent and free are different; a zero corrupts a total."""
    ledger.record(_Reply(), duration_s=1.5)
    (row,) = _rows(ledger_dir / "llm_calls.jsonl")
    assert row["cost_usd"] is None


def test_cost_survives_a_nonsense_value(ledger_dir):
    ledger.record(_Reply(cost="not a number"), duration_s=1.0)
    (row,) = _rows(ledger_dir / "llm_calls.jsonl")
    assert row["cost_usd"] is None


def test_a_failed_call_still_gets_a_row(ledger_dir):
    ledger.record(None, duration_s=0.2, ok=False, error="LLMTimeoutError: x")
    (row,) = _rows(ledger_dir / "llm_calls.jsonl")
    assert row["ok"] is False
    assert row["cost_usd"] is None
    assert row["input_tokens"] is None


def test_ledger_row_carries_no_prompt_text(ledger_dir, monkeypatch):
    """The text lives in the trace now, so a cost total need not stream it."""
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    ledger.record(_Reply(), duration_s=1.0, prompt="the sample description")
    (row,) = _rows(ledger_dir / "llm_calls.jsonl")
    assert "prompt" not in row
    assert "response" not in row


# ── The trace ───────────────────────────────────────────────────────────


def test_trace_is_off_by_default(ledger_dir):
    ledger.record(_Reply(), duration_s=1.0, prompt="private sample description")
    assert not (ledger_dir / "llm_trace.jsonl").exists()


def test_trace_records_the_whole_exchange_untruncated(ledger_dir, monkeypatch):
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    long_prompt = "x" * 9000
    long_reply = "y" * 9000
    ledger.record(_Reply(content=long_reply), duration_s=1.0, prompt=long_prompt)
    (row,) = _rows(ledger_dir / "llm_trace.jsonl")
    assert row["messages"] == [{"role": "user", "content": long_prompt}]
    assert row["completion"] == long_reply


def test_seq_joins_the_two_files(ledger_dir, monkeypatch):
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    for i in range(3):
        ledger.record(_Reply(content=f"reply {i}"), duration_s=1.0, prompt=f"ask {i}")
    calls = _rows(ledger_dir / "llm_calls.jsonl")
    traces = _rows(ledger_dir / "llm_trace.jsonl")
    assert [r["seq"] for r in calls] == [1, 2, 3]
    assert [r["seq"] for r in traces] == [1, 2, 3]
    assert traces[2]["completion"] == "reply 2"


def test_seq_does_not_drift_when_the_trace_is_enabled_midway(ledger_dir, monkeypatch):
    """The ledger numbers the calls; the trace only ever joins to it."""
    ledger.record(_Reply(), duration_s=1.0, prompt="first, untraced")
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    ledger.record(_Reply(), duration_s=1.0, prompt="second, traced")

    calls = _rows(ledger_dir / "llm_calls.jsonl")
    (trace,) = _rows(ledger_dir / "llm_trace.jsonl")
    assert [r["seq"] for r in calls] == [1, 2]
    assert trace["seq"] == 2
    assert trace["messages"][0]["content"] == "second, traced"


def test_node_label_reaches_both_files(ledger_dir, monkeypatch):
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    ledger.set_node("modeling")
    ledger.record(_Reply(), duration_s=1.0, prompt="ask")
    assert _rows(ledger_dir / "llm_calls.jsonl")[0]["node"] == "modeling"
    assert _rows(ledger_dir / "llm_trace.jsonl")[0]["node"] == "modeling"


# ── Message shapes ──────────────────────────────────────────────────────


def test_langchain_messages_are_normalised(ledger_dir, monkeypatch):
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    ledger.record(
        _Reply(),
        duration_s=1.0,
        prompt=[SystemMessage(content="be terse"), HumanMessage(content="fit this")],
    )
    (row,) = _rows(ledger_dir / "llm_trace.jsonl")
    assert row["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "fit this"},
    ]


def test_role_tuples_are_normalised(ledger_dir, monkeypatch):
    """nr-workbench's aure_adapter.complete sends this shape."""
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    ledger.record(
        _Reply(), duration_s=1.0, prompt=[("system", "be terse"), ("human", "fit this")]
    )
    (row,) = _rows(ledger_dir / "llm_trace.jsonl")
    assert row["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "fit this"},
    ]


def test_block_list_content_is_flattened(ledger_dir, monkeypatch):
    """Anthropic-shaped content arrives as a list of typed blocks."""
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    reply = _Reply(
        content=[
            {"type": "text", "text": "half "},
            {"type": "text", "text": "and half"},
        ]
    )
    ledger.record(reply, duration_s=1.0, prompt="ask")
    (row,) = _rows(ledger_dir / "llm_trace.jsonl")
    assert row["completion"] == "half and half"


def test_an_unrecognised_prompt_object_is_kept_as_text(ledger_dir, monkeypatch):
    """Admitting it was not understood beats silently dropping it."""
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")

    class _Odd:
        def __str__(self):
            return "some exotic prompt"

    ledger.record(_Reply(), duration_s=1.0, prompt=[_Odd()])
    (row,) = _rows(ledger_dir / "llm_trace.jsonl")
    assert row["messages"] == [{"role": "unknown", "content": "some exotic prompt"}]


# ── Failure isolation ───────────────────────────────────────────────────


def test_a_broken_trace_does_not_cost_the_ledger_a_row(ledger_dir, monkeypatch):
    """The trace is the expendable half of the pair."""
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    monkeypatch.setattr(
        ledger, "_trace_row", lambda *a, **k: {"seq": None, "bad": {1, 2}}
    )
    ledger.record(_Reply(cost=0.02), duration_s=1.0, prompt="ask")
    (row,) = _rows(ledger_dir / "llm_calls.jsonl")
    assert row["cost_usd"] == 0.02


def test_recording_without_a_sink_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.delenv("AURE_LLM_LOG", raising=False)
    ledger.set_sink(None)
    ledger.record(_Reply(), duration_s=1.0, prompt="ask")
    assert not list(tmp_path.iterdir())


# ── Destination ─────────────────────────────────────────────────────────


def test_aure_llm_log_moves_both_files(tmp_path, monkeypatch):
    elsewhere = tmp_path / "sweep"
    monkeypatch.setenv("AURE_LLM_LOG", str(elsewhere / "all_calls.jsonl"))
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    monkeypatch.setattr(ledger, "_seq", 0)
    ledger.set_sink(str(tmp_path / "ignored.jsonl"))
    try:
        ledger.record(_Reply(), duration_s=1.0, prompt="ask")
        assert (elsewhere / "all_calls.jsonl").exists()
        assert (elsewhere / "llm_trace.jsonl").exists()
        assert not (tmp_path / "ignored.jsonl").exists()
    finally:
        ledger.set_sink(None)


def test_the_answering_model_wins_over_the_requested_one(ledger_dir, monkeypatch):
    """claude_code lets the CLI resolve a model; the row should name the result."""
    monkeypatch.setenv("AURE_LLM_LOG_TEXT", "1")
    reply = _Reply()
    reply.response_metadata["model"] = "claude-sonnet-5"
    ledger.record(reply, duration_s=1.0, model="claude-code-default", prompt="ask")
    assert _rows(ledger_dir / "llm_calls.jsonl")[0]["model"] == "claude-sonnet-5"
    assert _rows(ledger_dir / "llm_trace.jsonl")[0]["model"] == "claude-sonnet-5"


def test_the_requested_model_stands_when_the_response_names_none(ledger_dir):
    ledger.record(_Reply(), duration_s=1.0, model="gpt-4o-mini")
    assert _rows(ledger_dir / "llm_calls.jsonl")[0]["model"] == "gpt-4o-mini"
