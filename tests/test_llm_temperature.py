"""Tests for how the configured temperature reaches the provider.

Three things went wrong together before this was fixed:

* every node called ``get_llm(temperature=0)``, so ``LLM_TEMPERATURE`` had no
  effect anywhere;
* langchain-openai drops any temperature but 1 for "gpt-5*" model names, so
  Azure's gpt-5.4, which accepts 0, silently ran at its default;
* a model that accepts only its default (gpt-6.1-sol) refused every call.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from aure.llm import ledger, providers
from aure.llm.providers import openai_compat

BASE = "http://example.invalid/v1"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key in ("LLM_TEMPERATURE", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_API_KEY", "x")
    openai_compat._TEMPERATURE_REFUSED.clear()
    yield
    openai_compat._TEMPERATURE_REFUSED.clear()


def _refusal() -> openai.BadRequestError:
    request = httpx.Request("POST", f"{BASE}/chat/completions")
    message = "Unsupported value: 'temperature' does not support 0.0 with this model."
    return openai.BadRequestError(
        message,
        response=httpx.Response(400, request=request),
        body={"message": message, "type": "invalid_request_error",
              "param": "temperature", "code": "unsupported_value"},
    )


def _result(text: str = "ok") -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def test_llm_temperature_is_honoured(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-oss-120b")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    assert providers.get_llm().temperature == 0.3


def test_no_node_overrides_the_configured_temperature():
    src = Path(providers.__file__).resolve().parents[2]
    offenders = [
        f"{path.relative_to(src)}:{n}"
        for path in src.rglob("*.py")
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "get_llm(temperature=" in line
    ]
    assert offenders == []


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5-mini", "gpt-oss-120b"])
def test_a_gpt5_name_keeps_the_temperature_asked_for(monkeypatch, model):
    monkeypatch.setenv("LLM_MODEL", model)
    assert providers.get_llm().temperature == 0.0


def test_a_refused_temperature_is_dropped_and_remembered(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-6.1-sol")
    sent = []

    def fake_generate(self, *args, **kwargs):
        sent.append(self.temperature)
        if self.temperature is not None:
            raise _refusal()
        return _result()

    monkeypatch.setattr(ChatOpenAI, "_generate", fake_generate)
    llm = providers.get_llm()
    assert llm.invoke("hi").content == "ok"
    assert sent == [0.0, None]
    assert llm.temperature is None
    # The next client for this model starts without one; other models keep 0.
    assert providers.get_llm().temperature is None
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    assert providers.get_llm().temperature == 0.0


def test_other_errors_are_not_retried(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    calls = []

    def fake_generate(self, *args, **kwargs):
        calls.append(self.temperature)
        raise ValueError("boom")

    monkeypatch.setattr(ChatOpenAI, "_generate", fake_generate)
    with pytest.raises(ValueError):
        providers.get_llm().invoke("hi")
    assert calls == [0.0]
    assert openai_compat._TEMPERATURE_REFUSED == set()


def test_the_ledger_records_the_temperature_sent(monkeypatch, tmp_path):
    from aure.llm.timeout import invoke_with_timeout

    monkeypatch.setenv("LLM_MODEL", "gpt-6.1-sol")

    def fake_generate(self, *args, **kwargs):
        if self.temperature is not None:
            raise _refusal()
        return _result()

    monkeypatch.setattr(ChatOpenAI, "_generate", fake_generate)
    sink = tmp_path / "llm_calls.jsonl"
    ledger.set_sink(sink)
    try:
        invoke_with_timeout(providers.get_llm(), "hi")
        monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
        monkeypatch.setattr(ChatOpenAI, "_generate", lambda self, *a, **k: _result())
        invoke_with_timeout(providers.get_llm(), "hi")
    finally:
        ledger.set_sink(None)
    rows = [json.loads(line) for line in sink.read_text().splitlines()]
    assert [row["temperature"] for row in rows] == [None, 0.0]
