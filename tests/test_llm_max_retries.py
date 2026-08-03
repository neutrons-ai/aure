"""Tests for LLM_MAX_RETRIES on the OpenAI-compatible providers.

A local Ollama server evicts an idle model after ~5 minutes, and AuRE's refl1d
fits run far longer, so the LLM call after a fit routinely lands mid-reload.
With max_retries=0 a single blip aborted the whole analysis.
"""

from __future__ import annotations

import os

import pytest

from aure.llm.providers import openai_compat


@pytest.fixture(autouse=True)
def _clear_env():
    saved = os.environ.pop("LLM_MAX_RETRIES", None)
    yield
    os.environ.pop("LLM_MAX_RETRIES", None)
    if saved is not None:
        os.environ["LLM_MAX_RETRIES"] = saved


def test_retries_by_default():
    assert openai_compat._max_retries() == openai_compat._DEFAULT_MAX_RETRIES
    assert openai_compat._max_retries() > 0


@pytest.mark.parametrize("raw,expected", [("0", 0), ("5", 5), ("  3  ", 3)])
def test_env_override(raw, expected):
    os.environ["LLM_MAX_RETRIES"] = raw
    assert openai_compat._max_retries() == expected


def test_negative_is_clamped_to_zero():
    os.environ["LLM_MAX_RETRIES"] = "-2"
    assert openai_compat._max_retries() == 0


@pytest.mark.parametrize("raw", ["", "lots", "2.5"])
def test_unusable_values_fall_back_to_default(raw):
    os.environ["LLM_MAX_RETRIES"] = raw
    assert openai_compat._max_retries() == openai_compat._DEFAULT_MAX_RETRIES


def test_value_reaches_the_client():
    client = openai_compat.create_openai_compatible(
        {"model": "gpt-oss:120b"},
        0.0,
        api_key="not-a-real-key",
        base_url="http://localhost:11434/v1",
    )
    assert client.max_retries == openai_compat._DEFAULT_MAX_RETRIES

    os.environ["LLM_MAX_RETRIES"] = "0"
    fail_fast = openai_compat.create_openai_compatible(
        {"model": "gpt-oss:120b"},
        0.0,
        api_key="not-a-real-key",
        base_url="http://localhost:11434/v1",
    )
    assert fail_fast.max_retries == 0
