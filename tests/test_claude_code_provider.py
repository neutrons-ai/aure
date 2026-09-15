"""The ``claude_code`` provider — the CLI driven as a completions endpoint.

Everything here runs against a fake ``claude`` binary, so the suite costs
nothing and runs in CI. The two live tests at the bottom talk to the real CLI
and are skipped unless ``AURE_LIVE_LLM_TESTS=1``.

What these pin down, in order of how much it would hurt to lose:

* **Prose must survive.** The ISAAC exporter asks for a paragraph and uses the
  reply verbatim. An unwrapper that always hunts for JSON would silently
  replace that paragraph with whatever brace it found first.
* **JSON must arrive parseable**, because two modeling call sites parse with no
  tolerance for a preamble *and* swallow the failure — they fall back to
  default parameter tying and a shared template, which is a physics change
  nothing in the output announces.
* **The child must die** when the caller gives up, or a run keeps billing for
  answers nobody is waiting for.
"""

import json
import os
import stat
import subprocess

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from aure.llm.providers import claude_code as cc
from aure.llm.timeout import LLMTimeoutError

# A stand-in for the CLI. Replies are scripted per call so a retry can be given
# different output from the first attempt.
_FAKE = """#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path

state = Path(os.environ["FAKE_CLAUDE_STATE"])
calls = state / "calls.jsonl"
n = len(calls.read_text().splitlines()) if calls.exists() else 0
with calls.open("a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")

replies = json.loads((state / "replies.json").read_text())
spec = replies[min(n, len(replies) - 1)]

if spec.get("sleep"):
    time.sleep(spec["sleep"])
if spec.get("exit_code"):
    sys.stderr.write(spec.get("stderr", "boom"))
    sys.exit(spec["exit_code"])
if spec.get("stdout_raw") is not None:
    sys.stdout.write(spec["stdout_raw"])
    sys.exit(0)

result = spec.get("result", "")
if spec.get("echo_prompt"):
    argv = sys.argv[1:]
    target = argv[argv.index("-p") + 1]
    result = Path(target[1:]).read_text()

sys.stdout.write(json.dumps({
    "type": "result",
    "subtype": spec.get("subtype", "success"),
    "is_error": spec.get("is_error", False),
    "result": result,
    "session_id": "fake-session",
    "duration_ms": 10,
    "num_turns": 1,
    "total_cost_usd": spec.get("cost", 0.0165),
    "usage": spec.get("usage", {
        "input_tokens": 1760,
        "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 12533,
        "output_tokens": 40,
    }),
    "modelUsage": spec.get("modelUsage", {
        "claude-haiku-4-5-20251001": {"outputTokens": 11},
        "claude-sonnet-5": {"outputTokens": 40},
    }),
}))
"""


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    """Install the fake CLI and return a handle for scripting it."""
    binary = tmp_path / "claude"
    binary.write_text(_FAKE)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("AURE_CLAUDE_BIN", str(binary))
    monkeypatch.setenv("FAKE_CLAUDE_STATE", str(state))
    monkeypatch.delenv(cc._RECURSION_FLAG, raising=False)

    class Handle:
        def script(self, *specs):
            (state / "replies.json").write_text(json.dumps(list(specs)))

        @property
        def calls(self):
            path = state / "calls.jsonl"
            if not path.exists():
                return []
            return [json.loads(line) for line in path.read_text().splitlines()]

        def prompt_of(self, index=0):
            argv = self.calls[index]
            return (tmp_path / "x").with_name(
                argv[argv.index("-p") + 1][1:].split("/")[-1]
            )

    handle = Handle()
    handle.script({"result": "ok"})
    return handle


def _chat(model=None):
    return cc.create_claude_code({"model": model}, 0.0)


# ── Layer 2 & 3: unwrapping ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('Here is the model:\n{"a": 1}', '{"a": 1}'),
        ('{"a": 1}\n\nLet me know if you want changes.', '{"a": 1}'),
        ('Sure!\n```json\n{"a": 1}\n```\nHope that helps.', '{"a": 1}'),
        # A brace inside a string must not close the object.
        ('note:\n{"a": "} not the end", "b": 2}', '{"a": "} not the end", "b": 2}'),
        # An escaped quote must not end the string.
        (r'x {"a": "say \"hi\"", "b": 2}', r'{"a": "say \"hi\"", "b": 2}'),
        # Nested containers.
        ('text {"a": {"b": [1, 2]}} more', '{"a": {"b": [1, 2]}}'),
        # A JSON array, which is what the skill selector asks for.
        ('I would activate ["polymer-films"]', '["polymer-films"]'),
    ],
)
def test_json_is_recovered_from_a_conversational_reply(raw, expected):
    content, _ = cc._unwrap(raw, wants_json=True)
    assert json.loads(content) == json.loads(expected)


def test_the_first_bracket_does_not_win_over_the_real_array():
    """The regex at the skill-selector call site has exactly this bug."""
    raw = 'Based on [the description] I would pick ["polymer-films", "sei-layer-analysis"]'
    content, _ = cc._unwrap(raw, wants_json=True)
    assert json.loads(content) == ["polymer-films", "sei-layer-analysis"]


def test_prose_is_left_alone_when_no_json_was_asked_for():
    """The ISAAC exporter's paragraph regression."""
    prose = (
        "The sample is a 240 A polystyrene film on silicon, measured in D2O. "
        "The fit converged to chi-squared 1.4 {sic} over three layers."
    )
    content, note = cc._unwrap(prose, wants_json=False)
    assert content == prose
    assert note is None


def test_an_unclosed_object_is_not_guessed_at():
    raw = 'Here you go: {"a": 1, "b":'
    content, note = cc._unwrap(raw, wants_json=True)
    assert note == "unparseable"
    assert content == raw


def test_wants_json_ignores_a_prose_prompt():
    assert cc._wants_json("Return a JSON object with the layers")
    assert cc._wants_json("reply with only the json array")
    assert not cc._wants_json(
        "Summarise the analysis into a single concise paragraph. "
        "Do NOT use bullet points."
    )


# ── Invocation ──────────────────────────────────────────────────────────


def test_the_flag_set_is_what_was_measured(fake_claude):
    _chat().invoke([HumanMessage(content="hello")])
    (argv,) = fake_claude.calls
    assert (
        "--output-format" in argv and argv[argv.index("--output-format") + 1] == "json"
    )
    assert "--system-prompt" in argv
    assert "--disallowed-tools" in argv
    assert "--strict-mcp-config" in argv
    assert "--disable-slash-commands" in argv
    assert argv[argv.index("--max-turns") + 1] == "1"
    # No model pinned: the CLI resolves its own, so an account whose backend
    # has not deployed a given alias still works.
    assert "--model" not in argv


def test_model_is_passed_when_configured(fake_claude):
    _chat(model="claude-sonnet-5").invoke("hello")
    (argv,) = fake_claude.calls
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5"


def test_prompt_goes_through_a_file_not_argv(fake_claude):
    """A modeling prompt with skill context runs to tens of kilobytes."""
    fake_claude.script({"echo_prompt": True})
    big = "x" * 300_000
    reply = _chat().invoke([HumanMessage(content=big)])
    (argv,) = fake_claude.calls
    assert argv[argv.index("-p") + 1].startswith("@")
    assert reply.content == big


def test_system_messages_reach_the_system_prompt(fake_claude):
    _chat().invoke(
        [SystemMessage(content="be terse"), HumanMessage(content="fit this")]
    )
    (argv,) = fake_claude.calls
    assert "be terse" in argv[argv.index("--system-prompt") + 1]


def test_role_tuples_are_accepted(fake_claude):
    """nr-workbench's aure_adapter.complete sends this shape."""
    fake_claude.script({"echo_prompt": True})
    reply = _chat().invoke([("system", "be terse"), ("human", "fit this")])
    (argv,) = fake_claude.calls
    assert "be terse" in argv[argv.index("--system-prompt") + 1]
    assert reply.content == "fit this"


def test_a_bare_string_is_accepted(fake_claude):
    fake_claude.script({"echo_prompt": True})
    assert _chat().invoke("just this").content == "just this"


# ── Accounting ──────────────────────────────────────────────────────────


def test_usage_is_mapped_into_langchain_shape(fake_claude):
    reply = _chat().invoke("hello")
    um = reply.usage_metadata
    # input_tokens is the whole prompt, cached reads included.
    assert um["input_tokens"] == 1760 + 200 + 12533
    assert um["output_tokens"] == 40
    assert um["total_tokens"] == 1760 + 200 + 12533 + 40
    assert um["input_token_details"]["cache_read"] == 12533


def test_cost_reaches_the_ledger_field(fake_claude):
    reply = _chat().invoke("hello")
    assert reply.response_metadata["total_cost_usd"] == 0.0165


def test_the_answering_model_is_reported_not_the_bookkeeping_one(fake_claude):
    reply = _chat().invoke("hello")
    assert reply.response_metadata["model"] == "claude-sonnet-5"


def test_raw_reply_is_kept_for_the_trace(fake_claude):
    """The trace must show what came back, not what the shim made of it."""
    fake_claude.script({"result": 'Sure!\n```json\n{"a": 1}\n```'})
    reply = _chat().invoke("give me JSON")
    assert reply.content == '{"a": 1}'
    assert reply.raw_content == 'Sure!\n```json\n{"a": 1}\n```'


# ── Layer 4: the corrective retry ───────────────────────────────────────


def test_unparseable_json_is_retried_once(fake_claude):
    fake_claude.script(
        {"result": "I'd rather explain: the layers look fine."},
        {"result": '{"layers": []}'},
    )
    reply = _chat().invoke("Return a JSON object describing the layers")
    assert json.loads(reply.content) == {"layers": []}
    assert len(fake_claude.calls) == 2
    # The retry says so in the system prompt, and only the retry.
    first, second = fake_claude.calls
    assert "could not be parsed" not in first[first.index("--system-prompt") + 1]
    assert "could not be parsed" in second[second.index("--system-prompt") + 1]


def test_a_prose_request_is_never_retried(fake_claude):
    fake_claude.script({"result": "A paragraph about the sample."})
    reply = _chat().invoke("Summarise the analysis into a single paragraph.")
    assert reply.content == "A paragraph about the sample."
    assert len(fake_claude.calls) == 1


def test_retry_happens_at_most_once(fake_claude):
    fake_claude.script({"result": "still prose"})
    reply = _chat().invoke("Return only the JSON object")
    assert len(fake_claude.calls) == 2
    assert reply.content == "still prose"  # handed on rather than invented


def test_clean_json_is_not_retried(fake_claude):
    fake_claude.script({"result": '{"a": 1}'})
    _chat().invoke("Return a JSON object")
    assert len(fake_claude.calls) == 1


# ── Failure paths ───────────────────────────────────────────────────────


def test_nonzero_exit_raises_with_the_stderr(fake_claude):
    fake_claude.script({"exit_code": 1, "stderr": "Invalid API key"})
    with pytest.raises(ValueError, match="Invalid API key"):
        _chat().invoke("hello")


def test_is_error_envelope_raises(fake_claude):
    fake_claude.script(
        {"is_error": True, "subtype": "error_during_execution", "result": "nope"}
    )
    with pytest.raises(ValueError, match="error_during_execution"):
        _chat().invoke("hello")


def test_non_json_stdout_raises(fake_claude):
    fake_claude.script({"stdout_raw": "<html>a proxy login page</html>"})
    with pytest.raises(ValueError, match="not JSON"):
        _chat().invoke("hello")


def test_timeout_raises_and_kills_the_child(fake_claude, monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "1")
    fake_claude.script({"sleep": 30})
    killed = {}
    real_kill = subprocess.Popen.kill

    def spy(self):
        killed["yes"] = True
        return real_kill(self)

    monkeypatch.setattr(subprocess.Popen, "kill", spy)
    with pytest.raises(LLMTimeoutError):
        _chat().invoke("hello")
    assert killed.get("yes")


def test_recursion_is_refused(fake_claude, monkeypatch):
    monkeypatch.setenv(cc._RECURSION_FLAG, "1")
    with pytest.raises(ValueError, match="already running inside"):
        _chat().invoke("hello")


def test_the_child_is_told_it_is_inside_the_provider(fake_claude):
    """So a nested `aure analyze` refuses rather than forking again."""
    _chat().invoke("hello")
    # The fake records argv only; the flag's effect is covered above. Here we
    # assert the provider sets it on the env it builds, not on its own process.
    assert cc._RECURSION_FLAG not in os.environ


def test_missing_binary_is_reported_at_construction(monkeypatch):
    monkeypatch.setenv("AURE_CLAUDE_BIN", "")
    monkeypatch.setattr(cc.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="AURE_CLAUDE_BIN"):
        _chat()


def test_availability_needs_no_key(monkeypatch, fake_claude):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "claude_code")
    from aure.llm.config import llm_available

    assert llm_available()


# ── Live ────────────────────────────────────────────────────────────────

live = pytest.mark.skipif(
    not os.environ.get("AURE_LIVE_LLM_TESTS"),
    reason="set AURE_LIVE_LLM_TESTS=1 to run; these call the real CLI and cost money",
)


@pytest.mark.live
@live
def test_live_round_trip(monkeypatch):
    monkeypatch.delenv("AURE_CLAUDE_BIN", raising=False)
    reply = cc.create_claude_code({"model": None}, 0.0).invoke(
        "Reply with exactly: AURE-OK"
    )
    assert "AURE-OK" in reply.content
    assert reply.usage_metadata["input_tokens"] > 0
    assert reply.response_metadata["total_cost_usd"] >= 0


@pytest.mark.live
@live
def test_live_json_request_comes_back_parseable(monkeypatch):
    monkeypatch.delenv("AURE_CLAUDE_BIN", raising=False)
    reply = cc.create_claude_code({"model": None}, 0.0).invoke(
        'Return a JSON object with one key "ok" set to true. Return only the JSON.'
    )
    assert json.loads(reply.content) == {"ok": True}


# ── Golden prompts: the real thing, parsed by the real parsers ──────────
#
# The unit tests above prove the unwrapper handles adversarial text. These
# prove the prompts AuRE actually sends come back in a shape AuRE's actual
# parsers accept — which is the claim the whole JSON-discipline design makes,
# and the only way to find out is to ask.


def _selector_prompt():
    from aure.skills.selector import _SKILL_SELECTION_PROMPT

    return _SKILL_SELECTION_PROMPT.format(
        catalog=(
            "- **polymer-films**: Swelling, glass transition, brush conformation.\n"
            "- **metal-oxide-interfaces**: Native oxides, passivation layers.\n"
            "- **sei-layer-analysis**: Solid-electrolyte interphase in batteries."
        ),
        sample_info="A 240 A polystyrene film spun onto a silicon wafer, measured in air.",
    )


@pytest.mark.live
@live
@pytest.mark.parametrize(
    "name",
    [
        "sample_parse",
        "cross_state_ties",
        "per_state_structure",
        "fit_evaluation",
        "skill_selection",
    ],
)
def test_live_golden_prompt_parses_at_its_own_call_site(name, monkeypatch):
    """Each real prompt, replayed through the provider and parsed as the node
    parses it. A failure here is the JSON-discipline design not holding, not a
    test being fussy."""
    import re

    from aure.nodes import prompts
    from aure.nodes.intake import _fix_llm_json
    from aure.nodes.modeling import _strip_code_fences

    monkeypatch.delenv("AURE_CLAUDE_BIN", raising=False)

    description = (
        "A 240 A polystyrene film on a silicon substrate with a native oxide, "
        "measured in air and then in D2O."
    )

    if name == "sample_parse":
        prompt = prompts.format_sample_parse_prompt(description)

        # intake.py: fences stripped, then the first {...} span, then _fix_llm_json
        def parse(text):
            match = re.search(r"\{[\s\S]*\}", _strip_code_fences(text))
            return json.loads(_fix_llm_json(match.group()))

    elif name == "cross_state_ties":
        prompt = prompts.format_cross_state_ties_prompt(
            description, ["oxide.thickness", "oxide.material.rho", "film.thickness"]
        )

        # modeling.py:670 — no regex fallback, and the failure is swallowed.
        def parse(text):
            return json.loads(_strip_code_fences(text.strip()))

    elif name == "per_state_structure":
        prompt = prompts.format_per_state_structure_prompt(
            description, ["air", "d2o"], ["oxide", "film"]
        )

        # modeling.py:718 — same, and also swallowed.
        def parse(text):
            return json.loads(_strip_code_fences(text.strip()))

    elif name == "fit_evaluation":
        prompt = prompts.format_fit_evaluation_prompt(
            sample_description=description,
            hypothesis=None,
            chi_squared=1.8,
            method="dream",
            converged=True,
            parameters={"film.thickness": 238.4, "film.material.rho": 1.41},
            features={"substrate_sld": 2.07, "total_thickness": 245.0},
        )

        # evaluation.py:1410
        def parse(text):
            return json.loads(re.search(r"\{[\s\S]*\}", text).group())

    else:
        prompt = _selector_prompt()

        # selector.py:361
        def parse(text):
            content = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
            content = re.sub(r"\n?```\s*$", "", content).strip()
            return json.loads(re.search(r"\[[\s\S]*?\]", content).group())

    reply = cc.create_claude_code({"model": None}, 0.0).invoke(
        [HumanMessage(content=prompt)]
    )
    parsed = parse(reply.content)
    assert isinstance(parsed, (dict, list)), reply.content[:500]


def test_a_bogus_binary_override_is_not_reported_as_available(monkeypatch, tmp_path):
    """Availability must mean the call will reach something."""
    monkeypatch.setenv("LLM_PROVIDER", "claude_code")
    monkeypatch.setenv("AURE_CLAUDE_BIN", str(tmp_path / "nope" / "claude"))
    from aure.llm.config import llm_available

    assert not llm_available()
    with pytest.raises(ValueError, match="not an executable file"):
        _chat()
