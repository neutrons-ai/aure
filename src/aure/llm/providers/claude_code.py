"""Claude Code as an LLM provider.

Every other provider here is a completions API reached with a key. This one is
a subprocess: ``claude -p``, the same binary a person drives interactively.

**Why it exists.** nr-workbench users have Claude Code and frequently nothing
else. ``nrw aure run`` refuses to start without an endpoint, so the only way
through was to configure a second, weaker model purely to satisfy that gate —
which is precisely the trade nr-workbench's own documentation argues against.

**It carries no credentials.** Whatever ``claude`` is already authenticated
with — a subscription, ``ANTHROPIC_API_KEY``, Bedrock, Vertex, Foundry — is
what this provider uses. AuRE's rule that obtaining and refreshing a facility
token is the facility tooling's job survives intact: this adds no credential
code, it delegates to a binary that already owns the problem.

**What it costs.** Every call carries Claude Code's own preamble — system
prompt plus tool definitions a completions API would never charge for.
Measured against Sonnet 5 with the flags below: ~12.3k input tokens per call,
about $0.077 cold and $0.017 warm. The prompt cache has a 1 h TTL and is reused
*across processes*, so only the first call in a run pays the cold price and no
session juggling is needed to get the warm one. Budget roughly $0.2–0.4 per
analysis of pure overhead on top of the prompts themselves.

**Why not ``--bare``.** It would cut the preamble further, but it also
restricts authentication to ``ANTHROPIC_API_KEY`` and never reads OAuth or the
keychain — removing the one property this provider exists for.

**Temperature is not a knob here.** ``claude`` exposes none. Every AuRE call
site asks for 0, which is what an agentic harness approximates anyway, so the
argument is accepted and ignored rather than being quietly honoured.

**``LLM_MAX_RETRIES`` does not apply.** It exists for a local server evicting
an idle model mid-run; a CLI failure here is a configuration or auth problem
that a retry does not fix. The one retry this provider does make is the JSON
correction below.

JSON discipline
---------------

Claude Code is trained to be conversational, and most AuRE nodes parse JSON out
of the reply — two of them (``modeling._cross_state_ties``,
``modeling._per_state_structure``) with no tolerance for a preamble at all, and
they degrade *silently* when parsing fails. Four layers keep that from
happening, each of which can be tested on its own:

1. **Replace the system prompt.** ``--system-prompt`` (replace, not append)
   with a completions-style instruction. This is the layer that does most of
   the work: the conversational habit is largely Claude Code's own prompt.
2. **Strip code fences**, unconditionally. Safe for prose, which is never
   fenced.
3. **Extract a balanced JSON span**, but only when the prompt asked for JSON,
   the reply does not already parse, and a balanced span exists. All three
   conditions matter: the ISAAC exporter asks for a prose paragraph and uses
   the reply verbatim, so an unconditional extractor would destroy it.
4. **Retry once**, with a blunter instruction, if the prompt asked for JSON and
   nothing parseable came back.

Layers 3 and 4 log when they fire. If layer 4 shows up routinely in a run's
logs, layer 1's wording is wrong and that is the fix — the retry is a
backstop, not the mechanism.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..config import get_llm_timeout

logger = logging.getLogger(__name__)

#: Set in the child's environment so a nested AuRE run — an agent session that
#: shells out to `aure analyze`, which would otherwise shell back to `claude` —
#: fails with an explanation instead of forking until something gives out.
_RECURSION_FLAG = "AURE_CLAUDE_CODE_PROVIDER"

#: Denied explicitly rather than trusted to the prompt. The completions-style
#: system prompt says not to use tools; this makes it so even if a future
#: default makes one of them attractive.
_TOOL_DENY = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "Task",
    "TodoWrite",
    "NotebookEdit",
)

#: Replaces Claude Code's own system prompt. Layer 1 of the JSON discipline,
#: and the reason the preamble drops from ~29k tokens to ~12k.
_SYSTEM_PROMPT = (
    "You are a text-completion endpoint. You are not an assistant and not an "
    "agent.\n"
    "Answer the request directly and completely.\n"
    "Do not use any tool. Do not read or write any file.\n"
    "Do not add a preamble, a restatement of the question, a summary, an "
    "explanation of your reasoning, or an offer of further help.\n"
    "If the request asks for JSON, reply with that JSON value alone: no prose "
    "before or after it, and no markdown code fence."
)

#: Appended for the layer-4 retry only.
_JSON_REMINDER = (
    "\nYour previous reply could not be parsed. Output the JSON value and "
    "nothing else. No prose. No code fence. Start your reply with { or [."
)


def _binary() -> str:
    """Path to the ``claude`` executable.

    Raises:
        ValueError: If it cannot be found, naming the fix.
    """
    explicit = os.environ.get("AURE_CLAUDE_BIN")
    if explicit:
        # Resolved, not trusted: an override pointing at nothing would
        # otherwise report the provider as available and fail at the first
        # call, which is a worse diagnostic than failing here.
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        resolved = shutil.which(explicit)
        if resolved:
            return resolved
        raise ValueError(
            f"AURE_CLAUDE_BIN points at {explicit!r}, which is not an executable file."
        )
    found = shutil.which("claude")
    if not found:
        raise ValueError(
            "The claude_code provider needs the Claude Code CLI on PATH. "
            "Install it, or point AURE_CLAUDE_BIN at the binary. "
            "See https://claude.com/claude-code"
        )
    return found


def available() -> bool:
    """Whether the CLI can be found — no key, no network call."""
    try:
        return bool(_binary())
    except ValueError:
        return False


# ── JSON discipline ─────────────────────────────────────────────────────


def _wants_json(prompt_text: str) -> bool:
    """Whether the prompt asked for JSON.

    A heuristic, deliberately narrow. It gates layers 3 and 4, and the cost of
    a false positive is real: the ISAAC exporter's prompt asks for a prose
    paragraph, and mangling that into a JSON fragment would be worse than any
    preamble. That prompt mentions JSON nowhere, and every AuRE prompt that
    does want JSON says so in as many words.
    """
    return "json" in prompt_text.lower()


def _strip_fences(text: str) -> str:
    """Remove a wrapping markdown code fence, if there is one."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence line (```json, ```, …) and a closing one if present.
    lines = lines[1:]
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


def _balanced_span(text: str, start: int) -> Optional[str]:
    """The balanced ``{…}``/``[…]`` beginning at *start*, or ``None``.

    String-literal aware, which the regexes at the call sites are not: a
    ``"}"`` inside a value does not close the object, and a ``\\"`` inside a
    string does not end it.
    """
    opener = text[start]
    closer = {"{": "}", "[": "]"}[opener]
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> Optional[str]:
    """The first balanced JSON span in *text* that actually parses.

    Returns ``None`` rather than a guess, so the caller can leave the reply
    alone instead of handing a downstream parser something worse than what it
    started with.
    """
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        span = _balanced_span(text, i)
        if span is None:
            continue
        try:
            json.loads(span)
        except ValueError:
            continue
        return span
    return None


def _parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _unwrap(raw: str, wants_json: bool) -> tuple:
    """Apply layers 2 and 3. Returns ``(content, note)``.

    *note* is ``None`` when nothing but fence-stripping happened, and a short
    label otherwise, so the caller can log that a reply needed rescuing.
    """
    content = _strip_fences(raw)
    if not wants_json or _parses(content):
        return content, None
    extracted = _extract_json(content)
    if extracted is None:
        return content, "unparseable"
    return extracted, "extracted"


# ── The provider ────────────────────────────────────────────────────────


class _Reply:
    """A provider response in the shape the rest of AuRE expects.

    ``content`` is what the nodes parse; ``raw_content`` is what the CLI
    actually returned, which is what the call trace records — the point of the
    trace is to show the reply, not this module's opinion of it.
    """

    def __init__(self, content, raw_content, usage_metadata, response_metadata):
        self.content = content
        self.raw_content = raw_content
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


class _ClaudeCodeChat:
    """Minimal chat model over the Claude Code CLI.

    Not a ``BaseChatModel`` subclass: nothing in AuRE binds tools, requests
    structured output, or streams, so the surface that matters is ``.invoke``
    returning something with ``.content``. Subclassing would buy pydantic
    validation of arguments this never takes.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or None
        self.model_name = model or "claude-code-default"

    # -- message handling ------------------------------------------------

    @staticmethod
    def _split(payload: Any) -> tuple:
        """Split whatever was passed into ``(system_text, user_text)``.

        Handles the three shapes that reach a provider in this codebase: a
        bare string, LangChain message objects, and ``(role, content)``
        tuples.
        """
        if isinstance(payload, str):
            return "", payload
        if not isinstance(payload, (list, tuple)):
            payload = [payload]

        systems, users = [], []
        for item in payload:
            role, text = _ClaudeCodeChat._one(item)
            (systems if role == "system" else users).append(text)
        return "\n\n".join(s for s in systems if s), "\n\n".join(u for u in users if u)

    @staticmethod
    def _one(item: Any) -> tuple:
        content = getattr(item, "content", None)
        if content is not None:
            return str(getattr(item, "type", "human")), _as_text(content)
        if isinstance(item, dict):
            role = str(item.get("role") or item.get("type") or "human")
            return role, _as_text(item.get("content"))
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return str(item[0]), _as_text(item[1])
        return "human", _as_text(item)

    # -- invocation ------------------------------------------------------

    def invoke(self, payload: Any) -> _Reply:
        system_text, user_text = self._split(payload)
        wants_json = _wants_json(user_text)

        envelope = self._call(user_text, system_text, retry=False)
        raw = envelope.get("result") or ""
        content, note = _unwrap(raw, wants_json)

        if note == "extracted":
            logger.info(
                "[claude_code] Reply carried prose around the JSON; extracted it."
            )
        elif note == "unparseable" and wants_json:
            logger.warning(
                "[claude_code] Reply was not parseable JSON; retrying once with "
                "a stricter instruction."
            )
            envelope = self._call(user_text, system_text, retry=True)
            raw = envelope.get("result") or ""
            content, note = _unwrap(raw, wants_json)
            if note == "unparseable":
                logger.warning(
                    "[claude_code] Retry did not produce parseable JSON either; "
                    "handing the reply to the caller as-is."
                )

        return _Reply(
            content=content,
            raw_content=raw,
            usage_metadata=_usage_metadata(envelope),
            response_metadata=_response_metadata(envelope),
        )

    def _argv(self, prompt_file: Path, system_prompt: str) -> list:
        argv = [
            _binary(),
            "-p",
            f"@{prompt_file}",
            "--output-format",
            "json",
            "--system-prompt",
            system_prompt,
            # Tool definitions still cost tokens even denied, but a denied tool
            # cannot be called, and this is what takes the preamble from ~29k
            # to ~12.3k with --disable-slash-commands.
            "--disallowed-tools",
            *_TOOL_DENY,
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            # Nothing to iterate on with no tools; a belt for a future default.
            "--max-turns",
            "1",
        ]
        if self.model:
            argv += ["--model", self.model]
        return argv

    def _call(self, user_text: str, system_text: str, *, retry: bool) -> dict:
        if os.environ.get(_RECURSION_FLAG):
            raise ValueError(
                "The claude_code provider is already running inside a "
                "claude_code call. Configure a different LLM_PROVIDER for the "
                "inner run, or drive AuRE directly rather than from a session "
                "this provider started."
            )

        system_prompt = _SYSTEM_PROMPT
        if system_text:
            system_prompt = f"{system_prompt}\n\n{system_text}"
        if retry:
            system_prompt = f"{system_prompt}\n{_JSON_REMINDER}"

        env = dict(os.environ)
        env[_RECURSION_FLAG] = "1"

        timeout = float(get_llm_timeout())
        with tempfile.TemporaryDirectory(prefix="aure-claude-") as tmp:
            # Via a file, not argv: a modeling prompt with skill context runs to
            # tens of kilobytes and ARG_MAX is a real ceiling.
            prompt_file = Path(tmp) / "prompt.txt"
            prompt_file.write_text(user_text, encoding="utf-8")
            argv = self._argv(prompt_file, system_prompt)
            code, out, err = _run(argv, env, timeout)

        if code != 0:
            raise ValueError(
                f"claude exited {code}: {(err or out or '').strip()[:500]}"
            )
        try:
            envelope = json.loads(out)
        except ValueError as exc:
            raise ValueError(
                f"claude returned output that is not JSON: {out[:500]}"
            ) from exc
        if envelope.get("is_error"):
            raise ValueError(
                f"claude reported an error ({envelope.get('subtype')}): "
                f"{str(envelope.get('result'))[:500]}"
            )
        return envelope


def _run(argv: list, env: dict, timeout: float) -> tuple:
    """Run *argv*, killing the child if anything interrupts the wait.

    The kill matters twice over. ``subprocess.TimeoutExpired`` leaves the child
    running, and so does the ``LLMTimeoutError`` that :mod:`aure.llm.timeout`
    raises from a SIGALRM handler *inside* this call — in both cases the
    request would otherwise keep going, and keep billing, after the caller has
    given up on it.
    """
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        from ..timeout import LLMTimeoutError

        raise LLMTimeoutError(
            f"claude did not answer within {timeout:.0f}s (LLM_TIMEOUT)"
        ) from None
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    return proc.returncode, out, err


def _as_text(content: Any) -> str:
    """Flatten message content, including provider block lists, to text."""
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


def _usage_metadata(envelope: dict) -> dict:
    """The CLI's usage block in LangChain's shape.

    ``input_tokens`` is the *total* prompt count including cached reads, which
    is LangChain's convention and what the ledger sums; the cache read is
    broken out under ``input_token_details`` exactly as the ledger expects.
    """
    usage = envelope.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    plain = usage.get("input_tokens") or 0
    created = usage.get("cache_creation_input_tokens") or 0
    read = usage.get("cache_read_input_tokens") or 0
    out = usage.get("output_tokens") or 0
    total_in = plain + created + read
    return {
        "input_tokens": total_in,
        "output_tokens": out,
        "total_tokens": total_in + out,
        "input_token_details": {"cache_read": read, "cache_creation": created},
    }


def _response_metadata(envelope: dict) -> dict:
    """Cost and session identity, for the ledger and for provenance."""
    meta = {}
    cost = envelope.get("total_cost_usd")
    if cost is not None:
        meta["total_cost_usd"] = cost
    for key in ("session_id", "duration_ms", "num_turns", "stop_reason"):
        if envelope.get(key) is not None:
            meta[key] = envelope[key]
    model = _dominant_model(envelope)
    if model:
        meta["model"] = model
    return meta


def _dominant_model(envelope: dict) -> Optional[str]:
    """Which model actually answered.

    ``modelUsage`` lists every model a turn touched, including the small one
    Claude Code uses for its own bookkeeping. The one that produced the reply
    is the one that produced the output tokens.
    """
    usage = envelope.get("modelUsage")
    if not isinstance(usage, dict) or not usage:
        return None
    try:
        return max(usage.items(), key=lambda kv: (kv[1] or {}).get("outputTokens", 0))[
            0
        ]
    except Exception:
        return None


def create_claude_code(config: dict, temperature: float):
    """Create a Claude Code chat model.

    *temperature* is accepted for interface parity and ignored: the CLI exposes
    no such control. Every AuRE call site asks for 0.
    """
    if temperature:
        logger.debug(
            "[claude_code] temperature=%s ignored; the CLI exposes no such control",
            temperature,
        )
    _binary()  # fail now, with an actionable message, not at the first call
    return _ClaudeCodeChat(model=config.get("model") or None)
