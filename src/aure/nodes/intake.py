"""
INTAKE node: Load data and parse sample description.

This is the first node in the workflow. It:
1. Loads and validates the reflectivity data file
2. Inspects file headers to determine dQ convention (FWHM vs 1-sigma)
3. Records the header's free-form run title (deterministically, always)
4. Parses the sample description using an LLM
5. Populates the initial state for analysis

Header handling deliberately separates *extraction* from *interpretation*.
Extraction is a regex and cannot hallucinate, so it always runs and the value
is always checkpointed. Interpretation — letting the operator's free-form run
title influence the analysis — is gated behind ``USE_RUN_TITLE`` and, when on,
enters only as ranked *hypotheses* the refinement loop can reject, never as a
correction to the user's sample description. See :func:`_use_run_title_enabled`.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, Any

from ..state import ReflectivityState, Message, LLMCallRecord
from ..tools.data_tools import load_reflectivity_data, validate_reflectivity_data
from ..llm import llm_available, get_llm, invoke_with_timeout
from ..skills import SkillRegistry, select_skills, load_skill_context
from .hypotheses import normalize_hypothesis_states
from .prompts import format_sample_parse_prompt, format_structural_hypothesis_prompt

logger = logging.getLogger(__name__)


# ============================================================================
# dQ convention detection
# ============================================================================

_MAX_HEADER_LINES = 40  # Read at most this many lines from each file


def _read_file_header(file_path: str) -> str:
    """Read the header/comment lines from a data file.

    Returns the first ``_MAX_HEADER_LINES`` lines of the file (or fewer
    if the file is shorter).  This provides enough context for the LLM
    to determine the dQ convention without reading the entire file.
    """
    lines = []
    try:
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                if i >= _MAX_HEADER_LINES:
                    break
                lines.append(line.rstrip("\n"))
    except Exception:
        pass
    return "\n".join(lines)


def _parse_theta_from_header(file_path: str) -> float:
    """Extract incident angle (theta) from a REF_L data file header.

    Looks for the metadata table that contains ``TwoTheta(deg)`` and
    extracts the value.  For a single-segment file the table has one
    data row; theta is half of TwoTheta.

    Returns 0.0 if the angle cannot be determined (e.g. combined file
    with multiple segments, or non-REF_L format).
    """
    header = _read_file_header(file_path)
    if not header:
        return 0.0

    lines = header.split("\n")
    # Find the column header line containing TwoTheta
    col_idx = -1
    header_line_idx = -1
    for i, line in enumerate(lines):
        if "TwoTheta" in line and line.startswith("#"):
            cols = line.lstrip("# ").split()
            for j, col in enumerate(cols):
                if col.startswith("TwoTheta"):
                    col_idx = j
                    header_line_idx = i
                    break
            break

    if col_idx < 0:
        return 0.0

    # Collect data rows after the header line (comment lines with numeric data)
    data_rows = []
    for line in lines[header_line_idx + 1 :]:
        if not line.startswith("#"):
            break
        parts = line.lstrip("# ").split()
        if len(parts) > col_idx:
            try:
                float(parts[col_idx])
                data_rows.append(parts)
            except ValueError:
                continue

    if len(data_rows) != 1:
        # Combined file (multiple segments) or no data — can't assign a single theta
        return 0.0

    try:
        two_theta = float(data_rows[0][col_idx])
        return two_theta / 2.0
    except (ValueError, IndexError):
        return 0.0


# Header label spellings that carry the operator's free-form run title, e.g.
# ``# Run title: CuPt_d8-THF_FullQ-218386-1.``. Matched against comment lines
# only, case-insensitively.
_RUN_TITLE_RE = re.compile(r"^#\s*(?:run\s+)?title\s*:\s*(.+?)\s*$", re.IGNORECASE)

#: Longest run title retained. A pathological header line must not be able to
#: dominate a downstream prompt.
_MAX_RUN_TITLE_LEN = 200


def _parse_run_title_from_header(file_path: str) -> str:
    """Return the free-form run title from a data file header, or ``""``.

    Deterministic by design — no LLM. The title is text typed by the
    instrument operator at measurement time, so it may be stale, abbreviated,
    copied from a previous run, or simply wrong. This function therefore only
    records *what the file says*; whether that text is allowed to influence the
    analysis is a separate decision, gated by :func:`_use_run_title_enabled`.

    The value is kept verbatim apart from surrounding whitespace (trailing
    punctuation included) because it is provenance, not data — normalizing it
    would make the checkpoint disagree with the file. The first matching
    comment line wins. Returns ``""`` when no title line is present, which is
    the common case outside REF_L.
    """
    header = _read_file_header(file_path)
    if not header:
        return ""
    for line in header.split("\n"):
        if not line.strip():
            continue
        if not line.startswith("#"):
            break  # reached the data block; no title in the header
        m = _RUN_TITLE_RE.match(line)
        if m:
            return m.group(1)[:_MAX_RUN_TITLE_LEN]
    return ""


def _use_run_title_enabled() -> bool:
    """Whether the header's run title may influence the analysis (default off).

    Opt-in via the ``USE_RUN_TITLE`` env var. Extraction is unconditional — see
    :func:`_parse_run_title_from_header` — so the title is recorded in the
    checkpoint either way; this flag governs only whether it is *interpreted*,
    by seeding ``origin="header"`` structural hypotheses.

    Off by default for two reasons: turning it on changes results for every
    existing run (breaking comparability with the reference fits in
    ``validation/``), and a wrong title in a multi-state co-refinement can
    mislabel which state is which contrast, which corrupts the cross-state
    parameter ties rather than merely adding a bad layer.
    """
    return os.environ.get("USE_RUN_TITLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _reconcile_run_titles(datasets: list[dict]) -> tuple[str, str | None]:
    """Collapse one state's per-file run titles into a single title.

    Files sharing a state describe one physical measurement, so they should
    carry one title. Disagreement means something upstream is mislabeled, so
    the caller is handed a warning to surface instead of a silently merged
    value, and *no* title is returned: picking one of two contradictory labels
    is exactly the failure this feature is gated to avoid.

    Returns ``(title, warning)``, where ``title`` is ``""`` when the titles are
    absent or ambiguous and ``warning`` is ``None`` unless they conflict.
    """
    titles = [
        t for t in (str(ds.get("run_title") or "").strip() for ds in datasets) if t
    ]
    unique = list(dict.fromkeys(titles))
    if not unique:
        return "", None
    if len(unique) == 1:
        return unique[0], None
    return "", (
        "data files carry different run titles ("
        + "; ".join(repr(t) for t in unique)
        + ") — ignoring them; files sharing a state should describe one "
        "measurement, so this suggests a mislabeled file"
    )


def _run_titles_for_prompt(
    states: list[dict],
    data_files: list[dict],
    data_file: str,
) -> list[Dict[str, str]]:
    """Per-state run titles as ``[{"state": name, "title": text}, ...]``.

    Empty and ambiguous titles are dropped. Falls back to the flat file list,
    then to the primary ``data_file``, so a programmatic single-file call that
    predates the states-only setup shape still yields its title.
    """
    out: list[Dict[str, str]] = []
    if states:
        for st in states:
            title, _ = _reconcile_run_titles(st.get("data_files") or [])
            if title:
                out.append({"state": str(st.get("name") or ""), "title": title})
        return out
    if data_files:
        title, _ = _reconcile_run_titles(data_files)
        if title:
            out.append({"state": "", "title": title})
        return out
    title = _parse_run_title_from_header(data_file) if data_file else ""
    if title:
        out.append({"state": "", "title": title})
    return out


_HEADER_METADATA_PROMPT = """\
You are inspecting a neutron reflectometry data file header.  Extract the
metadata listed below and return it as a single JSON object.

Header:
```
{header}
```

Return a JSON object with these fields:

- "dq_is_fwhm" (bool): true if the dQ (resolution) column is FWHM,
  false if it is 1-sigma.  Most reduction software outputs FWHM; assume
  true unless the header explicitly says otherwise (e.g. "dQ (1-sigma)",
  "sigma_Q", "dQ [sigma]").

- "num_segments" (int): number of measurement segments / runs listed in
  the metadata table (count the data rows in the table that lists
  DataRun, TwoTheta, etc.).  1 for a single-segment file.  0 if no such
  table is present.

- "theta" (float): incident angle **in degrees**.  For single-segment
  files (num_segments == 1), compute TwoTheta / 2.  For combined files
  (num_segments > 1) or if the angle cannot be determined, use 0.0.

- "instrument" (string | null): instrument identifier if recognisable
  from the header (e.g. "REF_L", "REF_M", "INTER").  null if unknown.

If you are unsure about any field, use the safe defaults:
{{"dq_is_fwhm": true, "num_segments": 0, "theta": 0.0, "instrument": null}}

Respond with ONLY the JSON object.
"""


# -- Default metadata returned when parsing is not possible ---------------
_DEFAULT_HEADER_METADATA = {
    "dq_is_fwhm": True,
    "num_segments": 0,
    "theta": 0.0,
    "instrument": None,
}


def parse_file_header(file_path: str) -> dict:
    """Extract structured metadata from a reflectometry data file header.

    Makes a single LLM call that returns dQ convention, incident angle,
    number of segments, and instrument.  Falls back to deterministic
    heuristics (``_parse_theta_from_header``) when the LLM is
    unavailable, and safe defaults for anything that cannot be inferred.

    Returns
    -------
    dict
        Keys: ``dq_is_fwhm``, ``num_segments``, ``theta``, ``instrument``.
    """
    header = _read_file_header(file_path)
    if not header:
        logger.debug(
            "[INTAKE] Could not read header from %s; using defaults", file_path
        )
        return dict(_DEFAULT_HEADER_METADATA)

    if not llm_available():
        logger.debug("[INTAKE] LLM not available; falling back to heuristics")
        result = dict(_DEFAULT_HEADER_METADATA)
        result["theta"] = _parse_theta_from_header(file_path)
        return result

    try:
        from langchain_core.messages import HumanMessage

        prompt = _HEADER_METADATA_PROMPT.format(header=header)
        llm = get_llm(temperature=0)
        response = invoke_with_timeout(llm, [HumanMessage(content=prompt)])
        content = response.content.strip()

        # Strip markdown fences
        content = re.sub(r"^```(?:json)?\s*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)
        content = content.strip()

        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            parsed = json.loads(_fix_llm_json(match.group()))
            # Merge with defaults so missing keys never cause KeyErrors
            result = dict(_DEFAULT_HEADER_METADATA)
            for key in result:
                if key in parsed:
                    result[key] = parsed[key]
            logger.info(
                "[INTAKE] Header metadata for %s: dq_is_fwhm=%s, theta=%.4f, "
                "segments=%d, instrument=%s",
                file_path,
                result["dq_is_fwhm"],
                result["theta"],
                result["num_segments"],
                result["instrument"],
            )
            return result
    except Exception as e:
        logger.warning(
            "[INTAKE] Header metadata extraction failed for %s: %s; "
            "falling back to heuristics",
            file_path,
            e,
        )

    # Fallback: heuristic theta + safe defaults for the rest
    result = dict(_DEFAULT_HEADER_METADATA)
    result["theta"] = _parse_theta_from_header(file_path)
    return result


def detect_dq_convention(file_path: str) -> bool:
    """Determine whether the dQ column in a data file is FWHM.

    Convenience wrapper that delegates to :func:`parse_file_header` and
    returns only the ``dq_is_fwhm`` flag.
    """
    return parse_file_header(file_path)["dq_is_fwhm"]


# ============================================================================
# Sample parsing
# ============================================================================


def parse_sample_with_llm(
    description: str,
    hypothesis: str | None = None,
    skill_context: str = "",
) -> Dict[str, Any]:
    """
    Parse sample description into structured format using the configured LLM.

    Args:
        description: Free-form sample description from the user
        hypothesis: Optional hypothesis to test

    Returns:
        Parsed sample dictionary with substrate, layers, ambient, etc.

    Raises:
        ValueError: If LLM is not available or parsing fails
        LLMTimeoutError: If the LLM call times out (likely quota issue)
    """
    if not llm_available():
        raise ValueError(
            "LLM is required for sample parsing. Please configure LLM_PROVIDER "
            "and appropriate API keys. See .env.example for options."
        )

    llm = get_llm(temperature=0)
    prompt = format_sample_parse_prompt(
        description, hypothesis, skill_context=skill_context
    )

    from langchain_core.messages import HumanMessage

    response = invoke_with_timeout(llm, [HumanMessage(content=prompt)])

    # Extract JSON from response
    content = response.content

    # Strip markdown code fences the LLM may wrap the JSON in
    content = re.sub(r"^```(?:json)?\s*\n", "", content)
    content = re.sub(r"\n```\s*$", "", content)
    content = content.strip()

    # Try to find JSON block in the response
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        raw_json = json_match.group()
        return json.loads(_fix_llm_json(raw_json))

    raise ValueError("Could not extract JSON from LLM response")


def _fix_llm_json(text: str) -> str:
    """Best-effort fix-up of common LLM JSON mistakes.

    Handles trailing commas before ``}`` or ``]`` and single-line
    ``// ...`` comments.
    """
    # Remove single-line comments (// ...)
    text = re.sub(r"//[^\n]*", "", text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


# ============================================================================
# Structural hypothesis generation
# ============================================================================

# Keys we expect on each hypothesis object; used to filter/validate
_HYPOTHESIS_TEXT_FIELDS = ("title", "rationale", "change", "skill_source")


def generate_structural_hypotheses_with_llm(
    sample_description: str,
    parsed_sample: Dict[str, Any],
    skill_context: str,
    hypothesis: str | None = None,
    run_titles: list[Dict[str, str]] | None = None,
    state_names: list[str] | None = None,
) -> list[Dict[str, Any]]:
    """Ask the LLM for a ranked list of candidate structural changes.

    Uses the ``structural-hypothesis-ranking`` skill plus all other active
    skill bodies to enumerate plausible modifications to the baseline model
    (e.g. adding a native oxide, splitting a layer, etc.). The user's stated
    ``hypothesis`` is folded in as one or more high-priority entries
    (``origin="user"``) ranked above the skill-derived ones. Each entry is
    stamped with a sequential ``id`` and ``status: "pending"``.

    ``run_titles`` (only ever non-empty when ``USE_RUN_TITLE`` is enabled)
    supplies the data files' free-form header titles as a *weak* extra source,
    yielding ``origin="header"`` entries ranked between the user's and the
    skills'. They enter as hypotheses rather than as baseline structure so a
    wrong title costs one refinement iteration and is then rejected by
    evaluation's regression guardrail, instead of silently poisoning every
    iteration — the same treatment the user's own hypothesis gets.

    ``state_names`` lists the run's measurement states, letting a hypothesis be
    *scoped* to a subset of them (``states``) — the "sample != structure" case
    where a layer is present in some states and absent in others. Without it
    the prompt sees one flattened stack and such a change cannot be proposed at
    all. Scopes are validated against these names; unknown ones are dropped.

    Returns an empty list if the LLM is unavailable or returns nothing
    parseable. Never raises.
    """
    if not llm_available():
        return []

    from langchain_core.messages import HumanMessage

    prompt = format_structural_hypothesis_prompt(
        sample_description=sample_description,
        parsed_sample=parsed_sample,
        skill_context=skill_context,
        hypothesis=hypothesis,
        run_titles=run_titles,
        state_names=state_names,
    )
    llm = get_llm(temperature=0)
    try:
        response = invoke_with_timeout(llm, [HumanMessage(content=prompt)])
    except Exception as e:
        logger.warning("[INTAKE] Structural hypothesis LLM call failed: %s", e)
        return []

    content = response.content.strip()
    # Strip markdown fences
    content = re.sub(r"^```(?:json)?\s*\n?", "", content)
    content = re.sub(r"\n?```\s*$", "", content)
    content = content.strip()

    match = re.search(r"\[[\s\S]*\]", content)
    if not match:
        logger.debug("[INTAKE] No JSON array in hypothesis response")
        return []

    try:
        raw = json.loads(_fix_llm_json(match.group()))
    except json.JSONDecodeError as e:
        logger.warning("[INTAKE] Hypothesis JSON parse error: %s", e)
        return []

    if not isinstance(raw, list):
        return []

    # Build entries (without ids yet), tagging provenance from skill_source.
    parsed_items: list[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        skill_source = str(item.get("skill_source", "")).strip()
        lowered = skill_source.lower()
        if lowered == "user":
            origin = "user"
        elif lowered == "header":
            origin = "header"
        else:
            origin = "skill"
        h = {
            "title": str(item.get("title", "")).strip(),
            "rationale": str(item.get("rationale", "")).strip(),
            "change": str(item.get("change", "")).strip(),
            "skill_source": skill_source,
            "origin": origin,
            "states": normalize_hypothesis_states(item.get("states"), state_names),
            "status": "pending",
            "tried_in_iteration": None,
            "created_in_iteration": None,
            "notes": "",
        }
        if h["title"]:
            parsed_items.append(h)

    if not run_titles:
        # No title was supplied (feature off, or no header carried one), so a
        # "header"-sourced entry has no evidence behind it and must not enter
        # the list — the flag has to actually gate the behaviour even when the
        # model volunteers the label on its own.
        dropped = [h for h in parsed_items if h["origin"] == "header"]
        if dropped:
            logger.debug(
                "[INTAKE] Dropped %d header-sourced hypotheses (no run title)",
                len(dropped),
            )
        parsed_items = [h for h in parsed_items if h["origin"] != "header"]

    # Rank user > header > skill: a hypothesis the user typed for this run
    # outranks the operator's free-form title, which in turn describes *this*
    # measurement and so outranks the skills' generic domain priors. Preserve
    # the LLM's relative order within each group, then assign stable ids in
    # rank order.
    user_items = [h for h in parsed_items if h["origin"] == "user"]
    header_items = [h for h in parsed_items if h["origin"] == "header"]
    skill_items = [h for h in parsed_items if h["origin"] not in ("user", "header")]
    hypotheses = [
        {"id": i, **h}
        for i, h in enumerate(user_items + header_items + skill_items, start=1)
    ]

    logger.info(
        "[INTAKE] Generated %d structural hypotheses (%d from user, %d from header)",
        len(hypotheses),
        len(user_items),
        len(header_items),
    )
    return hypotheses


def _format_hypotheses_summary(hypotheses: list[Dict[str, Any]]) -> str:
    """Format the hypothesis list for display in the checkpoint transcript."""
    lines = ["**Ranked structural hypotheses (consulted if parameter tuning stalls):**"]
    for h in hypotheses:
        lines.append(f"{h['id']}. **{h['title']}** — {h['change']}")
        if h.get("rationale"):
            lines.append(
                f"   _Rationale ({h.get('skill_source', '?')}):_ {h['rationale']}"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Multi-state helpers (Ticket 05)
# ----------------------------------------------------------------------

# Filename heuristics for REF_L instrument set_id detection.
_SET_ID_COMBINED_RE = re.compile(r"REFL_(\d+)_combined_data_auto\.txt$", re.IGNORECASE)
_SET_ID_PARTIAL_RE = re.compile(r"REFL_(\d+)_\d+_\d+_partial\.txt$", re.IGNORECASE)


def _extract_set_id(file_path: str) -> str | None:
    """Return the REF_L set_id encoded in *file_path* or None if absent."""
    import os

    name = os.path.basename(file_path)
    for pattern in (_SET_ID_COMBINED_RE, _SET_ID_PARTIAL_RE):
        m = pattern.search(name)
        if m:
            return m.group(1)
    return None


def _enrich_dataset(ds: dict) -> dict:
    """Load Q/R/dR + parse header metadata for a single ``DatasetInfo``.

    Returns a new dict; raises on load failure so the caller can decide
    whether to abort or downgrade to a warning.
    """
    enriched = dict(ds)
    data = load_reflectivity_data(ds["file"])
    validate_reflectivity_data(data["Q"], data["R"], data.get("dR"))
    enriched["Q"] = data["Q"].tolist()
    enriched["R"] = data["R"].tolist()
    if data.get("dR") is not None:
        enriched["dR"] = data["dR"].tolist()
    else:
        enriched["dR"] = [0.0] * len(data["Q"])

    meta = parse_file_header(ds["file"])
    enriched.setdefault("theta", meta["theta"])
    enriched.setdefault("dq_is_fwhm", meta["dq_is_fwhm"])
    enriched.setdefault("num_segments", meta.get("num_segments", 0))
    # Deterministic, unconditional, and independent of the LLM header call --
    # recorded even when USE_RUN_TITLE is off (see _parse_run_title_from_header).
    enriched.setdefault("run_title", _parse_run_title_from_header(ds["file"]))
    return enriched


def intake_node(state: ReflectivityState) -> Dict[str, Any]:
    """
    Load data and parse sample description.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    updates = {
        "current_node": "intake",
        "messages": [],
        "llm_calls": [],
    }

    # ========== 1. Load Data ==========
    try:
        data = load_reflectivity_data(state["data_file"])
        updates["Q"] = data["Q"].tolist()
        updates["R"] = data["R"].tolist()
        updates["dR"] = data.get("dR", [0.0] * len(data["Q"])).tolist()

        # Validate
        validation = validate_reflectivity_data(data["Q"], data["R"], data.get("dR"))

        if validation["issues"]:
            updates["messages"] = [
                Message(
                    role="system",
                    content=f"Data validation warnings: {', '.join(validation['issues'])}",
                    timestamp=None,
                )
            ]

        # Build data_files list for multi-file co-refinement

        existing_data_files = state.get("data_files", [])
        states_in = state.get("states", []) or []
        is_multi = len(states_in) > 1

        if states_in:
            # ===== States path (single or multi) =====
            # Every state's data_files is enriched in place with header
            # metadata (theta, dq convention) + Q/R/dR for plotting, and the
            # flat ``data_files`` list is rebuilt for downstream nodes that
            # still iterate it. The *single*-state case is enriched too: that
            # is what lets a single state's theta_offset / sample_broadening
            # load an angle-based NeutronProbe — they silently no-op on the
            # Q-only QProbe produced when a dataset carries no theta.
            enriched_states: list[dict] = []
            flat_files: list[dict] = []
            kind_errors: list[str] = []
            for st in states_in:
                st_files = st.get("data_files") or []
                enriched_files: list[dict] = []
                set_ids: set[str] = set()
                for ds in st_files:
                    try:
                        enriched_ds = _enrich_dataset(ds)
                    except Exception as exc:
                        updates["messages"].append(
                            Message(
                                role="system",
                                content=(
                                    f"Warning: could not load "
                                    f"{ds.get('label', ds.get('file'))}: {exc}"
                                ),
                                timestamp=None,
                            )
                        )
                        continue
                    sid = _extract_set_id(ds["file"])
                    if sid:
                        set_ids.add(sid)
                    enriched_files.append(enriched_ds)
                    flat_files.append(enriched_ds)
                if is_multi and len(set_ids) > 1:
                    # Single-state set_id consistency is already validated in
                    # config._detect_kind; only re-check across a multi-state
                    # run here (and don't reject single-state combined splices).
                    kind_errors.append(
                        f"State {st.get('name')!r}: partial files mix set_ids "
                        f"{sorted(set_ids)} — each state must contain files "
                        f"from a single REF_L set."
                    )
                _, title_warning = _reconcile_run_titles(enriched_files)
                if title_warning:
                    updates["messages"].append(
                        Message(
                            role="system",
                            content=f"State {st.get('name')!r}: {title_warning}.",
                            timestamp=None,
                        )
                    )
                new_state = dict(st)
                new_state["data_files"] = enriched_files
                enriched_states.append(new_state)

            if kind_errors:
                updates["error"] = "; ".join(kind_errors)
                updates["messages"].append(
                    Message(
                        role="system",
                        content=updates["error"],
                        timestamp=None,
                    )
                )
                return updates

            updates["states"] = enriched_states
            updates["data_files"] = flat_files
            n_files = len(flat_files)
            if is_multi:
                labels = ", ".join(s["name"] for s in enriched_states)
                updates["messages"].append(
                    Message(
                        role="system",
                        content=(
                            f"Multi-state co-refinement: {len(enriched_states)} "
                            f"states ({labels}), {n_files} files total"
                        ),
                        timestamp=None,
                    )
                )
            elif n_files > 1:
                updates["messages"].append(
                    Message(
                        role="system",
                        content=(
                            f"Single-state co-refinement: {n_files} files in "
                            f"state {enriched_states[0]['name']!r}"
                        ),
                        timestamp=None,
                    )
                )
        elif existing_data_files:
            # ===== Legacy flat multi-file path =====
            # The CLI/web layers now always wrap multi-file invocations
            # in a single ``state0`` and pass them via ``states``, so this
            # branch only fires when the runner is invoked programmatically
            # without states. Multi-file mode: validate all additional files.
            validated_files = []
            for ds in existing_data_files:
                try:
                    enriched = _enrich_dataset(ds)
                    validated_files.append(enriched)
                except Exception as e:
                    updates["messages"].append(
                        Message(
                            role="system",
                            content=f"Warning: could not load {ds['label']}: {e}",
                            timestamp=None,
                        )
                    )
            updates["data_files"] = validated_files
            if len(validated_files) > 1:
                labels = ", ".join(ds["label"] for ds in validated_files)
                updates["messages"].append(
                    Message(
                        role="system",
                        content=f"Multi-file co-refinement: {len(validated_files)} files loaded ({labels})",
                        timestamp=None,
                    )
                )

    except Exception as e:
        updates["error"] = f"Failed to load data: {str(e)}"
        updates["messages"] = [
            Message(
                role="system",
                content=f"Error loading data file: {str(e)}",
                timestamp=None,
            )
        ]
        return updates

    # ========== 2. Detect dQ Convention ==========
    # Inspect the primary data file header to determine if dQ is FWHM.
    # The result is stored in the state so model_builder can pass it to load4.
    primary_meta = parse_file_header(state["data_file"])
    updates["dq_is_fwhm"] = primary_meta["dq_is_fwhm"]

    # ========== 2b. Record the header's run title ==========
    # Extraction is unconditional; only interpretation is gated. Recording the
    # title even when it goes unused is what lets a corpus of past runs answer
    # "would the title have helped?" before USE_RUN_TITLE is ever made the
    # default — see docs/approach.md.
    observed_titles = _run_titles_for_prompt(
        updates.get("states") or [],
        updates.get("data_files") or [],
        state.get("data_file", ""),
    )
    use_run_title = _use_run_title_enabled()
    if observed_titles:
        rendered = ", ".join(
            f"{t['state']}: {t['title']!r}" if t["state"] else repr(t["title"])
            for t in observed_titles
        )
        updates["messages"].append(
            Message(
                role="system",
                content=(
                    f"Run title from file header ({rendered}) — "
                    + (
                        "seeding structural hypotheses (USE_RUN_TITLE is on)."
                        if use_run_title
                        else "recorded only, not used (USE_RUN_TITLE is off)."
                    )
                ),
                timestamp=None,
            )
        )

    # ========== 3. Parse Sample Description ==========
    if state["sample_description"]:
        # Select and load skills based on sample description
        registry = SkillRegistry()
        active_skills = select_skills(
            state["sample_description"],
            parsed_sample=None,
            registry=registry,
            states=state.get("states"),
        )
        skill_context = load_skill_context(active_skills, registry)

        try:
            parsed = parse_sample_with_llm(
                state["sample_description"],
                hypothesis=state.get("hypothesis"),
                skill_context=skill_context,
            )
            updates["parsed_sample"] = parsed

            # Re-select skills now that we have parsed sample info
            active_skills = select_skills(
                state["sample_description"],
                parsed_sample=parsed,
                registry=registry,
                states=state.get("states"),
            )
            # Activate the multi-state co-refinement skill when applicable.
            if len(state.get("states") or []) > 1 and (
                "multi-state-corefinement" not in active_skills
            ):
                active_skills = sorted(
                    set(active_skills) | {"multi-state-corefinement"}
                )
            updates["active_skills"] = active_skills
            updates["llm_calls"].append(
                LLMCallRecord(
                    node="intake",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    success=True,
                    used_fallback=False,
                    fallback_reason=None,
                    error=None,
                )
            )

            # Add confirmation message
            updates["messages"].append(
                Message(
                    role="assistant",
                    content=_format_parsed_summary(parsed),
                    timestamp=None,
                )
            )

            # ========== 4. Generate Ranked Structural Hypotheses ==========
            # Ask the LLM to enumerate candidate structural changes, drawing
            # on the active skills. Consumed by evaluation/modeling when
            # parameter-only refinement stalls.
            try:
                skill_context_full = load_skill_context(active_skills, registry)
                hypotheses = generate_structural_hypotheses_with_llm(
                    sample_description=state["sample_description"],
                    parsed_sample=parsed,
                    skill_context=skill_context_full,
                    hypothesis=state.get("hypothesis"),
                    run_titles=observed_titles if use_run_title else None,
                    state_names=[
                        st.get("name")
                        for st in (state.get("states") or [])
                        if st.get("name")
                    ],
                )
                updates["structural_hypotheses"] = hypotheses
                updates["llm_calls"].append(
                    LLMCallRecord(
                        node="intake",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        success=True,
                        used_fallback=False,
                        fallback_reason=None,
                        error=None,
                    )
                )
                if hypotheses:
                    updates["messages"].append(
                        Message(
                            role="assistant",
                            content=_format_hypotheses_summary(hypotheses),
                            timestamp=None,
                        )
                    )
            except Exception as e:
                logger.warning(
                    "[INTAKE] Could not generate structural hypotheses: %s", e
                )
                updates["structural_hypotheses"] = []
                updates["llm_calls"].append(
                    LLMCallRecord(
                        node="intake",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        success=False,
                        used_fallback=True,
                        fallback_reason="Structural hypothesis generation failed; proceeding without list",
                        error=str(e)[:200],
                    )
                )

        except Exception as e:
            # Non-fatal - we can still proceed with feature extraction
            updates["llm_calls"].append(
                LLMCallRecord(
                    node="intake",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    success=False,
                    used_fallback=True,
                    fallback_reason="Proceeding with feature extraction only",
                    error=str(e)[:200],
                )
            )
            updates["messages"].append(
                Message(
                    role="system",
                    content=f"Could not parse sample description: {str(e)}. Will rely on feature extraction.",
                    timestamp=None,
                )
            )

    return updates


def _format_parsed_summary(parsed: dict) -> str:
    """Format parsed sample info for display."""
    lines = ["**Understood sample structure:**"]

    if parsed.get("substrate"):
        sub = parsed["substrate"]
        lines.append(f"- Substrate: {sub['name']} (SLD = {sub['sld']:.2f})")

    if parsed.get("layers"):
        for i, layer in enumerate(parsed["layers"], 1):
            lines.append(
                f"- Layer {i}: {layer['name']} "
                f"(~{layer['thickness']:.0f} Å, SLD ≈ {layer['sld']:.2f})"
            )

    if parsed.get("ambient"):
        amb = parsed["ambient"]
        lines.append(f"- Ambient: {amb['name']} (SLD = {amb['sld']:.2f})")

    if parsed.get("constraints"):
        lines.append(f"- Constraints: {', '.join(parsed['constraints'])}")

    return "\n".join(lines)
