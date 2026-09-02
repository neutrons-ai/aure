"""Tests for run-title handling: deterministic extraction, gated interpretation.

Two separable behaviours are covered here, and the split is the point:

* **Extraction** (always on) — a regex reads ``# Run title:`` from the header
  and records it verbatim on each ``DatasetInfo``. It cannot hallucinate, so it
  is unconditional and always lands in the checkpoint.
* **Interpretation** (opt-in via ``USE_RUN_TITLE``) — the title seeds
  ``origin="header"`` structural hypotheses ranked between the user's and the
  skills'. Because they are *hypotheses*, a wrong title is rejected by
  evaluation's regression guardrail rather than poisoning the baseline.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np

REAL_FILE = os.path.join(os.path.dirname(__file__), "data", "REFL_218386_combined_data_auto.txt")
REAL_TITLE = "CuPt_d8-THF_FullQ-218386-1."

_PARSED = {
    "substrate": {"name": "Si", "sld": 2.07},
    "layers": [{"name": "Cu", "thickness": 500, "sld": 6.5}],
    "ambient": {"name": "THF", "sld": 0.0},
    "back_reflection": False,
}


def _make_data_file(name_prefix: str, header: str = "", n: int = 60) -> str:
    """A loadable synthetic data file with an arbitrary comment header."""
    Q = np.linspace(0.01, 0.10, n)
    R = np.clip((0.0217 / (2 * np.maximum(Q, 0.001))) ** 4, 1e-10, 1.0)
    path = os.path.join(tempfile.mkdtemp(), name_prefix)
    with open(path, "w") as f:
        if header:
            f.write(header if header.endswith("\n") else header + "\n")
        f.write("# Q R dR dQ\n")
        for q, r in zip(Q, R):
            f.write(f"{q:.6f}  {r:.6e}  {0.05 * r:.6e}  {0.02 * q:.6e}\n")
    return path


def _ds(file_path: str, label: str | None = None) -> dict:
    return {"file": file_path, "label": label or os.path.basename(file_path)}


# ----------------------------------------------------------------------
# Part 1: extraction (deterministic, always on)
# ----------------------------------------------------------------------


def test_run_title_extracted_verbatim_from_real_header():
    """The real REF_L header's title is returned exactly, trailing period kept.

    Verbatim matters: the field is provenance, so a normalized value would make
    the checkpoint disagree with the file it came from.
    """
    from aure.nodes.intake import _parse_run_title_from_header

    assert _parse_run_title_from_header(REAL_FILE) == REAL_TITLE


def test_run_title_label_variants_and_case():
    from aure.nodes.intake import _parse_run_title_from_header

    for header, expected in (
        ("# Run title: abc_d8", "abc_d8"),
        ("# RUN TITLE:   abc_d8   ", "abc_d8"),
        ("# Title: abc_d8", "abc_d8"),
        ("#run title:abc_d8", "abc_d8"),
    ):
        path = _make_data_file("REFL_1_combined_data_auto.txt", header=header)
        try:
            assert _parse_run_title_from_header(path) == expected, header
        finally:
            os.unlink(path)


def test_run_title_absent_returns_empty():
    from aure.nodes.intake import _parse_run_title_from_header

    path = _make_data_file("REFL_1_combined_data_auto.txt", header="# Reduction 2.2.0")
    try:
        assert _parse_run_title_from_header(path) == ""
    finally:
        os.unlink(path)


def test_run_title_missing_file_returns_empty():
    """Extraction is best-effort: an unreadable file must never raise."""
    from aure.nodes.intake import _parse_run_title_from_header

    assert _parse_run_title_from_header("/nonexistent/nope.txt") == ""


def test_run_title_not_scavenged_from_data_block():
    """Scanning stops at the data block, so stray text below it is ignored."""
    from aure.nodes.intake import _parse_run_title_from_header

    path = _make_data_file("REFL_1_combined_data_auto.txt")
    with open(path, "a") as f:
        f.write("# Title: sneaked_in_after_the_data\n")
    try:
        assert _parse_run_title_from_header(path) == ""
    finally:
        os.unlink(path)


def test_run_title_length_capped():
    """A pathological header line cannot dominate a downstream prompt."""
    from aure.nodes.intake import _MAX_RUN_TITLE_LEN, _parse_run_title_from_header

    path = _make_data_file(
        "REFL_1_combined_data_auto.txt", header="# Run title: " + "x" * 5000
    )
    try:
        assert len(_parse_run_title_from_header(path)) == _MAX_RUN_TITLE_LEN
    finally:
        os.unlink(path)


def test_enrich_dataset_records_run_title():
    from aure.nodes.intake import _enrich_dataset

    assert _enrich_dataset(_ds(REAL_FILE))["run_title"] == REAL_TITLE


def test_enrich_dataset_preserves_explicit_run_title():
    """An explicitly supplied title wins, matching theta/dq_is_fwhm behaviour."""
    from aure.nodes.intake import _enrich_dataset

    ds = _ds(REAL_FILE)
    ds["run_title"] = "supplied by caller"
    assert _enrich_dataset(ds)["run_title"] == "supplied by caller"


def test_intake_records_run_title_even_with_flag_off(monkeypatch):
    """The title reaches the checkpoint regardless of USE_RUN_TITLE.

    This is what makes "would the title have helped?" answerable from a corpus
    of past runs before the default is ever flipped.
    """
    from aure.nodes.intake import intake_node
    from aure.state import create_initial_state

    monkeypatch.delenv("USE_RUN_TITLE", raising=False)
    path = _make_data_file(
        "REFL_226642_combined_data_auto.txt", header="# Run title: CuPt_d8-THF"
    )
    try:
        state = create_initial_state(
            data_file=path,
            sample_description="",
            states=[{"name": "d8", "data_files": [_ds(path, "d8")]}],
        )
        result = intake_node(state)
        assert "error" not in result, result.get("error")
        assert result["states"][0]["data_files"][0]["run_title"] == "CuPt_d8-THF"
        assert any(
            "Run title from file header" in m["content"] and "not used" in m["content"]
            for m in result["messages"]
        ), result["messages"]
    finally:
        os.unlink(path)


# ----------------------------------------------------------------------
# Reconciliation across files within one state
# ----------------------------------------------------------------------


def test_reconcile_agreeing_titles():
    from aure.nodes.intake import _reconcile_run_titles

    assert _reconcile_run_titles([{"run_title": "A"}, {"run_title": "A"}]) == ("A", None)


def test_reconcile_conflicting_titles_yields_no_title_and_a_warning():
    """Conflict is a signal, not noise: warn, and use neither label."""
    from aure.nodes.intake import _reconcile_run_titles

    title, warning = _reconcile_run_titles([{"run_title": "A"}, {"run_title": "B"}])
    assert title == ""
    assert warning and "different run titles" in warning
    assert "'A'" in warning and "'B'" in warning


def test_reconcile_absent_titles_is_silent():
    from aure.nodes.intake import _reconcile_run_titles

    assert _reconcile_run_titles([{}, {"run_title": ""}]) == ("", None)
    assert _reconcile_run_titles([]) == ("", None)


def test_intake_warns_on_conflicting_titles_within_one_state():
    from aure.nodes.intake import intake_node
    from aure.state import create_initial_state

    f1 = _make_data_file("REFL_226642_1_2001_partial.txt", header="# Run title: A_d8")
    f2 = _make_data_file("REFL_226642_2_2002_partial.txt", header="# Run title: B_h8")
    try:
        state = create_initial_state(
            data_file=f1,
            sample_description="",
            states=[{"name": "s0", "data_files": [_ds(f1, "lo"), _ds(f2, "hi")]}],
        )
        result = intake_node(state)
        assert "error" not in result, result.get("error")
        assert any(
            "different run titles" in m["content"] for m in result["messages"]
        ), result["messages"]
    finally:
        os.unlink(f1)
        os.unlink(f2)


def test_run_titles_for_prompt_per_state():
    from aure.nodes.intake import _run_titles_for_prompt

    assert _run_titles_for_prompt(
        [
            {"name": "d8", "data_files": [{"run_title": "X_d8"}]},
            {"name": "h8", "data_files": [{"run_title": "X_h8"}]},
            {"name": "bad", "data_files": [{"run_title": "P"}, {"run_title": "Q"}]},
        ],
        [],
        "",
    ) == [{"state": "d8", "title": "X_d8"}, {"state": "h8", "title": "X_h8"}]


def test_run_titles_for_prompt_falls_back_to_primary_file():
    from aure.nodes.intake import _run_titles_for_prompt

    assert _run_titles_for_prompt([], [], REAL_FILE) == [
        {"state": "", "title": REAL_TITLE}
    ]
    assert _run_titles_for_prompt([], [{"run_title": "flat"}], "") == [
        {"state": "", "title": "flat"}
    ]


# ----------------------------------------------------------------------
# Part 2: the USE_RUN_TITLE gate
# ----------------------------------------------------------------------


def test_use_run_title_off_by_default(monkeypatch):
    from aure.nodes.intake import _use_run_title_enabled

    monkeypatch.delenv("USE_RUN_TITLE", raising=False)
    assert _use_run_title_enabled() is False


def test_use_run_title_env_values(monkeypatch):
    from aure.nodes.intake import _use_run_title_enabled

    for raw in ("1", "true", "TRUE", "yes", "on", " on "):
        monkeypatch.setenv("USE_RUN_TITLE", raw)
        assert _use_run_title_enabled() is True, raw
    for raw in ("0", "false", "off", "", "no"):
        monkeypatch.setenv("USE_RUN_TITLE", raw)
        assert _use_run_title_enabled() is False, raw


# ----------------------------------------------------------------------
# Part 2: prompt rendering
# ----------------------------------------------------------------------


def test_prompt_includes_titles_and_weak_evidence_framing():
    from aure.nodes.prompts import format_structural_hypothesis_prompt

    prompt = format_structural_hypothesis_prompt(
        sample_description="Si / Cu in THF",
        parsed_sample=_PARSED,
        skill_context="",
        run_titles=[{"state": "d8", "title": "CuPt_d8-THF"}],
    )
    assert 'd8: "CuPt_d8-THF"' in prompt
    assert "WEAK evidence" in prompt
    # The title must never be licensed to overrule the user's description.
    assert "never as a correction to the Sample" in prompt
    assert '`skill_source` set to "header"' in prompt


def test_prompt_says_none_available_when_titles_absent():
    """An explicit "none" keeps the model from substituting the file name."""
    from aure.nodes.prompts import format_structural_hypothesis_prompt

    prompt = format_structural_hypothesis_prompt(
        sample_description="Si / Cu in THF",
        parsed_sample=_PARSED,
        skill_context="",
    )
    assert "none available" in prompt
    assert "infer nothing from the file name" in prompt


def test_format_run_titles_skips_malformed_entries():
    from aure.nodes.prompts import _format_run_titles

    out = _format_run_titles([{"state": "a", "title": ""}, "junk", {"title": "keep"}])
    assert out == '- "keep"'


# ----------------------------------------------------------------------
# Part 2: hypothesis routing and ranking
# ----------------------------------------------------------------------


def _mock_llm(monkeypatch, payload):
    from aure.nodes import intake

    monkeypatch.setattr(intake, "llm_available", lambda: True)
    monkeypatch.setattr(intake, "get_llm", lambda temperature=0: MagicMock())
    monkeypatch.setattr(
        intake,
        "invoke_with_timeout",
        lambda llm, msgs: MagicMock(content=json.dumps(payload)),
    )


_THREE_SOURCES = [
    {
        "title": "Add native CuO",
        "rationale": "metal-oxide-interfaces",
        "change": "insert CuO",
        "skill_source": "metal-oxide-interfaces",
    },
    {
        "title": "Deuterated THF ambient",
        "rationale": 'header: "d8-THF" implies a deuterated solvent',
        "change": "reinterpret THF as d8-THF (SLD ~6.3)",
        "skill_source": "header",
    },
    {
        "title": "Oxide on top (user)",
        "rationale": "user said so",
        "change": "insert oxide",
        "skill_source": "user",
    },
]


def test_header_hypotheses_rank_between_user_and_skill(monkeypatch):
    """user > header > skill, regardless of the order the LLM emitted them."""
    from aure.nodes import intake

    _mock_llm(monkeypatch, _THREE_SOURCES)
    out = intake.generate_structural_hypotheses_with_llm(
        sample_description="Si / Cu in THF",
        parsed_sample=_PARSED,
        skill_context="",
        hypothesis="there may be an oxide on top",
        run_titles=[{"state": "", "title": "CuPt_d8-THF"}],
    )
    assert [h["origin"] for h in out] == ["user", "header", "skill"]
    assert [h["id"] for h in out] == [1, 2, 3]
    assert out[1]["skill_source"] == "header"
    assert out[1]["status"] == "pending"


def test_header_hypotheses_dropped_when_no_run_title(monkeypatch):
    """The flag has to gate the behaviour even if the model volunteers the label.

    With no title supplied there is no evidence behind a header-sourced entry,
    so it must not enter the list at all.
    """
    from aure.nodes import intake

    _mock_llm(monkeypatch, _THREE_SOURCES)
    out = intake.generate_structural_hypotheses_with_llm(
        sample_description="Si / Cu in THF",
        parsed_sample=_PARSED,
        skill_context="",
        hypothesis="there may be an oxide on top",
        run_titles=None,
    )
    assert [h["origin"] for h in out] == ["user", "skill"]
    assert all(h["skill_source"] != "header" for h in out)


def test_intake_passes_titles_only_when_flag_enabled(monkeypatch):
    """intake_node forwards the titles under the flag, and withholds them without it."""
    from aure.nodes import intake
    from aure.state import create_initial_state

    captured: list = []

    monkeypatch.setattr(intake, "llm_available", lambda: True)
    monkeypatch.setattr(intake, "parse_sample_with_llm", lambda *a, **k: dict(_PARSED))
    monkeypatch.setattr(
        intake,
        "generate_structural_hypotheses_with_llm",
        lambda **kwargs: captured.append(kwargs.get("run_titles")) or [],
    )

    path = _make_data_file(
        "REFL_226642_combined_data_auto.txt", header="# Run title: CuPt_d8-THF"
    )
    try:
        state = create_initial_state(
            data_file=path,
            sample_description="Si / Cu in THF",
            states=[{"name": "d8", "data_files": [_ds(path, "d8")]}],
        )

        monkeypatch.delenv("USE_RUN_TITLE", raising=False)
        intake.intake_node(state)
        assert captured[-1] is None

        monkeypatch.setenv("USE_RUN_TITLE", "1")
        result = intake.intake_node(state)
        assert captured[-1] == [{"state": "d8", "title": "CuPt_d8-THF"}]
        assert any(
            "seeding structural hypotheses" in m["content"] for m in result["messages"]
        ), result["messages"]
    finally:
        os.unlink(path)
