"""Tests for the untried-improvements report emitted by `finalize`.

A run now stops as soon as χ² meets the acceptance threshold, so the ranked
`structural_hypotheses` backlog is normally not exhausted. These tests pin the
rendering of the leftovers (and of the outcomes of what *was* attempted), and
the two return paths of `finalize_node` that must carry it.
"""

from aure.nodes.finalize import _format_remaining_improvements, finalize_node

# ======================================================================
# Fixtures
# ======================================================================


def _model():
    return {
        "layers": [{"name": "Cu", "thickness": 100.0, "sld": 6.4, "roughness": 5.0}],
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "D2O", "sld": 6.3},
        "intensity": {"value": 1.0},
    }


def _fit(iteration, chi2):
    return {
        "iteration": iteration,
        "method": "lm",
        "chi_squared": chi2,
        "bic": None,
        "parameters": {},
        "uncertainties": None,
    }


def _hypothesis(id_, title, status="pending", **extra):
    h = {
        "id": id_,
        "title": title,
        "rationale": f"why {title.lower()} is plausible",
        "change": f"concrete edit for {title.lower()}",
        "skill_source": "metal-oxide-interfaces",
        "origin": "skill",
        "status": status,
        "tried_in_iteration": None,
        "created_in_iteration": None,
        "notes": "",
    }
    h.update(extra)
    return h


def _state(fit_results, hypotheses=None, **extra):
    state = {
        "fit_results": fit_results,
        "model_history": [{"iteration": 0, "definition": _model()}],
        "current_model": _model(),
        "iteration": len(fit_results),
        "max_iterations": len(fit_results),
    }
    if hypotheses is not None:
        state["structural_hypotheses"] = hypotheses
    state.update(extra)
    return state


def _improvement_messages(updates):
    return [
        m
        for m in updates.get("messages", [])
        if not str(m.get("content", "")).startswith("**Final model:**")
    ]


# ======================================================================
# Formatter
# ======================================================================


def test_pending_entries_are_listed_with_their_change():
    text = _format_remaining_improvements(
        [
            _hypothesis(1, "Add native CuO on top of Cu"),
            _hypothesis(2, "Split Cu into two slabs"),
        ]
    )

    assert "Possible further improvements (not tried)" in text
    assert "Add native CuO on top of Cu" in text
    assert "concrete edit for add native cuo on top of cu" in text
    assert "concrete edit for split cu into two slabs" in text
    assert "metal-oxide-interfaces" in text
    assert "Already attempted" not in text


def test_pending_entries_keep_rank_order():
    """Rank is list position, not id — a re-ranked list must render as ranked."""
    text = _format_remaining_improvements(
        [_hypothesis(3, "Second best idea"), _hypothesis(1, "Best idea")]
    )

    assert text.index("Second best idea") < text.index("Best idea")


def test_outcomes_are_summarised_when_nothing_is_pending():
    text = _format_remaining_improvements(
        [
            _hypothesis(1, "Add native CuO on top of Cu", status="confirmed"),
            _hypothesis(2, "Split Cu into two slabs", status="rejected"),
            _hypothesis(3, "Roughen the Cu/D2O interface", status="rejected"),
        ]
    )

    assert "Possible further improvements" not in text
    assert "confirmed (1): Add native CuO on top of Cu" in text
    assert "rejected (2): Split Cu into two slabs, Roughen the Cu/D2O interface" in text


def test_tried_is_reported_as_inconclusive_not_failed():
    """The acceptance path never runs evaluation's outcome bookkeeping."""
    text = _format_remaining_improvements(
        [_hypothesis(1, "Add native CuO on top of Cu", status="tried")]
    )

    assert "tried, inconclusive (1): Add native CuO on top of Cu" in text
    assert "rejected" not in text


def test_pending_and_attempted_are_both_rendered():
    text = _format_remaining_improvements(
        [
            _hypothesis(1, "Add native CuO on top of Cu", status="rejected"),
            _hypothesis(2, "Split Cu into two slabs"),
        ]
    )

    lines = text.splitlines()
    assert lines[0].startswith("**Possible further improvements")
    assert lines[-1].startswith("**Already attempted:**")
    assert "Split Cu into two slabs" in lines[1]


def test_empty_or_absent_backlog_renders_nothing():
    assert _format_remaining_improvements(None) == ""
    assert _format_remaining_improvements([]) == ""


def test_untitled_entry_does_not_crash():
    text = _format_remaining_improvements([{"status": "pending"}])

    assert "untitled" in text


# ======================================================================
# Node wiring
# ======================================================================


def test_selection_message_comes_first_then_the_improvements():
    updates = finalize_node(
        _state([_fit(0, 1.5)], [_hypothesis(1, "Add native CuO on top of Cu")])
    )

    contents = [m["content"] for m in updates["messages"]]
    assert len(contents) == 2
    assert contents[0].startswith("**Final model:**")
    assert contents[1].startswith("**Possible further improvements")


def test_no_extra_message_without_a_backlog():
    for state in (_state([_fit(0, 1.5)]), _state([_fit(0, 1.5)], [])):
        updates = finalize_node(state)

        assert len(updates["messages"]) == 1
        assert updates["messages"][0]["content"].startswith("**Final model:**")


def test_improvements_are_reported_when_no_fit_was_usable():
    """A run that produced nothing fittable is when the ideas matter most."""
    updates = finalize_node(
        _state([_fit(0, float("inf"))], [_hypothesis(1, "Add native CuO on top of Cu")])
    )

    assert updates["final_selection"]["selected"] is False
    assert len(updates["messages"]) == 1
    assert updates["messages"][0]["content"].startswith("**Possible further")
    assert "Add native CuO on top of Cu" in updates["messages"][0]["content"]


def test_no_usable_fits_and_no_backlog_emits_no_message():
    updates = finalize_node(_state([]))

    assert updates["messages"] == []
    assert updates["finalized"] is True


def test_already_finalized_state_is_still_a_noop():
    state = _state([_fit(0, 1.5)], [_hypothesis(1, "Add native CuO on top of Cu")])

    first = finalize_node(state)
    assert _improvement_messages(first)

    state.update(first)
    assert finalize_node(state) == {}


def test_hypothesis_statuses_are_not_mutated():
    hypotheses = [
        _hypothesis(1, "Add native CuO on top of Cu", status="tried"),
        _hypothesis(2, "Split Cu into two slabs"),
    ]
    before = [dict(h) for h in hypotheses]

    updates = finalize_node(_state([_fit(0, 1.5)], hypotheses))

    assert hypotheses == before
    assert "structural_hypotheses" not in updates
    assert "structural_hypotheses" not in updates["final_selection"]
