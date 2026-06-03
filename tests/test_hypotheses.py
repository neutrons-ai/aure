"""Tests for the structural-hypothesis merge guard (nodes/hypotheses.py).

These are pure-function tests — the guard is the single point that keeps the
``structural_hypotheses`` list coherent across nodes, so its semantics are
pinned here independently of any LLM.
"""

import logging

from aure.nodes.hypotheses import (
    merge_structural_hypotheses,
    next_hypothesis_id,
    rerank_hypotheses,
)


def _hyp(hid, status="pending", origin="skill", **kw):
    base = {
        "id": hid,
        "title": f"H{hid}",
        "rationale": "r",
        "change": "c",
        "skill_source": "metal-oxide-interfaces",
        "origin": origin,
        "status": status,
        "tried_in_iteration": None,
        "created_in_iteration": None,
        "notes": "",
    }
    base.update(kw)
    return base


def test_next_hypothesis_id():
    assert next_hypothesis_id([]) == 1
    assert next_hypothesis_id([_hyp(1), _hyp(3)]) == 4


def test_merge_status_only_updates_existing():
    prior = [_hyp(1, status="pending"), _hyp(2, status="pending")]
    returned = [
        {"id": 1, "status": "tried", "tried_in_iteration": 2, "notes": "trying"},
        {"id": 2, "status": "confirmed"},
    ]
    out = merge_structural_hypotheses(
        prior, returned, allow_new=False, current_iteration=2
    )
    assert len(out) == 2
    assert out[0]["status"] == "tried"
    assert out[0]["tried_in_iteration"] == 2
    assert out[0]["notes"] == "trying"
    assert out[1]["status"] == "confirmed"


def test_merge_preserves_identity_fields():
    """The LLM cannot rename or re-scope an existing hypothesis via merge."""
    prior = [_hyp(1, status="pending")]
    returned = [
        {
            "id": 1,
            "title": "HIJACKED",
            "change": "evil",
            "skill_source": "x",
            "origin": "user",
            "status": "tried",
        }
    ]
    out = merge_structural_hypotheses(
        prior, returned, allow_new=False, current_iteration=1
    )
    assert out[0]["title"] == "H1"
    assert out[0]["change"] == "c"
    assert out[0]["skill_source"] == "metal-oxide-interfaces"
    assert out[0]["origin"] == "skill"
    assert out[0]["status"] == "tried"  # only the mutable field changed


def test_merge_invalid_status_ignored():
    prior = [_hyp(1, status="pending")]
    out = merge_structural_hypotheses(
        prior, [{"id": 1, "status": "bogus"}], allow_new=False, current_iteration=1
    )
    assert out[0]["status"] == "pending"


def test_merge_drops_new_when_not_allowed(caplog):
    prior = [_hyp(1)]
    returned = [{"id": 1, "status": "tried"}, {"title": "fabricated", "change": "x"}]
    with caplog.at_level(logging.WARNING, logger="aure.nodes.hypotheses"):
        out = merge_structural_hypotheses(
            prior, returned, allow_new=False, current_iteration=1
        )
    assert len(out) == 1
    assert any("dropped" in r.message.lower() for r in caplog.records)


def test_merge_appends_new_when_allowed():
    prior = [_hyp(1, status="confirmed")]
    returned = [
        {
            "title": "New from eval",
            "rationale": "rr",
            "change": "cc",
            "skill_source": "sei-layer-analysis",
        }
    ]
    out = merge_structural_hypotheses(
        prior, returned, allow_new=True, current_iteration=4
    )
    assert len(out) == 2
    new = out[1]
    assert new["id"] == 2
    assert new["title"] == "New from eval"
    assert new["origin"] == "evaluation"
    assert new["status"] == "pending"
    assert new["created_in_iteration"] == 4
    assert out[0]["status"] == "confirmed"  # existing untouched


def test_merge_new_without_title_dropped():
    prior = [_hyp(1)]
    out = merge_structural_hypotheses(
        prior, [{"rationale": "no title"}], allow_new=True, current_iteration=2
    )
    assert len(out) == 1


def test_rerank_reorders_by_ids():
    hyps = [_hyp(1), _hyp(2), _hyp(3)]
    out = rerank_hypotheses(hyps, [3, 1])
    # unranked (#2) keeps its relative position at the end
    assert [h["id"] for h in out] == [3, 1, 2]


def test_rerank_ignores_unknown_ids_and_empty():
    hyps = [_hyp(1), _hyp(2)]
    assert [h["id"] for h in rerank_hypotheses(hyps, [])] == [1, 2]
    assert [h["id"] for h in rerank_hypotheses(hyps, [9, 2])] == [2, 1]
