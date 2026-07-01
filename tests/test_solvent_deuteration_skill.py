"""Skill-content guards for the unspecified-deuteration behavior.

These lock in the guidance the workflow relies on: the solvent skill must teach
the unspecified-deuteration / rewind strategy and advertise a low activation
threshold, and the hypothesis-ranking skill must recognize reinterpretation
hypotheses and the rewind-to-baseline rule.
"""

from aure.skills.loader import SkillRegistry


def test_solvent_skill_description_low_threshold():
    meta = SkillRegistry().get_metadata("solvent-contrast-matching")
    desc = meta.description.lower()
    assert "electrolyte" in desc
    assert "deuterat" in desc
    # Framed around any liquid medium, not only stated-deuterated cases.
    assert "liquid" in desc


def test_solvent_skill_body_covers_unspecified_deuteration_and_rewind():
    body = SkillRegistry().load_body("solvent-contrast-matching").lower()
    assert "unspecified deuteration" in body
    assert "rewind" in body
    assert "baseline" in body


def test_hypothesis_skill_covers_reinterpretation_and_rewind():
    body = SkillRegistry().load_body("structural-hypothesis-ranking").lower()
    assert "reinterpret" in body
    assert "rewind" in body
    assert "deuterat" in body
