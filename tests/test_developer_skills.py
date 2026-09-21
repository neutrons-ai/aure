"""The developer skills in ``.claude/skills/`` are not the science skills.

Two directories hold files called ``SKILL.md`` and they are for opposite
audiences:

``src/aure/skills/``
    Domain knowledge, selected by ``selector.select_skills`` and rendered into
    LLM prompts at run time. Read by a model, about reflectometry.
``.claude/skills/``
    Checklists for someone editing this repository. Read by a developer (or a
    coding agent), about AuRE's own code.

Putting a developer checklist in the first would feed a refactoring procedure
into a modelling prompt. Putting a science skill in the second would stop it
reaching the model at all. Neither failure announces itself, so the separation
is pinned here.
"""

from pathlib import Path

import pytest

import aure.skills as science_skills
from aure.skills import SkillRegistry

REPO = Path(__file__).resolve().parents[1]
DEV_SKILLS = REPO / ".claude" / "skills"


def _dev_skill_dirs():
    if not DEV_SKILLS.is_dir():
        return []
    return sorted(p for p in DEV_SKILLS.iterdir() if (p / "SKILL.md").is_file())


def test_the_add_data_format_skill_exists():
    assert (DEV_SKILLS / "add-data-format" / "SKILL.md").is_file()


def test_no_developer_skill_is_loaded_into_a_prompt():
    """The science registry must not pick these up.

    It scans its own package directory, so this holds by construction — which
    is exactly why it is worth a test: a later change to how skills are
    discovered could start sweeping the repository and nothing else would
    notice.
    """
    loaded = set(SkillRegistry().skill_names)
    developer = {p.name for p in _dev_skill_dirs()}

    assert developer, "no developer skills found; has the directory moved?"
    assert not (loaded & developer)


def test_the_science_skills_live_where_the_registry_looks():
    package_dir = Path(science_skills.__file__).resolve().parent

    assert package_dir != DEV_SKILLS
    assert DEV_SKILLS not in package_dir.parents


@pytest.mark.parametrize("skill", _dev_skill_dirs(), ids=lambda p: p.name)
def test_a_developer_skill_says_which_kind_it_is(skill):
    """Stated in the file, because the directory alone is easy to mistake."""
    text = (skill / "SKILL.md").read_text()

    assert "developer skill" in text.lower()
    assert "src/aure/skills" in text  # names the other one, to disambiguate


@pytest.mark.parametrize("skill", _dev_skill_dirs(), ids=lambda p: p.name)
def test_a_developer_skill_has_the_frontmatter_the_spec_wants(skill):
    text = (skill / "SKILL.md").read_text()

    assert text.startswith("---\n")
    front = text.split("---", 2)[1]
    assert f"name: {skill.name}" in front
    assert "description:" in front


def test_the_add_data_format_skill_carries_the_negative_test():
    """The rule that keeps the seam a seam.

    If adding a format means editing a node, the protocol is short a member —
    that is the thing this skill exists to say, so a rewrite that drops it has
    lost the point.
    """
    text = (DEV_SKILLS / "add-data-format" / "SKILL.md").read_text()

    assert "src/aure/nodes/" in text
    assert "2.355" in text  # the resolution trap
    assert "per-acquisition" in text  # the array-length trap
