"""Tests for the Agent Skills infrastructure."""

import pytest
from pathlib import Path

from aure.skills.loader import SkillRegistry, SkillMetadata
from aure.skills.selector import select_skills, load_skill_context


# ============================================================================
# SkillRegistry Tests
# ============================================================================


class TestSkillRegistry:
    """Tests for skill scanning, parsing, and loading."""

    def test_scan_finds_all_skills(self):
        registry = SkillRegistry()
        names = registry.skill_names
        assert "neutron-reflectometry" in names
        assert "sei-layer-analysis" in names
        assert "polymer-films" in names
        assert "metal-oxide-interfaces" in names
        assert "solvent-contrast-matching" in names
        assert len(names) == 5

    def test_metadata_parsed_correctly(self):
        registry = SkillRegistry()
        meta = registry.get_metadata("neutron-reflectometry")
        assert meta is not None
        assert meta.name == "neutron-reflectometry"
        assert "reflectometry" in meta.description.lower()
        assert isinstance(meta.path, Path)

    def test_all_metadata_returns_list(self):
        registry = SkillRegistry()
        all_meta = registry.all_metadata()
        assert len(all_meta) == 5
        assert all(isinstance(m, SkillMetadata) for m in all_meta)

    def test_load_body(self):
        registry = SkillRegistry()
        body = registry.load_body("neutron-reflectometry")
        assert "SLD" in body
        assert "χ²" in body or "chi" in body.lower()

    def test_load_body_caches(self):
        registry = SkillRegistry()
        body1 = registry.load_body("neutron-reflectometry")
        body2 = registry.load_body("neutron-reflectometry")
        assert body1 is body2  # Same object (cached)

    def test_load_body_unknown_skill_raises(self):
        registry = SkillRegistry()
        with pytest.raises(KeyError, match="nonexistent"):
            registry.load_body("nonexistent")

    def test_name_matches_directory(self):
        """Verify each skill's name field matches its directory name."""
        registry = SkillRegistry()
        for meta in registry.all_metadata():
            assert meta.name == meta.path.name

    def test_descriptions_nonempty(self):
        registry = SkillRegistry()
        for meta in registry.all_metadata():
            assert len(meta.description) > 10, f"Skill '{meta.name}' has too short a description"

    def test_empty_directory(self, tmp_path):
        """Registry handles empty skills directory gracefully."""
        registry = SkillRegistry(tmp_path)
        assert registry.skill_names == []

    def test_invalid_skill_skipped(self, tmp_path):
        """Skills with invalid frontmatter are skipped."""
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("no frontmatter here")
        registry = SkillRegistry(tmp_path)
        assert registry.skill_names == []

    def test_path_traversal_blocked(self):
        registry = SkillRegistry()
        with pytest.raises(ValueError, match="traversal"):
            registry.load_reference("neutron-reflectometry", "../../pyproject.toml")


# ============================================================================
# Skill Selector Tests
# ============================================================================


class TestSelectSkills:
    """Tests for skill selection heuristic."""

    def test_baseline_always_selected(self):
        result = select_skills("a simple sample", registry=SkillRegistry())
        assert "neutron-reflectometry" in result

    def test_sei_keywords(self):
        result = select_skills(
            "copper electrode with SEI layer in battery cell",
            registry=SkillRegistry(),
        )
        assert "sei-layer-analysis" in result
        assert "neutron-reflectometry" in result

    def test_polymer_keywords(self):
        result = select_skills(
            "polystyrene thin film spin-coated on silicon",
            registry=SkillRegistry(),
        )
        assert "polymer-films" in result

    def test_oxide_keywords(self):
        result = select_skills(
            "copper with native oxide layer",
            registry=SkillRegistry(),
        )
        assert "metal-oxide-interfaces" in result

    def test_solvent_keywords(self):
        result = select_skills(
            "sample measured in D2O",
            registry=SkillRegistry(),
        )
        assert "solvent-contrast-matching" in result

    def test_parsed_sample_materials_trigger_skills(self):
        parsed = {
            "substrate": {"name": "silicon"},
            "layers": [{"name": "polystyrene"}],
            "ambient": {"name": "D2O"},
            "constraints": [],
        }
        result = select_skills("sample", parsed_sample=parsed, registry=SkillRegistry())
        assert "polymer-films" in result
        assert "solvent-contrast-matching" in result

    def test_no_false_positives_plain_sample(self):
        result = select_skills(
            "silicon substrate with gold layer in air",
            registry=SkillRegistry(),
        )
        # Should only get baseline
        assert "sei-layer-analysis" not in result
        assert "polymer-films" not in result

    def test_short_keyword_word_boundary(self):
        """Short keywords like 'ps' should use word boundaries."""
        result = select_skills(
            "sample with steps and ramps",
            registry=SkillRegistry(),
        )
        # 'ps' should NOT match inside 'steps'
        assert "polymer-films" not in result


# ============================================================================
# Skill Context Loading Tests
# ============================================================================


class TestLoadSkillContext:
    """Tests for combined skill body loading."""

    def test_single_skill(self):
        registry = SkillRegistry()
        ctx = load_skill_context(["neutron-reflectometry"], registry)
        assert "neutron-reflectometry" in ctx
        assert "SLD" in ctx

    def test_multiple_skills(self):
        registry = SkillRegistry()
        ctx = load_skill_context(
            ["neutron-reflectometry", "sei-layer-analysis"], registry
        )
        assert "neutron-reflectometry" in ctx
        assert "sei-layer-analysis" in ctx
        assert "SEI" in ctx

    def test_empty_list(self):
        registry = SkillRegistry()
        ctx = load_skill_context([], registry)
        assert ctx == ""

    def test_unknown_skill_skipped(self):
        registry = SkillRegistry()
        ctx = load_skill_context(["nonexistent"], registry)
        assert ctx == ""
