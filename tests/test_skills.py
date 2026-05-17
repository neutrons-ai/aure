"""Tests for the Agent Skills infrastructure."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from aure.skills.loader import SkillRegistry, SkillMetadata
from aure.skills.selector import (
    select_skills,
    load_skill_context,
    _build_catalog,
    _build_sample_info,
)


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
        assert "structural-hypothesis-ranking" in names
        assert "multi-state-corefinement" in names
        assert len(names) == 7

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
        assert len(all_meta) == 7
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
            assert len(meta.description) > 10, (
                f"Skill '{meta.name}' has too short a description"
            )

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
# LLM Skill Selection Tests
# ============================================================================


class TestLLMSkillSelection:
    """Tests for LLM-based skill selection."""

    def test_build_catalog(self):
        registry = SkillRegistry()
        catalog = _build_catalog(registry)
        assert "metal-oxide-interfaces" in catalog
        assert "polymer-films" in catalog
        assert "sei-layer-analysis" in catalog
        assert "solvent-contrast-matching" in catalog
        # Baseline skill is excluded from catalog (always on)
        assert "- **neutron-reflectometry**" not in catalog

    def test_build_sample_info_description_only(self):
        info = _build_sample_info("copper on silicon in dTHF")
        assert "copper on silicon in dTHF" in info

    def test_build_sample_info_with_parsed(self):
        parsed = {
            "substrate": {"name": "silicon"},
            "layers": [{"name": "titanium"}, {"name": "copper"}],
            "ambient": {"name": "dTHF"},
        }
        info = _build_sample_info("my sample", parsed_sample=parsed)
        assert "titanium" in info
        assert "copper" in info
        assert "silicon" in info
        assert "dTHF" in info

    def test_llm_selection_valid_response(self):
        """LLM returning a valid JSON array of skill names."""
        mock_response = MagicMock()
        mock_response.content = '["metal-oxide-interfaces", "sei-layer-analysis"]'

        with (
            patch("aure.llm.llm_available", return_value=True),
            patch("aure.llm.get_llm") as mock_get,
            patch("aure.llm.invoke_with_timeout", return_value=mock_response),
        ):
            mock_get.return_value = MagicMock()
            result = select_skills(
                "copper electrode with SEI",
                registry=SkillRegistry(),
            )
        assert "neutron-reflectometry" in result
        assert "metal-oxide-interfaces" in result
        assert "sei-layer-analysis" in result

    def test_llm_selection_filters_invalid_names(self):
        """LLM hallucinating a nonexistent skill name is filtered out."""
        mock_response = MagicMock()
        mock_response.content = '["metal-oxide-interfaces", "nonexistent-skill"]'

        with (
            patch("aure.llm.llm_available", return_value=True),
            patch("aure.llm.get_llm") as mock_get,
            patch("aure.llm.invoke_with_timeout", return_value=mock_response),
        ):
            mock_get.return_value = MagicMock()
            result = select_skills(
                "copper sample",
                registry=SkillRegistry(),
            )
        assert "metal-oxide-interfaces" in result
        assert "nonexistent-skill" not in result

    def test_llm_selection_empty_array(self):
        """LLM returning empty array still includes baseline."""
        mock_response = MagicMock()
        mock_response.content = "[]"

        with (
            patch("aure.llm.llm_available", return_value=True),
            patch("aure.llm.get_llm") as mock_get,
            patch("aure.llm.invoke_with_timeout", return_value=mock_response),
        ):
            mock_get.return_value = MagicMock()
            result = select_skills(
                "simple silicon wafer",
                registry=SkillRegistry(),
            )
        assert set(result) == {"neutron-reflectometry", "structural-hypothesis-ranking"}

    def test_llm_failure_returns_baseline_only(self):
        """When LLM call raises, return baseline skill only."""
        with (
            patch("aure.llm.llm_available", return_value=True),
            patch("aure.llm.get_llm", side_effect=RuntimeError("no LLM")),
        ):
            result = select_skills(
                "copper with native oxide layer",
                registry=SkillRegistry(),
            )
        assert set(result) == {"neutron-reflectometry", "structural-hypothesis-ranking"}

    def test_llm_unavailable_returns_baseline_only(self):
        """When no LLM is configured, return baseline skill only."""
        with patch("aure.llm.llm_available", return_value=False):
            result = select_skills(
                "polystyrene thin film",
                registry=SkillRegistry(),
            )
        assert set(result) == {"neutron-reflectometry", "structural-hypothesis-ranking"}

    def test_llm_response_with_markdown_fences(self):
        """LLM wrapping response in ```json fences."""
        mock_response = MagicMock()
        mock_response.content = '```json\n["polymer-films"]\n```'

        with (
            patch("aure.llm.llm_available", return_value=True),
            patch("aure.llm.get_llm") as mock_get,
            patch("aure.llm.invoke_with_timeout", return_value=mock_response),
        ):
            mock_get.return_value = MagicMock()
            result = select_skills(
                "polystyrene sample",
                registry=SkillRegistry(),
            )
        assert "polymer-films" in result


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
