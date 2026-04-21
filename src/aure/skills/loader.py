"""
Skill loader: scan, parse, and serve Agent Skills from disk.

Follows the Agent Skills specification (https://agentskills.io/specification).
Metadata (name + description) is loaded eagerly; the full Markdown body and
referenced files are loaded on demand.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Validation patterns from the spec
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")
_MAX_NAME_LEN = 64
_MAX_DESC_LEN = 1024


@dataclass
class SkillMetadata:
    """Parsed SKILL.md frontmatter."""

    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    allowed_tools: Optional[str] = None
    path: Path = field(default_factory=lambda: Path("."))


class SkillRegistry:
    """Registry of available skills, lazily loading bodies on demand."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        if skills_dir is None:
            skills_dir = Path(__file__).parent
        self._skills_dir = skills_dir
        self._metadata: Dict[str, SkillMetadata] = {}
        self._body_cache: Dict[str, str] = {}
        self._scan()

    def _scan(self) -> None:
        """Scan the skills directory for valid skill subdirectories."""
        if not self._skills_dir.is_dir():
            logger.warning("Skills directory does not exist: %s", self._skills_dir)
            return

        for child in sorted(self._skills_dir.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                meta = self._parse_frontmatter(skill_md, child.name)
                self._metadata[meta.name] = meta
                logger.debug("Loaded skill metadata: %s", meta.name)
            except Exception as e:
                logger.warning("Skipping invalid skill '%s': %s", child.name, e)

    @staticmethod
    def _parse_frontmatter(skill_md: Path, dir_name: str) -> SkillMetadata:
        """Parse YAML frontmatter from a SKILL.md file."""
        text = skill_md.read_text(encoding="utf-8")

        if not text.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter (---)")

        end = text.index("---", 3)
        front = yaml.safe_load(text[3:end])
        if not isinstance(front, dict):
            raise ValueError("Frontmatter must be a YAML mapping")

        name = front.get("name", "")
        if not name:
            raise ValueError("'name' is required")
        if name != dir_name:
            raise ValueError(
                f"name '{name}' does not match directory name '{dir_name}'"
            )
        if len(name) > _MAX_NAME_LEN or not _NAME_RE.match(name):
            raise ValueError(f"Invalid skill name: '{name}'")
        if "--" in name:
            raise ValueError("Consecutive hyphens not allowed in name")

        desc = front.get("description", "")
        if not desc:
            raise ValueError("'description' is required")
        if len(desc) > _MAX_DESC_LEN:
            raise ValueError("description exceeds 1024 characters")

        return SkillMetadata(
            name=name,
            description=desc,
            license=front.get("license"),
            compatibility=front.get("compatibility"),
            metadata=front.get("metadata", {}),
            allowed_tools=front.get("allowed-tools"),
            path=skill_md.parent,
        )

    @property
    def skill_names(self) -> list[str]:
        """Return all registered skill names."""
        return list(self._metadata.keys())

    def get_metadata(self, name: str) -> SkillMetadata | None:
        """Return metadata for a skill by name."""
        return self._metadata.get(name)

    def all_metadata(self) -> list[SkillMetadata]:
        """Return metadata for all registered skills."""
        return list(self._metadata.values())

    def load_body(self, name: str) -> str:
        """Load the full Markdown body (after frontmatter) for a skill."""
        if name in self._body_cache:
            return self._body_cache[name]

        meta = self._metadata.get(name)
        if meta is None:
            raise KeyError(f"Unknown skill: '{name}'")

        text = (meta.path / "SKILL.md").read_text(encoding="utf-8")
        # Skip frontmatter
        end = text.index("---", 3)
        body = text[end + 3:].strip()
        self._body_cache[name] = body
        return body

    def load_reference(self, skill_name: str, ref_path: str) -> str:
        """Load a reference file from a skill's directory.

        Parameters
        ----------
        skill_name
            Name of the skill.
        ref_path
            Relative path within the skill directory (e.g. ``references/materials.md``).
        """
        meta = self._metadata.get(skill_name)
        if meta is None:
            raise KeyError(f"Unknown skill: '{skill_name}'")
        base_path = meta.path.resolve()
        target = (base_path / ref_path).resolve()
        # Ensure the target is under the skill directory (path traversal guard)
        if target != base_path and base_path not in target.parents:
            raise ValueError(f"Path traversal detected: {ref_path}")
        return target.read_text(encoding="utf-8")
