"""
Agent Skills system for domain-specific LLM prompting.

Skills follow the Agent Skills specification (https://agentskills.io/specification)
and provide domain knowledge that is injected into LLM prompts based on the
sample being analyzed.
"""

from .loader import SkillRegistry
from .selector import select_skills, load_skill_context

__all__ = ["SkillRegistry", "select_skills", "load_skill_context"]
