"""
Agentic Reflectivity Modeling Workflow.

This module provides an intelligent assistant for analyzing neutron reflectivity data
using a hand-written workflow state machine and refl1d for optimization.

Main components:
- workflow: the analysis pipeline state machine (runner.py)
- state: State definitions for the workflow
- tools: LangChain tools for data loading, feature extraction, and model building
- cli: Click-based command-line interface

Usage:
    # Python API
    from aure import run_analysis
    result = run_analysis("data.dat", "100 nm polystyrene on silicon")

    # CLI
    python -m aure.cli analyze data.dat "100 nm polystyrene on silicon"
"""

import warnings
from importlib import metadata

# Suppress compatibility warnings for Python 3.14+
warnings.filterwarnings(
    "ignore", message="Core Pydantic V1 functionality isn't compatible"
)

from .state import ReflectivityState, create_initial_state
from .workflow import run_analysis

try:
    #: The installed version, read from package metadata so there is exactly
    #: one place it is declared: ``pyproject.toml``. It used to be spelled out
    #: again in ``cli.py``'s ``--version``, which then reported ``0.1.0`` for
    #: every release after 0.1 — a version string that disagrees with the
    #: package it names is worse than none, because it is recorded into
    #: provenance and believed.
    __version__ = metadata.version("aure")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    # Imported from a tree that was never installed. "unknown" rather than a
    # literal: a stale literal is exactly the failure above.
    __version__ = "unknown"

__all__ = [
    "ReflectivityState",
    "create_initial_state",
    "run_analysis",
    "__version__",
]
