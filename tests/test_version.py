"""The version is declared once, in ``pyproject.toml``.

It used to be declared twice. ``cli.py`` carried ``version="0.1.0"`` in its
``--version`` option and never moved, so every release from 0.1 onwards
reported 0.1.0 from the command line while the package metadata said something
else. nr-workbench felt that downstream: its provenance module records the
*commit* rather than the version precisely because "the version string does not
identify a build".

A version that disagrees with the package it names is worse than no version,
because it gets recorded into provenance and believed.
"""

from importlib import metadata
from pathlib import Path

import pytest
from click.testing import CliRunner

import aure
from aure.cli import cli


def test_the_package_reports_the_installed_version():
    assert aure.__version__ == metadata.version("aure")


def test_the_cli_reports_the_same_version():
    """`--version` is the copy that drifted, so this is the one that matters."""
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert aure.__version__ in result.output


def test_the_version_is_not_a_literal_in_the_cli():
    """Pinned because a literal is how the drift happened, not a style rule."""
    source = (Path(aure.__file__).parent / "cli.py").read_text()

    assert 'version_option(version="' not in source


def test_the_installed_version_matches_pyproject():
    """Catches a stale editable install as well as a code-side drift.

    Skipped when this is not a source checkout — a wheel has no pyproject, and
    the question does not arise.
    """
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        pytest.skip("tomllib needs Python 3.11+")

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():  # pragma: no cover - installed, not a checkout
        pytest.skip("not a source checkout")

    declared = tomllib.loads(pyproject.read_text())["project"]["version"]

    assert aure.__version__ == declared, (
        f"installed metadata says {aure.__version__}, pyproject says "
        f"{declared} — reinstall with `pip install -e . --no-deps`"
    )
