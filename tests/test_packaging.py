"""Packaging tests: assert the data files actually ship in a built wheel.

Why this file exists
--------------------
``tests/test_skills.py::test_scan_finds_all_skills`` asserts eight skills load,
and it passes -- but only because CI installs with ``pip install -e .``, where
``SkillRegistry`` reads straight from the source tree. It says nothing about
what a *built distribution* contains.

The skill directories carry no ``__init__.py``, so ``packages.find`` does not
collect them. Until they were declared in ``[tool.setuptools.package-data]``,
no ``SKILL.md`` appeared in the wheel at all. Every non-editable install --
notably the ``ghcr.io/neutrons-ai/aure`` image, whose Dockerfile runs
``pip install ".[export]"`` -- therefore started with an empty registry and ran
every prompt with no domain knowledge, silently.

These tests build a real wheel and look inside it, so the gap cannot reopen.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Never copied into the pristine build tree. `*.egg-info` matters most: a
# SOURCES.txt left over from an earlier build is reused by setuptools and will
# happily re-include files the current configuration no longer packages, so a
# developer's stale egg-info can mask a genuine packaging regression.
_BUILD_EXCLUDES = shutil.ignore_patterns(
    "*.egg-info",
    ".git",
    ".venv",
    "venv",
    "build",
    "dist",
    "htmlcov",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)

# Kept in step with tests/test_skills.py::test_scan_finds_all_skills.
EXPECTED_SKILLS = {
    "metal-oxide-interfaces",
    "multi-state-corefinement",
    "neutron-reflectometry",
    "polymer-films",
    "sei-layer-analysis",
    "solvent-contrast-matching",
    "structural-hypothesis-ranking",
    "thin-layer-degeneracy",
}


@pytest.fixture(scope="module")
def wheel_namelist(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """Build a wheel from a pristine copy of the repo and list its contents.

    The build runs against a copy with all build artifacts stripped, so the
    result reflects the committed configuration alone and not whatever an
    earlier local build happened to leave behind.

    Skips (rather than fails) when the wheel cannot be built at all -- no
    network for build isolation, a read-only checkout, and so on. A build that
    *succeeds* but omits files is a real failure and is asserted on below.
    """
    if not (REPO_ROOT / "pyproject.toml").is_file():
        pytest.skip("not running from a source checkout")

    workdir = tmp_path_factory.mktemp("src")
    pristine = workdir / "aure"
    shutil.copytree(REPO_ROOT, pristine, ignore=_BUILD_EXCLUDES, symlinks=True)

    outdir = tmp_path_factory.mktemp("wheel")
    base = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-cache-dir",
        "--wheel-dir",
        str(outdir),
    ]

    # Try without build isolation first: it needs no network and is much
    # faster. It fails when setuptools is absent from the active venv (the
    # default for Python 3.12+ venvs), so fall back to an isolated build.
    attempts = [[*base, "--no-build-isolation", str(pristine)], [*base, str(pristine)]]
    result = None
    for command in attempts:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        if result.returncode == 0:
            break
    if result is None or result.returncode != 0:
        stderr = result.stderr[-2000:] if result else "no attempt ran"
        pytest.skip(f"could not build a wheel in this environment:\n{stderr}")

    wheels = list(outdir.glob("aure-*.whl"))
    if not wheels:
        pytest.skip("pip reported success but produced no aure wheel")

    with zipfile.ZipFile(wheels[0]) as archive:
        return archive.namelist()


def test_wheel_contains_every_skill_md(wheel_namelist: list[str]) -> None:
    """Every skill directory's SKILL.md must be present in the wheel."""
    packaged = {
        name.split("/")[2]
        for name in wheel_namelist
        if name.startswith("aure/skills/") and name.endswith("/SKILL.md")
    }
    missing = EXPECTED_SKILLS - packaged
    assert not missing, (
        f"SKILL.md missing from the wheel for: {sorted(missing)}. "
        "Check the 'aure.skills' entry in [tool.setuptools.package-data]. "
        "Without it, non-editable installs run with no domain knowledge."
    )


def test_wheel_contains_skill_references(wheel_namelist: list[str]) -> None:
    """Any references/ material next to a skill must ship with it.

    Skills may cite reference files via ``SkillRegistry.load_reference``; a
    packaged skill whose references were dropped fails only at call time.
    """
    on_disk = sorted(
        path.relative_to(REPO_ROOT / "src").as_posix()
        for path in (REPO_ROOT / "src" / "aure" / "skills").glob("*/references/*")
        if path.is_file()
    )
    if not on_disk:
        pytest.skip("no skill reference files in this checkout")

    packaged = set(wheel_namelist)
    missing = [path for path in on_disk if path not in packaged]
    assert not missing, f"skill reference files missing from the wheel: {missing}"


def test_wheel_contains_web_assets(wheel_namelist: list[str]) -> None:
    """The Flask templates and static assets must ship (`aure serve`)."""
    assert any(
        name.startswith("aure/web/templates/") and name.endswith(".html")
        for name in wheel_namelist
    ), "no aure/web/templates/*.html in the wheel"
    assert any(name.startswith("aure/web/static/") for name in wheel_namelist), (
        "no aure/web/static/* in the wheel"
    )


def test_registry_is_not_silently_empty(caplog: pytest.LogCaptureFixture) -> None:
    """An empty skills directory must log an error, not fail silently.

    The original failure mode was invisible: no exception, no warning, just
    prompts with no domain knowledge. Absence of skills is always a defect, so
    it must be loud.
    """
    import logging

    from aure.skills.loader import SkillRegistry

    empty = REPO_ROOT / "tests" / "_no_such_skills_dir_"
    empty.mkdir(exist_ok=True)
    try:
        with caplog.at_level(logging.ERROR):
            registry = SkillRegistry(skills_dir=empty)
        assert registry.skill_names == []
        assert any(record.levelno >= logging.ERROR for record in caplog.records), (
            "an empty skills directory must be reported at ERROR level"
        )
    finally:
        empty.rmdir()


# ======================================================================
# requires-python must not undersell what the code actually needs
# ======================================================================
#
# `requires-python` said ">=3.9" while six modules used PEP 604 unions
# (`str | None`) in function signatures. Signature annotations are evaluated at
# import time, and `type.__or__` arrived in 3.10, so `import aure` raised
# TypeError on 3.9 -- the floor was never real. refl1d and bumps independently
# require >=3.10, so the claim was false twice over.
#
# CI only tested 3.12, which is how the two drifted apart. These tests derive
# the true floor from the source and compare it to the declaration, so a lowered
# `requires-python` fails here rather than on a user's machine.

_PEP604_MIN = (3, 10)


def _declared_requires_python() -> str:
    """Read `project.requires-python` out of pyproject.toml.

    `tomllib` is 3.11+ (PEP 680) and the declared floor is 3.10, which CI now
    tests -- so this must not depend on it. Falls back to a line scan, which is
    sufficient for a single flat string key.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import re

        m = re.search(r'^requires-python\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert m, "could not find requires-python in pyproject.toml"
        return m.group(1)
    return tomllib.loads(text)["project"]["requires-python"]


def _declared_floor() -> tuple[int, int]:
    spec = _declared_requires_python()
    assert spec.startswith(">="), f"expected a >= floor, got {spec!r}"
    major, _, minor = spec[2:].strip().partition(".")
    return int(major), int(minor)


def _modules_using_evaluated_pep604() -> dict[str, list[int]]:
    """Files with `X | Y` in an annotation Python evaluates at import time.

    Skips files carrying `from __future__ import annotations`, where every
    annotation becomes a string and PEP 604 is therefore legal on 3.9. Local
    variable annotations inside function bodies are never evaluated, so only
    signatures, return types, and module-/class-level annotations count.
    """
    import ast

    def has_future(tree: ast.Module) -> bool:
        return any(
            isinstance(n, ast.ImportFrom)
            and n.module == "__future__"
            and any(a.name == "annotations" for a in n.names)
            for n in tree.body
        )

    def uses_bitor(node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr)
            for n in ast.walk(node)
        )

    found: dict[str, list[int]] = {}
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if has_future(tree):
            continue
        lines: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg in (
                    *a.posonlyargs,
                    *a.args,
                    *a.kwonlyargs,
                    a.vararg,
                    a.kwarg,
                ):
                    if (
                        arg is not None
                        and arg.annotation
                        and uses_bitor(arg.annotation)
                    ):
                        lines.append(arg.annotation.lineno)
                if node.returns is not None and uses_bitor(node.returns):
                    lines.append(node.returns.lineno)
        scopes = [tree.body]
        scopes += [b.body for b in tree.body if isinstance(b, ast.ClassDef)]
        for body in scopes:
            for node in body:
                if (
                    isinstance(node, ast.AnnAssign)
                    and node.annotation
                    and uses_bitor(node.annotation)
                ):
                    lines.append(node.annotation.lineno)
        if lines:
            found[str(path.relative_to(REPO_ROOT))] = sorted(set(lines))
    return found


def test_requires_python_covers_the_syntax_the_code_uses() -> None:
    """The declared floor must be >= what evaluated PEP 604 annotations need."""
    offenders = _modules_using_evaluated_pep604()
    if not offenders:
        pytest.skip("no evaluated PEP 604 annotations remain in src/")

    declared = _declared_floor()
    detail = ", ".join(f"{f} (L{v[0]}…)" for f, v in list(offenders.items())[:4])
    assert declared >= _PEP604_MIN, (
        f"requires-python declares >={declared[0]}.{declared[1]} but "
        f"{len(offenders)} module(s) use PEP 604 unions in annotations Python "
        f"evaluates at import time, which need "
        f">={_PEP604_MIN[0]}.{_PEP604_MIN[1]}: {detail}. Either raise "
        "requires-python or add `from __future__ import annotations` to those "
        "modules."
    )


def test_requires_python_is_at_least_what_our_dependencies_demand() -> None:
    """refl1d and bumps set their own floors; ours cannot sit below them."""
    import importlib.metadata as md

    declared = _declared_floor()
    checked = 0
    for dep in ("refl1d", "bumps"):
        try:
            spec = md.metadata(dep).get("Requires-Python")
        except md.PackageNotFoundError:
            continue
        if not spec or not spec.strip().startswith(">="):
            continue
        major, _, minor = spec.strip()[2:].partition(".")
        try:
            dep_floor = (int(major), int(minor.split(",")[0]))
        except ValueError:
            continue
        checked += 1
        assert declared >= dep_floor, (
            f"requires-python is >={declared[0]}.{declared[1]} but {dep} "
            f"requires {spec}"
        )
    if not checked:
        pytest.skip("neither refl1d nor bumps is installed")
