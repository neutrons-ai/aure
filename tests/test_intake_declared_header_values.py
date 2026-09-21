"""A declared ``theta`` / ``dq_is_fwhm`` outranks the header parse.

``config`` accepts these on a ``data_files`` entry (see
``test_setup_declared_header_values.py``); these tests pin that intake then
honours them, including on the single-file path, where ``build_problem``
reads the *state-level* flag rather than the dataset's.

The files here deliberately carry a header AuRE cannot read, which is the
situation the feature exists for: a format no registered instrument
understands falls back to ``dq_is_fwhm=True, theta=0.0``, and that default is
exactly what a sigma-column file needs overridden.
"""

import os
import tempfile

import numpy as np
import pytest

from aure.nodes import intake
from aure.nodes.intake import intake_node
from aure.state import create_initial_state


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Pin the deterministic parse, whatever else the session has imported.

    Not a convenience. ``aure.cli`` calls ``load_dotenv`` at import, so any
    test module importing it leaves a developer's real ``LLM_PROVIDER`` in
    ``os.environ`` for the rest of the process — and these tests would then
    send the fixture header to a live model and assert on what it replied.
    What is under test here is which value wins, not who read the file.
    """
    monkeypatch.setattr(intake, "llm_available", lambda: False)


def _unreadable_header_file(name: str) -> str:
    """A file whose header no instrument claims — the fallback case."""
    Q = np.linspace(0.01, 0.10, 40)
    R = np.clip((0.0217 / (2 * np.maximum(Q, 0.001))) ** 4, 1e-10, 1.0)
    path = os.path.join(tempfile.mkdtemp(), name)
    with open(path, "w") as f:
        f.write("# columns = Q, R, dR, dQ (sigma)\n")
        for q, r in zip(Q, R):
            f.write(f"{q:.6f}  {r:.6e}  {0.05 * r:.6e}  {0.02 * q:.6e}\n")
    return path


def test_the_unclaimed_default_is_what_we_are_overriding():
    """Guard for the tests below: without a declaration, FWHM and no angle."""
    from aure import instruments

    path = _unreadable_header_file("REFL_234277_3_234279_autoreduction.dat")
    try:
        meta = instruments.header_metadata(path)
        assert meta["dq_is_fwhm"] is True
        assert meta["theta"] == 0.0
    finally:
        os.unlink(path)


def test_declared_values_survive_enrichment():
    path = _unreadable_header_file("REFL_234277_3_234279_autoreduction.dat")
    try:
        state = create_initial_state(
            data_file=path,
            sample_description="",
            states=[
                {
                    "name": "state0",
                    "data_files": [
                        {
                            "file": path,
                            "label": "seg3",
                            "theta": 3.5,
                            "dq_is_fwhm": False,
                        }
                    ],
                }
            ],
        )

        result = intake_node(state)

        assert "error" not in result, result.get("error")
        ds = result["states"][0]["data_files"][0]
        assert ds["dq_is_fwhm"] is False
        assert ds["theta"] == 3.5
    finally:
        os.unlink(path)


def test_a_declared_convention_reaches_the_state_level_flag():
    """The single-file path reads only this; a per-file value alone is not enough.

    ``build_problem`` takes ``definition["dq_is_fwhm"]``, which
    ``modeling`` copies from the state. Before this, intake set it from a
    fresh header parse and a declaration on the primary file was silently
    outvoted for any single-file analysis.
    """
    path = _unreadable_header_file("REFL_234277_1_234277_autoreduction.dat")
    try:
        state = create_initial_state(
            data_file=path,
            sample_description="",
            states=[
                {
                    "name": "state0",
                    "data_files": [{"file": path, "dq_is_fwhm": False}],
                }
            ],
        )

        result = intake_node(state)

        assert result["dq_is_fwhm"] is False
    finally:
        os.unlink(path)


def test_an_undeclared_file_still_takes_the_parsed_convention():
    """No declaration must leave the previous behaviour untouched."""
    path = _unreadable_header_file("REFL_234277_1_234277_autoreduction.dat")
    try:
        state = create_initial_state(
            data_file=path,
            sample_description="",
            states=[{"name": "state0", "data_files": [{"file": path}]}],
        )

        result = intake_node(state)

        assert result["dq_is_fwhm"] is True
        assert result["states"][0]["data_files"][0]["theta"] == 0.0
    finally:
        os.unlink(path)


def test_each_file_keeps_its_own_declaration():
    """The three segments of one measurement each carry their own angle."""
    paths = [
        _unreadable_header_file(f"REFL_234277_{i}_23427{6 + i}_autoreduction.dat")
        for i in (1, 2, 3)
    ]
    try:
        state = create_initial_state(
            data_file=paths[0],
            sample_description="",
            states=[
                {
                    "name": "state0",
                    "data_files": [
                        {"file": p, "theta": t, "dq_is_fwhm": False}
                        for p, t in zip(paths, (0.45, 1.251, 3.5))
                    ],
                }
            ],
        )

        result = intake_node(state)

        assert "error" not in result, result.get("error")
        thetas = [ds["theta"] for ds in result["states"][0]["data_files"]]
        assert thetas == [0.45, 1.251, 3.5]
        assert all(not ds["dq_is_fwhm"] for ds in result["states"][0]["data_files"])
    finally:
        for p in paths:
            os.unlink(p)
