"""Tests for intake_node multi-state behaviour (Ticket 05)."""

from __future__ import annotations

import os
import tempfile

import numpy as np


def _make_data_file(name_prefix: str, q_min=0.01, q_max=0.10, n=60) -> str:
    """Create a synthetic data file with a chosen filename prefix.

    The basename is chosen so that ``_extract_set_id`` can decode a
    REF_L set id (e.g. ``REFL_226642_combined_data_auto.txt``).
    """
    Q = np.linspace(q_min, q_max, n)
    R = np.clip((0.0217 / (2 * np.maximum(Q, 0.001))) ** 4, 1e-10, 1.0)
    dR = 0.05 * R
    dQ = 0.02 * Q
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, name_prefix)
    with open(path, "w") as f:
        f.write("# Q R dR dQ\n")
        for q, r, dr, dq in zip(Q, R, dR, dQ):
            f.write(f"{q:.6f}  {r:.6e}  {dr:.6e}  {dq:.6e}\n")
    return path


def _ds(file_path: str, label: str | None = None) -> dict:
    return {"file": file_path, "label": label or os.path.basename(file_path)}


# ----------------------------------------------------------------------
# _extract_set_id
# ----------------------------------------------------------------------


def test_extract_set_id_combined():
    from aure.nodes.intake import _extract_set_id

    assert _extract_set_id("/tmp/REFL_226642_combined_data_auto.txt") == "226642"


def test_extract_set_id_partial():
    from aure.nodes.intake import _extract_set_id

    assert _extract_set_id("/tmp/REFL_226642_3_2003_partial.txt") == "226642"


def test_extract_set_id_unknown():
    from aure.nodes.intake import _extract_set_id

    assert _extract_set_id("/tmp/random.dat") is None


# ----------------------------------------------------------------------
# intake_node multi-state branches
# ----------------------------------------------------------------------


def test_intake_multi_state_enriches_each_state():
    """Two states, each with one combined file: data_files should be enriched."""
    from aure.nodes.intake import intake_node
    from aure.state import create_initial_state

    f1 = _make_data_file("REFL_226642_combined_data_auto.txt")
    f2 = _make_data_file("REFL_226660_combined_data_auto.txt")
    try:
        states = [
            {"name": "D2O", "data_files": [_ds(f1, "D2O")]},
            {"name": "H2O", "data_files": [_ds(f2, "H2O")]},
        ]
        state = create_initial_state(
            data_file=f1,
            sample_description="",
            states=states,
        )
        result = intake_node(state)

        assert "error" not in result, result.get("error")
        assert "states" in result
        assert len(result["states"]) == 2
        # Each state's data_files have Q/R/dR + theta/dq_is_fwhm enriched
        for st in result["states"]:
            for ds in st["data_files"]:
                assert "Q" in ds and len(ds["Q"]) > 0
                assert "R" in ds
                assert "dR" in ds
                assert "dq_is_fwhm" in ds
                assert "theta" in ds
        # Flat data_files mirrors all enriched files (2 total)
        assert len(result["data_files"]) == 2
    finally:
        os.unlink(f1)
        os.unlink(f2)


def test_intake_flat_partials_same_set_id_passes():
    """Legacy partials path (same set_id) is unaffected."""
    from aure.nodes.intake import intake_node
    from aure.state import create_initial_state

    f1 = _make_data_file("REFL_226642_1_2001_partial.txt")
    f2 = _make_data_file("REFL_226642_2_2002_partial.txt")
    try:
        state = create_initial_state(
            data_file=f1,
            sample_description="",
            data_files=[_ds(f1, "lo"), _ds(f2, "hi")],
        )
        result = intake_node(state)
        assert "error" not in result, result.get("error")
        assert len(result["data_files"]) == 2
    finally:
        os.unlink(f1)
        os.unlink(f2)


def test_intake_single_file_passes():
    """Bare single-file invocation continues to work."""
    from aure.nodes.intake import intake_node
    from aure.state import create_initial_state

    f = _make_data_file("REFL_226642_combined_data_auto.txt")
    try:
        state = create_initial_state(data_file=f, sample_description="")
        result = intake_node(state)
        assert "error" not in result, result.get("error")
        # No states populated and no flat data_files (single-file path)
        assert not result.get("states")
    finally:
        os.unlink(f)


def test_intake_multi_state_partials_mixed_set_ids_errors():
    """Within one state, partial files must share a set_id."""
    from aure.nodes.intake import intake_node
    from aure.state import create_initial_state

    f1 = _make_data_file("REFL_111_1_1_partial.txt")
    f2 = _make_data_file("REFL_222_1_1_partial.txt")
    f3 = _make_data_file("REFL_333_combined_data_auto.txt")
    try:
        states = [
            {"name": "bad", "data_files": [_ds(f1), _ds(f2)]},
            {"name": "good", "data_files": [_ds(f3)]},
        ]
        state = create_initial_state(
            data_file=f1,
            sample_description="",
            states=states,
        )
        result = intake_node(state)
        assert "error" in result
        assert "set_ids" in result["error"] or "set ids" in result["error"].lower()
    finally:
        os.unlink(f1)
        os.unlink(f2)
        os.unlink(f3)
