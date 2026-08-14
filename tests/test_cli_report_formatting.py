"""Tests that the end-of-run report survives fields the workflow left unset.

``aure analyze`` prints its summary *after* the fit has converged and every
artifact — ``problem.json``, ``final_state.json``, the checkpoint trail — is
already on disk. So an exception raised while formatting that summary destroys
nothing but the report, yet it still propagates out of the click command and
turns a successful run into exit status 1. Any orchestrator reading that exit
code (``ndip-run``'s AuRE backend, for one) then records the analysis as failed
and discards a fit that actually succeeded.

The values in the summary are not all guaranteed to be present. A sample
description that never pins down a layer's thickness reaches ``parsed_sample``
with ``thickness: None``, and ``f"{None:.0f}"`` raises TypeError. This is a
recurring shape of bug in this function rather than a one-off — the parameter
block carries its own scar tissue for an explicit ``uncertainties=None``.

Hence :func:`_fmt_num`: formatting a missing number yields ``"n/a"`` and the
report keeps going, so a gap in the model description costs a line of output
instead of the whole run.
"""

import pytest

from aure.cli import _fmt_num, _print_analysis_results


# --- _fmt_num ---


@pytest.mark.parametrize(
    "value,expected",
    [
        (12.345, "12.35"),
        (0, "0.00"),
        (-1.95, "-1.95"),
    ],
)
def test_fmt_num_formats_real_numbers(value, expected):
    assert _fmt_num(value) == expected


def test_fmt_num_honors_the_format_spec():
    assert _fmt_num(29.6955, ".0f") == "30"


@pytest.mark.parametrize(
    "value",
    [
        None,             # the case that took down a converged run
        float("nan"),
        float("inf"),
        float("-inf"),
        "50 nm",          # an unparsed description that leaked through
        True,             # a bool is an int subclass but never a thickness
    ],
)
def test_fmt_num_returns_na_for_anything_unformattable(value):
    assert _fmt_num(value) == "n/a"


# --- the report itself ---


def _result_with_thickness(thickness):
    """A minimal converged-run result carrying one layer of *thickness*."""
    return {
        "Q": [0.01, 0.02, 0.03],
        "parsed_sample": {
            "substrate": {"name": "Si", "sld": 2.07},
            "layers": [
                {"name": "Cu", "sld": 6.55, "thickness": 500},
                {"name": "CuOx", "sld": 5.5, "thickness": thickness},
            ],
            "ambient": {"name": "air", "sld": 0.0},
        },
        "current_chi2": 1.493,
    }


def test_report_survives_a_layer_with_no_thickness(capsys):
    """The regression: an unspecified thickness must not abort the summary."""
    _print_analysis_results(_result_with_thickness(None))

    out = capsys.readouterr().out
    assert "CuOx" in out
    assert "n/a Å" in out
    # Everything downstream of the offending layer still has to print — that is
    # the whole point, since the fit results are what the run was for.
    assert "Ambient: air" in out
    assert "1.4930" in out


def test_report_prints_a_known_thickness_normally(capsys):
    """The guard must not degrade the output when the value is present."""
    _print_analysis_results(_result_with_thickness(30))

    out = capsys.readouterr().out
    assert "30 Å" in out
    assert "n/a" not in out


def test_report_survives_a_missing_substrate_and_ambient(capsys):
    """`parsed_sample` may be partially populated, not just partially valued."""
    _print_analysis_results({"Q": [0.01], "parsed_sample": {"layers": []}})

    out = capsys.readouterr().out
    assert "Substrate: unknown" in out
    assert "Ambient: unknown" in out
