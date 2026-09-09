"""Loading an ORSO ``.ort`` file into a refl1d probe.

`tests/test_instruments.py` covers the instrument seam — which instrument
claims a file and what metadata it reads. That is all on the metadata side:
nothing there opens a probe, and `load_probe` had no test at all, which is how
the limitation below went unnoticed while `.ort` was advertised as supported.

**refl1d's loader accepts one shape.** `load4` reads an ORSO file only when it
carries the incident angle *and* the wavelength as tagged data columns, each
with a matching ``ErrorColumn``. Every other spec-valid shape raises
``AttributeError: 'NoneType' object has no attribute 'error_value'`` from
``refl1d/probe/data_loaders/load4.py:110``, which dereferences ``v.error``
without a guard when no matching resolution column was found.

The failing shapes are marked ``xfail(strict=True)`` deliberately: they
document the limitation where someone will meet it, and the day refl1d guards
that dereference the strict xfail turns into a failure, which is the only
notification this repository would otherwise get. When that happens, drop the
marks — do not add a workaround.

Note the file still gets *part* way through a run without this: intake and
analysis read data through ``data_tools.parse_ort_file``, a lenient 4-column
reader that succeeds on all four shapes. Only ``fitting`` calls ``load_probe``,
so a header-only ORSO file completes intake, analysis and modeling before
failing.
"""

from __future__ import annotations

import numpy as np
import pytest

N_POINTS = 40


def _write_ort(path, *, geometry: str, header_error: bool = False) -> str:
    """Write a spec-valid ORSO file with orsopy's own writer.

    Args:
        geometry: how the incident angle and wavelength are carried —
            ``"header"`` (stated once in ``instrument_settings``, orsopy's
            default and the natural form for a reduced R(Q) curve),
            ``"columns"`` (also repeated per point as tagged columns), or
            ``"columns+error"`` (each tagged column with its ``ErrorColumn``).
        header_error: attach an explicit uncertainty to the header values. The
            standard permits this and orsopy does not write one by default.
    """
    from orsopy import fileio
    from orsopy.fileio.base import ErrorValue

    angle = fileio.Value(0.6, "deg")
    wavelength = fileio.ValueRange(2.5, 16.5, "angstrom")
    if header_error:
        angle.error = ErrorValue(0.01, "resolution", "sigma")
        wavelength.error = ErrorValue(0.1, "resolution", "sigma")

    columns = [
        fileio.Column(
            "Qz", "1/angstrom", physical_quantity="normal_wavevector_transfer"
        ),
        fileio.Column("R", physical_quantity="reflectivity"),
        fileio.ErrorColumn("R", "uncertainty", "sigma"),
        fileio.ErrorColumn("Qz", "resolution", "sigma"),
    ]
    data = [
        np.linspace(0.01, 0.10, N_POINTS),
        np.full(N_POINTS, 1e-3),
        np.full(N_POINTS, 5e-5),
        np.full(N_POINTS, 2e-4),
    ]
    if geometry != "header":
        columns.append(
            fileio.Column("Theta", "deg", physical_quantity="incident_angle")
        )
        data.append(np.full(N_POINTS, 0.6))
        if geometry == "columns+error":
            columns.append(fileio.ErrorColumn("Theta", "resolution", "sigma"))
            data.append(np.full(N_POINTS, 0.01))
        columns.append(
            fileio.Column("Lambda", "angstrom", physical_quantity="wavelength")
        )
        data.append(np.full(N_POINTS, 5.0))
        if geometry == "columns+error":
            columns.append(fileio.ErrorColumn("Lambda", "resolution", "sigma"))
            data.append(np.full(N_POINTS, 0.1))

    info = fileio.Orso(
        data_source=fileio.DataSource(
            owner=fileio.Person(name="Beamline Scientist", affiliation="ORNL"),
            experiment=fileio.Experiment(
                title="orso probe fixture",
                instrument="REF_L",
                start_date="2026-09-01",
                probe="neutron",
            ),
            sample=fileio.Sample(name="reference sample"),
            measurement=fileio.Measurement(
                instrument_settings=fileio.InstrumentSettings(
                    incident_angle=angle, wavelength=wavelength
                ),
                data_files=["REFL_12345_combined.nxs"],
            ),
        ),
        reduction=fileio.Reduction(software=fileio.Software("mantid")),
        columns=columns,
    )
    out = str(path)
    fileio.save_orso([fileio.OrsoDataset(info, np.column_stack(data))], out)
    return out


@pytest.mark.parametrize(
    "geometry,header_error",
    [("header", False), ("header", True), ("columns", False)],
    ids=["header-only", "header-only-with-error", "value-columns-no-error"],
)
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="refl1d 1.0.1 load4.py:110 dereferences v.error unguarded when no "
    "matching resolution COLUMN was found. Reported upstream with a "
    "reproduction. When this xpasses, remove the mark — do not work around it.",
)
def test_load_probe_reads_every_spec_valid_orso_shape(tmp_path, geometry, header_error):
    """All three of these are written and read back by orsopy without complaint.

    The second fails differently from the other two — `ValueRange.error` (what
    `wavelength` normally is) round-trips as an untyped dict rather than an
    `ErrorValue`, so the same line raises `'dict' object has no attribute
    'error_value'`. `raises=AttributeError` covers both.
    """
    from aure.nodes.model_builder import load_probe

    path = _write_ort(
        tmp_path / f"{geometry}.ort", geometry=geometry, header_error=header_error
    )
    probe = load_probe(path)
    assert len(probe.Q) == N_POINTS


def test_load_probe_reads_an_orso_file_with_geometry_error_columns(tmp_path):
    """The one shape refl1d accepts, and the reason `.ort` support is not simply
    broken: a TOF reduction that writes Theta/sTheta and Lambda/sLambda takes
    this path. It must keep working.
    """
    from aure.nodes.model_builder import load_probe

    path = _write_ort(tmp_path / "ok.ort", geometry="columns+error")
    probe = load_probe(path)
    assert len(probe.Q) == N_POINTS
    # The sQz column is one sigma by specification and reaches the probe as-is.
    assert float(probe.dQ[0]) == pytest.approx(2e-4)


def test_the_lenient_reader_accepts_what_the_probe_loader_rejects(tmp_path):
    """Why a header-only file gets three nodes into a run before failing:
    intake and analysis read through `parse_ort_file`, which does not care."""
    from aure.tools.data_tools import parse_ort_file

    path = _write_ort(tmp_path / "header.ort", geometry="header")
    parsed = parse_ort_file(path)
    assert len(parsed["Q"]) == N_POINTS
