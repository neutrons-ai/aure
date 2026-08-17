"""The SLD sequence the artifact detector judges a profile against.

`_ordered_slds_for_artifacts` used to emit ``[ambient, layers..., substrate]``.
A ModelDefinition lists its layers substrate-first, so that put both terminal
media on the wrong ends of an otherwise substrate-first sequence — an order
that describes no stack in either geometry. The interior is untouched by the
mistake, but the substrate and the ambient are exactly the two neighbours that
decide whether the first and last layers count as turning points, so the
detector expected the wrong extrema and reported the real ones as excursions.

On the 2026-08-17 cu_film sweep that vetoed every candidate in 38 of 51 runs,
and `finalize` ranks by the veto — so it chose which model each run reported.
"""

import numpy as np

from aure.nodes.evaluation import _ordered_slds_for_artifacts
from aure.tools.feature_tools import detect_profile_artifacts

# The Cu_0/201211 stack as AuRE modelled it: substrate-first layers, as every
# ModelDefinition carries them (`prompts`: "Layers are listed from substrate to
# ambient"), whatever the beam geometry.
MODEL = {
    "back_reflection": True,
    "substrate": {"name": "silicon", "sld": 2.07},
    "layers": [
        {"name": "silicon native oxide (SiO2)", "sld": 2.807},
        {"name": "titanium", "sld": -2.489},
        {"name": "copper", "sld": 6.56},
        {"name": "copper oxide", "sld": 4.725},
    ],
    "ambient": {"name": "dTHF", "sld": 6.187},
}


def test_the_sequence_is_the_physical_stack_order():
    """substrate, then layers as listed, then ambient — the adjacency `modeling`
    builds in both geometries (back reflection writes the same stack in beam
    order, layers reversed along with the terminals)."""
    seq = _ordered_slds_for_artifacts(MODEL, {})

    assert seq == [2.07, 2.807, -2.489, 6.56, 4.725, 6.187]
    assert seq[0] == MODEL["substrate"]["sld"]
    assert seq[-1] == MODEL["ambient"]["sld"]


def test_fitted_values_win_over_seeds_without_disturbing_the_order():
    seq = _ordered_slds_for_artifacts(MODEL, {"copper rho": 6.44, "dTHF rho": 6.20})

    assert seq == [2.07, 2.807, -2.489, 6.44, 4.725, 6.20]


def test_the_terminals_are_not_interchangeable():
    """The bug this pins: swapping the two ends leaves the interior identical
    and still changes the verdict, because it changes which interior media are
    turning points."""
    correct = _ordered_slds_for_artifacts(MODEL, {})
    swapped = [correct[-1]] + correct[1:-1] + [correct[0]]

    from aure.tools.feature_tools import _turning_point_slds

    assert sorted(_turning_point_slds(correct)) != sorted(_turning_point_slds(swapped))


def test_the_detector_does_not_care_which_way_the_profile_runs():
    """Why the mismatch between the two conventions is safe *here*.

    refl1d renders the profile in beam order — ambient-first under back
    reflection — while the sequence above is substrate-first. Nothing flips
    either one, and nothing needs to: the detector compares extremum values
    against turning-point values, and both sets survive reversal. Any future
    consumer that reads position rather than value does need to orient first.
    """
    from scipy.special import erf

    seq = _ordered_slds_for_artifacts(MODEL, {})
    z = np.linspace(-100.0, 700.0, 2000)
    rho = np.full_like(z, seq[0])
    prev = seq[0]
    for boundary, value in zip(np.linspace(0.0, 600.0, len(seq) - 1), seq[1:]):
        rho = rho + 0.5 * (value - prev) * (1 + erf((z - boundary) / (8.0 * np.sqrt(2))))
        prev = value

    forward = detect_profile_artifacts(z, rho, seq)
    reversed_profile = detect_profile_artifacts(z, rho[::-1], seq)
    reversed_media = detect_profile_artifacts(z, rho, seq[::-1])

    assert forward["has_artifact"] is False
    assert reversed_profile["has_artifact"] == forward["has_artifact"]
    assert reversed_media["has_artifact"] == forward["has_artifact"]
