"""A built parameter must start inside its own range, and be uniquely named.

Both invariants failed silently before: an out-of-range starting value makes
the whole problem infeasible (bumps short-circuits without evaluating the
model, and the fit reports `inf` before it starts), and two parameters sharing
a name collapse wherever results are keyed by name. Neither shows up as
anything wrong in the model JSON.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest

from aure.nodes.model_builder import build_problem, build_states_problem

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def data_file():
    Q = np.linspace(0.01, 0.10, 60)
    R = np.clip((0.0217 / (2 * Q)) ** 4, 1e-10, 1.0)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for q, r in zip(Q, R):
        f.write(f"{q:.6f} {r:.6e} {0.05 * r:.6e} {0.02 * q:.6e}\n")
    f.close()
    yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


def _defn(data_file: str, layer: dict, **extra) -> dict:
    d = {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "air", "sld": 0.0},
        "layers": [layer],
        "data_file": data_file,
        "intensity": {"fixed": True},
    }
    d.update(extra)
    return d


def _param(problem, name: str):
    return next(p for p in problem._parameters if str(p.name) == name)


def _feasible(problem) -> bool:
    return not problem._nllf_components()[3]


# ----------------------------------------------------------------------
# The built-in roughness floor must not contradict a declared roughness
# ----------------------------------------------------------------------


def test_declared_roughness_below_the_default_floor_is_honoured(data_file):
    """A chemically sharp buried interface — "do not impose a roughness floor
    on it" — used to be built at 3 Å with bounds (5, 20): outside its own
    range, so the problem was infeasible before the optimizer ran."""
    problem = build_problem(
        _defn(data_file, {"name": "SiO2", "sld": 2.9, "thickness": 25.0,
                          "roughness": 3.0, "roughness_max": 20.0})
    )
    par = _param(problem, "SiO2 interface")
    assert par.value == pytest.approx(3.0)
    assert par.prior.limits[0] <= 3.0
    assert _feasible(problem)


def test_the_floor_still_applies_when_it_does_not_conflict(data_file):
    problem = build_problem(
        _defn(data_file, {"name": "film", "sld": 2.0, "thickness": 100.0,
                          "roughness": 8.0, "roughness_max": 30.0})
    )
    assert _param(problem, "film interface").prior.limits[0] == pytest.approx(5.0)


def test_an_explicit_roughness_min_is_obeyed(data_file):
    problem = build_problem(
        _defn(data_file, {"name": "film", "sld": 2.0, "thickness": 100.0,
                          "roughness": 8.0, "roughness_min": 0.0,
                          "roughness_max": 30.0})
    )
    assert _param(problem, "film interface").prior.limits[0] == pytest.approx(0.0)


def test_an_explicit_bound_excluding_the_start_clamps_and_stays_feasible(data_file):
    """The model contradicting itself: clamp into the range and log, rather
    than hand the optimizer a point it will refuse to evaluate."""
    problem = build_problem(
        _defn(data_file, {"name": "film", "sld": 2.0, "thickness": 100.0,
                          "roughness": 2.0, "roughness_min": 6.0,
                          "roughness_max": 30.0})
    )
    par = _param(problem, "film interface")
    assert par.value == pytest.approx(6.0)
    assert _feasible(problem)


@pytest.mark.parametrize(
    "layer",
    [
        {"name": "f", "sld": 2.0, "thickness": 100.0, "thickness_min": 150.0,
         "thickness_max": 400.0, "roughness": 5.0},
        {"name": "f", "sld": 2.0, "thickness": 100.0, "sld_min": 3.0,
         "sld_max": 6.0, "roughness": 5.0},
    ],
    ids=["thickness below its min", "sld below its min"],
)
def test_no_parameter_is_ever_built_outside_its_range(data_file, layer):
    problem = build_problem(_defn(data_file, layer))
    for par in problem._parameters:
        lo, hi = par.prior.limits
        assert lo <= par.value <= hi, f"{par.name} = {par.value} outside ({lo}, {hi})"
    assert _feasible(problem)


def test_negative_ambient_sld_does_not_invert_its_range(data_file):
    """H2O (-0.56) hits the multiplicative defaults backwards: min=-0.448,
    max=-0.672. The defaults are left alone (see the TODO at the site) but the
    range must not reach refl1d inverted."""
    problem = build_problem(
        _defn(data_file,
              {"name": "film", "sld": 2.0, "thickness": 100.0, "roughness": 5.0},
              ambient={"name": "H2O", "sld": -0.56})
    )
    par = _param(problem, "H2O rho")
    lo, hi = par.prior.limits
    assert lo < hi
    assert lo <= par.value <= hi


# ----------------------------------------------------------------------
# Every free parameter needs a unique name
# ----------------------------------------------------------------------


def test_back_reflection_states_do_not_share_an_ambient_interface_name(data_file):
    """In back reflection the ambient carries the outer interface, so it is a
    fitted parameter — one per state, and all called "<ambient> interface"
    until they are prefixed."""
    defn = {
        "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0,
                      "roughness_max": 15.0},
        "ambient": {"name": "dTHF", "sld": 6.2},
        "back_reflection": True,
        "layers": [{"name": "Cu", "sld": 6.55, "thickness": 500.0,
                    "roughness": 8.0, "roughness_max": 25.0}],
        "states": [
            {"name": "a", "back_reflection": True,
             "data_files": [{"file": data_file, "label": "a"}]},
            {"name": "b", "back_reflection": True,
             "data_files": [{"file": data_file, "label": "b"}]},
        ],
    }
    problem, _exps, _ = build_states_problem(defn)
    names = [str(p.name) for p in problem._parameters]
    dupes = [n for n in names if names.count(n) > 1]
    assert len(names) == len(set(names)), f"duplicates: {dupes}"
    assert {"a dTHF interface", "b dTHF interface"} <= set(names)


# ---------------------------------------------------------------------------
# An unknown SLD must not be fenced into one half of a bimodal distribution
# ---------------------------------------------------------------------------


def test_an_unparsed_sld_gets_bounds_spanning_both_hd_clusters():
    """A layer whose SLD nothing supplied must be free to reach either cluster.

    Neutron SLDs are bimodal — protiated organics near 0.4, deuterated near
    5.5, nothing in between. The seed lands mid-gap by necessity, so applying
    the usual ±2.5 window to it produced (-0.5, 4.5) and excluded every
    deuterated material. The bound, not the data, then decided what the layer
    was made of.
    """
    from aure.nodes.modeling import _build_layers

    layer = _build_layers({"layers": [{"name": "unknown", "thickness": 100.0}]}, {})[0]
    assert layer["sld_min"] <= -0.5, layer
    assert layer["sld_max"] >= 7.0, layer
    # Both clusters reachable: a protiated organic and its deuterated form.
    assert layer["sld_min"] < 0.4 < layer["sld_max"]
    assert layer["sld_min"] < 6.4 < layer["sld_max"]


def test_a_parsed_sld_keeps_the_narrow_window():
    """The wide span is for ignorance only; a stated SLD still gets ±2.5."""
    from aure.nodes.modeling import _build_layers

    layer = _build_layers({"layers": [{"name": "dPS", "sld": 6.4}]}, {})[0]
    assert layer["sld_min"] == pytest.approx(3.9)
    assert layer["sld_max"] == pytest.approx(8.9)


def test_feature_estimated_layers_span_both_clusters_too():
    """Fringe counting says how many layers and how thick, not what they are."""
    from aure.nodes.modeling import _build_layers

    layer = _build_layers(
        {}, {"estimated_n_layers": 1, "estimated_total_thickness": 200.0}
    )[0]
    assert layer["sld_min"] < 0.4 < layer["sld_max"]
    assert layer["sld_min"] < 6.4 < layer["sld_max"]


def test_the_default_substrate_needs_no_materials_database():
    """`_get_substrate` used to import `aure.database` to look up a constant."""
    import aure.nodes.modeling as modeling
    from aure.nodes.modeling import _get_substrate

    assert not hasattr(modeling, "get_sld")
    assert _get_substrate({}, {})["sld"] == pytest.approx(2.07, abs=0.01)


# ---------------------------------------------------------------------------
# The 5 Å roughness floor is a default, not an assertion
# ---------------------------------------------------------------------------


def test_build_layers_does_not_inject_a_roughness_floor():
    """`_build_layers` used to write `roughness_min: 5.0` into every layer.

    That turned the builder's *default* floor into a *declared* bound, and
    `_ranged` treats the two differently on purpose: a default yields to a
    smaller declared roughness, a declared bound clamps it. So a parse that
    said 2 Å was silently refitted from 5 Å with (5, 30) bounds, and no
    iteration could get below the floor.
    """
    from aure.nodes.modeling import _build_layers

    parsed = {"name": "SiO2", "sld": 3.47, "thickness": 15.0, "roughness": 2.0}
    layer = _build_layers({"layers": [parsed]}, {})[0]
    assert "roughness_min" not in layer, layer
    assert layer["roughness"] == pytest.approx(2.0)


def test_feature_estimated_layers_do_not_inject_a_floor_either():
    from aure.nodes.modeling import _build_layers

    layer = _build_layers(
        {}, {"estimated_n_layers": 1, "estimated_total_thickness": 200.0}
    )[0]
    assert "roughness_min" not in layer, layer


def test_a_sharp_parsed_interface_survives_into_the_fit(data_file):
    """The default floor must yield to it, start value and bound together."""
    from aure.nodes.modeling import _build_layers

    parsed = {"name": "SiO2", "sld": 3.47, "thickness": 15.0, "roughness": 2.0}
    layer = _build_layers({"layers": [parsed]}, {})[0]
    problem = build_problem(_defn(data_file, layer))
    par = _param(problem, "SiO2 interface")
    assert par.value == pytest.approx(2.0)
    assert par.prior.limits[0] == pytest.approx(2.0)


def test_the_floor_still_binds_an_ordinary_interface(data_file):
    """Dropping the hardcode must not drop the floor where it does not conflict."""
    from aure.nodes.modeling import _build_layers

    parsed = {"name": "film", "sld": 3.47, "thickness": 100.0, "roughness": 8.0}
    layer = _build_layers({"layers": [parsed]}, {})[0]
    problem = build_problem(_defn(data_file, layer))
    assert _param(problem, "film interface").prior.limits[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# The ambient: bounded SLD, and no roughness of its own
# ---------------------------------------------------------------------------


#: One ordinary layer, so these tests can vary only the ambient.
_CU_LAYER = {
    "name": "Cu",
    "sld": 6.5,
    "thickness": 500.0,
    "roughness": 7.0,
    "roughness_max": 25.0,
}


def test_ambient_sld_bounds_are_honoured(data_file):
    """`AmbientInfo` did not declare these, but `_build_sample` has always read
    them — the ambient SLD is a fitted parameter whenever the ambient is not
    air and its SLD is non-zero."""
    defn = _defn(data_file, dict(_CU_LAYER))
    defn["ambient"] = {"name": "dTHF", "sld": 6.2, "sld_min": 5.8, "sld_max": 6.5}
    problem = build_problem(defn)
    par = _param(problem, "dTHF rho")
    assert par.prior.limits[0] == pytest.approx(5.8)
    assert par.prior.limits[1] == pytest.approx(6.5)


@pytest.mark.parametrize("back", [False, True], ids=["normal", "back_reflection"])
def test_a_roughness_on_the_ambient_warns_rather_than_vanishing(
    data_file, back, caplog
):
    """Nothing reads `ambient["roughness"]` in either geometry: an interface
    belongs to the slab below it, so the outer surface is the outermost
    layer's. A declaration that disappears without comment is the worst option.
    """
    defn = _defn(data_file, dict(_CU_LAYER))
    defn["ambient"] = {"name": "dTHF", "sld": 6.2, "roughness": 22.0}
    defn["back_reflection"] = back
    with caplog.at_level("WARNING"):
        build_problem(defn)
    messages = [r.getMessage() for r in caplog.records]
    assert any("declares roughness" in m for m in messages), messages
    # It must name the layer that actually owns the outer surface.
    assert any("Cu" in m for m in messages), messages


def test_a_clean_ambient_warns_about_nothing(data_file, caplog):
    defn = _defn(data_file, dict(_CU_LAYER))
    defn["ambient"] = {"name": "dTHF", "sld": 6.2, "sld_min": 5.8, "sld_max": 6.5}
    with caplog.at_level("WARNING"):
        build_problem(defn)
    assert not [r for r in caplog.records if "ambient" in r.getMessage()]
