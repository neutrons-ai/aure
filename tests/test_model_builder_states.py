"""Tests for build_states_problem (cross-state co-refinement)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _make_data_file(q_min: float = 0.01, q_max: float = 0.10, n: int = 80) -> str:
    Q = np.linspace(q_min, q_max, n)
    Qc = 0.0217
    R = np.where(Q < Qc, 1.0, (Qc / (2 * Q)) ** 4)
    R *= 1 + 0.2 * np.cos(2 * Q * 100.0)
    R = np.clip(R, 1e-10, 1.0)
    dR = 0.05 * R
    dQ = 0.02 * Q
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for q, r, dr, dq in zip(Q, R, dR, dQ):
        f.write(f"{q:.6f}  {r:.6e}  {dr:.6e}  {dq:.6e}\n")
    f.close()
    return f.name


@pytest.fixture
def two_files():
    paths = [_make_data_file(0.01, 0.10, 60), _make_data_file(0.01, 0.10, 60)]
    yield paths
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _two_state_definition(files: list[str], **extra) -> dict:
    base = {
        "substrate": {
            "name": "silicon",
            "sld": 2.07,
            "roughness": 3.0,
            "roughness_max": 15.0,
        },
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "sld_min": 4.0,
                "sld_max": 8.0,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            },
            {
                "name": "Ti",
                "sld": -1.95,
                "sld_min": -3.0,
                "sld_max": 0.0,
                "thickness": 30.0,
                "thickness_min": 10.0,
                "thickness_max": 60.0,
                "roughness": 3.0,
                "roughness_max": 15.0,
            },
        ],
        "ambient": {"name": "air", "sld": 0.0},
        "back_reflection": False,
        "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        "states": [
            {
                "name": "D2O",
                "data_files": [{"file": files[0], "label": "D2O-comb"}],
                "ambient": {"name": "D2O", "sld": 6.4, "sld_min": 5.0, "sld_max": 6.6},
            },
            {
                "name": "H2O",
                "data_files": [{"file": files[1], "label": "H2O-comb"}],
                "ambient": {
                    "name": "H2O",
                    "sld": -0.56,
                    "sld_min": -0.6,
                    "sld_max": 0.0,
                },
            },
        ],
    }
    base.update(extra)
    return base


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_default_tied_set_aliases_structural_params(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    problem, by_state, by_files = build_states_problem(defn)

    assert set(by_state.keys()) == {"D2O", "H2O"}
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample
    # samples themselves are distinct objects (different ambients)
    assert s0 is not s1
    # but the Cu thickness Parameter object is shared
    assert s1[1].thickness is s0[1].thickness
    assert s1[1].material.rho is s0[1].material.rho
    assert s1[1].interface is s0[1].interface
    # Ti
    assert s1[2].thickness is s0[2].thickness
    assert s1[2].material.rho is s0[2].material.rho
    # substrate.interface tied
    assert s1[0].interface is s0[0].interface
    # ambient SLD NOT tied (different solvents)
    assert s1[3].material.rho is not s0[3].material.rho


def test_shared_parameters_whitelist_replaces_default(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, shared_parameters=["Cu.thickness"])
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # Cu.thickness tied
    assert s1[1].thickness is s0[1].thickness
    # everything else NOT tied
    assert s1[1].material.rho is not s0[1].material.rho
    assert s1[1].interface is not s0[1].interface
    assert s1[2].thickness is not s0[2].thickness


def test_unshared_parameters_subtracts_from_default(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, unshared_parameters=["Cu.thickness"])
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # Cu.thickness explicitly not tied
    assert s1[1].thickness is not s0[1].thickness
    # but Cu.material.rho still tied (default)
    assert s1[1].material.rho is s0[1].material.rho


def test_unshared_substrate_alias_actually_unties(two_files):
    """`substrate.<attr>` in unshared_parameters must match the default tie.

    Regression: the default substrate tie is keyed by ``sub_name`` (e.g.
    ``"silicon"``), but users / docs spell it ``"substrate"``. Before
    normalization, the blacklist was silently ignored for the substrate.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, unshared_parameters=["substrate.interface"])
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # substrate interface explicitly NOT tied
    assert s1[0].interface is not s0[0].interface
    # but Cu.thickness still tied (default)
    assert s1[1].thickness is s0[1].thickness


def test_shared_substrate_alias_ties_and_keeps_canonical_name(two_files):
    """`substrate.<attr>` in shared_parameters ties and uses the canonical name.

    Regression: the rename pass keyed off ``sub_name``; with the alias
    spelling, the shared parameter was relabeled with a state prefix
    even though it was effectively shared.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, shared_parameters=["substrate.interface"])
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # substrate.interface tied via alias spelling
    assert s1[0].interface is s0[0].interface
    # and the shared parameter keeps the tied (state-unprefixed) name
    assert s0[0].interface.name == "silicon interface"


def test_unknown_layer_in_shared_parameters_raises(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, shared_parameters=["Aluminum.thickness"])
    with pytest.raises(ValueError, match="Aluminum"):
        build_states_problem(defn)


def test_shared_and_unshared_mutually_exclusive(two_files):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(
        two_files,
        shared_parameters=["Cu.thickness"],
        unshared_parameters=["Cu.interface"],
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_states_problem(defn)


def test_returns_fit_problem_with_all_experiments(two_files):
    from aure.nodes.model_builder import build_states_problem
    from bumps.fitproblem import FitProblem

    defn = _two_state_definition(two_files)
    problem, by_state, _ = build_states_problem(defn)
    assert isinstance(problem, FitProblem)
    n = sum(len(v) for v in by_state.values())
    assert n == 2


def test_single_state_via_build_states_problem(two_files):
    from aure.nodes.model_builder import build_states_problem
    from bumps.fitproblem import FitProblem

    defn = _two_state_definition(two_files)
    defn["states"] = defn["states"][:1]
    problem, by_state, _ = build_states_problem(defn)
    assert isinstance(problem, FitProblem)
    assert list(by_state.keys()) == ["D2O"]


def test_save_problem_json_routes_multi_state(tmp_path, two_files):
    from aure.nodes.model_builder import save_problem_json

    defn = _two_state_definition(two_files)
    out = save_problem_json(defn, tmp_path / "problem.json")
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_save_problem_json_round_trip_with_bumps(tmp_path, two_files):
    from aure.nodes.model_builder import save_problem_json
    from bumps.serialize import load_file

    defn = _two_state_definition(two_files)
    out = save_problem_json(defn, tmp_path / "problem.json")
    loaded = load_file(out)
    assert loaded is not None


def test_ui_ambient_rho_override_merges_into_model_ambient(two_files):
    """A partial ambient override from the UI (``{rho: X}``) must merge
    into the model-level ambient and translate ``rho`` → ``sld``.

    Regression: ``_state_overrides`` used to replace ``eff["ambient"]``
    wholesale with the UI payload, leaving the resulting dict without
    ``name``/``sld``. ``_build_sample`` then raised ``KeyError`` reading
    ``ambient_info["name"]``.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    # Simulate the UI shape: only `rho`, no `name`/`sld`.
    defn["states"][0]["ambient"] = {"rho": 6.4}
    defn["states"][1]["ambient"] = {"rho": -0.56}

    # Should build without KeyError.
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample
    s1 = by_state["H2O"][0].sample

    # State-0 ambient SLD should reflect the override (6.4); state-1 (-0.56).
    # Ambient slot is sample[n+1] in normal orientation (n=2 layers).
    assert abs(float(s0[3].material.rho.value) - 6.4) < 1e-9
    assert abs(float(s1[3].material.rho.value) - (-0.56)) < 1e-9


def test_mixed_back_reflection_orientations_rejected(two_files):
    """Mixed back-reflection orientations across states are rejected.

    Refl1d ranges substrate.interface on sample[0] in the normal stack
    and on sample[n+1] in the back-reflection stack, so cross-state
    aliasing silently drops the range on one side. Reject up front.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["states"][1]["back_reflection"] = True
    with pytest.raises(ValueError, match="back_reflection"):
        build_states_problem(defn)


def test_back_reflection_substrate_interface_indexed_correctly(two_files):
    """Substrate interface must alias to sample[n+1] in back-reflection.

    Regression: ``_layer_index`` returned ``n`` for substrate in
    back-reflection geometry, colliding with the topmost layer
    (``sample[n - 0] = sample[n]``). The default tie ``(sub_name,
    "interface")`` would then silently alias the wrong layer's
    interface across states.

    Stack ordering for back_reflection with layers=[Cu, Ti] (n=2):
      sample[0]=ambient, sample[1]=Ti, sample[2]=Cu, sample[3]=substrate.
    """
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files, back_reflection=True)
    _, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample  # back: [amb, Ti, Cu, sub]
    s1 = by_state["H2O"][0].sample  # back: [amb, Ti, Cu, sub]

    # Substrate.interface tied: both at sample[3]
    assert s1[3].interface is s0[3].interface
    # And it must NOT have been incorrectly aliased onto Cu (sample[2]).
    assert s1[2].interface is not s0[3].interface
    # Cu.interface tied at sample[2]
    assert s1[2].interface is s0[2].interface
    # Ti.interface tied at sample[1]
    assert s1[1].interface is s0[1].interface


# ----------------------------------------------------------------------
# Per-state structure ("sample != structure")
# ----------------------------------------------------------------------

_OXIDE = {
    "name": "Cu oxide",
    "sld": 4.0,
    "sld_min": 2.0,
    "sld_max": 6.0,
    "thickness": 30.0,
    "thickness_min": 5.0,
    "thickness_max": 100.0,
    "roughness": 5.0,
    "roughness_max": 20.0,
}


def test_resolve_tied_set_union_does_not_raise_for_per_state_layer(two_files):
    """A layer present in only some states is a valid tie target (union); the
    default tied set spans the union and resolution must not raise."""
    from aure.nodes.model_builder import _resolve_tied_set, _valid_layer_names

    defn = _two_state_definition(two_files)
    defn["layers"] = [_OXIDE, *defn["layers"]]  # template: [Cu oxide, Cu, Ti]
    defn["states"][1]["layers"] = list(defn["layers"][1:])  # H2O omits the oxide
    valid = _valid_layer_names(defn)
    assert "Cu oxide" in valid  # still a valid name (present in D2O / template)

    # shared/unshared naming the per-state-absent oxide must NOT raise.
    defn_u = dict(defn, unshared_parameters=["Cu oxide.interface"])
    _resolve_tied_set(defn_u)  # no raise
    pairs = _resolve_tied_set(defn)
    assert ("Cu oxide", "thickness") in pairs  # default set spans the union
    assert ("Cu", "thickness") in pairs


def test_per_state_structure_oxide_absent_in_one_state(two_files):
    """state D2O has the oxide; H2O omits it. Build succeeds; layers shared by
    BOTH states tie by identity across the differing stacks; the oxide tie
    simply does not apply to H2O."""
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["layers"] = [_OXIDE, *defn["layers"]]  # [Cu oxide, Cu, Ti]
    defn["states"][1]["layers"] = list(defn["layers"][1:])  # H2O = [Cu, Ti] (no oxide)

    problem, by_state, _ = build_states_problem(defn)
    s0 = by_state["D2O"][0].sample  # [Si, Cu oxide, Cu, Ti, D2O]
    s1 = by_state["H2O"][0].sample  # [Si, Cu, Ti, H2O]

    assert len(s1) == len(s0) - 1  # H2O is missing the oxide layer
    # Cu and Ti exist in both states and are tied by IDENTITY despite the
    # oxide shifting D2O's indices (Cu: s0[2] vs s1[1]).
    assert s1[1].thickness is s0[2].thickness  # Cu tied
    assert s1[1].material.rho is s0[2].material.rho
    assert s1[2].thickness is s0[3].thickness  # Ti tied
    # substrate.interface still tied; ambient (different solvents) not tied
    assert s1[0].interface is s0[0].interface
    assert s1[-1].material.rho is not s0[-1].material.rho


# ----------------------------------------------------------------------
# Per-state structure: ties and names follow the states' own stacks
# ----------------------------------------------------------------------

_OXIDE_ATTRS = ("thickness", "rho", "interface")


def _three_state_definition(files: list[str]) -> dict:
    """D2O, H2O and S2 on the template [Cu, Ti]."""
    defn = _two_state_definition(files[:2])
    defn["states"].append(
        {
            "name": "S2",
            "data_files": [{"file": files[1], "label": "S2-comb"}],
            "ambient": {"name": "S2amb", "sld": 3.0, "sld_min": 2.0, "sld_max": 4.0},
        }
    )
    return defn


def _with_oxide(defn: dict, *state_indices: int) -> dict:
    """Give those states their own stack: the template plus an oxide the
    template does not have."""
    import copy

    for i in state_indices:
        defn["states"][i]["layers"] = [
            copy.deepcopy(_OXIDE),
            *copy.deepcopy(defn["layers"]),
        ]
    return defn


def _oxide(by_state, name):
    return next(
        (s for s in by_state[name][0].sample if s.material.name == "Cu oxide"), None
    )


def _names(problem) -> list[str]:
    return [str(par.name) for par in problem._parameters]


def test_a_layer_absent_from_the_first_state_is_still_tied(two_files):
    """Ties were anchored on state 0, so an oxide the first state lacks was
    fitted independently in each state that had it — and both copies kept the
    same unprefixed name."""
    from aure.nodes.model_builder import build_states_problem

    defn = _with_oxide(_three_state_definition(two_files), 1, 2)
    problem, by_state, _ = build_states_problem(defn)

    assert _oxide(by_state, "D2O") is None
    h2o, s2 = _oxide(by_state, "H2O"), _oxide(by_state, "S2")
    assert h2o.thickness is s2.thickness
    assert h2o.material.rho is s2.material.rho
    assert h2o.interface is s2.interface
    names = _names(problem)
    assert len(names) == len(set(names))
    # One object, two states: the shared spelling, once.
    assert sorted(n for n in names if "oxide" in n) == sorted(
        f"Cu oxide {attr}" for attr in _OXIDE_ATTRS
    )


def test_an_untied_state_only_layer_is_named_per_state(two_files):
    """The renaming pass took its targets from the template, so a layer that
    lives only in states' own stacks was never renamed: two parameters, one
    name, and whatever is keyed by name saw only one of them."""
    from aure.nodes.model_builder import build_states_problem

    defn = _with_oxide(_three_state_definition(two_files), 0, 1)
    defn["unshared_parameters"] = [
        "Cu oxide.thickness",
        "Cu oxide.material.rho",
        "Cu oxide.interface",
    ]
    problem, by_state, _ = build_states_problem(defn)

    assert _oxide(by_state, "D2O").thickness is not _oxide(by_state, "H2O").thickness
    names = _names(problem)
    assert len(names) == len(set(names))
    assert sorted(n for n in names if "oxide" in n) == sorted(
        f"{state} Cu oxide {attr}" for state in ("D2O", "H2O") for attr in _OXIDE_ATTRS
    )


def test_an_unprefixed_name_means_one_parameter_for_several_states(two_files):
    """A layer only one state has is that state's own, so it is prefixed too:
    an unprefixed name then always says "shared"."""
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["layers"] = [_OXIDE, *defn["layers"]]  # template has the oxide
    defn["states"][1]["layers"] = list(defn["layers"][1:])  # H2O omits it
    problem, _, _ = build_states_problem(defn)

    names = _names(problem)
    assert sorted(n for n in names if "oxide" in n) == sorted(
        f"D2O Cu oxide {attr}" for attr in _OXIDE_ATTRS
    )
    # Cu and Ti are in both states, tied, and keep the shared spelling.
    assert {"Cu thickness", "Ti thickness"} <= set(names)


@pytest.mark.parametrize("tied", [True, False])
def test_a_roughness_tie_layer_builds_tied_or_untied(two_files, tied):
    """Its interface is an expression, which has no name to set. Untied, the
    renaming pass tried anyway and the build ended with an AttributeError; the
    free parameter is the fraction, and that is what takes the state prefix."""
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["layers"][0]["roughness_tie"] = {"fraction_max": 0.5}
    if not tied:
        defn["unshared_parameters"] = ["Cu.interface"]
    problem, _, _ = build_states_problem(defn)

    fractions = sorted(n for n in _names(problem) if "rough_frac" in n)
    expected = ["Cu rough_frac"] if tied else ["D2O Cu rough_frac", "H2O Cu rough_frac"]
    assert fractions == expected
    assert len(_names(problem)) == len(set(_names(problem)))


def test_two_parameters_under_one_name_are_refused(two_files):
    """Whatever produces it next, a name collision is an error at build, not a
    wrong number downstream. Two layers of one stack sharing a name is the
    case nothing else catches."""
    import copy

    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["layers"].insert(1, copy.deepcopy(defn["layers"][0]))  # Cu, Cu, Ti
    with pytest.raises(ValueError, match="belong to more than one parameter") as err:
        build_states_problem(defn)
    assert "Cu thickness" in str(err.value)


@pytest.mark.parametrize(
    "spec, value, fixed",
    [
        ({"value": 0.95, "fixed": True}, 0.95, True),
        ({"init": 0.9, "fixed": True}, 0.9, True),
        ({"fixed": True}, 1.0, True),
        ({"init": 0.9, "min": 0.8, "max": 1.0}, 0.9, False),
    ],
)
def test_state_intensity_value_applies_whether_or_not_fixed(
    two_files, spec, value, fixed
):
    from aure.nodes.model_builder import build_states_problem

    defn = _two_state_definition(two_files)
    defn["states"][0]["intensity"] = spec
    _, by_state, _ = build_states_problem(defn)
    intensity = by_state["D2O"][0].probe.intensity
    assert intensity.value == pytest.approx(value)
    assert intensity.fixed is fixed


def test_model_level_fixed_intensity_value_applies_on_every_builder(two_files):
    from aure.nodes.model_builder import build_experiment, build_multi_problem

    defn = _two_state_definition(two_files)
    defn.pop("states")
    defn["intensity"] = {"value": 0.95, "fixed": True}

    exp = build_experiment({**defn, "data_file": two_files[0]})
    assert exp.probe.intensity.value == pytest.approx(0.95)
    assert exp.probe.intensity.fixed

    _, experiments, _ = build_multi_problem(
        defn, [{"file": f, "label": f"f{i}"} for i, f in enumerate(two_files)]
    )
    assert [e.probe.intensity.value for e in experiments] == pytest.approx([0.95] * 2)
    assert all(e.probe.intensity.fixed for e in experiments)


@pytest.mark.parametrize("attr", ["theta_offset", "sample_broadening"])
def test_a_held_angle_nuisance_is_not_fitted(two_files, attr):
    """`fixed: true` on theta_offset / sample_broadening used to be ranged
    anyway, so a parameter declared held was fitted. Both builder paths."""
    from aure.nodes.model_builder import build_multi_problem, build_states_problem

    spec = {"init": 0.004, "min": 0.0, "max": 0.02, "fixed": True}
    files = [
        {"file": f, "label": f"f{i}", "theta": 0.6} for i, f in enumerate(two_files)
    ]

    defn = _two_state_definition(two_files)
    defn["states"] = [{"name": "A", "data_files": files, attr: spec}]
    _, by_state, _ = build_states_problem(defn)
    pars = [getattr(e.probe, attr) for e in by_state["A"]]

    defn = _two_state_definition(two_files)
    defn.pop("states")
    defn[attr] = spec
    _, experiments, _ = build_multi_problem(defn, files)
    pars_multi = [getattr(e.probe, attr) for e in experiments]

    for group in (pars, pars_multi):
        assert group[0] is group[1]  # still one parameter shared across files
        assert group[0].fixed
        assert group[0].value == pytest.approx(0.004)


def test_a_fitted_angle_nuisance_starts_from_init_on_both_paths(two_files):
    from aure.nodes.model_builder import build_multi_problem, build_states_problem

    spec = {"init": 0.004, "min": -0.01, "max": 0.01}
    files = [{"file": two_files[0], "label": "f0", "theta": 0.6}]

    defn = _two_state_definition(two_files)
    defn["states"] = [{"name": "A", "data_files": files, "theta_offset": spec}]
    _, by_state, _ = build_states_problem(defn)

    defn = _two_state_definition(two_files)
    defn.pop("states")
    defn["theta_offset"] = spec
    _, experiments, _ = build_multi_problem(defn, files)

    for par in (by_state["A"][0].probe.theta_offset, experiments[0].probe.theta_offset):
        assert not par.fixed
        assert par.value == pytest.approx(0.004)
        assert tuple(par.bounds) == (-0.01, 0.01)


# ----------------------------------------------------------------------
# A single state is the one-state case of the states builder
# ----------------------------------------------------------------------


def _flat_definition(files: list[str]) -> dict:
    defn = _two_state_definition(files)
    defn.pop("states")
    defn["ambient"] = {"name": "THF", "sld": 6.2}
    return defn


#: The parameters of `_flat_definition`, as the flat single-sample builders
#: always spelled and ranged them. Pinned as literals: the flat builders are
#: now views onto the state builder, so comparing the two would compare a
#: thing with itself.
_FLAT_STRUCTURE = [
    ("silicon interface", 3.0, (0, 15.0)),
    ("Cu interface", 5.0, (5.0, 20.0)),
    ("Cu rho", 6.5, (4.0, 8.0)),
    ("Cu thickness", 500.0, (250.0, 750.0)),
    ("Ti interface", 3.0, (3.0, 15.0)),
    ("Ti rho", -1.95, (-3.0, 0.0)),
    ("Ti thickness", 30.0, (10.0, 60.0)),
    ("THF rho", 6.2, (pytest.approx(4.96), pytest.approx(7.44))),
]


def _describe(problem) -> list:
    return [(p.name, p.value, tuple(p.bounds)) for p in problem._parameters]


def _intensity(label: str) -> tuple:
    return (f"intensity {label}", 1.0, (0.7, 1.1))


def test_a_single_curve_reads_like_a_flat_problem(two_files):
    """No state prefix anywhere, and the intensity named after the file —
    whether the model carries its one state, or is given it."""
    from aure.nodes.model_builder import build_problem, build_states_problem

    flat = _flat_definition(two_files)
    stem = os.path.splitext(os.path.basename(two_files[0]))[0]
    expected = [_intensity(stem), *_FLAT_STRUCTURE]

    one_state = dict(
        flat, states=[{"name": "OCV", "data_files": [{"file": two_files[0]}]}]
    )
    carried, _, _ = build_states_problem(one_state)
    implied = build_problem(dict(flat, data_file=two_files[0]))

    assert _describe(carried) == expected
    assert _describe(implied) == expected
    assert float(carried.chisq()) == pytest.approx(float(implied.chisq()))


def test_several_files_of_one_state_read_like_a_flat_co_refinement(two_files):
    from aure.nodes.model_builder import build_multi_problem, build_states_problem

    flat = _flat_definition(two_files)
    files = [{"file": f, "label": f"seg{i}"} for i, f in enumerate(two_files)]
    expected = [_intensity("seg0"), *_FLAT_STRUCTURE, _intensity("seg1")]

    carried, _, _ = build_states_problem(
        dict(flat, states=[{"name": "OCV", "data_files": files}])
    )
    implied, experiments, sorted_files = build_multi_problem(flat, files)

    assert _describe(carried) == expected
    assert _describe(implied) == expected
    assert len(experiments) == 2
    assert [ds["label"] for ds in sorted_files] == ["seg0", "seg1"]
    # One sample shared by both experiments: every structural parameter tied.
    assert experiments[0].sample is experiments[1].sample


def test_with_states_gives_a_stateless_model_the_state_it_implies():
    from aure.nodes.model_builder import with_states

    flat = {"layers": [], "data_file": "/x/a.txt", "ambient": {"name": "air"}}
    two = [{"file": "/x/a.txt", "label": "a"}, {"file": "/x/b.txt", "label": "b"}]

    assert with_states(flat)["states"] == [
        {"name": "state0", "data_files": [{"file": "/x/a.txt"}]}
    ]
    assert with_states(flat, two)["states"] == [{"name": "state0", "data_files": two}]
    assert "states" not in flat  # a view, not a mutation
    # No overrides on the implied state: the model-level fields apply as written.
    assert set(with_states(flat)["states"][0]) == {"name", "data_files"}

    # A model that brings its states keeps them, whatever files are passed.
    own = dict(flat, states=[{"name": "OCV", "data_files": [{"file": "/x/c.txt"}]}])
    assert with_states(own, two) is own


def test_the_flat_entry_points_honour_a_models_own_state(two_files):
    """They used to build from the model-level fields and `data_file` alone,
    ignoring a state the model carried. A state's override now reaches every
    entry point, because there is one builder behind them."""
    from aure.nodes.model_builder import build_experiment, build_problem

    model = dict(
        _flat_definition(two_files),
        data_file="/nonexistent/ignored.dat",
        states=[
            {
                "name": "OCV",
                "data_files": [{"file": two_files[0], "label": "curve"}],
                "ambient": {"rho": 5.5},
                "intensity": {"init": 0.9, "fixed": True},
            }
        ],
    )
    by_name = {p.name: p for p in build_problem(model)._parameters}
    assert by_name["THF rho"].value == pytest.approx(5.5)
    assert "intensity curve" not in by_name  # held, so not a fit parameter
    assert build_experiment(model).probe.intensity.value == pytest.approx(0.9)


def test_the_single_shape_views_refuse_other_shapes(two_files):
    from aure.nodes.model_builder import build_experiment, build_multi_problem

    files = [{"file": f, "label": f"seg{i}"} for i, f in enumerate(two_files)]
    flat = _flat_definition(two_files)
    with pytest.raises(ValueError, match="single-curve"):
        build_experiment(dict(flat, states=[{"name": "OCV", "data_files": files}]))
    with pytest.raises(ValueError, match="single-state"):
        build_multi_problem(_two_state_definition(two_files), files)


def test_one_state_ignores_tie_specs(two_files):
    """Nothing to tie: a spec naming a layer the stack lacks must not fail
    the build the way it does when it would silently tie nothing across
    several states."""
    from aure.nodes.model_builder import build_states_problem

    one_state = dict(
        _flat_definition(two_files),
        states=[{"name": "OCV", "data_files": [{"file": two_files[0]}]}],
        shared_parameters=["NoSuchLayer.thickness"],
    )
    problem, _, _ = build_states_problem(one_state)
    assert "Cu thickness" in problem.labels()


def _legacy_named_problem(two_files, *, n_files: int):
    from aure.nodes.model_builder import build_states_problem

    files = [
        {"file": f, "label": f"seg{i}", "theta": 0.6}
        for i, f in enumerate(two_files[:n_files])
    ]
    defn = dict(
        _flat_definition(two_files),
        states=[
            {
                "name": "OCV",
                "data_files": files,
                "background": {"init": 1e-6, "min": 0.0, "max": 1e-5},
                "theta_offset": {"init": 0.0, "min": -0.02, "max": 0.02},
            }
        ],
    )
    problem, _, _ = build_states_problem(defn)
    return problem, {p.name: p for p in problem._parameters}


def test_names_stored_with_the_state_prefix_still_apply(two_files):
    """Single-state runs with a nuisance key, and import-refl1d workspaces,
    stored "<state> ..." names before a single state stopped prefixing them.
    Their fits must still land on a rebuilt problem."""
    from aure.nodes.model_builder import apply_bounds, apply_parameters

    problem, by_name = _legacy_named_problem(two_files, n_files=1)
    assert {"THF rho", "background", "theta_offset", "intensity seg0"} <= set(by_name)

    apply_parameters(
        problem,
        {
            "OCV THF rho": 5.9,
            "OCV background": 3e-6,
            "OCV theta_offset": 0.004,
            "OCV intensity": 0.93,
        },
    )
    assert by_name["THF rho"].value == pytest.approx(5.9)
    assert by_name["background"].value == pytest.approx(3e-6)
    assert by_name["theta_offset"].value == pytest.approx(0.004)
    assert by_name["intensity seg0"].value == pytest.approx(0.93)

    apply_bounds(problem, {"OCV THF rho": [4.0, 7.5]})
    lo, hi = by_name["THF rho"].bounds
    assert (lo, hi) == (4.0, 7.5)


def test_prefixed_per_file_intensities_find_their_own_probe(two_files):
    from aure.nodes.model_builder import apply_parameters

    problem, by_name = _legacy_named_problem(two_files, n_files=2)
    apply_parameters(problem, {"OCV seg0 intensity": 0.83, "OCV seg1 intensity": 1.07})
    assert by_name["intensity seg0"].value == pytest.approx(0.83)
    assert by_name["intensity seg1"].value == pytest.approx(1.07)


def test_the_flat_single_file_background_name_still_applies(two_files):
    """The flat single-file builder left refl1d's "background <probe>" name in
    place; every path now says "background". A fit stored under the old name
    must still land."""
    from aure.nodes.model_builder import apply_parameters, build_problem

    model = dict(
        _flat_definition(two_files),
        data_file=two_files[0],
        background={"enabled": True, "min": 0.0, "max": 1e-5},
    )
    problem = build_problem(model)
    by_name = {p.name: p for p in problem._parameters}
    assert "background" in by_name

    stem = os.path.splitext(os.path.basename(two_files[0]))[0]
    apply_parameters(problem, {f"background {stem}": 4e-6})
    assert by_name["background"].value == pytest.approx(4e-6)


def test_an_unknown_name_is_still_ignored(two_files):
    from aure.nodes.model_builder import apply_parameters

    problem, by_name = _legacy_named_problem(two_files, n_files=1)
    before = {name: par.value for name, par in by_name.items()}
    apply_parameters(problem, {"OCV no such thing": 1.0, "elsewhere rho": 2.0})
    assert {name: par.value for name, par in by_name.items()} == before
