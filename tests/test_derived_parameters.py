"""Reparametrization: fit a combination, derive the raw parameter from it.

The data pins some combinations far better than the coordinates they are
written in — a thin layer's ``Δρ·t`` is determined while ``Δρ`` and ``t``
separately are not — and what an independent measurement gives you (a surface
excess, a volume fraction) is usually a combination too. These tests cover the
mechanism that lets a model be written in those coordinates instead.
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest

from aure.config import ConfigError, _parse_derived_parameters
from aure.nodes.evaluation import _count_free_params
from aure.nodes.expressions import (
    ExpressionError,
    evaluate,
    evaluate_constraint,
    referenced_names,
)
from aure.nodes.model_builder import (
    build_problem,
    build_states_problem,
    data_chisq,
    save_problem_json,
    validate_derived_parameters,
)

warnings.filterwarnings("ignore")


def _data_file() -> str:
    Q = np.linspace(0.01, 0.10, 60)
    R = np.clip((0.0217 / (2 * Q)) ** 4, 1e-10, 1.0)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for q, r in zip(Q, R):
        f.write(f"{q:.6f} {r:.6e} {0.05 * r:.6e} {0.02 * q:.6e}\n")
    f.close()
    return f.name


@pytest.fixture(scope="module")
def data_file():
    path = _data_file()
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def _single_state(data_file: str, **extra) -> dict:
    defn = {
        "substrate": {"name": "silicon", "sld": 2.07, "roughness": 3.0},
        "ambient": {"name": "air", "sld": 0.0},
        "layers": [
            {
                "name": "Cu",
                "sld": 6.5,
                "thickness": 500.0,
                "thickness_min": 250.0,
                "thickness_max": 750.0,
                "roughness": 8.0,
                "roughness_max": 30.0,
            }
        ],
        "data_file": data_file,
        "intensity": {"fixed": True},
    }
    defn.update(extra)
    return defn


_EXCESS = [
    {
        "name": "G",
        "free": {"init": 3250.0, "min": 1000.0, "max": 5000.0},
        "assign": {"Cu.rho": "air.rho + G / Cu.thickness"},
        "keep_physical": ["Cu.rho > 0", "Cu.rho < 9"],
        "source": "test",
    }
]


# ----------------------------------------------------------------------
# The expression evaluator is the attack surface: it runs LLM-writable text
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "(lambda: 1)()",
        "a if b else 0",
        "a % b",
        "a // b",
        "SEI.rho[0]",
        "[a, b]",
        "{'a': 1}",
        "a and b",
    ],
)
def test_evaluator_refuses_everything_that_is_not_arithmetic(expr):
    with pytest.raises(ExpressionError):
        evaluate(expr, {"a": 1.0, "b": 2.0, "SEI.material.rho": 3.0})


def test_evaluator_arithmetic_and_name_spellings():
    ns = {"SEI.material.rho": 1.5, "SEI.thickness": 80.0, "s.material.rho": 6.35}
    # `X.rho` and `X.material.rho` are the same parameter
    assert evaluate("(SEI.rho - s.rho) * SEI.thickness", ns) == pytest.approx(-388.0)
    assert evaluate("(SEI.material.rho - s.material.rho) * SEI.thickness", ns) == (
        pytest.approx(-388.0)
    )
    assert evaluate("-2 ** 2 + 10 / 4", ns) == pytest.approx(-1.5)


def test_unknown_name_is_reported_with_the_known_ones():
    with pytest.raises(ExpressionError) as exc:
        evaluate("nope.rho + 1", {"Cu.thickness": 1.0})
    assert "nope.rho" in str(exc.value) and "Cu.thickness" in str(exc.value)


def test_referenced_names_does_not_leak_dotted_prefixes():
    assert referenced_names("(SEI.rho - dTHF.rho) * SEI.thickness") == {
        "SEI.material.rho",
        "dTHF.material.rho",
        "SEI.thickness",
    }


def test_constraint_must_be_a_single_comparison():
    ns = {"a": 1.0, "b": 2.0}
    assert evaluate_constraint("a < b", ns) is not None
    with pytest.raises(ExpressionError):
        evaluate_constraint("a < b < 3", ns)
    with pytest.raises(ExpressionError):
        evaluate_constraint("a + b", ns)


# ----------------------------------------------------------------------
# Single state: the raw parameter leaves the free set
# ----------------------------------------------------------------------


def test_assigned_parameter_is_no_longer_free(data_file):
    plain = {str(p.name) for p in build_problem(_single_state(data_file))._parameters}
    reparam = build_problem(_single_state(data_file, derived_parameters=_EXCESS))
    names = {str(p.name) for p in reparam._parameters}
    assert "Cu rho" in plain and "G" not in plain
    assert "G" in names and "Cu rho" not in names
    # one coordinate swapped for another, not added
    assert len(names) == len(plain)


def test_derived_value_tracks_the_free_parameter(data_file):
    problem = build_problem(_single_state(data_file, derived_parameters=_EXCESS))
    sample = list(problem.models)[0].sample
    for thickness in (400.0, 650.0):
        sample[1].thickness.value = thickness
        assert sample[1].material.rho.value == pytest.approx(3250.0 / thickness)


def test_reparametrization_does_not_change_the_physics(data_file):
    """Same model, different coordinates — the fit quality must be identical."""
    plain = build_problem(_single_state(data_file))
    reparam = build_problem(_single_state(data_file, derived_parameters=_EXCESS))
    assert data_chisq(reparam) == pytest.approx(data_chisq(plain))


# ----------------------------------------------------------------------
# χ² must stay a statement about the DATA
# ----------------------------------------------------------------------


def test_data_chisq_matches_chisq_without_priors_or_constraints(data_file):
    problem = build_problem(_single_state(data_file))
    assert data_chisq(problem) == pytest.approx(problem.chisq())


def test_violated_constraint_reports_infeasible_not_a_perfect_fit(data_file):
    """bumps returns pmodel = 0.0 when a constraint fails; scaling that would
    report χ² = 0 — which reads as a flawless fit AND lands under the
    acceptance floor, filing an impossible model as an error-bar problem."""
    defn = _single_state(
        data_file,
        derived_parameters=[
            {
                "name": "G",
                "free": {"init": -3250.0, "min": -5000.0, "max": -1000.0},
                "assign": {"Cu.rho": "air.rho + G / Cu.thickness"},
                "keep_physical": ["Cu.rho > 0"],  # unsatisfiable with G < 0
            }
        ],
    )
    problem = build_problem(defn)
    assert problem.chisq() > 1e6  # bumps' own penalty
    assert data_chisq(problem) == float("inf")
    assert data_chisq(problem) != 0.0


# ----------------------------------------------------------------------
# Multi-state: the invariant is shared, the derived value is not
# ----------------------------------------------------------------------


def _two_state(data_file: str, derived: list) -> dict:
    return {
        "substrate": {"name": "silicon", "sld": 2.07, "roughness": 3.0,
                      "roughness_max": 15.0},
        "ambient": {"name": "solvent", "sld": 6.35},
        "layers": [
            {
                "name": "film",
                "sld": 5.0,
                "thickness": 80.0,
                "thickness_min": 20.0,
                "thickness_max": 200.0,
                "roughness": 5.0,
                "roughness_max": 20.0,
            }
        ],
        "states": [
            {
                "name": "dTHF",
                "data_files": [{"file": data_file, "label": "a"}],
                "ambient": {"name": "solvent", "sld": 6.35},
            },
            {
                "name": "hTHF",
                "data_files": [{"file": data_file, "label": "b"}],
                "ambient": {"name": "solvent", "sld": 0.18},
            },
        ],
        "derived_parameters": derived,
    }


_SOLVATION = [
    {
        "name": "phi",
        "free": {"init": 0.30, "min": 0.05, "max": 1.0},
        "assign": {"film.rho": "phi * rho_dry + (1 - phi) * solvent.rho"},
        "source": "solvated film, two-component mixing",
    },
    {"name": "rho_dry", "free": {"init": 2.0, "min": 0.5, "max": 4.0}},
]


def test_contrast_variation_shares_the_invariant_not_the_sld(data_file):
    """The relationship contrast variation actually has: one volume fraction
    and one dry SLD across states, each state's layer SLD following from its
    OWN solvent. No shared_parameters entry can express this — the invariant
    is not a layer attribute."""
    problem, experiments, _ = build_states_problem(_two_state(data_file, _SOLVATION))
    names = {str(p.name) for p in problem._parameters}
    assert {"phi", "rho_dry"} <= names
    # one shared phi/rho_dry, not one per state
    assert not any(n.startswith(("dTHF phi", "hTHF phi")) for n in names)
    assert "film rho" not in names and "dTHF film rho" not in names

    rho_by_state = {
        name: exps[0].sample[1].material.rho.value
        for name, exps in experiments.items()
    }
    assert rho_by_state["dTHF"] == pytest.approx(0.3 * 2.0 + 0.7 * 6.35)
    assert rho_by_state["hTHF"] == pytest.approx(0.3 * 2.0 + 0.7 * 0.18)
    assert rho_by_state["dTHF"] != rho_by_state["hTHF"]


def test_untied_derived_parameter_is_per_state(data_file):
    derived = [dict(_SOLVATION[0], tied=False), _SOLVATION[1]]
    problem, _exps, _ = build_states_problem(_two_state(data_file, derived))
    names = {str(p.name) for p in problem._parameters}
    assert {"dTHF phi", "hTHF phi"} <= names
    assert "phi" not in names


def test_scoped_derived_parameter_leaves_other_states_alone(data_file):
    derived = [dict(_SOLVATION[0], states=["dTHF"]), _SOLVATION[1]]
    problem, _exps, _ = build_states_problem(_two_state(data_file, derived))
    names = {str(p.name) for p in problem._parameters}
    # hTHF keeps an ordinary free SLD; dTHF's is derived
    assert "hTHF film rho" in names
    assert "phi" in names


# ----------------------------------------------------------------------
# Free-parameter accounting (BIC)
# ----------------------------------------------------------------------


def test_free_param_count_is_neutral_for_a_one_for_one_swap():
    base = {
        "layers": [{"name": "SEI"}],
        "substrate": {"name": "Si", "roughness_max": 15},
        "ambient": {"name": "air", "sld": 0.0},
        "intensity": {},
    }
    swapped = dict(
        base, derived_parameters=[{"name": "G", "assign": {"SEI.rho": "..."}}]
    )
    assert _count_free_params(swapped) == _count_free_params(base)


def test_free_param_count_charges_for_a_two_for_one():
    base = {
        "layers": [{"name": "SEI"}],
        "substrate": {"name": "Si", "roughness_max": 15},
        "ambient": {"name": "air", "sld": 0.0},
        "intensity": {},
    }
    solvation = dict(
        base,
        derived_parameters=[
            {"name": "phi", "assign": {"SEI.rho": "..."}},
            {"name": "rho_dry"},
        ],
    )
    assert _count_free_params(solvation) == _count_free_params(base) + 1


# ----------------------------------------------------------------------
# Validation and refusals
# ----------------------------------------------------------------------


def test_validation_rejects_a_target_that_names_no_layer(data_file):
    defn = _single_state(
        data_file,
        derived_parameters=[
            {
                "name": "G",
                "free": {"min": 1.0, "max": 2.0},
                "assign": {"Nonesuch.rho": "G"},
            }
        ],
    )
    with pytest.raises(ValueError, match="names no layer"):
        validate_derived_parameters(defn)


def test_validation_rejects_an_unreachable_free_parameter(data_file):
    defn = _single_state(
        data_file,
        derived_parameters=[{"name": "orphan", "free": {"min": 0.0, "max": 1.0}}],
    )
    with pytest.raises(ValueError, match="reaches nothing"):
        validate_derived_parameters(defn)


def test_validation_allows_an_auxiliary_parameter_another_entry_uses(data_file):
    defn = _two_state(data_file, _SOLVATION)
    validate_derived_parameters(defn)  # must not raise


def test_validation_rejects_a_name_colliding_with_a_layer(data_file):
    defn = _single_state(
        data_file,
        derived_parameters=[
            {"name": "Cu", "free": {"min": 0.0, "max": 1.0},
             "assign": {"Cu.thickness": "Cu * 100"}}
        ],
    )
    with pytest.raises(ValueError, match="collides"):
        validate_derived_parameters(defn)


def test_problem_json_export_refuses_rather_than_dropping_the_reparametrization(
    data_file, tmp_path
):
    """bumps serialization does not round-trip expression parameters, so the
    exported problem would be a different model with no warning."""
    defn = _single_state(data_file, derived_parameters=_EXCESS)
    with pytest.raises(ValueError, match="derived_parameters"):
        save_problem_json(defn, tmp_path / "problem.json")


def test_config_layer_shape_checks():
    assert _parse_derived_parameters(None) == []
    with pytest.raises(ConfigError, match="must be a list"):
        _parse_derived_parameters({"name": "phi"})
    with pytest.raises(ConfigError, match="missing a non-empty `name`"):
        _parse_derived_parameters([{"free": {"min": 0, "max": 1}}])
    with pytest.raises(ConfigError, match="Duplicate"):
        _parse_derived_parameters(
            [
                {"name": "a", "free": {"min": 0, "max": 1}},
                {"name": "a", "free": {"min": 0, "max": 1}},
            ]
        )
    parsed = _parse_derived_parameters(
        [
            {
                "name": "phi",
                "free": {"init": 0.3, "min": 0.0, "max": 1.0},
                "assign": {"film.rho": "phi"},
                "keep_physical": "film.rho > 0",
                "source": "QCM-D",
            }
        ]
    )
    assert parsed[0]["keep_physical"] == ["film.rho > 0"]  # scalar accepted
    assert parsed[0]["source"] == "QCM-D"


# ----------------------------------------------------------------------
# A structural edit that invalidates a declaration prunes it, not the run
# ----------------------------------------------------------------------


def test_build_prunes_a_declaration_whose_layer_was_removed(data_file):
    """A refinement can remove the layer a reparametrization is written
    against. Raising would end the run and forfeit every remaining iteration
    over an edit the model was never told was forbidden."""
    defn = _single_state(data_file, derived_parameters=_EXCESS)
    defn["layers"] = [
        {"name": "Ti", "sld": -1.9, "thickness": 50.0, "roughness": 5.0,
         "roughness_max": 20.0}
    ]
    problem = build_problem(defn)  # must not raise
    names = {str(p.name) for p in problem._parameters}
    assert "G" not in names
    assert "Ti rho" in names


def test_build_does_not_mutate_the_callers_definition(data_file):
    defn = _single_state(data_file, derived_parameters=_EXCESS)
    defn["layers"] = []
    build_problem(defn)
    assert len(defn["derived_parameters"]) == 1


def test_pruning_cascades_to_orphaned_auxiliaries(data_file):
    from aure.nodes.model_builder import surviving_derived_parameters

    defn = _two_state(data_file, _SOLVATION)
    defn["layers"] = []  # the film is gone; phi has nothing to assign
    kept, notes = surviving_derived_parameters(defn)
    assert kept == []
    assert any("names no layer" in n for n in notes)
    assert any("unreferenced" in n for n in notes)


def test_prune_reports_what_it_dropped(data_file):
    from aure.nodes.model_builder import prune_derived_parameters

    defn = _single_state(data_file, derived_parameters=_EXCESS)
    defn["layers"] = []
    notes = prune_derived_parameters(defn)
    assert notes and "'G'" in notes[0]
    assert defn["derived_parameters"] == []  # pruned in place, like tie specs


def test_a_still_valid_declaration_is_untouched_by_pruning(data_file):
    from aure.nodes.model_builder import prune_derived_parameters

    defn = _single_state(data_file, derived_parameters=_EXCESS)
    assert prune_derived_parameters(defn) == []
    assert len(defn["derived_parameters"]) == 1


# ----------------------------------------------------------------------
# The opt-in gate
# ----------------------------------------------------------------------


def test_feature_is_off_by_default(monkeypatch):
    from aure.config import derived_parameters_enabled

    monkeypatch.delenv("ALLOW_DERIVED_PARAMETERS", raising=False)
    assert derived_parameters_enabled() is False
    assert derived_parameters_enabled(None) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "ON"])
def test_env_var_enables_it(monkeypatch, value):
    from aure.config import derived_parameters_enabled

    monkeypatch.setenv("ALLOW_DERIVED_PARAMETERS", value)
    assert derived_parameters_enabled() is True


def test_explicit_config_key_wins_over_the_env(monkeypatch):
    from aure.config import derived_parameters_enabled

    monkeypatch.setenv("ALLOW_DERIVED_PARAMETERS", "1")
    assert derived_parameters_enabled(False) is False


def test_declaring_without_the_gate_is_refused_not_ignored():
    """Ignoring the block would fit a model measurably different from the one
    described, with nothing in the report to say why."""
    from aure.config import check_derived_parameters_allowed

    cfg = {"derived_parameters": [{"name": "phi"}], "allow_derived_parameters": False}
    with pytest.raises(ConfigError, match="allow_derived_parameters"):
        check_derived_parameters_allowed(cfg, source="setup.yaml")
    cfg["allow_derived_parameters"] = True
    check_derived_parameters_allowed(cfg, source="setup.yaml")  # must not raise


def test_refinement_prompt_carries_the_rule_only_when_the_model_has_a_block():
    from aure.nodes.prompts import format_model_refinement_prompt_json

    base = dict(sample_description="x", fit_result={"chi_squared": 1.0}, features={})
    model = {"substrate": {"sld": 1}, "layers": [], "ambient": {"sld": 0}}
    plain = format_model_refinement_prompt_json(current_model=model, **base)
    reparam = format_model_refinement_prompt_json(
        current_model={**model, "derived_parameters": [{"name": "G"}]}, **base
    )
    assert "REPARAMETRIZED PARAMETERS" not in plain
    assert "REPARAMETRIZED PARAMETERS" in reparam


def test_skill_activation_follows_the_gate():
    from aure.nodes.intake import _gate_functional_constraints_skill as gate

    assert gate(["neutron-reflectometry"], {}) == ["neutron-reflectometry"]
    # stripped even if the selector picked it out of the catalog
    assert gate(["functional-constraints", "neutron-reflectometry"], {}) == [
        "neutron-reflectometry"
    ]
    assert "functional-constraints" in gate(
        ["neutron-reflectometry"], {"derived_parameters": [{"name": "G"}]}
    )
    assert "functional-constraints" in gate(
        ["neutron-reflectometry"], {"allow_derived_parameters": True}
    )


def test_the_skill_is_installed_and_loadable():
    from aure.skills import SkillRegistry, load_skill_context

    registry = SkillRegistry()
    assert "functional-constraints" in registry.skill_names
    body = load_skill_context(["functional-constraints"], registry)
    assert "derived_parameters" in body and "contrast" in body.lower()
