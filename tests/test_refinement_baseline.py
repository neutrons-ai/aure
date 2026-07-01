"""Baseline (intake) rewind plumbing for reinterpretation hypotheses.

When the working model has structurally diverged from the clean intake model
(e.g. the optimizer inflated a layer to fake a contrast), the refinement prompt
exposes the intake baseline so the LLM can rebuild from it instead of stacking a
deuterated-ambient reinterpretation on top of the speculative structure.
"""

import json

from aure.nodes.prompts import (
    _format_baseline_model_section,
    _structural_skeleton,
    format_model_refinement_prompt_json,
)


def _baseline():
    return {
        "substrate": {"name": "Si", "sld": 2.07},
        "layers": [
            {"name": "Ti", "sld": -1.95, "thickness": 30.0},
            {"name": "Cu", "sld": 6.55, "thickness": 500.0},
            {"name": "native copper oxide", "sld": 5.0, "thickness": 20.0},
        ],
        "ambient": {"name": "electrolyte", "sld": 0.0},
        "back_reflection": False,
        "data_file": "/tmp/x.dat",
    }


def _diverged_current():
    cur = _baseline()
    # Optimizer ballooned Cu and bolted on a speculative surface layer to fake
    # the low-Q contrast a deuterated ambient would supply for free.
    cur["layers"][1]["thickness"] = 993.0
    cur["layers"].append({"name": "hydrated layer", "sld": 2.0, "thickness": 50.0})
    return cur


def _fit_result():
    return {
        "chi_squared": 19.77,
        "method": "dream",
        "converged": True,
        "parameters": {},
    }


def test_baseline_section_shown_when_diverged():
    prompt = format_model_refinement_prompt_json(
        current_model=_diverged_current(),
        sample_description="Cu on Si in electrolyte",
        fit_result=_fit_result(),
        features={},
        baseline_model=_baseline(),
        next_action="structural_change",
        proposed_hypothesis_id=1,
    )
    assert "Baseline (Intake) Model" in prompt
    # Baseline shows the clean Cu thickness (500), not the ballooned 993.
    assert '"thickness": 500.0' in prompt
    # The rewind rule is always present in the template.
    assert "REWIND FOR REINTERPRETATION" in prompt


def test_baseline_section_omitted_when_identical():
    prompt = format_model_refinement_prompt_json(
        current_model=_baseline(),
        sample_description="x",
        fit_result=_fit_result(),
        features={},
        baseline_model=_baseline(),
    )
    assert "Baseline (Intake) Model" not in prompt


def test_baseline_section_omitted_when_none():
    prompt = format_model_refinement_prompt_json(
        current_model=_baseline(),
        sample_description="x",
        fit_result=_fit_result(),
        features={},
        baseline_model=None,
    )
    assert "Baseline (Intake) Model" not in prompt


def test_format_baseline_section_direct():
    assert _format_baseline_model_section(None, _baseline()) == ""
    assert _format_baseline_model_section(_baseline(), _baseline()) == ""
    section = _format_baseline_model_section(_baseline(), _diverged_current())
    assert "Baseline (Intake) Model" in section


def test_baseline_section_shown_when_only_values_inflated():
    # The real signature: SAME layer stack, but the optimizer ballooned Cu
    # thickness and pinned the oxide SLD high — no layer added/removed. The
    # baseline must still surface so the LLM can rewind.
    cur = _baseline()
    cur["layers"][1]["thickness"] = 993.0  # Cu inflated ~2x
    cur["layers"][2]["sld"] = 6.96  # oxide rho pinned toward ambient
    assert len(cur["layers"]) == len(_baseline()["layers"])
    section = _format_baseline_model_section(_baseline(), cur)
    assert "Baseline (Intake) Model" in section
    assert '"thickness": 500.0' in section


def test_structural_skeleton_drops_runtime_data():
    model = {
        "substrate": {"name": "Si", "sld": 2.07},
        "layers": [{"name": "Cu", "sld": 6.5, "thickness": 500}],
        "ambient": {"name": "D2O", "sld": 6.4},
        "states": [
            {
                "name": "s0",
                "data_files": [
                    {"file": "a.dat", "Q": [1, 2, 3], "R": [1, 2, 3], "dR": [1, 2, 3]}
                ],
                "ambient": {"name": "D2O", "sld": 6.4},
            }
        ],
        "fitted_parameters": {"Cu thickness": 500},
    }
    skel = _structural_skeleton(model)
    dumped = json.dumps(skel)
    assert "data_files" not in dumped
    assert "fitted_parameters" not in dumped
    assert skel["states"][0]["name"] == "s0"
    assert skel["states"][0]["ambient"]["sld"] == 6.4


def test_strip_dataset_arrays_drops_qrdr_keeps_metadata():
    from aure.nodes.modeling import _strip_dataset_arrays

    model = {
        "substrate": {"name": "Si", "sld": 2.07},
        "layers": [{"name": "Cu", "sld": 6.5, "thickness": 500}],
        "ambient": {"name": "D2O", "sld": 6.4},
        "states": [
            {
                "name": "s0",
                "ambient": {"name": "D2O", "sld": 6.4},
                "data_files": [
                    {
                        "file": "a.dat",
                        "label": "low-Q",
                        "theta": 0.5,
                        "Q": [1, 2, 3],
                        "R": [1, 2, 3],
                        "dR": [1, 2, 3],
                    }
                ],
            }
        ],
    }
    slim = _strip_dataset_arrays(model)
    ds = slim["states"][0]["data_files"][0]
    assert "Q" not in ds and "R" not in ds and "dR" not in ds
    assert ds["file"] == "a.dat" and ds["label"] == "low-Q" and ds["theta"] == 0.5
    # Original is untouched (deepcopy).
    assert model["states"][0]["data_files"][0]["Q"] == [1, 2, 3]


def test_build_initial_model_snapshots_baseline_deepcopied():
    from aure.nodes.modeling import _build_initial_model

    state = {
        "data_file": "/tmp/x.dat",
        "parsed_sample": {
            "substrate": {
                "name": "Si",
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
                }
            ],
            "ambient": {"name": "electrolyte", "sld": 0.0},
            "back_reflection": False,
            "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
            "constraints": [],
        },
        "extracted_features": {},
        "iteration": 0,
        "messages": [],
        "states": [],
        "user_config": {},
        "dq_is_fwhm": True,
    }
    out = _build_initial_model(state)
    assert "error" not in out
    assert out["baseline_model"] == out["current_model"]
    # Deepcopy: mutating current_model must not reach the baseline snapshot.
    out["current_model"]["layers"][0]["thickness"] = 9999.0
    assert out["baseline_model"]["layers"][0]["thickness"] == 500.0
