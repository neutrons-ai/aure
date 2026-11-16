"""Which fit a run reports: the adopted final polish, not the fit it polished.

``final_fit`` appends its DREAM result after the exploration fits and, when it
adopts it, writes the DREAM values into ``current_model`` / ``current_chi2``.
Every reporting surface used to index ``final_selection`` on its own — five
copies — and so showed the exploration fit's χ², method and ± under a model
holding the polish's values. With an ``lm`` exploration the table even carried
``lm``'s covariance estimates, which look like the polish's error bars and are
not. ``aure.state.reported_fit`` is the one resolution they share now.
"""

from __future__ import annotations

import copy
import io
import re
from contextlib import redirect_stdout
from unittest.mock import patch

from aure.state import exploration_fits, final_fit_result, reported_fit, selected_fit

EXPLORED = {
    "iteration": 1,
    "method": "lm",
    "chi_squared": 1.30,
    "parameters": {"Cu thickness": 495.0, "Cu rho": 6.50},
    "uncertainties": {"Cu thickness": 0.2, "Cu rho": 0.01},
    "issues": [],
}
EARLIER = {
    "iteration": 0,
    "method": "lm",
    "chi_squared": 4.70,
    "parameters": {"Cu thickness": 120.0, "Cu rho": 6.10},
    "uncertainties": None,
    "issues": [],
}
POLISH = {
    "iteration": 1,  # the iteration it polished — shared with EXPLORED
    "method": "dream",
    "chi_squared": 1.25,
    "parameters": {"Cu thickness": 497.5, "Cu rho": 6.52},
    "uncertainties": {"Cu thickness": 3.1, "Cu rho": 0.04},
    "issues": [],
    "final_fit": True,
}
MODEL = {
    "substrate": {"name": "Si", "sld": 2.07, "roughness": 3.0},
    "layers": [{"name": "Cu", "sld": 6.52, "thickness": 497.5, "roughness": 5.0}],
    "ambient": {"name": "air", "sld": 0.0},
}


def _run(*, adopted: bool = True, polish: bool = True) -> dict:
    """A finished run: two exploration fits, iteration 1 selected, then a polish."""
    state = {
        "Q": [0.01],
        "fit_results": [copy.deepcopy(EARLIER), copy.deepcopy(EXPLORED)],
        "final_selection": {"selected": True, "index": 1, "iteration": 1},
        "current_model": copy.deepcopy(MODEL),
        "current_chi2": 1.30,
        "model_history": [
            {"iteration": 0, "definition": {"layers": [], "tag": "iteration 0"}},
            {"iteration": 1, "definition": {"layers": [], "tag": "iteration 1"}},
        ],
    }
    if polish:
        state["fit_results"].append(copy.deepcopy(POLISH))
        state["final_fit"] = {"ran": True, "adopted": adopted, "method": "dream"}
        if adopted:
            state["current_chi2"] = 1.25
    return state


# ----------------------------------------------------------------------
# The resolver
# ----------------------------------------------------------------------


def test_an_adopted_polish_is_the_reported_fit():
    state = _run()
    fit, source = reported_fit(state)

    assert source == "final_fit"
    assert fit is state["fit_results"][2]
    # The fit that was judged, and that the polish started from, is unchanged.
    assert selected_fit(state) is state["fit_results"][1]


def test_a_polish_that_was_not_adopted_is_only_recorded():
    """It did not hold the selected fit's χ², so the model keeps the selected
    values — and the report must not hang the polish's ± on them."""
    state = _run(adopted=False)
    fit, source = reported_fit(state)

    assert source == "selected"
    assert fit is state["fit_results"][1]
    assert final_fit_result(state) is state["fit_results"][2]


def test_without_a_polish_the_selected_fit_is_reported():
    state = _run(polish=False)
    assert reported_fit(state) == (state["fit_results"][1], "selected")

    # No selection record (legacy / imported): the last fit.
    state["final_selection"] = {}
    assert reported_fit(state)[0] is state["fit_results"][-1]
    assert reported_fit({"fit_results": []}) == (None, None)


def test_the_polish_is_not_an_iteration():
    state = _run()
    assert exploration_fits(state) == state["fit_results"][:2]
    assert len(exploration_fits(_run(polish=False))) == 2


def test_a_polish_stored_before_results_were_stamped_is_still_found():
    state = _run()
    del state["fit_results"][2]["final_fit"]

    assert reported_fit(state) == (state["fit_results"][2], "final_fit")
    assert len(exploration_fits(state)) == 2


def test_a_polish_followed_by_a_later_fit_describes_an_earlier_answer():
    """A restarted run appends new fits after the old polish. Until its own
    polish runs, the old one is not the run's latest word."""
    state = _run()
    state["fit_results"].append(dict(EXPLORED, iteration=2, chi_squared=1.10))

    assert final_fit_result(state) is None
    assert reported_fit(state)[1] == "selected"


def test_a_restart_drops_the_previous_polish():
    """`final_fit` is a no-op when the feature is off, so a restarted run
    would otherwise keep an "adopted" record for an answer it replaced — and
    the report and problem.json would follow it."""
    from aure.workflow.runner import prepare_state_for_restart

    restarted = prepare_state_for_restart(_run(), "try an oxide", "modeling")
    assert restarted["final_fit"] is None
    assert reported_fit(restarted)[1] == "selected"


def test_final_fit_stamps_its_result(monkeypatch):
    from aure.nodes import final_fit as ff

    monkeypatch.setenv("FIT_METHOD", "lm")
    monkeypatch.setenv("FIT_METHOD_FINAL", "dream")
    result = {k: v for k, v in POLISH.items() if k != "final_fit"}
    monkeypatch.setattr(ff, "run_fit_for_model", lambda **_k: result)

    state = _run(polish=False)
    out = ff.final_fit_node(state)

    assert out["final_fit"]["adopted"] is True
    assert out["fit_results"][0]["final_fit"] is True


# ----------------------------------------------------------------------
# The terminal report
# ----------------------------------------------------------------------


def _report(state: dict) -> str:
    from aure import cli

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cli._print_analysis_results(state, None)
    text = re.sub(r"\x1b\[[0-9;]*m", "", buffer.getvalue())
    return text[text.index("Fit Results") :]


def test_the_report_shows_the_adopted_polish():
    text = _report(_run())

    assert "χ²: 1.2500" in text
    assert "Method: dream — final fit of the selected iteration" in text
    assert "lm exploration χ² = 1.3000" in text
    assert "Cu thickness: 497.5000 ± 3.1000" in text
    # Not lm's covariance estimate, and not lm's value.
    assert "495.0000" not in text and "± 0.2000" not in text
    # Two iterations were fitted; the polish is not a third.
    assert "Iterations: 2" in text
    assert "Selected iteration: 1" in text


def test_the_report_keeps_the_selected_fit_for_an_unadopted_polish():
    text = _report(_run(adopted=False))

    assert "χ²: 1.3000" in text
    assert "Method: lm" in text and "final fit" not in text
    assert "Cu thickness: 495.0000 ± 0.2000" in text
    assert "497.5000" not in text and "± 3.1000" not in text
    assert "Iterations: 2" in text


def test_the_selection_payload_says_which_fit_and_reads_the_veto_off_the_judged_one():
    from aure import cli

    state = _run()
    state["fit_results"][1]["profile_artifact"] = True
    payload = cli._selection_payload(state)

    assert payload["reported_fit"] == "final_fit"
    assert payload["of_iterations_fitted"] == 2
    # The evaluator stamps the fit it judged; the polish carries no stamp, so
    # reading the reported fit would call a vetoed selection clean.
    assert payload["profile_artifact"] is True
    assert cli._selection_payload(_run(polish=False))["reported_fit"] == "selected"


# ----------------------------------------------------------------------
# The ISAAC exporter's context prompt
# ----------------------------------------------------------------------


def test_the_export_context_quotes_the_reported_fits_parameters(monkeypatch):
    """`current_chi2` is the polish's χ² once adopted; the values quoted next
    to it were the exploration fit's."""
    from aure.exporters import isaac

    seen = {}

    def fake_invoke(_llm, prompt, **_kwargs):
        seen["prompt"] = prompt
        raise RuntimeError("stop here")

    with (
        patch("aure.llm.get_llm", return_value=object()),
        patch("aure.llm.invoke_with_timeout", side_effect=fake_invoke),
    ):
        isaac._generate_context_description(_run())

    assert "Cu thickness: 497.5" in seen["prompt"]
    assert "495.0" not in seen["prompt"]


# ----------------------------------------------------------------------
# The web Results tab
# ----------------------------------------------------------------------


def _run_data(state: dict, tmp_path):
    from aure.web.data import RunData

    rd = RunData(str(tmp_path))
    return patch.object(rd, "get_final_state", return_value=state), rd


def test_the_results_table_defaults_to_the_adopted_polish(tmp_path):
    ctx, rd = _run_data(_run(), tmp_path)
    with ctx:
        table = rd.get_fit_parameters()

    assert (table["iteration"], table["method"], table["chi_squared"]) == (
        2,
        "dream",
        1.25,
    )
    row = next(r for r in table["parameters"] if r["name"] == "Cu thickness")
    assert (row["value"], row["uncertainty"]) == (497.5, 3.1)


def test_the_results_table_defaults_to_the_selected_fit_otherwise(tmp_path):
    for state in (_run(adopted=False), _run(polish=False)):
        ctx, rd = _run_data(state, tmp_path)
        with ctx:
            table = rd.get_fit_parameters()
        assert (table["iteration"], table["method"]) == (1, "lm")


def test_the_polish_is_its_own_selectable_entry(tmp_path):
    """It shares its iteration number with the fit it polished, so it was
    merged into that entry: one dropdown item, two curves with one label, and
    no way to select the fit the run reports."""
    ctx, rd = _run_data(_run(), tmp_path)
    with ctx:
        data = rd.get_reflectivity_data()

    assert [(m["iteration"], m["label"]) for m in data["models"]] == [
        (0, "Iteration 0 (χ²=4.70)"),
        (1, "Iteration 1 (χ²=1.30)"),
        (2, "Final dream fit (χ²=1.25)"),
    ]
    assert [i["iteration"] for i in data["iterations"]] == [0, 1, 2]
    assert data["iterations"][2]["label"] == "Final dream fit (χ²=1.25)"
    assert data["best_iteration"] == 2


def test_the_polish_is_simulated_on_the_model_it_was_fitted_on(tmp_path):
    """finalize writes the selected fit's values into `current_model` before
    the polish runs, which moves every bound derived from a value — so the
    polish's parameters can be infeasible against the selected iteration's
    stored model."""
    state = _run()
    ctx, rd = _run_data(state, tmp_path)
    with ctx:
        assert rd._model_for_fit(2) is state["current_model"]
        assert rd._model_for_fit(1)["tag"] == "iteration 1"
        assert rd._model_for_fit(0)["tag"] == "iteration 0"
