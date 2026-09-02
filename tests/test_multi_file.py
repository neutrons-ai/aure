"""Tests for multi-file co-refinement support."""

import os
import tempfile

import numpy as np
import pytest

from aure.state import (
    create_initial_state,
    PerFileFitResult,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_data_file(q_min: float = 0.01, q_max: float = 0.10, n: int = 80) -> str:
    """Create a synthetic reflectivity data file for a Q-range segment."""
    Q = np.linspace(q_min, q_max, n)
    Qc = 0.0217
    R = np.where(Q < Qc, 1.0, (Qc / (2 * Q)) ** 4)
    R *= 1 + 0.2 * np.cos(2 * Q * 100.0)
    R = np.clip(R, 1e-10, 1.0)
    dR = 0.05 * R
    dQ = 0.02 * Q  # 2% resolution

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
    f.write("# Q R dR dQ\n")
    for q, r, dr, dq in zip(Q, R, dR, dQ):
        f.write(f"{q:.6f}  {r:.6e}  {dr:.6e}  {dq:.6e}\n")
    f.close()
    return f.name


def _make_data_files():
    """Create three Q-range segment files."""
    f1 = _make_data_file(0.008, 0.06, 60)
    f2 = _make_data_file(0.05, 0.12, 80)
    f3 = _make_data_file(0.10, 0.25, 100)
    return [f1, f2, f3]


def _cleanup(files):
    for f in files:
        try:
            os.unlink(f)
        except OSError:
            pass


# ============================================================================
# State Tests
# ============================================================================


class TestStateMultiFile:
    """Tests for multi-file state fields."""

    def test_data_files_default_empty(self):
        state = create_initial_state("dummy.dat", "test sample")
        assert state["data_files"] == []

    def test_data_files_set_from_arg(self):
        ds = [
            {"file": "/tmp/a.dat", "label": "low-Q"},
            {"file": "/tmp/b.dat", "label": "high-Q"},
        ]
        state = create_initial_state("dummy.dat", "test", data_files=ds)
        assert len(state["data_files"]) == 2
        assert state["data_files"][0]["label"] == "low-Q"

    def test_data_file_still_set_as_primary(self):
        state = create_initial_state(
            "primary.dat",
            "test",
            data_files=[{"file": "/tmp/a.dat", "label": "a"}],
        )
        assert state["data_file"] == "primary.dat"

    def test_per_file_fit_result_type(self):
        pf = PerFileFitResult(
            file="/tmp/a.dat",
            label="low-Q",
            chi_squared=2.5,
            Q_fit=[0.01, 0.02],
            R_fit=[0.9, 0.8],
            residuals=[0.1, -0.1],
            residual_ratio=[1.01, 0.99],
        )
        assert pf["chi_squared"] == 2.5
        assert pf["label"] == "low-Q"


# ============================================================================
# Model Builder Tests
# ============================================================================


class TestModelBuilderMultiFile:
    """Tests for build_multi_problem and _build_sample."""

    @pytest.fixture(autouse=True)
    def _data_files(self):
        self.files = _make_data_files()
        yield
        _cleanup(self.files)

    def _make_definition(self):
        return {
            "substrate": {
                "name": "silicon",
                "sld": 2.07,
                "roughness": 3.0,
                "roughness_max": 15.0,
            },
            "layers": [
                {
                    "name": "polystyrene",
                    "sld": 1.41,
                    "sld_min": -1.0,
                    "sld_max": 4.0,
                    "thickness": 100.0,
                    "thickness_min": 50.0,
                    "thickness_max": 200.0,
                    "roughness": 5.0,
                    "roughness_max": 20.0,
                }
            ],
            "ambient": {"name": "air", "sld": 0.0},
            "constraints": [],
            "back_reflection": False,
            "data_file": self.files[0],
            "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        }

    def test_build_sample_returns_stack(self):
        from aure.nodes.model_builder import _build_sample

        defn = self._make_definition()
        sample = _build_sample(defn)
        # Should be a refl1d Stack with substrate + 1 layer + ambient = 3 slabs
        assert len(sample) == 3

    def test_build_multi_problem_creates_joint_problem(self):
        from aure.nodes.model_builder import build_multi_problem

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "low-Q"},
            {"file": self.files[1], "label": "mid-Q"},
            {"file": self.files[2], "label": "high-Q"},
        ]
        problem, experiments, sorted_df = build_multi_problem(defn, data_files)

        assert len(experiments) == 3
        # All experiments share the same sample object
        assert experiments[0].sample is experiments[1].sample
        assert experiments[1].sample is experiments[2].sample

    def test_build_multi_problem_independent_probes(self):
        from aure.nodes.model_builder import build_multi_problem

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "low-Q"},
            {"file": self.files[1], "label": "mid-Q"},
        ]
        problem, experiments, _ = build_multi_problem(defn, data_files)

        # Probes should be different objects (independent intensity)
        assert experiments[0].probe is not experiments[1].probe

    def test_build_multi_problem_returns_fit_problem(self):
        from aure.nodes.model_builder import build_multi_problem
        from bumps.fitproblem import FitProblem

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "low-Q"},
            {"file": self.files[1], "label": "mid-Q"},
        ]
        problem, _, _ = build_multi_problem(defn, data_files)
        assert isinstance(problem, FitProblem)

    def test_single_file_build_problem_still_works(self):
        from aure.nodes.model_builder import build_problem
        from bumps.fitproblem import FitProblem

        defn = self._make_definition()
        problem = build_problem(defn)
        assert isinstance(problem, FitProblem)

    def test_shared_parameters_are_identical_objects(self):
        from aure.nodes.model_builder import build_multi_problem

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "a"},
            {"file": self.files[1], "label": "b"},
        ]
        problem, experiments, _ = build_multi_problem(defn, data_files)

        # The thickness parameter of the layer in both experiments should be
        # the exact same Parameter object (shared via shared Sample)
        sample = experiments[0].sample
        layer_thickness = sample[1].thickness
        # Verify it's also the same object in the second experiment
        assert experiments[1].sample[1].thickness is layer_thickness

    def test_intensity_parameters_have_unique_per_file_names(self):
        """Each probe's intensity is named ``"intensity <label>"``.

        Regression: when every probe's intensity kept refl1d's default
        ``"intensity"`` name, the names collided and the Results-page
        sliders (which send the ``"intensity <label>"`` keys produced by
        ``_extract_multi_bumps_results``) could not be matched back, so
        intensity overrides were silently dropped.
        """
        from aure.nodes.model_builder import build_multi_problem

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "low-Q"},
            {"file": self.files[1], "label": "mid-Q"},
            {"file": self.files[2], "label": "high-Q"},
        ]
        problem, _, _ = build_multi_problem(defn, data_files)

        intensity_names = {
            str(p.name) for p in problem._parameters if "intensity" in str(p.name)
        }
        assert intensity_names == {
            "intensity low-Q",
            "intensity mid-Q",
            "intensity high-Q",
        }

    def test_apply_parameters_sets_distinct_intensities(self):
        """``apply_parameters`` resolves the per-file intensity keys and
        sets a *distinct* value on each probe (the Results-page path)."""
        from aure.nodes.model_builder import apply_parameters, build_multi_problem

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "low-Q"},
            {"file": self.files[1], "label": "mid-Q"},
            {"file": self.files[2], "label": "high-Q"},
        ]
        problem, _, _ = build_multi_problem(defn, data_files)

        wanted = {
            "intensity low-Q": 0.80,
            "intensity mid-Q": 0.95,
            "intensity high-Q": 1.05,
        }
        apply_parameters(problem, wanted)

        applied = {
            str(p.name): float(p.value)
            for p in problem._parameters
            if "intensity" in str(p.name)
        }
        assert applied == pytest.approx(wanted)


# ============================================================================
# Prompt Formatting Tests
# ============================================================================


class TestPromptMultiFile:
    """Tests for per-file chi2 formatting in prompts."""

    def test_format_per_file_chi2_none(self):
        from aure.nodes.prompts import _format_per_file_chi2

        result = _format_per_file_chi2(None)
        assert "single-file" in result.lower()

    def test_format_per_file_chi2_empty(self):
        from aure.nodes.prompts import _format_per_file_chi2

        result = _format_per_file_chi2([])
        assert "single-file" in result.lower()

    def test_format_per_file_chi2_with_data(self):
        from aure.nodes.prompts import _format_per_file_chi2

        per_file = [
            {"label": "low-Q", "chi_squared": 1.5},
            {"label": "mid-Q", "chi_squared": 3.2},
            {"label": "high-Q", "chi_squared": 8.1},
        ]
        result = _format_per_file_chi2(per_file)
        assert "low-Q" in result
        assert "mid-Q" in result
        assert "high-Q" in result
        assert "1.500" in result
        assert "8.100" in result
        assert "jointly" in result.lower()

    def test_evaluation_prompt_includes_per_file_section(self):
        from aure.nodes.prompts import format_fit_evaluation_prompt

        per_file = [
            {"label": "segment-1", "chi_squared": 2.0},
            {"label": "segment-2", "chi_squared": 4.0},
        ]
        prompt = format_fit_evaluation_prompt(
            sample_description="test",
            hypothesis=None,
            chi_squared=3.0,
            method="dream",
            converged=True,
            parameters={"Si thickness": 100.0},
            features={},
            per_file_results=per_file,
        )
        assert "segment-1" in prompt
        assert "segment-2" in prompt
        assert "Per-File / Per-State Fit Quality" in prompt
        # No PerFileFitResult carries a state -> the multi-file (spliced
        # Q-segments) wording, not the multi-state one.
        assert "Q-range segments" in prompt

    def test_evaluation_prompt_single_file_shows_not_applicable(self):
        from aure.nodes.prompts import format_fit_evaluation_prompt

        prompt = format_fit_evaluation_prompt(
            sample_description="test",
            hypothesis=None,
            chi_squared=3.0,
            method="dream",
            converged=True,
            parameters={},
            features={},
        )
        assert "single-file" in prompt.lower()


# ============================================================================
# Fitting Multi-File Tests
# ============================================================================


class TestFittingMultiFile:
    """Tests for multi-file fit execution."""

    @pytest.fixture(autouse=True)
    def _data_files(self):
        self.files = _make_data_files()
        yield
        _cleanup(self.files)

    def _make_definition(self):
        return {
            "substrate": {
                "name": "silicon",
                "sld": 2.07,
                "roughness": 3.0,
                "roughness_max": 15.0,
            },
            "layers": [
                {
                    "name": "polystyrene",
                    "sld": 1.41,
                    "sld_min": -1.0,
                    "sld_max": 4.0,
                    "thickness": 100.0,
                    "thickness_min": 50.0,
                    "thickness_max": 200.0,
                    "roughness": 5.0,
                    "roughness_max": 20.0,
                }
            ],
            "ambient": {"name": "air", "sld": 0.0},
            "constraints": [],
            "back_reflection": False,
            "data_file": self.files[0],
            "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        }

    def test_run_multi_refl1d_fit_returns_fit_result(self):
        from aure.nodes.fitting import run_multi_refl1d_fit

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "low-Q"},
            {"file": self.files[1], "label": "mid-Q"},
        ]
        result = run_multi_refl1d_fit(
            model_definition=defn,
            data_files=data_files,
            method="lm",
            iteration=0,
            steps=10,
        )
        assert "chi_squared" in result
        assert result["chi_squared"] > 0
        assert result["per_file_results"] is not None
        assert len(result["per_file_results"]) == 2

    def test_per_file_results_have_labels(self):
        from aure.nodes.fitting import run_multi_refl1d_fit

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "seg-A"},
            {"file": self.files[1], "label": "seg-B"},
        ]
        result = run_multi_refl1d_fit(
            model_definition=defn,
            data_files=data_files,
            method="lm",
            iteration=0,
            steps=10,
        )
        labels = [pf["label"] for pf in result["per_file_results"]]
        assert "seg-A" in labels
        assert "seg-B" in labels

    def test_per_file_results_have_chi2(self):
        from aure.nodes.fitting import run_multi_refl1d_fit

        defn = self._make_definition()
        data_files = [
            {"file": self.files[0], "label": "low"},
            {"file": self.files[1], "label": "mid"},
            {"file": self.files[2], "label": "high"},
        ]
        result = run_multi_refl1d_fit(
            model_definition=defn,
            data_files=data_files,
            method="lm",
            iteration=0,
            steps=10,
        )
        for pf in result["per_file_results"]:
            assert "chi_squared" in pf
            assert pf["chi_squared"] > 0

    def test_single_file_fit_has_no_per_file(self):
        from aure.nodes.fitting import run_refl1d_fit

        defn = self._make_definition()
        result = run_refl1d_fit(
            model_definition=defn,
            method="lm",
            iteration=0,
            steps=10,
        )
        assert result["per_file_results"] is None

    def test_three_file_fit_completes(self):
        from aure.nodes.fitting import run_multi_refl1d_fit

        defn = self._make_definition()
        data_files = [
            {"file": f, "label": f"seg-{i}"} for i, f in enumerate(self.files)
        ]
        result = run_multi_refl1d_fit(
            model_definition=defn,
            data_files=data_files,
            method="lm",
            iteration=0,
            steps=10,
        )
        assert len(result["per_file_results"]) == 3
        assert result["parameters"]  # Should have fitted params


# ============================================================================
# Fitting Node Dispatch Tests
# ============================================================================


class TestFittingNodeDispatch:
    """Tests that fitting_node dispatches correctly to multi-file."""

    @pytest.fixture(autouse=True)
    def _data_files(self):
        self.files = _make_data_files()
        yield
        _cleanup(self.files)

    def _make_state(self, multi: bool = False):
        defn = {
            "substrate": {
                "name": "silicon",
                "sld": 2.07,
                "roughness": 3.0,
                "roughness_max": 15.0,
            },
            "layers": [
                {
                    "name": "polystyrene",
                    "sld": 1.41,
                    "sld_min": -1.0,
                    "sld_max": 4.0,
                    "thickness": 100.0,
                    "thickness_min": 50.0,
                    "thickness_max": 200.0,
                    "roughness": 5.0,
                    "roughness_max": 20.0,
                }
            ],
            "ambient": {"name": "air", "sld": 0.0},
            "constraints": [],
            "back_reflection": False,
            "data_file": self.files[0],
            "intensity": {"value": 1.0, "min": 0.7, "max": 1.1, "fixed": False},
        }

        data_files = []
        if multi:
            data_files = [
                {"file": self.files[0], "label": "low-Q"},
                {"file": self.files[1], "label": "mid-Q"},
                {"file": self.files[2], "label": "high-Q"},
            ]

        return {
            "data_file": self.files[0],
            "data_files": data_files,
            "Q": [0.01, 0.02, 0.05],
            "R": [1.0, 0.9, 0.5],
            "dR": [0.01, 0.01, 0.01],
            "current_model": defn,
            "iteration": 0,
            "output_dir": None,
            "best_chi2": None,
            "best_model": None,
            "best_bic": None,
            "best_bic_model": None,
            "fit_steps": None,
            "fit_burn": None,
        }

    def test_single_file_dispatch(self, monkeypatch):
        """Single-file state should NOT produce per_file_results."""
        monkeypatch.setenv("FIT_METHOD", "lm")
        monkeypatch.setenv("FIT_STEPS", "5")

        from aure.nodes.fitting import fitting_node

        state = self._make_state(multi=False)
        updates = fitting_node(state)
        assert "error" not in updates or updates.get("error") is None
        fit = updates["fit_results"][0]
        assert fit["per_file_results"] is None

    def test_multi_file_dispatch(self, monkeypatch):
        """Multi-file state should produce per_file_results."""
        monkeypatch.setenv("FIT_METHOD", "lm")
        monkeypatch.setenv("FIT_STEPS", "5")

        from aure.nodes.fitting import fitting_node

        state = self._make_state(multi=True)
        updates = fitting_node(state)
        assert "error" not in updates or updates.get("error") is None
        fit = updates["fit_results"][0]
        assert fit["per_file_results"] is not None
        assert len(fit["per_file_results"]) == 3
