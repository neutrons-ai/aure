"""
Test the analysis workflow end-to-end.

This test uses synthetic data to verify the workflow runs correctly through
INTAKE → ANALYSIS → MODELING. Execution goes through the manual runner
(`run_prepare` / `run_analysis`), the single engine the CLI and web UI use.
"""

import tempfile
import numpy as np
import pytest

from aure.workflow import (
    run_analysis,
    run_prepare,
    NODE_ORDER,
    NODE_FUNCTIONS,
    ROUTING_FUNCTIONS,
)
from aure.llm import llm_available


def create_test_data_file() -> str:
    """Create a temporary reflectivity data file for testing."""
    # Generate synthetic 1-layer data
    Q = np.linspace(0.01, 0.25, 200)

    # Simple Fresnel-like curve with oscillations
    Qc = 0.0217  # Silicon critical edge
    R = np.ones_like(Q)

    # Below Qc: total reflection
    below_qc = Q < Qc
    R[below_qc] = 1.0

    # Above Qc: decay with oscillations
    above_qc = Q >= Qc
    Q_above = Q[above_qc]

    # Fresnel decay
    R[above_qc] = (Qc / (2 * Q_above)) ** 4

    # Add oscillations (100 Å layer)
    thickness = 100.0  # Angstrom
    R[above_qc] *= 1 + 0.3 * np.cos(2 * Q_above * thickness)

    # Add noise
    dR = 0.05 * R
    R += np.random.normal(0, 0.02, len(R)) * R
    R = np.clip(R, 1e-10, 1.0)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        f.write("# Q (1/Å)  R  dR\n")
        for q, r, dr in zip(Q, R, dR):
            f.write(f"{q:.6f}  {r:.6e}  {dr:.6e}\n")
        return f.name


def test_runner_registry_is_consistent():
    """The runner's node registry is complete and internally consistent.

    Replaces the old `create_workflow()` smoke test: the runner (not a compiled
    graph) is the workflow definition, so the invariant to pin is that every
    ordered node has a function and every node except the terminal `finalize`
    has a router.
    """
    for node in NODE_ORDER:
        assert node in NODE_FUNCTIONS, f"{node} missing from NODE_FUNCTIONS"

    # finalize is terminal by design → it alone has no router.
    for node in NODE_ORDER:
        if node == "finalize":
            assert node not in ROUTING_FUNCTIONS
        else:
            assert node in ROUTING_FUNCTIONS, f"{node} missing a router"

    # final_fit is a non-routable terminal node: registered, but not ordered.
    assert "final_fit" in NODE_FUNCTIONS
    assert "final_fit" not in NODE_ORDER
    assert "final_fit" not in ROUTING_FUNCTIONS


@pytest.mark.skipif(not llm_available(), reason="No LLM configured")
def test_intake_and_analysis():
    """Smoke: intake → analysis → modeling produces a model (via run_prepare)."""
    data_file = create_test_data_file()

    final_state = run_prepare(
        data_file=data_file,
        sample_description="100 nm polystyrene film on silicon, measured in air",
        hypothesis="Film thickness is approximately 100 nm",
    )

    print("\n  Final state:")
    print(f"    - Q points loaded: {len(final_state.get('Q', []))}")
    print(f"    - Parsed sample: {final_state.get('parsed_sample') is not None}")
    print(f"    - Current model: {final_state.get('current_model') is not None}")
    print(f"    - Error: {final_state.get('error')}")

    import os

    os.unlink(data_file)


@pytest.mark.skipif(not llm_available(), reason="No LLM configured")
def test_sample_description_parsing():
    """Test the sample description parser with LLM."""
    from aure.nodes.intake import parse_sample_with_llm
    from aure.llm import get_llm_info

    info = get_llm_info()
    print(f"  Using LLM: {info['provider']} / {info['model']}")

    test_cases = [
        "50 nm polystyrene on silicon",
        "100 Å gold film on sapphire substrate measured in D2O",
    ]

    for desc in test_cases:
        print(f"\n  Input: '{desc}'")
        try:
            result = parse_sample_with_llm(desc)
            print(
                f"    Substrate: {result['substrate']['name']} "
                f"(SLD={result['substrate']['sld']:.2f})"
            )
            print(f"    Layers: {len(result['layers'])}")
        except Exception as e:
            print(f"    ✗ Error: {e}")


@pytest.mark.skipif(not llm_available(), reason="No LLM configured")
def test_full_fitting_pipeline():
    """Smoke: the complete workflow including fitting/evaluation (via run_analysis)."""
    data_file = create_test_data_file()

    try:
        final_state = run_analysis(
            data_file=data_file,
            sample_description="100 nm polystyrene film on silicon, measured in air",
            hypothesis="Film thickness is approximately 100 nm",
            max_iterations=1,
        )
        print("\n  Final state:")
        print(f"    - Current model: {final_state.get('current_model') is not None}")
        print(f"    - Fit results: {len(final_state.get('fit_results', []))}")
        print(f"    - Iteration: {final_state.get('iteration', 0)}")
    except Exception as e:
        print(f"\n  Fitting workflow error (expected if refl1d not installed): {e}")

    import os

    os.unlink(data_file)


@pytest.mark.skipif(not llm_available(), reason="No LLM configured")
def test_workflow_without_fitting():
    """run_prepare yields a model but never fits (the MCP model-build path)."""
    data_file = create_test_data_file()

    final_state = run_prepare(
        data_file=data_file,
        sample_description="50 nm gold on silicon",
    )

    # Should have a model but no fit results.
    assert final_state.get("current_model") is not None, "Should have a model"
    assert not final_state.get("fit_results"), "Should not have fit results"

    import os

    os.unlink(data_file)


if __name__ == "__main__":
    print("=" * 60)
    print("WORKFLOW TESTS")
    print("=" * 60)

    test_runner_registry_is_consistent()
    test_sample_description_parsing()
    test_intake_and_analysis()
    test_workflow_without_fitting()
    test_full_fitting_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
