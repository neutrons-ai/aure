"""Posterior diagnostics and evidence from a sampler draw (docs/evidence.md).

The estimators are checked against posteriors whose evidence is known in closed
form, and the trust rules against the failure modes they exist to catch:
chains that never mixed, a population drifting together, a best point that is
not where the mass is, and a box that truncates the posterior.
"""

import json
import math

import numpy as np
import pytest

from aure.tools.posterior_tools import (
    POSTERIOR_FORMULA,
    chains_from_ids,
    compare_evidence,
    effective_size,
    evidence_baseline_current,
    evidence_record,
    evidence_usable,
    gelman_rubin,
    locate_point,
    occam_factors,
    posterior_digest,
    posterior_for_report,
    prior_width_adjustment,
    split_rhat,
    summarize_posterior,
)


def _flatten(chains):
    """(G, C, k) → generation-major draws plus the chain id of each."""
    g, c, k = chains.shape
    return chains.reshape(g * c, k), np.tile(np.arange(c), g)


class _Gaussian:
    """ℓ(θ) = ℓ̂ − ½ (θ−μ)ᵀ Σ⁻¹ (θ−μ) in a box; exact evidence when the box is wide."""

    def __init__(self, k=4, seed=1, ell_max=-50.0, half_width=40.0):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(k, k))
        self.cov = a @ a.T / k + 0.2 * np.eye(k)
        self.mu = np.linspace(-1.0, 2.0, k)
        self.lo, self.hi = self.mu - half_width, self.mu + half_width
        self.icov = np.linalg.inv(self.cov)
        self.ell_max = ell_max
        self.k = k

    def ell(self, x):
        d = np.atleast_2d(x) - self.mu
        return self.ell_max - 0.5 * np.einsum("ij,jk,ik->i", d, self.icov, d)

    def exact(self):
        return (
            self.ell_max
            + 0.5 * math.log(np.linalg.det(2 * math.pi * self.cov))
            - float(np.sum(np.log(self.hi - self.lo)))
        )

    def chains(self, generations=250, chains=40, seed=2):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((generations, chains, self.k))
        return self.mu + z @ np.linalg.cholesky(self.cov).T

    def summary(self, chains=None, best=None, **kw):
        chains = self.chains() if chains is None else chains
        x, ids = _flatten(chains)
        kw.setdefault("log_density", self.ell)
        return summarize_posterior(
            x,
            self.ell(x),
            best_x=self.mu if best is None else best,
            best_logp=float(self.ell(self.mu if best is None else best)[0]),
            lower=self.lo,
            upper=self.hi,
            labels=[f"p{i}" for i in range(self.k)],
            n_data=200,
            chain_ids=ids,
            **kw,
        )


# ----------------------------------------------------------------------
# The estimators
# ----------------------------------------------------------------------


def test_both_estimators_recover_a_gaussian_evidence():
    g = _Gaussian()
    s = g.summary()
    exact = g.exact()
    assert s["log_evidence_laplace"] == pytest.approx(exact, abs=0.1)
    err = s["log_evidence_bridge_error"]
    assert 0 < err < 0.05
    assert s["log_evidence_bridge"] == pytest.approx(exact, abs=max(4 * err, 0.02))
    assert s["log_evidence"] == s["log_evidence_bridge"]
    assert s["log_evidence_method"] == "bridge"


def test_gaussian_draw_is_trusted_and_central():
    s = _Gaussian().summary()
    assert s["checks"]["converged"] is True
    assert s["checks"]["best_central"] is True
    assert s["fit_trusted"] is True and s["evidence_trusted"] is True
    assert s["best_mahalanobis2_per_param"] < 0.01
    # p_D and d̃ both count the constrained parameters: k for a Gaussian.
    assert s["p_d"] == pytest.approx(4, abs=0.3)
    assert s["d_bayes"] == pytest.approx(4, abs=0.5)
    assert s["concerns"] == []


def test_occam_decomposition_is_an_identity():
    """O = Σ o_j + o_R = ln V − ½ ln|2πΣ̂|, and BIC's error is O − (k/2) ln n."""
    g = _Gaussian()
    s = g.summary()
    sigma = np.array(list(s["std"].values()))
    widths = g.hi - g.lo
    assert list(s["occam"].values()) == pytest.approx(
        list(occam_factors(sigma, widths))
    )
    assert s["occam_correlation"] >= 0
    occam_total = s["occam_total"]
    assert s["log_evidence_laplace"] == pytest.approx(
        s["log_likelihood_best"] - occam_total
    )
    k, n = g.k, 200
    assert s["bic_error_laplace"] == pytest.approx(occam_total - 0.5 * k * math.log(n))
    assert s["log_evidence_bic"] == pytest.approx(g.ell_max - 0.5 * k * math.log(n))


def test_uncorrelated_parameters_carry_no_correlation_penalty():
    g = _Gaussian()
    g.cov = np.diag([0.5, 1.0, 2.0, 0.1])
    g.icov = np.linalg.inv(g.cov)
    s = g.summary()
    assert s["occam_correlation"] == pytest.approx(0.0, abs=0.02)


def test_truncation_biases_laplace_but_not_bridge():
    """A half-Gaussian against its bound: Laplace is off by ln(2√(1−2/π)) ≈ 0.19."""
    rng = np.random.default_rng(3)
    k, sig = 2, np.array([1.0, 0.5])
    lo, hi = np.array([0.0, -20.0]), np.array([30.0, 20.0])
    mu = np.array([0.0, 0.0])  # the mode of parameter 0 sits on its lower bound

    def ell(x):
        x = np.atleast_2d(x)
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=1)

    z = rng.standard_normal((300, 40, k)) * sig
    z[..., 0] = np.abs(z[..., 0])
    x, ids = _flatten(z)
    s = summarize_posterior(
        x,
        ell(x),
        best_x=mu,
        best_logp=0.0,
        lower=lo,
        upper=hi,
        labels=["rough", "other"],
        n_data=100,
        chain_ids=ids,
        log_density=ell,
    )
    exact = (
        math.log(0.5 * math.sqrt(2 * math.pi) * 1.0)
        + math.log(math.sqrt(2 * math.pi) * 0.5)
        - float(np.sum(np.log(hi - lo)))
    )
    assert s["log_evidence_bridge"] == pytest.approx(exact, abs=0.05)
    assert s["log_evidence_laplace"] - exact == pytest.approx(
        math.log(2 * math.sqrt(1 - 2 / math.pi)), abs=0.05
    )
    assert s["truncated_by_bounds"] == ["rough"]
    assert any("truncates" in c for c in s["concerns"])
    # Pressed against a bound is not "away from the bulk": not an extreme.
    assert s["checks"]["best_extreme_percentile"] == []


def test_bridge_error_grows_when_the_proposal_cannot_cover_two_modes():
    """One Gaussian over two well-separated modes overlaps badly; RE says so."""
    rng = np.random.default_rng(4)
    k = 2
    lo, hi = np.full(k, -30.0), np.full(k, 30.0)
    centers = np.array([[-6.0, 0.0], [6.0, 0.0]])

    def ell(x):
        x = np.atleast_2d(x)
        d0 = np.sum((x - centers[0]) ** 2, axis=1)
        d1 = np.sum((x - centers[1]) ** 2, axis=1)
        return np.logaddexp(-0.5 * d0, -0.5 * d1) - math.log(2)

    # Half the chains in each mode — a sampler that never crossed over.
    z = rng.standard_normal((200, 40, k))
    z[:, :20] += centers[0]
    z[:, 20:] += centers[1]
    x, ids = _flatten(z)
    s = summarize_posterior(
        x,
        ell(x),
        best_x=centers[0],
        best_logp=float(ell(centers[0])[0]),
        lower=lo,
        upper=hi,
        labels=["a", "b"],
        n_data=100,
        chain_ids=ids,
        log_density=ell,
    )
    exact = math.log(2 * math.pi) - float(np.sum(np.log(hi - lo)))
    # Either the bridge still lands, or its own error says it did not; never a
    # confident wrong answer.
    if (
        s["log_evidence_bridge_error"] is not None
        and s["log_evidence_bridge_error"] < 0.5
    ):
        assert s["log_evidence_bridge"] == pytest.approx(exact, abs=0.5)
    # The chains in different modes never mixed, and the best point is not
    # where the mass is (the mean sits between the modes).
    assert s["checks"]["converged"] is False
    assert s["fit_trusted"] is False and s["evidence_trusted"] is False


# ----------------------------------------------------------------------
# Convergence
# ----------------------------------------------------------------------


def test_split_rhat_is_one_for_mixed_chains():
    chains = _Gaussian().chains()
    assert np.all(split_rhat(chains) < 1.05)


def test_chains_stuck_in_different_places_fail_rhat():
    g = _Gaussian()
    chains = g.chains()
    offsets = np.random.default_rng(5).normal(scale=3.0, size=(1, chains.shape[1], g.k))
    stuck = 0.05 * (chains - g.mu) + g.mu + offsets
    s = g.summary(chains=stuck)
    assert s["checks"]["rhat_ok"] is False
    assert s["evidence_trusted"] is False and s["fit_trusted"] is False
    assert any("did not mix" in c for c in s["concerns"])


def test_split_rhat_catches_a_population_drifting_together():
    """Plain R̂ cannot see a common drift; the split form can."""
    g = _Gaussian()
    chains = g.chains()
    drift = np.linspace(0, 6, chains.shape[0])[:, None, None]
    drifting = chains + drift
    assert np.max(gelman_rubin(drifting)) < 1.2
    assert np.max(split_rhat(drifting)) > 1.2


def test_without_chain_labels_there_is_no_convergence_verdict():
    g = _Gaussian()
    x, _ = _flatten(g.chains())
    s = summarize_posterior(
        x,
        g.ell(x),
        best_x=g.mu,
        best_logp=g.ell_max,
        lower=g.lo,
        upper=g.hi,
        labels=list("abcd"),
        n_data=200,
        log_density=g.ell,
    )
    assert s["checks"]["converged"] is None
    assert s["evidence_trusted"] is False and s["fit_trusted"] is False
    assert any("No convergence diagnostic" in c for c in s["concerns"])


def test_too_few_distinct_points_is_not_converged():
    g = _Gaussian()
    chains = g.chains(generations=300, chains=40)
    # A sampler that accepts nothing repeats its points: 40 distinct draws,
    # fewer than the max(50, 10(k + 1)) a k = 4 posterior needs.
    frozen = np.repeat(chains[:1], 300, axis=0)
    s = g.summary(chains=frozen)
    assert s["checks"]["enough_unique_draws"] is False
    assert s["checks"]["converged"] is False


# ----------------------------------------------------------------------
# Is the best point representative?
# ----------------------------------------------------------------------


def test_a_best_point_away_from_the_mass_is_not_central():
    g = _Gaussian()
    best = g.mu + 3.0 * np.sqrt(np.diag(g.cov))
    s = g.summary(best=best)
    assert s["best_mahalanobis2_per_param"] > 1.0
    assert s["checks"]["best_central"] is False
    assert s["fit_trusted"] is False
    # The evidence does not depend on the best point, only the fit verdict does.
    assert s["evidence_trusted"] is True
    assert any("not where the posterior mass is" in c for c in s["concerns"])


def test_an_interior_extreme_percentile_is_flagged():
    g = _Gaussian()
    best = g.mu.copy()
    best[1] += 3.5 * math.sqrt(g.cov[1, 1])
    s = g.summary(best=best)
    assert "p1" in s["checks"]["best_extreme_percentile"]
    assert s["best_percentile"]["p1"] > 0.99


# ----------------------------------------------------------------------
# Degenerate inputs never raise
# ----------------------------------------------------------------------


def test_a_parameter_that_never_moved_disables_the_covariance_quantities():
    g = _Gaussian()
    chains = g.chains()
    chains[..., 2] = g.mu[2]
    s = g.summary(chains=chains)
    assert s["log_evidence_laplace"] is None
    assert s["best_mahalanobis2"] is None
    assert any("never moved" in c for c in s["concerns"])
    json.dumps(s, allow_nan=False)


def test_a_non_uniform_prior_gets_no_evidence():
    s = _Gaussian().summary(uniform_prior=False)
    assert s["log_evidence"] is None and s["log_evidence_bridge"] is None
    assert any("uniform box prior" in c for c in s["concerns"])


def test_a_failing_likelihood_leaves_the_rest_of_the_summary():
    def boom(_x):
        raise RuntimeError("model blew up")

    s = _Gaussian().summary(log_density=boom)
    assert s["log_evidence_bridge"] is None
    assert s["log_evidence_method"] == "laplace"
    assert s["evidence_trusted"] is False
    assert s["log_evidence_laplace"] is not None


def test_an_empty_draw_is_reported_not_raised():
    s = summarize_posterior(
        np.zeros((0, 2)),
        np.zeros(0),
        best_x=[0, 0],
        best_logp=0.0,
        lower=[0, 0],
        upper=[1, 1],
        labels=["a", "b"],
        n_data=10,
    )
    assert s["fit_trusted"] is False and s["evidence_trusted"] is False
    json.dumps(s, allow_nan=False)


def test_the_summary_is_strict_json():
    s = _Gaussian().summary()
    text = json.dumps(s, allow_nan=False)
    assert json.loads(text)["formula"] == POSTERIOR_FORMULA


def test_no_bridge_without_proposals():
    s = _Gaussian().summary(n_proposal=0)
    assert s["log_evidence_bridge"] is None
    assert s["evidence_trusted"] is False


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def test_effective_size_is_n_for_independent_draws_and_less_for_sticky_ones():
    rng = np.random.default_rng(6)
    iid = rng.standard_normal((400, 20))
    x, ids = iid.reshape(-1), np.tile(np.arange(20), 400)
    assert effective_size(x, ids) == pytest.approx(len(x), rel=0.35)
    sticky = np.repeat(rng.standard_normal((40, 20)), 10, axis=0).reshape(-1)
    assert effective_size(sticky, ids) < 0.3 * len(sticky)


def test_chains_from_ids_restacks_generation_major_draws():
    chains = np.arange(5 * 3 * 2, dtype=float).reshape(5, 3, 2)
    x, ids = _flatten(chains)
    assert np.array_equal(chains_from_ids(x, ids), chains)


def _record(lnz, err=0.1, k=8, bounds=None, it=1, index=0, bic=None):
    return {
        "formula": POSTERIOR_FORMULA,
        "log_evidence": lnz,
        "log_evidence_error": err,
        "log_evidence_bic": bic,
        "chi_squared": 1.0,
        "iteration": it,
        "fit_index": index,
        "n_free_params": k,
        "bounds": bounds or {},
    }


@pytest.mark.parametrize(
    "delta, k_new, verdict, revert, becomes",
    [
        (+3.0, 11, "better", False, True),
        (-3.0, 8, "worse", True, False),
        (-0.5, 11, "inconclusive", True, False),  # more complex, data cannot tell
        (-0.5, 5, "inconclusive", False, True),  # simpler wins the tie
        (+0.5, 8, "inconclusive", False, True),  # same k, higher
        (-0.5, 8, "inconclusive", False, False),  # same k, lower: keep both as they are
    ],
)
def test_compare_evidence_verdicts_and_the_parsimony_band(
    delta, k_new, verdict, revert, becomes
):
    out = compare_evidence(
        _record(-90.0 + delta, k=k_new), _record(-90.0, k=8), margin=1.0
    )
    assert out["verdict"] == verdict
    assert out["revert"] is revert
    assert out["becomes_baseline"] is becomes


def test_revert_and_becoming_the_baseline_are_never_both_true():
    for delta in np.linspace(-4, 4, 33):
        for k_new in (5, 8, 11):
            out = compare_evidence(_record(delta, k=k_new), _record(0.0, k=8))
            assert not (out["revert"] and out["becomes_baseline"])


def test_the_margin_widens_with_the_estimate_error():
    out = compare_evidence(_record(-88.5, err=0.6), _record(-90.0, err=0.6), margin=1.0)
    assert out["threshold"] == pytest.approx(2 * math.hypot(0.6, 0.6))
    assert out["verdict"] == "inconclusive"


def test_shared_parameter_boxes_are_compared_on_a_common_prior():
    base = {"t": [0, 50], "rho": [0, 6]}
    wider = {"t": [0, 100], "rho": [0, 6], "x thickness": [0, 40]}
    assert prior_width_adjustment(wider, base) == pytest.approx(math.log(2))
    # Widening a box the posterior does not touch costs ln 2 raw, nothing adjusted.
    out = compare_evidence(
        _record(-90.0 - math.log(2), bounds=wider), _record(-90.0, bounds=base)
    )
    assert out["delta_log_evidence"] == pytest.approx(0.0)


def test_compare_records_what_bic_alone_would_have_said():
    out = compare_evidence(
        _record(-88.0, k=8, bic=-85.0), _record(-90.0, k=5, bic=-80.0), margin=1.0
    )
    assert out["verdict"] == "better"
    assert out["bic_verdict"] == "worse"
    assert out["bic_agrees"] is False


def test_evidence_usable_requires_the_current_formula_and_a_trusted_verdict():
    post = {
        "formula": POSTERIOR_FORMULA,
        "evidence_trusted": True,
        "log_evidence": -9.0,
    }
    assert evidence_usable(post)
    assert not evidence_usable({**post, "formula": "old"})
    assert not evidence_usable({**post, "evidence_trusted": False})
    assert not evidence_usable({**post, "log_evidence": None})
    assert not evidence_usable(None)


def test_evidence_record_round_trip():
    fit = {
        "chi_squared": 1.1,
        "iteration": 3,
        "bounds": {"a": [0, 1]},
        "posterior": {
            "formula": POSTERIOR_FORMULA,
            "log_evidence": -9.0,
            "log_evidence_error": 0.2,
            "log_evidence_bic": -7.0,
            "k": 4,
        },
    }
    rec = evidence_record(fit, fit_index=5)
    assert rec["fit_index"] == 5 and rec["n_free_params"] == 4
    assert evidence_baseline_current(rec)
    assert not evidence_baseline_current({**rec, "formula": "old"})


def test_locate_point_places_a_point_in_a_stored_posterior():
    g = _Gaussian()
    s = g.summary()
    centre = locate_point(s, dict(zip(s["mean"], g.mu)), chi2_total=-2 * g.ell_max)
    assert centre["mahalanobis2"] < 0.01
    assert all(0.4 < p < 0.6 for p in centre["percentiles"].values())
    assert centre["share_of_draws_with_lower_chi2"] == 0.0
    far = locate_point(
        s, dict(zip(s["mean"], g.mu + 5.0)), chi2_total=-2 * g.ell_max + 50
    )
    assert far["mahalanobis2_per_param"] > 1.0
    assert far["share_of_draws_with_lower_chi2"] == 1.0
    assert locate_point(s, {"missing": 1.0}) is None


def test_digest_and_report_resolution():
    post = _Gaussian().summary()
    digest = posterior_digest(post)
    assert digest["fit_trusted"] is True and digest["k"] == 4
    explore = {"chi_squared": 1.0, "posterior": {"formula": "x"}}
    final = {"chi_squared": 1.0, "posterior": post}
    state = {
        "fit_results": [explore, final],
        "final_selection": {"selected": True, "index": 0},
        "final_fit": {"ran": True},
    }
    assert posterior_for_report(state) == (post, "final_fit")
    state["final_fit"] = {"ran": False}
    assert posterior_for_report(state) == ({"formula": "x"}, "selected")
    assert posterior_for_report({"fit_results": []}) == (None, None)
