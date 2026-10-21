"""Posterior diagnostics and Bayesian evidence from a DREAM draw.

BIC scores a model by its best point alone and charges every free parameter the
same ``½·ln n``. That is the large-sample limit of the Bayesian evidence for a
unimodal, Gaussian posterior under a "unit-information" prior, and it is only as
good as the optimum it is evaluated at. A DREAM fit leaves a draw from the
posterior behind, which measures what BIC assumes:

* how much of each parameter's range the data leave open (its Occam factor),
  and therefore how far BIC's flat ``½·ln n`` per parameter is off;
* whether the best point sits where the posterior mass is, or is a spike, an
  edge, or one basin of several;
* whether the parameter box truncates the posterior;
* whether the sampler converged at all.

:func:`summarize_posterior` turns a draw into a JSON-safe summary, carried on a
``FitResult`` as ``posterior``, with two evidence estimates:

* **Laplace with the sampled covariance**, ``ℓ̂ + ½·ln|2πΣ̂| − ln V``: free, and
  exact only for a Gaussian posterior well inside its box;
* **bridge sampling** (Meng & Wong 1996) with a Gaussian proposal fitted to half
  of the chains: consistent for any posterior shape the sampler explored, at the
  cost of ``n_proposal`` extra likelihood evaluations, and it reports its error.

Conventions. ``ℓ(θ) = −nllf(θ) = −½·χ²(θ)`` inside the box, which is exactly
bumps' DREAM ``logp``. Priors are the parameters' uniform boxes, so
``ln π(θ) = −ln V``. The data-only constant ``−Σ ln(√(2π)·dR)`` is dropped, as
BIC drops it, so values compare across models of the same data and never across
data sets. All logarithms are natural; evidence values are in nats.

``docs/evidence.md`` has the derivations, the trust rules and the validation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Marker for the convention a stored evidence was computed under (cf.
#: ``evaluation.BIC_FORMULA``). Bump it when a definition here changes, so a
#: baseline carried across the change is discarded instead of compared.
POSTERIOR_FORMULA = "box-prior-bridge-v1"

#: Proposal draws for bridge sampling; each costs one likelihood evaluation.
DEFAULT_PROPOSALS = 2000
#: Gelman-Rubin ceiling for "the chains mixed" (Vrugt's DREAM convention).
DEFAULT_RHAT_MAX = 1.2
#: Largest bridge-sampling error, in nats, at which an evidence is compared.
DEFAULT_MAX_ERROR = 0.5
#: Smallest log-evidence difference, in nats, that decides a comparison.
#: ``ln B = 1`` is the bottom of "positive" on the Kass & Raftery (1995) scale.
DEFAULT_MARGIN = 1.0

#: Marginal quantiles stored per parameter (enough to place a point later).
QUANTILE_LEVELS = (0.005, 0.025, 0.16, 0.5, 0.84, 0.975, 0.995)
#: χ² quantiles of the draw, for placing another fit's χ² in the posterior.
CHI2_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

_BOUND_BAND = 0.01  # "at a bound" means within 1 % of the range of it
_TRUNCATION_MASS = 0.05  # more posterior mass than this at a bound truncates it
_CENTRAL_D2_PER_PARAM = 1.0  # best point farther than a typical draw is not central
_EXTREME_PERCENTILE = 0.01  # best value in the outer 1 % of an interior marginal
_GAUSSIAN_GAP = 1.0  # nats between Laplace and bridge before "not Gaussian"
_WEAK_OCCAM = 1.0  # nats: a parameter the data barely constrain
_UNCONSTRAINED_PARAMS = 2.0  # k − p_D at or above this: parameters the data do not see


def _finite(value: Any) -> Optional[float]:
    """``float(value)`` when finite, else None — the JSON-safe spelling of 'no number'."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _per_name(
    names: Sequence[str], values: Sequence[Any]
) -> Dict[str, Optional[float]]:
    return {n: _finite(v) for n, v in zip(names, values)}


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def gelman_rubin(chains: np.ndarray) -> Optional[np.ndarray]:
    """Gelman-Rubin R̂ per variable for chains shaped ``(generations, chains, k)``.

    Same estimator as ``bumps.dream.gelman``. A variable whose chains never move
    has no within-chain variance and gets ``inf``: a stuck sampler is not a
    converged one.
    """
    chains = np.asarray(chains, float)
    if chains.ndim != 3:
        return None
    n, m, _ = chains.shape
    if n < 2 or m < 2:
        return None
    b_over_n = chains.mean(axis=0).var(axis=0, ddof=1)
    w = chains.var(axis=0, ddof=1).mean(axis=0)
    sigma2 = (n - 1) / n * w + b_over_n
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = (m + 1) / m * sigma2 / w - (n - 1) / (m * n)
    return np.sqrt(np.where(np.isfinite(r2) & (r2 > 0), r2, np.inf))


def split_rhat(chains: np.ndarray) -> Optional[np.ndarray]:
    """Split-R̂: :func:`gelman_rubin` with every chain cut into two halves.

    Plain R̂ cannot see a population that drifts together — every chain moves
    the same way, so the chain means agree while each chain's variance absorbs
    the drift. Comparing first halves with second halves catches it
    (Gelman et al., *Bayesian Data Analysis*, 3rd ed., §11.4), which is why this
    is the single convergence diagnostic used here.
    """
    chains = np.asarray(chains, float)
    if chains.ndim != 3 or chains.shape[0] < 4:
        return None
    half = chains.shape[0] // 2
    return gelman_rubin(
        np.concatenate([chains[:half], chains[half : 2 * half]], axis=1)
    )


def chains_from_ids(
    values: np.ndarray, chain_ids: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Re-stack draws ``(N, …)`` into ``(generations, chains, …)`` by chain id.

    Draws of one chain must appear in generation order. Chains are cut to the
    shortest one's length, so the result is rectangular.
    """
    if chain_ids is None:
        return None
    values = np.asarray(values, float)
    ids = np.unique(chain_ids)
    rows = [values[chain_ids == c] for c in ids]
    length = min(len(r) for r in rows) if rows else 0
    if len(ids) < 2 or length < 4:
        return None
    return np.stack([r[:length] for r in rows], axis=1)


def occam_factors(sigma: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Marginal Occam penalty per parameter, ``ln(w / (√(2π)·σ))``, in nats.

    The factor by which the data shrink a uniform prior of width *w* to a
    Gaussian posterior of standard deviation σ. BIC charges ``½·ln n`` for every
    parameter instead; the difference is BIC's error for that parameter.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(widths) - 0.5 * math.log(2 * math.pi) - np.log(sigma)


def _chain_means(values: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(chain_ids, return_inverse=True)
    counts = np.bincount(inverse)
    return np.bincount(inverse, weights=values) / counts


def effective_size(values: np.ndarray, chain_ids: Optional[np.ndarray]) -> float:
    """Effective sample size of *values* from between-chain batch means.

    ``Var(mean) ≈ Var_c(chain means) / m`` for *m* chains of equal length, so the
    iid-equivalent size is ``Var(values)·m / Var_c(chain means)``, clipped to
    ``[m, n]``. Without chain labels every draw counts as independent.
    """
    values = np.asarray(values, float)
    n = len(values)
    if chain_ids is None or n < 2:
        return float(n)
    means = _chain_means(values, chain_ids)
    m = len(means)
    if m < 2:
        return float(n)
    var_means = float(np.var(means, ddof=1))
    var_all = float(np.var(values, ddof=1))
    if var_means <= 0 or var_all <= 0:
        return float(n)
    return float(min(max(var_all * m / var_means, m), n))


def bridge_log_evidence(
    points: np.ndarray,
    logp: np.ndarray,
    log_density: Callable[[np.ndarray], np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    chain_ids: Optional[np.ndarray] = None,
    n_proposal: int = DEFAULT_PROPOSALS,
    rng: Optional[np.random.Generator] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Optional[Dict[str, Any]]:
    """``ln ∫_box exp(ℓ(θ)) dθ`` by iterative bridge sampling, with its error.

    Meng & Wong (1996), in the form of Gronau et al. (2017):

    * half of the chains fit the proposal ``q = N(μ, Σ)``; the other half are
      the posterior draws the estimator uses, so ``q`` is not tuned to the very
      points it is scored on;
    * ``n_proposal`` draws from ``q`` are evaluated with *log_density* (the only
      new likelihood calls; a draw outside the box has ``ℓ = −∞``);
    * the optimal bridge is iterated to a fixed point;
    * the relative error is Frühwirth-Schnatter's (2004) estimate, with the
      posterior term's autocorrelation taken from between-chain batch means.

    Returns ``{log_integral, error, ...}`` or None when the proposal cannot be
    built or no proposal lands where the likelihood is finite.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    X = np.asarray(points, float)
    lp = np.asarray(logp, float)
    n, k = X.shape
    if n < 4 or n_proposal <= 0:
        return None

    # Split by chain so the proposal and the estimate use different chains.
    if chain_ids is not None and len(np.unique(chain_ids)) >= 4:
        ids = np.unique(chain_ids)
        fit_mask = np.isin(chain_ids, ids[0::2])
    else:
        fit_mask = np.zeros(n, bool)
        fit_mask[rng.permutation(n)[: n // 2]] = True
    Xf, Xe, lpe = X[fit_mask], X[~fit_mask], lp[~fit_mask]
    ce = chain_ids[~fit_mask] if chain_ids is not None else None
    if len(Xf) <= k or len(Xe) < 2:
        return None

    mu = Xf.mean(axis=0)
    cov = np.atleast_2d(np.cov(Xf.T, ddof=1))
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return None
    half_logdet = float(np.sum(np.log(np.diag(L))))
    norm = 0.5 * k * math.log(2 * math.pi) + half_logdet

    def log_q(Y: np.ndarray) -> np.ndarray:
        z = np.linalg.solve(L, (Y - mu).T)
        return -0.5 * np.sum(z * z, axis=0) - norm

    Y = mu + rng.standard_normal((n_proposal, k)) @ L.T
    inside = np.all((Y >= lower) & (Y <= upper), axis=1)
    lp_prop = np.full(n_proposal, -np.inf)
    if inside.any():
        values = np.asarray(log_density(Y[inside]), float)
        lp_prop[inside] = np.where(np.isfinite(values), values, -np.inf)
    if not np.isfinite(lp_prop).any():
        return None

    l1 = lpe - log_q(Xe)  # posterior draws: ln(p̃/q)
    l2 = lp_prop - log_q(Y)  # proposal draws: ln(p̃/q), −∞ where p̃ = 0
    n1_eff = effective_size(lpe, ce)
    s1 = n1_eff / (n1_eff + n_proposal)
    s2 = n_proposal / (n1_eff + n_proposal)
    lstar = float(np.median(l1))  # rescaling constant; cancels in the result

    # Written so that nothing overflows: exp(+big) only ever lands in a
    # denominator, and 1/inf is the right limit.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        e1 = np.exp(l1 - lstar)
        inv_e2 = np.exp(-(l2 - lstar))
        r = 1.0
        iterations = 0
        converged = False
        for iterations in range(1, max_iter + 1):
            num = 1.0 / (s1 + s2 * r * inv_e2)  # f1 on the proposals
            den = 1.0 / (s1 * e1 + s2 * r)  # ∝ f2 on the posterior draws
            r_new = float(np.mean(num) / np.mean(den))
            if not (math.isfinite(r_new) and r_new > 0):
                return None
            if abs(math.log(r_new) - math.log(r)) < tol:
                r = r_new
                converged = True
                break
            r = r_new
        num = 1.0 / (s1 + s2 * r * inv_e2)
        den = 1.0 / (s1 * e1 + s2 * r)

    # Relative mean-square error (Frühwirth-Schnatter 2004, eq. 17 of Gronau et
    # al. 2017). Proposal draws are iid; posterior draws are not, so their term
    # uses the variance of chain means rather than of single draws.
    mean_num, mean_den = float(np.mean(num)), float(np.mean(den))
    if mean_num <= 0 or mean_den <= 0:
        return None
    re2 = float(np.var(num, ddof=1)) / (n_proposal * mean_num**2)
    if ce is not None and len(np.unique(ce)) >= 2:
        cm = _chain_means(den, ce)
        re2 += float(np.var(cm, ddof=1)) / (len(cm) * mean_den**2)
    else:
        re2 += float(np.var(den, ddof=1)) / (max(n1_eff, 1.0) * mean_den**2)

    return {
        "log_integral": math.log(r) + lstar,
        "error": math.sqrt(re2),
        "n_posterior": int(len(Xe)),
        "n_posterior_effective": float(n1_eff),
        "n_proposal": int(n_proposal),
        "proposal_in_box": float(np.mean(inside)),
        "iterations": int(iterations),
        "converged": bool(converged),
    }


# ----------------------------------------------------------------------
# The summary
# ----------------------------------------------------------------------


def summarize_posterior(
    points: np.ndarray,
    logp: np.ndarray,
    *,
    best_x: Sequence[float],
    best_logp: float,
    lower: Sequence[float],
    upper: Sequence[float],
    labels: Sequence[str],
    n_data: int,
    chain_ids: Optional[np.ndarray] = None,
    log_density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    uniform_prior: bool = True,
    n_proposal: int = DEFAULT_PROPOSALS,
    rhat_max: float = DEFAULT_RHAT_MAX,
    max_error: float = DEFAULT_MAX_ERROR,
    method: str = "dream",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Summarize a posterior draw: diagnostics, Occam factors, evidence, verdicts.

    *points* ``(N, k)`` and *logp* ``(N,)`` are the draw, with ``logp = ℓ(θ)``.
    *best_x*/*best_logp* are the best point the sampler saw (it may predate the
    retained draw). *lower*/*upper* are the boxes, i.e. the uniform prior.
    *chain_ids* labels each draw's chain — each chain's draws in generation
    order — for split-R̂, batch-mean errors and the bridge split. Without it
    there is no convergence diagnostic, which is never read as a pass.
    *log_density* evaluates ``ℓ`` on an ``(m, k)`` array and enables bridge
    sampling.

    Every field is a plain Python value (None for "no number"), so the result
    can go straight into a checkpoint. Never raises for a degenerate draw: the
    affected fields are None and the reason is listed in ``concerns``.
    """
    X = np.asarray(points, float)
    lp = np.asarray(logp, float)
    cid = None if chain_ids is None else np.asarray(chain_ids)
    keep = np.isfinite(lp) & np.all(np.isfinite(X), axis=1)
    X, lp = X[keep], lp[keep]
    if cid is not None:
        cid = cid[keep]
    n_draws, k = X.shape
    names = [str(s) for s in labels]
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    width = hi - lo
    best = np.asarray(best_x, float)
    best_logp = float(best_logp)
    concerns: List[str] = []

    n_unique = int(len(np.unique(X, axis=0))) if n_draws else 0
    n_chains = int(len(np.unique(cid))) if cid is not None else None
    min_unique = max(50, 10 * (k + 1))
    # Split-R̂ on every parameter and on ℓ itself: a likelihood still climbing is
    # a draw still burning in, whatever the parameters look like.
    stacked = chains_from_ids(np.column_stack([X, lp]), cid) if n_draws else None
    rhat_all = split_rhat(stacked) if stacked is not None else None
    rhat_arr = rhat_all[:k] if rhat_all is not None else None
    rhat_logp = float(rhat_all[k]) if rhat_all is not None else None
    rhat_max_seen = float(np.max(rhat_all)) if rhat_all is not None else None

    out: Dict[str, Any] = {
        "formula": POSTERIOR_FORMULA,
        "method": method,
        "k": int(k),
        "n_data": int(n_data),
        "draw": {
            "n_draws": int(n_draws),
            "n_unique": n_unique,
            "n_chains": n_chains,
            "n_generations": int(stacked.shape[0]) if stacked is not None else None,
            "rhat_max": _finite(rhat_max_seen),
            "rhat_max_is_infinite": bool(
                rhat_max_seen is not None and math.isinf(rhat_max_seen)
            ),
            "rhat": _per_name(names, rhat_arr) if rhat_arr is not None else None,
            "rhat_logp": _finite(rhat_logp),
        },
        "thresholds": {
            "rhat_max": float(rhat_max),
            "max_error": float(max_error),
            "n_unique_min": int(min_unique),
        },
        "concerns": concerns,
    }

    box_ok = bool(k) and bool(np.all(np.isfinite(width) & (width > 0)))
    log_v = float(np.sum(np.log(width))) if box_ok else None
    out["log_prior_volume"] = _finite(log_v)
    out["log_likelihood_best"] = _finite(best_logp)
    out["chi2_total_best"] = _finite(-2.0 * best_logp)
    out["log_evidence_bic"] = (
        _finite(best_logp - 0.5 * k * math.log(n_data)) if n_data > 0 else None
    )

    if n_draws < 2 or k == 0:
        concerns.append("The posterior draw is empty or has no free parameters.")
        out.update(fit_trusted=False, evidence_trusted=False, log_evidence=None)
        return out

    # ---- Moments and marginals ------------------------------------------
    mu = X.mean(axis=0)
    cov = np.atleast_2d(np.cov(X.T, ddof=1))
    sigma = np.sqrt(np.clip(np.diag(cov), 0, None))
    quant = np.quantile(X, QUANTILE_LEVELS, axis=0)
    out["mean"] = _per_name(names, mu)
    out["std"] = _per_name(names, sigma)
    out["cov"] = [[_finite(v) for v in row] for row in cov]
    out["quantiles"] = {
        n: [_finite(v) for v in quant[:, j]] for j, n in enumerate(names)
    }
    out["quantile_levels"] = list(QUANTILE_LEVELS)
    chi2_draw = -2.0 * lp
    out["chi2_total_quantiles"] = [
        _finite(v) for v in np.quantile(chi2_draw, CHI2_LEVELS)
    ]
    out["chi2_levels"] = list(CHI2_LEVELS)

    below = np.sum(X < best, axis=0)
    ties = np.sum(X == best, axis=0)
    percentile = (below + 0.5 * ties) / n_draws
    out["best_percentile"] = _per_name(names, percentile)

    if box_ok:
        band = _BOUND_BAND * width
        near_lo = np.mean(X <= lo + band, axis=0)
        near_hi = np.mean(X >= hi - band, axis=0)
    else:
        near_lo = near_hi = np.zeros(k)
    at_bound = np.maximum(near_lo, near_hi)
    truncated = [n for n, m in zip(names, at_bound) if m > _TRUNCATION_MASS]
    out["mass_at_bound"] = {
        n: {"lower": _finite(a), "upper": _finite(b)}
        for n, a, b in zip(names, near_lo, near_hi)
        if max(a, b) > 0
    }
    out["truncated_by_bounds"] = truncated

    # ---- Effective dimension --------------------------------------------
    mean_lp = float(np.mean(lp))
    out["log_likelihood_mean"] = _finite(mean_lp)
    out["p_d"] = _finite(2.0 * (best_logp - mean_lp))
    out["d_bayes"] = _finite(2.0 * float(np.var(lp, ddof=1)))

    # ---- Covariance-based quantities ------------------------------------
    sign, logdet = np.linalg.slogdet(cov)
    cov_ok = bool(sign > 0 and math.isfinite(logdet) and np.all(sigma > 0))
    laplace = None
    occam = None
    if cov_ok:
        d = best - mu
        try:
            d2 = float(d @ np.linalg.solve(cov, d))
        except np.linalg.LinAlgError:
            d2 = None
        out["best_mahalanobis2"] = _finite(d2)
        out["best_mahalanobis2_per_param"] = _finite(d2 / k) if d2 is not None else None
        occam = occam_factors(sigma, width) if box_ok else None
        if occam is not None:
            o_corr = -0.5 * (logdet - 2.0 * float(np.sum(np.log(sigma))))
            out["occam"] = _per_name(names, occam)
            out["occam_correlation"] = _finite(o_corr)
            out["occam_total"] = _finite(float(np.sum(occam)) + o_corr)
            out["occam_bic_per_param"] = (
                _finite(0.5 * math.log(n_data)) if n_data > 0 else None
            )
        if log_v is not None and uniform_prior:
            laplace = best_logp + 0.5 * (k * math.log(2 * math.pi) + logdet) - log_v
    else:
        out["best_mahalanobis2"] = None
        out["best_mahalanobis2_per_param"] = None
        frozen = [n for n, s in zip(names, sigma) if not s > 0]
        concerns.append(
            "The draw's covariance is singular"
            + (f" ({', '.join(frozen)} never moved)" if frozen else "")
            + ", so the Laplace evidence and the best-point test are unavailable."
        )
    out["log_evidence_laplace"] = _finite(laplace)
    if laplace is not None and out["log_evidence_bic"] is not None:
        # ε = (−½·BIC) − ln Z_L = Σ(o_j − ½ ln n) + o_R: BIC's Occam error.
        out["bic_error_laplace"] = _finite(out["log_evidence_bic"] - laplace)
    else:
        out["bic_error_laplace"] = None

    # ---- Bridge sampling -------------------------------------------------
    bridge = None
    if not uniform_prior:
        concerns.append(
            "Not every parameter has a uniform box prior, so the prior volume "
            "is undefined and no evidence is computed."
        )
    elif (
        log_density is not None
        and n_proposal > 0
        and log_v is not None
        and n_unique >= min_unique
    ):
        try:
            bridge = bridge_log_evidence(
                X,
                lp,
                log_density,
                lo,
                hi,
                chain_ids=cid,
                n_proposal=n_proposal,
                rng=rng,
            )
        except Exception as exc:  # measurement only — never fail a fit
            logger.warning("[POSTERIOR] Bridge sampling failed: %s", exc)
            bridge = None
    if bridge is not None:
        lz_bridge = bridge["log_integral"] - log_v
        out["log_evidence_bridge"] = _finite(lz_bridge)
        out["log_evidence_bridge_error"] = _finite(bridge["error"])
        out["bridge"] = {k_: v for k_, v in bridge.items() if k_ != "log_integral"}
        if out["log_evidence_bic"] is not None:
            out["bic_error"] = _finite(out["log_evidence_bic"] - lz_bridge)
    else:
        out["log_evidence_bridge"] = None
        out["log_evidence_bridge_error"] = None
        out["bridge"] = None
        out["bic_error"] = out.get("bic_error_laplace")

    if out["log_evidence_bridge"] is not None:
        out["log_evidence"] = out["log_evidence_bridge"]
        out["log_evidence_error"] = out["log_evidence_bridge_error"]
        out["log_evidence_method"] = "bridge"
    else:
        out["log_evidence"] = out["log_evidence_laplace"]
        out["log_evidence_error"] = None
        out["log_evidence_method"] = "laplace" if laplace is not None else None

    # ---- Checks and verdicts --------------------------------------------
    enough = n_unique >= min_unique
    rhat_ok = None if rhat_max_seen is None else bool(rhat_max_seen <= rhat_max)
    converged: Optional[bool] = None if rhat_ok is None else bool(enough and rhat_ok)

    d2k = out.get("best_mahalanobis2_per_param")
    interior_extremes = [
        n
        for n, u, m in zip(names, percentile, at_bound)
        if m <= _TRUNCATION_MASS
        and (u < _EXTREME_PERCENTILE or u > 1 - _EXTREME_PERCENTILE)
    ]
    best_central = (
        None
        if d2k is None
        else bool(d2k <= _CENTRAL_D2_PER_PARAM and not interior_extremes)
    )
    gap = (
        abs(out["log_evidence_laplace"] - out["log_evidence_bridge"])
        if out["log_evidence_laplace"] is not None
        and out["log_evidence_bridge"] is not None
        else None
    )
    bridge_precise = (
        None
        if out["log_evidence_bridge_error"] is None
        else bool(out["log_evidence_bridge_error"] <= max_error)
    )
    out["checks"] = {
        "enough_unique_draws": bool(enough),
        "rhat_ok": rhat_ok,
        "converged": converged,
        "best_central": best_central,
        "best_extreme_percentile": interior_extremes,
        "untruncated": not truncated,
        "bridge_precise": bridge_precise,
        "laplace_bridge_gap": _finite(gap),
        "gaussian": None if gap is None else bool(gap <= _GAUSSIAN_GAP),
    }
    out["fit_trusted"] = bool(converged is True and best_central is True)
    out["evidence_trusted"] = bool(converged is True and bridge_precise is True)

    # ---- Plain-language concerns ----------------------------------------
    if not enough:
        concerns.append(
            f"Only {n_unique} distinct points in the draw (need {min_unique} for "
            f"k = {k}); the posterior is not resolved. Raise the DREAM sample count."
        )
    if rhat_ok is False:
        worst = _worst(names + ["log-likelihood"], rhat_all)
        concerns.append(
            f"The chains did not mix: split-R̂ reaches {_fmt(rhat_max_seen)} "
            f"(limit {rhat_max:g}) on {worst}, so the draw does not yet represent "
            f"the posterior and its ±σ and evidence are provisional."
        )
    if converged is None:
        concerns.append("No convergence diagnostic was available for this draw.")
    if d2k is not None and d2k > _CENTRAL_D2_PER_PARAM:
        concerns.append(
            f"The best point is not where the posterior mass is (Mahalanobis "
            f"D²/k = {d2k:.2f}): the posterior is multimodal, curved or skewed, so "
            f"the Gaussian picture behind BIC and the quoted ±σ does not hold."
        )
    for n in interior_extremes:
        concerns.append(
            f"The best value of {n} sits in the outer 1 % of its own marginal "
            f"(percentile {percentile[names.index(n)]:.3f}), away from any bound."
        )
    if truncated:
        concerns.append(
            "The box truncates the posterior of "
            + ", ".join(truncated)
            + "; the optimum is a boundary point, and the Laplace evidence "
            "overstates the evidence by 0.2–0.9 nats per truncated parameter."
        )
    p_d = out.get("p_d")
    if p_d is not None and k and k - p_d >= _UNCONSTRAINED_PARAMS:
        concerns.append(
            f"The data constrain about {p_d:.1f} of the {k} parameters "
            f"(p_D = {p_d:.1f}); BIC charges for all {k}."
        )
    if occam is not None and n_data > 0:
        weak = [n for n, o in zip(names, occam) if o < _WEAK_OCCAM]
        if weak:
            concerns.append(
                "Barely constrained (Occam < 1 nat, BIC charges "
                f"{0.5 * math.log(n_data):.2f}): " + ", ".join(weak) + "."
            )
    if bridge_precise is False:
        concerns.append(
            f"The bridge-sampling error (±{out['log_evidence_bridge_error']:.2f} "
            f"nats) exceeds {max_error:g}; the evidence is not precise enough to "
            f"compare models."
        )
    if gap is not None and gap > _GAUSSIAN_GAP:
        concerns.append(
            f"Laplace and bridge evidences differ by {gap:.1f} nats: the posterior "
            f"is not Gaussian, so BIC's large-sample picture does not apply."
        )
    return out


def _worst(names: Sequence[str], values: Optional[np.ndarray]) -> str:
    if values is None or not len(values):
        return "?"
    return names[int(np.argmax(values))]


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "∞" if math.isinf(value) else f"{value:.2f}"


# ----------------------------------------------------------------------
# bumps adapter
# ----------------------------------------------------------------------


def _problem_log_density(problem: Any) -> Callable[[np.ndarray], np.ndarray]:
    """``ℓ(θ) = −nllf(θ)`` on rows of an array; −∞ where the model fails."""

    def log_density(Y: np.ndarray) -> np.ndarray:
        out = np.full(len(Y), -np.inf)
        for i, y in enumerate(Y):
            problem.setp(y)
            value = float(problem.nllf())
            if math.isfinite(value):
                out[i] = -value
        return out

    return log_density


def summarize_dream_fit(
    problem: Any,
    fit_result: Any,
    *,
    n_proposal: int = DEFAULT_PROPOSALS,
    rhat_max: float = DEFAULT_RHAT_MAX,
    max_error: float = DEFAULT_MAX_ERROR,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict[str, Any]]:
    """:func:`summarize_posterior` for a bumps DREAM ``OptimizeResult``.

    Uses the retained draw over the chains bumps did not mark as outliers,
    honouring the state's trim portion — the same points ``state.draw()``
    returns. Returns None when the result carries no DREAM state.

    bumps' own convergence test (``ks_converged``) is deliberately not used: it
    fails any draw whose best point improved inside the retained window, by any
    amount, which a well-mixed sampler does by chance. Split-R̂ is used instead.

    Bridge sampling moves the problem's parameters; they are restored before
    returning, so a caller that reads the problem afterwards sees the fit.
    """
    state = getattr(fit_result, "state", None)
    if state is None or not hasattr(state, "chains") or not hasattr(state, "best"):
        return None
    params = list(getattr(problem, "_parameters", []))
    _, chains, logp = state.chains()
    chains = np.asarray(chains, float)
    logp = np.asarray(logp, float)
    if chains.ndim != 3 or chains.shape[0] < 2 or chains.shape[2] != len(params):
        return None
    portion = float(getattr(state, "portion", 1.0) or 1.0)
    start = int((1.0 - portion) * chains.shape[0])
    good = getattr(state, "_good_chains", slice(None))
    chains = chains[start:][:, good, :]
    logp = logp[start:][:, good]
    n_gen, n_chain, k = chains.shape

    lo, hi = (np.asarray(b, float) for b in problem.bounds())
    best_x, best_logp = state.best()
    uniform = all(type(getattr(p, "prior", None)).__name__ == "Bounded" for p in params)
    p0 = np.asarray(problem.getp(), float)
    try:
        summary = summarize_posterior(
            chains.reshape(n_gen * n_chain, k),
            logp.reshape(n_gen * n_chain),
            best_x=np.asarray(best_x, float),
            best_logp=float(best_logp),
            lower=lo,
            upper=hi,
            labels=[str(p.name) for p in params],
            n_data=int(problem.model_points()),
            chain_ids=np.tile(np.arange(n_chain), n_gen),
            log_density=_problem_log_density(problem),
            uniform_prior=uniform,
            n_proposal=n_proposal,
            rhat_max=rhat_max,
            max_error=max_error,
            rng=rng,
        )
    finally:
        problem.setp(p0)
    return summary


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def _env_number(name: str, default: float, *, cast=float, minimum: float = 0.0):
    import os

    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        logger.warning(
            "[POSTERIOR] %s=%r is not a number — using %s", name, raw, default
        )
        return default
    if not math.isfinite(value) or value < minimum:
        logger.warning(
            "[POSTERIOR] %s=%r is out of range — using %s", name, raw, default
        )
        return default
    return value


def evidence_settings() -> Dict[str, Any]:
    """The evidence knobs, from the environment (``docs/evidence.md`` §7.6).

    ``EVIDENCE_PROPOSALS`` (bridge proposal draws, ``0`` turns bridge sampling
    off), ``EVIDENCE_RHAT_MAX``, ``EVIDENCE_MAX_ERROR`` (nats) and
    ``EVIDENCE_MARGIN`` (nats). An unusable value falls back to its default
    with a warning rather than failing the run.
    """
    return {
        "n_proposal": int(
            _env_number("EVIDENCE_PROPOSALS", DEFAULT_PROPOSALS, cast=int)
        ),
        "rhat_max": _env_number("EVIDENCE_RHAT_MAX", DEFAULT_RHAT_MAX, minimum=1.0),
        "max_error": _env_number("EVIDENCE_MAX_ERROR", DEFAULT_MAX_ERROR),
        "margin": _env_number("EVIDENCE_MARGIN", DEFAULT_MARGIN),
    }


# ----------------------------------------------------------------------
# Using a summary
# ----------------------------------------------------------------------


def evidence_usable(posterior: Any) -> bool:
    """True when *posterior* carries a **trusted** evidence: one that may decide."""
    return (
        isinstance(posterior, dict)
        and posterior.get("formula") == POSTERIOR_FORMULA
        and posterior.get("evidence_trusted") is True
        and _finite(posterior.get("log_evidence")) is not None
    )


def bridge_available(posterior: Any) -> bool:
    """True when *posterior* carries a bridge-sampling evidence, trusted or not.

    An untrusted one is *provisional*: it may contest a BIC verdict (the revert
    still happens, the hypothesis is just not recorded as ruled out), never
    decide one. The Laplace estimate does not qualify — on the thin-layer ridges
    of docs/evidence.md §6 it was off by up to 9 nats.
    """
    return (
        isinstance(posterior, dict)
        and posterior.get("formula") == POSTERIOR_FORMULA
        and posterior.get("log_evidence_method") == "bridge"
        and _finite(posterior.get("log_evidence")) is not None
    )


def evidence_record(fit: Mapping[str, Any], fit_index: int) -> Dict[str, Any]:
    """The slice of a fit the evidence baseline keeps (see ``state.best_evidence``)."""
    post = fit.get("posterior") or {}
    return {
        "formula": post.get("formula"),
        "log_evidence": _finite(post.get("log_evidence")),
        "log_evidence_error": _finite(post.get("log_evidence_error")) or 0.0,
        "log_evidence_bic": _finite(post.get("log_evidence_bic")),
        "chi_squared": _finite(fit.get("chi_squared")),
        "iteration": fit.get("iteration"),
        "fit_index": int(fit_index),
        "n_free_params": int(post.get("k") or 0),
        "bounds": dict(fit.get("bounds") or {}),
    }


def evidence_baseline_current(record: Any) -> bool:
    """Whether a stored evidence baseline was computed under this convention."""
    return (
        isinstance(record, dict)
        and record.get("formula") == POSTERIOR_FORMULA
        and _finite(record.get("log_evidence")) is not None
    )


def prior_width_adjustment(
    new_bounds: Mapping[str, Sequence[float]],
    base_bounds: Mapping[str, Sequence[float]],
) -> float:
    """``Σ ln(w_new / w_base)`` over the parameters both fits share by name.

    Widening the box of a parameter whose posterior sits well inside both boxes
    lowers ``ln Z`` by exactly ``ln(w_new / w_base)`` and says nothing about the
    data; adding this back compares the two fits on a common prior for every
    shared parameter. Parameters only one fit has keep their own box: that is
    the Occam factor of the structure being tested.
    """
    total = 0.0
    for name, nb in (new_bounds or {}).items():
        bb = (base_bounds or {}).get(name)
        if not (nb and bb and len(nb) == 2 and len(bb) == 2):
            continue
        w_new, w_base = float(nb[1]) - float(nb[0]), float(bb[1]) - float(bb[0])
        if w_new > 0 and w_base > 0 and math.isfinite(w_new) and math.isfinite(w_base):
            total += math.log(w_new / w_base)
    return total


def compare_evidence(
    new: Mapping[str, Any], base: Mapping[str, Any], *, margin: float = DEFAULT_MARGIN
) -> Dict[str, Any]:
    """Compare two :func:`evidence_record` snapshots; positive Δ favours *new*.

    The verdict is ``better``/``worse`` when ``|Δ|`` exceeds
    ``max(margin, 2σ_Δ)`` and ``inconclusive`` otherwise. ``revert`` applies the
    parsimony rule inside that band: an inconclusive comparison goes to the model
    with fewer free parameters. ``bic_*`` record what BIC alone would have said,
    so a disagreement is visible.
    """
    raw = float(new["log_evidence"]) - float(base["log_evidence"])
    adjust = prior_width_adjustment(new.get("bounds") or {}, base.get("bounds") or {})
    delta = raw + adjust
    error = math.hypot(
        float(new.get("log_evidence_error") or 0.0),
        float(base.get("log_evidence_error") or 0.0),
    )
    threshold = max(float(margin), 2.0 * error)
    if delta > threshold:
        verdict = "better"
    elif delta < -threshold:
        verdict = "worse"
    else:
        verdict = "inconclusive"
    k_new, k_base = (
        int(new.get("n_free_params") or 0),
        int(base.get("n_free_params") or 0),
    )
    revert = verdict == "worse" or (verdict == "inconclusive" and k_new > k_base)
    becomes_baseline = verdict == "better" or (
        verdict == "inconclusive"
        and (k_new < k_base or (k_new == k_base and delta > 0))
    )
    bic_new, bic_base = (
        _finite(new.get("log_evidence_bic")),
        _finite(base.get("log_evidence_bic")),
    )
    bic_delta = (
        bic_new - bic_base if bic_new is not None and bic_base is not None else None
    )
    return {
        "criterion": "evidence",
        "baseline_iteration": base.get("iteration"),
        "baseline_fit_index": base.get("fit_index"),
        "delta_log_evidence": delta,
        "delta_log_evidence_raw": raw,
        "prior_width_adjustment": adjust,
        "error": error,
        "threshold": threshold,
        "verdict": verdict,
        "revert": bool(revert),
        "becomes_baseline": bool(becomes_baseline),
        "n_free_params": k_new,
        "baseline_n_free_params": k_base,
        "delta_log_evidence_bic": bic_delta,
        "bic_verdict": None
        if bic_delta is None
        else ("better" if bic_delta > 0 else "worse"),
        "bic_agrees": None
        if bic_delta is None or verdict == "inconclusive"
        else bool((bic_delta > 0) == (verdict == "better")),
    }


def locate_point(
    posterior: Mapping[str, Any],
    parameters: Mapping[str, float],
    chi2_total: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Where a point (say, the exploration optimum) sits in a stored posterior.

    Uses only what the summary keeps — mean, covariance, marginal quantiles and
    the draw's χ² quantiles — so it works from a checkpoint without the chain.
    Returns its Mahalanobis distance, approximate marginal percentiles, and the
    approximate share of posterior draws with a lower χ² than *chi2_total*.
    """
    mean = posterior.get("mean") or {}
    names = list(mean)
    cov = posterior.get("cov")
    if not names or not cov or any(n not in parameters for n in names):
        return None
    try:
        x = np.array([float(parameters[n]) for n in names])
        mu = np.array([float(mean[n]) for n in names])
        C = np.array(cov, float)
        d = x - mu
        d2 = float(d @ np.linalg.solve(C, d))
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    levels = np.asarray(posterior.get("quantile_levels") or QUANTILE_LEVELS, float)
    pct = {}
    for n in names:
        qs = (posterior.get("quantiles") or {}).get(n)
        if not qs or any(q is None for q in qs):
            pct[n] = None
            continue
        pct[n] = _interp_level(float(parameters[n]), np.asarray(qs, float), levels)
    share_better = None
    cq = posterior.get("chi2_total_quantiles")
    if chi2_total is not None and cq and all(q is not None for q in cq):
        share_better = _interp_level(
            float(chi2_total),
            np.asarray(cq, float),
            np.asarray(posterior.get("chi2_levels") or CHI2_LEVELS, float),
        )
    k = len(names)
    return {
        "mahalanobis2": _finite(d2),
        "mahalanobis2_per_param": _finite(d2 / k),
        "percentiles": pct,
        "share_of_draws_with_lower_chi2": share_better,
        "delta_chi2_vs_best": (
            _finite(float(chi2_total) - float(posterior["chi2_total_best"]))
            if chi2_total is not None and posterior.get("chi2_total_best") is not None
            else None
        ),
    }


def _interp_level(value: float, quantiles: np.ndarray, levels: np.ndarray) -> float:
    """Approximate CDF at *value* from a quantile table (clamped to its ends)."""
    if value <= quantiles[0]:
        return float(levels[0]) if value == quantiles[0] else 0.0
    if value >= quantiles[-1]:
        return float(levels[-1]) if value == quantiles[-1] else 1.0
    return float(np.interp(value, quantiles, levels))


def posterior_digest(posterior: Any) -> Optional[Dict[str, Any]]:
    """The headline numbers of a summary, for reports and machine payloads."""
    if not isinstance(posterior, dict):
        return None
    draw = posterior.get("draw") or {}
    checks = posterior.get("checks") or {}
    return {
        "fit_trusted": posterior.get("fit_trusted"),
        "evidence_trusted": posterior.get("evidence_trusted"),
        "log_evidence": posterior.get("log_evidence"),
        "log_evidence_error": posterior.get("log_evidence_error"),
        "log_evidence_method": posterior.get("log_evidence_method"),
        "log_evidence_laplace": posterior.get("log_evidence_laplace"),
        "log_evidence_bic": posterior.get("log_evidence_bic"),
        "bic_error": posterior.get("bic_error"),
        "p_d": posterior.get("p_d"),
        "k": posterior.get("k"),
        "best_mahalanobis2_per_param": posterior.get("best_mahalanobis2_per_param"),
        "rhat_max": draw.get("rhat_max"),
        "rhat_max_is_infinite": draw.get("rhat_max_is_infinite"),
        "rhat_limit": (posterior.get("thresholds") or {}).get("rhat_max"),
        "n_draws": draw.get("n_draws"),
        "n_unique": draw.get("n_unique"),
        "converged": checks.get("converged"),
        "best_central": checks.get("best_central"),
        "truncated_by_bounds": posterior.get("truncated_by_bounds") or [],
        "concerns": list(posterior.get("concerns") or []),
    }


def posterior_for_report(state: Mapping[str, Any]) -> tuple:
    """``(posterior, source)`` for the fit a run reports as its answer.

    The final DREAM polish, when it ran, characterizes the reported structure
    with the longest chain of the run, so its summary wins. Otherwise the
    finalize-selected fit's own summary (a ``FIT_METHOD=dream`` run), else the
    last fit's. *source* is ``"final_fit"``, ``"selected"`` or ``"last"``;
    ``(None, None)`` when no fit carries a summary.
    """
    fits = list(state.get("fit_results") or [])
    if not fits:
        return None, None
    final = state.get("final_fit") or {}
    if final.get("ran") and isinstance(fits[-1].get("posterior"), dict):
        return fits[-1]["posterior"], "final_fit"
    selection = state.get("final_selection") or {}
    index = selection.get("index") if selection.get("selected") else None
    if isinstance(index, int) and 0 <= index < len(fits):
        post = fits[index].get("posterior")
        return (post, "selected") if isinstance(post, dict) else (None, None)
    post = fits[-1].get("posterior")
    return (post, "last") if isinstance(post, dict) else (None, None)
