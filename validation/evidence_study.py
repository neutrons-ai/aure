"""Reproduce the validation tables of docs/evidence.md §6.

Two synthetic reflectivity scenarios, fitted with DREAM exactly as aure
configures it and summarized by ``aure.tools.posterior_tools``, scored against
nested sampling (``dynesty``) on the same data and the same box priors::

    python -m validation.evidence_study run   --out evidence_study/      # DREAM fits
    python -m validation.evidence_study nested --out evidence_study/     # needs dynesty
    python -m validation.evidence_study report --out evidence_study/     # the tables

``run`` and ``nested`` take ``--models`` and ``--budgets`` / ``--seeds`` to do
a subset; each job writes its own JSON, so jobs can run in parallel processes.
The study takes about an hour of CPU in total; ``nested`` is most of it.

Scenario A (spurious layer): Si | SiO2 12 Å | polymer 220 Å | air, 4 % noise.
  A1 no oxide (k=5), A2 the true structure (k=8), A3 true + a spurious top layer (k=11).
Scenario B (sub-resolution layer): Si | 5 Å interlayer, SLD 2.40 | polymer | air,
  10 % noise. B1 no interlayer (k=5), B2 the true structure (k=8).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

SCENARIOS = {
    "A": dict(seed=7, ox_t=12.0, ox_rho=3.47, ox_sig=3.0, noise=0.04),
    "B": dict(seed=11, ox_t=5.0, ox_rho=2.40, ox_sig=2.0, noise=0.10),
}
MODELS = ("A1", "A2", "A3", "B1", "B2")
#: FIT_STEPS / FIT_BURN as aure passes them to bumps (samples / burn generations).
BUDGETS = {
    "explore": (1000, 1000),  # the exploration defaults
    "final": (10000, 10000),  # final_fit's defaults
    "long": (100000, 10000),
    "verylong": (800000, 10000),
}


def _si():
    from refl1d.names import SLD

    return SLD("Si", rho=2.07)


def simulate(scenario: str):
    """160 points on 0.008–0.25 Å⁻¹, dQ/Q = 2.5 %, Gaussian noise of ``noise``·R."""
    from refl1d.names import SLD, Experiment, QProbe, air

    p = SCENARIOS[scenario]
    rng = np.random.default_rng(p["seed"])
    q = np.linspace(0.008, 0.25, 160)
    dq = 0.025 * q
    truth = (
        _si()(0, 3)
        | SLD("SiO2", rho=p["ox_rho"])(p["ox_t"], p["ox_sig"])
        | SLD("poly", rho=1.6)(220, 6)
        | air
    )
    _, r = Experiment(probe=QProbe(q, dq), sample=truth).reflectivity()
    dr = p["noise"] * r + 2e-8
    return q, dq, r + rng.normal(0, dr), dr


def build(name: str, data):
    """Model ``<scenario><1|2|3>`` with the study's boxes (the priors)."""
    from refl1d.names import SLD, Experiment, FitProblem, QProbe, air

    q, dq, r, dr = data
    probe = QProbe(q, dq, R=r, dR=dr)
    probe.intensity.range(0.8, 1.2)
    kind = name[-1]
    if kind == "1":
        s = _si()(0, 3) | SLD("poly", rho=1.6)(220, 6) | air
        poly = s[1]
    else:
        s = _si()(0, 3) | SLD("SiO2", rho=3.0)(15, 3) | SLD("poly", rho=1.6)(220, 6)
        if kind == "3":
            s = s | SLD("x", rho=1.0)(10, 3)
        s = s | air
        ox, poly = s[1], s[2]
        ox.thickness.range(0, 40)
        ox.material.rho.range(0, 6)
        ox.interface.range(0, 10)
        if kind == "3":
            x = s[3]
            x.thickness.range(0, 50)
            x.material.rho.range(0, 6)
            x.interface.range(0, 10)
    s[0].interface.range(0, 15)
    poly.thickness.range(100, 400)
    poly.material.rho.range(0, 4)
    poly.interface.range(0, 20)
    problem = FitProblem(Experiment(probe=probe, sample=s))
    problem.name = name
    return problem


def run_dream(name: str, budget: str, out: Path, seed: int = 1) -> dict:
    from bumps.fitters import fit

    from aure.nodes.fitting import _build_fit_options
    from aure.tools.posterior_tools import summarize_dream_fit

    np.random.seed(seed)
    steps, burn = BUDGETS[budget]
    problem = build(name, simulate(name[0]))
    t0 = time.time()
    result = fit(problem, **_build_fit_options("dream", steps, burn))
    seconds = time.time() - t0
    problem.setp(result.x)
    post = summarize_dream_fit(problem, result, rng=np.random.default_rng(0))
    rec = {
        "model": name,
        "budget": budget,
        "seed": seed,
        "steps": steps,
        "burn": burn,
        "seconds": seconds,
        "evaluations": (burn + post["draw"]["n_generations"])
        * post["draw"]["n_chains"],
        "posterior": post,
    }
    (out / f"dream_{name}_{budget}.json").write_text(json.dumps(rec, indent=1))
    return rec


def run_nested(name: str, out: Path, seed: int, nlive: int = 1000) -> dict:
    import dynesty

    problem = build(name, simulate(name[0]))
    lo, hi = (np.asarray(b, float) for b in problem.bounds())

    def loglike(theta):
        problem.setp(theta)
        v = problem.nllf()
        return -float(v) if np.isfinite(v) else -1e300

    sampler = dynesty.NestedSampler(
        loglike,
        lambda u: lo + u * (hi - lo),
        len(lo),
        nlive=nlive,
        bound="multi",
        sample="rslice",
        rstate=np.random.default_rng(seed),
    )
    t0 = time.time()
    sampler.run_nested(dlogz=0.05, print_progress=False)
    res = sampler.results
    rec = {
        "model": name,
        "seed": seed,
        "nlive": nlive,
        "logz": float(res.logz[-1]),
        "logzerr": float(res.logzerr[-1]),
        "ncall": int(np.sum(res.ncall)),
        "seconds": time.time() - t0,
    }
    (out / f"nested_{name}_s{seed}.json").write_text(json.dumps(rec, indent=1))
    return rec


def report(out: Path) -> None:
    nested: dict = {}
    for f in glob.glob(str(out / "nested_*.json")):
        r = json.loads(Path(f).read_text())
        nested.setdefault(r["model"], []).append(r["logz"])
    ref = {
        m: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        for m, v in nested.items()
    }
    runs = {}
    for f in glob.glob(str(out / "dream_*.json")):
        r = json.loads(Path(f).read_text())
        runs[(r["model"], r["budget"])] = r

    def fmt(v, d=2):
        return "—" if v is None else f"{v:.{d}f}"

    print(
        "| Model | Budget | Draws (distinct) | split-R̂ max | ln Z_BIC | Laplace | Bridge ± σ | Nested | Bridge − nested | Trusted |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for b in BUDGETS:
            r = runs.get((m, b))
            if r is None:
                continue
            p = r["posterior"]
            ns = ref.get(m, (None, None))[0]
            diff = (
                None
                if ns is None or p["log_evidence_bridge"] is None
                else p["log_evidence_bridge"] - ns
            )
            print(
                f"| {m} | {b} | {p['draw']['n_draws']:,} ({p['draw']['n_unique']:,}) | "
                f"{fmt(p['draw']['rhat_max'])} | {fmt(p['log_evidence_bic'], 1)} | "
                f"{fmt(p['log_evidence_laplace'], 1)} | {fmt(p['log_evidence_bridge'])} ± "
                f"{fmt(p['log_evidence_bridge_error'])} | {fmt(ns)} | "
                f"{'—' if diff is None else f'{diff:+.2f}'} | {'yes' if p['evidence_trusted'] else 'no'} |"
            )
    print()
    for a, b in (("A2", "A3"), ("B2", "B1")):
        if a in ref and b in ref:
            d = ref[a][0] - ref[b][0]
            print(
                f"{a} − {b}: nested Δln Z = {d:+.2f} ± {math.hypot(ref[a][1], ref[b][1]):.2f}"
            )
        for bud in BUDGETS:
            ra, rb = runs.get((a, bud)), runs.get((b, bud))
            if not (ra and rb):
                continue
            pa, pb = ra["posterior"], rb["posterior"]
            print(
                f"  {bud:9s} Δ(−BIC/2) = {pa['log_evidence_bic'] - pb['log_evidence_bic']:+.2f}  "
                f"ΔLaplace = {pa['log_evidence_laplace'] - pb['log_evidence_laplace']:+.2f}  "
                f"Δbridge = {pa['log_evidence_bridge'] - pb['log_evidence_bridge']:+.2f}"
            )


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("run", "nested", "report"))
    ap.add_argument("--out", default="evidence_study", type=Path)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--budgets", default="explore,final,long")
    ap.add_argument("--seeds", default="11,12,13")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m]
    if args.command == "run":
        for m in models:
            for b in args.budgets.split(","):
                rec = run_dream(m, b, args.out)
                print(
                    f"{m} {b}: ln Z = {rec['posterior']['log_evidence']} ({rec['seconds']:.0f} s)"
                )
    elif args.command == "nested":
        for m in models:
            for s in args.seeds.split(","):
                rec = run_nested(m, args.out, int(s))
                print(f"{m} seed {s}: ln Z = {rec['logz']:.2f} ({rec['ncall']} calls)")
    else:
        report(args.out)


if __name__ == "__main__":
    main()
