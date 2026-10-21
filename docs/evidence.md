# Posterior evidence: how far to trust BIC, and what to use instead

This document explains how aure judges a fitted model from its posterior after
a DREAM fit, instead of from the best point alone. It covers three things:

1. **A measure of how far a BIC value, and a BIC comparison, can be trusted.**
   The draw measures the quantities BIC assumes: Occam factors, correlations,
   truncation, convergence, and where the best point sits.
2. **A better replacement for BIC.** This is the Bayesian evidence estimated
   from the same draw by bridge sampling, with its own error bar.
3. **A trust verdict on the fit itself.** It asks whether the reported optimum
   represents its posterior, which is what an experienced user checks by eye
   on the marginal distributions.

The code is [`tools/posterior_tools.py`](../src/aure/tools/posterior_tools.py).
§7 describes how the workflow uses it, and §6 validates it against nested
sampling. BIC itself, and the regression guardrail built on it, are described
in [metrics.md §3–4](metrics.md#3-bic--complexity-penalty).

---

## 1. Why BIC is not enough

aure's BIC is the known-variance form

$$\mathrm{BIC} = \chi^2_\text{min} + k\ln n .$$

It approximates $-2\ln Z$, where $Z$ is the Bayesian evidence (§2). The
approximation makes three assumptions, and reflectivity fits routinely break
each of them.

1. **The best point is the global optimum of that model.** BIC uses
   $\chi^2_\text{min}$ verbatim. A fit stuck in a local minimum, such as the
   thin-layer $\Delta\rho\cdot t$ ridge of [approach.md §6.8](approach.md),
   has too high a χ², and its BIC inherits the error one for one. In that
   regime BIC is no better than χ²: the penalty is right and the likelihood
   term is wrong.
2. **The posterior is one Gaussian, well inside its box.** Multimodal,
   curved or truncated posteriors break the Laplace step BIC rests on (§2.2).
   The sub-resolution ridge is curved, and roughnesses pile up against zero.
3. **Every parameter is constrained by "one observation's worth" of prior.**
   BIC charges exactly $\tfrac12\ln n$ nats per parameter, whatever the data
   actually constrain and whatever box the parameter has (§2.3). A parameter
   the data barely see is over-charged. A sharply constrained one in a wide
   box is under-charged, and so is a group of strongly correlated ones.

The draw DREAM leaves behind can test all three. It cannot see a basin the
chains never visited, and §4.1 says what that limitation costs.

---

## 2. The evidence, and the approximations that lead to BIC

### 2.1 Definitions

| Symbol | Meaning |
|---|---|
| $n$ | data points: every point of every dataset of every state (`problem.model_points()`) |
| $k$ | free parameters (`len(problem.getp())`) |
| $\theta\in B$ | parameter vector in the box $B=\prod_j[\mathrm{lo}_j,\mathrm{hi}_j]$ |
| $w_j$, $V$ | box widths $w_j=\mathrm{hi}_j-\mathrm{lo}_j$ and volume $V=\prod_j w_j$ |
| $\chi^2(\theta)$ | $\sum_i\big((R_i-R_i(\theta))/dR_i\big)^2$ |
| $\ell(\theta)$ | $-\tfrac12\chi^2(\theta)$: bumps' `-nllf`, which is exactly DREAM's `logp` |
| $\hat\theta$, $\hat\ell$ | the best point, and $\hat\ell=\ell(\hat\theta)=-\tfrac12\chi^2_\text{min}$ |
| $\Sigma$, $\sigma_j$, $R$ | posterior covariance, marginal standard deviations, correlation matrix |

**Likelihood.** With Gaussian, independent, known uncertainties,

$$L(\theta)=\prod_i\frac{1}{\sqrt{2\pi}\,dR_i}\,e^{-r_i^2/2}
\quad\Rightarrow\quad
\ln L(\theta)=\ell(\theta)+c_D,\qquad c_D=-\sum_i\ln\!\big(\sqrt{2\pi}\,dR_i\big).$$

$c_D$ depends only on the data, so it is identical for every model fitted to
them.

**Prior.** aure gives every parameter a uniform box (`.range(lo, hi)`, a bumps
`Bounded` prior), so $\pi(\theta)=1/V$ on $B$. A box is a prior. That is the
main practical consequence of using the evidence, and §5.1 deals with it.

**Evidence.**

$$Z=\int_B L(\theta)\,\pi(\theta)\,d\theta
 =\frac{e^{c_D}}{V}\int_B e^{\ell(\theta)}\,d\theta .$$

As BIC does, aure drops $c_D$ and reports

$$\ln Z\;\equiv\;\ln\!\int_B e^{\ell(\theta)}\,d\theta\;-\;\ln V
\qquad\text{(nats)}.\tag{2.1}$$

Values compare across models of the *same data* and never across data sets.

**Bayes factor.** $\ln B_{ab}=\ln Z_a-\ln Z_b$. With equal prior model
probabilities, $B_{ab}$ is the posterior odds of $a$ over $b$. Kass & Raftery
(1995) read $|\ln B|$ as follows: below 1 is "not worth more than a bare
mention", 1–3 is positive, 3–5 is strong, and above 5 is very strong. Their
table is in $2\ln B$: 0–2, 2–6, 6–10 and above 10.

### 2.2 The Laplace approximation and the Occam factor

Suppose $\ell$ has one maximum $\hat\theta$ inside $B$ and is close to
quadratic around it:

$$\ell(\theta)\approx\hat\ell-\tfrac12(\theta-\hat\theta)^{\mathsf T}\Sigma^{-1}(\theta-\hat\theta),
\qquad \Sigma^{-1}=-\nabla\nabla\ell(\hat\theta)=\tfrac12\nabla\nabla\chi^2(\hat\theta).$$

If the box also cuts off a negligible part of that Gaussian, the integral
becomes Gaussian:

$$\int_B e^{\ell}\,d\theta\approx e^{\hat\ell}\,(2\pi)^{k/2}|\Sigma|^{1/2}
\quad\Rightarrow\quad
\ln Z_L=\hat\ell+\tfrac12\ln|2\pi\Sigma|-\ln V .\tag{2.2}$$

Written as $\ln Z=\hat\ell-O$, the **Occam factor**

$$O=\ln V-\tfrac12\ln|2\pi\Sigma|$$

is the log of the prior volume over the posterior volume
$(2\pi)^{k/2}|\Sigma|^{1/2}$. It splits exactly into per-parameter terms and
one correlation term. Use $\ln|\Sigma|=\sum_j\ln\sigma_j^2+\ln|R|$:

$$O=\sum_j o_j+o_R,\qquad
o_j=\ln\frac{w_j}{\sqrt{2\pi}\,\sigma_j},\qquad
o_R=-\tfrac12\ln|R|\;\ge 0 .\tag{2.3}$$

$o_j$ is the log of how many posterior widths fit in the box of parameter
$j$. $o_R$ is non-negative by Hadamard's inequality ($|R|\le1$).
Correlations shrink the posterior volume below the product of the marginal
widths, so they add to the penalty. On the oxide models of scenario A in §6
the correlation term alone is about 5 nats. That is the $\Delta\rho\cdot t$
ridge seen as a volume.

### 2.3 From Laplace to BIC

For $n$ independent observations, $\Sigma^{-1}=n\bar I$, where $\bar I$ is the
average Fisher information per observation and stays $O(1)$ as $n$ grows. Then

$$\ln Z_L=\hat\ell-\frac k2\ln n+\Big[\tfrac12\ln|2\pi\bar I^{-1}|-\ln V\Big].$$

Schwarz (1978) drops the bracket, because it is $O(1)$ against $O(\ln n)$.
Multiplying by $-2$ and restoring $c_D$ gives
$\mathrm{BIC}=\chi^2_\text{min}+k\ln n$. Kass & Wasserman (1995) showed that
the bracket vanishes *exactly* when the prior is Gaussian with covariance
$\bar I^{-1}$. That is the **unit-information prior**, which carries as much
information as one data point. Per parameter, BIC therefore charges

$$o_j^\text{BIC}=\tfrac12\ln n ,$$

which is the same as assuming $w_j=\sqrt{2\pi n}\,\sigma_j$. For $n=160$ that
box is 31.7 posterior widths wide, and for $n=1000$ it is 79. BIC assumes it
whatever the real box is and whatever the data constrain.

### 2.4 The error BIC makes, measured from the draw

Let $\ln Z_\text{BIC}\equiv-\tfrac12\mathrm{BIC}=\hat\ell-\tfrac k2\ln n$ be
the log-evidence BIC implies. Its difference from the Laplace evidence is

$$\varepsilon\;=\;\ln Z_\text{BIC}-\ln Z_L\;=\;O-\frac k2\ln n
\;=\;\sum_j\Big(o_j-\tfrac12\ln n\Big)+o_R .\tag{2.4}$$

- $\varepsilon>0$: BIC is **optimistic** about the model. Its parameters are
  constrained more tightly, relative to their boxes, than one observation's
  worth, or they are correlated.
- $\varepsilon<0$: BIC **over-charges**, typically for parameters the data
  barely constrain ($o_j\ll\tfrac12\ln n$).

Per parameter, $o_j-\tfrac12\ln n=\ln\!\big(w_j/(\sqrt{2\pi n}\,\sigma_j)\big)$.
That compares the real box with the unit-information box BIC assumes.

For a comparison between models $a$ and $b$,

$$\Delta\ln Z_\text{BIC}-\Delta\ln Z_L=\varepsilon_a-\varepsilon_b .$$

In a nested comparison, shared parameters whose posterior widths do not change
cancel. What remains is the added parameters' $o_j-\tfrac12\ln n$, plus any
change in the shared parameters' widths and correlations. **A BIC verdict is
robust to its own Occam error when $|\Delta\ln Z_\text{BIC}|>|\varepsilon_a-\varepsilon_b|$.**
Otherwise its sign is set by the approximation, not by the data.

$\varepsilon$ is only the part of BIC's error that a Gaussian can see. When the
posterior is not Gaussian, Laplace is wrong as well (§6 has a 3.5-nat example).
The measured total is therefore taken against bridge sampling (§3.3):

$$\varepsilon_\text{total}=\ln Z_\text{BIC}-\ln Z_B .$$

### 2.5 The information identity and the effective number of parameters

Take the posterior expectation of
$\ln p(\theta\mid D)=\ell(\theta)-\ln V-\ln Z$, with $c_D$ dropped:

$$\ln Z=\langle\ell\rangle_p-D_\text{KL}(p\,\|\,\pi),\qquad
D_\text{KL}=\int p\ln\frac p\pi\,d\theta=\ln V-H[p] ,\tag{2.5}$$

where $H[p]$ is the posterior's differential entropy. This is exact. It is the
identity nested sampling integrates, and $D_\text{KL}$ is the "information
gain" of Treece et al. (2019). For a Gaussian posterior,
$\langle\ell\rangle=\hat\ell-k/2$ and $H=\tfrac12\ln|2\pi e\Sigma|$, which
gives back (2.2).

Put the Gaussian entropy of the draw's covariance in place of $H$. That is the
largest entropy any distribution with that covariance can have, so (2.5)
becomes an **upper bound** on $\ln Z$. Its distance from (2.2) is

$$\ln Z_{H,G}-\ln Z_L=\langle\ell\rangle-\hat\ell+\frac k2=\frac{k-p_D}{2},$$

so it is not an independent estimate. It is the $p_D$ check below in other
units, which is why aure does not report it as one.

The **deviance spread** (Spiegelhalter et al. 2002, with the mode as the
plug-in point) and the **Bayesian model dimensionality** (Handley & Lemos 2019)

$$p_D=2\big(\hat\ell-\langle\ell\rangle\big),\qquad \tilde d=2\,\mathrm{Var}_p(\ell)$$

both equal $k$ for a Gaussian posterior. There $2(\hat\ell-\ell)\sim\chi^2_k$,
with mean $k$ and variance $2k$. A parameter whose likelihood is flat across
its box adds nothing to either. So $p_D$ estimates **how many parameters the
data actually constrain**, which is the number BIC ought to be charging for.

---

## 3. Estimating the evidence from a DREAM draw

### 3.1 What the draw provides

After a DREAM fit, bumps keeps every chain's last $G$ generations (the
*retained window*) and the best point ever seen. From that, aure uses:

- the points $\theta_t$ and their $\ell(\theta_t)$. DREAM's `logp` is exactly
  $\ell$, because a `Bounded` prior contributes 0 inside its box;
- the chain identity of each point. Consecutive DREAM draws are correlated,
  and a rejected proposal repeats the current point, so every error estimate
  below works from chains, not from single draws;
- only the chains bumps did not mark as outliers, which is what `state.draw()`
  returns.

$\ln V$ is not part of `logp`. It is added from `problem.bounds()`.

aure passes `FIT_STEPS` to bumps as `samples` and `FIT_BURN` as burn-in
generations, so the retained draw has about `FIT_STEPS` points. In
exploration that is 1 000 by default, about 20 generations for $k=8$. The
final polish (`FIT_STEPS_FINAL`) keeps about 10 000, roughly 130 generations.

### 3.2 Laplace with the sampled covariance

$$\ln\hat Z_L=\hat\ell+\tfrac12\ln|2\pi\hat\Sigma|-\ln V,\qquad\hat\Sigma=\text{sample covariance of the draw}.$$

This costs no likelihood evaluations. It uses the covariance the posterior
actually has, not a Hessian at the optimum, and it fails three ways.

- **Truncation by the box.** This is common for roughnesses near zero. Take a
  one-dimensional posterior on $[0,\infty)$ whose draw has standard deviation
  $s$. Laplace replaces its integral by $\sqrt{2\pi}\,s$:
  - for a half-Gaussian with its mode at the bound, $s=\sigma\sqrt{1-2/\pi}$
    against a true integral of $\sigma\sqrt{\pi/2}$, so Laplace overstates
    $\ln Z$ by $\ln\!\big(2\sqrt{1-2/\pi}\big)=0.19$ nats;
  - for a likelihood still rising through the bound (exponential, $s=\lambda$,
    integral $\lambda$), it overstates by $\ln\sqrt{2\pi}=0.92$ nats.

  So the error is 0.19–0.92 nats per truncated parameter. A Hessian-based
  Laplace would be off by $\ln2$ in the first case and undefined in the
  second.
- **Curvature.** The covariance ellipsoid of a curved ridge holds far more
  volume than the ridge, so Laplace *overstates* the evidence. The oxide
  models of §6 show 2–9 nats of this.
- **Multiple modes.** The covariance then spans the gap between modes.

The sampling error is small by comparison. $\ln|\hat\Sigma|$ from $n_\text{eff}$
effective points has a Wishart bias of about $-k(k+1)/(2n_\text{eff})$ and a
standard deviation of about $\sqrt{2k/n_\text{eff}}$.

aure reports $\ln\hat Z_L$ and the Occam factors (2.3) built from $\hat\Sigma$.
Their gap to the bridge estimate measures how far the posterior is from the
Gaussian picture that BIC and Laplace share.

### 3.3 Bridge sampling

**Identity** (Meng & Wong 1996). Let $\tilde p=e^\ell$ be the unnormalized
posterior on $B$, $I=\int_B\tilde p$, $p=\tilde p/I$, and $q$ any normalized
density. For any *bridge function* $h$ with $\int\tilde p\,q\,h>0$,

$$I=\frac{\mathbb E_q[\tilde p\,h]}{\mathbb E_p[q\,h]} ,\tag{3.1}$$

because $\mathbb E_q[\tilde p h]=\int\tilde p\,h\,q$ and
$\mathbb E_p[qh]=\tfrac1I\int q\,h\,\tilde p$. Replacing the expectations by
averages over $n_2$ draws $\tilde\theta_i\sim q$ and $n_1$ posterior draws
$\theta_j$ gives the estimator. The variance-minimizing bridge is
$h\propto1/(s_1\tilde p+s_2 I q)$, with $s_1=n_1/(n_1+n_2)$ and
$s_2=n_2/(n_1+n_2)$. Because it depends on $I$, it is iterated to a fixed
point:

$$\hat I^{(t+1)}=
\frac{\frac1{n_2}\sum_i \dfrac{\rho_{2i}}{s_1\rho_{2i}+s_2\hat I^{(t)}}}
     {\frac1{n_1}\sum_j \dfrac{1}{s_1\rho_{1j}+s_2\hat I^{(t)}}},
\qquad
\rho_{1j}=\frac{\tilde p(\theta_j)}{q(\theta_j)},\quad
\rho_{2i}=\frac{\tilde p(\tilde\theta_i)}{q(\tilde\theta_i)} .\tag{3.2}$$

Then $\ln\hat Z_B=\ln\hat I-\ln V$.

**aure's choices.**

- **Proposal.** $q=\mathcal N(\hat\mu,\hat\Sigma)$, fitted to half of the
  chains. The other half supply the $n_1$ posterior draws, so $q$ is not tuned
  to the points it is scored on. Gronau et al. (2017) split the sample in
  halves; aure splits by chain, because consecutive DREAM draws are not
  independent.
- **Proposal draws.** $n_2$ = `EVIDENCE_PROPOSALS`, 2 000 by default. Each
  costs one likelihood evaluation, and these are the only new ones. A draw
  outside the box has $\tilde p=0$.
- **Weights.** $n_1$ in $s_1,s_2$ is replaced by its effective size, taken
  from between-chain batch means, as the `bridgesampling` R package does with
  `use_neff`. This affects efficiency only, not consistency.
- **Numerics.** Everything is computed in logs, rescaled by the median of
  $\ln\rho_1$.

**Error** (Frühwirth-Schnatter 2004; Gronau et al. 2017, eq. 17):

$$\mathrm{RE}^2=\frac{\mathbb E[(\hat I-I)^2]}{I^2}\approx
\frac1{n_2}\frac{\mathrm V_q[f_1]}{\mathbb E_q[f_1]^2}
+\frac1{n_{1,\text{eff}}}\frac{\mathrm V_p[f_2]}{\mathbb E_p[f_2]^2},
\qquad
f_1=\frac{\tilde p}{s_1\tilde p+s_2 Iq},\quad f_2=\frac{q}{s_1\tilde p+s_2 Iq} .$$

The posterior term uses the variance of per-chain means of $f_2$ divided by the
number of chains, which accounts for autocorrelation. The error on
$\ln\hat Z_B$ is $\approx\mathrm{RE}$.

**What it does and does not assume.** The estimator is consistent for any
posterior shape, because $q$ only needs to cover it, and its efficiency depends
on how well $q$ overlaps $p$. A multimodal posterior that one Gaussian covers
badly shows up as a large RE. But the estimator treats the retained points as
posterior samples. A draw that is still burning in, or that never spread along
a ridge, biases it in a way RE cannot see. §6 has errors of 1.6–6.7 nats with
a quoted RE of 0.3–0.7, and §4.1 is the check that guards against it.

**Cost.** $n_2$ likelihood evaluations. That is about 2–4 % of an exploration
DREAM fit at the default budget, and under 0.4 % of the final polish.

### 3.4 Estimators considered and not used

- **bumps' `MCMCDraw.entropy()`, method `llf`** (Kramer et al. 2010). Internally
  it computes $\hat N=\langle e^{\ell}/\hat p(\theta)\rangle\approx\int_Be^\ell$,
  which is exactly the integral needed, using a kernel density estimate
  $\hat p$. It then discards $\hat N$ and returns the entropy. Density
  estimation in 8–15 dimensions is its weak point. bumps 1.0.4 also has three
  defects here:
  - the normalization subset is sliced with the entropy subset's size;
  - the two subsets can overlap;
  - the scipy fallback passes the sample untransposed to `gaussian_kde`, so it
    fails whenever there are more draws than parameters.

  The preferred path needs scikit-learn, which aure does not install.
- **bumps' `mvn` method** is the Gaussian entropy: the upper bound of §2.5,
  equivalent to the $p_D$ check.
- **bumps' `wnn` method** (nearest neighbours) needs distinct points, so
  DREAM's repeated draws would have to be de-duplicated first, and that
  biases it.
- **The harmonic-mean estimator** has infinite variance.
- **Nested sampling** (Skilling 2006; `dynesty`) is the reference in §6. There
  it took 0.9–2.8 million likelihood calls per model, against 0.05–1.6 million
  for DREAM's own fits. Running it routinely would multiply the cost of every
  fit.

---

## 4. Can the fit, the BIC and the evidence be trusted?

### 4.1 Did the sampler converge?

aure uses one convergence diagnostic: **split-R̂** (Gelman & Rubin 1992, split
form as in Gelman et al., *BDA3*), computed over the non-outlier chains for
every parameter and for $\ell$. Each chain is cut into two halves and the
halves are treated as separate chains. With $m$ chains of length $n$,

$$W=\overline{s^2_c},\qquad \frac Bn=\mathrm{Var}_c(\bar\theta_c),\qquad
\hat\sigma^2=\frac{n-1}{n}W+\frac Bn,\qquad
\hat R=\sqrt{\frac{m+1}{m}\frac{\hat\sigma^2}{W}-\frac{n-1}{mn}} .$$

$\hat R\to1$ once each chain has explored the whole posterior within the
window. Splitting catches a population that drifts together, which plain R̂
misses: all the chain means agree while each chain's variance absorbs the
drift. Including $\ell$ catches a likelihood that is still climbing.

The draw also needs at least $\max(50,\,10(k+1))$ distinct points. The limit
is $\hat R\le1.2$ (`EVIDENCE_RHAT_MAX`), Vrugt's (2016) convention for DREAM.
`converged` requires both.

Why not bumps' own `ks_converged`? It compares the distribution of $\ell$ at the
head and tail of the window, which is reasonable. But it also refuses any draw
whose best point improved inside the window, *by any amount*, and a
well-mixed sampler does that by chance. In §6 it failed a draw with
$\hat R=1.05$ over a 0.03 improvement in $\ell$.

DREAM's acceptance rate on these problems is low, 2–5 %, so a chain moves
only a handful of times per 100 generations. $\hat R$ is then dominated by the
window length. With 20 retained generations it cannot pass, and the
thin-oxide model of §6 still had $\hat R=2.2$ after 1 250 generations and 1.39
after 10 000.

**No sampler diagnostic sees a basin the chains never visited.** "Converged"
means the draw represents the basins that were explored, not that the global
optimum was found. §4.5 is the partial remedy.

### 4.2 Is the best point representative of the posterior?

This is the check an experienced user makes by eye on the marginal plots. For a
Gaussian posterior the mode, the mean and every marginal median coincide.
With $\hat\mu$ and $\hat\Sigma$ from the draw,

$$D^2=(\hat\theta-\hat\mu)^{\mathsf T}\hat\Sigma^{-1}(\hat\theta-\hat\mu),
\qquad u_j=\hat F_j(\hat\theta_j)\ \ \text{(marginal percentile of the best value)}.$$

For a Gaussian, $D^2\approx0$ and $u_j\approx\tfrac12$. A typical draw has
$\mathbb E[D^2]=k$, so **$D^2/k>1$ means the best point is farther from the
posterior's centre than a typical posterior sample is**. The reported optimum
is then not where the probability mass is: the posterior is skewed, curved or
multimodal, or the optimum is a narrow spike. Separately, $u_j<0.01$ or
$u_j>0.99$ for a parameter not pressed against a bound means the best value
sits in the extreme tail of its own marginal. `best_central` requires neither
to happen.

### 4.3 Truncation by the box

For each parameter, aure measures the share of draws within 1 % of the range
from either bound. Above 5 % the parameter is listed in `truncated_by_bounds`.
The Laplace evidence then overstates the evidence by 0.19–0.92 nats per such
parameter (§3.2), while the bridge estimate is unaffected. BIC does not
integrate at all, so truncation does not enter it directly. It does mean the
optimum is a boundary point, where the asymptotics behind (2.3) do not hold. The boundary-hit check in
[evaluation](../src/aure/nodes/evaluation.py) and bound widening already act on
the same signal.

### 4.4 Gaussianity and constrained dimension

- $|\ln\hat Z_L-\ln\hat Z_B|>1$ nat means the posterior is far from Gaussian in
  a way that moves the evidence. BIC's large-sample picture does not hold for
  this model at this $n$.
- $k-p_D\ge2$ means at least two parameters' worth of the model is not
  constrained by the data, and BIC charges for parameters the data do not see.
- Any $o_j<1$ nat marks a parameter the data barely constrain. BIC charges it
  $\tfrac12\ln n$, which is 2.54 nats at $n=160$.

### 4.5 The exploration optimum against the final posterior

Suppose exploration used another optimizer (`de`, `lm`) and `final_fit` runs
DREAM on the selected model. The exploration optimum $\theta_e$ can then be
placed in the DREAM posterior using the stored mean, covariance and quantiles:
its $D^2/k$, its marginal percentiles, and the share of DREAM draws with a
lower χ² than $\theta_e$.

If more than half of the posterior beats the exploration optimum, or
$D^2/k\gg1$, the exploration fit was not at the minimum of its own structure.
Every χ²/BIC comparison the loop made with that structure was then made at a
local minimum. This directly tests assumption 1 of §1. It is available at the
end of any run that sets `FIT_METHOD_FINAL=dream`, and the final-fit message
reports it.

### 4.6 The verdicts

| Verdict | Definition | Meaning |
|---|---|---|
| `fit_trusted` | `converged` and `best_central` | the quoted ±σ describe the reported point, and the reported point describes the posterior |
| `evidence_trusted` | `converged` and bridge error ≤ `EVIDENCE_MAX_ERROR` (0.5 nat) | $\ln\hat Z_B$ may be compared with another trusted evidence |

Every summary also records the measured BIC error ($\varepsilon$, or
$\varepsilon_\text{total}$ when bridge sampling ran) and a plain-language
`concerns` list.

---

## 5. Comparing two models with the evidence

### 5.1 A common prior for shared parameters

$\ln Z$ depends on every box. Widening the box of a parameter whose posterior
sits well inside both boxes leaves the integral in (2.1) unchanged and grows
$V$, so it lowers $\ln Z$ by exactly $\ln(w^\text{new}/w^\text{old})$. aure
widens boxes itself after a boundary hit, and the modeling node restates them
every iteration. A raw comparison would therefore partly be a comparison of box
choices. For parameters present in both fits (same refl1d name), aure compares
on a common prior:

$$\Delta=\big[\ln Z_\text{new}-\ln Z_\text{base}\big]
+\sum_{j\in\text{shared}}\ln\frac{w^\text{new}_j}{w^\text{base}_j} .\tag{5.1}$$

This is exact when the narrower box does not truncate the posterior. When it
does, which is usually why the box was widened, the wider box keeps the credit
for the posterior mass it uncovered, since that mass is in the integral, but not
the penalty for its width.

Parameters only one model has **keep their own box**. That box is the Occam
factor of the structure being tested, and it depends on the range the modeling
node chose for the new layer. Under the evidence, those ranges are the prior of
the hypothesis.

### 5.2 Decision rule

$$\sigma_\Delta=\sqrt{\sigma_\text{new}^2+\sigma_\text{base}^2},\qquad
m=\max\big(\texttt{EVIDENCE\_MARGIN},\;2\sigma_\Delta\big)$$

The margin defaults to 1 nat, which is $\ln B=1$, the bottom of Kass &
Raftery's "positive".

| $\Delta$ | Verdict |
|---|---|
| $\Delta>m$ | better |
| $\Delta<-m$ | worse |
| otherwise | inconclusive |

**Parsimony inside the band.** An inconclusive comparison goes to the model
with fewer free parameters. The evidence already includes its own Occam
penalty, so this only breaks ties within the estimate's resolution.

The rule is applied in two nodes, and the two halves are complementary by
construction, so the nodes never disagree about a pair:

- **fitting.** A new trusted fit becomes the evidence baseline when it is
  better, or when it is inconclusive and either simpler or of equal $k$ with
  $\Delta>0$.
- **evaluation.** The model reverts to the baseline when the new fit is worse,
  or when it is inconclusive and more complex.

### 5.3 Which criterion decides

The evidence decides only when the latest fit and an earlier fit **both** carry
a trusted evidence. Every other comparison is decided by BIC, unchanged from
[metrics.md §4.2](metrics.md). The two baselines are kept apart: `best_bic`
covers every fit, and `best_evidence` covers trusted fits only. Whichever
criterion decided is recorded on the fit as `model_comparison`, together with
what BIC alone would have said.

**Hypothesis outcomes.**

- **Worse:** the hypothesis is `rejected`, with $\Delta\ln Z\pm\sigma$ in its
  notes.
- **Inconclusive and reverted:** the hypothesis stays `tried`, which the report
  prints as "tried, inconclusive", with a note. The data cannot tell the change
  from the simpler model at this noise level. That is not evidence that the
  structure is absent.

---

## 6. Validation

### 6.1 Closed-form checks

[`tests/test_posterior_tools.py`](../tests/test_posterior_tools.py) scores the
estimators against posteriors whose evidence is known exactly.

- **A correlated 4-D Gaussian in a wide box.** Bridge sampling lands within
  0.001 nats of the exact value, inside its own error estimate. Laplace lands
  within 0.03 nats, which is the sampling error of $\hat\Sigma$.
  $p_D=\tilde d=k$, $D^2/k\approx0$, and every check passes.
- **A half-Gaussian against a bound.** Bridge sampling is exact. Laplace is
  high by $\ln(2\sqrt{1-2/\pi})=0.19$ nats, as §3.2 predicts, and the
  parameter is listed as truncated but not flagged as an extreme.
- **Failure modes.** The tests cover two modes the chains never crossed between,
  chains stuck in different places, a population drifting together (which plain
  R̂ misses and split-R̂ catches), a frozen sampler, a best point 3σ from the
  mass, and an interior extreme percentile. Each one fails the check it should,
  and no case produces a confident wrong evidence.
- **The comparison rule.** Revert and "becomes the baseline" are never both
  true, the parsimony band behaves as specified, and widening a box the
  posterior does not touch costs nothing after the adjustment of (5.1).

### 6.2 Synthetic reflectivity against nested sampling

[`validation/evidence_study.py`](../validation/evidence_study.py) reproduces
everything in this section.

**Data.** 160 points on $Q=0.008$–$0.25$ Å⁻¹, $dQ/Q=2.5$ %, Gaussian noise.

**Scenario A: a spurious layer.**
- Truth: Si | SiO₂ 12 Å, ρ = 3.47 | polymer 220 Å, ρ = 1.6 | air, with 4 % noise.
- Models: A1 has no oxide ($k=5$). A2 is the true structure ($k=8$). A3 adds a
  spurious top layer ($k=11$).

**Scenario B: a real layer below the resolution limit.**
- Truth: Si | 5 Å interlayer, ρ = 2.40 | polymer | air, with 10 % noise.
  The resolution limit here is $2\pi/Q_\text{max}=25$ Å.
- Models: B1 has no interlayer ($k=5$). B2 is the true structure ($k=8$).

**Priors.** Every model uses the same boxes:

| Parameter | Box |
|---|---|
| Oxide or interlayer thickness | 0–40 Å |
| Oxide or interlayer SLD | 0–6 |
| Oxide or interlayer roughness | 0–10 Å |
| Spurious layer thickness | 0–50 Å |
| Spurious layer SLD | 0–6 |
| Spurious layer roughness | 0–10 Å |
| Polymer thickness | 100–400 Å |
| Polymer SLD | 0–4 |
| Polymer roughness | 0–20 Å |
| Si roughness | 0–15 Å |
| Intensity | 0.8–1.2 |

**Method.** DREAM ran exactly as aure configures it (`_build_fit_options`) at
the exploration budget, the final-polish budget, and longer ones. The reference
is `dynesty` nested sampling: slice sampling, 1 000 live points, Δln Z < 0.05,
three seeds, on the same likelihood and boxes. The mean is quoted. Seed-to-seed
scatter was 0.07–0.14 nats, except 0.42 for B2.

**Table 6.1: log-evidence in nats.** ln Z_BIC is $-\tfrac12$BIC. "Trusted"
means `evidence_trusted`.

| Model | Budget | Draws (distinct) | split-R̂ | ln Z_BIC | Laplace | Bridge ± σ | Nested | Bridge − nested | Trusted |
|---|---|---|---|---|---|---|---|---|---|
| A1 | explore | 1,000 (374) | 2.08 | −7032.2 | −7049.1 | −7050.59 ± 0.04 | −7050.82 | +0.24 | no |
| A1 | final | 10,000 (3,603) | 1.14 | −7032.2 | −7049.3 | −7050.57 ± 0.03 | −7050.82 | +0.25 | **yes** |
| A1 | long | 100,000 (35,516) | 1.02 | −7032.2 | −7049.2 | −7050.59 ± 0.02 | −7050.82 | +0.23 | **yes** |
| A2 | explore | 1,600 (140) | 11.41 | −81.1 | −91.9 | −94.19 ± 0.13 | −93.63 | −0.56 | no |
| A2 | final | 10,400 (327) | 6.87 | −80.8 | −90.2 | −94.18 ± 0.20 | −93.63 | −0.55 | no |
| A2 | long | 100,000 (3,003) | 2.24 | −80.9 | −90.7 | −94.25 ± 0.15 | −93.63 | −0.62 | no |
| A2 | 800 k | 800,000 (18,054) | 1.39 | −80.7 | −89.5 | −94.16 ± 0.14 | −93.63 | −0.53 | no |
| A3 | explore | 1,100 (126) | 51.7 | −91.2 | −91.6 | −102.65 ± 0.33 | −95.92 | −6.72 | no |
| A3 | final | 11,000 (252) | 24.0 | −88.4 | −91.4 | −97.51 ± 0.49 | −95.92 | −1.58 | no |
| A3 | long | 100,100 (1,101) | 11.5 | −88.5 | −90.2 | −97.50 ± 0.73 | −95.92 | −1.58 | no |
| B1 | explore | 1,000 (321) | 2.02 | −87.2 | −95.3 | −95.40 ± 0.02 | −95.52 | +0.12 | no |
| B1 | final | 10,000 (2,920) | 1.12 | −87.2 | −95.2 | −95.40 ± 0.01 | −95.52 | +0.11 | **yes** |
| B1 | long | 100,000 (29,816) | 1.01 | −87.2 | −95.2 | −95.40 ± 0.01 | −95.52 | +0.11 | **yes** |
| B2 | explore | 1,600 (167) | 10.35 | −89.1 | −90.3 | −93.32 ± 0.18 | −92.84 | −0.48 | no |
| B2 | final | 10,400 (362) | 6.55 | −88.7 | −88.2 | −93.48 ± 0.28 | −92.84 | −0.64 | no |
| B2 | long | 100,000 (2,621) | 3.44 | −88.8 | −87.8 | −92.74 ± 0.24 | −92.84 | +0.10 | no |
| B2 | 800 k | 800,000 (20,043) | 3.95 | −88.7 | −83.9 | −92.24 ± 0.54 | −92.84 | +0.60 | no |

**Table 6.2: the comparisons.** $\Delta$ is the first model minus the second,
in nats. The nested value is the truth.

| Comparison | Nested | Budget | Δ ln Z_BIC | Δ Laplace | Δ Bridge | What aure records |
|---|---|---|---|---|---|---|
| A2 − A3 (spurious layer) | **+2.29 ± 0.16** | explore | +10.16 | −0.30 | +8.45 ± 0.35 | BIC reverts A3, and the provisional evidence agrees, so the hypothesis is rejected |
| | | final | +7.55 | +1.28 | +3.32 ± 0.53 | same |
| | | long | +7.66 | −0.52 | +3.25 ± 0.75 | same |
| B2 − B1 (real interlayer) | **+2.68 ± 0.42** | explore | **−1.85** | +4.97 | +2.08 ± 0.18 | BIC reverts B2, and the provisional evidence **contests** it, so the hypothesis stays "tried, inconclusive" |
| | | final | **−1.52** | +6.94 | +1.92 ± 0.28 | same; B1 trusted, B2 not |
| | | long | **−1.58** | +7.38 | +2.67 ± 0.24 | same |

**Table 6.3: BIC's error decomposed.** These come from the long runs. $\varepsilon$
is the Laplace part (2.4), and $\varepsilon_\text{total}$ is measured against
nested sampling.

| Model | $k$ | $p_D$ | BIC charges $\tfrac k2\ln n$ | $\sum_j o_j$ | $o_R$ | $\varepsilon$ | $\varepsilon_\text{total}$ |
|---|---|---|---|---|---|---|---|
| A2 | 8 | 6.1 | 20.3 | 24.9 | 5.2 | +9.8 | +12.8 |
| A3 | 11 | 6.9 | 27.9 | 24.3 | 5.3 | +1.7 | +7.4 |
| B1 | 5 | 5.0 | 12.7 | 20.2 | 0.6 | +8.0 | +8.3 |
| B2 | 8 | 6.9 | 20.3 | 17.9 | 1.5 | −0.9 | +4.1 |

Per-parameter Occam factors ranged from 0.44 nats (B2's Si roughness) to 6.2
nats (B1's polymer thickness), against BIC's flat 2.54. The spurious layer's
three parameters cost 0.76–1.61 each. B2's real interlayer cost 0.92–1.09
each, and its presence loosened the Si roughness to 0.44.

### 6.3 What the study shows

1. **When the chains converge, the bridge evidence is right.** The four
   trusted rows are within 0.11–0.25 nats of nested sampling. The quoted error
   of 0.01–0.03 is a little optimistic, but the offset is far inside the 1-nat
   decision margin.
2. **When they do not, the error estimate cannot be trusted, and R̂ is what
   says so.** On the thin-oxide ridge (A2, B2) the unconverged bridge was off by
   up to 0.65 nats. On the unconstrained spurious layer (A3) it was off by
   1.6–6.7 nats while quoting 0.3–0.7. In a first pass of this study, bumps'
   `ks_converged` made both kinds of mistake. It passed an A3 draw with plain
   R̂ = 7.4 whose bridge estimate was 2.2 nats low. It failed an A1 draw with
   plain R̂ = 1.05, because the best $\ell$ had improved by 0.03 inside the
   window. That is why split-R̂ alone gates trust (§4.1).
3. **Laplace is not a safe shortcut on reflectivity posteriors.** It ran 1.3–1.7
   nats high on A1, where three parameters press against bounds, and 2–9 nats
   high on every oxide model, where the posterior is a curved ridge. It even
   reversed the sign of A2 − A3 at two of three budgets.
4. **BIC overstated every model's evidence.** The error was 4–13 nats for the
   models of Table 6.3, and 19 for A1. The polymer parameters are constrained
   to $o_j$ = 4–6 nats inside wide boxes, and the oxide ridge adds a
   correlation penalty of 5 nats. For the spurious layer, BIC got the
   direction right but was three times too confident: +7.6 against +2.3. For
   the real sub-resolution interlayer it got the direction **wrong**: it
   rejected the layer by 1.6 nats where the evidence favours it by 2.7.
   Charging $\tfrac12\ln n$ for parameters the data constrain at the ~1-nat
   level is exactly the failure §1 describes.
5. **The provisional evidence pointed the right way in every comparison and at
   every budget**, including the exploration default. That is why aure lets it
   *contest* a BIC revert (§5.3) without letting it decide one.
6. **DREAM converges slowly on ridges.** Only the two models without a thin
   layer passed $\hat R\le1.2$, and both did so at the final-polish budget.
   With a thin layer, R̂ was still 1.39 (A2) and 3.95 (B2) after 800 000 draws.
   In B2, 2 of 80 chains sat in a second, worse basin (polymer SLD ≈ 0.6, χ²
   about 29 higher) that bumps' outlier test did not remove. The interlayer's
   thickness–SLD ridge spans 0.1–33 Å.

**Cost.** DREAM used 51 000–1 600 000 likelihood evaluations per fit. Bridge
sampling added 2 000. The nested references used 0.9–2.8 million calls each, in
2–6 minutes on this 160-point problem.

### 6.4 Reproducing

```bash
python -m validation.evidence_study run --out evidence_study/      # ≈ 25 min
python -m validation.evidence_study nested --out evidence_study/   # needs dynesty; ≈ 40 min
python -m validation.evidence_study report --out evidence_study/
```

Add `--budgets explore,final,long,verylong` for the 800 000-draw rows.

---

## 7. How aure uses it

### 7.1 Where the summary is computed

Every DREAM fit gets one: the exploration loop's (`FIT_METHOD=dream`) and the
final polish's (`FIT_METHOD_FINAL=dream`). All three fit paths compute it the
same way. [`fitting._attach_posterior`](../src/aure/nodes/fitting.py) runs right
after extraction and calls
[`summarize_dream_fit`](../src/aure/tools/posterior_tools.py), which:

- reads the retained draw over the non-outlier chains;
- adds $\ln V$ from `problem.bounds()`;
- runs bridge sampling with `EVIDENCE_PROPOSALS` likelihood evaluations;
- restores the problem's parameters afterwards.

It is measurement only. A failure is logged and the fit is kept without a
summary. Fits from other optimizers have none.

### 7.2 The `posterior` block on a `FitResult`

| Field | Meaning |
|---|---|
| `draw.n_draws`, `n_unique`, `n_chains`, `n_generations` | the retained draw; distinct points show how often DREAM accepted |
| `draw.rhat`, `rhat_max`, `rhat_logp` | split-R̂ per parameter, the maximum including $\ell$, and $\ell$'s own (§4.1) |
| `mean`, `std`, `cov`, `quantiles`, `chi2_total_quantiles` | enough of the posterior to place another point in it later (§4.5) |
| `best_mahalanobis2(_per_param)`, `best_percentile` | where the best point sits (§4.2) |
| `mass_at_bound`, `truncated_by_bounds` | truncation by the box (§4.3) |
| `p_d`, `d_bayes` | constrained dimension (§2.5) |
| `occam`, `occam_correlation`, `occam_total`, `occam_bic_per_param` | (2.3) and BIC's $\tfrac12\ln n$ |
| `log_evidence_bic`, `log_evidence_laplace`, `log_evidence_bridge(_error)` | the three estimates: (2.4), §3.2 and (3.2) |
| `log_evidence`, `log_evidence_error`, `log_evidence_method` | the one to use: bridge when it ran, else Laplace |
| `bic_error_laplace`, `bic_error` | $\varepsilon$, and $\varepsilon_\text{total}$ against the bridge estimate |
| `checks`, `fit_trusted`, `evidence_trusted`, `concerns` | §4.6, with plain-language reasons |
| `formula`, `thresholds` | the convention marker and the limits that were applied |

All values are plain JSON: floats, with `null` for "no number". Checkpoints and
`final_state.json` carry the block verbatim.

### 7.3 Evaluation

`_complexity_comparison` settles **which criterion judges the latest fit**
before the LLM is called, and stores it on the fit as `model_comparison`.

- **Evidence.** Both the fit and the evidence baseline are trusted, so
  `fitting` has already compared them with (5.1)–(5.2).
- **BIC.** Otherwise, the verdict comes from BIC as before. The record also
  carries the provisional bridge comparison against the same baseline when both
  fits have one, and `contested` is set when that comparison clearly favours a
  change BIC is about to revert.

The evaluator's prompt receives:
- the convergence verdict;
- whether the best point is central;
- $p_D$;
- the evidence, with what BIC implies and by how much it is off;
- the barely constrained parameters;
- the comparison.

The guardrail acts on the record.

| Criterion | Verdict | Action | Hypothesis |
|---|---|---|---|
| evidence | worse | revert to `best_evidence_model` | `rejected`, note "Evidence guardrail …" |
| evidence | inconclusive, more complex | revert by parsimony | stays `tried`, note "Evidence inconclusive …" |
| evidence | better, or inconclusive and not more complex | keep | χ² bookkeeping as before |
| BIC | worse, not contested | revert to `best_bic_model` | `rejected` (unchanged) |
| BIC | worse, **contested** | revert to `best_bic_model` | stays `tried`, note "… provisional posterior evidence favours it …" |

### 7.4 Finalize

When every candidate in the preferred tier carries a trusted evidence,
[`finalize._rank_by_evidence`](../src/aure/nodes/finalize.py) ranks them by
$\ln Z$ on one common prior: each parameter name gets the widest box any
candidate gave it. Parsimony breaks ties within the evidence margin. This
replaces the χ²-plus-`FINAL_SELECTION_TOL` rule. `final_selection` records
`evidence_ranked`, the scores and the margin. Any other pool is ranked by χ²
exactly as before.

### 7.5 Final fit

The polish's posterior is the reported answer's posterior check, recorded as
`final_fit.posterior`. The exploration optimum is placed in it as
`final_fit.exploration_in_posterior`, which holds $D^2/k$, percentiles, the
share of draws with a lower χ², and `superseded` when more than half the draws
beat the optimum or $D^2/k>1$. The final-fit message says in plain words
whether the chains converged, whether the best point is central, what the
evidence is, and whether the exploration fit was at the minimum of its own
structure (§4.5).

### 7.6 Report surfaces

The posterior check of the reported answer is resolved one way everywhere, by
`posterior_tools.posterior_for_report`: the final polish when it ran, else the
selected fit.

| Surface | What it shows |
|---|---|
| `aure analyze` report | a "Posterior Check" block |
| `--json` | `selection.posterior` |
| `aure batch` | a one-line verdict |
| web Results tab | an alert for the displayed iteration |
| ISAAC export | a note in its context and warnings when the posterior is not trusted |

### 7.7 Configuration

| Variable | Default | Meaning |
|---|---|---|
| `EVIDENCE_PROPOSALS` | 2000 | bridge-sampling proposal draws, one likelihood evaluation each; `0` disables bridge sampling, and so the evidence |
| `EVIDENCE_RHAT_MAX` | 1.2 | split-R̂ ceiling for "converged" |
| `EVIDENCE_MAX_ERROR` | 0.5 | largest bridge error, in nats, at which an evidence may decide |
| `EVIDENCE_MARGIN` | 1.0 | smallest $\Delta\ln Z$, in nats, that decides a comparison; the effective margin is $\max(\text{margin},2\sigma_\Delta)$ |

**Budgets.** `FIT_STEPS` is the retained draw for DREAM, so the
exploration default of 1 000 is far too small for a trusted posterior. In §6,
a 5-parameter model converged at 10 000 draws, the `final_fit` default. The
thin-layer models had not converged at 800 000. For runs that need the evidence
to decide, use `FIT_METHOD_FINAL=dream` with a large `FIT_STEPS_FINAL` for the
answer, or run exploration with `FIT_METHOD=dream` and `FIT_STEPS` ≥ 10⁵ when
the extra evaluations are affordable.

Relaxing `EVIDENCE_RHAT_MAX` trades safety for coverage. In §6, $\hat R\le5$
would have admitted the thin-oxide rows, which were within 0.65 nats of the
truth, and still excluded every spurious-layer row, which was off by up to 6.7.
That is one study, so it is a judgement for the user rather than a default.

### 7.8 State

| Key | Holds |
|---|---|
| `best_evidence`, `best_evidence_model` | the evidence regression baseline, for trusted fits only |
| `best_bic_evidence` | the provisional evidence of the fit holding the BIC baseline, which the contest check reads |

Each record carries `formula`. A baseline written under another convention is
ignored, as `bic_formula` is for BIC.

---

## 8. Limitations

- **It needs a converged draw, and DREAM converges slowly on ridges.** At the
  default budgets the evidence is usually flagged untrusted, and BIC keeps
  deciding. The difference is that BIC's reliability is now measured and
  reported. §7.7 gives the budgets that converged in §6.
- **No diagnostic sees a basin the chains never visited** (§4.1). The
  exploration-versus-posterior check of §4.5 and thin-layer mode enumeration
  ([approach.md §6.8](approach.md)) are partial remedies.
- **Boxes are priors** (§5.1). The added structure's boxes enter the
  comparison directly, which is the Bayesian answer, and it makes those boxes
  part of the hypothesis.
- **$\chi^2_\text{red}\ll1$** corrupts every likelihood-based quantity equally.
  It is a statement about the `dR` column. The existing acceptance floor
  ([finalization.md](finalization.md)) still applies, and so does the
  sub-floor rule for baselines.
- **Dimension.** Bridge sampling with one Gaussian proposal loses efficiency
  as $k$ grows and as the posterior departs from Gaussian. The error estimate
  reports that loss, and a large error makes the evidence untrusted rather
  than wrong.

---

## 9. References

- Frühwirth-Schnatter, S. (2004). Estimating marginal likelihoods for mixture
  and Markov switching models using bridge sampling techniques. *The
  Econometrics Journal* 7, 143–167.
- Gelman, A. & Rubin, D. B. (1992). Inference from iterative simulation using
  multiple sequences. *Statistical Science* 7, 457–472.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. & Rubin,
  D. B. (2013). *Bayesian Data Analysis*, 3rd ed., §11.4 (split-R̂).
- Gronau, Q. F. et al. (2017). A tutorial on bridge sampling. *Journal of
  Mathematical Psychology* 81, 80–97.
- Handley, W. & Lemos, P. (2019). Quantifying dimensionality: Bayesian
  cosmological model complexities. *Physical Review D* 100, 023512.
- Kass, R. E. & Raftery, A. E. (1995). Bayes factors. *JASA* 90, 773–795.
- Kass, R. E. & Wasserman, L. (1995). A reference Bayesian test for nested
  hypotheses and its relationship to the Schwarz criterion. *JASA* 90,
  928–934.
- Kramer, A., Hasenauer, J., Allgöwer, F. & Radde, N. (2010). Computation of
  the posterior entropy in a Bayesian framework for parameter estimation in
  biological networks. *IEEE CCA*, doi:10.1109/CCA.2010.5611198.
- Meng, X.-L. & Wong, W. H. (1996). Simulating ratios of normalizing constants
  via a simple identity. *Statistica Sinica* 6, 831–860.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of
  Statistics* 6, 461–464.
- Skilling, J. (2006). Nested sampling for general Bayesian computation.
  *Bayesian Analysis* 1, 833–859.
- Spiegelhalter, D. J., Best, N. G., Carlin, B. P. & van der Linde, A. (2002).
  Bayesian measures of model complexity and fit. *JRSS B* 64, 583–639.
- Treece, B. W., Kienzle, P. A., Hoogerheide, D. P., Majkrzak, C. F., Lösche,
  M. & Heinrich, F. (2019). Optimization of reflectometry experiments using
  information theory. *J. Appl. Cryst.* 52, 47–59.
- Vrugt, J. A. (2016). Markov chain Monte Carlo simulation using the DREAM
  software package. *Environmental Modelling & Software* 75, 273–316.
