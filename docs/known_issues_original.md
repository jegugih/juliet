# Issues found in juliet 2.2.10 and its dependencies

This document lists the bugs and pitfalls found while porting juliet to JAX. Every entry was **reproduced** on the
original code (juliet 2.2.10, git `HEAD` before the port) or on the dependency in question. Environment: Python 3.14,
NumPy 2.5.3, astropy 8.0.1, matplotlib 3.11.2, batman 2.5.2, catwoman 1.1.0, celerite 0.4.3, celerite2 0.3.3,
george 0.4.4, dynesty 3.1.0, emcee 3.1.6, kelp 0.5.0 (PyPI) / `main` (commit 219a922), numpyro 0.22.0,
jax 0.11.2, jaxoplanet 0.1.0.

**Status** refers to the two backends of the ported juliet: `jax` (default) and `legacy` (the original implementation,
selected with `juliet.load(..., backend='legacy')`). The legacy backend only received fixes for bugs that crash or
return clearly wrong outputs; modelling conventions were left untouched so that previous results are reproduced.

## Summary

| ID | Component | Severity | Symptom | jax backend | legacy backend |
|----|-----------|----------|---------|-------------|----------------|
| J1 | juliet | crash | Global models without a GP: log-likelihood is `None` | fixed | fixed |
| J2 | juliet | crash | `extra_loglikelihood` raises `NameError` | fixed | fixed |
| J3 | juliet | crash | Re-loading posteriors with `sigma_w_rv_*` names raises `AttributeError` | fixed | fixed |
| J4 | juliet | minor | Unknown model type raises `NameError` instead of the intended error | fixed | fixed |
| J5 | juliet | wrong output | `evaluate()` returns a zero model if the posterior has fewer samples than `nsamples` | fixed | fixed |
| J6 | juliet | crash / minor | `get_quantiles` fails for small sample sizes; even-size median off by half a step | crash fixed | crash fixed |
| J7 | juliet | wrong output | Beta prior log-density is `-inf` whenever `b >= 1` (MCMC samplers) | fixed | fixed |
| J8 | juliet | crash | Exponential prior crashes dynesty with NumPy 2 | fixed | fixed |
| J9 | juliet | crash | Prior dictionaries with `sigma_w_rv_*` names turn RV models global, then `KeyError` | fixed | fixed |
| J10 | juliet | side effect | `juliet.load` sorts the user's GP regressor array in place | fixed | fixed |
| J11 | juliet | crash | `evaluate(return_components=True)` with several RV instruments | fixed | not fixed |
| J12 | juliet | crash | `evaluate(t=...)` with non-linear functions | clear error | not fixed |
| J13 | juliet | crash | `evaluate(t=...)` with TTVs | fixed | not fixed |
| J14 | juliet | crash | `juliet.plots` fails to import with current matplotlib | fixed | fixed |
| J15 | juliet | packaging | `setup.py` uses `extras_requires` (ignored) | fixed | fixed |
| J16 | juliet | wrong model | Exp-sine-squared GP kernel uses `log(GP_Gamma)` | fixed (option) | fixed (option) |
| J17 | juliet + george | wrong model | Multi-dimensional SE / Matern-3/2 GP kernels have variance `nX * GP_sigma**2` | fixed (option) | fixed (option) |
| J18 | juliet + george | inaccurate | Default HODLR solver gives approximate likelihoods (errors > 100 in lnL observed) | exact solver | unchanged default |
| J19 | juliet + batman | convention | Phase-curve phases measured from a `t0` silently changed by batman | reproduced | unchanged |
| J20 | juliet + batman | wrong output | Fixed `t_secondary`: eclipse placement changes after the first evaluation | fixed | unchanged |
| J21 | juliet + batman | inaccurate | Non-quadratic limb-darkening laws: 2-7 ppm errors | accurate (1e-9) | unchanged |
| J22 | juliet | dead code | `CeleriteTripleSHOKernel` can never be selected | not used | unchanged |
| J23 | juliet + dynesty | biased | Evidences from juliet's default dynesty settings are biased low (0.4-2.4 nats verified) | nautilus available | unchanged |
| B1 | batman | side effect | Secondary-eclipse `light_curve` overwrites `params.t0` | - | - |
| B2 | batman | stateful | Secondary-eclipse positions depend on whether `t_secondary` changed since the last call | - | - |
| B3 | batman | inaccurate | Integration step fixed at initialization; `max_err` not honoured after parameters change | - | - |
| B4 | batman | inaccurate | Exponential law: ~50 ppm errors near the stellar limb | - | - |
| B5 | batman | packaging | Imports `distutils` (removed in Python 3.12) | - | - |
| G1 | george | wrong output | `ConstantKernel` evaluates to `ndim * exp(log_constant)` for multi-dimensional inputs | - | - |
| C1 | celerite2 | limitation | JAX operations: no `vmap` rules, CPU only, no forward-mode derivatives | not used | - |
| K1-K6 | kelp | packaging / crash | Released kelp does not install or import with current Python / NumPy / JAX / astropy | vendored, fixed | vendored, fixed |
| N1 | numpyro | pitfall | ESS shuffles walkers between iterations by default | handled | - |
| N2 | numpyro | pitfall | NUTS mass-matrix regularization breaks on parameters with tiny posterior widths | handled | - |
| X1 | blackjax | pitfall | Nested slice sampling: `lnZ` scatter ~4x larger than its error estimate on multimodal posteriors | documented | - |

## juliet

**J1. Global models without a GP return no log-likelihood.** In `model.get_log_likelihood`, the branch for global
models without a GP computes `self.gaussian_log_likelihood(...)` but does not `return` it, so the method returns
`None` and `fit.loglike` fails with a `TypeError`. Any global lightcurve or RV model without a GP was unusable.
*Reproduction:* two lightcurve instruments plus a prior containing the `lc` token: `get_log_likelihood` returned `None`.

**J2. `extra_loglikelihood` raises `NameError`.** `fit.loglike` calls `extra_loglikelihood['loglikelihood'](...)`, an
undefined global name, instead of `self.extra_loglikelihood`. Any fit using this option crashed at the first
likelihood evaluation (reproduced with dynesty).

**J3. Re-loading posteriors with `sigma_w_rv_*` names crashes.** The back-compatibility branch of `fit.__init__` uses
`self.sigmaw_iname`, an attribute of `model`, not `fit`: `AttributeError: 'fit' object has no attribute 'sigmaw_iname'`.

**J4. Unknown model types raise `NameError`.** `model.__init__` builds its error message with an undefined variable
`lc` (should be `modeltype`).

**J5. `evaluate()` returns a zero model when the posterior has fewer samples than `nsamples`.** The arrays holding the
model samples have `nsamples` rows (default 1000), but only `len(posterior)` rows are filled; the median over the
remaining zero rows is 0. Reproduced with 100 posterior samples: the "median model" was exactly 0 everywhere.

**J6. `get_quantiles` fails for small sample sizes.** The index `median + nsamples * alpha / 2 + 1` exceeds the array
for small `nsamples` (`IndexError` for 3 and 10 samples). Also, for an even number of samples the "median" averages
the order statistics `n/2` and `n/2 + 1` instead of `n/2 - 1` and `n/2` (a bias of half a sample spacing, negligible
for typical posteriors; kept for compatibility).

**J7. Beta prior log-density is `-inf` whenever `b >= 1`.** `evaluate_beta` checks `a > 0 and b < 1` (the
hyperparameters) instead of `0 < x < 1`, so e.g. `Beta(2, 5)` returns `-inf` for every value. This only affects the
MCMC samplers (emcee, zeus), which use the `evaluate_*` functions; nested samplers use the prior transforms.

**J8. Exponential priors crash dynesty with NumPy 2.** Hyperparameters of exponential priors are stored as a
one-element list, so `transform_exponential` and `evaluate_exponential` return one-element arrays. With NumPy 2,
assigning them into the parameter vector raises `ValueError: setting an array element with a sequence` (reproduced
with dynesty; emcee ran).

**J9. Prior dictionaries with `sigma_w_rv_*` names break RV fits.** `readpriors` renames `sigma_w_rv_<inst>` to
`sigma_w_<inst>` for prior *files*, but not for prior *dictionaries*. The `rv` token in the name then makes juliet
treat the RV model as global, and the likelihood fails with `KeyError: 'sigma_w_X'`.

**J10. `juliet.load` modifies the user's GP regressors in place.** For global models with celerite kernels, `sort_GP`
sorts `GP_regressors_lc['lc']` / `GP_regressors_rv['rv']` in place, silently changing the user's array.

**J11. `evaluate(..., return_components=True)` crashes with several RV instruments.** For non-global RV models the
planet and trend components are computed on the times of all instruments and then stored into arrays sized for one
instrument: `ValueError: could not broadcast input array from shape (27,) into shape (15,)`.

**J12. `evaluate(t=...)` crashes with non-linear functions.** The non-linear model is always evaluated on the regressor
of the data, so its length does not match the new times (`ValueError: operands could not be broadcast together`). The
jax backend raises an explicit error instead (there is no regressor for the new times).

**J13. `evaluate(t=...)` crashes with TTVs.** The TTV-shifted times are computed from the data times while the model
is evaluated at the new times (`ValueError: operands could not be broadcast together`).

**J14. `juliet.plots` does not import.** It imports `mpl_toolkits.axes_grid`, removed from matplotlib (now
`mpl_toolkits.axes_grid1`).

**J15. Optional dependencies are ignored.** `setup.py` passes `extras_requires`, which setuptools ignores (the keyword
is `extras_require`).

**J16. The exp-sine-squared kernel uses `log(GP_Gamma)` as its amplitude.** george's `ExpSine2Kernel` has a *linear*
`gamma` parameter (only the period is logarithmic), but juliet set it to `log(GP_Gamma)`. The fitted kernel was
`exp(-alpha tau^2 / 2 - ln(GP_Gamma) sin^2(pi tau / P))` instead of the documented
`exp(... - GP_Gamma sin^2(pi tau / P))`; for `GP_Gamma < 1` the coefficient of the sine term is negative. Verified
against george directly. *Resolution:* fixed by default; `juliet.load(..., legacy_gp_parametrization=True)`
restores the old behaviour (the default of the legacy backend).

**J17. Multi-dimensional squared-exponential and Matern-3/2 kernels have variance `nX * GP_sigma**2`.** juliet builds
these kernels as `1. * george.kernels.ExpSquaredKernel(..., ndim=nX)`; because of G1, the constant kernel evaluates to
`nX * GP_sigma**2` after `set_parameter_vector`. For one regressor (`nX = 1`) there is no effect. *Resolution:* as J16.

**J18. The default george solver is approximate.** juliet uses george's HODLR solver by default (`george_hodlr=True`).
For a squared-exponential kernel on 400 points the log-likelihood was off by more than 100 compared with the exact
solver (which matched a direct NumPy calculation to 1e-12). The jax backend computes dense GPs exactly; the legacy
backend keeps the old default.

**J19. Phase-curve phases are measured from a `t0` changed by batman.** The sinusoidal and kelp phase curves use
`params.t0`, but batman's secondary-eclipse model overwrites `params.t0` with the time of transit implied by
`t_secondary` (B1). The phase curves were therefore phased with the eclipse ephemeris, not with the fitted `t0` (they
coincide only if `t_secondary` is consistent with the orbit). The jax backend reproduces this explicitly.

**J20. With a fixed `t_secondary`, the eclipse moves after the first evaluation.** Because of B1 and B2, the first
likelihood evaluation places the eclipse using `t_secondary`, and later evaluations (with the same parameters) use
`t0` and the orbit instead. Reproduced on an eccentric orbit: two evaluations with identical parameters differed by
5e-4 in relative flux (the full eclipse depth). The jax backend always uses `t_secondary`.

**J21. Non-quadratic limb-darkening laws are less accurate than requested.** batman's integration step is computed
once, for the coefficients juliet uses at initialization (B3). During fits, the actual errors were 2-7 ppm (nominal
`max_err` is 1 ppm) for the square-root, power-2 and non-linear laws, and up to ~50 ppm near the limb for the
exponential law (B4). The jax backend integrates to ~1e-9 (verified against adaptive quadrature).

**J22. `CeleriteTripleSHOKernel` is dead code.** `set_parameter_vector` has a branch for it, but the kernel is never
defined or selectable.

**J23. Evidences from juliet's default dynesty settings are biased low.** With juliet's defaults (`bound='multi'`,
`sample='rwalk'`, 500 live points), dynesty's log-evidence was compared with independent importance-sampling references
(multivariate-t proposals, effective sample sizes of 10^4-10^5, statistical errors ~0.001-0.01): for a 7-parameter transit
fit it was 0.44 and 0.76 nats low (two runs; quoted error 0.49), and for an 11-parameter fit of 18,000 TESS points with a
GP it was 2.4 and 1.2 nats low (two runs; quoted errors 0.51 and 0.50). On the 29-parameter K2-32 tutorial fit, two runs were 8.0 and 5.2 nats below nautilus (no
independent reference was computed for that case; quoted errors 0.48 and 0.55). Such offsets can flip model comparisons. nautilus matched the references
to 0.004 and 0.016 nats.

Also note: `evaluate(..., evaluate_transit=True)` switches off the linear models and GPs but keeps non-linear
functions; both backends keep this behaviour.

## Dependencies

**B1 (batman). `light_curve` mutates its input.** For secondary-eclipse models, `TransitModel.light_curve(params)`
sets `params.t0 = self.get_t_conjunction(params)` whenever `t_secondary` changed, overwriting the user's `t0` on the
shared parameter object.

**B2 (batman). Secondary-eclipse positions depend on the previous call.** Positions are recomputed from `t_secondary`
only if `t_secondary` changed since the last call; otherwise, if another parameter (e.g., `t0`) changed, they are
computed from `params.t0`. The same parameters can give different light curves depending on call history.

**B3 (batman). `max_err` is only honoured for the initial parameters.** For laws integrated numerically, the step-size
factor `fac` is computed in `TransitModel.__init__` for the initial parameters and not updated when `params.u` or
`params.rp` change.

**B4 (batman). Exponential limb darkening near the limb.** Even with `max_err=0.001`, the exponential law is off by up
to 5e-5 near the stellar limb (the intensity diverges as mu -> 0), compared with adaptive quadrature (which agrees with
the jax implementation to 1e-11).

**B5 (batman). `import batman` fails on Python >= 3.12 without setuptools**, because `batman/openmp.py` imports
`distutils`.

**G1 (george). `ConstantKernel` is wrong for multi-dimensional inputs.** `ConstantKernel(log_constant=c, ndim=n)`
evaluates to `n * exp(c)` (e.g., 2.0 instead of 1.0 for `c = 0`, `ndim = 2`); `1. * Kernel(ndim=n)` picks up the factor
after `set_parameter_vector`.

**C1 (celerite2). JAX interface limitations.** celerite2's JAX operations have no batching (`vmap`) rules, are only
lowered for CPUs, and forward-mode differentiation is not implemented (`Evaluation rule for 'celerite2_factor_jvp' not
implemented`). juliet now uses celerite2's kernel definitions with a pure-JAX implementation of the celerite algorithm.

**K1-K6 (kelp).** The released kelp (0.5.0) pins `cython==0.29.14` as a build requirement, which fails on Python >= 3.13
(K1, the `cgi` module was removed), and its `_astropy_init.py` uses an astropy function removed in recent versions
(K2; fixed on kelp's `main`, unreleased). kelp's `main` still uses `from jax.config import config` (K3, removed in JAX
0.4.25) and `numpy.cast` / `numpy.trapz` (K4, removed in NumPy 2). `reflected_phase_curve_inhomogeneous` uses NumPy on
its inputs, so it cannot be traced or jit-compiled by JAX (K5), and `reflected_phase_curve` returns NaN at phases
exactly 0 and 0.5 (K6). juliet now vendors kelp's JAX phase curves (`juliet/kelp_jax.py`) with K3-K5 fixed.

**N1 (numpyro). ESS shuffles walkers.** `numpyro.infer.ESS` defaults to `randomize_split=True`, which permutes the
walkers on every iteration without restoring their order, so the per-walker chains are not walker trajectories (stuck
walkers then appear as outliers in every chain). juliet uses `randomize_split=False` by default.

**N2 (numpyro). NUTS mass-matrix regularization.** NUTS shrinks the adapted mass matrix towards 1e-3, which dominates
for parameters with tiny posterior variances (e.g., `t0` with ~1e-10 day^2): trajectories hit the maximum tree depth
(~840 leapfrog steps per iteration instead of ~6). juliet adapts a dense mass matrix without regularization by default.

**X1 (blackjax). Nested slice sampling evidences on multimodal posteriors.** On an RV problem with three separated
posterior modes, 12 runs of `blackjax.nss` (with 100 or 500 points replaced per iteration and 10 or 30 slice steps per
point) gave log-evidences with a scatter of ~0.45 nats, while the reported errors were ~0.11-0.14 (nautilus and dynesty
agreed with the mean). New points rarely move between modes, so the share of live points in each mode fluctuates.

Accuracy notes (not bugs): jaxoplanet's limb-darkening solver with its default quadrature order (10) is accurate to
~2e-8 in relative flux (juliet uses order 20, ~1e-9); batman's analytic quadratic law is accurate to ~5e-9.
