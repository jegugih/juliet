juliet --- a versatile modelling tool for transiting and non-transiting exoplanetary systems
---

![Juliet logo](juliet.png?raw=true)

Authors: Néstor Espinoza (nespinoza@stsci.edu) & Diana Kossakowski (kossakowski@mpia.de)

> For usage and dependencies, **check the documentation at http://juliet.readthedocs.io.**
> Don't forget to [check out the paper](https://arxiv.org/abs/1812.08549).
> For citation, see [this link](https://github.com/nespinoza/juliet/wiki/Citing-the-code).

### JAX backend

This version of `juliet` runs entirely on [JAX](https://github.com/jax-ml/jax): lightcurves (transits, eclipses and phase curves) are computed with [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet), radial velocities with `jaxoplanet`'s Keplerian systems and Gaussian Processes with [celerite2](https://github.com/exoplanet-dev/celerite2)'s kernels and a pure-JAX implementation of the celerite algorithm (the former `george` kernels use dense JAX linear algebra). The likelihood is a single jit-compiled function, and posteriors are sampled with JAX samplers:

```python
results = dataset.fit(sampler = 'nested')   # default: batched nested slice sampling (blackjax); returns lnZ
results = dataset.fit(sampler = 'nuts')     # numpyro's No-U-Turn sampler
results = dataset.fit(sampler = 'emcee')    # numpyro's affine-invariant ensemble sampler
results = dataset.fit(sampler = 'zeus')     # numpyro's ensemble slice sampler
results = dataset.fit(sampler = 'nautilus') # nautilus (neural-network-boosted importance nested sampling); returns lnZ
```

The original implementation (`batman`, `catwoman`, `radvel`, `george`, `celerite`; MultiNest, dynesty, UltraNest, emcee and zeus) is still available with `juliet.load(..., backend = 'legacy')`, which requires those packages.

Install the dependencies with `pip install jax jaxoplanet celerite2 numpyro blackjax astropy scipy h5py` (use `jax[cuda12]` to run on NVIDIA GPUs). Everything runs on CPUs and GPUs; GPUs pay off for large datasets and batched samplers (e.g., the nested sampler), while small fits are usually faster on CPUs (`JAX_PLATFORMS=cpu`). On GPUs, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid JAX reserving most of the GPU memory.

Linear and quadratic limb-darkening are computed exactly with `jaxoplanet`; the square-root, logarithmic, exponential, power-2 and non-linear laws, as well as `catwoman`-like transits of planets with asymmetric limbs, by numerical integration in JAX. `kelp` phase curves are included in `juliet` (vendored from `kelp`). Functions passed through `non_linear_functions` or `extra_loglikelihood` are best written with `jax.numpy`; NumPy functions also work (evaluated through `jax.pure_callback`), except with `sampler = 'nuts'`.

Two GP kernels were parametrized incorrectly in juliet <= 2.2.10 (the exp-sine-squared kernel used `log(GP_Gamma)`, and multi-dimensional squared-exponential and Matern 3/2 kernels had a variance of `nX * GP_sigma**2`); this is fixed, and `juliet.load(..., legacy_gp_parametrization = True)` reproduces the old behaviour (the default with `backend = 'legacy'`). See [docs/known_issues_original.md](docs/known_issues_original.md) for all the issues found in juliet 2.2.10 and its dependencies.

Acknowledgments: We would like to thank our referee, Daniel Foreman-Mackey (https://github.com/dfm) for very useful code and paper review and suggestions. Also to Raphael Gyory (https://github.com/raphaelgyory) for helping us with releasing the pip package and readthedocs for the project.
