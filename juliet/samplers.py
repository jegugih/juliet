# JAX samplers used by juliet.fit. All of them work on an unconstrained parameter vector z
# (see juliet.fit.fit for the mapping between z and the physical parameters).
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def chunked_vmap(fn, batch_size):
    """
    Returns fn with a batching rule that evaluates batches in chunks of ``batch_size`` (vectorized within each chunk,
    sequentially over chunks), which bounds the memory used when samplers vmap the likelihood over many points of large
    datasets. Not differentiable under vmap (use for gradient-free samplers only).
    """
    @jax.custom_batching.custom_vmap
    def chunked(x):
        return fn(x)

    @chunked.def_vmap
    def rule(axis_size, in_batched, x):
        if not in_batched[0]:
            return fn(x), False
        return jax.lax.map(fn, x, batch_size=min(batch_size, axis_size)), True

    return chunked


def run_nested(logprior_fn, loglikelihood_fn, initial_z, rng_key, num_delete, num_inner_steps,
               dlogz=0.1, steps_per_chunk=None, max_iterations=1000000, n_posterior_samples=None, verbose=False):
    """
    Nested sampling with blackjax's (batched) Nested Slice Sampler. ``num_delete`` live points are replaced
    in parallel on every iteration. Stops when the estimated remaining evidence in the live points
    changes log(Z) by less than ``dlogz`` (same criterion as dynesty's ``dlogz``).

    Returns a dictionary with the dead points, their log-likelihoods and log-weights, log-evidence
    (and its error) and equally-weighted posterior samples (all in z-space).
    """
    import blackjax
    from blackjax.ns.base import NSInfo
    from blackjax.ns.utils import finalise, log_weights

    # Iterations are run in jit-compiled chunks; by default each chunk replaces about one full set of live points:
    if steps_per_chunk is None:
        steps_per_chunk = max(1, int(np.ceil(initial_z.shape[0] / num_delete)))

    algorithm = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_fn,
                             num_inner_steps=num_inner_steps, num_delete=num_delete)

    rng_key, init_key = jax.random.split(rng_key)
    state = algorithm.init(initial_z, rng_key=init_key)
    if not np.any(np.isfinite(np.asarray(state.particles.loglikelihood))):
        raise Exception('All the initial live points (drawn from the priors) have a zero likelihood. Check that the priors '
                        'are compatible with the data and the model (e.g., periods ordered, b < 1 + p, ecc < ecclim).')

    @jax.jit
    def run_chunk(state, key):
        def one_step(state, key):
            state, info = algorithm.step(key, state)
            return state, info.particles
        return jax.lax.scan(one_step, state, jax.random.split(key, steps_per_chunk))

    dead = []
    iteration = 0
    tstart = time.time()
    while True:
        rng_key, chunk_key = jax.random.split(rng_key)
        state, particles = run_chunk(state, chunk_key)
        # Flatten (steps, num_delete, ...) -> (steps * num_delete, ...):
        dead.append(jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), particles))
        iteration += steps_per_chunk
        logZ, logZ_live = float(state.integrator.logZ), float(state.integrator.logZ_live)
        remaining = np.logaddexp(logZ, logZ_live) - logZ if np.isfinite(logZ) else np.inf
        if verbose:
            print('\t nested sampling | iterations: {0:} | dead points: {1:} | lnZ: {2:.3f} | dlnZ remaining: {3:.3f} | {4:.1f} s'.format(
                  iteration, iteration * num_delete, logZ, remaining, time.time() - tstart), flush=True)
        if remaining < dlogz or iteration >= max_iterations:
            break

    final = finalise(state, [NSInfo(d, None) for d in dead], update_info=False)

    rng_key, w_key, s_key = jax.random.split(rng_key, 3)
    logw = log_weights(w_key, final, shape=100)
    logZs = logsumexp(logw, axis=0)
    mean_logw = np.asarray(logw.mean(axis=-1))
    mean_logw = np.where(np.isfinite(mean_logw), mean_logw, -np.inf)

    weights = np.exp(mean_logw - mean_logw.max())
    weights /= weights.sum()
    ess = 1. / np.sum(weights**2)
    if n_posterior_samples is None:
        n_posterior_samples = int(max(ess, 1000))
    idx = np.asarray(jax.random.choice(s_key, weights.shape[0], p=jnp.asarray(weights),
                                       shape=(n_posterior_samples,), replace=True))

    return {'samples': np.asarray(final.particles.position),
            'loglikelihood': np.asarray(final.particles.loglikelihood),
            'logwt': mean_logw,
            'logz': float(np.mean(logZs)),
            'logzerr': float(np.std(logZs)),
            'ess': float(ess),
            'niterations': iteration,
            'posterior_samples': np.asarray(final.particles.position)[idx]}


def run_numpyro(kernel_name, potential_fn, initial_z, rng_key, num_warmup, num_samples, num_chains,
                kernel_kwargs=None, mcmc_kwargs=None):
    """
    Runs one of numpyro's MCMC kernels on potential_fn (= -log posterior) with ``num_chains`` chains
    (walkers for the ensemble samplers) run vectorized. ``initial_z`` has shape (num_chains, ndim).
    Returns samples grouped by chain, shape (num_chains, num_samples, ndim).
    """
    from numpyro.infer import AIES, ESS, MCMC, NUTS

    kernels = {'nuts': NUTS, 'aies': AIES, 'ess': ESS}
    kernel = kernels[kernel_name](potential_fn=potential_fn, **(kernel_kwargs or {}))
    mcmc_kwargs = dict(mcmc_kwargs or {})
    mcmc_kwargs.setdefault('progress_bar', True)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method='vectorized', **mcmc_kwargs)
    mcmc.run(rng_key, init_params=jnp.asarray(initial_z))
    return np.asarray(mcmc.get_samples(group_by_chain=True)), mcmc
