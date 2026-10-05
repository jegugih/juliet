"""
Likelihood throughput of juliet's JAX backend (single calls and batched calls, as used by the samplers) and, with
--legacy, of the legacy backend (the original implementation, one call at a time, as used by dynesty/emcee).

    python likelihood_throughput.py [--legacy] [--batch 250 1000 4096] [--problems 0 1 2]

Runs on the default JAX device (set JAX_PLATFORMS=cpu to use CPUs; on GPUs, XLA_PYTHON_CLIENT_PREALLOCATE=false).
"""
import sys, io, time, copy, argparse, contextlib, warnings
import numpy as np
warnings.filterwarnings('ignore')
import jax
import jax.numpy as jnp
import juliet
from juliet.samplers import chunked_vmap
from problems import PROBLEMS

parser = argparse.ArgumentParser()
parser.add_argument('--legacy', action='store_true')
parser.add_argument('--batch', type=int, nargs='+', default=[250, 1000])
parser.add_argument('--problems', type=int, nargs='+', default=None, help='problem indices (default: all)')
args = parser.parse_args()
jf = sys.modules['juliet.fit']


class LikelihoodOnly(jf.fit):
    """juliet.fit without running a sampler (sets up the priors and models only)."""
    def __init__(self, data):
        self.data, self.extra_loglikelihood_boolean, self.sampler = data, False, 'nested'
        self.paramnames = [p for p in data.priors if data.priors[p]['distribution'].lower() != 'fixed']
        self.fixed_values = {p: float(data.priors[p]['hyperparameters']) for p in data.priors
                             if data.priors[p]['distribution'].lower() == 'fixed'}
        self.prior_dists = [jf.jm.prior_distribution(data.priors[p]['distribution'], data.priors[p]['hyperparameters'])
                            for p in self.paramnames]
        self.transforms = [jf.biject_to(d.support) for d in self.prior_dists]
        if data.t_lc is not None:
            self.lc = jf.model(data, 'lc')
        if data.t_rv is not None:
            self.rv = jf.model(data, 'rv')
        self.batch_size = min(m.batch_size for m in [getattr(self, 'lc', None), getattr(self, 'rv', None)] if m is not None)


def timeit(fn, x, n):
    jax.block_until_ready(fn(x))
    t0 = time.time()
    for _ in range(n):
        out = fn(x)
    jax.block_until_ready(out)
    return (time.time() - t0) / n


print(f'device: {jax.devices()[0]}')
for index, (name, make) in enumerate(PROBLEMS.items()):
    if args.problems is not None and index not in args.problems:
        continue
    priors, kw = make()
    with contextlib.redirect_stdout(io.StringIO()):
        data = juliet.load(priors=copy.deepcopy(priors), **copy.deepcopy(kw))
    f = LikelihoodOnly(data)
    # Valid prior draws (finite likelihood):
    nmax = max(args.batch)
    x = np.asarray(f._sample_prior(jax.random.PRNGKey(0), 4 * nmax))
    ll = np.asarray(jax.lax.map(jax.jit(f._loglike_x), jnp.asarray(x), batch_size=f.batch_size))
    X = x[np.isfinite(ll)]
    single = jax.jit(f._loglike_x)
    line = f'{name:52s} | single call: {timeit(single, jnp.asarray(X[0]), 30) * 1e3:9.3f} ms | chunk {f.batch_size}'
    batched = jax.jit(jax.vmap(chunked_vmap(jax.jit(f._loglike_x), f.batch_size)))
    for nb in args.batch:
        if nb <= len(X):
            line += f' | batch {nb}: {timeit(batched, jnp.asarray(X[:nb]), 3) / nb * 1e3:9.4f} ms/eval'
    if args.legacy:
        with contextlib.redirect_stdout(io.StringIO()):
            dold = juliet.load(priors=copy.deepcopy(priors), backend='legacy', **copy.deepcopy(kw))
            models = [juliet.model(dold, mt, log_like_calc=True) for mt in ['lc', 'rv'] if getattr(dold, 't_' + mt) is not None]
        fixed = {p: v['hyperparameters'] for p, v in dold.priors.items() if v['distribution'] == 'fixed'}
        pvs = [dict(fixed, **dict(zip(f.paramnames, xi))) for xi in X[:200]]
        t0 = time.time()
        for pv in pvs:
            for m in models:
                m.generate(pv)
                if m.modelOK:
                    m.get_log_likelihood(pv)
        line += f' | legacy (original): {(time.time() - t0) / len(pvs) * 1e3:8.3f} ms/eval'
    print(line, flush=True)
