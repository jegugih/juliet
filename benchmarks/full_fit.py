"""
Full fit of one benchmark problem: wall time, number of likelihood evaluations, log-evidence and posterior summary.

    python full_fit.py <problem index (0, 1, 2)> <backend: jax|legacy> <sampler> [--seed 1] [--n_live 500] [--train_procs 1]
                       [--key value ...]

e.g., python full_fit.py 0 legacy dynesty --nthreads 12;  python full_fit.py 2 jax nautilus --n_live 2000 --train_procs 4
--train_procs: number of processes training nautilus' neural networks (the likelihood stays in the main process).
"""
import sys, io, time, copy, shutil, tempfile, argparse, contextlib, warnings, multiprocessing
import numpy as np
warnings.filterwarnings('ignore')
import jax
import juliet
from problems import PROBLEMS

# Count the likelihood evaluations of the JAX samplers (batched calls report their size to the host):
counter = {'n': 0}
def counting_chunked_vmap(fn, batch_size):
    @jax.custom_batching.custom_vmap
    def f(x):
        jax.debug.callback(lambda: counter.__setitem__('n', counter['n'] + 1))
        return fn(x)
    @f.def_vmap
    def rule(axis_size, in_batched, x):
        jax.debug.callback(lambda: counter.__setitem__('n', counter['n'] + axis_size))
        return jax.lax.map(fn, x, batch_size=min(batch_size, axis_size)), True
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type=int)
    parser.add_argument('backend')
    parser.add_argument('sampler')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_live', type=int, default=500)
    parser.add_argument('--train_procs', type=int, default=1)
    args, unknown = parser.parse_known_args()
    extra = {k.lstrip('-'): (int(v) if v.lstrip('-').isdigit() else float(v) if v.replace('.', '', 1).replace('e-', '').isdigit() else v)
             for k, v in zip(unknown[::2], unknown[1::2])}
    name, make = list(PROBLEMS.items())[args.problem]
    priors, kw = make()
    out = tempfile.mkdtemp() + '/'
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        data = juliet.load(priors=copy.deepcopy(priors), out_folder=out, backend=args.backend, **copy.deepcopy(kw))
        fit_kwargs = dict(sampler=args.sampler, n_live_points=args.n_live, **extra)
        if args.backend == 'jax':
            fit_kwargs['seed'] = args.seed
        if args.sampler == 'nautilus' and args.train_procs > 1:
            fit_kwargs['pool'] = (None, multiprocessing.get_context('fork').Pool(args.train_procs))
        results = data.fit(**fit_kwargs)
    elapsed = time.time() - t0
    post = results.posteriors
    if args.backend == 'legacy' and 'dynesty' in args.sampler:
        counter['n'] = int(np.sum(post['dynesty_output'].ncall))
    ps = post['posterior_samples']
    summary = ' '.join(f'{k}={np.median(ps[k]):.6g}+-{np.std(ps[k]):.2g}' for k in [k for k in ps if k not in ('unnamed', 'loglike')][:6])
    print(f'{name} | {args.backend} {args.sampler} seed={args.seed} n_live={args.n_live} train_procs={args.train_procs} {extra} | {jax.devices()[0].platform} | '
          f'{elapsed:9.1f} s | {counter["n"]} likelihood evaluations | lnZ = {post.get("lnZ", float("nan")):.3f} +- '
          f'{post.get("lnZerr", float("nan")):.3f} | {summary}')
    shutil.rmtree(out, ignore_errors=True)


if __name__ == '__main__':
    # (guarded: the original juliet's nthreads option starts worker processes that import this script)
    sys.modules['juliet.fit'].chunked_vmap = counting_chunked_vmap
    main()
