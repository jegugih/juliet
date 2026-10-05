# Benchmarks

Scripts to time juliet's JAX backend against the original implementation (`backend='legacy'`, which needs batman,
radvel, celerite, george and dynesty installed). Run them from this folder.

```bash
# Likelihood throughput (add --legacy to also time the original implementation, one call at a time)
JAX_PLATFORMS=cpu python likelihood_throughput.py --legacy            # on CPUs
XLA_PYTHON_CLIENT_PREALLOCATE=false python likelihood_throughput.py --batch 250 1000 4096   # on a GPU

# Full fits: python full_fit.py <problem> <backend> <sampler> [--n_live N] [--seed S] [--any_fit_keyword value]
python full_fit.py 0 legacy dynesty                  # original juliet (dynesty, 500 live points)
python full_fit.py 1 legacy dynesty --nthreads 12    # original juliet, dynesty with 12 processes
python full_fit.py 0 jax nested                      # JAX nested slice sampling
python full_fit.py 2 jax nautilus --n_live 2000 --train_procs 4   # nautilus, networks trained by 4 processes
```

Problems (`problems.py`): 0 = small transit (300 points, 7 free parameters); 1 = one TESS sector at 2-minute cadence
with a transit and a Matern GP (18,000 points, 11 parameters); 2 = the K2-32 tutorial dataset (3 planets; K2 long-cadence
photometry supersampled 20x with a GP, plus RVs from 3 instruments; 29 parameters).

Tips for large GPUs (e.g., H100): batches of likelihood evaluations are what make GPUs fast, so use more live points
(e.g., `--n_live 4000`, which also gives more accurate evidences; the nested sampler replaces `n_live/2` points per
iteration), a large nautilus batch (`--n_batch 4096`), and, if memory allows, a larger `batch_size` (the default uses
~1/5 of the GPU memory per batch). nautilus trains its neural networks on the CPU; train them in parallel with
`pool=(None, multiprocessing.get_context('fork').Pool(4))` in `fit` (4 = nautilus' default number of networks).
celerite GPs are evaluated with a sequential recursion whose cost on GPUs is set by
latency, so they only become efficient with large batches.

## Results on a laptop (Intel i9-12900HK, NVIDIA RTX 3080 Ti Laptop GPU 16 GB; October 2026)

Likelihood cost per evaluation (original = juliet 2.2.10 with batman/celerite, one call at a time on one core; batched =
how the JAX samplers evaluate it):

| Problem | original | JAX, single call | JAX CPU, batched | JAX GPU, batched |
|---|---|---|---|---|
| small transit (300 points) | 0.033 ms | 0.107 ms | 0.023 ms (1.4x) | 0.004 ms (8x) |
| TESS sector + GP (18,000 points) | 1.44 ms | 3.57 ms | 1.45 ms (1.0x) | 1.32 ms (1.1x) |
| K2-32 joint (73,280 supersampled points + GP + RV) | 6.0 ms | 13.1 ms | 9.9 ms (0.6x) | 3.1 ms (1.9x) |

Full fits (`n_live = 500` unless noted; evidences compared with independent importance-sampling references):

| Problem | Sampler | Device | Wall time | Likelihood evaluations | lnZ |
|---|---|---|---|---|---|
| small transit | original juliet + dynesty | CPU (1 core) | 103 s | 566k | 2317.21 / 2316.89 (+-0.49) |
| small transit | original juliet + dynesty (`nthreads = 4`) | CPU (4 processes) | 52 s | 568k | 2317.08 +- 0.49 |
| small transit | JAX nested (nss) | CPU | 64 s | 3.27M | 2317.50 / 2317.21 (+-0.33) |
| small transit | JAX nested (nss) | GPU | 28 s | 3.27M | 2317.50 (+-0.33) |
| small transit | nautilus (`n_live = 1000`) | CPU / GPU | 150 s / 162 s | 139k | 2317.648 / 2317.648 |
| small transit | importance-sampling reference | | | 1M | 2317.652 +- 0.001 |
| TESS + GP | original juliet + dynesty | CPU (1 core) | 1095 s | 695k | 107705.62 +- 0.51 |
| TESS + GP | original juliet + dynesty (`nthreads = 12`) | CPU (12 processes) | 388 s | 679k | 107706.88 +- 0.50 |
| TESS + GP | nautilus (`n_live = 1000`) | CPU / GPU | 617 s / 632 s | 162k | 107708.060 |
| TESS + GP | nautilus (`n_live = 1000`), networks trained by 4 processes | GPU | 379 s | 162k | 107708.062 |
| TESS + GP | importance-sampling reference | | | 300k | 107708.044 +- 0.010 |
| K2-32 | original juliet + dynesty | CPU (1 core) | 6006 s | 1.45M | 13539.98 +- 0.48 |
| K2-32 | original juliet + dynesty (`nthreads = 12`) | CPU (12 processes) | 1102 s | 1.31M | 13542.77 +- 0.55 |
| K2-32 | nautilus (`n_live = 2000`) | GPU | 6755 s | 367k | 13548.006 +- 0.010 |
| K2-32 | nautilus (`n_live = 2000`), networks trained by 4 processes | GPU | 3093 s | 367k | 13548.007 |

(Two values: two seeds.) The JAX nested sampler needs ~6x more likelihood evaluations than dynesty, so it is the fastest
option for cheap likelihoods (especially on GPUs) but not for expensive ones. nautilus needs ~4x fewer evaluations than
dynesty and gives evidences accurate to ~0.01-0.02, but its run time is dominated by training its neural networks on one
CPU core (minutes, growing with `n_live` and the number of parameters), which GPUs do not accelerate.
For K2-32, likelihood evaluations took ~1140 s of the nautilus runs; the rest was neural-network training (~5600 s on one
core, ~1950 s with 4 processes), which GPUs do not accelerate. The JAX nested sampler would need an estimated 20-30M
evaluations (about a day on this GPU). The two dynesty evidences differ by 2.8 nats (quoted errors ~0.5) and are 5.2 and
8.0 nats below nautilus's, consistent with dynesty's 0.4-2.4 nat underestimates against the references above. With
parallel processes, the original juliet is the fastest option here for expensive likelihoods (12 processes gave 2.8x on
TESS and 5.4x on K2-32), but its evidences are biased.

### Multimodal posteriors

An RV fit with a daily alias and wide priors on P and t0 has three separated posterior modes. All nested samplers found
all three. emcee (50 walkers x 20,000 steps) and NUTS stayed in the mode they started in, so longer MCMC chains do not fix
this. Log-evidences (two seeds): nautilus -70.738 / -70.682 (quoted error 0.01, so the true scatter is larger); dynesty
-70.767 / -70.386 (+-0.27); JAX nested -71.34 / -70.15. Across 12 JAX nested runs with different settings, the evidence
scatter was ~0.45 nats against quoted errors of ~0.11-0.14.

