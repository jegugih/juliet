"""Benchmark problems: each function returns (priors, keyword arguments of juliet.load)."""
import os
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '')


def transit(t, t0=0., P=1., a=3.6, inc=90., rp=0.1, u=(0.2, 0.3)):
    """Circular-orbit transit (simulated with juliet's JAX model, so no legacy packages are needed)."""
    import jax.numpy as jnp
    from juliet import jaxmodels as jm
    tp = jm.time_of_periastron(t0, P, 0., np.pi / 2.)
    sini, cosi = np.sin(np.deg2rad(inc)), np.cos(np.deg2rad(inc))
    d, z, _, _ = jm.sky_position(jnp.asarray(t), tp, P, a, 0., 1., 0., sini, cosi)
    return np.asarray(jm.transit_flux(d, z, rp, jnp.asarray(u)))


def small_transit():
    rng = np.random.default_rng(42)
    t = np.linspace(-0.1, 0.1, 300)
    y = transit(t) + rng.normal(0, 100e-6, t.size)
    params = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_inst', 'q2_inst', 'ecc_p1', 'omega_p1', 'a_p1', 'mdilution_inst', 'mflux_inst', 'sigma_w_inst']
    dists = ['fixed', 'normal', 'uniform', 'fixed', 'uniform', 'uniform', 'fixed', 'fixed', 'normal', 'fixed', 'normal', 'loguniform']
    hyperps = [1.0, [0.0, 0.1], [0., 1], 0.0, [0., 1.], [0., 1.], 0.0, 90., [3.6, 0.1], 1.0, [0., 0.1], [0.1, 1000.]]
    priors = dict(zip(params, [{'distribution': d, 'hyperparameters': h} for d, h in zip(dists, hyperps)]))
    return priors, dict(t_lc={'inst': t}, y_lc={'inst': y}, yerr_lc={'inst': np.full(t.size, 10e-6)})


def tess_sector():
    """One TESS sector at 2-minute cadence (18,000 points, 11 free parameters): a P = 3.3 d hot Jupiter plus correlated noise (Matern GP)."""
    rng = np.random.default_rng(1)
    t = 1325. + np.arange(18000) * 2. / 60. / 24.
    flux = transit(t, t0=1326.3, P=3.3, a=9., inc=88.5, rp=0.1, u=(0.35, 0.2))
    # Correlated noise: smoothed random walk with ~300 ppm amplitude on ~0.3 day timescales.
    noise = np.convolve(rng.normal(0, 1, t.size + 400), np.exp(-np.arange(400) / 200.), mode='valid')[:t.size]
    y = flux + 300e-6 * noise / noise.std() + rng.normal(0, 600e-6, t.size)
    params = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'a_p1', 'q1_TESS', 'q2_TESS', 'ecc_p1', 'omega_p1', 'mdilution_TESS', 'mflux_TESS',
              'sigma_w_TESS', 'GP_sigma_TESS', 'GP_rho_TESS']
    dists = ['normal', 'normal', 'uniform', 'uniform', 'loguniform', 'uniform', 'uniform', 'fixed', 'fixed', 'fixed', 'normal',
             'loguniform', 'loguniform', 'loguniform']
    hyperps = [[3.3, 0.01], [1326.3, 0.02], [0., 0.3], [0., 1.], [2., 30.], [0., 1.], [0., 1.], 0., 90., 1., [0., 0.01],
               [1., 5000.], [1e-6, 1e-2], [1e-3, 10.]]
    priors = dict(zip(params, [{'distribution': d, 'hyperparameters': h} for d, h in zip(dists, hyperps)]))
    return priors, dict(t_lc={'TESS': t}, y_lc={'TESS': y}, yerr_lc={'TESS': np.full(t.size, 600e-6)}, GP_regressors_lc={'TESS': t})


def k2_32():
    """The juliet tutorial dataset of K2-32 (3 planets; K2 photometry with a GP and RVs from HIRES, HARPS and PFS)."""
    d = REPO + 'tutorial2/'
    return d + 'priors/K2-32_priors.dat', dict(lcfilename=d + 'data/K2-32_lc.dat', rvfilename=d + 'data/K2-32_rvs.dat',
                                                GPlceparamfile=d + 'data/K2-32_lceparams.dat', lc_instrument_supersamp=['K2'],
                                                lc_n_supersamp=[20], lc_exptime_supersamp=[0.020434])


PROBLEMS = {'small transit (300 pts, 7 params)': small_transit,
            'TESS sector + GP (18000 pts, 11 params)': tess_sector,
            'K2-32 joint lc+GP+RV (3664x20 + 80 pts, 29 params)': k2_32}
