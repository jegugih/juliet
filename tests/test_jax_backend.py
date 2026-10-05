# Tests of the JAX backend of juliet. Run with: python -m pytest tests/test_jax_backend.py
import os

import numpy as np
import pytest

import juliet
import jax
import jax.numpy as jnp
from juliet import jaxmodels as jm
from juliet.jaxgp import JaxGP, celerite_term

rng = np.random.default_rng(0)


def test_uniform_transit_depth_is_p_squared():
    # Central transit of a small planet on a uniform star: depth = p^2.
    p = 0.05
    d, z = jnp.array([0.0, 0.3]), jnp.array([1.0, 1.0])
    flux = jm.transit_flux(d, z, p, jnp.array([0.0]))
    np.testing.assert_allclose(1. - np.asarray(flux), p**2, rtol=1e-10)


def test_no_transit_behind_the_star_and_full_eclipse():
    p, fp = 0.1, 1e-3
    d, z = jnp.array([0.0]), jnp.array([-1.0])
    assert float(jm.transit_flux(d, z, p, jnp.array([0.3, 0.2]))[0]) == 1.0
    # Planet fully behind the star: only the star's flux is left.
    assert float(jm.eclipse_flux(d, z, p, fp)[0]) == pytest.approx(1.0)
    # Planet in front (or away from the star): star + full planet flux.
    assert float(jm.eclipse_flux(jnp.array([3.0]), z, p, fp)[0]) == pytest.approx(1.0 + fp)


def test_transit_and_secondary_times():
    P, ecc, omega = 3.0, 0.3, np.deg2rad(40.)
    t0 = 1.2
    tp = jm.time_of_periastron(t0, P, ecc, omega)
    sinw, cosw = np.sin(omega), np.cos(omega)
    # Sky separation is minimal (and planet in front) at t0 for an edge-on orbit:
    t = jnp.linspace(t0 - 0.05, t0 + 0.05, 2001)
    d, z, _, _ = jm.sky_position(t, tp, P, 10., ecc, sinw, cosw, 1.0, 0.0)
    assert abs(float(t[jnp.argmin(d)]) - t0) < 1e-4 and float(z[1000]) > 0
    # Time of secondary eclipse and the conjunction implied by it are consistent:
    t_sec = jm.time_of_secondary(t0, P, ecc, omega)
    assert float(jm.time_of_conjunction(t_sec, P, ecc, omega)) == pytest.approx(t0)
    t = jnp.linspace(t_sec - 0.05, t_sec + 0.05, 2001)
    d, z, _, _ = jm.sky_position(t, tp, P, 10., ecc, sinw, cosw, 1.0, 0.0)
    assert abs(float(t[jnp.argmin(d)]) - float(t_sec)) < 1e-4 and float(z[1000]) < 0


def test_circular_rv():
    t = jnp.linspace(0, 10, 500)
    rv = jm.rv_keplerian(t, 2.5, 0.3, 0.0, np.pi / 2., 7.0)
    # For a circular orbit, RV = -K sin(2 pi (t - t0) / P):
    np.testing.assert_allclose(np.asarray(rv), -7.0 * np.sin(2 * np.pi * (np.asarray(t) - 0.3) / 2.5), atol=1e-10)


@pytest.mark.parametrize('kernel_name, h', [
    ('CeleriteMaternKernel', {'sigma': 2.0, 'rho': 0.7}),
    ('CeleriteSHOKernel', {'S0': 1.5, 'Q': 0.3, 'omega0': 4.0}),
    ('CeleriteQPKernel', {'B': 1.0, 'L': 3.0, 'Prot': 1.3, 'C': 0.5}),
    ('CeleriteMaternExpKernel', {'sigma': 1.5, 'timescale': 0.5, 'rho': 0.8}),
])
def test_celerite_gp_matches_dense_computation(kernel_name, h):
    X = np.sort(rng.uniform(0, 10, 80))
    r, diag = rng.normal(0, 1, 80), np.full(80, 0.3)
    gp = JaxGP(kernel_name, X[::-1].copy(), 1.0)   # unsorted input: sorted internally
    ll = float(gp.log_likelihood(h, jnp.asarray(r[::-1].copy()), jnp.asarray(diag)))
    K = np.asarray(celerite_term(kernel_name, h, 0.01).get_value(np.abs(X[:, None] - X[None, :]))) + np.diag(diag)
    expected = -0.5 * r @ np.linalg.solve(K, r) - 0.5 * np.linalg.slogdet(K)[1] - 40 * np.log(2 * np.pi)
    assert ll == pytest.approx(expected, rel=1e-8)


def simulated_transit(n=300, sigma=100e-6):
    t = np.linspace(-0.1, 0.1, n)
    pv = dict(P=1.0, t0=0.0, a=3.6, inc=np.pi / 2., p=0.1, u=jnp.array([0.2, 0.3]))
    tp = jm.time_of_periastron(pv['t0'], pv['P'], 0., np.pi / 2.)
    d, z, _, _ = jm.sky_position(jnp.asarray(t), tp, pv['P'], pv['a'], 0., 1., 0., 1., 0.)
    flux = np.asarray(jm.transit_flux(d, z, pv['p'], pv['u']))
    return t, flux + rng.normal(0, sigma, n)


def transit_dataset(out_folder=None):
    t, y = simulated_transit()
    params = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_inst', 'q2_inst', 'ecc_p1', 'omega_p1', 'a_p1', 'mdilution_inst', 'mflux_inst',
              'sigma_w_inst']
    dists = ['fixed', 'normal', 'uniform', 'fixed', 'uniform', 'uniform', 'fixed', 'fixed', 'normal', 'fixed', 'normal', 'loguniform']
    hyperps = [1.0, [0.0, 0.01], [0., 0.3], 0.0, [0., 1.], [0., 1.], 0.0, 90., [3.6, 0.1], 1.0, [0., 0.01], [1., 1000.]]
    priors = juliet.generate_priors(params, dists, hyperps)
    return juliet.load(priors=priors, t_lc={'inst': t}, y_lc={'inst': y}, yerr_lc={'inst': np.full(t.size, 10e-6)},
                       out_folder=out_folder)


def test_likelihood_gradient_is_finite():
    dataset = transit_dataset()
    m = juliet.model(dataset, 'lc')
    pv = {p: jnp.asarray(v) for p, v in m.fixed_values.items()}
    free = dict(t0_p1=0.001, p_p1=0.1, q1_inst=0.3, q2_inst=0.4, a_p1=3.6, mflux_inst=0., sigma_w_inst=100.)
    grad = jax.grad(lambda f: m.log_likelihood_fn({**pv, **f}))({k: jnp.asarray(v) for k, v in free.items()})
    assert all(np.isfinite(float(v)) for v in grad.values())


def test_invalid_models_have_zero_likelihood():
    t, y = simulated_transit()
    params = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_inst', 'q2_inst', 'ecc_p1', 'omega_p1', 'a_p1', 'mdilution_inst', 'mflux_inst',
              'sigma_w_inst']
    dists = ['fixed'] * 3 + ['uniform'] + ['fixed'] * 8
    hyperps = [1.0, 0.0, 0.1, [0., 2.], 0.3, 0.4, 0.0, 90., 3.6, 1.0, 0.0, 100.]
    dataset = juliet.load(priors=juliet.generate_priors(params, dists, hyperps), t_lc={'inst': t}, y_lc={'inst': y},
                          yerr_lc={'inst': np.full(t.size, 10e-6)})
    m = juliet.model(dataset, 'lc')
    assert np.isfinite(m.get_log_likelihood({'b_p1': 0.5}))
    assert m.get_log_likelihood({'b_p1': 1.2}) == -np.inf   # b > 1 + p


@pytest.mark.parametrize('sampler', ['nested', 'nuts'])
def test_fit_with_celerite_gp(tmp_path, sampler):
    # celerite2.jax has no batching rules: this checks GP likelihoods work under the samplers' vmap (and grad, for NUTS).
    t = np.linspace(0, 10, 150)
    y = 1. + 1e-3 * np.sin(t) + rng.normal(0, 1e-4, t.size)
    priors = juliet.generate_priors(['mdilution_inst', 'mflux_inst', 'sigma_w_inst', 'GP_sigma_inst', 'GP_rho_inst'],
                                    ['fixed', 'normal', 'loguniform', 'loguniform', 'loguniform'],
                                    [1.0, [0., 0.01], [1., 1000.], [1e-5, 1e-2], [0.1, 100.]])
    dataset = juliet.load(priors=priors, t_lc={'inst': t}, y_lc={'inst': y}, yerr_lc={'inst': np.full(t.size, 1e-4)},
                          GP_regressors_lc={'inst': t}, out_folder=str(tmp_path) + '/')
    kwargs = dict(n_live_points=100) if sampler == 'nested' else dict(nsteps=100, nburnin=100, num_chains=2, progress_bar=False)
    results = dataset.fit(sampler=sampler, seed=0, **kwargs)
    assert np.all(np.isfinite(results.posteriors['posterior_samples']['loglike']))
    model = results.lc.evaluate('inst', nsamples=50)
    assert np.std(y - model) < 3e-4


@pytest.mark.parametrize('sampler', ['nested', 'nuts'])
def test_fit_recovers_transit(tmp_path, sampler):
    dataset = transit_dataset(out_folder=str(tmp_path) + '/')
    kwargs = dict(n_live_points=200) if sampler == 'nested' else dict(nsteps=300, nburnin=300, num_chains=2, progress_bar=False)
    results = dataset.fit(sampler=sampler, seed=0, **kwargs)
    post = results.posteriors['posterior_samples']
    assert abs(np.median(post['p_p1']) - 0.1) < 5 * np.std(post['p_p1']) + 1e-3
    assert abs(np.median(post['a_p1']) - 3.6) < 5 * np.std(post['a_p1']) + 1e-2
    if sampler == 'nested':
        assert np.isfinite(results.posteriors['lnZ']) and results.posteriors['lnZerr'] > 0
    model, up, low = results.lc.evaluate('inst', return_err=True, nsamples=100)
    assert model.shape == dataset.t_lc.shape and np.all(up >= low)
    assert os.path.exists(str(tmp_path) + '/' + sampler + '_posteriors.pkl')


@pytest.mark.parametrize('ld_law, u', [('squareroot', [0.2, 0.5]), ('logarithmic', [0.6, 0.3]), ('exponential', [0.4, 0.2]),
                                       ('power2', [0.6, 0.6]), ('nonlinear', [0.6, -0.4, 0.8, -0.3])])
def test_numerical_limb_darkening_against_adaptive_quadrature(ld_law, u):
    from scipy.integrate import quad
    p, ds = 0.1, np.array([0.0, 0.5, 0.95, 1.03, 1.09])
    I = lambda r: float(jm.intensity(ld_law, jnp.asarray(u), jnp.sqrt(1. - r**2)))
    F = 2. * np.pi * quad(lambda r: I(r) * r, 0., 1., limit=500, epsabs=1e-14)[0]

    def expected(d):
        kappa = lambda r: np.pi if r <= p - d else np.arccos(np.clip((r**2 + d**2 - p**2) / (2. * r * d), -1., 1.))
        lo, hi = max(0., d - p), min(1., d + p)
        points = [x for x in [p - d] if lo < x < hi] or None
        return 1. - quad(lambda r: I(r) * 2. * r * kappa(r), lo, hi, points=points, limit=500, epsabs=1e-14)[0] / F

    flux = np.asarray(jm.numerical_transit_flux(jnp.asarray(ds), jnp.ones(ds.size), p, ld_law, jnp.asarray(u)))
    np.testing.assert_allclose(flux, [expected(d) for d in ds], atol=1e-9)


def test_numerical_limb_darkening_matches_jaxoplanet_for_quadratic_law():
    from jaxoplanet.core.limb_dark import light_curve
    d = jnp.linspace(0., 1.15, 300)
    u = jnp.array([0.4, 0.25])
    expected = 1. + light_curve(u, d, 0.1 * jnp.ones_like(d), order=100)
    np.testing.assert_allclose(np.asarray(jm.numerical_transit_flux(d, jnp.ones_like(d), 0.1, 'quadratic', u)),
                               np.asarray(expected), atol=1e-9)


def test_semicircles_with_equal_radii_is_a_circular_transit():
    t = jnp.linspace(-0.08, 0.08, 400)
    tp = jm.time_of_periastron(0., 1., 0.2, 1.0)
    X, Y, z, direction = jm.sky_coordinates(t, tp, 1., 4., 0.2, np.sin(1.0), np.cos(1.0), np.sin(1.5), np.cos(1.5))
    u = jnp.array([0.3, 0.2])
    flux = jm.semicircles_transit_flux(X, Y, z, direction, 0.1, 0.1, 0.3, 'quadratic', u)
    np.testing.assert_allclose(np.asarray(flux), np.asarray(jm.transit_flux(jnp.hypot(X, Y), z, 0.1, u)), atol=1e-8)


def test_kelp_phase_curves_are_traceable():
    from juliet import kelp_jax
    phases = jnp.asarray(rng.uniform(0., 1., 50))

    def total(params):
        f1 = kelp_jax.reflected_phase_curve(phases, params[0], params[1], 20.)[0]
        f2 = kelp_jax.reflected_phase_curve_inhomogeneous(phases, params[0], 0.2, -0.5, 0.5, 0.3, 20.)[0]
        return jnp.sum(f1) + jnp.sum(f2)

    params = jnp.array([0.5, 0.1])
    assert np.isfinite(float(jax.jit(total)(params)))
    assert np.all(np.isfinite(np.asarray(jax.grad(total)(params))))
    assert np.all(np.isfinite(np.asarray(jax.vmap(total)(jnp.stack([params, params * 0.9])))))


def test_celerite_prediction_matches_dense_computation():
    X = np.sort(rng.uniform(0, 10, 60))
    r, diag = rng.normal(0, 1, 60), np.full(60, 0.3)
    h = {'S0': 1.5, 'Q': 3.0, 'omega0': 2.0}
    gp = JaxGP('CeleriteSHOKernel', X, 1.0)
    X_new = rng.uniform(-1, 11, 25)
    k = celerite_term('CeleriteSHOKernel', h, 0.01)
    K = np.asarray(k.get_value(np.abs(X[:, None] - X[None, :]))) + np.diag(diag)
    Ks = np.asarray(k.get_value(np.abs(X_new[:, None] - X[None, :])))
    np.testing.assert_allclose(np.asarray(gp.predict(h, jnp.asarray(r), jnp.asarray(diag), X_new)), Ks @ np.linalg.solve(K, r), atol=1e-10)


def test_numpy_user_functions_are_evaluated_with_callbacks(tmp_path):
    t = np.linspace(0, 1, 100)
    y = 1. + 1e-3 * t + rng.normal(0, 1e-4, t.size)

    def slope(regressor, parameters):
        # Written with NumPy (np.asarray on the inputs would fail under JAX tracing):
        return np.asarray(parameters['slope_inst']) * np.asarray(regressor)

    priors = juliet.generate_priors(['mdilution_inst', 'mflux_inst', 'sigma_w_inst', 'slope_inst'],
                                    ['fixed', 'normal', 'loguniform', 'uniform'], [1.0, [0., 0.01], [1., 1000.], [-0.01, 0.01]])
    dataset = juliet.load(priors=priors, t_lc={'inst': t}, y_lc={'inst': y}, yerr_lc={'inst': np.full(t.size, 1e-4)},
                          non_linear_functions={'inst': {'function': slope, 'regressor': t}}, out_folder=str(tmp_path) + '/')
    with pytest.raises(Exception, match='nuts'):
        dataset.fit(sampler='nuts')
    results = dataset.fit(sampler='nested', n_live_points=100, seed=0)
    assert abs(np.median(results.posteriors['posterior_samples']['slope_inst']) - 1e-3) < 2e-4


@pytest.mark.parametrize('legacy', [False, True])
def test_gp_parametrization_option(legacy):
    # Exp-sine-squared kernel: GP_Gamma (or log(GP_Gamma) with the legacy parametrization) multiplies sin^2;
    # multi-dimensional squared-exponential kernel: variance GP_sigma^2 (or nX * GP_sigma^2 with the legacy parametrization).
    X1, X2 = rng.uniform(0, 1, 40), rng.uniform(0, 1, (40, 2))
    r, diag = rng.normal(0, 1e-3, 40), np.full(40, 1e-6)
    h1 = {'sigma': 300., 'alpha': 5., 'Gamma': 3., 'Prot': 0.3}
    h2 = {'sigma': 300., 'alpha0': 5., 'alpha1': 2.}
    gamma = np.log(3.) if legacy else 3.
    tau = X1[:, None] - X1[None, :]
    K1 = (300e-6)**2 * np.exp(-0.5 * 5. * tau**2 - gamma * np.sin(np.pi * np.abs(tau) / 0.3)**2)
    D = X2[:, None, :] - X2[None, :, :]
    K2 = (2. if legacy else 1.) * (300e-6)**2 * np.exp(-0.5 * (5. * D[..., 0]**2 + 2. * D[..., 1]**2))
    for kernel_name, X, h, K in [('ExpSineSquaredSEKernel', X1, h1, K1), ('SEKernel', X2, h2, K2)]:
        gp = JaxGP(kernel_name, X, 1e-6, legacy_parametrization=legacy)
        C = K + np.diag(diag)
        expected = -0.5 * r @ np.linalg.solve(C, r) - 0.5 * np.linalg.slogdet(C)[1] - 20. * np.log(2. * np.pi)
        assert float(gp.log_likelihood(h, jnp.asarray(r), jnp.asarray(diag))) == pytest.approx(expected, rel=1e-9)


def test_legacy_backend_dispatch():
    pytest.importorskip('batman')
    t, y = simulated_transit()
    params = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_inst', 'q2_inst', 'ecc_p1', 'omega_p1', 'a_p1', 'mdilution_inst', 'mflux_inst', 'sigma_w_inst']
    priors = juliet.generate_priors(params, ['fixed'] * len(params), [1., 0., 0.1, 0., 0.3, 0.4, 0., 90., 3.6, 1., 0., 100.])
    kw = dict(priors=priors, t_lc={'inst': t}, y_lc={'inst': y}, yerr_lc={'inst': np.full(t.size, 10e-6)})
    legacy, new = juliet.load(backend='legacy', **kw), juliet.load(**kw)
    assert legacy.backend == 'legacy' and type(legacy).__module__ == 'juliet.legacy.fit'
    m_legacy, m_new = juliet.model(legacy, 'lc', log_like_calc=True), juliet.model(new, 'lc')
    assert type(m_legacy).__module__ == 'juliet.legacy.fit'
    pv = {p: priors[p]['hyperparameters'] for p in priors}
    m_legacy.generate(pv)
    # (batman's analytic quadratic law is accurate to ~5e-9 in flux)
    assert m_new.get_log_likelihood(pv) == pytest.approx(m_legacy.get_log_likelihood(pv), rel=1e-5)


def test_nautilus_sampler(tmp_path):
    pytest.importorskip('nautilus')
    dataset = transit_dataset(out_folder=str(tmp_path) + '/')
    results = dataset.fit(sampler='nautilus', n_live_points=500, seed=0)
    post = results.posteriors['posterior_samples']
    assert abs(np.median(post['p_p1']) - 0.1) < 5 * np.std(post['p_p1']) + 1e-3
    assert np.isfinite(results.posteriors['lnZ'])
