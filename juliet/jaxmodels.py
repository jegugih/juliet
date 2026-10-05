# Pure-JAX building blocks used by juliet's lightcurve and radial-velocity models.
#
# Lightcurves are computed with jaxoplanet's Kepler solver and its polynomial limb-darkening
# solver (Agol et al. 2020); radial velocities with jaxoplanet's Keplerian System. The
# conventions follow the ones juliet always used through batman/radvel: t0 is the time of
# inferior conjunction (transit), omega is the argument of periastron of the planet's orbit
# and a is in units of stellar radii.
import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet.core.kepler import kepler
from jaxoplanet.core.limb_dark import light_curve as _limb_dark_light_curve
from jaxoplanet.orbits.keplerian import Body, Central, System

import numpyro.distributions as dist
from numpyro.distributions import constraints

# Speed of light (m/s) and solar radius (m), same values as astropy.constants (CODATA 2018 / IAU 2015):
c_light = 299792458.0
R_sun = 695700000.0

# Limb-darkening laws. Polynomial laws are computed exactly with jaxoplanet's limb-darkening solver ('none' is used
# for instruments with no limb-darkening priors, e.g., eclipse-only fits); the others (same definitions as in batman)
# by numerical integration of the stellar intensity over the occulted area (see numerical_transit_flux):
POLYNOMIAL_LD_LAWS = ['linear', 'quadratic', 'none']
NUMERICAL_LD_LAWS = ['squareroot', 'logarithmic', 'exponential', 'power2', 'nonlinear']
SUPPORTED_LD_LAWS = POLYNOMIAL_LD_LAWS + NUMERICAL_LD_LAWS


def safe_sqrt(x):
    """sqrt that has a finite gradient at (and below) zero."""
    pos = x > 0.
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, x, 1.)), 0.)


def reverse_ld_coeffs(ld_law, q1, q2):
    """JAX version of juliet.utils.reverse_ld_coeffs for the laws supported by the JAX backend."""
    sq1 = safe_sqrt(q1)
    if ld_law == 'quadratic':
        return 2. * sq1 * q2, sq1 * (1. - 2. * q2)
    elif ld_law == 'squareroot':
        return sq1 * (1. - 2. * q2), 2. * sq1 * q2
    elif ld_law == 'logarithmic':
        return 1. - sq1 * q2, 1. - sq1
    elif ld_law == 'linear':
        return q1, q2
    raise NotImplementedError('The (q1, q2) parametrization is not available for the ' + ld_law + ' limb-darkening law; '
                              'use (u1, u2) instead.')


def mean_anomaly_at(ecc, f):
    """Mean anomaly at true anomaly f; uses the same branch of arctan as batman and radvel."""
    E = 2. * jnp.arctan(jnp.sqrt((1. - ecc) / (1. + ecc)) * jnp.tan(f / 2.))
    return E - ecc * jnp.sin(E)


def time_of_periastron(t0, P, ecc, omega, secondary=False):
    """
    Time of periastron given the time of transit (or of secondary eclipse if secondary = True).
    omega is in radians.
    """
    f = (1.5 * jnp.pi if secondary else 0.5 * jnp.pi) - omega
    return t0 - P * mean_anomaly_at(ecc, f) / (2. * jnp.pi)


def time_of_secondary(t0, P, ecc, omega):
    """Geometric time of secondary eclipse (same definition as batman's get_t_secondary). omega in radians."""
    M1 = mean_anomaly_at(ecc, 0.5 * jnp.pi - omega)
    M2 = mean_anomaly_at(ecc, 1.5 * jnp.pi - omega)
    return t0 + P * (M2 - M1) / (2. * jnp.pi)


def time_of_conjunction(t_secondary, P, ecc, omega):
    """Time of transit implied by a time of secondary eclipse (same definition as batman's get_t_conjunction). omega in radians."""
    M1 = mean_anomaly_at(ecc, 0.5 * jnp.pi - omega)
    M2 = mean_anomaly_at(ecc, 1.5 * jnp.pi - omega)
    return t_secondary + P * (M1 - M2) / (2. * jnp.pi)


def true_anomaly(t, tp, P, ecc):
    """sin and cos of the true anomaly at times t."""
    M = 2. * jnp.pi * (t - tp) / P
    return kepler(M, ecc * jnp.ones_like(M))


def sky_position(t, tp, P, a, ecc, sinw, cosw, sini, cosi):
    """
    Returns the sky-projected separation d (in stellar radii), the line-of-sight coordinate z
    (positive towards the observer, i.e., planet in front of the star) and sin/cos of the true anomaly.
    """
    sinf, cosf = true_anomaly(t, tp, P, ecc)
    r = a * (1. - ecc**2) / (1. + ecc * cosf)
    sinwf = sinw * cosf + cosw * sinf
    coswf = cosw * cosf - sinw * sinf
    d = r * jnp.sqrt(coswf**2 + (sinwf * cosi)**2)
    z = r * sinwf * sini
    return d, z, sinf, cosf


def supersample(t, nresampling, etresampling):
    """Times at which to evaluate a supersampled model (same offsets as batman); shape (len(t), nresampling)."""
    if nresampling is None or etresampling is None:
        return t[:, None]
    offsets = jnp.linspace(-etresampling / 2., etresampling / 2., nresampling)
    return t[:, None] + offsets[None, :]


def transit_flux(d, z, p, u, order=20):
    """
    Limb-darkened transit (relative flux, 1 out-of-transit) using jaxoplanet's solver. With order = 20 the
    quadrature error is below 1e-9 in relative flux (batman's analytic quadratic law is good to ~5e-9).
    """
    lc = _limb_dark_light_curve(u, d, p * jnp.ones_like(d), order=order)
    return 1. + jnp.where(z > 0., lc, 0.)


def intensity(ld_law, u, mu):
    """Stellar intensity I(mu) / I(mu = 1) for the limb-darkening laws, as defined in batman."""
    if ld_law in ['quadratic', 'none']:
        return 1. - u[0] * (1. - mu) - u[1] * (1. - mu)**2
    elif ld_law == 'linear':
        return 1. - u[0] * (1. - mu)
    elif ld_law == 'squareroot':
        return 1. - u[0] * (1. - mu) - u[1] * (1. - jnp.sqrt(mu))
    elif ld_law == 'logarithmic':
        return 1. - u[0] * (1. - mu) - u[1] * mu * jnp.log(mu)
    elif ld_law == 'exponential':
        return 1. - u[0] * (1. - mu) - u[1] / (1. - jnp.exp(mu))
    elif ld_law == 'power2':
        return 1. - u[0] * (1. - mu**u[1])
    elif ld_law == 'nonlinear':
        smu = jnp.sqrt(mu)
        return 1. - u[0] * (1. - smu) - u[1] * (1. - mu) - u[2] * (1. - smu**3) - u[3] * (1. - mu**2)
    raise NotImplementedError('Limb-darkening law ' + ld_law + ' not implemented.')


# Gauss-Legendre nodes on (0, pi) for the radial integrals, with the substitution r = r0 + (r1 - r0) (1 - cos theta) / 2
# (which removes the square-root behaviour of the integrands at the ends of the integration intervals), and on (0, 1)
# for the total stellar flux:
_x, _w = np.polynomial.legendre.leggauss(24)
_s, _ds, _wtheta = (1. - np.cos((_x + 1.) * np.pi / 2.)) / 2., np.sin((_x + 1.) * np.pi / 2.) / 2., _w * np.pi / 2.
_xm, _wm = np.polynomial.legendre.leggauss(64)
_mu_nodes, _mu_weights = (_xm + 1.) / 2., _wm / 2.


def numerical_transit_flux(d, z, p, ld_law, u):
    """
    Transit of a planet of radius p at sky separations d (in stellar radii) for any radial limb-darkening law, by
    integrating I(r) times the length of the arc of radius r covered by the planet. The integration is split at
    r = p - d (where the arcs become full circles), and is converged to ~1e-10 in relative flux.
    """
    def radial_integral(r0, r1, kappa):
        r = r0[..., None] + (r1 - r0)[..., None] * _s
        # (mu >= 1e-15 keeps the intensity finite at the limb, e.g., for the exponential law)
        mu = jnp.sqrt(jnp.clip(1. - r**2, 1e-30, 1.))
        return jnp.sum(_wtheta * intensity(ld_law, u, mu) * 2. * r * kappa(r) * (r1 - r0)[..., None] * _ds, axis=-1)

    # Annuli fully covered by the planet (r < p - d, only if the planet covers the centre of the star):
    zero = jnp.zeros_like(d)
    inner = radial_integral(zero, jnp.clip(p - d, 0., 1.), lambda r: jnp.pi)
    # Partially covered annuli, |d - p| < r < min(1, d + p): half-angle of the arc of radius r inside the planet:
    safe_d = jnp.where(d > 0., d, 1.)[..., None]

    def kappa(r):
        return jnp.arccos(jnp.clip((r**2 + d[..., None]**2 - p**2) / (2. * r * safe_d), -1., 1.))
    outer = radial_integral(jnp.clip(jnp.abs(d - p), 0., 1.), jnp.clip(d + p, 0., 1.), kappa)

    total = 2. * np.pi * jnp.sum(_mu_weights * _mu_nodes * intensity(ld_law, u, jnp.asarray(_mu_nodes)))
    return jnp.where((z > 0.) & (d < 1. + p), 1. - (inner + outer) / total, 1.)


def sky_coordinates(t, tp, P, a, ecc, sinw, cosw, sini, cosi):
    """
    Sky coordinates (X, Y) of the planet with respect to the star (in stellar radii; X along the projected orbital
    motion at transit, as in catwoman), line-of-sight coordinate z (positive towards the observer) and the angle of the
    sky-projected direction perpendicular to the position vector within the orbital plane.
    """
    sinf, cosf = true_anomaly(t, tp, P, ecc)
    r = a * (1. - ecc**2) / (1. + ecc * cosf)
    sinwf = sinw * cosf + cosw * sinf
    coswf = cosw * cosf - sinw * sinf
    return -r * coswf, -r * sinwf * cosi, r * sinwf * sini, jnp.arctan2(-coswf * cosi, sinwf)


# Gauss-Legendre nodes for the integrals of the semicircles' occultations (one set per interval between breakpoints):
_xc, _wc = np.polynomial.legendre.leggauss(16)
_sc, _dsc, _wthetac = (1. - np.cos((_xc + 1.) * np.pi / 2.)) / 2., np.sin((_xc + 1.) * np.pi / 2.) / 2., _wc * np.pi / 2.


def _arc_overlap(delta, kappa, beta):
    """Length of the intersection of the arcs (delta - kappa, delta + kappa) and (-beta, beta) of a circle."""
    total = 0.
    for k in (-1, 0, 1):
        shift = 2. * np.pi * k
        total = total + jnp.maximum(0., jnp.minimum(delta + kappa + shift, beta) - jnp.maximum(delta - kappa + shift, -beta))
    return total


def semicircles_transit_flux(X, Y, z, direction, R1, R2, phi, ld_law, u):
    """
    Transit of a planet made of two semicircles of radii R1 and R2 (as in catwoman; Jones & Espinoza 2020). The
    semicircles share a diameter whose direction is the angle ``direction`` (the sky-projected direction perpendicular
    to the position vector in the orbital plane, see sky_coordinates) rotated by ``phi`` (radians); the semicircle of
    radius R1 lies to its left. The flux is computed by integrating the stellar intensity times the length of the arc of
    radius r covered by the planet, splitting the integral at every radius where that length has a kink.
    """
    d = jnp.sqrt(X**2 + Y**2)
    alpha = direction + phi
    tx, ty = jnp.cos(alpha), jnp.sin(alpha)
    h = -ty * X + tx * Y                       # signed distance from the star to the line of the shared diameter
    Rmax = jnp.maximum(R1, R2)
    rmin, rmax = jnp.clip(d - Rmax, 0., 1.), jnp.clip(d + Rmax, 0., 1.)
    breaks = jnp.stack([jnp.abs(d - R1), d + R1, jnp.abs(d - R2), d + R2, jnp.abs(h),
                        jnp.hypot(X + R1 * tx, Y + R1 * ty), jnp.hypot(X - R1 * tx, Y - R1 * ty),
                        jnp.hypot(X + R2 * tx, Y + R2 * ty), jnp.hypot(X - R2 * tx, Y - R2 * ty)], axis=-1)
    edges = jnp.sort(jnp.concatenate([rmin[..., None], rmax[..., None],
                                      jnp.clip(breaks, rmin[..., None], rmax[..., None])], axis=-1), axis=-1)
    r0, r1 = edges[..., :-1, None], edges[..., 1:, None]
    r = r0 + (r1 - r0) * _sc
    mu = jnp.sqrt(jnp.clip(1. - r**2, 1e-30, 1.))

    # Angular measure of the circle of radius r inside each semicircle (arc inside the disk intersected with the
    # corresponding half-plane):
    e = (..., None, None)
    safe = jnp.maximum(r * d[e], 1e-300)
    kappa1 = jnp.arccos(jnp.clip((r**2 + d[e]**2 - R1**2) / (2. * safe), -1., 1.))
    kappa2 = jnp.arccos(jnp.clip((r**2 + d[e]**2 - R2**2) / (2. * safe), -1., 1.))
    beta = jnp.arccos(jnp.clip(h[e] / jnp.maximum(r, 1e-300), -1., 1.))
    delta = jnp.mod(jnp.arctan2(Y, X) - jnp.arctan2(tx, -ty) + np.pi, 2. * np.pi)[e] - np.pi
    arc = _arc_overlap(delta, kappa1, beta) + 2. * kappa2 - _arc_overlap(delta, kappa2, beta)

    blocked = jnp.sum(_wthetac * intensity(ld_law, u, mu) * arc * r * (r1 - r0) * _dsc, axis=(-2, -1))
    total = 2. * np.pi * jnp.sum(_mu_weights * _mu_nodes * intensity(ld_law, u, jnp.asarray(_mu_nodes)))
    return jnp.where((z > 0.) & (d < 1. + Rmax), 1. - blocked / total, 1.)


def eclipse_flux(d, z, p, fp):
    """
    Secondary eclipse of a uniform planetary disk (same normalization as batman: 1 + fp out of eclipse,
    1 in total eclipse). The occulted area is computed with jaxoplanet's solver, swapping the roles of the
    star and the planet (the star occulting a uniform disk of radius p).
    """
    occulted = _limb_dark_light_curve(jnp.array([]), d / p, jnp.ones_like(d) / p)
    visible = jnp.where(z < 0., 1. + occulted, 1.)
    return 1. + fp * visible


def light_travel_corrected_times(t, t0, tp, P, a, ecc, sinw, cosw, sini, stellar_radius):
    """
    JAX version of juliet.utils.correct_light_travel_time (first-order correction for the light travel time
    across the orbit, measured with respect to the planet's position at transit). Returns corrected times.
    """
    def los(tt):
        sinf, cosf = true_anomaly(tt, tp, P, ecc)
        r = a * (1. - ecc**2) / (1. + ecc * cosf)
        return r * (sinw * cosf + cosw * sinf) * sini

    meters = stellar_radius * R_sun
    delta_x = (los(jnp.atleast_1d(t0))[0] - los(t)) * meters
    return t - (delta_x / c_light) / (3600. * 24.)


def rv_keplerian(t, P, t0, ecc, omega, K):
    """Keplerian RV signal of the star due to one planet, via jaxoplanet. omega in radians."""
    system = System(Central()).add_body(Body(time_transit=t0, period=P, eccentricity=ecc,
                                             omega_peri=omega, radial_velocity_semiamplitude=K))
    return system.radial_velocity(t)[0]


def sinusoidal_phase_curve(t, t0, P, fp, phase_offset):
    orbital_phase = ((t - t0) / P) % 1
    sine_model = jnp.sin(2. * jnp.pi * orbital_phase - jnp.pi / 2. + phase_offset * (jnp.pi / 180.))
    return fp * (sine_model + 1.) * 0.5


def cowan_agol_phase_curve(t, t_secondary, P, fp, C1, D1, C2, D2):
    omega_t = 2. * jnp.pi * (t - t_secondary) / P
    return fp + C1 * (jnp.cos(omega_t) - 1.) + D1 * jnp.sin(omega_t) + \
           C2 * (jnp.cos(2. * omega_t) - 1.) + D2 * jnp.sin(2. * omega_t)


def lambertian_phase_curve(sinf, cosf, sinw, cosw, sini, ecc, p, a, Ag):
    # Deline et al. (2022), Section 4.4.3:
    sinwf = sinw * cosf + cosw * sinf
    alpha_phs = jnp.arccos(jnp.clip(-sinwf * sini, -1. + 1e-15, 1. - 1e-15))
    ecc_facs = (1. + ecc * cosf) / (1. - ecc**2)
    return Ag * (p * ecc_facs / a)**2 * (jnp.sin(alpha_phs) + (jnp.pi - alpha_phs) * jnp.cos(alpha_phs)) / jnp.pi


def example_parameter_values(priors):
    """Representative values of all the parameters (fixed values, or prior means), e.g., to evaluate user functions once."""
    values = {}
    for k, v in priors.items():
        h = np.atleast_1d(np.asarray(v['hyperparameters'], dtype=float))
        if v['distribution'].lower() == 'fixed':
            values[k] = float(h[0])
            continue
        try:
            value = float(prior_distribution(v['distribution'], v['hyperparameters']).mean)
        except Exception:
            value = np.nan
        values[k] = value if np.isfinite(value) else float(h[0])
    return values


def jax_compatible(fn, example_parameters, label):
    """
    Returns (function, is_callback). ``fn`` takes a dictionary of parameter values. If it can be traced by JAX (i.e., it
    is written with jax.numpy), it is returned as is. Otherwise, it is wrapped in a jax.pure_callback, so it is evaluated
    with concrete (NumPy) values on the host: this works with all samplers except NUTS, which needs gradients.
    """
    structure = {k: jax.ShapeDtypeStruct((), jnp.float64) for k in example_parameters}
    try:
        jax.eval_shape(fn, structure)
        return fn, False
    except Exception:
        pass

    def host(parameters):
        parameters = {k: float(v) for k, v in parameters.items()}
        return np.asarray(fn(parameters), dtype=np.float64)

    # Shape of the output, from one evaluation with concrete values:
    output = jax.ShapeDtypeStruct(host(example_parameters).shape, jnp.float64)
    print('Note: ' + label + ' is not written with jax.numpy; it will be evaluated with NumPy on the host via '
          'jax.pure_callback (this is slower, and does not work with sampler = "nuts", which needs gradients).')

    def wrapped(parameters):
        return jax.pure_callback(host, output, {k: parameters[k] for k in example_parameters}, vmap_method='sequential')

    return wrapped, True


def gaussian_log_likelihood(residuals, variances):
    return -0.5 * (residuals.shape[0] * np.log(2. * np.pi) + jnp.sum(jnp.log(variances) + residuals**2 / variances))


class ModifiedJeffreys(dist.Distribution):
    """Modified Jeffreys prior, p(x) ~ 1 / (x + turn) on (0, hi), as defined in juliet.utils."""
    arg_constraints = {'turn': constraints.positive, 'hi': constraints.positive}
    reparametrized_params = ['turn', 'hi']

    def __init__(self, turn, hi, validate_args=None):
        self.turn, self.hi = turn, hi
        self._support = constraints.interval(0., hi)
        super().__init__(batch_shape=(), validate_args=validate_args)

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def icdf(self, u):
        return self.turn * (jnp.exp(u * jnp.log(self.hi / self.turn + 1.)) - 1.)

    def sample(self, key, sample_shape=()):
        return self.icdf(jax.random.uniform(key, sample_shape + self.batch_shape))

    def log_prob(self, x):
        return -jnp.log(x + self.turn) - jnp.log(jnp.log((self.turn + self.hi) / self.turn))


def prior_distribution(distribution, hyperparameters):
    """numpyro distribution matching a juliet prior."""
    name = distribution.lower()
    h = np.atleast_1d(np.asarray(hyperparameters, dtype=float))
    if name == 'uniform':
        return dist.Uniform(h[0], h[1])
    elif name == 'normal':
        return dist.Normal(h[0], h[1])
    elif name == 'truncatednormal':
        return dist.TruncatedNormal(h[0], h[1], low=h[2], high=h[3])
    elif name in ['jeffreys', 'loguniform']:
        return dist.LogUniform(h[0], h[1])
    elif name == 'beta':
        return dist.Beta(h[0], h[1])
    elif name == 'exponential':
        # As in juliet.utils.transform_exponential, this is a gamma distribution with shape a:
        return dist.Gamma(h[0], 1.)
    elif name == 'modjeffreys':
        return ModifiedJeffreys(h[0], h[1])
    raise Exception('INPUT ERROR: prior distribution ' + distribution + ' not recognized.')
