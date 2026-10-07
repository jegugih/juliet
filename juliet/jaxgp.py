# Gaussian Process kernels for the JAX backend. celerite kernels are defined with celerite2.jax's terms and
# evaluated with the celerite algorithm (Foreman-Mackey et al. 2017) written in pure JAX, so they run on CPUs and
# GPUs and can be vmapped and differentiated. The (former) george kernels, which also accept multi-dimensional
# regressors, are evaluated with a dense Cholesky factorization in JAX.
#
# Parametrizations reproduce exactly the parameter vectors juliet used to pass to celerite and george,
# so posteriors are directly comparable with the ones obtained with previous juliet versions.
import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

from celerite2.jax import terms

from .jaxmodels import safe_sqrt

# Hyperparameters (in order) defining each kernel. 'alpha'/'malpha' kernels get one hyperparameter
# per GP regressor (see kernel_variables):
_kernel_variables = {
    'ExpSineSquaredSEKernel': ['sigma', 'alpha', 'Gamma', 'Prot'],
    'CeleriteQPKernel': ['B', 'L', 'Prot', 'C'],
    'CeleriteExpKernel': ['sigma', 'timescale'],
    'CeleriteMaternKernel': ['sigma', 'rho'],
    'CeleriteMaternExpKernel': ['sigma', 'timescale', 'rho'],
    'CeleriteSHOKernel': ['S0', 'Q', 'omega0'],
    'CeleriteDoubleSHOKernel': ['sigma', 'Q0', 'period', 'f', 'dQ'],
    'CeleriteMaternSHOKernel': ['sigma', 'rho', 'S0', 'Q', 'omega0'],
}


def kernel_variables(nX):
    all_kernel_variables = {'SEKernel': ['sigma'] + ['alpha' + str(i) for i in range(nX)],
                            'M32Kernel': ['sigma'] + ['malpha' + str(i) for i in range(nX)]}
    all_kernel_variables.update(_kernel_variables)
    return all_kernel_variables


def celerite_term(kernel_name, h, matern_eps):
    """celerite2.jax term for kernel_name given the hyperparameter dictionary h."""
    if kernel_name == 'CeleriteQPKernel':
        # Rotation term of Foreman-Mackey et al. (2017), as previously implemented in juliet:
        B, L, Prot, C = h['B'], h['L'], h['Prot'], h['C']
        return terms.RealTerm(a=B * (1. + C) / (2. + C), c=1. / L) + \
               terms.ComplexTerm(a=B / (2. + C), b=0., c=1. / L, d=2. * jnp.pi / Prot)
    elif kernel_name == 'CeleriteExpKernel':
        return terms.RealTerm(a=h['sigma'], c=h['timescale'])
    elif kernel_name == 'CeleriteMaternKernel':
        return terms.Matern32Term(sigma=h['sigma'], rho=h['rho'], eps=matern_eps)
    elif kernel_name == 'CeleriteMaternExpKernel':
        return terms.RealTerm(a=h['sigma'], c=h['timescale']) * \
               terms.Matern32Term(sigma=1., rho=h['rho'], eps=matern_eps)
    elif kernel_name == 'CeleriteSHOKernel':
        return terms.SHOTerm(S0=h['S0'], Q=h['Q'], w0=h['omega0'])
    elif kernel_name == 'CeleriteDoubleSHOKernel':
        # Adapted from celerite2's RotationTerm (formulas kept as in previous juliet versions):
        sigma, Q0, P, f, dQ = h['sigma'], h['Q0'], h['period'], h['f'], h['dQ']
        Q1 = 1. / 2. + Q0 + dQ
        omega1 = 4. * jnp.pi * Q1 / (P * jnp.sqrt(4. * Q1**2 - 1.))
        S1 = sigma**2 / ((1. + f) * omega1 * Q1)
        Q2 = 1. / 2. + Q0
        omega2 = 8. * jnp.pi * Q1 / (P * jnp.sqrt(4. * Q1**2 - 1.))
        S2 = f * sigma**2 / ((1. + f) * omega2 * Q2)
        return terms.SHOTerm(S0=S1, Q=Q1, w0=omega1) + terms.SHOTerm(S0=S2, Q=Q2, w0=omega2)
    elif kernel_name == 'CeleriteMaternSHOKernel':
        return terms.Matern32Term(sigma=h['sigma'], rho=h['rho'], eps=matern_eps) + \
               terms.SHOTerm(S0=h['S0'], Q=h['Q'], w0=h['omega0'])
    raise Exception('Kernel ' + kernel_name + ' is not a celerite kernel.')


##########################################################################################################
# celerite solver. A celerite covariance matrix is K = diag(a) + tril(U phi V^T) + triu(V phi U^T), with
# phi_nm = exp(-c (t_n - t_m)); (c, a, U, V) are given by celerite2's get_celerite_matrices.
##########################################################################################################

# Number of data points per iteration of the celerite recursions (lax.scan's unroll). On GPUs each loop iteration has a
# fixed launch cost, so unrolling cuts the latency of long time series; set with the JULIET_CELERITE_UNROLL variable.
CELERITE_UNROLL = int(os.environ.get('JULIET_CELERITE_UNROLL', 1))


def _celerite_step(c, carry, inputs):
    """One step of the Cholesky factorization K = L diag(d) L^T and of the forward solve L z = y."""
    S, F, t_prev, d_prev, W_prev, z_prev = carry
    tn, an, Un, Vn, yn = inputs
    p = jnp.exp(-c * (tn - t_prev))
    S = p[:, None] * (S + d_prev * W_prev[:, None] * W_prev[None, :]) * p[None, :]
    SU = jnp.sum(S * Un[None, :], axis=1)
    d = an - jnp.sum(Un * SU)
    W = (Vn - SU) / d
    F = p * (F + W_prev * z_prev)
    z = yn - jnp.sum(Un * F)
    return (S, F, tn, d, W, z), (d, z, W)


def _celerite_initial_carry(t, c, dtype):
    J = c.shape[0]
    return (jnp.zeros((J, J), dtype), jnp.zeros(J, dtype), t[0], jnp.zeros((), dtype), jnp.zeros(J, dtype),
            jnp.zeros((), dtype))


def _celerite_forward(t, c, a, U, V, y, store_carries=False):
    def step(carry, inputs):
        new_carry, out = _celerite_step(c, carry, inputs)
        return new_carry, (out, carry) if store_carries else out
    return jax.lax.scan(step, _celerite_initial_carry(t, c, y.dtype), (t, a, U, V, y), unroll=CELERITE_UNROLL)[1]


def _log_likelihood_from_innovations(d, z):
    return -0.5 * (jnp.sum(z**2 / d) + jnp.sum(jnp.log(d)) + z.shape[0] * np.log(2. * np.pi))


@jax.custom_vjp
def celerite_log_likelihood(t, c, a, U, V, y):
    """GP log-likelihood of y for a celerite matrix (t must be sorted)."""
    d, z, _ = _celerite_forward(t, c, a, U, V, y)
    return _log_likelihood_from_innovations(d, z)


# The gradient is computed with an explicit reverse recursion over the stored states of the forward pass
# (several times faster than differentiating through lax.scan automatically):
def _celerite_log_likelihood_fwd(t, c, a, U, V, y):
    (d, z, _), carries = _celerite_forward(t, c, a, U, V, y, store_carries=True)
    return _log_likelihood_from_innovations(d, z), (t, c, a, U, V, y, d, z, carries)


def _celerite_log_likelihood_bwd(residuals, g):
    t, c, a, U, V, y, d, z, carries = residuals
    d_bar = -0.5 * g * (1. / d - z**2 / d**2)
    z_bar = -g * z / d

    def back(state, inputs):
        carry_bar, c_bar = state
        carry, step_inputs, db, zb = inputs
        _, vjp = jax.vjp(_celerite_step, c, carry, step_inputs)
        c_b, carry_b, inputs_b = vjp((carry_bar, (db, zb, jnp.zeros_like(carry[4]))))
        return (carry_b, c_bar + c_b), inputs_b

    zero_carry = jax.tree.map(lambda x: jnp.zeros_like(x[0]), carries)
    (_, c_bar), (t_b, a_b, U_b, V_b, y_b) = jax.lax.scan(back, (zero_carry, jnp.zeros_like(c)),
                                                        (carries, (t, a, U, V, y), d_bar, z_bar), reverse=True,
                                                        unroll=CELERITE_UNROLL)
    return t_b, c_bar, a_b, U_b, V_b, y_b


celerite_log_likelihood.defvjp(_celerite_log_likelihood_fwd, _celerite_log_likelihood_bwd)


def celerite_apply_inverse(t, c, a, U, V, y):
    """K^-1 y for a celerite matrix (t must be sorted)."""
    d, z, W = _celerite_forward(t, c, a, U, V, y)
    J = c.shape[0]

    # Backward solve L^T x = z / d:
    def step(carry, inputs):
        F, t_next, U_next, x_next = carry
        tn, Un, Wn, zn = inputs
        F = jnp.exp(-c * (t_next - tn)) * (F + U_next * x_next)
        x = zn - jnp.sum(Wn * F)
        return (F, tn, Un, x), x
    init = (jnp.zeros(J, y.dtype), t[-1], jnp.zeros(J, y.dtype), jnp.zeros((), y.dtype))
    return jax.lax.scan(step, init, (t, U, W, z / d), reverse=True, unroll=CELERITE_UNROLL)[1]


def celerite_predict(term, t, diag, y, t_new):
    """
    Conditional mean K(t_new, t) K^-1 y of a celerite GP in O(N + M) operations. t must be sorted; t_new can be in
    any order.
    """
    c, a, U, V = term.get_celerite_matrices(t, diag)
    alpha = celerite_apply_inverse(t, c, a, U, V, y)
    _, _, Us, Vs = term.get_celerite_matrices(t_new, jnp.zeros_like(t_new))
    J = c.shape[0]

    # Q_n = sum_{m <= n} V_m alpha_m exp(-c (t_n - t_m)):
    def forward(carry, inputs):
        Q, t_prev = carry
        tn, Vn, an = inputs
        Q = jnp.exp(-c * (tn - t_prev)) * Q + Vn * an
        return (Q, tn), Q
    Q = jax.lax.scan(forward, (jnp.zeros(J, y.dtype), t[0]), (t, V, alpha), unroll=CELERITE_UNROLL)[1]

    # R_n = sum_{m > n} U_m alpha_m exp(-c (t_m - t_n)):
    def backward(carry, inputs):
        R, t_next, U_next, a_next = carry
        tn, Un, an = inputs
        R = jnp.exp(-c * (t_next - tn)) * (R + U_next * a_next)
        return (R, tn, Un, an), R
    R = jax.lax.scan(backward, (jnp.zeros(J, y.dtype), t[-1], jnp.zeros(J, y.dtype), jnp.zeros((), y.dtype)),
                     (t, U, alpha), reverse=True, unroll=CELERITE_UNROLL)[1]

    # Contribution of the data points before (t_m <= t_new) and after (t_m > t_new) each prediction time:
    i = jnp.searchsorted(t, t_new, side='right') - 1
    ic = jnp.clip(i, 0, t.shape[0] - 1)
    past = jnp.sum(Us * jnp.exp(-c[None, :] * (t_new - t[ic])[:, None]) * Q[ic], axis=1)
    jc = jnp.clip(i + 1, 0, t.shape[0] - 1)
    future = jnp.sum(Vs * jnp.exp(-c[None, :] * (t[jc] - t_new)[:, None]) * (R[jc] + U[jc] * alpha[jc][:, None]), axis=1)
    return jnp.where(i >= 0, past, 0.) + jnp.where(i + 1 < t.shape[0], future, 0.)


def dense_covariance(kernel_name, h, X1, X2, sigma_factor, nX, legacy_parametrization=False):
    """Covariance matrix between regressors X1 and X2 (each of shape (n,) or (n, nX)) for the george-like kernels."""
    X1 = X1.reshape(X1.shape[0], -1)
    X2 = X2.reshape(X2.shape[0], -1)
    delta = X1[:, None, :] - X2[None, :, :]
    amplitude2 = (h['sigma'] * sigma_factor)**2
    if kernel_name in ['SEKernel', 'M32Kernel']:
        prefix = 'alpha' if kernel_name == 'SEKernel' else 'malpha'
        inverse_metric = jnp.stack([h[prefix + str(i)] for i in range(nX)])
        r2 = jnp.sum(delta**2 * inverse_metric, axis=-1)
        # With nX-dimensional regressors, the george kernels used by juliet <= 2.2.10 had a variance of nX * GP_sigma^2
        # (george's ConstantKernel evaluates to ndim * exp(log_constant)); legacy_parametrization reproduces this:
        if legacy_parametrization:
            amplitude2 = amplitude2 * nX
        if kernel_name == 'SEKernel':
            return amplitude2 * jnp.exp(-0.5 * r2)
        r = safe_sqrt(3. * r2)
        return amplitude2 * (1. + r) * jnp.exp(-r)
    elif kernel_name == 'ExpSineSquaredSEKernel':
        tau = delta[..., 0]
        # juliet <= 2.2.10 set george's (linear) gamma parameter of the ExpSine2Kernel to log(GP_Gamma);
        # legacy_parametrization reproduces this:
        gamma = jnp.log(h['Gamma']) if legacy_parametrization else h['Gamma']
        return amplitude2 * jnp.exp(-0.5 * h['alpha'] * tau**2) * \
               jnp.exp(-gamma * jnp.sin(jnp.pi * jnp.abs(tau) / h['Prot'])**2)
    raise Exception('Kernel ' + kernel_name + ' is not a dense kernel.')


class JaxGP(object):
    """
    Evaluates GP log-likelihoods and predictions for a given kernel and fixed regressors X. The
    variances added to the diagonal (errorbars plus jitters) are passed on each call.
    """

    def __init__(self, kernel_name, X, sigma_factor, matern_eps=0.01, legacy_parametrization=False):
        self.kernel_name = kernel_name
        self.legacy_parametrization = legacy_parametrization
        self.use_celerite = 'Celerite' in kernel_name
        self.sigma_factor = sigma_factor
        self.matern_eps = matern_eps
        X = np.asarray(X, dtype=float)
        self.nX = 1 if X.ndim == 1 else X.shape[1]
        if self.use_celerite:
            if X.ndim != 1:
                raise Exception('celerite kernels (' + kernel_name + ') only accept one-dimensional GP regressors.')
            # celerite needs sorted regressors; sort internally and undo the sorting on the outputs:
            self.idx_sort = np.argsort(X, kind='stable')
            self.idx_unsort = np.argsort(self.idx_sort, kind='stable')
            self.X = jnp.asarray(X[self.idx_sort])
        else:
            self.X = jnp.asarray(X)

    def log_likelihood(self, h, residuals, diag):
        if self.use_celerite:
            term = celerite_term(self.kernel_name, h, self.matern_eps)
            c, a, U, V = term.get_celerite_matrices(self.X, diag[self.idx_sort])
            return celerite_log_likelihood(self.X, c, a, U, V, residuals[self.idx_sort])
        K = dense_covariance(self.kernel_name, h, self.X, self.X, self.sigma_factor, self.nX, self.legacy_parametrization)
        L = jnp.linalg.cholesky(K + jnp.diag(diag))
        alpha = jsl.cho_solve((L, True), residuals)
        return -0.5 * (residuals @ alpha) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * residuals.shape[0] * np.log(2. * np.pi)

    def predict(self, h, residuals, diag, X_new=None):
        """Mean GP prediction at X_new (default: the regressors of the fit) conditioned on the residuals."""
        if self.use_celerite:
            term = celerite_term(self.kernel_name, h, self.matern_eps)
            y, diag = residuals[self.idx_sort], diag[self.idx_sort]
            if X_new is None:
                return celerite_predict(term, self.X, diag, y, self.X)[self.idx_unsort]
            return celerite_predict(term, self.X, diag, y, jnp.asarray(X_new, dtype=float).reshape(-1))
        X_new = self.X if X_new is None else jnp.asarray(X_new)
        K = dense_covariance(self.kernel_name, h, self.X, self.X, self.sigma_factor, self.nX, self.legacy_parametrization)
        L = jnp.linalg.cholesky(K + jnp.diag(diag))
        Ks = dense_covariance(self.kernel_name, h, X_new, self.X, self.sigma_factor, self.nX, self.legacy_parametrization)
        return Ks @ jsl.cho_solve((L, True), residuals)
