from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from jax.scipy.integrate import trapezoid
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayTree, ArrayLikeTree, PRNGKey
from blackjax.util import generate_gaussian_noise

from . import dencalc


def odl_integrator():
    def one_step(
        position: ArrayLikeTree,
        logdensity_grad: ArrayLikeTree,
        divergence: ArrayLikeTree,
        noise: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ) -> ArrayTree:
        position = jax.tree.map(
            lambda p, g, d, n: p
            + step_size * g
            + step_size * d
            + jnp.sqrt(2 * temperature * step_size) * n,
            position,
            logdensity_grad,
            divergence,
            noise,
        )

        return position

    return one_step


def init(position: ArrayLikeTree) -> ArrayLikeTree:
    return position


def build_kernel() -> Callable:
    integrator = odl_integrator()

    def kernel(
        rng_key: PRNGKey,
        position: ArrayLikeTree,
        grad_estimator: Callable,
        preconditioner: Callable,
        minibatch: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ):
        gnmat = preconditioner(position)

        u, s, vh = jnp.linalg.svd(gnmat, hermitian=True)
        smax = s[..., 0]
        tol = 10 * jnp.finfo(gnmat.dtype).eps
        rtol = tol / smax
        s = jnp.where(s > tol, s, jnp.inf)
        prec = jnp.matmul(vh.mT, u.mT / s[..., None])
        prec_sqrt = jnp.matmul(vh.mT, u.mT / jnp.sqrt(s[..., None]))

        key_hutch, key_noise = jax.random.split(rng_key)
        eps = jax.random.rademacher(key_hutch, (20, *position.shape), dtype=gnmat.dtype)
        _, tangents = jax.vmap(
            jax.vmap(
                lambda i, j, vec: jax.jvp(
                    lambda x: jnp.linalg.pinv(
                        preconditioner(x)[i], hermitian=True, rtol=rtol[i]
                    )[j],
                    (position,),
                    (vec,),
                ),
                in_axes=[0, 0, None],
            ),
            in_axes=[None, None, 0],
        )(*jnp.indices(position.shape).reshape(2, -1), eps)
        trace = jnp.einsum(
            "...ij,...j", tangents.reshape((20, *gnmat.shape)), eps
        ).mean(axis=0)

        logden_grad = grad_estimator(position, minibatch)
        noise = generate_gaussian_noise(key_noise, position)

        grad_prec = jnp.einsum("...ij,...j", prec, logden_grad)
        noise_prec = jnp.einsum("...ij,...j", prec_sqrt, noise)

        new_position = integrator(
            position,
            grad_prec,
            trace,
            noise_prec,
            step_size,
            temperature,
        )
        return new_position

    return kernel


def prec_sgld(
    grad_estimator: Callable,
    preconditioner: Callable,
) -> SamplingAlgorithm:
    kernel = build_kernel()

    def init_fn(position: ArrayLikeTree, rng_key=None):
        del rng_key
        return init(position)

    def step_fn(
        rng_key: PRNGKey,
        state: ArrayLikeTree,
        minibatch: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ) -> ArrayTree:
        return kernel(
            rng_key,
            state,
            grad_estimator,
            preconditioner,
            minibatch,
            step_size,
            temperature,
        )

    return SamplingAlgorithm(init_fn, step_fn)


def loglik_fn(
    params,
    batch,
    target,
    coords,
    umat,
    occ,
    aty,
    D,
    sigma_n,
    data_size,
):
    pts, inds = batch[0], batch[1].astype(int)
    params_tr = transform_params(params)

    f_o = target[inds[:, 0], inds[:, 1], inds[:, 2]]
    f_c = dencalc.calc_f(coords, umat, occ, aty, params_tr, pts).sum(axis=0)

    D_s = D[inds[:, 0], inds[:, 1], inds[:, 2]]
    sg_n_s = sigma_n[inds[:, 0], inds[:, 1], inds[:, 2]]

    logpdf = (
        -jnp.mean(
            jnp.log(jnp.pi) + jnp.log(sg_n_s) + jnp.abs(f_o - D_s * f_c) ** 2 / sg_n_s
        )
        * data_size
    )

    return jnp.mean(logpdf)


def logprior_fn(params, means):
    params_tr = transform_params(params)
    logpdf_a = stats.norm.logpdf(params_tr[:, :5], loc=means[:, :5], scale=1.0)
    logpdf_b = stats.expon.logpdf(params_tr[:, 5:], scale=means[:, 5:])
    log_det_jac = log_jacobian_fn(params)

    return jnp.sum(logpdf_a) + jnp.sum(logpdf_b) + log_det_jac


def scheduler(k, start=1e-7):
    return start * k ** (-0.33)


def inference_loop(rng_key, kernel, batches, initial_state):
    @jax.jit
    def one_step(state, tree):
        rng_key, batch, step_size, itnum = tree
        jax.lax.cond(
            itnum % 100 == 0,
            lambda *_: jax.debug.print("iteration {}", itnum),
            lambda *_: None,
        )
        state = kernel(rng_key, state, batch, step_size)

        return state, state

    num_samples = batches.shape[0]
    counter = jnp.arange(num_samples) + 1
    step_size = scheduler(counter, start=1e-7)
    initial_state_tr = inv_transform_params(initial_state)

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(
        one_step,
        initial_state_tr,
        (keys, batches, step_size, counter),
    )

    return states, step_size


@jax.jit
def transform_params(params):
    return jnp.concatenate(
        [
            params[..., :5],
            jax.nn.softplus(params[..., 5:]),
        ],
        axis=-1,
    )


@jax.jit
def log_jacobian_fn(params):
    log_diag = jnp.concatenate(
        [
            jnp.zeros_like(params[..., :5]),
            -jax.nn.softplus(-params[..., 5:]),
        ],
        axis=-1,
    )
    logdet = jnp.sum(log_diag)
    return logdet


@jax.jit
def inv_transform_params(params):
    return jnp.concatenate(
        [
            params[..., :5],
            params[..., 5:] + jnp.log(-jnp.expm1(-params[..., 5:])),
        ],
        axis=-1,
    )


@jax.jit
def _calc_hess_atom(umat, occ, aty, gnmat, D, sigma_n, bins):
    b_eff = umat.trace() / 3
    b_cont = D**2 * occ**2 * jnp.exp(-b_eff * bins**2 / 2) / sigma_n
    integrand = 4 * jnp.pi * bins**2 * gnmat[aty] * b_cont

    spacing = bins[1] - bins[0]
    precond = trapezoid(integrand, dx=spacing)

    return precond


def calc_hess(params, umat, occ, aty, naty, D, sigma_n, bins):
    params_tr = transform_params(params)
    grad_a = dencalc.oc1d_vmap(
        jnp.ones_like(params[:, :5]),
        params_tr[:, 5:],
        bins,
    )
    grad_b = dencalc.oc1d_vmap(
        params_tr[:, :5],
        params_tr[:, 5:],
        bins,
    ) * (-(bins**2) / 4)
    grad_b_tr = (grad_b.T * jax.lax.logistic(params[:, 5:]).T).T

    grad = jnp.concatenate([grad_a, grad_b_tr], axis=1)
    gnmat = grad[:, None, ...] * grad[:, :, None, ...]

    prec_atoms = jax.vmap(
        _calc_hess_atom,
        in_axes=[0, 0, 0, None, None, None, None],
    )(umat, occ, aty, gnmat, D, sigma_n, bins)
    prec = jax.ops.segment_sum(prec_atoms, segment_ids=aty, num_segments=naty)

    return prec
