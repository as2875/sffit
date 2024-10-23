from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayTree, ArrayLikeTree, PRNGKey
from blackjax.util import generate_gaussian_noise

from . import dencalc


def odl_integrator():
    def one_step(
        position: ArrayLikeTree,
        logdensity_grad: ArrayLikeTree,
        noise: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ) -> ArrayTree:
        position = jax.tree.map(
            lambda p, g, n: p
            + step_size * g
            + jnp.sqrt(2 * temperature * step_size) * n,
            position,
            logdensity_grad,
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
        eigvals, eigvecs = jnp.linalg.eigh(gnmat)
        eigvals_inv = jnp.where(eigvals < 1e-6, 0.0, 1 / eigvals)
        prec = jnp.einsum("...ij,...kj", eigvals_inv[:, None, :] * eigvecs, eigvecs)
        prec_sqrt = jnp.einsum(
            "...ij,...kj", jnp.sqrt(eigvals_inv[:, None, :]) * eigvecs, eigvecs
        )

        logden_grad = grad_estimator(position, minibatch)
        noise = generate_gaussian_noise(rng_key, position)

        grad_prec = jnp.einsum("...ij,...j", prec, logden_grad)
        noise_prec = jnp.einsum("...ij,...j", prec_sqrt, noise)

        new_position = integrator(
            position,
            grad_prec,
            noise_prec,
            step_size,
            temperature,
        )
        new_position_filtered = jax.tree.map(
            lambda x, y: jnp.where(jnp.isnan(x), y, x),
            new_position,
            position,
        )

        return new_position_filtered

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
    fbins,
    nbins,
    coords,
    umat,
    occ,
    aty,
    D,
    sigma_n,
    data_size,
):
    pts, inds = batch[0], batch[1].astype(int)
    params_tr = params.at[:, 5:].power(2)

    f_o = target[inds[:, 0], inds[:, 1], inds[:, 2]]
    f_c = dencalc.calc_f(coords, umat, occ, aty, params_tr, pts).sum(axis=0)

    D_s = D[inds[:, 0], inds[:, 1], inds[:, 2]]
    sg_n_s = sigma_n[inds[:, 0], inds[:, 1], inds[:, 2]]

    logpdf = -(
        jnp.log(jnp.pi) + jnp.log(sg_n_s) + jnp.abs(f_o - D_s * f_c) ** 2 / sg_n_s
    ) * (data_size / len(pts))
    logpdf = jnp.where(
        sg_n_s < 1e-6,
        jnp.nan,
        logpdf,
    )

    return jnp.nanmean(logpdf)


def logprior_fn(params, means):
    logpdf_a = stats.norm.logpdf(params[:, :5], loc=means[:, :5], scale=1.0)
    logpdf_b = stats.expon.logpdf(params[:, 5:] ** 2, loc=means[:, 5:], scale=1.0)
    return jnp.sum(logpdf_a) + jnp.sum(logpdf_b)


def scheduler(k):
    return 1e-6 * jnp.ones_like(k)


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
    step_size = scheduler(counter)
    initial_state_tr = initial_state.at[:, 5:].power(0.5)

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(
        one_step,
        initial_state_tr,
        (keys, batches, step_size, counter),
    )

    return states
