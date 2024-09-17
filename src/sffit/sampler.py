import sys
from functools import partial
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


@jax.jit
def sqrtm2(mat):
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    trace = mat[0, 0] + mat[1, 1]
    s = jnp.sqrt(det)
    t = jnp.sqrt(trace + 2 * s)
    sqrt = (mat + s * jnp.identity(2)) / t

    return sqrt


sqrtm2_vmap = jax.vmap(sqrtm2)


@jax.jit
def inv2(mat):
    eigvals = jnp.linalg.eigvalsh(mat)
    emin = eigvals.min()
    modif = jax.lax.cond(
        emin < 1e-6,
        lambda mat, tau: mat + tau * jnp.identity(2),
        lambda mat, tau: mat,
        mat,
        1e-6 - emin,
    )

    inv = jnp.linalg.inv(modif)
    return inv


inv2_vmap = jax.vmap(inv2)


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
        prec = inv2_vmap(
            preconditioner(position),
        )
        prec_sqrt = sqrtm2_vmap(prec)
        logden_grad = grad_estimator(position, minibatch)
        noise = generate_gaussian_noise(rng_key, position)

        grad_prec = jnp.einsum(
            "ijk,ik->ij",
            prec,
            jnp.stack([logden_grad["weights"], logden_grad["sigma"]], axis=-1),
        )
        noise_prec = jnp.einsum(
            "ijk,ik->ij",
            prec_sqrt,
            jnp.stack([noise["weights"], noise["sigma"]], axis=-1),
        )

        new_position = integrator(
            position,
            {"weights": grad_prec[:, 0], "sigma": grad_prec[:, 1]},
            {"weights": noise_prec[:, 0], "sigma": noise_prec[:, 1]},
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
    it92,
    aty,
    D,
    sigma_n,
    data_size,
):
    weights, sigma = (
        params["weights"],
        params["sigma"] ** 2,
    )
    pts, inds = batch[0], batch[1].astype(int)

    f_o = target[inds[:, 0], inds[:, 1], inds[:, 2]]
    f_c = dencalc.calc_f(coords, umat, occ, it92, aty, weights, sigma, pts).sum(axis=0)

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


def logprior_fn(params):
    weights, sigma = (
        params["weights"],
        params["sigma"],
    )
    logpdf_wt = stats.norm.logpdf(weights, loc=1.0, scale=1.0)
    logpdf_sg = stats.norm.logpdf(sigma, loc=0.0, scale=1.0)

    return jnp.sum(logpdf_wt) + jnp.sum(logpdf_sg)


def scheduler(k):
    return 1e-6 * jnp.ones_like(k)


def inference_loop(rng_key, kernel, batches, initial_state, grad_vmap):
    @jax.jit
    def calc_lmax(state, batch, step_size):
        grads = grad_vmap(state, batch[..., None, :])
        grads_arr = jnp.concatenate([grads["weights"].T, grads["sigma"].T])
        grads_cov = jnp.cov(grads_arr)
        l_max = step_size * jnp.linalg.eigvalsh(grads_cov).max()

        return l_max

    @jax.jit
    def one_step(state, tree):
        rng_key, batch, step_size, itnum = tree
        jax.lax.cond(
            itnum % 100 == 0,
            lambda *_: jax.debug.print("iteration {}", itnum),
            lambda *_: None,
        )
        state = kernel(rng_key, state, batch, step_size)
        l_max = jax.lax.cond(
            False,
            calc_lmax,
            lambda *_: jnp.nan,
            state,
            batch,
            step_size,
        )
        acc = dict(statistic=l_max, **state)

        return state, acc

    num_samples = batches.shape[0]
    counter = jnp.arange(num_samples) + 1
    step_size = scheduler(counter)

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(
        one_step,
        initial_state,
        (keys, batches, step_size, counter),
    )

    return states
