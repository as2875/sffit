from typing import Callable

import jax
import jax.numpy as jnp
from blackjax.base import SamplingAlgorithm
from blackjax.types import ArrayTree, ArrayLikeTree, PRNGKey
from blackjax.util import generate_gaussian_noise


def odl_integrator():
    def one_step(
        position: ArrayLikeTree,
        logdensity_grad: ArrayLikeTree,
        noise: ArrayLikeTree,
        step_size: float,
        temperature: float = 1.0,
    ) -> ArrayTree:
        position = jax.tree_util.tree_map(
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
        prec = preconditioner(position)
        logden_grad = grad_estimator(position, minibatch)
        noise = generate_gaussian_noise(rng_key, position)

        grad_prec = jax.tree.map(
            lambda g, p: g / (p + 1e-3),
            logden_grad,
            prec,
        )
        noise_prec = jax.tree.map(
            lambda n, p: n / jnp.sqrt(p + 1e-3),
            noise,
            prec,
        )

        new_position = integrator(
            position,
            grad_prec,
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
