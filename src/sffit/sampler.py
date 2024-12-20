from functools import partial
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
        tol = 100 * jnp.finfo(gnmat.dtype).eps
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
    coords,
    umat,
    occ,
    aty,
    molind,
    data_size,
    nmol,
):
    def one_mol(tree):
        ind, molpts = tree
        fcmol = dencalc.calc_f(coords, umat, occ, aty, params_tr, molpts)
        msk = jnp.astype(molind == ind, int)
        return jnp.sum((fcmol.T * msk).T, axis=0)

    pts, f_o, D, sigma_n = batch
    params_tr = transform_params(params)
    f_c = jax.lax.map(one_mol, (jnp.arange(nmol), pts))

    logpdf = (
        -jnp.mean(
            jnp.log(jnp.pi) + jnp.log(sigma_n) + jnp.abs(f_o - D * f_c) ** 2 / sigma_n,
            axis=1,
        )
        * data_size
    )

    return logpdf.sum()


def logprior_fn(params):
    params_tr = transform_params(params)
    logpdf_a = stats.norm.logpdf(params_tr[:, :5], loc=0.0, scale=1.0)
    logpdf_b = stats.expon.logpdf(params_tr[:, 5:], scale=1.0)
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

    num_samples = len(batches[0])
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


@partial(jax.jit, static_argnames=["naty"])
def _calc_hess_mol(params, umat, occ, aty, naty, D, sigma_n, bins):
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


@partial(jax.jit, static_argnames=["nmol", "naty"])
def calc_hess(params, umat, occ, aty, molind, nmol, naty, D, sigma_n, bins):
    def one_mol(carry, tree):
        D, sigma_n, bins, ind = tree
        msk = jnp.astype(molind == ind, int)
        new = carry + _calc_hess_mol(
            params, umat, msk * occ, aty, naty, D, sigma_n, bins
        )
        return new, None

    prec, _ = jax.lax.scan(
        one_mol,
        jnp.zeros((naty, 10, 10)),
        (D, sigma_n, bins, jnp.arange(nmol)),
    )
    return prec


@jax.jit
def eval_sog(it92, freq, weights):
    def one_sample(sample):
        bc = jnp.broadcast_to(sample.T, (*freq.T.shape, *sample.T.shape)).T
        return jnp.sum(
            bc[:, :5, :] * jnp.exp(-bc[:, 5:, :] * freq**2 / 4), axis=1
        )

    sf = jax.lax.map(one_sample, it92)
    mean = jnp.average(sf, axis=0, weights=weights)
    return mean.T
