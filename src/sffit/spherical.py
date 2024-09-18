from functools import partial

import jax
import jax.numpy as jnp
import jax_finufft as finufft
import numpy as np


def mw_weights(m):
    if m == 1:
        return 1j * np.pi / 2

    elif m == -1:
        return -1j * np.pi / 2

    elif m % 2 == 0:
        return 2 / (1 - m**2)

    else:
        return 0


def quad_weights_mw_theta(lmax):
    w = np.zeros(2 * lmax - 1, dtype=complex)
    for i in range(-(lmax - 1), lmax):
        w[i + lmax - 1] = mw_weights(i)

    w *= np.exp(-1j * np.arange(-(lmax - 1), lmax) * np.pi / (2 * lmax - 1))
    wr = np.real(np.fft.fft(np.fft.ifftshift(w), norm="backward")) / (2 * lmax - 1)
    q = wr[:lmax]

    q[: lmax - 1] = q[: lmax - 1] + wr[-1 : lmax - 1 : -1]

    return q


def make_spherical_grid(s, N, lmax):
    p = np.arange(0, 2 * lmax - 1).reshape(1, -1)
    phi = 2 * p * np.pi / (2 * lmax - 1)
    t = np.arange(0, lmax).reshape(-1, 1)
    theta = (2 * t + 1) * np.pi / (2 * lmax - 1)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta) * np.ones(phi.shape)

    x, y, z = map(
        lambda arr: 2 * np.pi * np.ravel(arr.reshape(1, -1) * s.reshape(-1, 1)),
        [x, y, z],
    )

    return x, y, z


def make_quad_weights(shape, lmax):
    nphi = 2 * lmax - 1
    quad_weights_theta = quad_weights_mw_theta(lmax) * 2 * np.pi / (2 * lmax - 1)
    quadwt = jax.lax.broadcast(
        jnp.repeat(jnp.asarray(quad_weights_theta), nphi),
        shape,
    )
    return quadwt


@jax.jit
def calc_nufft(gaussians, freqs, x, y, z):
    _, nufft = jax.lax.scan(
        lambda _, ind: (
            None,
            finufft.nufft2(
                gaussians[ind].todense().astype(complex),
                x,
                y,
                z,
            ),
        ),
        None,
        jnp.arange(len(gaussians)),
    )

    return nufft.reshape(len(gaussians), len(freqs), -1)


@jax.jit
def calc_mat(nufft, quadwt):
    return jnp.einsum("i...j,k...j->ik...", nufft * quadwt, nufft.conj())


@jax.jit
def calc_vec(mpdata, nufft, freqs, quadwt, x, y, z):
    f_o = finufft.nufft2(mpdata.astype(complex), x, y, z).reshape(len(freqs), -1)
    vec = jnp.einsum("...j,i...j->i...", f_o * quadwt, nufft.conj())
    return vec


@jax.jit
def batch_lstsq(mats, vecs):
    lstsq_part = partial(jnp.linalg.lstsq, rcond=1e-6)
    return jax.vmap(lstsq_part)(mats.T, vecs.T)
