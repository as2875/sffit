import jax
import jax.numpy as jnp
import jax_finufft as finufft
from s2fft.sampling import s2_samples
from s2fft.utils import quadrature

from . import dencalc


def make_spherical_grid(s, N, lmax):
    phi = s2_samples.phis_equiang(lmax, sampling="mw").reshape(1, -1)
    theta = s2_samples.thetas(lmax, sampling="mw").reshape(-1, 1)

    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta) * jnp.ones(phi.shape)

    x, y, z = map(
        lambda arr: 2 * jnp.pi * jnp.ravel(arr.reshape(1, -1) * s.reshape(-1, 1)),
        [x, y, z],
    )

    return x, y, z


def make_quad_weights(shape, lmax):
    nphi = s2_samples.nphi_equiang(lmax, sampling="mw")
    quadwt = jax.lax.broadcast(
        jnp.repeat(quadrature.quad_weights(lmax, sampling="mw"), nphi),
        shape,
    )
    return quadwt


def calc_nufft(coords, umat, aty, mgrid, rcut, naty, freqs, lmax):
    gaussians = dencalc.calc_gaussians(coords, umat, aty, mgrid, rcut, naty)
    x, y, z = make_spherical_grid(freqs, mgrid.shape[1], lmax)
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
        jnp.arange(naty),
    )

    return nufft.reshape(naty, len(freqs), -1)


def calc_vec(mpdata, nufft, freqs, quadwt, lmax):
    x, y, z = make_spherical_grid(freqs, mpdata.shape[0], lmax)
    f_o = finufft.nufft2(mpdata.astype(complex), x, y, z).reshape(len(freqs), -1)
    vec = jnp.einsum("...j,i...j->i...", f_o * quadwt, nufft.conj())
    return vec
