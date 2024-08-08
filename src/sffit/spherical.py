from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import sph_harm

import jax_finufft as finufft
import scipy.special
import s2fft
from s2fft.sampling import s2_samples


# Spherical Bessel function wrappers are adapted from
# https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html


@jax.custom_jvp
def spherical_jn(n, z):
    def _scipy_sjn(n, z):
        return scipy.special.spherical_jn(n, z).astype(z.dtype)

    n, z = jnp.asarray(n), jnp.asarray(z)

    # Require the order n to be integer type
    assert jnp.issubdtype(n.dtype, jnp.integer)

    # Promote the input to inexact (float/complex).
    # Note that jnp.result_type() accounts for the enable_x64 flag.
    z = z.astype(jnp.result_type(float, z.dtype))

    # Define the expected shape & dtype of output.
    result_shape_dtype = jax.ShapeDtypeStruct(
        shape=jnp.broadcast_shapes(n.shape, z.shape), dtype=z.dtype
    )

    # We use vectorize=True because scipy.special.spherical_jn handles broadcasted inputs.
    return jax.pure_callback(_scipy_sjn, result_shape_dtype, n, z, vectorized=True)


@spherical_jn.defjvp
def _sjn_jvp(primals, tangents):
    n, z = primals
    _, z_dot = tangents  # Note: n_dot is always 0 because n is integer.
    sjn_minus_1, sjn, sjn_plus_1 = (
        spherical_jn(n - 1, z),
        spherical_jn(n, z),
        spherical_jn(n + 1, z),
    )
    dsjn_dz = jnp.where(n == 0, -sjn_plus_1, sjn_minus_1 - (n + 1) * sjn / z)
    return spherical_jn(n, z), z_dot * dsjn_dz


@partial(jax.jit, static_argnames=["lmax"], backend="cpu")
def _calc_delta_atom(carry, tree, freq, l_1d, m_1d, lmax):
    coord, b_iso, aty = tree
    x, y, z = coord
    hxy = jnp.hypot(x, y)
    r = jnp.hypot(hxy, z)
    phi = jnp.arctan2(y, x)
    theta = jnp.arctan2(z, hxy)

    sh_cont = sph_harm(
        m_1d,
        l_1d,
        jax.lax.broadcast(theta, l_1d.shape),
        jax.lax.broadcast(phi, l_1d.shape),
        n_max=lmax,
    )
    b_cont = jnp.exp(-b_iso * freq**2 / 4)

    bessel_arg = 2 * jnp.pi * freq * r
    bessel_cont = spherical_jn(
        jax.lax.broadcast(l_1d, freq.shape).T,
        jax.lax.broadcast(bessel_arg, l_1d.shape),
    )
    delta_at = jnp.sqrt(2 / jnp.pi) * (
        bessel_cont * b_cont * jax.lax.broadcast(sh_cont, freq.shape).T
    )

    new = carry.at[aty].add(delta_at)

    return new, None


@partial(jax.jit, static_argnames=["lmax", "naty"], backend="cpu")
def calc_delta(coords, b_iso, freq, l_1d, m_1d, lmax, aty, naty):
    wrapped = partial(
        _calc_delta_atom,
        freq=freq,
        l_1d=l_1d,
        m_1d=m_1d,
        lmax=lmax,
    )
    deltas, _ = jax.lax.scan(
        wrapped,
        jnp.zeros((naty, len(l_1d), len(freq)), dtype=complex),
        (coords, b_iso, aty),
    )
    return deltas


def calc_lm(lmax):
    l_1d = jnp.repeat(
        jnp.arange(lmax),
        jnp.arange(1, 2 * lmax, 2),
    )
    m_1d = jnp.arange(len(l_1d)) - l_1d**2 - l_1d
    return l_1d, m_1d


def calc_rhs(mpdata, deltas, freqs, l_1d, lmax):
    nphi = s2_samples.nphi_equiang(lmax, sampling="mw")
    ntheta = s2_samples.ntheta(lmax, sampling="mw")
    phi = s2_samples.phis_equiang(lmax, sampling="mw").reshape(1, -1)
    theta = s2_samples.thetas(lmax, sampling="mw").reshape(-1, 1)

    x = jnp.pi * jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.cos(phi)
    z = jnp.cos(theta) * jnp.ones(phi.shape)

    R = mpdata.shape[0] // 2
    x, y, z = map(
        lambda arr: jnp.ravel(arr.reshape(1, -1) * freqs.reshape(-1, 1) / R),
        [x, y, z],
    )

    nufft = finufft.nufft2(mpdata.astype(complex), x, y, z).reshape(
        len(freqs), ntheta, nphi
    )
    sht = jax.vmap(s2fft.forward_jax, in_axes=[0, None])(nufft, lmax)
    sht1d = jax.vmap(s2fft.sampling.reindex.flm_2d_to_1d_fast, in_axes=[0, None])(
        sht, lmax
    )
    sht1d = sht1d * 1j**l_1d
    inner = jnp.sum(deltas * sht1d.T.conj(), axis=1) / (2 * jnp.pi) ** 1.5

    return inner.real
