from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def _mask_inner(freq_index, inner, fbins):
    msk = (fbins == freq_index).astype(int)
    return jnp.sum(inner * msk)


@jax.jit
def calc_mats(gaussians, fbins, labels):
    @jax.jit
    def one_element(mat_index):
        inner = gaussians[mat_index[0]] * gaussians[mat_index[1]].conj()
        inner_binned = jax.lax.map(
            partial(_mask_inner, inner=inner, fbins=fbins),
            labels,
        )
        return inner_binned

    naty = len(gaussians)
    gaussians = gaussians.reshape(naty, -1)
    fbins = fbins.ravel()

    colx, coly = jnp.triu_indices(naty)
    mat_indices = jnp.column_stack((colx, coly))
    mats1d = jax.lax.map(one_element, mat_indices).T
    mats2d = jnp.zeros((len(labels), naty, naty), dtype=complex)
    mats2d = mats2d.at[:, colx, coly].set(mats1d)
    mats2d = mats2d.at[:, coly, colx].set(mats1d)

    return mats2d


@jax.jit
def calc_vecs(mpdata, gaussians, fbins, labels):
    @jax.jit
    def one_element(vec_index):
        inner = f_o * gaussians[vec_index].conj()
        inner_binned = jax.lax.map(
            partial(_mask_inner, inner=inner, fbins=fbins),
            labels,
        )
        return inner_binned

    naty = len(gaussians)
    f_o = jnp.fft.fftn(mpdata).ravel()
    gaussians = gaussians.reshape(naty, -1)
    fbins = fbins.ravel()
    vec_indices = jnp.arange(naty)

    vecs = jax.lax.map(one_element, vec_indices).T

    return vecs


@jax.jit
def batch_lstsq(mats, vecs):
    lstsq_part = partial(jnp.linalg.lstsq, rcond=None)
    soln, _, _, _ = jax.vmap(lstsq_part)(mats, vecs)
    return soln
