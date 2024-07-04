from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
from jax.lax.linalg import cholesky, triangular_solve
from jax.scipy.integrate import trapezoid


@jax.jit
def _add_and_cho(umat, b, sigma):
    umatb = umat + (b + sigma) * jnp.identity(3)
    ucho = cholesky(umatb, symmetrize_input=False)
    return ucho


@jax.jit
def one_coef_3d(a, b, umat, sigma, pts):
    ucho = _add_and_cho(umat, b, sigma)
    y = ucho @ pts.T
    s_U_s = jnp.linalg.vector_norm(y, axis=0) ** 2

    return a * jnp.exp(-s_U_s / 4)


@jax.jit
def _calc_f_atom(coord, umat, occ, it92, aty, weights, sigma, pts):
    f_at = (
        occ
        * weights[aty]
        * ocre_vmap(
            it92[:5],
            it92[5:],
            umat,
            sigma[aty],
            pts,
        ).sum(axis=0)
    )
    phase = jnp.exp(-2 * jnp.pi * 1j * coord @ pts.T)

    return f_at * phase


calc_f = jax.vmap(_calc_f_atom, in_axes=[0, 0, 0, 0, 0, None, None, None])


@jax.jit
def one_coef_re(a, b, umat, sigma, pts):
    ucho = _add_and_cho(umat, b, sigma)
    y = triangular_solve(ucho, pts.T, left_side=True)
    r_U_r = jnp.linalg.vector_norm(y, axis=0) ** 2

    udet = ucho[0, 0] * ucho[1, 1] * ucho[2, 2]

    den = (
        a * (4 * jnp.pi) * jnp.sqrt(4 * jnp.pi) * jnp.exp(-4 * jnp.pi**2 * r_U_r)
    ) / udet

    return den


ocre_vmap = jax.vmap(one_coef_re, in_axes=[0, 0, None, None, None])


@partial(jax.jit, static_argnames=["rcut"])
def _calc_v_atom_sparse(carry, tree, weights, sigma, mgrid, rcut):
    coord, umat, occ, it92, aty = tree

    dist = (mgrid.T - coord).T ** 2
    coords = jnp.argmin(dist, axis=1)
    dim = mgrid.shape[1]

    inds3d = jnp.indices((rcut, rcut, rcut))
    inds1d = jnp.column_stack(
        [inds3d[i].ravel() + coords[i] - rcut // 2 for i in range(3)]
    )

    angpix = mgrid[0, 1] - mgrid[0, 0]
    pts1d = jnp.column_stack([inds3d[i].ravel() - rcut // 2 for i in range(3)])
    pts1d *= angpix

    v_small = (
        occ
        * weights[aty]
        * ocre_vmap(
            it92[:5],
            it92[5:],
            umat,
            sigma[aty],
            pts1d,
        )
        .sum(axis=0)
        .reshape(rcut, rcut, rcut)
    )

    v_at = sparse.BCOO((v_small.ravel(), inds1d), shape=(dim, dim, dim))

    return carry + v_at, None


@partial(jax.jit, static_argnames=["rcut", "nsamples"])
def calc_v_sparse(coord, umat, occ, it92, aty, weights, sigma, rcut, bounds, nsamples):
    mgrid = make_grid(bounds, nsamples)
    wrapped = partial(
        _calc_v_atom_sparse,
        weights=weights,
        sigma=sigma,
        mgrid=mgrid,
        rcut=rcut,
    )
    v_mol, _ = jax.lax.scan(
        wrapped,
        jnp.zeros((nsamples, nsamples, nsamples)),
        (coord, umat, occ, it92, aty),
    )

    return v_mol


def make_grid(bounds, nsamples):
    axis = jnp.linspace(bounds[:, 0], bounds[:, 1], nsamples, axis=-1, endpoint=False)
    return axis


@partial(jax.jit, static_argnames=["nbins"])
def make_bins(data, spacing, dmax, nbins):
    axis = jnp.fft.fftfreq(data.shape[0], d=spacing)

    sx, sy, sz = jnp.meshgrid(axis, axis, axis)
    s = jnp.sqrt(sx**2 + sy**2 + sz**2)

    bins = jnp.linspace(0, dmax, nbins + 1)
    sdig = jnp.digitize(s, bins) - 1

    s_vec = jnp.stack([sx, sy, sz], axis=-1)

    return s_vec, sdig, bins


@jax.jit
def calc_ml_params(v_o, v_c, fbins, labels):
    @jax.jit
    def calc_D(carry, ind):
        msk = (fbins == ind).astype(int)
        D_bin = (prec_D1 * msk).sum() / (prec_D2 * msk).sum()
        return carry, D_bin

    @jax.jit
    def calc_S(carry, tree):
        ind, D = tree
        msk = (fbins == ind).astype(int)
        S_bin = (
            jnp.sum(
                jnp.abs(f_o - D * f_c) ** 2 * msk,
            )
            / msk.sum()
        )
        return carry, S_bin

    f_o = jnp.fft.fftn(v_o)
    f_c = jnp.fft.fftn(v_c)
    prec_D1 = jnp.real(f_o * f_c.conj())
    prec_D2 = jnp.abs(f_c) ** 2

    _, D = jax.lax.scan(calc_D, None, labels)
    _, S = jax.lax.scan(calc_S, None, (labels, D))

    return D, S


@jax.jit
def one_coef_1d(a, b, bins):
    return a * jnp.exp(-b * bins**2 / 4)


oc1d_vmap = jax.vmap(one_coef_1d, in_axes=[0, 0, None])


@jax.jit
def _calc_hess_atom(carry, tree, weights, sigma, sigma_n, bins):
    coord, umat, occ, it92, aty = tree
    fs_2 = occ**2 * oc1d_vmap(it92[:5], it92[5:], bins).sum(axis=0) ** 2
    b_eff = umat.trace() / 3
    b_cont = jnp.exp(-0.5 * (b_eff + sigma[aty]) * bins**2)

    spacing = bins[1] - bins[0]
    fun_wt = bins**2 * fs_2 * b_cont / sigma_n
    fun_sg = bins**6 * weights[aty] ** 2 * fs_2 * b_cont / sigma_n
    h_wt = 8 * jnp.pi * trapezoid(fun_wt, dx=spacing)
    h_sg = 0.5 * jnp.pi * trapezoid(fun_sg, dx=spacing)
    precond = {"weights": h_wt, "sigma": h_sg}

    return carry, precond


@partial(jax.jit, static_argnames=["naty"])
def calc_hess(params, coords, umat, occ, it92, aty, naty, sigma_n, bins):
    weights, sigma = (
        params["weights"],
        params["sigma"] ** 2,
    )
    wrapped = partial(
        _calc_hess_atom,
        weights=weights,
        sigma=sigma,
        sigma_n=sigma_n,
        bins=bins,
    )
    _, prec_atoms = jax.lax.scan(
        wrapped,
        None,
        (coords, umat, occ, it92, aty),
    )
    prec = jax.tree.map(
        partial(jax.ops.segment_sum, segment_ids=aty, num_segments=naty),
        prec_atoms,
    )

    return prec
