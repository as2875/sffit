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
    y = ucho.T @ pts.T
    s_U_s = jnp.linalg.vector_norm(y, axis=0) ** 2

    return a * jnp.exp(-s_U_s / 4)


oc3d_vmap = jax.vmap(one_coef_3d, in_axes=[0, 0, None, None, None])


@jax.jit
def _calc_f_atom(coord, umat, occ, it92, aty, weights, sigma, pts):
    f_at = (
        occ
        * weights[aty]
        * oc3d_vmap(
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
def calc_f_scan(coords, umat, occ, it92, aty, weights, sigma, pts):
    @jax.jit
    def one_atom(carry, tree):
        coord, umat, occ, it92, aty = tree
        vals = _calc_f_atom(coord, umat, occ, it92, aty, weights, sigma, pts1d)
        return carry + vals, None

    pts1d = pts.reshape(-1, 3)
    f_o, _ = jax.lax.scan(
        one_atom,
        jnp.zeros(len(pts1d), dtype=complex),
        (coords, umat, occ, it92, aty),
    )
    return f_o


@jax.jit
def one_coef_re(a, b, umat, sigma, pts):
    ucho = _add_and_cho(umat, b, sigma)
    y = triangular_solve(ucho.T, pts.T, left_side=True)
    r_U_r = jnp.linalg.vector_norm(y, axis=0) ** 2

    udet = ucho[0, 0] * ucho[1, 1] * ucho[2, 2]

    den = (a * 8 * jnp.pi * jnp.sqrt(jnp.pi) * jnp.exp(-4 * jnp.pi**2 * r_U_r)) / udet

    return den


ocre_vmap = jax.vmap(one_coef_re, in_axes=[0, 0, None, None, None])


@partial(jax.jit, static_argnames=["rcut"])
def _make_small_grid(coord, mgrid, rcut):
    dist = (mgrid.T - coord).T ** 2
    coords = jnp.argmin(dist, axis=1)

    inds3d = jnp.indices((rcut, rcut, rcut))
    inds1d = jnp.column_stack(
        [inds3d[i].ravel() + coords[i] - rcut // 2 for i in range(3)]
    )

    angpix = mgrid[0, 1] - mgrid[0, 0]
    pts1d = jnp.column_stack([inds3d[i].ravel() - rcut // 2 for i in range(3)])
    pts1d *= angpix

    return inds1d, pts1d


@partial(jax.jit, static_argnames=["rcut"])
def _calc_v_atom_sparse(carry, tree, weights, sigma, mgrid, rcut):
    coord, umat, occ, it92, aty = tree
    inds1d, pts1d = _make_small_grid(coord, mgrid, rcut)

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

    dim = mgrid.shape[1]
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


def calc_rcut(length, spacing):
    return int(length / (2 * spacing)) * 2


@partial(jax.jit, static_argnames=["nbins"])
def make_bins(data, bsize, spacing, dmax, nbins):
    axes = (
        jnp.fft.fftfreq(data.shape[0], d=spacing),
        jnp.fft.fftfreq(data.shape[0], d=spacing),
        jnp.fft.rfftfreq(data.shape[0], d=spacing),
    )

    sx, sy, sz = jnp.meshgrid(*axes, indexing="ij")
    s = jnp.sqrt(sx**2 + sy**2 + sz**2)

    dmin = 1 / (bsize * spacing)
    bins = jnp.linspace(dmin, dmax, nbins + 1)
    bin_cent = 0.5 * (bins[1:] + bins[:-1])
    sdig = jnp.digitize(s, bins) - 1

    s_vec = jnp.stack([sx, sy, sz], axis=-1)

    return s_vec, sdig, bin_cent


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
    fun_wt_sg = bins**4 * weights[aty] * fs_2 * b_cont / sigma_n

    h_wt = 8 * jnp.pi * trapezoid(fun_wt, dx=spacing)
    h_sg = 0.5 * jnp.pi * trapezoid(fun_sg, dx=spacing)
    h_wt_sg = -2 * jnp.pi * trapezoid(fun_wt_sg, dx=spacing)

    precond = jnp.array(
        [
            [h_wt, h_wt_sg],
            [h_wt_sg, h_sg],
        ],
    )

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
    prec = jax.ops.segment_sum(prec_atoms, segment_ids=aty, num_segments=naty)

    return prec


@partial(jax.jit, static_argnames=["rcut"])
def _calc_gaussian_atom(coord, umat, mgrid, rcut):
    inds, pts = _make_small_grid(coord, mgrid, rcut)
    gauss = one_coef_re(1, 0, umat, 0, pts)
    dim = mgrid.shape[1]
    gauss_coo = sparse.BCOO((gauss.ravel(), inds), shape=(dim, dim, dim))

    return gauss_coo


@partial(jax.jit, static_argnames=["rcut", "naty"])
def calc_gaussians(coords, umat, aty, mgrid, rcut, naty):
    gauss = jax.vmap(_calc_gaussian_atom, in_axes=[0, 0, None, None])(
        coords, umat, mgrid, rcut
    )
    dim = mgrid.shape[1]
    summed, _ = jax.lax.scan(
        lambda c, t: (c.at[t[1]].add(gauss[t[0]].todense()), None),
        jnp.zeros((naty, dim, dim, dim)),
        (jnp.arange(len(gauss)), aty),
    )

    f_c = jax.lax.map(lambda x: jnp.fft.fftn(x), summed)

    return f_c


@partial(jax.jit, static_argnames=["naty"])
def calc_gaussians_direct(coords, umat, aty, pts, naty, fft_scale):
    @jax.jit
    def one_gaussian(carry, tree):
        coord, umat, aty = tree
        vals = one_coef_3d(1, 0, umat, 0, pts1d)
        phase = jnp.exp(-2 * jnp.pi * 1j * coord @ pts1d.T)

        new = carry.at[aty].add(vals * phase / fft_scale)
        return new, None

    pts1d = pts.reshape(-1, 3)
    gauss, _ = jax.lax.scan(
        one_gaussian,
        jnp.zeros((naty, len(pts1d)), dtype=complex),
        (coords, umat, aty),
    )
    return gauss
