from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
from jax.lax.linalg import cholesky, triangular_solve


@jax.jit
def _add_and_cho(umat, b):
    umatb = umat + b * jnp.identity(3)
    ucho = cholesky(umatb, symmetrize_input=False)
    return ucho


@jax.jit
def one_coef_3d(a, b, umat, pts):
    ucho = _add_and_cho(umat, b)
    y = ucho.T @ pts.T
    s_U_s = jnp.linalg.vector_norm(y, axis=0) ** 2

    return a * jnp.exp(-s_U_s / 4)


oc3d_vmap = jax.vmap(one_coef_3d, in_axes=[0, 0, None, None])


@jax.jit
def _calc_f_atom(coord, umat, occ, aty, it92, pts):
    f_at = occ * oc3d_vmap(
        it92[aty, :5],
        it92[aty, 5:],
        umat,
        pts,
    ).sum(axis=0)
    phase = jnp.exp(-2 * jnp.pi * 1j * coord @ pts.T)

    return f_at * phase


calc_f = jax.vmap(_calc_f_atom, in_axes=[0, 0, 0, 0, None, None])


@jax.jit
def calc_f_scan(coords, umat, occ, aty, it92, pts, fft_scale):
    @jax.jit
    def one_atom(carry, tree):
        coord, umat, occ, aty = tree
        vals = _calc_f_atom(coord, umat, occ, aty, it92, pts1d)
        addend = vals / fft_scale
        return carry + addend.astype(jnp.complex64), None

    pts1d = pts.reshape(-1, 3)
    f_o, _ = jax.lax.scan(
        one_atom,
        jnp.zeros(len(pts1d), dtype=jnp.complex64),
        (coords, umat, occ, aty),
    )
    return f_o.reshape(*pts.shape[:3])


@jax.jit
def one_coef_re(a, b, umat, pts):
    ucho = _add_and_cho(umat, b)
    y = triangular_solve(ucho, pts.T, left_side=True, lower=True)
    r_U_r = jnp.linalg.vector_norm(y, axis=0) ** 2

    udet = ucho[0, 0] * ucho[1, 1] * ucho[2, 2]

    den = (a * 8 * jnp.pi * jnp.sqrt(jnp.pi) * jnp.exp(-4 * jnp.pi**2 * r_U_r)) / udet

    return den


ocre_vmap = jax.vmap(one_coef_re, in_axes=[0, 0, None, None])


@partial(jax.jit, static_argnames=["rcut"])
def _make_small_grid(coord, mgrid, rcut):
    dist = (mgrid.T - coord).T
    coords = jnp.argmax(dist >= 0, axis=1)

    inds3d = jnp.indices((rcut, rcut, rcut))
    inds1d = jnp.column_stack(
        [inds3d[i].ravel() + coords[i] - rcut // 2 for i in range(3)]
    )

    angpix = mgrid[0, 1] - mgrid[0, 0]
    offset = dist[(0, 1, 2), coords]
    pts1d = jnp.column_stack(
        [angpix * inds3d[i].ravel() + offset[i] - angpix * rcut / 2 for i in range(3)]
    )

    return inds1d, pts1d


@partial(jax.jit, static_argnames=["rcut"])
def _calc_v_atom_sparse(carry, tree, it92, mgrid, rcut):
    coord, umat, occ, aty = tree
    inds1d, pts1d = _make_small_grid(coord, mgrid, rcut)

    v_small = occ * ocre_vmap(
        it92[aty, :5],
        it92[aty, 5:],
        umat,
        pts1d,
    ).sum(axis=0).reshape(rcut, rcut, rcut)

    dim = mgrid.shape[1]
    v_at = sparse.BCOO((v_small.ravel(), inds1d), shape=(dim, dim, dim))

    return carry + v_at, None


@partial(jax.jit, static_argnames=["rcut", "nsamples"])
def calc_v_sparse(coord, umat, occ, aty, it92, rcut, bounds, nsamples):
    mgrid = make_grid(bounds, nsamples)
    wrapped = partial(
        _calc_v_atom_sparse,
        it92=it92,
        mgrid=mgrid,
        rcut=rcut,
    )
    v_mol, _ = jax.lax.scan(
        wrapped,
        jnp.zeros((nsamples, nsamples, nsamples)),
        (coord, umat, occ, aty),
    )

    return v_mol


def make_grid(bounds, nsamples):
    axis = jnp.linspace(bounds[:, 0], bounds[:, 1], nsamples, axis=-1, endpoint=False)
    return axis


def calc_rcut(length, spacing):
    return int(length / (2 * spacing)) * 2


@partial(jax.jit, static_argnames=["nbins"])
def make_bins(data, spacing, smin, smax, nbins):
    axes = (
        jnp.fft.fftfreq(data.shape[0], d=spacing),
        jnp.fft.fftfreq(data.shape[0], d=spacing),
        jnp.fft.rfftfreq(data.shape[0], d=spacing),
    )

    sx, sy, sz = jnp.meshgrid(*axes, indexing="ij")
    s = jnp.sqrt(sx**2 + sy**2 + sz**2)

    bins = jnp.linspace(smin, smax, nbins + 1)
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

    f_o = jnp.fft.rfftn(v_o)
    f_c = jnp.fft.rfftn(v_c)
    prec_D1 = jnp.real(f_o * f_c.conj())
    prec_D2 = jnp.abs(f_c) ** 2

    _, D = jax.lax.scan(calc_D, None, labels)
    _, S = jax.lax.scan(calc_S, None, (labels, D))

    return D, S


@jax.jit
def calc_power(fdata, fbins, labels):
    @jax.jit
    def one_bin(ind):
        msk = (fbins == ind).astype(int)
        return jnp.sum(sqabs * msk)

    sqabs = jnp.abs(fdata) ** 2
    pspec = jax.lax.map(one_bin, labels)
    return pspec


@jax.jit
def one_coef_1d(a, b, bins):
    return a * jnp.exp(-b * bins**2 / 4)


oc1d_vmap = jax.vmap(
    jax.vmap(one_coef_1d, in_axes=[0, 0, None]),
    in_axes=[0, 0, None],
)


@partial(jax.jit, static_argnames=["rcut"])
def _calc_gaussian_atom(coord, umat, mgrid, rcut):
    inds, pts = _make_small_grid(coord, mgrid, rcut)
    gauss = one_coef_re(1, 0, umat, pts)
    dim = mgrid.shape[1]
    gauss_coo = sparse.BCOO((gauss.ravel(), inds), shape=(dim, dim, dim))

    return gauss_coo


@partial(jax.jit, static_argnames=["rcut", "naty"])
def calc_gaussians_fft(coords, umat, aty, mgrid, rcut, naty):
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
def calc_gaussians_direct(coords, umat, occ, aty, pts, sigma_n, naty, fft_scale):
    @jax.jit
    def one_gaussian(carry, tree):
        coord, umat, occ, aty = tree
        vals = one_coef_3d(1, 0, umat, pts1d)
        phase = jnp.exp(-2 * jnp.pi * 1j * coord @ pts1d.T)
        addend = occ * vals * phase / fft_scale

        new = carry.at[aty].add(addend.astype(jnp.complex64))
        return new, None

    pts1d = pts.reshape(-1, 3)
    gauss, _ = jax.lax.scan(
        one_gaussian,
        jnp.zeros((naty, len(pts1d)), dtype=jnp.complex64),
        (coords, umat, occ, aty),
    )
    gauss = gauss.reshape(len(gauss), *pts.shape[:-1])
    gauss /= jnp.sqrt(sigma_n)

    return gauss


@partial(jax.jit, static_argnames=["rcut", "bsize"], donate_argnames=["mpdata"])
def subtract_density(mpdata, atmask, coords, umat, occ, aty, it92, rcut, bounds, bsize):
    excluded = calc_v_sparse(
        coords,
        umat,
        occ * (1 - atmask),
        aty,
        it92,
        rcut,
        bounds,
        bsize,
    )
    mpdata -= excluded
    return mpdata
