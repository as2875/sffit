from functools import partial

import jax
import jax.numpy as jnp
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
    ).sum(axis=0)
    new = carry.at[*inds1d.T].add(v_small)
    return new, None


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


@partial(jax.jit, static_argnames=["dim"])
def make_freq_grid(dim, spacing):
    axes = (
        jnp.fft.fftfreq(dim, d=spacing),
        jnp.fft.fftfreq(dim, d=spacing),
        jnp.fft.rfftfreq(dim, d=spacing),
    )
    sx, sy, sz = jnp.meshgrid(*axes, indexing="ij")
    s = jnp.sqrt(sx**2 + sy**2 + sz**2)
    s_vec = jnp.stack([sx, sy, sz], axis=-1)
    return s, s_vec


@partial(jax.jit, static_argnames=["dim", "nbins"])
def make_bins(dim, spacing, smin, smax, nbins):
    s, s_vec = make_freq_grid(dim, spacing)
    bins = jnp.linspace(smin, smax, nbins + 1)
    bin_cent = 0.5 * (bins[1:] + bins[:-1])
    sdig = jnp.digitize(s, bins) - 1
    return s_vec, sdig, bin_cent


@partial(jax.jit, static_argnames=["dim"])
def make_relion_bins(dim, bsize, spacing):
    @jax.jit
    def first_pass(carry, tree):
        bins, counts = carry
        ind, count = tree
        bins, counts = jax.lax.cond(
            count < 10,
            lambda x, y: (
                jnp.where(x == ind, ind + 1, x),
                counts.at[ind].set(0).at[ind + 1].add(count),
            ),
            lambda x, y: (x, y),
            bins,
            counts,
        )
        return (bins, counts), None

    @jax.jit
    def second_pass(fbins, tree):
        ind, count_this, count_prev = tree
        new = jax.lax.cond(
            count_this / count_prev < 0.5,
            lambda x: jnp.where(x == ind - 1, ind, x),
            lambda x: x,
            fbins,
        )
        return new, None

    @jax.jit
    def renumber(carry, ind_read):
        fbins, ind_set = carry
        msk = fbins == ind_read
        fbins = jnp.where(msk, ind_set, fbins)
        count = jnp.any(msk)
        ind_set = jax.lax.cond(count, lambda x: x + 1, lambda x: x, ind_set)
        return (fbins, ind_set), None

    @jax.jit
    def get_bin_cent(ind):
        masked = jnp.where(bins == ind, s, jnp.nan)
        return (jnp.nanmin(masked), jnp.nanmax(masked))

    cell_size = bsize * spacing
    s, s_vec = make_freq_grid(dim, spacing)
    bins = (cell_size * s + 0.5).astype(int)
    max_ind = dim // 2
    bins = jnp.where(bins > max_ind, max_ind + 1, bins)

    _, counts = jnp.unique_counts(bins, size=max_ind + 1, fill_value=0)
    (bins, counts), _ = jax.lax.scan(
        first_pass, (bins, counts), (jnp.arange(1, max_ind), counts[1:-1])
    )
    bins, _ = jax.lax.scan(
        second_pass,
        bins,
        (jnp.arange(max_ind, 0, -1), counts[:-1][::-1], counts[1:][::-1]),
    )

    bins = bins.at[0, 0, 0].set(-1)
    (bins, _), _ = jax.lax.scan(renumber, (bins, 0), jnp.arange(max_ind + 2))

    smin, smax = jax.lax.map(get_bin_cent, jnp.arange(max_ind + 1))
    bin_cent = 0.5 * (smin + smax)
    cutoff = jnp.argmax(jnp.isnan(bin_cent)) - 1

    return s_vec, bins, bin_cent, cutoff


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
def calc_power(fdata, fbins, labels, sigma_n):
    @jax.jit
    def one_bin(ind):
        msk = (fbins == ind).astype(int)
        return jnp.sum(sqabs * msk)

    sqabs = 2 * jnp.abs(fdata) ** 2 / sigma_n
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
def _calc_gaussian_atom(coord, umat, occ, mgrid, rcut):
    inds, pts = _make_small_grid(coord, mgrid, rcut)
    gauss = occ * one_coef_re(1, 0, umat, pts)
    return gauss, inds


@partial(jax.jit, static_argnames=["rcut", "nsamples", "naty"])
def calc_gaussians_fft(
    coords, umat, occ, aty, D, sigma_n, rcut, bounds, nsamples, naty, blur, pts
):
    @jax.jit
    def categorical_sum(carry, tree):
        coord, umat, occ, atyind = tree
        gauss, inds = _calc_gaussian_atom(coord, umat, occ, mgrid, rcut)
        atyind = jax.lax.broadcast(atyind, gauss.shape)
        new = carry.at[atyind, *inds.T].add(gauss.astype(jnp.float32))
        return new, None

    mgrid = make_grid(bounds, nsamples)
    summed, _ = jax.lax.scan(
        categorical_sum,
        jnp.zeros((naty, nsamples, nsamples, nsamples), dtype=jnp.float32),
        (coords, umat + blur * jnp.identity(3), occ, aty),
    )

    f_c = jnp.fft.rfftn(summed, axes=(1, 2, 3))
    b_corr = jnp.exp(blur * jnp.linalg.norm(pts, axis=-1) ** 2 / 4)
    f_c *= b_corr
    f_c /= jnp.sqrt(sigma_n.astype(jnp.float32)) / D.astype(jnp.float32)

    return f_c


@partial(jax.jit, static_argnames=["naty"])
def calc_gaussians_direct(coords, umat, occ, aty, pts, D, sigma_n, naty, fft_scale):
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
    gauss /= jnp.sqrt(sigma_n.astype(jnp.float32)) / D.astype(jnp.float32)

    return gauss


@partial(jax.jit, static_argnames=["rcut", "bsize"])
def subtract_density(
    mpdata, D, atmask, coords, umat, occ, aty, it92, rcut, bounds, bsize
):
    excluded = calc_v_sparse(
        coords,
        umat,
        occ * (1 - atmask.astype(float)),
        aty,
        it92,
        rcut,
        bounds,
        bsize,
    )
    f_obs = jnp.fft.rfftn(mpdata) - D * jnp.fft.rfftn(excluded)
    return f_obs


@partial(jax.jit, static_argnames=["nsamples"])
def calc_k_b(f_ref, f_scale, nsamples, spacing):
    s2, _ = make_freq_grid(nsamples, spacing)
    s2 **= 2
    f1, f2 = jnp.abs(f_ref), jnp.abs(f_scale)
    msk = ((f1 > 0) & (f2 > 0)).astype(int)

    diff = jnp.log(f2) - jnp.log(f1)
    g = jnp.array([2 * jnp.nansum(diff), -jnp.nansum(diff * s2) / 2])
    H = jnp.array(
        [
            [2 * jnp.count_nonzero(msk), -jnp.sum(msk * s2) / 2],
            [-jnp.sum(msk * s2) / 2, jnp.sum(msk * s2**2 / 8)],
        ]
    )
    soln = -jnp.linalg.solve(H, g)
    k, B = jnp.exp(soln[0]), soln[1]
    return k, B


@jax.jit
def calc_blur(umat, spacing):
    b_iso = jnp.trace(umat, axis1=1, axis2=2) / 3
    b_min = b_iso.min()
    blur = (5 * spacing) ** 2 - b_min
    return blur
