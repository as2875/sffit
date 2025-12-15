from functools import partial

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from . import dencalc, util, radn


@jax.jit
def make_coef_mats(coef_a1, coef_a2, coef_b1, coef_b2):
    return (
        jnp.multiply.outer(coef_a1, coef_a2),
        jnp.add.outer(coef_b1, coef_b2),
    )


def get_contacts(st, cutoff):
    lookup = {x.atom: i for i, x in enumerate(st[0].all())}
    ns = gemmi.NeighborSearch(st, cutoff).populate()
    cs = gemmi.ContactSearch(cutoff)
    cs.ignore = gemmi.ContactSearch.Ignore.Nothing
    contacts = [
        (lookup[c.partner1.atom], lookup[c.partner2.atom]) for c in cs.find_contacts(ns)
    ]
    return jnp.array(contacts)


def clear_altlocs(st):
    stcl = st.clone()
    for cra in stcl[0].all():
        cra.atom.altloc = "\0"
    return stcl


@jax.jit
def scale_map(f_obs, D, sigma_n, fbins):
    msk = radn.mask_extrema(jnp.ones_like(fbins), fbins)
    return msk * f_obs * D[fbins] / sigma_n[fbins]


@jax.jit
def approx_umat_iso(umat):
    b_iso = jnp.trace(umat, axis1=1, axis2=2) / 3
    umat_iso = jnp.broadcast_to(b_iso, (3, 3, len(umat))).T * jnp.identity(3)
    return b_iso, umat_iso


@jax.jit
def calc_occ_lhs(contacts, coords, it92, b_iso, aty, freqs, weights):
    @jax.jit
    def one_elem(pair):
        i1, i2 = pair
        adp_sum = b_iso[i1] + b_iso[i2]
        dist = jnp.linalg.norm(coords[i1] - coords[i2])

        coef_a_mat, coef_b_mat = make_coef_mats(
            it92[aty[i1], :5],
            it92[aty[i2], :5],
            it92[aty[i1], 5:],
            it92[aty[i2], 5:],
        )
        exponent = jnp.broadcast_to(freqs**2, (5, 5, len(freqs))).T * (
            adp_sum + coef_b_mat
        )
        overlap = jnp.sum(coef_a_mat * jnp.exp(-exponent / 4), axis=(1, 2))
        integrand = (
            4 * jnp.pi * overlap * weights * freqs**2 * jnp.sinc(2 * freqs * dist)
        )
        return jnp.trapezoid(integrand, dx=dx, axis=-1)

    natoms = len(coords)
    dx = jnp.diff(freqs).mean()
    values = jax.lax.map(one_elem, contacts)

    diaginds = jnp.column_stack([jnp.arange(natoms), jnp.arange(natoms)])
    diag = jax.lax.map(one_elem, diaginds)

    values = jnp.concatenate([values, values, diag])
    matinds = jnp.concatenate([contacts, contacts[..., ::-1], diaginds], axis=0)
    return (values, matinds.T)


@partial(jax.jit, static_argnames=["rcut"])
def calc_occ_rhs(coords, it92, umat, aty, mpdata, mgrid, rcut):
    @jax.jit
    def one_atom(coord, umat, aty):
        inds1d, pts1d = dencalc._make_small_grid(coord, mgrid, rcut)
        den_calc = dencalc.ocre_vmap(
            it92[aty, :5],
            it92[aty, 5:],
            umat,
            pts1d,
        ).sum(axis=0)

        den_obs = mpdata[*inds1d.T]
        return jnp.sum(den_obs * den_calc) / 8

    rhs = jax.vmap(one_atom)(coords, umat, aty)
    return rhs


@partial(jax.jit, static_argnames=["nbsize"])
def calc_occ_reg(contacts, coords, nbsize):
    @jax.jit
    def get_neighbors(carry, pair):
        table, counter = carry
        i1, i2 = pair
        table = table.at[i1, counter[i1]].set(i2)
        counter = counter.at[i1].add(1)
        return (table, counter), None

    def cho_row(tree):
        ind, nblist = tree
        coordset = coords[nblist]
        distmat = jnp.linalg.norm(coordset[:, None, :] - coordset[None, :, :], axis=-1)
        valid = nblist >= ind
        msk = jnp.logical_and.outer(valid, valid)
        kern = jnp.exp(-(distmat) / 4)
        kern = jnp.where(msk, kern, 0)

        soln, *_ = jnp.linalg.lstsq(kern, jnp.zeros(nbsize).at[0].set(1.0))
        soln = jnp.where(valid, soln, 0)
        soln /= jnp.sqrt(soln[0])
        matinds = jnp.column_stack([nblist, jnp.repeat(ind, nbsize)])
        return soln, matinds

    natoms = len(coords)
    (nbtable, _), _ = jax.lax.scan(
        get_neighbors,
        (jnp.full((natoms, nbsize), -1, dtype=int), jnp.zeros(natoms, dtype=int)),
        jnp.concatenate(
            [jnp.column_stack([jnp.arange(natoms), jnp.arange(natoms)]), contacts]
        ),
    )
    values, indices = jax.lax.map(cho_row, (jnp.arange(natoms), nbtable))
    indices = jnp.where(indices == -1, 0, indices)
    return (values.ravel(), indices.reshape(-1, 2).T)


def pair_to_numpy(pair):
    return (np.asarray(pair[0]), np.asarray(pair[1]))


def solve_linear_system(mat, vec, regcho):
    natoms = len(vec)
    regcho = scipy.sparse.csr_array(pair_to_numpy(regcho), (natoms, natoms))
    mat = scipy.sparse.csr_array(pair_to_numpy(mat), (natoms, natoms))
    weights = 1e1 * mat.diagonal()

    lhs = scipy.sparse.vstack([mat, regcho.T * weights])
    rhs = np.concatenate([vec, regcho.T @ weights])
    result = scipy.optimize.lsq_linear(lhs, rhs, verbose=2)

    return np.clip(result.x, 0, 1)


def assign_occ(
    st,
    f_obs,
    D,
    sigma_n,
    freqs,
    fbins,
    mgrid,
    spacing,
    fft_scale,
    basisco=4.0,
    maxnbsize=30,
):
    coords, it92, umat, _, aty, *_ = util.from_gemmi(
        st, nochangeh=True, addmissing=False
    )
    b_iso, umat_iso = approx_umat_iso(umat)
    rcut = dencalc.calc_rcut(20, spacing)

    f_obs = scale_map(f_obs, D, sigma_n, fbins)
    weights = D**2 / sigma_n

    lhs = calc_occ_lhs(
        get_contacts(clear_altlocs(st), basisco),
        coords,
        it92,
        b_iso,
        aty,
        freqs,
        weights,
    )
    rhs = calc_occ_rhs(
        coords, it92, umat_iso, aty, jnp.fft.irfftn(f_obs) / fft_scale, mgrid, rcut
    )
    regmat = calc_occ_reg(get_contacts(st, basisco), coords, maxnbsize)
    soln = solve_linear_system(lhs, rhs, regmat)

    for i, cra in enumerate(st[0].all()):
        cra.atom.occ = soln[i]

    return st
