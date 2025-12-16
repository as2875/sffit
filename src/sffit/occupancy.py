from functools import partial

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from . import util, spherical


@jax.jit
def make_coef_mats(coef_a1, coef_a2, coef_b1, coef_b2):
    return (
        jnp.multiply.outer(coef_a1, coef_a2),
        jnp.add.outer(coef_b1, coef_b2),
    )


def get_contacts(st, cutoff):
    lookup = {x.atom: i for i, x in enumerate(st[0].all())}
    ns = gemmi.NeighborSearch(st, cutoff).populate(include_h=False)
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


@jax.jit
def calc_occ_rhs(
    coords, it92, b_iso, aty, weights, f_obs, fbins, freqs, labels, freqel
):
    @jax.jit
    def one_atom(coord, umat, aty):
        sf1d = jax.vmap(lambda a, b: a * jnp.exp(-b * freqs**2 / 4))(
            it92[aty, :5],
            it92[aty, 5:],
        ).sum(axis=0)
        integrand = weights * sf1d * prec.conj() * freqs**2 * freqel
        return jnp.trapezoid(integrand, dx=dx).real

    dx = jnp.diff(freqs).mean()
    prec = jax.lax.map(partial(spherical._mask_inner, inner=f_obs, fbins=fbins), labels)
    rhs = jax.vmap(one_atom)(coords, b_iso, aty)
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
    weights = 2e1 * mat.diagonal()

    lhs = scipy.sparse.vstack([mat, regcho.T * weights])
    rhs = np.concatenate([vec, regcho.T @ weights])
    result = scipy.optimize.lsq_linear(lhs, rhs, verbose=2)

    return np.clip(result.x, 0, 1)


def assign_occ(
    st,
    contacts,
    f_obs,
    D,
    sigma_n,
    freqs,
    fbins,
    labels,
    spacing,
    nsamples,
    monlib,
    maxnbsize=30,
):
    st_noh = st.clone()
    st_noh.remove_hydrogens()

    coords, it92, umat, _, aty, *_ = util.from_gemmi(
        st_noh, nochangeh=True, addmissing=False
    )
    b_iso = jnp.trace(umat, axis1=1, axis2=2) / 3

    freqel = 1 / (spacing * nsamples) ** 3

    lhs = calc_occ_lhs(
        contacts[0],
        coords,
        it92,
        b_iso,
        aty,
        freqs,
        D**2 / sigma_n,
    )
    rhs = calc_occ_rhs(
        coords, it92, b_iso, aty, D / sigma_n, f_obs, fbins, freqs, labels, freqel
    )
    regmat = calc_occ_reg(contacts[1], coords, maxnbsize)
    soln = solve_linear_system(lhs, rhs, regmat)

    sel = gemmi.Selection("[!H]")
    i = 0
    for model in sel.models(st):
        for chain in sel.chains(model):
            for residue in sel.residues(chain):
                for atom in sel.atoms(residue):
                    atom.occ = soln[i]
                    i += 1

    topo = gemmi.prepare_topology(
        st,
        monlib,
        h_change=gemmi.HydrogenChange.NoChange,
    )
    for bond in topo.bonds:
        for i in [True, False]:
            if bond.atoms[i].element.atomic_number == 1:
                bond.atoms[i].occ = bond.atoms[not i].occ
                break

    return st
