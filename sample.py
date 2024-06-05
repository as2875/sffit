import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.experimental.sparse as sparse
from jax.lax.linalg import cholesky, triangular_solve

import gemmi
import numpy as np
import optax


@jax.jit
def one_coef(a, b, umat, sigma, pts):
    umatb = umat + (b + sigma) * jnp.identity(3)

    ucho = cholesky(umatb, symmetrize_input=False)
    y = triangular_solve(ucho, pts.T, left_side=True)
    r_U_r = jnp.linalg.vector_norm(y, axis=0) ** 2

    udet = ucho[0, 0] * ucho[1, 1] * ucho[2, 2]

    den = (
        a * (4 * jnp.pi) * jnp.sqrt(4 * jnp.pi) * jnp.exp(-4 * jnp.pi**2 * r_U_r)
    ) / udet

    return den


one_coef_vmap = jax.vmap(one_coef, in_axes=[0, 0, None, None, None])


@jax.jit
def _calc_v_atom(coord, umat, it92, aty, weights, sigma, pts):
    dist = pts - coord
    v_small = weights[aty] * one_coef_vmap(
        it92[:5],
        it92[5:],
        umat,
        sigma[aty],
        dist,
    ).sum(axis=0)

    return v_small


calc_v = jax.vmap(_calc_v_atom, in_axes=[0, 0, 0, 0, None, None, None])


def make_grid(bounds, nsamples):
    axis = jnp.linspace(bounds[:, 0], bounds[:, 1], nsamples, axis=-1, endpoint=False)
    return axis


@partial(jax.jit, static_argnames=["rcut"])
def _calc_v_atom_sparse(carry, tree, weights, sigma, mgrid, rcut):
    coord, umat, it92, aty = tree

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

    v_small = weights[aty] * one_coef_vmap(
        it92[:5],
        it92[5:],
        umat,
        sigma[aty],
        pts1d,
    ).sum(axis=0).reshape(rcut, rcut, rcut)

    v_at = sparse.BCOO((v_small.ravel(), inds1d), shape=(dim, dim, dim))

    return carry + v_at, None


def calc_v_sparse(coord, umat, it92, aty, weights, sigma, rcut, bounds, nsamples):
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
        (coord, umat, it92, aty),
    )

    return v_mol


def loglik_fn(params, pts, inds, target, coords, umat, it92, aty):
    weights, sigma, D, sigma_n = (
        jax.nn.softplus(params["weights"]),
        jax.nn.softplus(params["sigma"]),
        jax.nn.softplus(params["D"]),
        jax.nn.softplus(params["sigma_n"]),
    )
    v_mol = calc_v(coords, umat, it92, aty, weights, sigma, pts).sum(axis=0)

    logpdf = stats.norm.logpdf(
        target[inds[:, 0], inds[:, 1], inds[:, 2]] - D * v_mol,
        0,
        sigma_n,
    ).sum()

    return logpdf


def logprior_fn(params, scale_wt, scale_sg):
    weights, sigma = params["weights"], params["sigma"]
    logpdf_wt = stats.norm.logpdf(weights, loc=0, scale=scale_wt)
    logpdf_sg = stats.norm.logpdf(sigma, loc=0, scale=scale_sg)

    return jnp.sum(logpdf_wt) + jnp.sum(logpdf_sg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="input map")
    parser.add_argument("--model", required=True, help="input model")
    parser.add_argument("-om", required=True, help="output map")
    parser.add_argument(
        "--rcut",
        type=float,
        required=True,
        help="maximum radius for evaluation of Gaussians in output map",
    )

    pgroup = parser.add_mutually_exclusive_group()
    pgroup.add_argument("--params", help=".npz file with parameters")
    pgroup.add_argument("-op", help="output .npz with parameters")

    args = parser.parse_args()

    rng_key = jax.random.key(int(time.time()))

    # load observations
    ccp4 = gemmi.read_ccp4_map(args.map)
    mpdata = ccp4.grid.array
    st = gemmi.read_structure(args.model)
    st[0].remove_alternative_conformations()
    atmod = gemmi.expand_ncs_model(st[0], st.ncs, gemmi.HowToNameCopiedChain.Short)

    n_atoms = atmod.count_atom_sites()
    coords = np.empty((n_atoms, 3))
    it92 = np.empty((n_atoms, 10))
    umat = np.empty((n_atoms, 3, 3))
    atyhash = np.empty(n_atoms, dtype=np.uint32)

    # set up atom typing
    ns = gemmi.NeighborSearch(st[0], st.cell, 2).populate(include_h=True)

    for ind, cra in enumerate(atmod.all()):
        coords[ind] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
        it92[ind] = np.concatenate([cra.atom.element.c4322.a, cra.atom.element.c4322.b])
        if cra.atom.aniso.nonzero():
            umat[ind] = 8 * np.pi**2 * np.array(cra.atom.aniso.as_mat33().tolist())
        else:
            umat[ind] = cra.atom.b_iso * np.identity(3)

        envid = (
            sum(
                [
                    hash(mk.element.name)
                    for mk in ns.find_neighbors(cra.atom, min_dist=0.1, max_dist=2.0)
                ]
            )
            % 2**32
        )
        atyhash[ind] = envid

    unq, aty = np.unique(atyhash, return_inverse=True)
    naty = len(unq)

    mpdata, coords, it92, umat, aty = [
        jnp.array(a) for a in (mpdata, coords, it92, umat, aty)
    ]

    assert (
        ccp4.grid.nu == ccp4.grid.nv == ccp4.grid.nw
    ), "Only cubic boxes are supported"
    assert (
        ccp4.grid.spacing[0] == ccp4.grid.spacing[1] == ccp4.grid.spacing[2]
    ), "Only cubic boxes are supported"

    bsize = ccp4.grid.nu
    spacing = ccp4.grid.spacing[0]
    inds3d = jnp.indices((bsize, bsize, bsize))
    inds1d = jnp.column_stack([inds3d[i].ravel() for i in range(3)])

    inds1dr = jax.random.permutation(rng_key, inds1d)
    pts1dr = inds1dr * spacing

    indsbatched = inds1dr.reshape(-1, 16, 3)
    ptsbatched = pts1dr.reshape(-1, 16, 3)

    if not args.params:
        loglik = partial(
            loglik_fn,
            target=mpdata,
            coords=coords,
            umat=umat,
            it92=it92,
            aty=aty,
        )
        logprior = partial(logprior_fn, scale_wt=1.0, scale_sg=1.0)
        logden = jax.jit(
            lambda params, pts, inds: -loglik(params, pts, inds) - logprior(params),
        )

        @jax.jit
        def step(params, opt_state, ptbatch, indbatch):
            loss, grads = jax.value_and_grad(logden)(params, ptbatch, indbatch)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        params = {
            "weights": jnp.zeros(naty),
            "sigma": jnp.zeros(naty),
            "D": 0.0,
            "sigma_n": 0.0,
        }
        optim = optax.adam(learning_rate=0.001)
        opt_state = optim.init(params)

        for i, (ptbatch, indbatch) in enumerate(zip(ptsbatched, indsbatched)):
            params, opt_state, loss = step(params, opt_state, ptbatch, indbatch)

            if i % 1000 == 0:
                print(f"step {i}, loss: {loss}")

        jnp.savez(args.op, **params)

    else:
        params = jnp.load(args.params)

    wt_post, sg_post = (
        jax.nn.softplus(params["weights"]),
        jax.nn.softplus(params["sigma"]),
    )

    bounds = jnp.array([[0, bsize * spacing] for i in range(3)])
    rcut = int(args.rcut / (2 * spacing)) * 2
    v_approx = calc_v_sparse(
        coords,
        umat,
        it92,
        aty,
        wt_post,
        sg_post,
        rcut,
        bounds,
        bsize,
    ).reshape(bsize, bsize, bsize)

    result_map = gemmi.Ccp4Map()
    result_map.grid = gemmi.FloatGrid(np.array(v_approx, dtype=np.float32))
    result_map.grid.copy_metadata_from(ccp4.grid)
    result_map.update_ccp4_header()
    result_map.write_ccp4_map(args.om)
