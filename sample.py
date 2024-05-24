import argparse
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.experimental.sparse as sparse
import blackjax

import gemmi
import numpy as np

from datetime import date

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))


def make_grid(bounds, nsamples):
    axis = jnp.linspace(bounds[:, 0], bounds[:, 1], nsamples, axis=-1, endpoint=False)
    return axis


@jax.jit
def one_coef(a, b, umat, sigma, pts):
    umatb = umat + b * jnp.identity(3)
    uinv = jnp.linalg.inv(umatb)
    udet = jnp.linalg.det(umatb)

    r_U_r = jnp.einsum(
        "ki,ij,jk->k",
        pts,
        uinv,
        pts.T,
    )

    den = (
        a * (4 * jnp.pi) * jnp.sqrt(4 * np.pi) * jnp.exp(-4 * jnp.pi**2 * r_U_r)
    ) / jnp.sqrt(udet)

    return den


one_coef_vmap = jax.vmap(one_coef, in_axes=[0, 0, None, None, None])


@partial(jax.jit, static_argnames=["rcut"])
def _calc_v_atom(coord, umat, it92, weight, sigma, mgrid, rcut):
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

    v_small = weight * one_coef_vmap(
        it92[:5],
        it92[5:],
        umat,
        sigma,
        pts1d,
    ).sum(axis=0).reshape(rcut, rcut, rcut)

    v_at = sparse.BCOO((v_small.ravel(), inds1d), shape=(dim, dim, dim))

    return v_at


calc_v = jax.vmap(_calc_v_atom, in_axes=[0, 0, 0, 0, 0, None, None])


def loglik_fn(params, data, coords, umat, it92, sigma_n, mgrid, rcut):
    weights, sigma = jnp.exp(params["weights"]), jnp.exp(params["sigma"])
    v_mol = calc_v(coords, umat, it92, weights, sigma, mgrid, rcut).sum(axis=0)

    logpdf = stats.norm.logpdf(data - v_mol.todense(), 0, sigma_n)

    return jnp.sum(logpdf)


def logprior_fn(params, scale_wt, scale_sg):
    weights, sigma = params["weights"], params["sigma"]
    logpdf_wt = stats.norm.logpdf(weights, loc=0, scale=scale_wt)
    logpdf_sg = stats.norm.logpdf(sigma, loc=0, scale=scale_sg)

    return jnp.sum(logpdf_wt) + jnp.sum(logpdf_sg)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="input map")
    parser.add_argument("--model", required=True, help="input model")
    parser.add_argument("-om", required=True, help="output map")

    pgroup = parser.add_mutually_exclusive_group()
    pgroup.add_argument("--params", help=".npz file with parameters")
    pgroup.add_argument("-op", help="output .npz with parameters")

    args = parser.parse_args()

    # load observations
    ccp4 = gemmi.read_ccp4_map("/lmb/home/ashtyrov/Downloads/emd_17962.map")
    mpdata = ccp4.grid.array
    st = gemmi.read_structure("/lmb/home/ashtyrov/Downloads/pdb8pvd_h.ent")

    n_atoms = st[0].count_atom_sites()
    coords = np.empty((n_atoms, 3))
    it92 = np.empty((n_atoms, 10))
    umat = np.empty((n_atoms, 3, 3))

    for ind, cra in enumerate(st[0].all()):
        coords[ind] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
        it92[ind] = np.concatenate([cra.atom.element.c4322.a, cra.atom.element.c4322.b])
        if cra.atom.aniso.nonzero():
            umat[ind] = np.array(cra.atom.aniso.as_mat33().tolist())
        else:
            B = cra.atom.b_iso
            umat[ind] = np.diag([B, B, B])

    assert (
        ccp4.grid.nu == ccp4.grid.nv == ccp4.grid.nw
    ), "Only cubic boxes are supported"
    assert (
        ccp4.grid.spacing[0] == ccp4.grid.spacing[1] == ccp4.grid.spacing[2]
    ), "Only cubic boxes are supported"

    bsize = ccp4.grid.nu
    spacing = ccp4.grid.spacing[0]
    bounds = jnp.array([[0, bsize * spacing] for i in range(3)])
    rcut = int(10 / spacing)
    mgrid = make_grid(bounds, bsize)

    if not args.params:
        loglik = partial(
            loglik_fn,
            data=mpdata,
            coords=coords,
            umat=umat,
            it92=it92,
            sigma_n=1.0,
            mgrid=mgrid,
            rcut=rcut,
        )
        logprior = partial(logprior_fn, scale_wt=1.0, scale_sg=1.0)
        logden = jax.jit(lambda x: loglik(x) + logprior(x))

        rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
        init_params = {"weights": jnp.zeros(n_atoms), "sigma": jnp.zeros(n_atoms)}

        warmup = blackjax.window_adaptation(blackjax.nuts, logden)
        (init_states, tuned_params), _ = warmup.run(warmup_key, init_params, 100)
        kernel = blackjax.nuts(logden, **tuned_params).step
        states = inference_loop(sample_key, jax.jit(kernel), init_states, 1000)

        mcmc_samples = states.position
        jnp.savez(args.op, **mcmc_samples)
    else:
        mcmc_samples = jnp.load(args.params)

    wt_post = jnp.exp(mcmc_samples["weights"]).mean(axis=0)
    sg_post = jnp.exp(mcmc_samples["sigma"]).mean(axis=0)
    v_approx = calc_v(coords, it92, wt_post, sg_post, mgrid)

    result_map = gemmi.Ccp4Map()
    result_map.grid = gemmi.FloatGrid(np.array(v_approx, dtype=np.float32))
    result_map.grid.copy_metadata_from(ccp4.grid)
    result_map.update_ccp4_header()
    result_map.write_ccp4_map(args.om)
