import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import jax.experimental.sparse as sparse
from jax.lax.linalg import cholesky, triangular_solve

import blackjax
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
def _calc_v_atom(coord, umat, occ, it92, aty, weights, sigma, pts):
    dist = pts - coord
    v_small = (
        occ
        * weights[aty]
        * one_coef_vmap(
            it92[:5],
            it92[5:],
            umat,
            sigma[aty],
            dist,
        ).sum(axis=0)
    )

    return v_small


calc_v = jax.vmap(_calc_v_atom, in_axes=[0, 0, 0, 0, 0, None, None, None])


def make_grid(bounds, nsamples):
    axis = jnp.linspace(bounds[:, 0], bounds[:, 1], nsamples, axis=-1, endpoint=False)
    return axis


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
        * one_coef_vmap(
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


def loglik_fn(
    params, pts, inds, target, fbins, nbins, coords, umat, occ, it92, aty, D, sigma_n
):
    weights, sigma = (
        jnp.exp(params["weights"]),
        jnp.exp(params["sigma"]),
    )
    v_mol = calc_v(coords, umat, occ, it92, aty, weights, sigma, pts).sum(axis=0)
    v_sparse = sparse.BCOO((v_mol, inds), shape=target.shape).todense()
    target_sparse = sparse.BCOO(
        (target[inds[:, 0], inds[:, 1], inds[:, 2]], inds),
        shape=target.shape,
    ).todense()

    f_c = jnp.fft.fftn(v_sparse)
    f_o = jnp.fft.fftn(target_sparse)

    N, n = target.size, v_mol.size
    logpdf = -(
        jnp.log(jnp.pi)
        + jnp.log(sigma_n)
        + (N / n) * jnp.abs(f_o - D * f_c) ** 2 / sigma_n
    )

    logpdf = jnp.where(
        (fbins == nbins) | (sigma_n < 1e-6),
        0,
        logpdf,
    )

    return logpdf.mean()


def logprior(params):
    weights, sigma = params["weights"], params["sigma"]
    logpdf_wt = stats.norm.logpdf(weights, loc=0.0, scale=1.0)
    logpdf_sg = stats.norm.logpdf(sigma, loc=0.0, scale=1.0)

    return jnp.sum(logpdf_wt) + jnp.sum(logpdf_sg)


@partial(jax.jit, static_argnames=["nbins"])
def make_bins(data, spacing, nbins):
    axis = jnp.fft.fftfreq(data.shape[0], d=spacing)
    sx, sy, sz = jnp.meshgrid(axis, axis, axis)
    s = jnp.sqrt(sx**2 + sy**2 + sz**2)

    nyq = 1 / (2 * spacing)
    bins = jnp.linspace(0, nyq, nbins + 1)
    sdig = jnp.digitize(s, bins)

    return sdig


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
def scheduler(k):
    return 1e-6 * k ** (-0.33)


def inference_loop(rng_key, kernel, batches, initial_state, grad_vmap):
    @jax.jit
    def calc_lmax(state, batch, step_size):
        grads = grad_vmap(state, batch[..., None, :])
        grads_arr = jnp.concatenate([grads["weights"].T, grads["sigma"].T])
        grads_cov = jnp.cov(grads_arr)
        l_max = step_size * jnp.linalg.eigvalsh(grads_cov).max()

        return l_max

    @jax.jit
    def one_step(state, tree):
        rng_key, batch, step_size, itnum = tree
        state = kernel(rng_key, state, batch, step_size)
        l_max = jax.lax.cond(
            (itnum - 1) % 1000 == 0,
            calc_lmax,
            lambda *_: jnp.nan,
            state,
            batch,
            step_size,
        )
        acc = dict(statistic=l_max, **state)

        return state, acc

    num_samples = batches.shape[0]
    counter = jnp.arange(num_samples) + 1
    step_size = scheduler(counter)

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(
        one_step,
        initial_state,
        (keys, batches, step_size, counter),
    )

    return states


def opt_loop(optim, opt_fn, batches, init_params):
    @jax.jit
    def opt_step(carry, batch):
        params, opt_state = carry
        loss, grads = opt_fn(params, batch)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return (params, opt_state), loss

    opt_state = optim.init(init_params)
    (params, _), _ = jax.lax.scan(
        opt_step,
        (init_params, opt_state),
        batches,
    )

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--map", required=True, help="input map")
    parser.add_argument("--model", required=True, help="input model")
    parser.add_argument("-om", required=True, help="output map")

    pgroup = parser.add_mutually_exclusive_group(required=True)
    pgroup.add_argument("--params", help=".npz file with parameters")
    pgroup.add_argument("-op", help="output .npz with parameters")

    # algorithm parameters
    parser.add_argument("--noml", action="store_true", help="do not estimate D, S")
    parser.add_argument(
        "--rcut",
        type=float,
        required=True,
        help="maximum radius for evaluation of Gaussians in output map",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=100,
        help="number of frequency bins",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=50000,
        help="number of MCMC samples",
    )
    parser.add_argument(
        "--ninit",
        type=int,
        default=50000,
        help="number of steps in the initial optimisation",
    )

    args = parser.parse_args()

    rng_key = jax.random.key(int(time.time()))

    # load observations
    print("loading data")
    ccp4 = gemmi.read_ccp4_map(args.map)
    mpdata = ccp4.grid.array
    st = gemmi.read_structure(args.model)
    atmod = gemmi.expand_ncs_model(st[0], st.ncs, gemmi.HowToNameCopiedChain.Short)

    n_atoms = atmod.count_atom_sites()
    coords = np.empty((n_atoms, 3))
    it92 = np.empty((n_atoms, 10))
    umat = np.empty((n_atoms, 3, 3))
    occ = np.empty(n_atoms)
    atyhash = np.empty(n_atoms, dtype=int)
    atydesc = np.zeros((n_atoms, 10), dtype=int)

    # set up atom typing
    ns = gemmi.NeighborSearch(st[0], st.cell, 2).populate(include_h=True)

    for ind, cra in enumerate(atmod.all()):
        coords[ind] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
        it92[ind] = np.concatenate([cra.atom.element.c4322.a, cra.atom.element.c4322.b])
        occ[ind] = cra.atom.occ

        if cra.atom.aniso.nonzero():
            umat[ind] = 8 * np.pi**2 * np.array(cra.atom.aniso.as_mat33().tolist())
        else:
            umat[ind] = cra.atom.b_iso * np.identity(3)

        neighbours = ns.find_neighbors(cra.atom, min_dist=0.1, max_dist=2.0)
        envid = [cra.atom.element.name] + [
            mk.element.name
            for mk in neighbours
        ]
        atyhash[ind] = hash(frozenset(envid))

        envdesc = set([mk.element.atomic_number for mk in neighbours])
        atydesc[ind, 0] = cra.atom.element.atomic_number
        atydesc[ind, 1 : 1 + len(envdesc)] = sorted(envdesc)

    unq, unq_ind, aty = np.unique(atyhash, return_index=True, return_inverse=True)
    naty = len(unq)

    mpdata, coords, it92, umat, occ, aty = [
        jnp.array(a) for a in (mpdata, coords, it92, umat, occ, aty)
    ]

    assert (
        ccp4.grid.nu == ccp4.grid.nv == ccp4.grid.nw
    ), "Only cubic boxes are supported"
    assert (
        ccp4.grid.spacing[0] == ccp4.grid.spacing[1] == ccp4.grid.spacing[2]
    ), "Only cubic boxes are supported"

    bsize = ccp4.grid.nu
    spacing = ccp4.grid.spacing[0]
    bounds = jnp.array([[0, bsize * spacing] for i in range(3)])
    rcut = int(args.rcut / (2 * spacing)) * 2
    fbins = make_bins(mpdata, spacing, args.nbins) - 1

    # estimation of D and S
    v_iam = calc_v_sparse(
        coords,
        umat,
        occ,
        it92,
        aty,
        jnp.ones(naty),
        jnp.zeros(naty),
        rcut,
        bounds,
        bsize,
    )

    if args.noml:
        D, sigma_n = jnp.ones(args.nbins), jnp.ones(args.nbins)
    else:
        D, sigma_n = calc_ml_params(mpdata, v_iam, fbins, jnp.arange(args.nbins))

    D_gr, sg_n_gr = D[fbins], sigma_n[fbins]

    inds3d = jnp.indices((bsize, bsize, bsize))
    inds1d = jnp.column_stack([inds3d[i].ravel() for i in range(3)])

    inds1dr = jax.random.choice(
        rng_key, inds1d, axis=0, shape=((args.ninit + args.nsamples) * 64,)
    )
    pts1dr = inds1dr * spacing

    indsbatched = inds1dr.reshape(-1, 64, 3)
    ptsbatched = pts1dr.reshape(-1, 64, 3)
    batched = jnp.stack([ptsbatched, indsbatched], axis=1)

    if not args.params:
        loglik = partial(
            loglik_fn,
            target=mpdata,
            fbins=fbins,
            nbins=args.nbins,
            coords=coords,
            umat=umat,
            occ=occ,
            it92=it92,
            aty=aty,
            D=D_gr,
            sigma_n=sg_n_gr,
        )
        logden = jax.jit(
            lambda params, batch: loglik(params, batch[0], batch[1].astype(int))
            + logprior(params),
        )
        grad_fn = jax.grad(logden)
        ll_grad_vmap = jax.vmap(jax.grad(loglik), in_axes=[None, 0, 0])
        lp_grad = jax.grad(logprior)
        grad_vmap = jax.jit(
            lambda params, batch: jax.tree_util.tree_map(
                jnp.add,
                ll_grad_vmap(params, batch[0], batch[1].astype(int)),
                lp_grad(params),
            ),
        )

        rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
        init_params = {
            "weights": jnp.zeros(naty),
            "sigma": jnp.zeros(naty),
        }

        print("finding starting parameters")
        optim = optax.adam(learning_rate=1e-3)
        opt_fn = jax.value_and_grad(lambda params, batch: -logden(params, batch))
        init_params = opt_loop(optim, opt_fn, batched[: args.ninit], init_params)

        # sample with SGLD
        print("sampling")
        sgld = blackjax.sgld(grad_fn)
        params = inference_loop(
            sample_key,
            jax.jit(sgld.step),
            batched[args.ninit :],
            init_params,
            grad_vmap,
        )

        print("saving parameters")
        params["steps"] = scheduler(jnp.arange(args.nsamples) + 1)
        jnp.savez(args.op, aty=atydesc[unq_ind], **params)

    else:
        params = jnp.load(args.params)

    print("writing output map")
    weights, sigma, step_size = (
        jnp.exp(params["weights"]),
        jnp.exp(params["sigma"]),
        params["steps"],
    )
    wt_post, sg_post = (
        jnp.average(weights, axis=0, weights=step_size),
        jnp.average(sigma, axis=0, weights=step_size),
    )

    v_approx = calc_v_sparse(
        coords,
        umat,
        occ,
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
