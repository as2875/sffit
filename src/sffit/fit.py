import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import gemmi

from . import dencalc
from . import sampler
from . import util


def loglik_fn(
    params,
    batch,
    target,
    fbins,
    nbins,
    coords,
    umat,
    occ,
    it92,
    aty,
    atycounts,
    D,
    sigma_n,
):
    weights, sigma = (
        params["weights"],
        params["sigma"] ** 2,
    )
    pts, inds = batch[0], batch[1].astype(int)

    f_o = target[inds[:, 0], inds[:, 1], inds[:, 2]]
    f_c = dencalc.calc_f(coords, umat, occ, it92, aty, weights, sigma, pts).sum(axis=0)

    D_s = D[inds[:, 0], inds[:, 1], inds[:, 2]]
    sg_n_s = sigma_n[inds[:, 0], inds[:, 1], inds[:, 2]]

    N, n = target.size, pts.shape[0]
    logpdf = -(
        jnp.log(jnp.pi) + jnp.log(sg_n_s) + jnp.abs(f_o - D_s * f_c) ** 2 / sg_n_s
    ) * (N / n)
    logpdf = jnp.where(
        sg_n_s < 1e-6,
        jnp.nan,
        logpdf,
    )

    return jnp.nanmean(logpdf)


def logprior_fn(params, atycounts):
    weights, sigma = (
        params["weights"],
        params["sigma"],
    )
    logpdf_wt = stats.norm.logpdf(weights, loc=1.0, scale=1.0)
    logpdf_sg = stats.norm.logpdf(sigma, loc=0.0, scale=1.0)

    return jnp.sum(logpdf_wt) + jnp.sum(logpdf_sg)


@jax.jit
def scheduler(k):
    return 1e-5 * k ** (-0.33)


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
            (itnum - 1) % 100 == 0,
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


def parse_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--map", required=True, help="input map")
    parser.add_argument("--model", required=True, help="input model")
    parser.add_argument("--mask", required=True, help="input mask")
    parser.add_argument("-im", help="initial calculated map")
    parser.add_argument("-om", help="final calculated map")

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
    parser.add_argument("-d", type=float, required=True, help="maximum resolution")
    parser.add_argument(
        "--nbins",
        type=int,
        default=100,
        help="number of frequency bins",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=50_000,
        help="number of MCMC samples",
    )
    parser.add_argument(
        "--nwarm",
        type=int,
        default=10_000,
        help="number of warm-up steps",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    rng_key = jax.random.key(int(time.time()))

    # load observations
    print("loading data")

    ccp4 = gemmi.read_ccp4_map(args.map)
    mpdata = ccp4.grid.array
    msk = gemmi.read_ccp4_map(args.mask)
    mskdata = msk.grid.array

    st = gemmi.read_structure(args.model)
    st_aty = gemmi.read_structure(args.model)
    coords, it92, umat, occ, aty, atycounts, atnames, atydesc, unq_ind = (
        util.from_gemmi(st, st_aty)
    )
    naty = len(atycounts)

    print(f"{naty} atom types identified")

    mpdata, coords, it92, umat, occ, aty, atycounts = [
        jnp.array(a) for a in (mpdata, coords, it92, umat, occ, aty, atycounts)
    ]
    f_obs = jnp.fft.fftn(mpdata * mskdata)

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

    # estimation of D and S
    if args.im:
        v_iam = dencalc.calc_v_sparse(
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
        util.write_map(v_iam, ccp4, args.im)

    if not args.params:
        init_params = {
            "weights": jnp.ones(naty),
            "sigma": jnp.zeros(naty),
        }

        # initialise frequency grid
        freqs, fbins, bin_edges = dencalc.make_bins(
            mpdata, spacing, 1 / args.d, args.nbins
        )
        bin_cent = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        inds1d = jnp.argwhere(fbins < args.nbins)
        inds1dr = jax.random.choice(
            rng_key, inds1d, axis=0, shape=((args.nsamples) * 512,)
        )
        pts1dr = freqs[inds1dr[:, 0], inds1dr[:, 1], inds1dr[:, 2]]

        indsbatched = inds1dr.reshape(-1, 512, 3)
        ptsbatched = pts1dr.reshape(-1, 512, 3)
        batched = jnp.stack([ptsbatched, indsbatched], axis=1)

        if args.noml:
            D, sigma_n = jnp.ones(args.nbins), jnp.ones(args.nbins)
        else:
            D, sigma_n = dencalc.calc_ml_params(
                mpdata, v_iam, fbins, jnp.arange(args.nbins)
            )

        D_gr, sg_n_gr = D[fbins], sigma_n[fbins]

        loglik = partial(
            loglik_fn,
            target=f_obs,
            fbins=fbins,
            nbins=args.nbins,
            coords=coords,
            umat=umat,
            occ=occ,
            it92=it92,
            aty=aty,
            atycounts=atycounts,
            D=D_gr,
            sigma_n=sg_n_gr,
        )
        logprior = partial(
            logprior_fn,
            atycounts=atycounts,
        )
        logden = jax.jit(
            lambda params, batch: loglik(params, batch) + logprior(params),
        )
        grad_fn = jax.grad(logden)
        ll_grad_vmap = jax.vmap(jax.grad(loglik), in_axes=[None, 0])
        lp_grad = jax.grad(logprior)
        grad_vmap = jax.jit(
            lambda params, batch: jax.tree.map(
                jnp.add,
                ll_grad_vmap(params, batch),
                lp_grad(params),
            ),
        )

        prec_fn = partial(
            dencalc.calc_hess,
            coords=coords,
            umat=umat,
            occ=occ,
            it92=it92,
            aty=aty,
            naty=naty,
            sigma_n=sigma_n,
            bins=bin_cent,
        )

        # sample with SGLD
        print("sampling")
        rng_key, init_key, sample_key = jax.random.split(rng_key, 3)
        sgld = sampler.prec_sgld(grad_fn, prec_fn)
        params = inference_loop(
            sample_key,
            jax.jit(sgld.step),
            batched,
            init_params,
            grad_vmap,
        )

        print("saving parameters")
        params["steps"] = scheduler(jnp.arange(args.nsamples) + 1)
        jnp.savez(
            args.op,
            aty=atydesc[unq_ind],
            atyex=atnames[unq_ind],
            atycounts=atycounts,
            **params,
        )

    else:
        params = jnp.load(args.params)

    if args.om:
        print("writing output map")
        weights, sigma, step_size = (
            params["weights"][args.nwarm :],
            params["sigma"][args.nwarm :] ** 2,
            params["steps"][args.nwarm :],
        )
        wt_post, sg_post = (
            jnp.average(weights, axis=0, weights=step_size),
            jnp.average(sigma, axis=0, weights=step_size),
        )

        v_approx = dencalc.calc_v_sparse(
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

        util.write_map(v_approx, ccp4, args.om)


if __name__ == "__main__":
    main()
