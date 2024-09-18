import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import gemmi

from . import dencalc
from . import util


def main():
    parser = argparse.ArgumentParser(
        description="utilities to fit scattering factors to cryo-EM SPA maps"
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

    parser_sample = subparsers.add_parser(
        "sample", description="sample parameters using MCMC"
    )
    parser_sample.add_argument("-im", metavar="FILE", help="initial calculated map")
    parser_sample.add_argument("-om", metavar="FILE", help="final calculated map")
    parser_sample.add_argument(
        "--nsamples",
        metavar="INT",
        type=int,
        required=True,
        help="number of MCMC samples",
    )
    parser_sample.add_argument(
        "--nwarm",
        metavar="INT",
        type=int,
        required=True,
        help="number of warm-up steps",
    )
    pgroup = parser_sample.add_mutually_exclusive_group(required=True)
    pgroup.add_argument("--params", metavar="FILE", help=".npz file with parameters")
    pgroup.add_argument("-op", metavar="FILE", help="output .npz with parameters")
    parser_sample.set_defaults(func=do_sample)

    parser_ml = subparsers.add_parser(
        "ml", description="estimate scattering factors using least squares and FFT"
    )
    parser_ml.add_argument(
        "--lmax",
        metavar="INT",
        type=int,
        default=128,
        help="order of spherical harmonics used in calculations",
    )
    parser_ml.add_argument(
        "-o",
        metavar="FILE",
        required=True,
        help="output .npz with parameters",
    )
    parser_ml.set_defaults(func=do_ml)

    for sp in (parser_sample, parser_ml):
        sp.add_argument("--map", metavar="FILE", required=True, help="input map")
        sp.add_argument("--model", metavar="FILE", required=True, help="input model")
        sp.add_argument("--mask", metavar="FILE", help="input mask")

        sp.add_argument("--noml", action="store_true", help="do not estimate D, S")
        sp.add_argument(
            "--rcut",
            metavar="LENGTH",
            type=float,
            default=10,
            help="maximum radius for evaluation of Gaussians in output map",
        )
        sp.add_argument(
            "-d",
            metavar="RESOLUTION",
            type=float,
            required=True,
            help="maximum resolution",
        )
        sp.add_argument(
            "--nbins",
            metavar="INT",
            type=int,
            default=50,
            help="number of frequency bins",
        )

    args = parser.parse_args()
    args.func(args)


def do_sample(args):
    from . import sampler

    rng_key = jax.random.key(int(time.time()))
    rng_key, init_key, sample_key = jax.random.split(rng_key, 3)

    # load observations
    print("loading data")
    mpgrid, mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(
        args.map, args.mask
    )
    rcut = dencalc.calc_rcut(args.rcut, spacing)

    st = gemmi.read_structure(args.model)
    st_aty = gemmi.read_structure(args.model)
    coords, it92, umat, occ, aty, atycounts, atnames, atydesc, unq_ind = (
        util.from_gemmi(st, st_aty)
    )
    naty = len(atycounts)

    print(f"{naty} atom types identified")

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
        util.write_map(v_iam, mpgrid, args.im)

    if not args.params:
        # FFT
        f_obs = jnp.fft.fftn(mpdata) * fft_scale

        # initialise frequency grid
        freqs, fbins, bin_cent = dencalc.make_bins(
            mpdata, spacing, 1 / args.d, args.nbins
        )

        # calculate D and S
        if args.noml:
            D, sigma_n = jnp.ones(args.nbins), jnp.ones(args.nbins)
        else:
            D, sigma_n = dencalc.calc_ml_params(
                mpdata, v_iam, fbins, jnp.arange(args.nbins)
            )

        D_gr, sg_n_gr = D[fbins], sigma_n[fbins]

        # generate minibatches
        inds1d = jnp.argwhere(fbins < args.nbins)
        inds1dr = jax.random.choice(
            rng_key, inds1d, axis=0, shape=((args.nsamples) * 512,)
        )
        pts1dr = freqs[inds1dr[:, 0], inds1dr[:, 1], inds1dr[:, 2]]

        indsbatched = inds1dr.reshape(-1, 512, 3)
        ptsbatched = pts1dr.reshape(-1, 512, 3)
        batched = jnp.stack([ptsbatched, indsbatched], axis=1)

        # set up distributions & preconditioner
        loglik = partial(
            sampler.loglik_fn,
            target=f_obs,
            fbins=fbins,
            nbins=args.nbins,
            coords=coords,
            umat=umat,
            occ=occ,
            it92=it92,
            aty=aty,
            D=D_gr,
            sigma_n=sg_n_gr,
            data_size=len(inds1d),
        )
        logprior = partial(
            sampler.logprior_fn,
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

        init_params = {
            "weights": jnp.ones(naty),
            "sigma": jnp.zeros(naty),
        }
        prec_mat = dencalc.calc_hess(
            init_params,
            coords=coords,
            umat=umat,
            occ=occ,
            it92=it92,
            aty=aty,
            naty=naty,
            sigma_n=sigma_n,
            bins=bin_cent,
        )
        prec_fn = jax.jit(lambda *_: prec_mat)

        # sample with SGLD
        print("sampling")

        sgld = sampler.prec_sgld(grad_fn, prec_fn)
        params = sampler.inference_loop(
            sample_key,
            jax.jit(sgld.step),
            batched,
            init_params,
            grad_vmap,
        )

        print("saving parameters")
        params["steps"] = sampler.scheduler(jnp.arange(args.nsamples) + 1)
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

        util.write_map(v_approx, mpgrid, args.om)


def do_ml(args):
    from . import spherical

    print("loading data")
    _, mpdata, _, bsize, spacing, bounds = util.read_mrc(args.map, args.mask)
    _, _, freqs = dencalc.make_bins(mpdata, spacing, spacing / args.d, args.nbins)
    rcut = dencalc.calc_rcut(args.rcut, spacing)
    mgrid = dencalc.make_grid(bounds, bsize)

    st = gemmi.read_structure(args.model)
    st_aty = gemmi.read_structure(args.model)
    coords, _, umat, _, aty, atycounts, _, atydesc, unq_id = util.from_gemmi(
        st, st_aty, b_iso=False
    )
    naty = len(atycounts)

    print("calculating FFT")
    gaussians = dencalc.calc_gaussians(coords, umat, aty, mgrid, rcut, naty)
    spherical_grid = spherical.make_spherical_grid(freqs, bsize, args.lmax)

    nufft = spherical.calc_nufft(gaussians, freqs, *spherical_grid)
    nufft.block_until_ready()

    quadwt = spherical.make_quad_weights(freqs.shape, args.lmax)

    print("performing quadrature with", quadwt.shape[1], "points")
    mats = spherical.calc_mat(nufft, quadwt)
    vecs = spherical.calc_vec(mpdata, nufft, freqs, quadwt, *spherical_grid)

    print("solving")
    soln, _, _, _ = spherical.batch_lstsq(mats, vecs)
    frang = freqs / spacing
    jnp.savez(
        args.o,
        soln=soln,
        mats=mats,
        vecs=vecs,
        freqs=frang,
        aty=atydesc[unq_id],
    )


if __name__ == "__main__":
    main()
