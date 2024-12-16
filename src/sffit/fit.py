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
        "-o",
        metavar="FILE",
        required=True,
        help="output .npz with parameters",
    )
    parser_ml.add_argument(
        "--direct",
        action="store_true",
        help="calculate observed structure factors from model using direct summation (for testing)",
    )
    parser_ml.set_defaults(func=do_ml)

    for sp in (parser_sample, parser_ml):
        sp.add_argument("--map", metavar="FILE", required=True, help="input map")
        sp.add_argument("--model", metavar="FILE", required=True, help="input model")
        sp.add_argument("--mask", metavar="FILE", help="input mask")
        sp.add_argument("-om", metavar="FILE", help="final calculated map")

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
        sp.add_argument(
            "--rcut",
            metavar="LENGTH",
            type=float,
            default=10,
            help="maximum radius for evaluation of Gaussians in output map",
        )
        sp.add_argument("--noml", action="store_true", help="do not estimate D, S")

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
    print(f"using cutoff {rcut}")

    st = gemmi.read_structure(args.model)
    coords, it92_init, umat, occ, aty, _, atycounts, atydesc = util.from_gemmi(st)
    naty = len(atycounts)

    print(f"{naty} atom types identified")

    if args.im or not (args.params or args.noml):
        v_iam = dencalc.calc_v_sparse(
            coords,
            umat,
            occ,
            aty,
            it92_init,
            rcut,
            bounds,
            bsize,
        )

    if args.im:
        util.write_map(v_iam, mpgrid, args.im)

    if not args.params:
        # FFT
        f_obs = jnp.fft.rfftn(mpdata) * fft_scale

        # initialise frequency grid
        freqs, fbins, bin_cent = dencalc.make_bins(
            mpdata, bsize, spacing, 1 / args.d, args.nbins
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
        inds1d = jnp.argwhere((fbins < args.nbins) & (fbins >= 0))
        inds1dr = jax.random.choice(rng_key, inds1d, axis=0, shape=(args.nsamples, 512))
        pts1dr = freqs[inds1dr[..., 0], inds1dr[..., 1], inds1dr[..., 2]]
        batched = jnp.stack([pts1dr, inds1dr], axis=1)

        # set up distributions & preconditioner
        loglik = partial(
            sampler.loglik_fn,
            target=f_obs,
            coords=coords,
            umat=umat,
            occ=occ,
            aty=aty,
            D=D_gr,
            sigma_n=sg_n_gr,
            data_size=len(inds1d),
        )
        logden = jax.jit(
            lambda params, batch: loglik(params, batch) + sampler.logprior_fn(params),
        )
        grad_fn = jax.grad(logden)
        prec_fn = jax.jit(
            partial(
                sampler.calc_hess,
                umat=umat,
                occ=occ,
                aty=aty,
                naty=naty,
                D=D,
                sigma_n=sigma_n,
                bins=bin_cent,
            )
        )

        # sample with SGLD
        print("sampling")

        sgld = sampler.prec_sgld(grad_fn, prec_fn)
        it92_samples, step_size = sampler.inference_loop(
            sample_key,
            jax.jit(sgld.step),
            batched,
            it92_init,
        )
        params = {"it92": sampler.transform_params(it92_samples), "steps": step_size}

        print("saving parameters")
        jnp.savez(
            args.op,
            aty=atydesc,
            atycounts=atycounts,
            **params,
        )

    else:
        params = jnp.load(args.params)

    if args.om:
        print("writing output map")
        step_size = params["steps"][args.nwarm :]
        params_post = jnp.average(
            params["it92"][args.nwarm :], axis=0, weights=step_size
        )

        v_approx = dencalc.calc_v_sparse(
            coords,
            umat,
            occ,
            aty,
            params_post,
            rcut,
            bounds,
            bsize,
        ).reshape(bsize, bsize, bsize)

        util.write_map(v_approx, mpgrid, args.om)


def do_ml(args):
    from . import spherical

    print("loading data")
    st = gemmi.read_structure(args.model)
    coords, it92, umat, occ, aty, _, atycounts, atydesc = util.from_gemmi(st)
    naty = len(atycounts)

    mpgrid, mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(
        args.map, args.mask
    )
    rcut = dencalc.calc_rcut(args.rcut, spacing)

    freqs, fbins, bin_cent = dencalc.make_bins(
        mpdata, bsize, spacing, 1 / args.d, args.nbins
    )
    flabels = jnp.arange(args.nbins)

    if args.noml:
        D, sigma_n = jnp.ones(args.nbins), jnp.ones(args.nbins)
    else:
        v_iam = dencalc.calc_v_sparse(
            coords,
            umat,
            occ,
            aty,
            it92,
            rcut,
            bounds,
            bsize,
        )
        D, sigma_n = dencalc.calc_ml_params(
            mpdata, v_iam, fbins, jnp.arange(args.nbins)
        )

    D_gr, sg_n_gr = D[fbins], sigma_n[fbins]
    gaussians = dencalc.calc_gaussians_direct(
        coords,
        umat,
        occ,
        aty,
        freqs,
        sg_n_gr,
        naty,
        fft_scale,
    )

    if args.direct:
        f_obs = dencalc.calc_f_scan(
            coords,
            umat,
            occ,
            aty,
            it92,
            freqs,
            fft_scale,
        )
    else:
        f_obs = jnp.fft.rfftn(mpdata)

    soln = spherical.solve(
        gaussians,
        f_obs,
        D_gr,
        sg_n_gr,
        fbins,
        flabels,
        bin_cent,
        jnp.identity(naty),
    )

    if args.om:
        reconstructed = spherical.reconstruct(
            gaussians,
            soln,
            sg_n_gr,
            fbins,
            flabels,
        )
        util.write_map(reconstructed, mpgrid, args.om)

    jnp.savez(
        args.o,
        soln=soln,
        freqs=bin_cent,
        aty=atydesc,
        atycounts=atycounts,
    )


if __name__ == "__main__":
    main()
