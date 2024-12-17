import argparse
import time
from functools import partial
from itertools import repeat

import jax
import jax.numpy as jnp
import gemmi
import numpy as np

from . import dencalc
from . import spherical
from . import sampler
from . import util


def main():
    parser = argparse.ArgumentParser(
        description="utilities to fit scattering factors to cryo-EM SPA maps"
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

    parser_sample = subparsers.add_parser(
        "sample", description="sample parameters using MCMC"
    )

    # I/O
    parser_sample.add_argument("--map", metavar="FILE", required=True, help="input map")
    parser_sample.add_argument(
        "--model", metavar="FILE", required=True, help="input model"
    )
    parser_sample.add_argument("--mask", metavar="FILE", help="input mask")
    parser_sample.add_argument("-im", metavar="FILE", help="initial calculated map")
    parser_sample.add_argument("-om", metavar="FILE", help="final calculated map")

    pgroup = parser_sample.add_mutually_exclusive_group(required=True)
    pgroup.add_argument("--params", metavar="FILE", help=".npz file with parameters")
    pgroup.add_argument("-op", metavar="FILE", help="output .npz with parameters")
    parser_sample.set_defaults(func=do_sample)

    # sampler parameters
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
    parser_sample.add_argument(
        "-d",
        metavar="RESOLUTION",
        type=float,
        required=True,
        help="resolution range",
    )

    parser_ml = subparsers.add_parser(
        "ml", description="estimate scattering factors using least squares and FFT"
    )

    # I/O
    parser_ml.add_argument(
        "--maps", nargs="+", metavar="FILE", required=True, help="input maps"
    )
    parser_ml.add_argument(
        "--models", nargs="+", metavar="FILE", required=True, help="input models"
    )
    parser_ml.add_argument("--masks", nargs="+", metavar="FILE", help="input masks")
    parser_ml.add_argument(
        "-o",
        metavar="FILE",
        required=True,
        help="output .npz with parameters",
    )

    # calculation parameters
    parser_ml.add_argument(
        "--direct",
        action="store_true",
        help="calculate observed structure factors from model using direct summation (for testing)",
    )
    parser_ml.set_defaults(func=do_ml)

    # shared parameters
    for sp in (parser_sample, parser_ml):
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
            mpdata, spacing, 1 / (bsize * spacing), 1 / args.d, args.nbins
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


def make_linear_system(
    model_paths,
    map_paths,
    mask_paths,
    nbins,
    rcut,
    noml=False,
    direct=False,
):
    flabels = jnp.arange(nbins)
    matlist, veclist, atylist = [], [], []

    smin, smax = 0.0, jnp.inf
    for map_path in map_paths:
        ccp4 = gemmi.read_ccp4_map(map_path)
        mpmin = 1 / ccp4.grid.unit_cell.a
        mpmax = 1 / (2 * ccp4.grid.spacing[0])
        if mpmin > smin:
            smin = mpmin
        if mpmax < smax:
            smax = mpmax

    print(f"using resolution range: {smin:.2f} 1/ang to {smax:.2f} 1/ang")
    bins = jnp.linspace(smin, smax, nbins + 1)
    bin_cent = 0.5 * (bins[1:] + bins[:-1])

    for model_path, map_path, mask_path in zip(model_paths, map_paths, mask_paths):
        print("loading", model_path)
        st = gemmi.read_structure(model_path)
        coords, it92, umat, occ, aty, _, _, atydesc = util.from_gemmi(st)
        mpgrid, mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(
            map_path, mask_path
        )
        rcut = dencalc.calc_rcut(rcut, spacing)

        freqs, fbins, _ = dencalc.make_bins(mpdata, spacing, smin, smax, nbins)

        if noml:
            D, sigma_n = jnp.ones(nbins), jnp.ones(nbins)
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
            D, sigma_n = dencalc.calc_ml_params(mpdata, v_iam, fbins, flabels)

        D_gr, sg_n_gr = D[fbins], sigma_n[fbins]
        gaussians = dencalc.calc_gaussians_direct(
            coords,
            umat,
            occ,
            aty,
            freqs,
            sg_n_gr,
            len(atydesc),
            fft_scale,
        )

        if direct:
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

        mats, vecs = spherical.calc_mats_and_vecs(
            gaussians, f_obs, D_gr, sg_n_gr, fbins, flabels
        )
        matlist += [mats]
        veclist += [vecs]
        atylist += [atydesc]

    atyref = np.unique(np.concatenate(atylist), axis=0)
    naty = len(atyref)
    refmats = jnp.zeros((nbins, naty, naty))
    refvecs = jnp.zeros((nbins, naty))

    for mats, vecs, aty in zip(matlist, veclist, atylist):
        refmats, refvecs = spherical.align_linsys(
            atyref, aty, refmats, refvecs, mats, vecs
        )

    return jnp.asarray(refmats), jnp.asarray(refvecs), atyref, bin_cent


def do_ml(args):
    print("loading data")
    if args.masks is None:
        mask_paths = repeat(None)
    else:
        mask_paths = args.masks

    mats, vecs, aty, bin_cent = make_linear_system(
        args.models,
        args.maps,
        mask_paths,
        args.nbins,
        args.rcut,
        args.noml,
        args.direct,
    )
    soln = spherical.solve(
        mats,
        vecs,
        bin_cent,
        jnp.identity(len(aty)),
    )
    jnp.savez(
        args.o,
        soln=soln,
        freqs=bin_cent,
        aty=aty,
    )


if __name__ == "__main__":
    main()
