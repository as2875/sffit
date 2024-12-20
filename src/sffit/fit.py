import argparse
import os.path
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
        "sample", description="sample scattering factors using MCMC"
    )
    parser_sample.add_argument(
        "--nsamples",
        metavar="INT",
        type=int,
        required=True,
        help="number of MCMC samples",
    )
    parser_sample.set_defaults(func=do_sample)

    parser_gp = subparsers.add_parser(
        "gp", description="estimate scattering factors using GP regression"
    )
    parser_gp.add_argument(
        "--direct",
        action="store_true",
        help="calculate observed structure factors from model using direct summation (debugging)",
    )
    parser_gp.set_defaults(func=do_gp)

    # shared parameters
    for sp in (parser_sample, parser_gp):
        sp.add_argument(
            "--maps", nargs="+", metavar="FILE", required=True, help="input maps"
        )
        sp.add_argument(
            "--models", nargs="+", metavar="FILE", required=True, help="input models"
        )
        sp.add_argument("--masks", nargs="+", metavar="FILE", help="input masks")
        sp.add_argument(
            "-o",
            metavar="FILE",
            required=True,
            help="output .npz with parameters",
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
            help="cutoff radius for evaluation of atom density",
        )
        sp.add_argument(
            "--noml",
            action="store_true",
            help="do not estimate scale parameters (debugging)",
        )

    parser_fcalc = subparsers.add_parser(
        "fcalc", description="compute ESP from model and provided scattering factors"
    )
    parser_fcalc.add_argument(
        "--params", metavar="FILE", required=True, help="input .npz with parameters"
    )
    parser_fcalc.add_argument(
        "--models", nargs="+", metavar="FILE", required=True, help="input models"
    )
    parser_fcalc.add_argument(
        "--maps",
        nargs="+",
        metavar="FILE",
        required=True,
        help="input maps (for unit cell and pixel size)",
    )
    parser_fcalc.add_argument(
        "-o", metavar="DIR", required=True, help="directory for output maps"
    )
    parser_fcalc.add_argument(
        "--nbins", metavar="INT", type=int, default=50, help="number of frequency bins"
    )
    parser_fcalc.add_argument(
        "--approx", action="store_true", help="allow approximate matches for atom types"
    )
    parser_fcalc.set_defaults(func=do_fcalc)

    args = parser.parse_args()
    args.func(args)


def make_batches(
    coords,
    umat,
    occ,
    aty,
    it92,
    molind,
    map_paths,
    mask_paths,
    nbins,
    nsamples,
    rcut,
    rng_key,
    noml=False,
):
    nst = len(map_paths)
    batch_size = 512 // nst
    batches = [
        np.empty((nsamples, nst, batch_size, 3)),
        np.empty((nsamples, nst, batch_size), dtype=complex),
        np.empty((nsamples, nst, batch_size)),
        np.empty((nsamples, nst, batch_size)),
    ]
    data_size = np.empty(nst)
    D_1d, sg_n_1d, bins_1d = [jnp.empty((nst, nbins)) for _ in range(3)]

    for i, (map_path, mask_path) in enumerate(zip(map_paths, mask_paths)):
        mpgrid, mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(
            map_path, mask_path
        )
        rcut = dencalc.calc_rcut(rcut, spacing)

        v_iam = dencalc.calc_v_sparse(
            coords[molind == i],
            umat[molind == i],
            occ[molind == i],
            aty[molind == i],
            it92,
            rcut,
            bounds,
            bsize,
        )
        f_obs = jnp.fft.rfftn(mpdata) * fft_scale
        freqs, fbins, bin_cent = dencalc.make_bins(
            mpdata, spacing, 1 / (bsize * spacing), 1 / (2 * spacing), nbins
        )

        # calculate D and S
        if noml:
            D, sigma_n = jnp.ones(nbins), jnp.ones(nbins)
        else:
            D, sigma_n = dencalc.calc_ml_params(mpdata, v_iam, fbins, jnp.arange(nbins))

        D_gr, sg_n_gr = D[fbins], sigma_n[fbins]
        inds1d = jnp.argwhere((fbins < nbins) & fbins >= 0)
        data_size[i] = len(inds1d)
        inds1dr = jax.random.choice(
            rng_key, inds1d, axis=0, shape=(nsamples, batch_size)
        )

        for j, arr in enumerate((freqs, f_obs, D_gr, sg_n_gr)):
            batches[j][:, i, :] = arr[inds1dr[..., 0], inds1dr[..., 1], inds1dr[..., 2]]

        D_1d = D_1d.at[i].set(D)
        sg_n_1d = sg_n_1d.at[i].set(sigma_n)
        bins_1d = bins_1d.at[i].set(bin_cent)

        # update random key
        _, rng_key = jax.random.split(rng_key)

    batches = tuple(map(jnp.asarray, batches))
    data_size = jnp.asarray(data_size)

    return batches, data_size, D_1d, sg_n_1d, bins_1d


def do_sample(args):
    rng_key = jax.random.key(int(time.time()))
    rng_key, sample_key = jax.random.split(rng_key)

    print("loading data")
    structures = [gemmi.read_structure(p) for p in args.models]
    coords, it92_init, umat, occ, aty, _, atycounts, atydesc, molind = (
        util.from_multiple(structures)
    )

    if args.masks is None:
        mask_paths = repeat(None)
    else:
        mask_paths = args.masks

    batches, data_size, D, sigma_n, bin_cent = make_batches(
        coords,
        umat,
        occ,
        aty,
        it92_init,
        molind,
        args.maps,
        mask_paths,
        args.nbins,
        args.nsamples,
        args.rcut,
        rng_key,
        args.noml,
    )
    print(data_size)
    nmol, naty = len(structures), len(atycounts)

    # set up distributions & preconditioner
    loglik = partial(
        sampler.loglik_fn,
        coords=coords,
        umat=umat,
        occ=occ,
        aty=aty,
        molind=molind,
        data_size=data_size,
        nmol=nmol,
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
            molind=molind,
            nmol=nmol,
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
        batches,
        it92_init,
    )
    it92_tr = sampler.transform_params(it92_samples)

    print("saving parameters")
    jnp.savez(
        args.o,
        it92=it92_tr,
        steps=step_size,
        aty=atydesc,
        atycounts=atycounts,
    )


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
    smin, smax = util.freq_range(map_paths)

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


def do_gp(args):
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
    soln, params = spherical.solve(
        mats,
        vecs,
        bin_cent,
        jnp.identity(len(aty)),
    )

    print("saving parameters")
    jnp.savez(
        args.o,
        soln=soln,
        freqs=bin_cent,
        aty=aty,
        **params,
    )


def do_fcalc(args):
    params = np.load(args.params)
    infmethod = util.InferenceMethod.from_npz(params)

    for map_path, model_path in zip(args.maps, args.models):
        print("loading", model_path)
        st = gemmi.read_structure(model_path)
        coords, _, umat, occ, aty, _, _, atydesc = util.from_gemmi(st)
        atymap = util.align_aty(params["aty"], atydesc, approx=args.approx)

        mpgrid, mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(map_path)
        freqs, fbins, bin_cent = dencalc.make_bins(
            mpdata, spacing, 1 / (bsize * spacing), 1 / (2 * spacing), args.nbins
        )

        print("computing scattering factors on new grid")
        match infmethod:
            case util.InferenceMethod.MCMC:
                soln = sampler.eval_sog(params["it92"], bin_cent, params["steps"])

            case util.InferenceMethod.GP:
                cov_params = {
                    k: v for k, v in params.items() if k in ["scale", "alpha", "beta"]
                }
                soln = spherical.eval_sf(
                    bin_cent, params["freqs"], params["soln"], cov_params
                )

        print("calculating map")
        naty = len(atydesc)
        sigma_n = jnp.ones_like(fbins)
        gaussians = dencalc.calc_gaussians_direct(
            coords,
            umat,
            occ,
            aty,
            freqs,
            sigma_n,
            naty,
            fft_scale,
        )
        coefs = jnp.zeros((args.nbins, naty))
        atymatch = atymap >= 0
        coefs = coefs.at[:, atymatch].set(soln[:, atymap[atymatch]])
        v_calc = spherical.reconstruct(
            gaussians,
            coefs,
            sigma_n,
            fbins,
            jnp.arange(args.nbins),
        )

        print("writing output")
        outname, _ = os.path.splitext(os.path.basename(model_path))
        outpath = os.path.join(args.o, outname + ".mrc")
        util.write_map(v_calc, mpgrid, outpath)


if __name__ == "__main__":
    main()
