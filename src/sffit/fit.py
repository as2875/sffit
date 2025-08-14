import argparse
import copy
import json
import multiprocessing as mp
import pathlib
import time
from functools import partial
from itertools import repeat

import jax
import jax.numpy as jnp
import gemmi
import numpy as np

from . import dencalc
from . import radn
from . import spherical
from . import sampler
from . import util


def main():
    parser = argparse.ArgumentParser(
        description="utilities to fit scattering factors to cryo-EM SPA maps"
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

    # I/O
    parser_sample = subparsers.add_parser(
        "sample", description="sample scattering factors using MCMC"
    )
    parser_sample.add_argument(
        "--maps", nargs="+", metavar="FILE", required=True, help="input maps"
    )
    parser_sample.add_argument(
        "--models", nargs="+", metavar="FILE", required=True, help="input models"
    )
    parser_sample.add_argument("--masks", nargs="+", metavar="FILE", help="input masks")
    parser_sample.add_argument(
        "-o",
        metavar="FILE",
        required=True,
        help="output .npz with parameters",
    )

    # density calculator
    parser_sample.add_argument(
        "--nbins",
        metavar="INT",
        type=int,
        default=50,
        help="number of frequency bins",
    )
    parser_sample.add_argument(
        "--rcut",
        metavar="LENGTH",
        type=float,
        default=10,
        help="cutoff radius for evaluation of atom density",
    )
    parser_sample.add_argument(
        "--noml",
        action="store_true",
        help="do not estimate scale parameters (debugging)",
    )

    # sampler
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

    # I/O
    parser_gp.add_argument(
        "-o",
        metavar="FILE",
        required=True,
        help="output .npz with parameters",
    )

    # these should form two mutually exclusive groups, but we can't do this with argparse
    # group 1: provide maps, models, masks, and path to write overlap integrals
    parser_gp.add_argument("--maps", nargs="+", metavar="FILE", help="input maps")
    parser_gp.add_argument("--models", nargs="+", metavar="FILE", help="input models")
    parser_gp.add_argument("--masks", nargs="+", metavar="FILE", help="input masks")
    parser_gp.add_argument(
        "-L", nargs="+", metavar="FILE", help="CIF dictionary files (optional)"
    )
    parser_gp.add_argument(
        "-oi", metavar="FILE", help="output .npz file with overlap integrals"
    )
    # group 2: provide precomputed overlap integrals
    parser_gp.add_argument(
        "-ii", metavar="FILE", help="input .npz file with overlap integrals"
    )

    # density calculator
    parser_gp.add_argument(
        "--nbins",
        metavar="INT",
        type=int,
        default=50,
        help="number of frequency bins",
    )
    parser_gp.add_argument(
        "--rcut",
        metavar="LENGTH",
        type=float,
        default=10,
        help="cutoff radius for evaluation of atom density",
    )
    parser_gp.add_argument(
        "--noml",
        action="store_true",
        help="do not estimate scale parameters (debugging)",
    )
    parser_gp.add_argument(
        "--no-change-h",
        action="store_true",
        help="do not re-add hydrogens to the model",
    )
    parser_gp.add_argument(
        "--no-filter",
        action="store_true",
        help="do not exclude atoms with large B-values from calculation",
    )

    # GP parameters
    parser_gp.add_argument(
        "--weight",
        metavar="FLOAT",
        type=float,
        help="additional weighting factor for observations",
    )
    parser_gp.set_defaults(func=do_gp)

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
        "-o",
        nargs="+",
        metavar="FILE",
        required=True,
        help="filenames for calculated maps",
    )
    parser_fcalc.add_argument(
        "--nbins", metavar="INT", type=int, default=50, help="number of frequency bins"
    )
    parser_fcalc.add_argument(
        "--rcut",
        metavar="LENGTH",
        type=float,
        default=10,
        help="cutoff radius for evaluation of atom density",
    )
    parser_fcalc.add_argument(
        "--approx", action="store_true", help="allow approximate matches for atom types"
    )
    parser_fcalc.set_defaults(func=do_fcalc)

    parser_iam = subparsers.add_parser(
        "iam", description="compute ESP from model and tabulated scattering factors"
    )
    parser_iam.add_argument(
        "--models", nargs="+", metavar="FILE", required=True, help="input models"
    )
    parser_iam.add_argument(
        "--maps",
        nargs="+",
        metavar="FILE",
        required=True,
        help="input maps (for unit cell and pixel size)",
    )
    parser_iam.add_argument(
        "-o",
        nargs="+",
        metavar="FILE",
        required=True,
        help="filenames for calculated maps",
    )
    parser_iam.set_defaults(func=do_iam)

    parser_mmcif = subparsers.add_parser(
        "mmcif", description="make mmCIF file with custom scattering factors"
    )
    parser_mmcif.add_argument(
        "--params", metavar="FILE", help="input .npz with parameters"
    )
    parser_mmcif.add_argument(
        "-ii", metavar="FILE", help="input .npz with overlap integrals"
    )
    parser_mmcif.add_argument(
        "--models", nargs="+", metavar="FILE", help="input models"
    )
    parser_mmcif.add_argument(
        "--approx", action="store_true", help="allow approximate matches for atom types"
    )
    parser_mmcif.add_argument(
        "-o",
        nargs="+",
        metavar="FILE",
        help="filenames for output mmCIF",
    )
    parser_mmcif.add_argument(
        "-oj",
        metavar="FILE",
        required=True,
        help="filename for output JSON with SoG parameters",
    )
    parser_mmcif.set_defaults(func=do_mmcif)

    parser_radn = subparsers.add_parser(
        "radn",
        help="perform joint fitting of scattering factors and radiation damage model",
    )

    # I/O
    parser_radn.add_argument(
        "--maps",
        nargs="+",
        metavar="FILE",
        required=True,
        help="input maps, in order of increasing dose",
    )
    parser_radn.add_argument("--mask", metavar="FILE", help="input mask")
    parser_radn.add_argument(
        "--model", metavar="FILE", required=True, help="input model"
    )
    parser_radn.add_argument(
        "--dose", metavar="FLOAT", type=float, required=True, help="total dose"
    )
    parser_radn.add_argument(
        "--scratch",
        metavar="DIR",
        required=True,
        help="scratch directory for temporary files",
    )

    # algorithm parameters
    parser_radn.add_argument(
        "--ncycle",
        metavar="INT",
        type=int,
        required=True,
        help="number of ECM cycles",
    )
    parser_radn.add_argument(
        "--dmin",
        metavar="ANG",
        required=True,
        type=float,
        help="refinement resolution",
    )
    parser_radn.add_argument(
        "--blur",
        metavar="ANG SQ",
        default=0,
        type=float,
        help="B value to add to the input structure before scale calculation",
    )
    parser_radn.add_argument(
        "--weight",
        metavar="FLOAT",
        default=0.1,
        type=float,
        help="weight of ADP and occupancy restraints",
    )

    parser_radn.set_defaults(func=do_radn)

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
        mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(map_path, mask_path)
        pixrcut = dencalc.calc_rcut(rcut, spacing)

        v_iam = dencalc.calc_v_sparse(
            coords[molind == i],
            umat[molind == i],
            occ[molind == i],
            aty[molind == i],
            it92,
            pixrcut,
            bounds,
            bsize,
        )
        f_obs = jnp.fft.rfftn(mpdata) * fft_scale
        freqs, fbins, bin_cent = dencalc.make_bins(
            mpdata.shape[0], spacing, 1 / (bsize * spacing), 1 / (2 * spacing), nbins
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
    cif_paths,
    nbins,
    rcut,
    noml=False,
    nochangeh=False,
    nofilter=False,
):
    flabels = jnp.arange(nbins)
    matlist, veclist, atylist, countlist = [], [], [], []
    smin, smax = util.freq_range(map_paths)

    print(f"using resolution range: {smin:.2f} 1/ang to {smax:.2f} 1/ang")
    bins = jnp.linspace(smin, smax, nbins + 1)
    bin_cent = 0.5 * (bins[1:] + bins[:-1])
    power = jnp.zeros_like(bin_cent)

    for model_path, map_path, mask_path, cif_path in zip(
        model_paths, map_paths, mask_paths, cif_paths
    ):
        print("loading", model_path)
        st = gemmi.read_structure(model_path)
        selections = None if nofilter else util.make_selections(st)
        coords, it92, umat, occ, aty, atmask, atycounts, atydesc = util.from_gemmi(
            st, selections=selections, cif=cif_path, nochangeh=nochangeh
        )
        mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(map_path, mask_path)
        pixrcut = dencalc.calc_rcut(rcut, spacing)
        blur = dencalc.calc_blur(umat, spacing)
        freqs, fbins, _ = dencalc.make_bins(mpdata.shape[0], spacing, smin, smax, nbins)

        if noml:
            D, sigma_n = jnp.ones(nbins), jnp.ones(nbins)
        else:
            v_iam = dencalc.calc_v_sparse(
                coords,
                umat,
                occ,
                aty,
                it92,
                pixrcut,
                bounds,
                bsize,
            )
            D, sigma_n = dencalc.calc_ml_params(mpdata, v_iam, fbins, flabels)

        D_gr, sg_n_gr = D[fbins], sigma_n[fbins]
        f_obs = dencalc.subtract_density(
            mpdata,
            D_gr,
            atmask,
            coords,
            umat,
            occ,
            aty,
            it92,
            pixrcut,
            bounds,
            bsize,
        )
        aty, atycounts, atydesc = util.reindex_excluded(atmask, aty, atydesc)
        gaussians = dencalc.calc_gaussians_fft(
            coords,
            umat,
            occ,
            aty,
            D_gr,
            sg_n_gr,
            pixrcut,
            bounds,
            bsize,
            len(atydesc),
            blur,
            freqs,
        )

        power += dencalc.calc_power(f_obs, fbins, flabels, sg_n_gr)
        mats, vecs = spherical.calc_mats_and_vecs(
            gaussians, f_obs, sg_n_gr, fbins, flabels
        )
        mats.block_until_ready()
        vecs.block_until_ready()
        matlist += [mats]
        veclist += [vecs]
        atylist += [atydesc]
        countlist += [atycounts]

    atyref = np.unique(np.concatenate(atylist), axis=0)
    naty = len(atyref)
    refmats = jnp.zeros((nbins, naty, naty))
    refvecs = jnp.zeros((nbins, naty))
    refcounts = np.zeros(naty, dtype=int)
    reflabels = np.zeros((naty, len(model_paths)), dtype=bool)

    for ind, (mats, vecs, aty, atycounts) in enumerate(
        zip(matlist, veclist, atylist, countlist)
    ):
        refmats, refvecs, refcounts, reflabels = spherical.align_linsys(
            atyref,
            aty,
            atycounts,
            refmats,
            refvecs,
            refcounts,
            reflabels,
            mats,
            vecs,
            ind,
        )

    return refmats, refvecs, power, atyref, refcounts, reflabels, bin_cent


def do_gp(args):
    print("loading data")
    assert (args.maps and args.models and args.oi) != args.ii, (
        "invalid argument combination"
    )
    precomputed = bool(args.ii)
    mask_paths = args.masks if args.masks else repeat(None)
    cif_paths = args.L if args.L else repeat(None)

    if precomputed:
        interm = jnp.load(args.ii)
        mats, vecs, power, aty, bin_cent = (
            interm["mats"],
            interm["vecs"],
            interm["power"],
            interm["aty"],
            interm["freqs"],
        )
    else:
        datapath = np.array(args.models)
        mats, vecs, power, aty, refcounts, reflabels, bin_cent = make_linear_system(
            args.models,
            args.maps,
            mask_paths,
            cif_paths,
            args.nbins,
            args.rcut,
            args.noml,
            args.no_change_h,
            args.no_filter,
        )
        jnp.savez(
            args.oi,
            mats=mats,
            vecs=vecs,
            power=power,
            freqs=bin_cent,
            aty=aty,
            atycounts=refcounts,
            atymat=reflabels,
            datapath=datapath,
        )

    atycov = np.identity(len(aty))
    atycov[aty[:, 0] == 255] = 0.0
    atycov = jnp.array(atycov)

    if not args.weight:
        weights = jnp.logspace(-6, -2, 100)
        cutoff = np.flatnonzero(bin_cent > 0.2).min()
        print(f"cutoff index for cross validation is {cutoff.item()}")
        _, _, _, loss = jax.lax.map(
            lambda wt: spherical.solve(
                mats=mats,
                vecs=vecs,
                power=power,
                bin_cent=bin_cent,
                aty_cov=atycov,
                co=cutoff,
                weight=wt,
            ),
            weights,
        )
        optimal_weight = weights[jnp.nanargmax(loss)]
    else:
        optimal_weight = args.weight
        weights, loss = np.array([]), np.array([])

    soln, var, params, _ = spherical.solve(
        mats,
        vecs,
        power,
        bin_cent,
        atycov,
        co=None,
        weight=optimal_weight,
    )

    print("saving parameters")
    jnp.savez(
        args.o,
        soln=soln,
        var=var,
        freqs=bin_cent,
        aty=aty,
        weights=weights,
        loss=loss,
        **params,
    )


def do_fcalc(args):
    params = np.load(args.params)
    infmethod = util.InferenceMethod.from_npz(params)

    for map_path, model_path, out_path in zip(args.maps, args.models, args.o):
        print("loading", model_path)
        st = gemmi.read_structure(model_path)
        coords, _, umat, occ, aty, _, _, atydesc = util.from_gemmi(st, nochangeh=True)
        atymap = util.align_aty(params["aty"], atydesc, approx=args.approx)

        mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(map_path)
        pixrcut = dencalc.calc_rcut(args.rcut, spacing)
        freqs, fbins, bin_cent = dencalc.make_bins(
            mpdata.shape[0],
            spacing,
            1 / (bsize * spacing),
            1 / (2 * spacing),
            args.nbins,
        )

        print("computing scattering factors on new grid")
        match infmethod:
            case util.InferenceMethod.MCMC:
                soln = sampler.eval_sog(params["it92"], bin_cent, params["steps"])

            case util.InferenceMethod.GP:
                cov_params = {k: v for k, v in params.items() if k in ["scale", "beta"]}
                soln = spherical.eval_sf(
                    bin_cent, params["freqs"], params["soln"], cov_params
                )

        print("calculating map")
        naty = len(atydesc)
        scale = jnp.ones_like(fbins)
        blur = dencalc.calc_blur(umat, spacing)
        gaussians = dencalc.calc_gaussians_fft(
            coords,
            umat,
            occ,
            aty,
            scale,
            scale,
            pixrcut,
            bounds,
            bsize,
            naty,
            blur,
            freqs,
        )
        coefs = jnp.zeros((args.nbins, naty))
        atymatch = atymap >= 0
        coefs = coefs.at[:, atymatch].set(soln[:, atymap[atymatch]])
        v_calc = spherical.reconstruct(
            gaussians,
            coefs,
            scale,
            fbins,
            jnp.arange(args.nbins),
        )

        print("writing output")
        util.write_map(v_calc, out_path, bsize, bsize * spacing)


def do_iam(args):
    for map_path, model_path, out_path in zip(args.maps, args.models, args.o):
        print("loading", model_path)
        st = gemmi.read_structure(model_path)
        ccp4 = gemmi.read_ccp4_map(map_path)

        st.setup_entities()
        st.expand_ncs(gemmi.HowToNameCopiedChain.Short)

        dc = gemmi.DensityCalculatorE()
        dc.grid.copy_metadata_from(ccp4.grid)
        dc.initialize_grid()
        dc.put_model_density_on_grid(st[0])

        print("writing output")
        dim, cell = ccp4.grid.nu, ccp4.grid.unit_cell.a
        util.write_map(dc.grid.array, out_path, dim, cell)


def do_mmcif(args):
    params = np.load(args.params)
    intparams = np.load(args.ii)
    infmethod = util.InferenceMethod.from_npz(params)
    if infmethod is not util.InferenceMethod.GP:
        raise NotImplementedError("Only output of gp subcommand is supported")

    print("fitting sum of Gaussians")
    cov_params = {k: v for k, v in params.items() if k in ["scale", "beta"]}
    newfreq = jnp.linspace(0, 2, 200)
    extrapolated = spherical.eval_sf(
        newfreq, params["freqs"], params["soln"], cov_params
    )
    sftab = np.zeros((len(params["aty"]), 10))
    for ind, atyrow in enumerate(params["aty"]):
        elem = gemmi.Element(atyrow[0])
        sftab[ind] = np.concatenate([elem.c4322.a, elem.c4322.b])

    fitted_sog = spherical.fit_sog(newfreq, extrapolated, sftab)
    eval_sog = sampler.eval_sog(fitted_sog[None], params["freqs"], None)
    err = jnp.mean((params["soln"] - eval_sog) ** 2, axis=0)
    print("MSE in fit:", err)

    # write results to a JSON file
    with open(args.oj, "w") as f:
        json.dump(
            {
                util.aty_to_str(t): {
                    "coefficients": [
                        {"a": round(a, 4), "b": round(b, 4)}
                        for a, b in zip(c[:5], c[5:])
                    ],
                    "count": n,
                }
                for t, c, n in zip(
                    params["aty"], fitted_sog.tolist(), intparams["atycounts"].tolist()
                )
                if t[0] != 255
            },
            f,
        )

    if not (args.models and args.o):
        return

    for model_path, out_path in zip(args.models, args.o):
        print("loading", model_path)
        st = gemmi.read_structure(model_path)
        _, _, _, _, aty, _, _, atydesc = util.from_gemmi(st, nochangeh=True)
        atymap = util.align_aty(params["aty"], atydesc, approx=args.approx)
        print("indices of matching atom types:", atymap)

        # if we don't have an atom type, use tabulated values
        fitted_mapped = np.zeros((len(atymap), 10))
        for i, j in enumerate(atymap):
            elem = gemmi.Element(atydesc[i, 0])
            fitted_mapped[i] = (
                fitted_sog[j]
                if j >= 0
                else np.concatenate([elem.c4322.a, elem.c4322.b])
            )

        sel = gemmi.Selection(";q=0")
        sel.remove_selected(st)

        block = st.make_mmcif_block()
        loop = block.find_loop("_atom_site.id").get_loop()
        loop.add_columns(["_atom_site.scat_id"], value="?")

        table = block.find("_atom_site.", ["scat_id"])
        assert len(table) == aty.size
        for ind, row in enumerate(table):
            row[0] = str(aty[ind])

        sfloop = block.init_loop(
            "_lmb_scat_coef.",
            [
                "scat_id",
                "coef_a1",
                "coef_a2",
                "coef_a3",
                "coef_a4",
                "coef_a5",
                "coef_b1",
                "coef_b2",
                "coef_b3",
                "coef_b4",
                "coef_b5",
            ],
        )

        for ind, _ in enumerate(atydesc):
            sfloop.add_row(
                [
                    str(ind),
                    *[f"{c:.4f}" for c in fitted_mapped[ind, :5]],
                    *[f"{c:.4f}" for c in fitted_mapped[ind, 5:]],
                ],
            )

        block.write_file(out_path)


def do_radn(args):
    print("loading data")
    mpdata, fft_scale, bsize, spacing, bounds = util.read_multiple(args.maps, args.mask)
    cell_size = bsize * spacing
    nmaps = len(args.maps)

    # prepare input structure
    input_st = gemmi.read_structure(args.model)
    input_st.setup_entities()
    input_st.cell = gemmi.UnitCell(cell_size, cell_size, cell_size, 90, 90, 90)
    input_st.setup_cell_images()

    monlib = util.setup_monlib(input_st)
    _ = gemmi.prepare_topology(
        input_st,
        monlib,
        h_change=gemmi.HydrogenChange.Remove,
    )
    structures = [copy.deepcopy(input_st) for ind in range(nmaps)]
    structures = [radn.shift_b(st, args.blur) for st in structures]

    scratch_dir = pathlib.Path(args.scratch)
    if not scratch_dir.exists():
        scratch_dir.mkdir()
    assert scratch_dir.is_dir()
    refn_dir = scratch_dir / "scratch"
    refn_dir.mkdir(exist_ok=True)
    result_dir = scratch_dir / "result"
    result_dir.mkdir(exist_ok=True)

    fbins, bin_cent, d_min_max = radn.make_servalcat_bins(bsize, spacing, args.dmin)

    dose = jnp.linspace(args.dose / nmaps, args.dose, nmaps, endpoint=True)
    flabels = jnp.arange(len(bin_cent))

    mpdata = radn.mask_extrema(mpdata, fbins)
    f_calc = radn.calc_f_gemmi_multiple(structures, bsize, d_min_max[0])

    print("- estimating hyperparameters")
    hparams = radn.calc_hyperparams(
        mpdata,
        fbins,
        flabels,
        bin_cent,
        dose,
    )
    jax.block_until_ready(hparams)

    print("- calculating posterior expectation")
    f_smoothed = radn.smooth_maps(hparams, mpdata, fbins, flabels, bin_cent, dose)
    jax.block_until_ready(f_smoothed)

    radn.scale_b(f_smoothed, f_calc, structures, bsize, spacing)

    # change multiprocessing start method
    mp.set_start_method("spawn")

    for outer_step in range(args.ncycle):
        print(f"cycle {outer_step + 1}")
        D = radn.calc_D(f_smoothed, f_calc, fbins, flabels)

        print("- majorizing")
        refn_objective = radn.calc_refn_objective(
            hparams, mpdata, f_calc, D, fbins, flabels, bin_cent, dose
        )

        print("- minimizing")
        refn_args = []
        for inner_step in range(nmaps):
            servalcat_cwd = refn_dir / f"refine{inner_step:03d}"
            servalcat_cwd.mkdir(exist_ok=True)

            map_path, model_path = radn.servalcat_setup_input(
                servalcat_cwd,
                refn_objective[inner_step],
                structures[inner_step],
                bsize,
                spacing,
                fft_scale,
            )
            refn_args.append(
                (
                    servalcat_cwd,
                    map_path,
                    model_path,
                    inner_step,
                    args.dmin,
                    args.weight,
                    D[inner_step],
                    hparams,
                    bin_cent,
                    dose,
                )
            )

        with mp.Pool() as pool:
            output_paths = pool.starmap(radn.servalcat_run, refn_args)

        print("- updating Fc")
        for inner_step, output_path in enumerate(output_paths):
            # update Fc
            structures[inner_step] = gemmi.read_structure(str(output_path))
            structures[inner_step].make_mmcif_document().write_file(
                str(result_dir / f"model_{outer_step:02d}_{inner_step:03d}.cif")
            )

            f_calc = radn.update_f_gemmi(
                f_calc, inner_step, structures[inner_step], bsize, d_min_max[0]
            )

        # write debug info
        elbo = radn.calc_elbo(
            hparams,
            mpdata,
            f_calc,
            D,
            fbins,
            bin_cent,
            dose,
        )
        print(f"ELBO {elbo}")
        jnp.savez(
            result_dir / f"params_{outer_step:02d}.npz",
            D=D,
            freqs=bin_cent,
            dose=dose,
            elbo=elbo,
            **hparams,
        )


if __name__ == "__main__":
    main()
