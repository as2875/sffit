# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import argparse
import base64
import json
import pathlib
from itertools import repeat

import jax
import jax.numpy as jnp
import gemmi
import numpy as np

from . import dencalc
from . import spherical
from . import util


def parse_args(*args):
    parser = argparse.ArgumentParser(
        description="utilities to fit scattering factors to cryo-EM SPA maps"
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

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
        "--params", metavar="FILE", help="input file containing parameters"
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
        help="filename for output JSON with SoG parameters",
    )
    parser_mmcif.set_defaults(func=do_mmcif)

    return parser.parse_args(*args)


def main():
    args = parse_args()
    args.func(args)


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
        cov_params = {k: v for k, v in params.items() if k in ["scale", "beta"]}
        soln = spherical.eval_sf(bin_cent, params["freqs"], params["soln"], cov_params)

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
    assert (args.ii is None) == (args.oj is None), "invalid argument combination"
    inpath = pathlib.Path(args.params)

    if inpath.suffix == ".npz":
        params = np.load(args.params)
        atyref = params["aty"]
        intparams = np.load(args.ii)

        print("fitting sum of Gaussians")
        cov_params = {k: v for k, v in params.items() if k in ["scale", "beta"]}
        newfreq = jnp.linspace(0, 2, 200)
        extrapolated = spherical.eval_sf(
            newfreq, params["freqs"], params["soln"], cov_params
        )
        sftab = np.zeros((len(atyref), 10))
        for ind, atyrow in enumerate(atyref):
            elem = gemmi.Element(atyrow[0])
            sftab[ind] = np.concatenate([elem.c4322.a, elem.c4322.b])

        fitted_sog = spherical.fit_sog(newfreq, extrapolated, sftab)
        eval_sog = spherical.eval_sog(fitted_sog[None], params["freqs"], None)
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
                        "function": {
                            "frequency": params["freqs"].tolist(),
                            "values": s,
                        },
                        "description": base64.b64encode(t).decode(),
                        "count": n,
                    }
                    for t, c, n, s in zip(
                        atyref,
                        fitted_sog.tolist(),
                        intparams["atycounts"].tolist(),
                        params["soln"].T.tolist(),
                    )
                    if t[0] != 255
                },
                f,
            )

    elif inpath.suffix == ".json":
        with open(inpath) as f:
            params = json.load(f)

        naty = len(params)
        atyref = np.empty((naty, 11), dtype=np.uint8)
        fitted_sog = np.empty((naty, 10))

        for ind, entry in enumerate(params.values()):
            atyref[ind] = np.frombuffer(
                base64.b64decode(entry["description"]), dtype=np.uint8
            )
            fitted_sog[ind, :5] = [c["a"] for c in entry["coefficients"]]
            fitted_sog[ind, 5:] = [c["b"] for c in entry["coefficients"]]

    if not (args.models and args.o):
        return

    for model_path, out_path in zip(args.models, args.o):
        print("loading", model_path)
        st = gemmi.read_structure(model_path)
        _, _, _, _, aty, _, _, atydesc = util.from_gemmi(st, nochangeh=True)
        atymap = util.align_aty(atyref, atydesc, approx=args.approx)
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


if __name__ == "__main__":
    main()
