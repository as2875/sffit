import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import gemmi
import optax

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
    parser_sample.add_argument(
        "--noml", action="store_true", help="do not estimate D, S"
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
        "--exclude",
        metavar="SELECTION",
        help="atoms to exclude from form factor calculations (GEMMI selection syntax)",
    )
    parser_ml.add_argument(
        "--contract",
        action="store_true",
        help="compute optimal contraction of basis",
    )
    parser_ml.set_defaults(func=do_ml)

    for sp in (parser_sample, parser_ml):
        sp.add_argument("--map", metavar="FILE", required=True, help="input map")
        sp.add_argument("--model", metavar="FILE", required=True, help="input model")
        sp.add_argument("--mask", metavar="FILE", help="input mask")

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
    st_aty = gemmi.read_structure(args.model)
    coords, it92_init, umat, occ, aty, _, atycounts, atnames, atydesc, unq_ind = (
        util.from_gemmi(st, st_aty, typing="identity")
    )
    naty = len(atycounts)

    print(f"{naty} atom types identified")

    if args.im:
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
            aty=aty,
            D=D_gr,
            sigma_n=sg_n_gr,
            data_size=len(inds1d),
        )
        logprior = partial(
            sampler.logprior_fn,
            means=it92_init,
        )
        logden = jax.jit(
            lambda params, batch: loglik(params, batch) + logprior(params),
        )
        grad_fn = jax.grad(logden)

        prec_mat = dencalc.calc_hess(
            coords,
            umat,
            occ,
            aty,
            it92_init,
            naty,
            D,
            sigma_n,
            bin_cent,
        )
        prec_fn = jax.jit(lambda *_: prec_mat)

        # sample with SGLD
        print("sampling")

        sgld = sampler.prec_sgld(grad_fn, prec_fn)
        step_size = sampler.scheduler(jnp.arange(args.nsamples) + 1)
        it92_samples = sampler.inference_loop(
            sample_key,
            jax.jit(sgld.step),
            batched,
            it92_init,
        )
        params = {"it92": it92_samples, "steps": step_size}

        print("saving parameters")
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
        params_tr = jnp.concatenate(
            [
                params["it92"][..., :5],
                params["it92"][..., 5:] ** 2,
            ],
            axis=-1,
        )
        step_size = params["steps"][args.nwarm :]
        params_post = jnp.average(params_tr[args.nwarm :], axis=0, weights=step_size)

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

    rng_key = jax.random.key(int(time.time()))

    print("loading data")
    st = gemmi.read_structure(args.model)
    st_aty = gemmi.read_structure(args.model)
    coords, it92, umat, occ, aty, atmask, atycounts, _, atydesc, unq_id = (
        util.from_gemmi(
            st, st_aty, selection=args.exclude, b_iso=False, typing="number"
        )
    )
    naty = len(atycounts)

    mpgrid, mpdata, fft_scale, bsize, spacing, bounds = util.read_mrc(
        args.map, args.mask
    )
    rcut = dencalc.calc_rcut(args.rcut, spacing)
    excluded = dencalc.calc_v_sparse(
        coords[~atmask],
        umat[~atmask],
        occ[~atmask],
        aty[~atmask],
        it92,
        rcut,
        bounds,
        bsize,
    )
    mpdata -= excluded

    freqs, fbins, bin_cent = dencalc.make_bins(
        mpdata, bsize, spacing, 1 / args.d, args.nbins
    )
    flabels = jnp.arange(args.nbins)

    gaussians = dencalc.calc_gaussians_direct(
        coords[atmask], umat[atmask], occ[atmask], aty[atmask], freqs, naty, fft_scale
    )
    f_obs = jnp.fft.rfftn(mpdata)
    f_obs = f_obs.at[0, 0, 0].set(0.0)
    pspec = dencalc.calc_power(f_obs, fbins, flabels)

    objective = partial(
        spherical.calc_loss,
        gaussians=gaussians,
        f_o=f_obs,
        fbins=fbins,
        flabels=flabels,
    )

    if args.contract:
        solver = optax.lbfgs(
            linesearch=optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=15,
            )
        )
        params = jax.random.normal(rng_key, (5, naty))
        params = spherical.opt_loop(solver, objective, params, 1000)
    else:
        params = jnp.identity(naty)

    _, soln, mats, vecs = spherical.solve(
        params,
        gaussians,
        f_obs,
        fbins,
        flabels,
    )
    transformed = jnp.einsum("ij,...j", params.T, soln)

    jnp.savez(
        args.o,
        soln=transformed,
        mats=mats,
        vecs=vecs,
        contr=params,
        freqs=bin_cent,
        aty=atydesc[unq_id],
        atycounts=atycounts,
        power=pspec,
    )


if __name__ == "__main__":
    main()
