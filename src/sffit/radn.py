import contextlib
from functools import partial

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import optax
from servalcat.refine import refine_spa
from servalcat.refine.spa import LL_SPA

from . import util
from .spherical import opt_loop, _mask_inner


def calc_den_gemmi(st, it92, bounds, nsamples):
    dc = gemmi.DensityCalculatorE()
    dc.grid.set_size(nsamples, nsamples, nsamples)
    dc.grid.set_unit_cell(gemmi.UnitCell(*bounds.T[1], 90, 90, 90))
    dc.put_model_density_on_grid(st[0])
    return dc.grid.array


def update_f_gemmi(f_calc, index, st, it92, bounds, nsamples):
    f_new = calc_den_gemmi(st, it92, bounds, nsamples)
    f_calc = f_calc.at[index].set(jnp.fft.rfftn(f_new))
    return f_calc


def calc_f_gemmi_multiple(structures, it92, bounds, nsamples):
    if nsamples % 2 == 0:
        f_calc = np.empty(
            (len(structures), nsamples, nsamples, nsamples // 2 + 1), dtype=np.complex64
        )
    else:
        f_calc = np.empty(
            (len(structures), nsamples, nsamples, (nsamples + 1) // 2),
            dtype=np.complex64,
        )

    for ind, st in enumerate(structures):
        den = calc_den_gemmi(st, it92, bounds, nsamples)
        # use np.fft for consistency
        f_calc[ind] = np.fft.rfftn(den)

    return f_calc


@jax.jit
def _reciprocal_pow(x, s, scale, order):
    return 1 / (1 + scale * s**2 * x) ** order


@jax.jit
def calc_cov(params, freq, dose, noisewt=1.0):
    @jax.jit
    def one_bin(tree):
        power, noise, s2 = tree
        mat = (
            power
            * _reciprocal_pow(t1**2 + t2**2, s2, parsp["a"], parsp["alpha"])
            * _reciprocal_pow(t1 + t2, 1, parsp["b"], parsp["beta"])
        )
        return mat + noisewt * noise * jnp.identity(len(mat))

    parsp = jax.tree.map(jax.nn.softplus, params)
    t1, t2 = jnp.meshgrid(dose, dose, indexing="xy")
    covmats = jax.lax.map(
        one_bin, (parsp["power"], parsp["noise"], freq), batch_size=64
    )
    return covmats


@jax.jit
def calc_empirical_cov(f_obs, fbins, labels):
    @jax.jit
    def one_bin(ind):
        msk = (fbins == ind).astype(int)
        cov = jnp.cov(f_obs.reshape(nmaps, -1), fweights=msk.ravel())
        return cov.real

    nmaps = f_obs.shape[0]
    covmats = jax.lax.map(one_bin, labels)
    return covmats


@jax.jit
def calc_D(f_obs, f_calc, fbins, labels):
    prec_D1 = jnp.real(f_obs * f_calc.conj())
    covar = jax.lax.map(partial(_mask_inner, inner=prec_D1, fbins=fbins), labels)
    prec_D2 = jnp.abs(f_calc) ** 2
    var = jax.lax.map(partial(_mask_inner, inner=prec_D2, fbins=fbins), labels)
    D = covar / var
    return D


@jax.jit
def calc_hyperparams(f_obs, fbins, labels, freq, dose):
    nbins = len(freq)
    init_params = {
        "a": jnp.array(1.0),
        "b": jnp.array(1.0),
        "alpha": jnp.array(1.0),
        "beta": jnp.array(1.0),
        "power": jnp.ones(nbins),
        "noise": jnp.ones(nbins),
    }

    nlab = nbins + 2
    _, obscounts = jnp.unique(fbins, return_counts=True, size=nlab)
    obscounts = obscounts[1:-1]

    cov_emp = calc_empirical_cov(f_obs, fbins, labels)
    norm = jnp.linalg.matrix_norm(cov_emp)
    cov_emp = (cov_emp.T / norm).T

    mll_fn = partial(
        calc_mll,
        cov_emp=cov_emp,
        freq=freq,
        dose=dose,
        obscounts=obscounts,
    )

    solver = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=50,
            initial_guess_strategy="one",
            verbose=True,
        ),
    )
    params = opt_loop(solver, mll_fn, init_params, 5000)

    parsp = jax.tree.map(jax.nn.softplus, params)
    parsp["power"] *= norm
    parsp["noise"] *= norm
    parscaled = jax.tree.map(lambda x: x + jnp.log(-jnp.expm1(-x)), parsp)

    return parscaled


@jax.jit
def calc_mll(params, cov_emp, freq, dose, obscounts):
    cov_calc = calc_cov(params, freq, dose)
    _, logdet = jnp.linalg.slogdet(cov_calc)
    prod = jnp.linalg.solve(cov_calc, cov_emp)
    loss = jnp.sum(obscounts * (logdet + jnp.trace(prod, axis1=1, axis2=2)))
    reg = jnp.sum((params["power"][None, :] - params["power"][:, None]) ** 2)
    return loss + reg


@partial(jax.jit, donate_argnames=["data"])
def mask_extrema(data, fbins):
    msk = jnp.astype((fbins != fbins.min()) & (fbins != fbins.max()), int)
    data *= msk
    return data


@jax.jit
def smooth_maps(params, f_obs, f_calc, D, fbins, freq, dose):
    @jax.jit
    def one_coef(tree):
        ind, coef_obs, coef_calc = tree
        vec = cov_calc[ind] @ coef_obs + noise[ind] * D[ind] * coef_calc
        soln = jax.scipy.linalg.cho_solve((cho_fac[ind], is_lower), vec)
        return soln

    noise = jax.nn.softplus(params["noise"])
    cov_calc_noise = calc_cov(params, freq, dose)
    cov_calc = calc_cov(params, freq, dose, noisewt=0.0)
    cho_fac, is_lower = jax.scipy.linalg.cho_factor(cov_calc_noise)

    nmaps = len(dose)
    smoothed = jax.lax.map(
        one_coef,
        (
            fbins.ravel(),
            f_obs.reshape(nmaps, -1).T,
            f_calc.reshape(nmaps, -1).T,
        ),
        batch_size=4096,
    )
    smoothed = smoothed.T.reshape(f_obs.shape)
    return smoothed


@jax.jit
def calc_residuals(f_obs, f_calc, D, fbins):
    @jax.jit
    def one_map(tree):
        fo, fc = tree
        return fo - D_gr * fc

    D_gr = D[fbins]
    residuals = jax.lax.map(one_map, (f_obs, f_calc))
    return residuals


@jax.jit
def calc_refn_objective(index, smoothed, residuals, fbins, params, freq, dose):
    @jax.jit
    def one_map(carry, tree):
        res, wt = tree
        wt_gr = wt[fbins]
        new = carry + res * wt_gr
        return new, None

    cov_calc = calc_cov(params, freq, dose, noisewt=0.0)
    cov_inv = jnp.linalg.pinv(cov_calc, hermitian=True)
    cov_weights = cov_inv[:, index].T
    cov_weights /= cov_weights[index]
    res_sum, _ = jax.lax.scan(
        one_map,
        smoothed[index],
        (residuals, cov_weights),
    )
    return res_sum


def servalcat_setup_input(path, in_map, in_model, bsize, spacing):
    # write model
    in_model.setup_entities()
    out_path_st = path / "input_model.cif"
    in_model.make_mmcif_document().write_file(str(out_path_st))

    # write map
    mpdata = np.fft.irfftn(in_map)
    out_path_map = path / "input_map.mrc"
    util.write_map(mpdata, str(out_path_map), bsize, bsize * spacing)

    return out_path_map, out_path_st


def _servalcat_calc_D_and_S(self, D, S, freq):
    bdf = self.hkldata.binned_df
    bin_cent = 0.5 / bdf["d_max"] + 0.5 / bdf["d_min"]

    D_interp = np.interp(bin_cent, freq, D)
    S_interp = np.interp(bin_cent, freq, S)

    bdf["D"] = 0.0
    bdf["S"] = 0.0

    for ind, (i_bin, _) in enumerate(self.hkldata.binned()):
        bdf.loc[i_bin, "D"] = D_interp[ind]
        bdf.loc[i_bin, "S"] = S_interp[ind]


def servalcat_run(cwd, map_path, model_path, index, spacing, D, params, freq, dose):
    cov_calc = calc_cov(params, freq, dose, noisewt=0.0)
    cov_inv = jnp.linalg.pinv(cov_calc, hermitian=True)
    sigvar = cov_inv[:, index, index]

    LL_SPA.update_ml_params = lambda self: _servalcat_calc_D_and_S(
        self,
        D=D,
        S=sigvar,
        freq=freq,
    )

    cmdline = [
        "--model",
        str(model_path),
        "--map",
        str(map_path),
        "--resolution",
        str(2 * spacing),
        "--ncsr",
        "--fix_xyz",
        "--hydrogen",
        "all",
        "--hout",
        "--ncycle",
        "1",
    ]

    with contextlib.chdir(cwd):
        args = refine_spa.parse_args(cmdline)
        with util.silence_stdout():
            refine_spa.main(args)

    outpath = cwd / "refined.mmcif"
    return outpath
