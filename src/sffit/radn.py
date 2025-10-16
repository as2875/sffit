# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import contextlib
from functools import partial

import gemmi
import jax
import jax.numpy as jnp
import numpy as np
import optax

from servalcat.refine import refine_spa
from servalcat.refine.spa import LL_SPA
from servalcat.utils import hkl
from servalcat.utils.model import calc_fc_fft

from . import util
from .dencalc import calc_k_b
from .spherical import _mask_inner, opt_loop


def calc_f_gemmi(st, nsamples, dmin):
    dmin = dmin - 1e-6
    with util.silence_stdout():
        asu = calc_fc_fft(st, d_min=dmin, source="electron")
    grid = asu.get_f_phi_on_grid((nsamples, nsamples, nsamples), half_l=True)
    return grid.array.conj()


def calc_f_gemmi_multiple(structures, nsamples, dmin):
    if nsamples % 2 == 0:
        f_calc = np.zeros(
            (len(structures), nsamples, nsamples, nsamples // 2 + 1), dtype=np.complex64
        )
    else:
        f_calc = np.zeros(
            (len(structures), nsamples, nsamples, (nsamples + 1) // 2),
            dtype=np.complex64,
        )

    for ind, st in enumerate(structures):
        f_calc[ind] = calc_f_gemmi(st, nsamples, dmin)

    return f_calc


def make_servalcat_bins(nsamples, spacing, dmin):
    cell_size = nsamples * spacing
    cell = gemmi.UnitCell(cell_size, cell_size, cell_size, 90, 90, 90)
    sg = gemmi.SpaceGroup("P 1")

    sf = gemmi.ReciprocalComplexGrid(
        np.zeros((nsamples, nsamples, nsamples), dtype=np.complex64),
        cell=cell,
        spacegroup=sg,
    )
    asu = sf.prepare_asu_data(dmin=dmin, with_000=False)
    with util.silence_stdout():
        hkldata = hkl.hkldata_from_asu_data(asu, label="")
        hkldata.setup_relion_binning()

    for i_bin, indices in hkldata.binned():
        asu.value_array[indices] = i_bin

    bins = asu.get_f_phi_on_grid((nsamples, nsamples, nsamples), half_l=True)
    friedel_mask = asu.get_f_phi_on_grid((nsamples, nsamples, nsamples), half_l=True)

    for point in friedel_mask:
        ih, ik, il = friedel_mask.to_hkl(point)
        if il == 0 and (ih <= 0) and not (ih == 0 and ik >= 0):
            point.value = 0
        else:
            point.value = 1

    bins = bins.array.real.astype(int)
    bins[bins == 0] = bins.max() + 1
    bins[0, 0, 0] = 0
    _, bins = np.unique(bins, return_inverse=True)
    bins -= 1

    friedel_mask = friedel_mask.array.real.astype(int)

    bdf = hkldata.binned_df
    bin_cent = 0.5 / bdf["d_min"] + 0.5 / bdf["d_max"]

    return bins, friedel_mask, bin_cent.to_numpy(), hkldata.d_min_max()


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
def calc_empirical_cov(f_obs, fbins, labels, friedel_mask):
    @jax.jit
    def one_coef(carry, tree):
        mats, counts = carry
        ind, coef = tree
        outer = jnp.outer(coef, coef.conj())
        mats = mats.at[ind].add(outer.real)
        counts = counts.at[ind].add(1)
        return (mats, counts), None

    nmaps, nbins = len(f_obs), len(labels)
    fbins = jnp.where((fbins == -1) | (friedel_mask == 0), nbins, fbins)
    covmats = jnp.zeros((nbins, nmaps, nmaps))
    counts = jnp.zeros(nbins, dtype=int)
    (covmats, counts), _ = jax.lax.scan(
        one_coef, (covmats, counts), (fbins.ravel(), f_obs.reshape(nmaps, -1).T)
    )
    covmats = (covmats.T / counts).T
    return covmats, counts


@jax.jit
def calc_posterior_cov(params, freq, dose):
    noise = jax.nn.softplus(params["noise"])
    cov_calc = calc_cov(params, freq, dose, noisewt=0.0)
    cov_calc_noise = calc_cov(params, freq, dose, noisewt=1.0)
    cov_posterior = (noise * jnp.linalg.solve(cov_calc_noise, cov_calc).T).T
    return cov_posterior


@jax.jit
def calc_residual_cov(f_smoothed, f_calc, D, fbins, labels, friedel_mask):
    residuals = calc_residuals(f_smoothed, f_calc, D, fbins)
    rescov, _ = calc_empirical_cov(residuals, fbins, labels, friedel_mask)
    return rescov


@jax.jit
def calc_variational_cov(cov_post, cov_res, obscounts):
    _, nmaps, _ = cov_post.shape
    cov_tot = cov_post + cov_res
    u, s, vh = jnp.linalg.svd(cov_tot, hermitian=True)
    lam = s[..., 0]
    trace = jnp.trace(cov_tot, axis1=1, axis2=2)
    alpha = (trace - lam) / (nmaps - 1)

    _, logdet_post = jnp.linalg.slogdet(cov_post)
    logdet_var = jnp.log(alpha + lam) + (nmaps - 1) * jnp.log(alpha)
    kldiv = jnp.sum(
        obscounts
        * (
            trace / alpha
            - lam**2 / (alpha * (alpha + lam))
            + logdet_var
            - logdet_post
            - nmaps
        )
    )
    return u[..., 0], alpha, lam, kldiv


@jax.jit
def calc_D(f_obs, f_calc, fbins, labels, friedel_mask):
    @jax.jit
    def one_map(tree):
        fo, fc = tree
        prec_D1 = jnp.real(fo * fc.conj())
        covar = jax.lax.map(partial(_mask_inner, inner=prec_D1, fbins=fbins), labels)
        prec_D2 = jnp.abs(fc) ** 2
        var = jax.lax.map(partial(_mask_inner, inner=prec_D2, fbins=fbins), labels)
        return covar / var

    fbins = jnp.where((fbins == -1) | (friedel_mask == 0), len(labels), fbins)
    D = jax.lax.map(one_map, (f_obs, f_calc))
    return D


@jax.jit
def calc_hyperparams(f_obs, fbins, labels, friedel_mask, freq, dose):
    nbins = len(freq)
    init_params = {
        "a": jnp.array(1.0),
        "b": jnp.array(1.0),
        "alpha": jnp.array(1.0),
        "beta": jnp.array(1.0),
        "power": jnp.ones(nbins),
        "noise": jnp.ones(nbins),
    }

    cov_emp, obscounts = calc_empirical_cov(f_obs, fbins, labels, friedel_mask)
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
            verbose=False,
        ),
    )
    params = opt_loop(solver, mll_fn, init_params, 5000)

    parsp = jax.tree.map(jax.nn.softplus, params)
    parsp["power"] *= norm
    parsp["noise"] *= norm
    parscaled = jax.tree.map(lambda x: x + jnp.log(-jnp.expm1(-x)), parsp)

    return parscaled, obscounts


@jax.jit
def calc_mll(params, cov_emp, freq, dose, obscounts):
    cov_calc = calc_cov(params, freq, dose)
    _, logdet = jnp.linalg.slogdet(cov_calc)
    prod = jnp.linalg.solve(cov_calc, cov_emp)
    loss = jnp.sum(obscounts * (logdet + jnp.trace(prod, axis1=1, axis2=2)))
    return loss


@partial(jax.jit, donate_argnames=["data"])
def mask_extrema(data, fbins):
    msk = jnp.astype((fbins != fbins.min()) & (fbins != fbins.max()), int)
    data *= msk
    return data


@jax.jit
def smooth_maps(params, f_obs, fbins, labels, freq, dose):
    @jax.jit
    def one_bin(carry, tree):
        ind, cov_calc, cho_fac = tree
        rhs = cov_calc @ f_obs
        soln = jax.scipy.linalg.cho_solve((cho_fac, is_lower), rhs)
        carry = carry + soln.astype(jnp.complex64) * (fbins == ind).astype(int)
        return carry, None

    cov_calc_noise = calc_cov(params, freq, dose)
    cov_calc = calc_cov(params, freq, dose, noisewt=0.0)
    cho_fac, is_lower = jax.scipy.linalg.cho_factor(cov_calc_noise)

    shape = f_obs.shape
    nmaps = len(dose)
    f_obs = f_obs.reshape(nmaps, -1)
    fbins = fbins.ravel()

    smoothed, _ = jax.lax.scan(
        one_bin,
        jnp.zeros_like(f_obs, dtype=jnp.complex64),
        (labels, cov_calc, cho_fac),
    )
    smoothed = smoothed.reshape(shape)
    return smoothed


@jax.jit
def calc_refn_objective(f_smoothed, f_calc, D, fbins, labels, vecs, alpha, lam):
    @jax.jit
    def one_bin(carry, tree):
        ind, vec, alpha, lam = tree
        proj = jnp.einsum("i,i...", vec, residuals).reshape(1, -1) * vec.reshape(-1, 1)
        soln = f_smoothed - lam / (alpha + lam) * proj
        carry = carry + soln.astype(jnp.complex64) * (fbins == ind).astype(int)
        return carry, None

    shape = f_smoothed.shape
    nmaps = shape[0]

    f_smoothed = f_smoothed.reshape(nmaps, -1)
    f_calc = f_calc.reshape(nmaps, -1)
    fbins = fbins.ravel()
    residuals = calc_residuals(f_smoothed, f_calc, D, fbins)

    smoothed, _ = jax.lax.scan(
        one_bin,
        jnp.zeros_like(f_smoothed, dtype=jnp.complex64),
        (labels, vecs, alpha, lam),
    )
    smoothed = smoothed.reshape(shape)
    return smoothed


@jax.jit
def calc_residuals(f_obs, f_calc, D, fbins):
    residuals = f_obs - D[:, fbins] * f_calc
    return residuals


@jax.jit
def calc_overall_scale(f_obs, f_calc, D, fbins, friedel_mask, sigvar):
    msk = mask_extrema(friedel_mask, fbins)
    residuals = calc_residuals(f_obs, f_calc, D, fbins)
    resvar = jnp.sum(
        msk * jnp.abs(residuals) ** 2 / sigvar[fbins], axis=(1, 2, 3)
    ) / jnp.count_nonzero(msk)
    scale = 1 / jnp.sqrt(resvar)
    return scale


def shift_b(st, b_scale):
    u_scale = b_scale / (8 * np.pi**2)
    for cra in st[0].all():
        cra.atom.b_iso += b_scale
        if cra.atom.aniso.nonzero():
            cra.atom.aniso.u11 += u_scale
            cra.atom.aniso.u22 += u_scale
            cra.atom.aniso.u33 += u_scale

    return st


def servalcat_setup_input(
    path,
    in_map,
    in_model,
    bsize,
    spacing,
    fft_scale,
):
    # write model
    in_model.setup_entities()
    out_path_st = path / "input_model.cif"
    in_model.make_mmcif_document().write_file(str(out_path_st))

    # write map
    mpdata = (
        np.fft.irfftn(in_map.astype(jnp.complex128), s=(bsize, bsize, bsize))
        / fft_scale
    )
    out_path_map = path / "input_map.mrc"
    util.write_map(mpdata, str(out_path_map), bsize, bsize * spacing)

    return out_path_map, out_path_st


def _servalcat_calc_D_and_S(self, D, S):
    bdf = self.hkldata.binned_df
    bdf["D"] = 0.0
    bdf["S"] = 0.0

    for ind, (i_bin, _) in enumerate(self.hkldata.binned()):
        bdf.loc[i_bin, "D"] = D[ind]
        bdf.loc[i_bin, "S"] = S[ind]


def servalcat_run(
    cwd,
    map_path,
    model_path,
    step,
    dmin,
    D,
    weight,
    sigvar,
):
    wtheur = np.exp(-1.7588 + dmin * 0.6311)
    weight *= wtheur

    LL_SPA.update_ml_params = lambda self: _servalcat_calc_D_and_S(
        self,
        D=D,
        S=sigvar,
    )
    LL_SPA.overall_scale = lambda *_: None

    ncycle = 10 if step == 0 else 5
    prefix = f"refined_{step:02d}"
    cmdline = [
        "--map",
        str(map_path),
        "--model",
        str(model_path),
        "--resolution",
        str(dmin),
        "--ncsr",
        "--no_mask",
        "--no_trim",
        "--blur",
        "0",
        "--weight",
        str(weight),
        "-s",
        "electron",
        "--hydrogen",
        "yes",
        "--ncycle",
        str(ncycle),
        "-o",
        prefix,
    ]

    with contextlib.chdir(cwd):
        with util.silence_stdout():
            args = refine_spa.parse_args(cmdline)
            refine_spa.main(args)

    outpath = (cwd / prefix).with_suffix(".mmcif")
    return outpath


def scale_b(f_obs, f_calc, fbins, friedel_mask, structures, nsamples, spacing):
    msk = mask_extrema(friedel_mask, fbins)
    k_scale, b_scale = jax.lax.map(
        lambda tree: calc_k_b(*tree, nsamples=nsamples, spacing=spacing),
        (msk * f_obs, msk * f_calc),
    )
    structures = [shift_b(st, b) for st, b in zip(structures, b_scale)]
    return structures, k_scale
