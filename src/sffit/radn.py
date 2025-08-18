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
from .spherical import opt_loop, _mask_inner


def calc_f_gemmi(st, nsamples, dmin):
    dmin = dmin - 1e-6
    with util.silence_stdout():
        asu = calc_fc_fft(st, d_min=dmin, source="electron")
    grid = asu.get_f_phi_on_grid((nsamples, nsamples, nsamples), half_l=True)
    return grid.array.conj()


def update_f_gemmi(f_calc, index, st, nsamples, dmin):
    f_new = calc_f_gemmi(st, nsamples, dmin)
    f_calc[index] = f_new
    return f_calc


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
    bins = bins.array.real.astype(int)
    bins[bins == 0] = bins.max() + 1
    bins[0, 0, 0] = 0
    _, bins = np.unique(bins, return_inverse=True)
    bins -= 1

    bdf = hkldata.binned_df
    bin_cent = 0.5 / bdf["d_min"] + 0.5 / bdf["d_max"]

    return bins, bin_cent.to_numpy(), hkldata.d_min_max()


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
    def one_coef(carry, tree):
        mats, counts = carry
        ind, coef = tree
        outer = jnp.outer(coef, coef.conj())
        mats = mats.at[ind].add(outer.real)
        counts = counts.at[ind].add(1)
        return (mats, counts), None

    nmaps, nbins = len(f_obs), len(labels)
    covmats = jnp.zeros((nbins, nmaps, nmaps))
    counts = jnp.zeros(nbins)
    (covmats, counts), _ = jax.lax.scan(
        one_coef, (covmats, counts), (fbins.ravel(), f_obs.reshape(nmaps, -1).T)
    )
    covmats = (covmats.T / (counts - 1)).T
    return 2 * covmats


@jax.jit
def calc_D(f_obs, f_calc, fbins, labels):
    @jax.jit
    def one_map(tree):
        fo, fc = tree
        prec_D1 = jnp.real(fo * fc.conj())
        covar = jax.lax.map(partial(_mask_inner, inner=prec_D1, fbins=fbins), labels)
        prec_D2 = jnp.abs(fc) ** 2
        var = jax.lax.map(partial(_mask_inner, inner=prec_D2, fbins=fbins), labels)
        return covar / var

    D = jax.lax.map(one_map, (f_obs, f_calc))
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
            verbose=False,
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
def calc_gamma(mats):
    return jnp.abs(mats).sum(axis=-1)


@partial(jax.jit, static_argnames=["rank"])
def calc_refn_objective(params, f_obs, f_calc, D, fbins, labels, freq, dose, rank):
    @jax.jit
    def one_bin(carry, tree):
        ind, mat1, mat2, D, gamma = tree
        scaled = (D * f_calc.T).T
        soln = (mat1.T / gamma).T @ f_obs - (mat2.T / gamma).T @ scaled + scaled
        carry = carry + soln.astype(jnp.complex64) * (fbins == ind).astype(int)
        return carry, None

    shape = f_obs.shape
    nmaps = len(dose)

    noise = jax.nn.softplus(params["noise"])
    cov_calc = calc_cov(params, freq, dose, noisewt=0.0)
    u, s, vh = jnp.linalg.svd(cov_calc, hermitian=True)

    s1 = s / (s.T + noise).T
    s1 = s1.at[..., :rank].set(1.0)
    s2 = s.at[..., rank:].set(jnp.inf)
    mat1 = jnp.matmul(u * s1[..., None, :], vh)
    mat2 = jnp.identity(nmaps) + (noise * jnp.matmul(vh.mT, u.mT / s2[..., None]).T).T
    gamma = calc_gamma(mat2)

    f_obs = f_obs.reshape(nmaps, -1)
    f_calc = f_calc.reshape(nmaps, -1)
    fbins = fbins.ravel()

    smoothed, _ = jax.lax.scan(
        one_bin,
        jnp.zeros_like(f_obs, dtype=jnp.complex64),
        (labels, mat1, mat2, D.T, gamma),
    )
    smoothed = smoothed.reshape(shape)
    return smoothed, cov_calc


@jax.jit
def calc_residuals(f_obs, f_calc, D, fbins):
    residuals = f_obs - D[:, fbins] * f_calc
    return residuals


@jax.jit
def make_friedel_mask(fbins):
    msk = jnp.ones_like(fbins)
    msk = msk.at[:, :, 0].set(0)  # exclude l=0
    msk = msk.at[1:150, :, 0].set(1)  # but include when l=0 and h>0
    msk = msk.at[0, :150, 0].set(1)  # or when l=0 and h=0 and k>=0
    return msk


@partial(jax.jit, static_argnames=["rank"])
def calc_elbo(params, f_obs, f_calc, D, fbins, freq, dose, rank):
    @jax.jit
    def one_coef(carry, tree):
        ind, coef, D, mskwt = tree
        loglik = mskwt * jnp.linalg.vector_norm(msqrt[ind] @ (D * coef)) ** 2
        return carry + loglik, None

    nmaps = len(dose)
    noise = jax.nn.softplus(params["noise"])
    cov_calc = calc_cov(params, freq, dose, noisewt=0.0)
    _, s, vh = jnp.linalg.svd(cov_calc, hermitian=True)
    s = s.at[..., rank:].set(jnp.inf)
    msqrt = vh / jnp.sqrt(s[..., None])
    msk = mask_extrema(make_friedel_mask(fbins), fbins)

    loglik = jnp.sum(
        msk * jnp.abs(calc_residuals(f_obs, f_calc, D, fbins)) ** 2 / noise[fbins]
    )
    logprior, _ = jax.lax.scan(
        one_coef,
        0.0,
        (
            fbins.ravel(),
            f_calc.reshape(nmaps, -1).T,
            D[:, fbins].reshape(nmaps, -1).T,
            msk.ravel(),
        ),
    )
    return -loglik - logprior


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
    mpdata = np.fft.irfftn(in_map.astype(jnp.complex128)) / fft_scale
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


def servalcat_run(
    cwd,
    map_path,
    model_path,
    index,
    dmin,
    weight,
    D,
    params,
    freq,
    dose,
    rank,
):
    noise = np.logaddexp(0, params["noise"])
    cov_calc = calc_cov(params, freq, dose, noisewt=0.0)
    u, s, vh = np.linalg.svd(cov_calc, hermitian=True)
    s[..., rank:] = np.inf
    gamma = calc_gamma(
        np.identity(len(dose)) + (noise * np.matmul(vh.mT, u.mT / s[..., None]).T).T
    )
    sigvar = 1 / gamma[:, index]

    LL_SPA.update_ml_params = lambda self: _servalcat_calc_D_and_S(
        self,
        D=D,
        S=sigvar,
        freq=freq,
    )
    LL_SPA.overall_scale = lambda *args, **kwargs: None

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
        "--hydrogen",
        "no",
        "--blur",
        "0",
        "--fix_xyz",
        "--adpr_weight",
        str(weight),
        "--occr_weight",
        str(weight),
        "-s",
        "electron",
        "--ncycle",
        "1",
    ]

    with contextlib.chdir(cwd):
        with util.silence_stdout():
            args = refine_spa.parse_args(cmdline)
            refine_spa.main(args)

    outpath = cwd / "refined.mmcif"
    return outpath


def scale_b(f_obs, f_calc, structures, nsamples, spacing):
    _, b_scale = jax.lax.map(
        lambda tree: calc_k_b(*tree, nsamples=nsamples, spacing=spacing),
        (f_obs, f_calc),
    )
    print(b_scale)
    structures = [shift_b(st, b) for st, b in zip(structures, b_scale)]
    return structures
