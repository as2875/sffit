from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optax.tree_utils as otu

from . import sampler


@jax.jit
def _mask_inner(freq_index, inner, fbins):
    msk = (fbins == freq_index).astype(int)
    return jnp.sum(inner * msk)


@jax.jit
def calc_mats(gaussians, fbins, labels):
    @jax.jit
    def one_element(mat_index):
        inner = gaussians[mat_index[0]] * gaussians[mat_index[1]].conj()
        inner_binned = jax.lax.map(
            partial(_mask_inner, inner=inner, fbins=fbins),
            labels,
        )
        return inner_binned

    naty = len(gaussians)
    gaussians = gaussians.reshape(naty, -1)
    fbins = fbins.ravel()

    colx, coly = jnp.triu_indices(naty)
    mat_indices = jnp.column_stack((colx, coly))
    mats1d = jax.lax.map(one_element, mat_indices).T
    mats2d = jnp.zeros((len(labels), naty, naty), dtype=complex)
    mats2d = mats2d.at[:, colx, coly].set(mats1d)
    mats2d = mats2d.at[:, coly, colx].set(mats1d)

    return 2 * mats2d.real


@jax.jit
def calc_vecs(f_o, gaussians, fbins, labels):
    @jax.jit
    def one_element(vec_index):
        inner = f_o * gaussians[vec_index].conj()
        inner_binned = jax.lax.map(
            partial(_mask_inner, inner=inner, fbins=fbins),
            labels,
        )
        return inner_binned

    naty = len(gaussians)
    f_o = f_o.ravel()
    gaussians = gaussians.reshape(naty, -1)
    fbins = fbins.ravel()
    vec_indices = jnp.arange(naty)

    vecs = jax.lax.map(one_element, vec_indices).T

    return 2 * vecs.real


@jax.jit
def calc_mats_and_vecs(gaussians, f_o, sigma_n, fbins, flabels):
    gaussians = gaussians.reshape(len(gaussians), -1)
    mats = calc_mats(
        gaussians,
        fbins,
        flabels,
    )
    vecs = calc_vecs(
        f_o / jnp.sqrt(sigma_n),
        gaussians,
        fbins,
        flabels,
    )
    return mats, vecs


def align_linsys(
    atyref,
    atynew,
    countsnew,
    refmats,
    refvecs,
    refcounts,
    reflabels,
    mats,
    vecs,
    dataind,
):
    search = jnp.all(atyref == atynew[:, None], axis=-1)
    _, align = jnp.nonzero(search, size=len(atynew))
    refvecs = refvecs.at[:, align].add(vecs)
    indsnew = jnp.indices((len(atynew), len(atynew)))
    indsref = align[indsnew]
    refmats = refmats.at[:, *indsref].add(mats)
    refcounts[align] += countsnew
    reflabels[align, dataind] = True

    return (
        refmats,
        refvecs,
        refcounts,
        reflabels,
    )


@jax.jit
def calc_cov_freq(params, freqs):
    s1, s2 = jnp.meshgrid(freqs, freqs, indexing="xy")
    parsp = jax.tree.map(jax.nn.softplus, params)
    cov = parsp["scale"] / (1 + parsp["beta"] * (s1**2 + s2**2))
    return cov


def calc_cov_aty(atydesc):
    naty = len(atydesc)
    alphabet = np.unique(atydesc)
    if alphabet[0] == 0:
        alphabet = alphabet[1:]
    counts = np.zeros((naty, len(alphabet)))

    for ind, elem in enumerate(alphabet):
        counts[:, ind] = np.count_nonzero(atydesc[:, 1:] == elem, axis=1)

    # if an atom has no bonds, the only subtree is the atom itself
    freeat = np.count_nonzero(counts, axis=1) == 0

    elemind = atydesc[freeat, 0]
    for ind in elemind:
        (alphind,) = np.nonzero(alphabet == ind)
        counts[freeat, alphind] = 1

    kern = np.empty((naty, naty))
    for i in range(naty):
        for j in range(i, naty):
            card = np.sum(np.minimum(counts[i], counts[j]))
            if atydesc[i, 0] == atydesc[j, 0]:
                kern[i, j] = 2**card
            else:
                kern[i, j] = 0

    inds = np.tril_indices(naty, k=-1)
    kern[inds] = kern.T[inds]
    diag = np.sqrt(np.diag(kern))
    kern /= np.outer(diag, diag)

    return kern


@partial(jax.jit, static_argnames=["nshells", "naty"])
def make_block_diagonal(mats, vecs, nshells, naty):
    block_inds = jnp.indices((naty, naty))
    shell_shifts = jnp.repeat(jnp.arange(nshells) * naty, naty**2)
    block_inds_flat = jnp.stack(
        [
            jnp.tile(block_inds[0].ravel(), nshells) + shell_shifts,
            jnp.tile(block_inds[1].ravel(), nshells) + shell_shifts,
        ]
    )
    dim = nshells * naty
    overlap_mat = jnp.zeros((dim, dim)).at[*block_inds_flat].add(mats.ravel())

    return overlap_mat, vecs.ravel()


@jax.jit
def reconstruct(gaussians, weights, sigma_n, fbins, labels):
    @jax.jit
    def one_shell(carry, tree):
        wtshell, freq_index = tree
        fcshell = jnp.where(
            fbins == freq_index,
            jnp.sum((wtshell * gaussians.T).T, axis=0),
            0,
        )
        new = carry + fcshell.reshape(fcshape)
        return new, None

    fcshape = fbins.shape
    gaussians *= jnp.sqrt(sigma_n)
    gaussians = gaussians.reshape(len(gaussians), -1)
    fbins = fbins.ravel()
    f_c, _ = jax.lax.scan(
        one_shell,
        jnp.zeros(fcshape, dtype=complex),
        (weights, labels),
    )
    v_c = jnp.fft.irfftn(f_c)

    return v_c


@partial(jax.jit, static_argnames=["co"])
def solve(mats, vecs, power, bin_cent, aty_cov, co, weight):
    include, exclude = slice(co, None), slice(None, co)
    nshells, naty = vecs[include].shape
    mats_stacked, vecs_stacked = make_block_diagonal(
        weight * mats[include], weight * vecs[include], nshells, naty
    )

    mll_fn = partial(
        calc_mll,
        mats_stacked=mats_stacked,
        vecs_stacked=vecs_stacked,
        aty_cov=aty_cov,
        freqs=bin_cent[include],
    )

    solver = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=50,
            initial_guess_strategy="one",
            verbose=False,
        ),
    )
    init_params = {
        "scale": jnp.array(1.0),
        "beta": jnp.array(1.0),
    }
    params = opt_loop(solver, mll_fn, init_params, 5000)

    soln, _ = _calc_posterior(
        params, mats_stacked, vecs_stacked, aty_cov, bin_cent[include]
    )
    soln = soln.reshape(nshells, naty)
    var = _calc_posterior_var(params, mats_stacked, aty_cov, bin_cent[include])
    var = var.reshape(nshells, naty)

    pred = eval_sf(bin_cent[exclude], bin_cent[include], soln, params)
    fcfc = jnp.einsum("...i,...ij,...j", pred, mats[exclude], pred)
    fofc = jnp.einsum("...i,...i", pred, vecs[exclude])
    loss = fofc.sum() / jnp.sqrt(fcfc.sum() * power[exclude].sum())

    return soln, var, params, loss


@jax.jit
def _calc_cov_kron(params, freqs, aty_cov):
    prior_cov = calc_cov_freq(params, freqs)
    cov_kron = jnp.kron(prior_cov, aty_cov)
    return cov_kron


@jax.jit
def _calc_posterior(params, mats_stacked, vecs_stacked, aty_cov, freqs):
    cov_kron = _calc_cov_kron(params, freqs, aty_cov)
    id_n = jnp.identity(len(freqs) * len(aty_cov))
    posterior_cov = id_n / 2 + mats_stacked @ cov_kron

    soln = jnp.linalg.solve(posterior_cov, vecs_stacked)
    soln = cov_kron @ soln
    _, logdet = jnp.linalg.slogdet(posterior_cov)

    return soln, logdet


@jax.jit
def _calc_posterior_var(params, mats_stacked, aty_cov, freqs):
    cov_kron = _calc_cov_kron(params, freqs, aty_cov)
    id_n = jnp.identity(len(freqs) * len(aty_cov))
    block_cov = id_n / 2 + cov_kron @ mats_stacked

    posterior_cov = jnp.linalg.solve(block_cov, cov_kron)
    posterior_var = jnp.diag(posterior_cov) / 2

    return posterior_var


@jax.jit
def calc_mll(params, mats_stacked, vecs_stacked, aty_cov, freqs):
    soln, logdet = _calc_posterior(params, mats_stacked, vecs_stacked, aty_cov, freqs)
    quad = jnp.vdot(vecs_stacked, soln)
    loglik = 0.5 * logdet - quad
    return loglik


@jax.jit
def eval_sf(s_test, s_train, sf_train, params):
    cov = calc_cov_freq(params, jnp.concat([s_train, s_test]))
    nbins = len(s_train)
    autocov = cov[:nbins, :nbins]
    crosscov = cov[nbins:, :nbins]

    y = jnp.linalg.solve(autocov, sf_train)
    sf_test = jnp.einsum("ij,...j", crosscov, y.T).T

    return sf_test


def opt_loop(solver, objective, params, max_steps):
    @jax.jit
    def one_step(carry):
        params, opt_state = carry
        loss, grad = value_and_grad(params, state=opt_state)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=loss, grad=grad, value_fn=objective
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state

    @jax.jit
    def has_converged(carry):
        params, opt_state = carry
        step = otu.tree_get(opt_state, "count")
        lr = otu.tree_get(opt_state, "learning_rate")
        grad = otu.tree_get(opt_state, "grad")
        err = otu.tree_l2_norm(grad) * lr
        return (step == 0) | ((step < max_steps) & (err >= 1e-6))

    opt_state = solver.init(params)
    value_and_grad = optax.value_and_grad_from_state(objective)
    params, _ = jax.lax.while_loop(has_converged, one_step, (params, opt_state))
    return params


@jax.jit
def sog_loss(params, freqs, target, is_monotonic):
    params_tr = jax.lax.cond(
        is_monotonic,
        jax.nn.softplus,
        sampler.transform_params,
        params,
    )
    sog_eval = sampler.eval_sog(
        params_tr[None, None],
        freqs,
        weights=None,
    )
    loss = jnp.mean((target - sog_eval.squeeze()) ** 2)
    return loss


@jax.jit
def fit_sog(freqs, soln, x0):
    @jax.jit
    def one_aty(tree):
        target, x0, is_monotonic = tree
        lossfn = partial(
            sog_loss, freqs=freqs, target=target, is_monotonic=is_monotonic
        )
        x0tr = jax.lax.cond(
            is_monotonic,
            lambda x: x + jnp.log(-jnp.expm1(-x)),
            sampler.inv_transform_params,
            x0,
        )
        solver = optax.lbfgs(
            linesearch=optax.scale_by_zoom_linesearch(
                max_linesearch_steps=50,
                initial_guess_strategy="one",
                verbose=False,
            ),
        )
        params = opt_loop(solver, lossfn, x0tr, 5000)
        params_tr = jax.lax.cond(
            is_monotonic,
            jax.nn.softplus,
            sampler.transform_params,
            params,
        )
        return params_tr

    is_monotonic = jnp.all(jnp.diff(soln, axis=0) <= 0, axis=0)
    fitted = jax.lax.map(one_aty, (soln.T, x0, is_monotonic))
    return fitted
