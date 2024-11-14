from functools import partial

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu


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
def calc_cov(params, freqs):
    s1, s2 = jnp.meshgrid(freqs, freqs, indexing="xy")
    cov = jax.nn.softplus(params["scale"]) * jnp.exp(
        -jax.nn.softplus(params["length"]) * (s1 - s2) ** 2
    ) + 1e-6 * jnp.identity(len(freqs))
    return cov


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
def reconstruct(gaussians, weights, fbins, labels):
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
    fbins = fbins.ravel()
    f_c, _ = jax.lax.scan(
        one_shell,
        jnp.zeros(fcshape, dtype=complex),
        (weights, labels),
    )

    return f_c


@jax.jit
def solve(gaussians, mpdata, sigma_n, fbins, flabels, bin_cent):
    gaussians = gaussians.reshape(len(gaussians), -1)
    f_o = jnp.fft.rfftn(mpdata) / jnp.sqrt(sigma_n)
    mats = calc_mats(
        gaussians,
        fbins,
        flabels,
    )
    vecs = calc_vecs(f_o, gaussians, fbins, flabels)
    nshells, naty = vecs.shape
    scale = jnp.linalg.matrix_norm(mats)
    scale_sqrt = jnp.sqrt(scale)
    scale_mat = jnp.outer(scale_sqrt, scale_sqrt)
    mats_stacked, vecs_stacked = make_block_diagonal(
        (mats.T / scale).T, (vecs.T / scale_sqrt).T, nshells, naty
    )

    mll_fn = partial(
        calc_mll,
        mats_stacked=mats_stacked,
        vecs_stacked=vecs_stacked,
        freqs=bin_cent,
        scale=scale_mat,
        naty=naty,
    )

    solver = optax.lbfgs(
        linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=50)
    )
    init_params = {"scale": jnp.array(1.0), "length": jnp.array(1.0)}
    params = opt_loop(solver, mll_fn, init_params, 5000)
    jax.debug.print("params: {}", params)

    prior_cov = calc_cov(params, bin_cent)
    posterior_cov = (
        jnp.kron(jnp.linalg.inv(prior_cov) / scale_mat, jnp.identity(naty))
        + mats_stacked
    )

    cho = jax.scipy.linalg.solve_triangular(
        jnp.linalg.cholesky(posterior_cov),
        jnp.identity(nshells * naty),
        lower=True,
    )
    var = jnp.diag(cho.T @ cho)
    var = var.reshape(nshells, naty)
    var = (var.T / scale).T

    soln = jnp.linalg.solve(posterior_cov, vecs_stacked)
    soln = soln.reshape(nshells, naty)
    soln = (soln.T / scale_sqrt).T

    return soln, var


@partial(jax.jit, static_argnames=["naty"])
def calc_mll(
    params,
    mats_stacked,
    vecs_stacked,
    freqs,
    scale,
    naty,
):
    prior_cov = calc_cov(params, freqs)
    mll_cov = (
        jnp.kron(jnp.linalg.inv(prior_cov) / scale, jnp.identity(naty)) + mats_stacked
    )
    quad = vecs_stacked.T @ jnp.linalg.solve(mll_cov, vecs_stacked)
    _, logdet_perturbation = jnp.linalg.slogdet(mll_cov)
    _, logdet_cov = jnp.linalg.slogdet(prior_cov)
    loglik = logdet_perturbation + naty * logdet_cov - quad
    return loglik


def opt_loop(solver, objective, params, max_steps):
    @jax.jit
    def one_step(carry):
        params, opt_state = carry
        step = otu.tree_get(opt_state, "count")
        loss, grad = value_and_grad(params)
        updates, opt_state = solver.update(
            grad, opt_state, params, value=loss, grad=grad, value_fn=objective
        )
        params = optax.apply_updates(params, updates)
        jax.debug.print("loss {step}: {loss}", step=step, loss=loss)
        return params, opt_state

    @jax.jit
    def has_converged(carry):
        params, opt_state = carry
        step = otu.tree_get(opt_state, "count")
        lr = otu.tree_get(opt_state, "learning_rate")
        grad = otu.tree_get(opt_state, "grad")
        err = otu.tree_l2_norm(grad) * lr
        jax.debug.print("err {err}", err=err)
        return (step == 0) | ((step < max_steps) & (err >= 1e-6))

    opt_state = solver.init(params)
    value_and_grad = jax.value_and_grad(objective)
    params, _ = jax.lax.while_loop(has_converged, one_step, (params, opt_state))
    return params
