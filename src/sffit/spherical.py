from functools import partial

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu
from jax.experimental import sparse
from jax.scipy.sparse.linalg import cg


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

    return mats2d


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

    return vecs


@partial(jax.jit, static_argnames=["nshells", "naty"])
def lstsq(mats, vecs, alpha, nshells, naty):
    mats_diags = jnp.trace(mats, axis1=1, axis2=2)
    mats_norm = (mats.T / mats_diags).T

    block_inds = jnp.indices((naty, naty))
    shell_shifts = jnp.repeat(jnp.arange(nshells) * naty, naty**2)
    block_inds_flat = jnp.column_stack(
        [
            jnp.tile(block_inds[0].ravel(), nshells) + shell_shifts,
            jnp.tile(block_inds[1].ravel(), nshells) + shell_shifts,
        ]
    )
    dim = nshells * naty
    overlap_mat = sparse.BCOO((mats_norm.ravel(), block_inds_flat), shape=(dim, dim))

    fd_stencil = jnp.concatenate(
        [
            jnp.full(dim, -2),
            jnp.full(dim - naty, 1),
            jnp.full(dim - naty, 1),
        ]
    )
    diag_inds = jnp.diag_indices(dim)
    diag_inds_flat = jnp.column_stack(
        [
            diag_inds[0].ravel(),
            diag_inds[1].ravel(),
        ]
    )
    diag_inds_trunc = diag_inds_flat[:-naty]
    fd_inds = jnp.concatenate(
        [
            diag_inds_flat,
            diag_inds_trunc.at[:, 0].add(naty),
            diag_inds_trunc.at[:, 1].add(naty),
        ]
    )
    fd_mat = sparse.BCOO((fd_stencil, fd_inds), shape=(dim, dim))

    A = overlap_mat + alpha * fd_mat.T @ fd_mat
    b = jnp.ravel((vecs.T / mats_diags).T)
    soln, _ = cg(A, b)
    soln_scaled = soln.reshape(nshells, naty)

    return soln_scaled


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
def solve(contract, gaussians, f_o, fbins, flabels):
    contracted = contract @ gaussians
    mats = calc_mats(
        contracted,
        fbins,
        flabels,
    )
    vecs = calc_vecs(f_o, contracted, fbins, flabels)
    soln = lstsq(mats.real, vecs.real, 1e-1, *vecs.shape)
    estimated = reconstruct(contracted, soln, fbins, flabels)

    return estimated, soln, mats, vecs


@jax.jit
def calc_loss(contract, gaussians, f_o, fbins, flabels):
    f_o = f_o * (fbins != -1).astype(int)
    estimated, soln, mats, vecs = solve(contract, gaussians, f_o, fbins, flabels)

    rec_err = jnp.sqrt(jnp.mean(jnp.abs(f_o - estimated) ** 2))
    cond_penalty = 1e-3 * jnp.sum(jnp.log(jnp.linalg.cond(mats)))
    loss = rec_err + cond_penalty

    return loss


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
        return (step == 0) | ((step < max_steps) & (lr >= 1e-15))

    opt_state = solver.init(params)
    value_and_grad = jax.value_and_grad(objective)
    params, _ = jax.lax.while_loop(has_converged, one_step, (params, opt_state))
    return params
