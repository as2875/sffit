import argparse
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.stats as stats
import blackjax

import gemmi
import numpy as np

from datetime import date

rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))


def make_grid(bounds, nsamples):
    axis = jnp.linspace(bounds[:, 0], bounds[:, 1], nsamples, axis=-1, endpoint=False)
    return jnp.stack(
        jnp.meshgrid(axis[0], axis[1], axis[2]),
    )


@jax.jit
def one_coef(a, b, sigma, dist):
    t = 4 * jnp.pi / (b + sigma)
    den = a * t**1.5 * jnp.exp(-t * dist**2 * jnp.pi)
    return den


one_coef_vmap = jax.vmap(one_coef, in_axes=[0, 0, None, None])


def _calc_v_atom(carry, tree, bounds, nsamples):
    coord, it92, weight, sigma = tree
    mgrid = make_grid(bounds, nsamples)
    dist = jnp.sqrt(
        (mgrid[0] - coord[0]) ** 2
        + (mgrid[1] - coord[1]) ** 2
        + (mgrid[2] - coord[2]) ** 2
    )
    v_at = weight * one_coef_vmap(
        it92[:5],
        it92[5:],
        sigma,
        dist,
    ).sum(axis=0)

    return v_at + carry, None


def calc_v(coords, it92, weights, sigma, bounds, nsamples):
    calcv_part = partial(_calc_v_atom, bounds=bounds, nsamples=nsamples)
    v_mol, _ = jax.lax.scan(
        calcv_part,
        jnp.zeros((nsamples, nsamples, nsamples)),
        (coords, it92, weights, sigma),
    )

    return v_mol


def loglik_fn(params, data, coords, it92, sigma_n, bounds, nsamples):
    weights, sigma = jnp.exp(params["weights"]), jnp.exp(params["sigma"])
    v_mol = calc_v(coords, it92, weights, sigma, bounds, nsamples)
    v_mol = jnp.swapaxes(v_mol, 0, 1)

    logpdf = stats.norm.logpdf(data - v_mol, 0, sigma_n)

    return jnp.sum(logpdf)


def logprior_fn(params, scale_wt, scale_sg):
    weights, sigma = params["weights"], params["sigma"]
    logpdf_wt = stats.norm.logpdf(weights, loc=0, scale=scale_wt)
    logpdf_sg = stats.norm.logpdf(sigma, loc=0, scale=scale_sg)

    return jnp.sum(logpdf_wt) + jnp.sum(logpdf_sg)


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True, help="input map")
    parser.add_argument("--model", required=True, help="input model")
    parser.add_argument("-om", required=True, help="output map")

    pgroup = parser.add_mutually_exclusive_group()
    pgroup.add_argument("--params", help=".npz file with parameters")
    pgroup.add_argument("-op", help="output .npz with parameters")

    args = parser.parse_args()

    # load observations
    ccp4 = gemmi.read_ccp4_map(args.map)
    mpdata = ccp4.grid.array
    st = gemmi.read_structure(args.model)

    n_atoms = st[0].count_atom_sites()
    coords = np.empty((n_atoms, 3))
    Z = np.empty(n_atoms, dtype=int)
    it92 = np.empty((n_atoms, 10))

    for ind, cra in enumerate(st[0].all()):
        Z[ind] = cra.atom.element.atomic_number
        coords[ind] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
        it92[ind] = np.concatenate([cra.atom.element.c4322.a, cra.atom.element.c4322.b])

    assert ccp4.grid.nu == ccp4.grid.nv == ccp4.grid.nw, "Only cubic boxes are supported"
    bounds = jnp.array(
        [
            [0, ccp4.grid.nu * ccp4.grid.spacing[0]],
            [0, ccp4.grid.nv * ccp4.grid.spacing[1]],
            [0, ccp4.grid.nw * ccp4.grid.spacing[2]],
        ]
    )
    nsamples = ccp4.grid.nu

    if not args.params:
        loglik = partial(
            loglik_fn,
            data=mpdata,
            coords=coords,
            it92=it92,
            sigma_n=1.0,
            bounds=bounds,
            nsamples=nsamples,
        )
        logprior = partial(logprior_fn, scale_wt=1.0, scale_sg=1.0)
        logden = jax.jit(lambda x: loglik(x) + logprior(x))

        rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
        init_params = {"weights": jnp.zeros(n_atoms), "sigma": jnp.zeros(n_atoms)}

        warmup = blackjax.window_adaptation(blackjax.nuts, logden)
        (init_states, tuned_params), _ = warmup.run(warmup_key, init_params, 1000)
        kernel = blackjax.nuts(logden, **tuned_params).step
        states = inference_loop(sample_key, jax.jit(kernel), init_states, 10000)

        mcmc_samples = states.position
        jnp.savez(args.op, **mcmc_samples)
    else:
        mcmc_samples = jnp.load(args.params)

    wt_post = jnp.exp(mcmc_samples["weights"]).mean(axis=0)
    sg_post = jnp.exp(mcmc_samples["sigma"]).mean(axis=0)
    v_approx = jnp.swapaxes(
        calc_v(coords, it92, wt_post, sg_post, bounds, nsamples), 0, 1
    )

    result_map = gemmi.Ccp4Map()
    result_map.grid = gemmi.FloatGrid(np.array(v_approx, dtype=np.float32))
    result_map.grid.copy_metadata_from(ccp4.grid)
    result_map.update_ccp4_header()
    result_map.write_ccp4_map(args.om)
