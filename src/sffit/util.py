import os
import sys
import warnings
from collections import defaultdict
from enum import Enum
from itertools import repeat

import gemmi
import jax.numpy as jnp
import numpy as np

from .spherical import calc_cov_aty


class InferenceMethod(Enum):
    MCMC = 1
    GP = 2

    @classmethod
    def from_npz(cls, npz):
        if "it92" in npz.keys():
            return cls(1)
        elif "soln" in npz.keys():
            return cls(2)
        else:
            raise TypeError("Unable to identify inference method")


def read_mrc(map_path, mask_path=None):
    ccp4 = gemmi.read_ccp4_map(map_path)
    mpdata = np.array(ccp4.grid.array)
    if mask_path:
        msk = gemmi.read_ccp4_map(mask_path)
        mskdata = np.array(msk.grid.array)
    else:
        mskdata = np.ones_like(mpdata)

    masked = mpdata * mskdata

    assert ccp4.grid.nu == ccp4.grid.nv == ccp4.grid.nw, (
        "Only cubic boxes are supported"
    )
    assert ccp4.grid.spacing[0] == ccp4.grid.spacing[1] == ccp4.grid.spacing[2], (
        "Only cubic boxes are supported"
    )

    fft_scale = ccp4.grid.unit_cell.volume / ccp4.grid.point_count

    bsize = ccp4.grid.nu
    spacing = ccp4.grid.spacing[0]
    bounds = jnp.array([[0, bsize * spacing] for i in range(3)])

    return ccp4.grid, masked, fft_scale, bsize, spacing, bounds


def freq_range(map_paths):
    smin, smax = 0.0, jnp.inf
    for map_path in map_paths:
        ccp4 = gemmi.read_ccp4_map(map_path)
        mpmin = 1 / ccp4.grid.unit_cell.a
        mpmax = 1 / (2 * ccp4.grid.spacing[0])
        if mpmin > smin:
            smin = mpmin
        if mpmax < smax:
            smax = mpmax

    return smin, smax


def write_map(data, template, path):
    result_map = gemmi.Ccp4Map()
    result_map.grid = gemmi.FloatGrid(np.array(data, dtype=np.float32))
    result_map.grid.copy_metadata_from(template)
    result_map.update_ccp4_header()
    result_map.write_ccp4_map(path)


def from_gemmi(st, selections=None):
    def label_from_cra(cra):
        crastr = str(cra)
        noalt, _, _ = crastr.partition(".")
        return noalt

    st.setup_entities()
    st.expand_ncs(gemmi.HowToNameCopiedChain.Short)

    # remove metal coordination
    ncon = len(st.connections)
    for ind in reversed(range(ncon)):
        if st.connections[ind].type is gemmi.ConnectionType.MetalC:
            del st.connections[ind]

    monlib_path = os.getenv("CLIBD_MON", default=None)
    if monlib_path:
        resnames = st[0].get_all_residue_names()
        monlib = gemmi.read_monomer_lib(monlib_path, resnames)
    else:
        warnings.warn(
            "Monomer Library not found, falling back to GEMMI defaults", RuntimeWarning
        )
        monlib = gemmi.MonLib()

    conlist = gemmi.ConnectionList(st.connections)
    topo = gemmi.prepare_topology(
        st,
        monlib,
        h_change=gemmi.HydrogenChange.ReAdd,
        warnings=sys.stderr,
    )
    missing = topo.find_missing_atoms(including_hydrogen=False)

    # add missing as 'dummy' atoms
    for m in missing:
        mon = monlib.monomers[m.res_id.name]
        monat = mon.find_atom(m.atom_name)

        atom = gemmi.Atom()
        atom.occ = 0.0
        atom.element = monat.el
        atom.name = m.atom_name

        cra = st[0].find_cra(m)
        cra.residue.add_atom(atom)

    # A side-effect of gemmi.prepare_topology is to modify the link_id
    # field of connections. We restore the original list of
    # connections to avoid inconsistencies.
    st.connections = conlist
    topo = gemmi.prepare_topology(
        st,
        monlib,
        h_change=gemmi.HydrogenChange.NoChange,
        warnings=sys.stderr,
    )

    lookup = {x.atom: label_from_cra(x) for x in st[0].all()}
    nbdict = defaultdict(list)

    for bond in topo.bonds:
        names = [(atom.element.atomic_number, lookup[atom]) for atom in bond.atoms]
        for i in [True, False]:
            nbdict[bond.atoms[i]].append(names[not i])

    # set flags from selection
    sels = [gemmi.Selection(s) for s in selections] if selections else []
    for sel in sels:
        for model in sel.models(st):
            for chain in sel.chains(model):
                for res in sel.residues(chain):
                    for atom in sel.atoms(res):
                        atom.flag = "e"

    # load model parameters into arrays
    n_atoms = st[0].count_atom_sites(gemmi.Selection(";q>0"))
    coords = np.empty((n_atoms, 3))
    umat = np.empty((n_atoms, 3, 3))
    occ = np.empty(n_atoms)
    atmask = np.empty(n_atoms, dtype=bool)
    # element 0: central atom, elements 1-9: first bonding neighours, element 10: flag
    atydesc = np.zeros((n_atoms, 11), dtype=np.uint8)
    ind = 0

    for cra in st[0].all():
        # ignore zero-occupancy atoms
        if cra.atom.occ == 0.0:
            continue

        coords[ind] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
        occ[ind] = cra.atom.occ
        atmask[ind] = cra.atom.flag != "e"

        if cra.atom.aniso.nonzero():
            umat[ind] = 8 * np.pi**2 * np.array(cra.atom.aniso.as_mat33().tolist())
        else:
            umat[ind] = cra.atom.b_iso * np.identity(3)

        envinds = set(nbdict[cra.atom])
        if len(envinds) > 0:
            envdesc, _ = zip(*envinds)
        else:
            envdesc = tuple()

        atomic_number = cra.atom.element.atomic_number
        envdesc = [atomic_number] + sorted(envdesc)
        trunc = min(10, len(envdesc))
        atydesc[ind, :trunc] = envdesc[:trunc]

        # set flag for carboxyl groups
        if cra.atom.name in ["OD1", "OD2", "OE1", "OE2"]:
            atydesc[ind, -1] = 201

        if not atmask[ind]:
            atydesc[ind] = 0
            atydesc[ind, 0] = 255

        ind += 1

    atydesc, aty, atycounts = np.unique(
        atydesc,
        return_inverse=True,
        return_counts=True,
        axis=0,
    )

    it92 = np.empty((len(atydesc), 10))
    for ind, envdesc in enumerate(atydesc):
        element = gemmi.Element(envdesc[0])
        it92[ind] = np.concatenate([element.c4322.a, element.c4322.b])

    # put on JAX device
    coords, it92, umat, occ, aty = [
        jnp.asarray(a) for a in (coords, it92, umat, occ, aty)
    ]

    return coords, it92, umat, occ, aty, atmask, atycounts, atydesc


def from_multiple(structures, selection=None):
    nst = len(structures)
    output = map(from_gemmi, structures, repeat(selection, nst))
    coords, it92, umat, occ, aty, atmask, atycounts, atydesc = zip(*output)
    naty = [len(a) for a in atycounts]
    natoms = np.array([len(a) for a in coords])
    molind = jnp.repeat(jnp.arange(nst), natoms)

    coords, umat, occ = [jnp.concatenate(a) for a in (coords, umat, occ)]
    atydesc, unq_ind, rev_ind = np.unique(
        np.concatenate(atydesc), return_index=True, return_inverse=True, axis=0
    )
    aty_shifted = np.concatenate(
        [aty[i] + sum(naty[:i]) for i in range(len(structures))]
    )
    aty = jnp.asarray(rev_ind[aty_shifted])
    it92 = jnp.concatenate(it92)[unq_ind]
    atmask = np.concatenate(atmask)

    atycounts_sum = np.zeros(len(atydesc), dtype=int)
    rev_seg = np.split(rev_ind, np.cumsum(naty))
    for counts, inds in zip(atycounts, rev_seg):
        atycounts_sum[inds] += counts

    return coords, it92, umat, occ, aty, atmask, atycounts_sum, atydesc, molind


def align_aty(ref, new, approx=False):
    cat = np.concatenate([ref, new])
    cov = calc_cov_aty(cat)
    crosscov = cov[len(ref) :, : len(ref)]
    maxind = np.argmax(crosscov, axis=1)
    maxval = np.round(crosscov[np.arange(len(new)), maxind], decimals=3)

    if not approx:
        maxind[maxval < 1.0] = -1
    else:
        maxind[maxval == 0.0] = -1

    return maxind


def make_selections(st):
    n_at = st[0].count_atom_sites()
    bval = np.empty(n_at)

    for ind, cra in enumerate(st[0].all()):
        bval[ind] = cra.atom.b_iso

    iqr = np.quantile(bval, 0.75) - np.quantile(bval, 0.25)
    bcut = np.median(bval) + 1.5 * iqr
    selections = [";q<1", f";b>{bcut:.1f}"]
    return selections
