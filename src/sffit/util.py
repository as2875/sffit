# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import contextlib
import os
import sys
import warnings
from collections import defaultdict
from itertools import repeat

import gemmi
import jax.numpy as jnp
import numpy as np

from .spherical import calc_cov_aty


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

    return masked, fft_scale, bsize, spacing, bounds


def read_multiple(map_paths, mask_path=None):
    mask_paths = repeat(mask_path)

    # get dimensions from first map
    _, fft_scale, bsize, spacing, bounds = read_mrc(map_paths[0])
    output = map(read_mrc, map_paths, mask_paths)
    if bsize % 2 == 0:
        masked = np.empty(
            (len(map_paths), bsize, bsize, bsize // 2 + 1), dtype=np.complex64
        )
    else:
        masked = np.empty(
            (len(map_paths), bsize, bsize, (bsize + 1) // 2), dtype=np.complex64
        )

    for ind, (data, *_) in enumerate(output):
        masked[ind] = jnp.fft.rfftn(data) * fft_scale

    return masked, fft_scale, bsize, spacing, bounds


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


def write_map(data, path, dim, cell):
    result_map = gemmi.Ccp4Map()
    result_map.grid = gemmi.FloatGrid(np.array(data, dtype=np.float32))
    result_map.grid.set_size(dim, dim, dim)
    result_map.grid.set_unit_cell(gemmi.UnitCell(cell, cell, cell, 90, 90, 90))
    result_map.update_ccp4_header()
    result_map.write_ccp4_map(path)


def setup_monlib(st):
    monlib_path = os.getenv("CLIBD_MON", default=None)
    if monlib_path:
        resnames = st[0].get_all_residue_names()
        try:
            monlib = gemmi.read_monomer_lib(monlib_path, resnames)
        except RuntimeError as err:
            (msg,) = err.args
            warnings.warn(msg, RuntimeWarning)
            monlib = gemmi.MonLib()
    else:
        warnings.warn(
            "Monomer Library not found, falling back to GEMMI defaults", RuntimeWarning
        )
        monlib = gemmi.MonLib()

    return monlib


def from_gemmi(st, selections=None, cif=None, nochangeh=False):
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

    monlib = setup_monlib(st)

    if cif:
        monlib.read_monomer_cif(cif)

    conlist = gemmi.ConnectionList(st.connections)
    h_change = (
        gemmi.HydrogenChange.NoChange if nochangeh else gemmi.HydrogenChange.ReAdd
    )
    topo = gemmi.prepare_topology(
        st,
        monlib,
        h_change=h_change,
        warnings=sys.stderr,
    )
    missing = topo.find_missing_atoms(including_hydrogen=nochangeh)

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
            if cra.residue.name in ["ASP", "GLU"]:
                atydesc[ind, -1] = 201
            elif cra.residue.name in ["ASN", "GLN"]:
                atydesc[ind, -1] = 202

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


def reindex_excluded(atmask, aty, atydesc):
    expanded = atydesc[aty]
    expanded[~atmask, :] = [255] + 10 * [0]
    atydesc, aty, atycounts = jnp.unique(
        expanded,
        return_inverse=True,
        return_counts=True,
        axis=0,
    )
    return aty, atycounts, atydesc


def make_selections(st):
    n_at = st[0].count_atom_sites()
    bval = np.empty(n_at)

    for ind, cra in enumerate(st[0].all()):
        bval[ind] = cra.atom.b_iso

    iqr = np.quantile(bval, 0.75) - np.quantile(bval, 0.25)
    bcut = np.median(bval) + 1.5 * iqr
    selections = [";q<1", f";b>{bcut:.1f}"]
    return selections


@contextlib.contextmanager
def silence_stdout():
    with (
        os.fdopen(os.dup(sys.stdout.fileno()), "wb") as copied,
        open(os.devnull, "wb") as devnull,
    ):
        sys.stdout.flush()
        try:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            yield sys.stdout
        finally:
            sys.stdout.flush()
            os.dup2(copied.fileno(), sys.stdout.fileno())


def aty_to_str(arr):
    env, flag = arr[:10], arr[10]
    syms = [gemmi.Element(n).name for n in arr[np.nonzero(env)]]
    match flag:
        case 201:
            syms.append(", carboxyl")
        case 202:
            syms.append(", amide")
    if len(syms) == 1:
        nb = "(0)"
    else:
        nb = "(" + "".join(syms[1:]) + ")"
    atystr = syms[0] + nb
    return atystr
