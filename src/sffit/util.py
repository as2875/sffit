import os
from collections import defaultdict

import gemmi
import jax.numpy as jnp
import numpy as np


def read_mrc(map_path, mask_path):
    ccp4 = gemmi.read_ccp4_map(map_path)
    mpdata = jnp.array(ccp4.grid.array)
    if mask_path:
        msk = gemmi.read_ccp4_map(mask_path)
        mskdata = jnp.array(msk.grid.array)
    else:
        mskdata = jnp.ones_like(mpdata)

    masked = mpdata * mskdata

    assert (
        ccp4.grid.nu == ccp4.grid.nv == ccp4.grid.nw
    ), "Only cubic boxes are supported"
    assert (
        ccp4.grid.spacing[0] == ccp4.grid.spacing[1] == ccp4.grid.spacing[2]
    ), "Only cubic boxes are supported"

    fft_scale = ccp4.grid.unit_cell.volume / ccp4.grid.point_count

    bsize = ccp4.grid.nu
    spacing = ccp4.grid.spacing[0]
    bounds = jnp.array([[0, bsize * spacing] for i in range(3)])

    return ccp4.grid, masked, fft_scale, bsize, spacing, bounds


def write_map(data, template, path):
    result_map = gemmi.Ccp4Map()
    result_map.grid = gemmi.FloatGrid(np.array(data, dtype=np.float32))
    result_map.grid.copy_metadata_from(template)
    result_map.update_ccp4_header()
    result_map.write_ccp4_map(path)


def from_gemmi(st, st_aty, selection=None, b_iso=False):
    st.expand_ncs(gemmi.HowToNameCopiedChain.Short)
    st_aty.remove_alternative_conformations()
    st_aty.expand_ncs(gemmi.HowToNameCopiedChain.Short)

    serial_map = {}
    for c1, c2 in zip(st[0], st_aty[0]):
        for r1, r2 in zip(c1, c2):
            for atref in r2:
                atname = atref.name
                for ateq in r1[atname]:
                    serial_map[ateq.serial] = atref.serial

    monlib_path = os.environ["CLIBD_MON"]
    resnames = st_aty[0].get_all_residue_names()
    monlib = gemmi.read_monomer_lib(monlib_path, resnames)
    topo = gemmi.prepare_topology(
        st_aty,
        monlib,
        h_change=gemmi.HydrogenChange.NoChange,
    )
    nbdict = defaultdict(list)

    for cra in st_aty[0].all():
        nbdict[cra.atom.serial].append(cra.atom.element.atomic_number)

    for bond in topo.bonds:
        elems = [atom.element.atomic_number for atom in bond.atoms]
        serials = [atom.serial for atom in bond.atoms]

        for i in [True, False]:
            nbdict[serials[i]].append(elems[not i])

    for k in nbdict.keys():
        nbdict[k] = (nbdict[k][0], len(nbdict[k][1:]))

    # set flags from selection
    if selection is not None:
        sel = gemmi.Selection(selection)
        for model in sel.models(st):
            for chain in sel.chains(model):
                for res in sel.residues(chain):
                    for atom in sel.atoms(res):
                        atom.flag = "e"

    # load model parameters into arrays
    n_atoms = st[0].count_atom_sites()
    coords = np.empty((n_atoms, 3))
    it92 = np.empty((n_atoms, 10))
    occ = np.empty(n_atoms)
    atyhash = np.empty(n_atoms, dtype=int)
    atydesc = np.zeros((n_atoms, 2), dtype=int)
    atnames = np.empty(n_atoms, dtype="<U20")
    atmask = np.empty(n_atoms, dtype=bool)

    if b_iso:
        umat = np.empty(n_atoms)
    else:
        umat = np.empty((n_atoms, 3, 3))

    for ind, cra in enumerate(st[0].all()):
        coords[ind] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
        it92[ind] = np.concatenate([cra.atom.element.c4322.a, cra.atom.element.c4322.b])
        occ[ind] = cra.atom.occ
        atmask[ind] = cra.atom.flag != "e"

        if cra.atom.aniso.nonzero():
            if b_iso:
                umat[ind] = cra.atom.b_eq()
            else:
                umat[ind] = 8 * np.pi**2 * np.array(cra.atom.aniso.as_mat33().tolist())

        else:
            if b_iso:
                umat[ind] = cra.atom.b_iso
            else:
                umat[ind] = cra.atom.b_iso * np.identity(3)

        envdesc = nbdict[serial_map[cra.atom.serial]]
        atnames[ind] = str(cra)
        atyhash[ind] = hash(envdesc)
        atydesc[ind, : len(envdesc)] = envdesc

    _, unq_ind, aty, atycounts = np.unique(
        atyhash,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    # put on JAX device
    coords, it92, umat, occ, aty = [
        jnp.array(a) for a in (coords, it92, umat, occ, aty)
    ]

    return coords, it92, umat, occ, aty, atmask, atycounts, atnames, atydesc, unq_ind
