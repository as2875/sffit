import os
from collections import defaultdict

import gemmi
import numpy as np


def write_map(data, template, path):
    result_map = gemmi.Ccp4Map()
    result_map.grid = gemmi.FloatGrid(np.array(data, dtype=np.float32))
    result_map.grid.copy_metadata_from(template.grid)
    result_map.update_ccp4_header()
    result_map.write_ccp4_map(path)


def from_gemmi(st, st_aty):
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
        h_change=gemmi.HydrogenChange.ReAdd,
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
        nbdict[k] = tuple(
            [nbdict[k][0]] + sorted(nbdict[k][1:]),
        )

    # load model parameters into arrays
    n_atoms = st[0].count_atom_sites()
    coords = np.empty((n_atoms, 3))
    it92 = np.empty((n_atoms, 10))
    umat = np.empty((n_atoms, 3, 3))
    occ = np.empty(n_atoms)
    atyhash = np.empty(n_atoms, dtype=int)
    atydesc = np.zeros((n_atoms, 10), dtype=int)
    atnames = np.empty(n_atoms, dtype="<U20")

    for ind, cra in enumerate(st[0].all()):
        coords[ind] = [cra.atom.pos.x, cra.atom.pos.y, cra.atom.pos.z]
        it92[ind] = np.concatenate([cra.atom.element.c4322.a, cra.atom.element.c4322.b])
        occ[ind] = cra.atom.occ

        if cra.atom.aniso.nonzero():
            umat[ind] = 8 * np.pi**2 * np.array(cra.atom.aniso.as_mat33().tolist())
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

    return coords, it92, umat, occ, aty, atycounts, atnames, atydesc, unq_ind
