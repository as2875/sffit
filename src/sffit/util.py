import os
import sys
from collections import defaultdict

import gemmi
import jax.numpy as jnp
import numpy as np


def read_mrc(map_path, mask_path):
    ccp4 = gemmi.read_ccp4_map(map_path)
    mpdata = np.array(ccp4.grid.array)
    if mask_path:
        msk = gemmi.read_ccp4_map(mask_path)
        mskdata = np.array(msk.grid.array)
    else:
        mskdata = np.ones_like(mpdata)

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


def from_gemmi(st, selection=None):
    def label_from_cra(cra):
        crastr = str(cra)
        noalt, _, _ = crastr.partition(".")
        return noalt

    st.setup_entities()
    st.expand_ncs(gemmi.HowToNameCopiedChain.Short)

    monlib_path = os.environ["CLIBD_MON"]
    resnames = st[0].get_all_residue_names()
    monlib = gemmi.read_monomer_lib(monlib_path, resnames)
    conlist = gemmi.ConnectionList(st.connections)

    topo = gemmi.prepare_topology(
        st,
        monlib,
        h_change=gemmi.HydrogenChange.NoChange,
        warnings=sys.stderr,
    )
    missing = topo.find_missing_atoms(including_hydrogen=True)

    # add missing hydrogens as 'dummy' atoms
    for m in missing:
        mon = monlib.monomers[m.res_id.name]
        monat = mon.find_atom(m.atom_name)

        if m.res_id.name == "HOH" or monat.el.atomic_number != 1:
            continue

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
    if selection is not None:
        sel = gemmi.Selection(selection)
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
    atydesc = np.zeros((n_atoms, 10), dtype=int)
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

        envdesc = [cra.atom.element.atomic_number] + sorted(envdesc)
        atydesc[ind, : len(envdesc)] = envdesc

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
        jnp.array(a) for a in (coords, it92, umat, occ, aty)
    ]

    return coords, it92, umat, occ, aty, atmask, atycounts, atydesc
