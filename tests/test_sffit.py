import json
import os
import unittest

import numpy as np

from sffit import fit


class TestSffit(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("test_output"):
            os.mkdir("test_output")

    def test_gp(self):
        cmdline = [
            "gp",
            "--maps",
            "data/8el9_map.mrc",
            "--models",
            "data/8el9_model.pdb",
            "-o",
            "test_output/8el9_out.npz",
            "-oi",
            "test_output/8el9_int.npz",
            "--no-change-h",
            "--weight",
            "0.05",
        ]
        args = fit.parse_args(cmdline)
        args.func(args)

        refnpz = np.load("data/8el9_out.npz")
        testnpz = np.load("test_output/8el9_out.npz")

        self.assertTrue(np.allclose(refnpz["soln"], testnpz["soln"]))

    def test_mmcif_from_npz(self):
        cmdline = [
            "mmcif",
            "--params",
            "data/8el9_out.npz",
            "-ii",
            "data/8el9_int.npz",
            "-oj",
            "test_output/8el9_sog.json",
            "--models",
            "data/8el9_model.pdb",
            "-o",
            "test_output/8el9_with_sf.cif",
        ]
        args = fit.parse_args(cmdline)
        args.func(args)

        with (
            open("data/8el9_sog.json") as fref,
            open("test_output/8el9_sog.json") as ftest,
        ):
            jref, jtest = json.load(fref), json.load(ftest)

        self.assertEqual(jref, jtest)

        with (
            open("data/8el9_with_sf.cif") as fref,
            open("test_output/8el9_with_sf.cif") as ftest,
        ):
            self.assertEqual(fref.read(), ftest.read())

    def test_mmcif_from_json(self):
        cmdline = [
            "mmcif",
            "--params",
            "data/8el9_sog.json",
            "--models",
            "data/8el9_model.pdb",
            "-o",
            "test_output/8el9_with_sf.cif",
        ]
        args = fit.parse_args(cmdline)
        args.func(args)

        with (
            open("data/8el9_with_sf.cif") as fref,
            open("test_output/8el9_with_sf.cif") as ftest,
        ):
            self.assertEqual(fref.read(), ftest.read())
