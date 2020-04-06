import logging
import random
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from context import pomato

class JuliaMockup():
    def __init__(self):
        self.is_alive = True
        self.solved = True
    def run(self, args):
        pass
    
# pylint: disable-msg=E1101
class TestPomatoGridRepresentation(unittest.TestCase):
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")
        self.options = pomato.tools.default_options()
        self.data = pomato.data.DataManagement(self.options, self.wdir)
        self.data.logger.setLevel(logging.ERROR)
        self.data.load_data('data_input/pglib_opf_case118_ieee.m')
        self.grid  = pomato.grid.GridModel(self.data.nodes, self.data.lines)
        self.cbco_module = pomato.cbco.CBCOModule(self.wdir, self.wdir, self.grid, self.data, self.options)
        self.cbco_module.logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def tessht_ntc(self):
        self.cbco_module.options["optimization"]["type"] = "ntc"
        self.cbco_module.create_grid_representation()
        grid_representation = self.cbco_module.grid_representation
        np.testing.assert_equal(grid_representation["redispatch_grid"][self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(grid_representation["redispatch_grid"]["ram"].values, self.data.lines.maxflow.values)

    def test_nodal(self):
        self.cbco_module.options["optimization"]["type"] = "nodal"
        self.cbco_module.create_grid_representation()
        grid_representation = self.cbco_module.grid_representation
        np.testing.assert_equal(grid_representation["grid"][self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(grid_representation["grid"]["ram"].values, self.data.lines.maxflow.values)

    def test_cbco_nodal(self):
        self.cbco_module.options["optimization"]["type"] = "cbco_nodal"
        self.cbco_module.create_grid_representation()
        grid_representation = self.cbco_module.grid_representation

        # test 10 contingency ptdf's
        c_ptdf = grid_representation["grid"]
        c_ptdf = c_ptdf[c_ptdf.co != "basecase"]
        test_contingencies = random.sample(range(0, len(c_ptdf)), 25)

        for contingency in test_contingencies:
            cb, co = c_ptdf.loc[c_ptdf.index[contingency], ["cb", "co"]]
            tmp_ptdf =  c_ptdf.loc[c_ptdf.index[contingency], 
            self.data.nodes.index].values.reshape((1, len(self.data.nodes)))
            np.testing.assert_equal(self.grid.create_n_1_ptdf_cbco(cb, co), tmp_ptdf)

    def test_cbco_nodal_no_precalc(self):
        self.cbco_module.options["optimization"]["type"] = "cbco_nodal"
        self.cbco_module.options["grid"]["precalc_filename"] = "asdasd"
        
        self.cbco_module.process_cbco_nodal()
        
        c_ptdf_fallback = self.cbco_module.grid_representation["grid"].copy()
        self.cbco_module.options["grid"]["precalc_filename"] = ""
        self.cbco_module.options["grid"]["cbco_option"] = "full"
        self.cbco_module.create_grid_representation()
        pd.testing.assert_frame_equal(c_ptdf_fallback, self.cbco_module.grid_representation["grid"])

    def test_cbco_nodal_clarkson(self):
        my_file = self.wdir.parent.joinpath('tests/test_data/ieee_cbco.csv')
        to_file = self.wdir.joinpath('data_temp/julia_files/cbco_data/ieee_cbco.csv')
        shutil.copyfile(str(my_file), str(to_file))
        self.cbco_module.jdir = self.wdir.joinpath('data_temp/julia_files')

        self.cbco_module.julia_instance = JuliaMockup()

        self.cbco_module.options["optimization"]["type"] = "cbco_nodal"
        self.cbco_module.options["grid"]["cbco_option"] = "clarkson_base"

        self.cbco_module.create_grid_representation()
        self.assertTrue(self.cbco_module.jdir.joinpath("cbco_data/A_py.csv").is_file())
        self.assertTrue(self.cbco_module.jdir.joinpath("cbco_data/b_py.csv").is_file())
        self.assertTrue(self.cbco_module.jdir.joinpath("cbco_data/I_py.csv").is_file())
        self.assertTrue(self.cbco_module.jdir.joinpath("cbco_data/x_bounds_py.csv").is_file())