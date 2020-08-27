import copy
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
        self.options["data"]["stacked"] = ["demand_el_rt", "demand_el_da", "availability_da", "availability_rt", "net_export"]
        self.data = pomato.data.DataManagement(self.options, self.wdir)
        self.data.logger.setLevel(logging.ERROR)
        self.data.load_data(r'data_input/nrel_118.zip')

        self.grid = pomato.grid.GridModel()
        self.grid.calculate_parameters(self.data.nodes, self.data.lines)
        self.grid_representation = pomato.cbco.GridRepresentation(self.wdir, self.grid, self.data, self.options)
        self.grid_representation.logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def test_ntc(self):

        self.grid_representation.options["optimization"]["type"] = "ntc"
        self.options["optimization"]["redispatch"]["include"] = True

        self.grid_representation.create_grid_representation()
        gr = self.grid_representation.grid_representation
        np.testing.assert_equal(gr.redispatch_grid[self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(gr.redispatch_grid["ram"].values, 
                                self.data.lines.maxflow.values*self.options["grid"]["capacity_multiplier"])

    def test_zonal(self):
        
        self.grid_representation.options["optimization"]["type"] = "zonal"
        self.grid_representation.options["grid"]["gsk"] = "gmax"

        self.grid_representation.create_grid_representation()
        grid_representation_gmax = copy.copy(self.grid_representation.grid_representation)

        self.grid_representation.options["grid"]["gsk"] = "flat"
        self.grid_representation.create_grid_representation()
        grid_representation_flat = self.grid_representation.grid_representation

        self.assertRaises(AssertionError, np.testing.assert_equal, 
                          grid_representation_flat.grid.values, grid_representation_gmax.grid.values)

        test_columns = list(self.grid_representation.data.zones.index) + ["ram"]
        self.assertTrue(all(grid_representation_flat.grid.columns == test_columns))
        self.assertTrue(all(grid_representation_gmax.grid.columns == test_columns))

    def test_nodal(self):
        self.grid_representation.options["optimization"]["type"] = "nodal"
        self.grid_representation.create_grid_representation()
        gr = self.grid_representation.grid_representation
        np.testing.assert_equal(gr.grid[self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(gr.grid["ram"].values, self.data.lines.maxflow.values)
        

    def test_cbco_nodal(self):
        self.grid_representation.options["optimization"]["type"] = "cbco_nodal"
        self.grid_representation.create_grid_representation()
        gr = self.grid_representation.grid_representation

        # test 10 contingency ptdf
        c_ptdf = gr.grid
        c_ptdf = c_ptdf[c_ptdf.co != "basecase"]
        test_contingencies = random.sample(range(0, len(c_ptdf)), 25)

        for contingency in test_contingencies:
            cb, co = c_ptdf.loc[c_ptdf.index[contingency], ["cb", "co"]]
            tmp_ptdf =  c_ptdf.loc[c_ptdf.index[contingency], 
            self.data.nodes.index].values.reshape((1, len(self.data.nodes)))
            np.testing.assert_equal(self.grid.create_n_1_ptdf_cbco(cb, co), tmp_ptdf)

    def test_cbco_nodal_no_precalc(self):

        self.grid_representation.options["optimization"]["type"] = "cbco_nodal"
        self.grid_representation.options["grid"]["precalc_filename"] = "random_words"
        
        self.grid_representation.process_cbco_nodal()
        
        c_ptdf_fallback = copy.copy(self.grid_representation.grid_representation.grid)
        self.grid_representation.options["grid"]["precalc_filename"] = ""
        self.grid_representation.options["grid"]["cbco_option"] = "full"
        self.grid_representation.create_grid_representation()
        pd.testing.assert_frame_equal(c_ptdf_fallback, self.grid_representation.grid_representation.grid)

    def test_cbco_nodal_clarkson(self):
        my_file = self.wdir.parent.joinpath('tests/test_data/nrel_cbco.csv')
        to_file = self.wdir.joinpath('data_temp/julia_files/cbco_data/nrel_cbco.csv')
        shutil.copyfile(str(my_file), str(to_file))
        self.grid_representation.julia_dir = self.wdir.joinpath('data_temp/julia_files')

        self.grid_representation.julia_instance = JuliaMockup()

        self.grid_representation.options["optimization"]["type"] = "cbco_nodal"
        self.grid_representation.options["grid"]["cbco_option"] = "clarkson_base"

        self.grid_representation.create_grid_representation()
        self.assertTrue(self.grid_representation.julia_dir.joinpath("cbco_data/A_py.csv").is_file())
        self.assertTrue(self.grid_representation.julia_dir.joinpath("cbco_data/b_py.csv").is_file())
        self.assertTrue(self.grid_representation.julia_dir.joinpath("cbco_data/I_py.csv").is_file())
        self.assertTrue(self.grid_representation.julia_dir.joinpath("cbco_data/x_bounds_py.csv").is_file())
