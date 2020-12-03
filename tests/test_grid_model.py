import copy
import logging
import random
import shutil
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from context import pomato
from pomato import tools

# pylint: disable-msg=E1101
class TestPomatoGridModel(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('Log.MarketModel')
        self.logger.setLevel(logging.ERROR)
        self.wdir = Path.cwd().joinpath("examples")
        pomato.tools.create_folder_structure(self.wdir)
        with open(self.wdir.joinpath("profiles/nrel118.json")) as opt_file:
                loaded_options = json.load(opt_file)
        self.options = pomato.tools.add_default_options(loaded_options) 

        self.data = pomato.data.DataManagement(self.options, self.wdir)
        self.data.logger.setLevel(logging.ERROR)
        self.data.load_data(r'data_input/nrel_118.zip')
        
        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]

        self.data.nodes.loc[R2_to_R3, "zone"] = "R3"

        self.grid = pomato.grid.GridTopology()
        self.grid.calculate_parameters(self.data.nodes, self.data.lines)
        self.grid_model = pomato.grid.GridModel(self.wdir, self.grid, self.data, self.options)
        self.grid_model.logger.setLevel(logging.ERROR)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def test_ntc(self):

        self.grid_model.options["optimization"]["type"] = "ntc"
        self.options["optimization"]["redispatch"]["include"] = True

        self.grid_model.create_grid_representation()
        gr = self.grid_model.grid_representation
        np.testing.assert_equal(gr.redispatch_grid[self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(gr.redispatch_grid["ram"].values, 
                                self.data.lines.maxflow.values*self.options["grid"]["capacity_multiplier"])

    def test_zonal(self):
        
        self.grid_model.options["optimization"]["type"] = "zonal"
        self.grid_model.options["grid"]["gsk"] = "gmax"

        self.grid_model.create_grid_representation()
        grid_representation_gmax = copy.copy(self.grid_model.grid_representation)

        self.grid_model.options["grid"]["gsk"] = "flat"
        self.grid_model.create_grid_representation()
        grid_representation_flat = self.grid_model.grid_representation

        self.assertRaises(AssertionError, np.testing.assert_equal, 
                          grid_representation_flat.grid.values, grid_representation_gmax.grid.values)

        test_columns = ["cb", "co", "ram"] + list(self.grid_model.data.zones.index)
        print(grid_representation_flat.grid.columns)
        self.assertTrue(all(grid_representation_flat.grid.columns == test_columns))
        self.assertTrue(all(grid_representation_gmax.grid.columns == test_columns))

    def test_nodal(self):
        self.grid_model.options["optimization"]["type"] = "nodal"
        self.grid_model.create_grid_representation()
        gr = self.grid_model.grid_representation
        np.testing.assert_equal(gr.grid[self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(gr.grid["ram"].values/self.grid_model.options["grid"]["capacity_multiplier"], self.data.lines.maxflow.values)
        

    def test_cbco_nodal(self):
        self.grid_model.options["optimization"]["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["cbco_option"] = "full"

        self.grid_model.create_grid_representation()
        gr = self.grid_model.grid_representation

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

        self.grid_model.options["optimization"]["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["precalc_filename"] = "random_words"
        grid = self.grid_model.create_cbco_nodal_grid_parameters()
        c_ptdf_fallback = copy.copy(grid)
        self.grid_model.options["grid"]["precalc_filename"] = ""
        self.grid_model.options["grid"]["cbco_option"] = "full"
        self.grid_model.create_grid_representation()
        pd.testing.assert_frame_equal(c_ptdf_fallback, self.grid_model.grid_representation.grid)

    def test_cbco_nodal_index_precalc(self):
        
        my_file = self.wdir.parent.joinpath('tests/test_data/cbco_nrel_118.csv')
        to_file = self.wdir.joinpath('data_temp/julia_files/cbco_data/cbco_nrel_118.csv')
        shutil.copyfile(str(my_file), str(to_file))

        self.grid_model.options["optimization"]["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["precalc_filename"] = "cbco_nrel_118"
        self.grid_model.options["grid"]["capacity_multiplier"] = 0.8
        self.grid_model.create_cbco_nodal_grid_parameters()

    def test_clarkson(self):

        test_configs = [("cbco_nodal", "clarkson_base"), 
                        # ("cbco_nodal", "clarkson"), too slow
                        ("nodal", "nodal_clarkson"), 
                        ("zonal", "clarkson"), 
                        ("cbco_zonal", "clarkson")]

        for (optimization_option, cbco_option) in test_configs:
            print(optimization_option)

            self.grid_model.options["optimization"]["type"] = optimization_option
            self.grid_model.options["grid"]["cbco_option"] = cbco_option
            self.grid_model.create_grid_representation()
            
            file = tools.newest_file_folder(self.grid_model.julia_dir.joinpath("cbco_data"), keyword="cbco")
            self.assertTrue(file.is_file())
            self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/A_py.csv").is_file())
            self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/b_py.csv").is_file())
            self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/I_py.csv").is_file())
            self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/x_bounds_py.csv").is_file())
              
        self.grid_model.julia_instance.join()