import copy
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pomato
from pomato.tools import copytree


class TestPomatoGridModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)
        copytree(Path.cwd().joinpath("tests/test_data/cbco_lists"), cls.wdir)

        pomato.tools.create_folder_structure(cls.wdir)
        with open(cls.wdir.joinpath("profiles/nrel118.json")) as opt_file:
                loaded_options = json.load(opt_file)
        cls.options = pomato.tools.add_default_options(loaded_options) 

        cls.data = pomato.data.DataManagement(cls.options, cls.wdir)
        cls.data.logger.setLevel(logging.ERROR)
        cls.data.load_data(r'data_input/nrel_118.zip')
        
        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]
        cls.data.nodes.loc[R2_to_R3, "zone"] = "R3"
        cls.grid = pomato.grid.GridTopology()
        cls.grid.calculate_parameters(cls.data.nodes, cls.data.lines)
        cls.grid_model = pomato.grid.GridModel(cls.wdir, cls.grid, cls.data, cls.options)
        cls.grid_model.logger.setLevel(logging.ERROR)

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        if cls.grid_model.julia_instance:
            cls.grid_model.julia_instance.join()
            cls.grid_model.julia_instance = None
            
        cls.grid_model = None
        cls.grid = None
        cls.data = None
        cls.options = None
        cls.wdir = None
        cls.temp_dir = None
        
    def test_ntc(self):

        self.grid_model.options["type"] = "ntc"
        self.options["redispatch"]["include"] = True

        self.grid_model.create_grid_representation()
        gr = self.grid_model.grid_representation
        np.testing.assert_equal(gr.redispatch_grid[self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(gr.redispatch_grid["ram"].values, 
                                self.data.lines.capacity.values*self.options["grid"]["capacity_multiplier"])

    def test_zonal(self):
        
        self.grid_model.options["type"] = "zonal"
        self.grid_model.options["grid"]["gsk"] = "gmax"

        self.grid_model.create_grid_representation()
        grid_representation_gmax = copy.copy(self.grid_model.grid_representation)

        self.grid_model.options["grid"]["gsk"] = "flat"
        self.grid_model.create_grid_representation()
        grid_representation_flat = self.grid_model.grid_representation

        self.assertRaises(AssertionError, np.testing.assert_equal, 
                          grid_representation_flat.grid.values, grid_representation_gmax.grid.values)

        test_columns = ["cb", "co", "ram"] + list(self.grid_model.data.zones.index)
        self.assertTrue(all(grid_representation_flat.grid.columns == test_columns))
        self.assertTrue(all(grid_representation_gmax.grid.columns == test_columns))

    def test_nodal(self):
        self.grid_model.options["type"] = "nodal"
        self.grid_model.create_grid_representation()
        gr = self.grid_model.grid_representation
        np.testing.assert_equal(gr.grid[self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(gr.grid["ram"].values/self.grid_model.options["grid"]["capacity_multiplier"], self.data.lines.capacity.values)
        

    def test_cbco_nodal(self):
        self.grid_model.options["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["redundancy_removal_option"] = "full"

        self.grid_model.create_grid_representation()
        gr = self.grid_model.grid_representation

        # test 10 contingency ptdf
        c_ptdf = gr.grid
        c_ptdf = c_ptdf[c_ptdf.co != "basecase"]
        test_contingencies = random.sample(range(0, len(c_ptdf)), 25)

        for contingency in test_contingencies:
            cb, co = c_ptdf.loc[c_ptdf.index[contingency], ["cb", "co"]]
            tmp_ptdf =  c_ptdf.loc[c_ptdf.index[contingency],  self.data.nodes.index].values
            tmp_ptdf = tmp_ptdf.astype(float).reshape((1, len(self.data.nodes)))
            tmp_ptdf_test = self.grid.create_n_1_ptdf_cbco(cb, co)
            np.testing.assert_allclose(tmp_ptdf_test, tmp_ptdf, atol=1e-6)

    def test_cbco_nodal_no_precalc(self):

        self.grid_model.options["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["precalc_filename"] = "random_words"
        grid = self.grid_model.create_cbco_nodal_grid_parameters()
        c_ptdf_fallback = copy.copy(grid)
        self.grid_model.options["grid"]["precalc_filename"] = ""
        self.grid_model.options["grid"]["redundancy_removal_option"] = "full"
        self.grid_model.create_grid_representation()
        pd.testing.assert_frame_equal(c_ptdf_fallback, self.grid_model.grid_representation.grid)

    def test_cbco_nodal_precalc_index(self):
        
        my_file = self.wdir.joinpath('cbco_nrel_118.csv')
        to_file = self.wdir.joinpath('data_temp/julia_files/cbco_data/cbco_nrel_118.csv')
        shutil.copyfile(str(my_file), str(to_file))

        self.grid_model.options["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["precalc_filename"] = "cbco_nrel_118"
        self.grid_model.options["grid"]["capacity_multiplier"] = 0.8
        self.grid_model.create_cbco_nodal_grid_parameters()

    def test_cbco_nodal_precalc_table(self):
        
        my_file = self.wdir.joinpath('cbco_nrel_118.csv')
        to_file = self.wdir.joinpath('data_temp/julia_files/cbco_data/nrel_cbco_table.csv')
        shutil.copyfile(str(my_file), str(to_file))

        self.grid_model.options["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["precalc_filename"] = "cbco_nrel_118"
        self.grid_model.create_cbco_nodal_grid_parameters()
    
    def test_cbco_nodal_save(self):
        self.grid_model.options["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["redundancy_removal_option"] = "save"
        self.grid_model.options["grid"]["precalc_filename"] = None
        self.grid_model.create_cbco_nodal_grid_parameters()

        self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/A_py_save.csv").is_file())
        self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/b_py_save.csv").is_file())
        self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/I_py_save.csv").is_file())
        self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/x_bounds_py_save.csv").is_file())
    
    def test_cbco_nodal_invalid_option(self):
        self.grid_model.options["type"] = "cbco_nodal"
        self.grid_model.options["grid"]["redundancy_removal_option"] = "invalid_option"
        self.grid_model.options["grid"]["precalc_filename"] = None
        self.assertRaises(AttributeError, self.grid_model.create_cbco_nodal_grid_parameters)

    def test_clarkson(self):

        test_configs = [("cbco_nodal", "clarkson_base"), 
                        # ("cbco_nodal", "clarkson"), too slow
                        ("nodal", "nodal_clarkson"), 
                        ("zonal", "clarkson"), 
                        ("cbco_zonal", "clarkson")]

        for (optimization_option, redundancy_removal_option) in test_configs:

            self.grid_model.options["type"] = optimization_option
            self.grid_model.options["grid"]["redundancy_removal_option"] = redundancy_removal_option
            self.grid_model.create_grid_representation()
            
            file = pomato.tools.newest_file_folder(self.grid_model.julia_dir.joinpath("cbco_data"), keyword="cbco")
            self.assertTrue(file.is_file())
            self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/A_py.csv").is_file())
            self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/b_py.csv").is_file())
            self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/I_py.csv").is_file())
            self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/x_bounds_py.csv").is_file())
              
        self.grid_model.julia_instance.join()
