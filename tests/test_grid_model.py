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
        copytree(Path.cwd().joinpath("tests/test_data"), cls.wdir)

        pomato.tools.create_folder_structure(cls.wdir)
        with open(cls.wdir.joinpath("profiles/nrel118.json")) as opt_file:
                loaded_options = json.load(opt_file)
        cls.options = pomato.tools.add_default_options(loaded_options) 

        cls.data = pomato.data.DataManagement(cls.options, cls.wdir)
        cls.data.logger.setLevel(logging.INFO)
        cls.data.load_data(r'data_input/nrel_118_original.zip')
        
    def setUp(self):
        self.grid = pomato.grid.GridTopology()
        self.grid.calculate_parameters(self.data.nodes, self.data.lines)
        self.grid_model = pomato.grid.GridModel(self.wdir, self.grid, self.data, self.options)
        self.grid_model.logger.setLevel(logging.INFO)

    def tearDown(self):
        if self.grid_model.julia_instance:
            self.grid_model.julia_instance.join()
            self.grid_model.julia_instance = None

    @classmethod
    def tearDownClass(cls):           
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
                                self.data.lines.capacity.values*self.options["grid"]["long_term_rating_factor"])

    def test_nodal(self):
        self.grid_model.options["type"] = "opf"
        self.grid_model.create_grid_representation()
        gr = self.grid_model.grid_representation
        np.testing.assert_equal(gr.grid[self.data.nodes.index].values, self.grid.ptdf)
        np.testing.assert_equal(gr.grid["ram"].values/self.grid_model.options["grid"]["long_term_rating_factor"], self.data.lines.capacity.values)
        

    def test_scopf(self):
        self.grid_model.options["type"] = "scopf"
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

    def test_scopf_no_precalc(self):

        self.grid_model.options["type"] = "scopf"
        self.grid_model.options["grid"]["precalc_filename"] = "random_words"
        grid = self.grid_model.create_scopf_grid_parameters()
        c_ptdf_fallback = copy.copy(grid)
        self.grid_model.options["grid"]["precalc_filename"] = ""
        self.grid_model.options["grid"]["redundancy_removal_option"] = "full"
        self.grid_model.create_grid_representation()
        pd.testing.assert_frame_equal(c_ptdf_fallback, self.grid_model.grid_representation.grid)

    def test_scopf_precalc_index(self):
        
        my_file = self.wdir.joinpath('nrel_cbco_indices.csv')
        to_file = self.wdir.joinpath('data_temp/julia_files/cbco_data/cbco_nrel_118.csv')
        shutil.copyfile(str(my_file), str(to_file))

        self.grid_model.options["type"] = "scopf"
        self.grid_model.options["grid"]["precalc_filename"] = "nrel_cbco_indices"
        self.grid_model.options["grid"]["long_term_rating_factor"] = 1
        self.grid_model.create_scopf_grid_parameters()

    def test_scopf_precalc_table(self):
        
        my_file = self.wdir.joinpath('nrel_cbco_table.csv')
        to_file = self.wdir.joinpath('data_temp/julia_files/cbco_data/nrel_cbco_table.csv')
        shutil.copyfile(str(my_file), str(to_file))

        self.grid_model.options["type"] = "scopf"
        self.grid_model.options["grid"]["precalc_filename"] = "nrel_cbco_table"
        self.grid_model.create_scopf_grid_parameters()
    
    def test_scopf_save(self):
        self.grid_model.options["type"] = "scopf"
        self.grid_model.options["grid"]["redundancy_removal_option"] = "save"
        self.grid_model.options["grid"]["precalc_filename"] = None
        self.grid_model.create_scopf_grid_parameters()

        self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/A_py_save.csv").is_file())
        self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/b_py_save.csv").is_file())
        self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/I_py_save.csv").is_file())
        self.assertTrue(self.grid_model.julia_dir.joinpath("cbco_data/x_bounds_py_save.csv").is_file())
    
    def test_scopf_invalid_option(self):
        self.grid_model.options["type"] = "scopf"
        self.grid_model.options["grid"]["redundancy_removal_option"] = "invalid_option"
        self.grid_model.options["grid"]["precalc_filename"] = None
        self.assertRaises(AttributeError, self.grid_model.create_scopf_grid_parameters)

    def test_clarkson(self):

        test_configs = [("scopf", "conditional_redundancy_removal"), 
                        # ("cbco_nodal", "clarkson"), too slow
                        ("opf", "redundancy_removal")]
        self.grid_model.options["grid"]["sensitivity"] = 2e-2
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
