import logging
import random
import shutil
import unittest
import json
from pathlib import Path
import time 

import numpy as np
import pandas as pd
import tempfile

from context import pomato, copytree

class TestPomatoData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)

        copytree(Path.cwd().joinpath("examples"), cls.wdir)
        copytree(Path.cwd().joinpath("tests/test_data/nrel_result"), cls.wdir)

        with open(cls.wdir.joinpath("profiles/nrel118.json")) as opt_file:
                loaded_options = json.load(opt_file)
        cls.options = pomato.tools.add_default_options(loaded_options) 
        cls.options["model_horizon"] = [0, 24]

        cls.data = pomato.data.DataManagement(cls.options, cls.wdir)
        cls.data.logger.setLevel(logging.ERROR)
        cls.data.load_data('data_input/nrel_118.zip')

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.data = None
        cls.temp_dir = None

    def test_plant_data(self):
        self.assertTrue(self.data.plants.loc[self.data.plants.mc_el.isna(), :].empty)

    def test_nodes_lines_data(self):
        self.assertEqual(len(self.data.nodes), 118)
        self.assertEqual(len(self.data.lines), 186)
        for data in ["slack"]:
            self.assertTrue(self.data.nodes.loc[self.data.nodes[data].isna(), :].empty)
        for data in ["node_i", "node_j", "x_pu", "capacity", "contingency"]:
            self.assertTrue(self.data.lines.loc[self.data.lines[data].isna(), :].empty)
    
    def test_data_preprocess(self):
        self.data.process_inflows()

    def test_save_data(self):
        folder = self.wdir.joinpath("data_output")
        if not folder.is_dir():
            folder.mkdir(parents=True)
        
        # Remove availability and demand for faster test execution
        self.data.availability_rt = pd.DataFrame()
        self.data.availability_da = pd.DataFrame()
        self.data.demand_el_rt = pd.DataFrame()
        self.data.demand_el_da = pd.DataFrame()
        self.data.net_export = pd.DataFrame()
 
        self.data.save_data(folder.joinpath("nrel_data"))
        self.assertTrue(folder.joinpath("nrel_data").with_suffix(".xlsx").is_file())
        self.assertTrue(folder.joinpath("nrel_data").with_suffix(".zip").is_file())

    def system_balance(self, result):
        model_horizon = result.result_attributes["model_horizon"]
        condition = result.data.demand_el.timestep.isin(model_horizon)

        return (result.G.G.sum() 
                - result.data.demand_el.loc[condition, "demand_el"].sum()  
                + result.INFEAS_EL_N_POS.INFEAS_EL_N_POS.sum() 
                - result.INFEAS_EL_N_NEG.INFEAS_EL_N_NEG.sum())

    def test_results_uniform_pricing(self):
        # obj 990129.5893227865
        # n-0 overloads = 15
        grid  = pomato.grid.GridTopology()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.joinpath("dispatch_market_results")
        result = pomato.data.Results(self.data, grid, folder)

        system_balance = self.system_balance(result)
        self.assertAlmostEqual(system_balance, 0)

        overload_n_0, _ = result.overloaded_lines_n_0()
        self.assertEqual(len(overload_n_0), 15)
        self.assertAlmostEqual(result.result_attributes["objective"]["Objective Value"], 
                               990129.5893227865)

        price = result.price()
        for t in result.model_horizon:
            self.assertTrue(all(price[price.t == t].marginal - price[price.t == t].marginal.mean() < 0.1))


    def test_results_nodal(self):
        # obj 2805962.178313506
        # n-0 : 0 OL; n-1 : 23 OL
        grid  = pomato.grid.GridTopology()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.joinpath("nodal_market_results")
        result = pomato.data.Results(self.data, grid, folder)

        system_balance = self.system_balance(result)
        self.assertAlmostEqual(system_balance, 0)

        overload_n_0, _ = result.overloaded_lines_n_0()
        overload_n_1, _ = result.overloaded_lines_n_1()
        
        self.assertEqual(len(overload_n_0), 0)
        self.assertEqual(len(overload_n_1), 23)
        self.assertAlmostEqual(result.result_attributes["objective"]["Objective Value"], 
                               2805962.178313506)
        self.assertTrue(len(result.price().marginal.unique()) > 1)

    def test_results_scopf(self):
        # obj 3899019.71757418
        # n-0 : 0 OL; n-1 : 29 OL
        grid  = pomato.grid.GridTopology()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.joinpath("scopf_market_results")
        result = pomato.data.Results(self.data, grid, folder)

        system_balance = self.system_balance(result)
        self.assertAlmostEqual(system_balance, 0)

        overload_n_0, _ = result.overloaded_lines_n_0()
        overload_n_1, _ = result.overloaded_lines_n_1()
        
        self.assertEqual(len(overload_n_0), 0)
        self.assertEqual(len(overload_n_1), 0)
        self.assertAlmostEqual(result.result_attributes["objective"]["Objective Value"], 
                               3899019.71757418)

class TestPomatoResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_dir = tempfile.TemporaryDirectory()
        wdir = Path(temp_dir.name)

        copytree(Path.cwd().joinpath("examples"), wdir)
        copytree(Path.cwd().joinpath("tests/test_data/nrel_result"), wdir)

        mato = pomato.POMATO(wdir=wdir, options_file="profiles/nrel118.json",
                                 logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/nrel_118.zip')
        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]
        mato.data.nodes.loc[R2_to_R3, "zone"] = "R3"
        mato.initialize_market_results(
            [wdir.joinpath("ntc_redispatch"),
             wdir.joinpath("ntc_market_results")])
        cls.market_result, cls.redispatch_result = mato.data.return_results()

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.market_result = None
        cls.redispatch_result = None

    def test_n_0_flow_cache(self):
        t0 = time.time()
        n_0 = self.market_result.n_0_flow()
        t1 = time.time()
        n_0_cached = self.market_result.n_0_flow()
        t2 = time.time()

        pd.testing.assert_frame_equal(n_0, n_0_cached)
        self.assertTrue( (t1-t0) > (t2 - t1))

    def test_n_1_flow_cache(self):
        t0 = time.time()
        n_1 = self.market_result.n_1_flow()
        t1 = time.time()
        n_1_cached = self.market_result.n_1_flow()
        t2 = time.time()

        pd.testing.assert_frame_equal(n_1, n_1_cached)
        self.assertTrue( (t1-t0) > (t2 - t1))

    def test_redispatch(self):
        # Not the corret market result set -> fails
        self.redispatch_result.result_attributes["corresponding_market_result_name"] = None
        gen = self.redispatch_result.redispatch()
        self.assertFalse(isinstance(gen, pd.DataFrame))

        self.redispatch_result.result_attributes["corresponding_market_result_name"] = "ntc_market_results"
        gen = self.redispatch_result.redispatch()
        self.assertTrue(isinstance(gen, pd.DataFrame))
        self.assertTrue(gen.delta_abs.sum() > 0)
        self.assertAlmostEqual(gen.delta.sum(), 0)

    def test_result_methods(self):

        self.market_result.net_position()
        self.market_result.infeasibility()
        self.market_result.demand()
        self.market_result.generation()
        self.market_result.storage_generation()
        self.market_result.price()     

        
if __name__ == '__main__':
    unittest.main()
