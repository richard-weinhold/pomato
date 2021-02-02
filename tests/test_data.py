import logging
import random
import shutil
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tempfile

from context import pomato, copytree

# pylint: disable-msg=E1101
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
        pass

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

    def test_save_results(self):
        grid  = pomato.grid.GridTopology()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.joinpath("dispatch_market_results")
        self.data.process_results(folder, grid)

        save_folder = self.wdir.joinpath("data_output")
        if not save_folder.is_dir():
            save_folder.mkdir()

        self.data.save_results(save_folder, "dummy_name")
        self.assertTrue(save_folder.joinpath("dummy_name_" + "dispatch_market_results").is_dir())

    def system_balance(self, result):
        model_horizon = result.result_attributes["model_horizon"]
        condition = result.data.demand_el.timestep.isin(model_horizon)

        return (result.G.G.sum() 
                - result.data.demand_el.loc[condition, "demand_el"].sum()  
                + result.INFEAS_EL_N_POS.INFEAS_EL_N_POS.sum() 
                - result.INFEAS_EL_N_NEG.INFEAS_EL_N_NEG.sum())

    def test_results_processing(self):
        grid  = pomato.grid.GridTopology()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.joinpath("dispatch_market_results")
        result = pomato.data.Results(self.data, grid, folder)

        system_balance = self.system_balance(result)
        self.assertAlmostEqual(system_balance, 0)

    def test_misc_result_methods(self):
        grid  = pomato.grid.GridTopology()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.joinpath("scopf_market_results")
        result = pomato.data.Results(self.data, grid, folder)
        result.output_folder = self.wdir.joinpath("data_output")
        
        result.curtailment()
        result.net_position()
        result.commercial_exchange("R1", "R2")
        result.infeasibility()
        res_share = result.res_share(["wind", "solar", "ror_ts"])
        result.demand()
        result.generation()
        result.storage_generation()
        result.price()

        self.assertTrue(0 < res_share < 1)
        self.assertRaises(TypeError, result.res_share)

    def test_redispatch_result(self):
        grid  = pomato.grid.GridTopology()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        market_folder = self.wdir.joinpath("dispatch_market_results")
        redispatch_folder = self.wdir.joinpath("dispatch_redispatch")

        self.data.process_results(market_folder, grid)
        self.data.process_results(redispatch_folder, grid)
        self.data.results["dispatch_redispatch"].result_attributes["corresponding_market_result_name"] = "dispatch_market_results"
        gen = self.data.results["dispatch_redispatch"].redispatch()

        self.assertTrue(gen.delta_abs.sum() > 0)
        self.assertAlmostEqual(gen.delta.sum(), 0)

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


if __name__ == '__main__':
    unittest.main()
