import logging
import random
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from context import pomato

# pylint: disable-msg=E1101
class TestPomatoData(unittest.TestCase):
    
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")
        self.options = pomato.tools.default_options()
        self.data = pomato.data.DataManagement(self.options, self.wdir)
        self.data.logger.setLevel(logging.ERROR)
        self.data.load_data('data_input/pglib_opf_case118_ieee.m')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def test_plant_data(self):
        self.assertTrue(self.data.plants.loc[self.data.plants.mc_el.isna(), :].empty)

    def test_nodes_lines_data(self):
        self.assertEqual(len(self.data.nodes), 118)
        self.assertEqual(len(self.data.lines), 186)
        for data in ["slack"]:
            self.assertTrue(self.data.nodes.loc[self.data.nodes[data].isna(), :].empty)
        for data in ["node_i", "node_j", "b", "maxflow", "contingency"]:
            self.assertTrue(self.data.lines.loc[self.data.lines[data].isna(), :].empty)

    def test_visualize_input_data(self):
        folder = self.wdir.joinpath("data_output")
        self.data.visualize_inputdata(folder, show_plot=False)
        self.assertTrue(folder.is_dir())
        self.assertTrue(folder.joinpath("zonal_demand.png").is_file())
        self.assertTrue(folder.joinpath("installed_capacity_by_type.png").is_file())

    def system_balance(self, result):
        return (result.G.G.sum() 
                - result.data.demand_el.demand_el.sum()  
                + result.INFEAS_EL_N_POS.INFEAS_EL_N_POS.sum() 
                - result.INFEAS_EL_N_NEG.INFEAS_EL_N_NEG.sum())

    def test_results_processing(self):
        grid  = pomato.grid.GridModel()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.parent.joinpath("tests/test_data/dispatch_result")
        result = pomato.data.ResultProcessing(self.data, grid, folder)

        system_balance = self.system_balance(result)
        self.assertAlmostEqual(system_balance, 0)

    def test_misc_result_methods(self):
        grid  = pomato.grid.GridModel()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.parent.joinpath("tests/test_data/scopf_result")
        result = pomato.data.ResultProcessing(self.data, grid, folder)
        result.output_folder = self.wdir.joinpath("data_output")
        
        result.check_infeasibilities()
        result.check_curtailment()
        result.net_position()
        result.commercial_exchange("z1", "z1")

        self.assertRaises(AttributeError, result.res_share)
        result.default_plots()

    def test_results_uniform_pricing(self):
        # obj 186053.45909199998
        # n-0 overloads = 3
        grid  = pomato.grid.GridModel()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.parent.joinpath("tests/test_data/dispatch_result")
        result = pomato.data.ResultProcessing(self.data, grid, folder)

        system_balance = self.system_balance(result)
        self.assertAlmostEqual(system_balance, 0)

        overload_n_0, _ = result.overloaded_lines_n_0()
        self.assertEqual(len(overload_n_0), 3)
        self.assertAlmostEqual(result.result_attributes["objective"]["Objective Value"], 
                               186053.45909199998)

        self.assertTrue(len(result.price().marginal.unique()) == 1)

    def test_results_nodal(self):
        # obj 186304.75403404975
        # n-0 : 0 OL; n-1 : 29 OL
        grid  = pomato.grid.GridModel()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.parent.joinpath("tests/test_data/nodal_result")
        result = pomato.data.ResultProcessing(self.data, grid, folder)

        system_balance = self.system_balance(result)
        self.assertAlmostEqual(system_balance, 0)

        overload_n_0, _ = result.overloaded_lines_n_0()
        overload_n_1, _ = result.overloaded_lines_n_1()
        
        self.assertEqual(len(overload_n_0), 0)
        self.assertEqual(len(overload_n_1), 29)
        self.assertAlmostEqual(result.result_attributes["objective"]["Objective Value"], 
                               186304.75403404975)
        self.assertTrue(len(result.price().marginal.unique()) > 1)

    def test_results_scopf(self):
        # obj 244192.23855578768
        # n-0 : 0 OL; n-1 : 29 OL
        grid  = pomato.grid.GridModel()
        grid.calculate_parameters(self.data.nodes, self.data.lines)
        folder = self.wdir.parent.joinpath("tests/test_data/scopf_result")
        result = pomato.data.ResultProcessing(self.data, grid, folder)

        system_balance = self.system_balance(result)
        self.assertAlmostEqual(system_balance, 0)

        overload_n_0, _ = result.overloaded_lines_n_0()
        overload_n_1, _ = result.overloaded_lines_n_1()
        
        self.assertEqual(len(overload_n_0), 0)
        self.assertEqual(len(overload_n_1), 0)
        self.assertAlmostEqual(result.result_attributes["objective"]["Objective Value"], 
                               244192.23855578768)

if __name__ == '__main__':
    unittest.main()
