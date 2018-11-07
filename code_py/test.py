"""
Unit Test for Grid and Market Model
"""
from pathlib import Path
import unittest
import pickle
import json
import pandas as pd
from pandas.util.testing import assert_frame_equal
from market_tool import MarketTool

class TestMarketTool(unittest.TestCase):
    """Test Class for the Market Tool"""
    @classmethod
    def setUpClass(cls):
        """Set up Market Tool"""
        cls.wdir = Path.cwd()
        cls.MT = MarketTool(cls.wdir, "test_data.xlsx", "opt_setup.json", 2017,
                         6, model_horizon=range(1, 100))
        cls.MT.data.nodes.net_injection = 0
        
    def test_cbco_slack_zones(self):
        """test cbco and slack zones"""
        grid_rep = self.MT.grid_representation
        with open(self.wdir.joinpath("unittest").joinpath("cbco.json"), "r") as jsonfile:
            reference_cbco = json.load(jsonfile)
            
        with open(self.wdir.joinpath("unittest").joinpath("slack_zones.json"), "r") as jsonfile:
            reference_slack_zones = json.load(jsonfile)
        self.assertEqual(reference_slack_zones, grid_rep["slack_zones"])
        self.assertEqual(len(grid_rep["cbco"]), len(reference_cbco)) 

class TestGridModel(TestMarketTool):
    """Test Class for Grid Model"""
    @classmethod
    def setUpClass(cls):
        """Setup Reference Grid """
        print("setUpClass Grid Model")

        super(TestGridModel, cls).setUpClass()
        with open(cls.wdir.joinpath("unittest").joinpath("grid.file"), "rb") as gridfile:
            cls.reference_grid = pickle.load(gridfile)

    def test_ptdf(self):
        self.assertTrue(len(self.MT.grid.ptdf) == len(self.reference_grid.ptdf))

    def test_lodf(self):
        self.assertTrue(len(self.MT.grid.lodf) == len(self.reference_grid.lodf))

    def test_n_1_ptdf(self):
        self.assertTrue(len(self.MT.grid.n_1_ptdf) == len(self.reference_grid.n_1_ptdf))

def load_jl_market_result_by_symbol(symb, directory):
    """Function to load Julia Results as reference"""
    return pd.read_json(directory.joinpath(symb + '.json'), orient="index").sort_index()

class TestMarketModel(TestMarketTool):
    """Test Ckass for Market Model"""
    @classmethod
    def setUpClass(cls):
        """Setup and run Market Model"""
        print("setUpClass Market Model")
        super(TestMarketModel, cls).setUpClass()
        cls.MT.init_market_model()
        cls.MT.market_model.run()

    def test_objective_value(self):
        """test objective value"""
        with open(self.wdir.joinpath("unittest/jl_result").joinpath("misc_result.json"), "r") as jsonfile:
                reference_objective = json.load(jsonfile)
        self.assertEqual(reference_objective["Objective Value"], self.MT.market_model.return_results("COST"))
        
    import os.path
    
    def test_results(self):
        """test all model results except objective value"""
        for file in self.wdir.joinpath("unittest/jl_result").rglob("*.json"):
            if file.name != "misc_result.json":
                symb = file.name.split(".")[0]
                reference_result = load_jl_market_result_by_symbol(symb, self.wdir.joinpath("unittest").joinpath("jl_result"))
                assert_frame_equal(reference_result, self.MT.market_model.return_results(symb))

    def test_n_1(self):
        """check N-1"""
        print("N-1 Test")
        overloaded_lines = self.MT.check_n_1_for_marketresult()
        self.assertEqual(overloaded_lines, {})

if __name__ == "__main__":
    unittest.main()





