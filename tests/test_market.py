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


class JuliaMockup():
    def __init__(self):
        self.is_alive = True
        self.solved = True
    def run(self, args):
        pass
    
class TestPomatoMarketModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)

        pomato.tools.create_folder_structure(cls.wdir)
        cls.options = pomato.tools.default_options()
        cls.data = pomato.data.DataManagement(cls.options, cls.wdir)
        cls.data.load_data('data_input/nrel_118_original.zip')
        cls.grid = pomato.grid.GridTopology()
        cls.grid.calculate_parameters(cls.data.nodes, cls.data.lines)
        cls.grid_model = pomato.grid.GridModel(cls.wdir, cls.grid, cls.data, cls.options)
    
        cls.market_model = pomato.market_model.MarketModel(cls.wdir, cls.options, cls.data, 
                                                            cls.grid_model.grid_representation)

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.grid_model = None
        cls.grid = None
        cls.data = None
        cls.temp_dir = None

    def test_init(self):
        self.grid_model.options["type"] = "ntc"
        self.grid_model.create_grid_representation()
        self.market_model.update_data()
    
    def test_save_files(self):
        self.grid_model.options["type"] = "ntc"
        self.grid_model.create_grid_representation()
        self.market_model.update_data()

        for data in ["availability", "dclines", "demand_el", "demand_h", "grid", "heatareas", 
                     "inflows", "net_export", "net_position", "nodes", "ntc", "plant_types", 
                     "plants", "redispatch_grid", "slack_zones", "zones"]:
            self.assertTrue(self.market_model.data_dir.joinpath(f'{data}.csv').is_file())
        self.assertTrue(self.market_model.data_dir.joinpath('options.json').is_file())

    
    def test_market_model_missing_result(self):
        self.market_model.julia_model = JuliaMockup()
        self.grid_model.options["type"] = "ntc"
        self.grid_model.create_grid_representation()
        self.market_model.update_data()
        self.assertRaises(FileNotFoundError, self.market_model.run)


    def test_market_model_rolling_horizon_storage_levels(self):
        
        self.options["timeseries"]["market_horizon"] = 20
        self.options["model_horizon"] = [0, 168]
        self.options["plant_types"]["es"] = ["es"]
        model_horizon = self.market_model.model_horizon
        # Add Storage Plant 
        es = [["bus026", 0, 100, "es", "water", 0, 0, 100, 1, 1, None, 1000]]
        self.data.plants = self.data.plants.append(
            pd.DataFrame(index=["es1"], 
            columns=self.data.plants.columns, 
            data=es)
        ) 
        self.data.storage_level = pd.DataFrame(
            columns=self.data.storage_level.columns,
            data=[["t0001", "es1", 0.5], ["t0168", "es1", 1]]
        )

        # print("asd")
        storage_level = self.market_model.save_rolling_horizon_storage_levels()
        self.assertEqual(
            len(storage_level), 
            round(self.options["model_horizon"][1]/self.options["timeseries"]["market_horizon"])
        )
        self.assertTrue(all([s in [0.5, 1] for s in storage_level.storage_start]))
        self.assertTrue(all([s in [0.5, 1] for s in storage_level.storage_end]))

        self.options["timeseries"]["smooth_storage_level"] = True
        storage_level = self.market_model.save_rolling_horizon_storage_levels()

        self.assertFalse(all([s in [0.5, 1] for s in storage_level.storage_start]))
        self.assertFalse(all([s in [0.5, 1] for s in storage_level.storage_end]))
