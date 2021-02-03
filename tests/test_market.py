import logging
import random
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tempfile

from context import pomato, copytree

class JuliaMockup():
    def __init__(self):
        self.is_alive = True
        self.solved = True
    def run(self, args):
        pass
    
# pylint: disable-msg=E1101
class TestPomatoMarketModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)

        pomato.tools.create_folder_structure(cls.wdir)
        cls.options = pomato.tools.default_options()
        cls.data = pomato.data.DataManagement(cls.options, cls.wdir)
        cls.data.logger.setLevel(logging.ERROR)        
        cls.data.load_data('data_input/pglib_opf_case118_ieee.m')
        cls.grid = pomato.grid.GridTopology()
        cls.grid.calculate_parameters(cls.data.nodes, cls.data.lines)
        cls.grid_model = pomato.grid.GridModel(cls.wdir, cls.grid, cls.data, cls.options)
        cls.grid_model.logger.setLevel(logging.ERROR)
    
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
            print(data)
            print(self.market_model.data_dir)
            self.assertTrue(self.market_model.data_dir.joinpath(f'{data}.csv').is_file())
        self.assertTrue(self.market_model.data_dir.joinpath('options.json').is_file())

    
    def test_market_model_missing_result(self):
        self.market_model.julia_model = JuliaMockup()
        self.grid_model.options["type"] = "ntc"
        self.grid_model.create_grid_representation()
        self.market_model.update_data()
        self.assertRaises(FileNotFoundError, self.market_model.run)