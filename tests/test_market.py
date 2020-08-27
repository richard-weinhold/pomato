import logging
import random
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from context import pomato, copytree

class JuliaMockup():
    def __init__(self):
        self.is_alive = True
        self.solved = True
    def run(self, args):
        pass
    
# pylint: disable-msg=E1101
class TestPomatoMarketModel(unittest.TestCase):
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")
        self.options = pomato.tools.default_options()
        self.data = pomato.data.DataManagement(self.options, self.wdir)
        self.data.logger.setLevel(logging.ERROR)        
        self.data.load_data('data_input/pglib_opf_case118_ieee.m')
        self.grid = pomato.grid.GridModel()
        self.grid.calculate_parameters(self.data.nodes, self.data.lines)
        self.grid_representation = pomato.cbco.GridRepresentation(self.wdir, self.grid, self.data, self.options)
        self.grid_representation.logger.setLevel(logging.ERROR)
    
        self.market_model = pomato.market_model.MarketModel(self.wdir, self.options, 
                                                            self.data, self.grid_representation)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def test_init(self):
        self.grid_representation.options["optimization"]["type"] = "ntc"
        self.grid_representation.create_grid_representation()
        self.market_model.update_data()
    
    def test_save_files(self):
        self.grid_representation.options["optimization"]["type"] = "ntc"
        self.grid_representation.create_grid_representation()
        self.market_model.update_data()

        for data in ["availability", "dclines", "demand_el", "demand_h", "grid", "heatareas", 
                     "inflows", "net_export", "net_position", "nodes", "ntc", "plant_types", 
                     "plants", "redispatch_grid", "slack_zones", "zones"]:
            print(data)
            print(self.market_model.data_dir)
            self.assertTrue(self.market_model.data_dir.joinpath(f'{data}.csv').is_file())
        self.assertTrue(self.market_model.data_dir.joinpath('options.json').is_file())

    def test_market_model_run(self):
        prepared_result = self.wdir.parent.joinpath('tests/test_data/dispatch_result/')
        to_folder = self.wdir.joinpath('data_temp/julia_files/results/dispatch_result') 
        to_folder.mkdir()
        copytree(prepared_result, to_folder)

        self.market_model.julia_model = JuliaMockup()
        self.grid_representation.options["optimization"]["type"] = "ntc"
        self.grid_representation.create_grid_representation()
        self.market_model.update_data()
        self.market_model.run()

        self.assertTrue(self.market_model.status == "solved")
    
    def test_market_model_missing_result(self):
        self.market_model.julia_model = JuliaMockup()
        self.grid_representation.options["optimization"]["type"] = "ntc"
        self.grid_representation.create_grid_representation()
        self.market_model.update_data()
        self.assertRaises(FileNotFoundError, self.market_model.run)