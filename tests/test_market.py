import logging
import random
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from context import pomato

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
        self.grid  = pomato.grid.GridModel(self.data.nodes, self.data.lines)
        self.cbco_module = pomato.cbco.CBCOModule(self.wdir, self.wdir, self.grid, self.data, self.options)
        self.cbco_module.logger.setLevel(logging.ERROR)

        self.market_model = pomato.market_model.MarketModel(self.wdir, Path("dummy"), self.options)

    # @classmethod
    # def tearDownClass(cls):
    #     shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
    #     shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
    #     shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def test_init(self):
        self.cbco_module.options["optimization"]["type"] = "ntc"
        self.cbco_module.create_grid_representation()
        self.market_model.update_data(self.data, self.options, self.cbco_module.grid_representation)
    
    def test_save_files(self):
        self.cbco_module.options["optimization"]["type"] = "ntc"
        self.cbco_module.create_grid_representation()
        self.market_model.update_data(self.data, self.options, self.cbco_module.grid_representation)

        for data in ["availability", "dclines", "demand_el", "demand_h", "grid", "heatareas", 
                     "inflows", "net_export", "net_position", "nodes", "ntc", "plant_types", 
                     "plants", "redispatch_grid", "slack_zones", "zones"]:
            self.assertTrue(self.market_model.data_dir.joinpath(f'data/{data}.csv').is_file())
        self.assertTrue(self.market_model.data_dir.joinpath('data/options.json').is_file())

    def test_market_model_run(self):
        self.market_model.julia_model = JuliaMockup()
        self.cbco_module.options["optimization"]["type"] = "ntc"
        self.cbco_module.create_grid_representation()
        self.market_model.update_data(self.data, self.options, self.cbco_module.grid_representation)
        
        self.market_model.run()
        self.assertTrue(self.market_model.status == "solved")
