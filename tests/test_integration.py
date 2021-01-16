import logging
import random
import shutil
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime
import time 

import numpy as np
import pandas as pd

from context import pomato, copytree	
           
# pylint: disable-msg=E1101
class TestPomatoMarketModel(unittest.TestCase):
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("domains"), ignore_errors=True)
    
    def test_run_nrel(self):
        # What takes how long
        mato = pomato.POMATO(wdir=self.wdir, options_file="profiles/nrel118.json",
                             logging_level=logging.INFO)
        mato.load_data('data_input/nrel_118.zip')
        
        my_file = self.wdir.parent.joinpath('tests/test_data/cbco_nrel_118.csv')
        to_file = self.wdir.joinpath('data_temp/julia_files/cbco_data/cbco_nrel_118.csv')
        shutil.copyfile(str(my_file), str(to_file))

        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]
        mato.data.nodes.loc[R2_to_R3, "zone"] = "R3"

        mato.options["model_horizon"] = [0, 1]
        mato.options["constrain_nex"] = False
        mato.options["redispatch"]["include"] = True
        mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
        mato.options["infeasibility"]["electricity"]["bound"] = 200
        mato.options["infeasibility"]["electricity"]["cost"] = 1000
        mato.options["redispatch"]["cost"] = 20

        # %% NTC Model NEX = 0
        mato.data.results = {}
        mato.options["type"] = "ntc"
        mato.options["constrain_nex"] = True
        mato.data.set_default_net_position(0)
        mato.create_grid_representation()
        # mato.update_market_model_data()
        # mato.run_market_model()

        # NTC Model NTC = 100
        mato.data.results = {}
        mato.options["type"] = "ntc"
        mato.options["constrain_nex"] = False
        mato.create_grid_representation()
        mato.grid_representation.ntc["ntc"] = \
            mato.grid_representation.ntc["ntc"]*0.001
        # mato.update_market_model_data()
        # mato.run_market_model()

        # %% Zonal PTDF model
        mato.data.results = {}
        mato.options["type"] = "zonal"
        mato.options["grid"]["gsk"] = "gmax"
        mato.create_grid_representation()
        # mato.update_market_model_data()
        # mato.run_market_model()
        
        # %% Nodal PTDF model
        mato.data.results = {}
        mato.options["type"] = "nodal"
        mato.create_grid_representation()
        mato.update_market_model_data()
        mato.run_market_model()

        # %% FBMC basecase
        # mato.data.results = {}
        mato.options["timeseries"]["market_horizon"] = 168
        mato.options["type"] = "cbco_nodal"
        mato.grid_model.options["grid"]["cbco_option"] = "clarkson_base"
        mato.options["redispatch"]["include"] = False
        mato.options["chance_constrained"]["include"] = False
        mato.options["grid"]["sensitivity"] = 0.05

        mato.grid_model.options["grid"]["precalc_filename"] = "cbco_nrel_118"
        mato.create_grid_representation()
        # mato.update_market_model_data()
        # mato.run_market_model()

        result_name = next(r for r in list(mato.data.results))
        basecase = mato.data.results[result_name]
        mato.options["grid"]["minram"] = 0.1
        mato.options["grid"]["sensitivity"] = 0.05
        fb_parameters = mato.create_flowbased_parameters(basecase, gsk_strategy="gmax", reduce=False)

        # %% FBMC market clearing
        mato.data.results = {}
        mato.options["timeseries"]["market_horizon"] = 100
        mato.options["redispatch"]["include"] = True
        mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
        mato.create_grid_representation(flowbased_paramters=fb_parameters)
        mato.update_market_model_data()
        mato.run_market_model()
        mato._join_julia_instances()
