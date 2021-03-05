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
import tempfile
from context import pomato, copytree
           
# pylint: disable-msg=E1101
class TestPomatoMarketModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)
        copytree(Path.cwd().joinpath("tests/test_data/cbco_lists"), cls.wdir)

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.mato = None
        cls.wdir = None
        cls.temp_dir = None
    
    def test_run_nrel(self):
        # What takes how long
        mato = pomato.POMATO(wdir=self.wdir, options_file="profiles/nrel118.json",
                                 logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/nrel_118.zip')
        
        my_file = self.wdir.joinpath('cbco_nrel_118.csv')
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

        # %% Nodal Basecase
        mato.data.results = {}
        mato.options["type"] = "nodal"
        mato.create_grid_representation()
        mato.update_market_model_data()
        mato.run_market_model()
        result_name = next(r for r in list(mato.data.results))
        basecase = mato.data.results[result_name]
        mato.options["grid"]["minram"] = 0.2
        mato.options["grid"]["sensitivity"] = 0.1
        fb_parameters = mato.create_flowbased_parameters(basecase, gsk_strategy="gmax", reduce=False)

        # %% FBMC market clearing
        mato.options["redispatch"]["include"] = True
        mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
        mato.create_grid_representation(flowbased_paramters=fb_parameters)
        mato.update_market_model_data()
        mato.run_market_model()
        mato.visualization.create_generation_overview(mato.data.results.values(), show_plot=False)
        mato._join_julia_instances()
