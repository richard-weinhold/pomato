import logging
import random
import shutil
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from context import pomato, copytree	
           
# pylint: disable-msg=E1101
class TestFBMCModule(unittest.TestCase):
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("domains"), ignore_errors=True)

    def test_run_nrel(self):
        mato = pomato.POMATO(wdir=self.wdir, options_file="profiles/nrel118.json",
                             logging_level=logging.INFO)
        mato.load_data('data_input/nrel_118.zip')
        
        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]
        mato.data.nodes.loc[R2_to_R3, "zone"] = "R3"

        mato.options["optimization"]["timeseries"]["market_horizon"] = 10000
        mato.options["optimization"]["timeseries"]["redispatch_horizon"] = 24
        mato.options["optimization"]["constrain_nex"] = False
        mato.options["optimization"]["redispatch"]["include"] = True
        mato.options["optimization"]["redispatch"]["zones"] = list(mato.data.zones.index)
        mato.options["optimization"]["infeasibility"]["electricity"]["bound"] = 200
        mato.options["optimization"]["infeasibility"]["electricity"]["cost"] = 1000
        mato.options["optimization"]["redispatch"]["cost"] = 20

        folder = self.wdir.parent.joinpath("tests/test_data/nrel_result/scopf_result")
        mato.data.process_results(folder, mato.grid)

        basecase = mato.data.results[folder.name]
        mato.options["grid"]["minram"] = 0.1
        mato.options["grid"]["sensitivity"] = 0.05

        fbmc = pomato.fbmc.FBMCModule(mato.wdir, mato.grid, mato.data, mato.options)
        
        fbmc_gridrep_Gmax = fbmc.create_flowbased_parameters(basecase, gsk_strategy="gmax", reduce=False)
        fbmc_gridrep_G = fbmc.create_flowbased_parameters(basecase, gsk_strategy="dynamic", reduce=False)

        self.assertRaises(AssertionError, np.testing.assert_array_equal, 
                          fbmc_gridrep_G.loc[:, mato.data.zones.index].values, 
                          fbmc_gridrep_Gmax.loc[:, mato.data.zones.index].values)

