import logging
import os
import random
import shutil
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pomato
from pomato.tools import copytree


# pylint: disable-msg=E1101
class TestPomatoMarketModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.wdir = None
        cls.temp_dir = None
    
    def test_run_nrel(self):
        # What takes how long
        mato = pomato.POMATO(wdir=self.wdir, options_file="profiles/nrel118.json",
                                 logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/nrel_118_original.zip')
        
        mato.options["model_horizon"] = [0, 1]
        mato.options["redispatch"]["include"] = False
        mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
        mato.options["infeasibility"]["electricity"]["bound"] = 200
        mato.options["infeasibility"]["electricity"]["cost"] = 1000
        mato.options["redispatch"]["cost"] = 20

        def system_balance(result):
            model_horizon = result.result_attributes["model_horizon"]
            condition = result.data.demand_el.timestep.isin(model_horizon)
            return (result.G.G.sum() 
                    - result.data.demand_el.loc[condition, "demand_el"].sum()  
                    + result.INFEASIBILITY_EL_POS.INFEASIBILITY_EL_POS.sum() 
                    - result.INFEASIBILITY_EL_NEG.INFEASIBILITY_EL_NEG.sum())

        # %% Nodal Basecase
        mato.data.results = {}
        mato.options["type"] = "opf"
        mato.create_grid_representation()
        mato.update_market_model_data()
        mato.run_market_model()
        result_name = next(r for r in list(mato.data.results))
        basecase = mato.data.results[result_name]

        self.assertAlmostEqual(system_balance(basecase), 0)
        self.assertAlmostEqual(basecase.INJ.INJ.sum(), 0)

        mato.options["fbmc"]["minram"] = 0.2
        mato.options["fbmc"]["lodf_sensitivity"] = 0.1
        mato.options["fbmc"]["cne_sensitivity"] = 0.2
        fb_parameters = mato.create_flowbased_parameters(basecase)
        fbmc_domain = pomato.visualization.FBDomainPlots(mato.data, fb_parameters)
        fbmc_domain.generate_flowbased_domains(("R1", "R2"), ["R1", "R3"], timesteps=["t0001"])
        mato.visualization.create_fb_domain_plot(fbmc_domain.fbmc_plots[0], show_plot=False)

        # %% FBMC market clearing
        mato.options["redispatch"]["include"] = True
        mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
        mato.create_grid_representation(flowbased_paramters=fb_parameters)
        mato.update_market_model_data()
        mato.run_market_model()
        mato.visualization.create_generation_overview(list(mato.data.results.values()), show_plot=False)
        mato._join_julia_instances()
