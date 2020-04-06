import logging
import random
import shutil
import os
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

def copytree(src, dst, symlinks=False, ignore=None):
    # https://stackoverflow.com/a/12514470
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
            
# pylint: disable-msg=E1101
class TestPomatoMarketModel(unittest.TestCase):
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def test_run_ieee_init_invalid_option(self):
        mato = pomato.POMATO(wdir=self.wdir, options_file="INVALID_PATH",
                             logging_level=logging.ERROR)
        self.assertTrue(mato.options == pomato.tools.default_options())

    def test_run_ieee_init_no_option(self):
        mato = pomato.POMATO(wdir=self.wdir,
                             logging_level=logging.ERROR)
        self.assertTrue(mato.options == pomato.tools.default_options())

    def test_run_ieee_init_invalid_data(self):
        mato = pomato.POMATO(wdir=self.wdir, 
                             logging_level=logging.ERROR)
        self.assertRaises(FileNotFoundError, mato.load_data, "INVALID_PATH")

    def test_run_ieee(self):
        """Simply run the ieee case"""
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        mato.load_data('data_input/pglib_opf_case118_ieee.m')
        mato.create_geo_plot(name="IEEE_blank", show=False)

        ### Set Mock Julia Model and copy precalculated results
        mato.cbco_module.julia_instance = JuliaMockup()
        mato.market_model.julia_model = JuliaMockup()
        prepared_result = self.wdir.parent.joinpath('tests/test_data/dispatch_result/')
        to_folder = self.wdir.joinpath('data_temp/julia_files/results/dispatch_result') 
        to_folder.mkdir()
        copytree(prepared_result, to_folder)

        mato.options["optimization"]["type"] = "dispatch"
        mato.create_grid_representation()
        mato.update_market_model_data()
        mato.run_market_model()

        result_folder = mato.market_model.result_folders[0]
        result = mato.data.results[result_folder.name]
        result.default_plots()
        
        # df1, df2 = result.overloaded_lines_n_1()
        # df3, df4 = result.overloaded_lines_n_0()
        mato._clear_data()

    def test_run_de(self):
        """Simply run the DE case"""

        mato = pomato.POMATO(wdir=self.wdir, options_file="profiles/de.json",
                             logging_level=logging.ERROR)
        mato.load_data('data_input/dataset_de.xlsx')
        
        ### Set Mock Julia Model and copy precalculated results
        mato.cbco_module.julia_instance = JuliaMockup()
        mato.market_model.julia_model = JuliaMockup()
        prepared_result_market = self.wdir.parent.joinpath('tests/test_data/market_result_DE/')
        prepared_result_redispatch = self.wdir.parent.joinpath('tests/test_data/redispatch_DE/')
        to_folder_market = self.wdir.joinpath('data_temp/julia_files/results/market_result_DE') 
        to_folder_redispatch = self.wdir.joinpath('data_temp/julia_files/results/redispatch_DE') 
        to_folder_redispatch.mkdir()
        to_folder_market.mkdir()
        copytree(prepared_result_market, to_folder_market)
        copytree(prepared_result_redispatch, to_folder_redispatch)

        # Init Model 
        mato.create_grid_representation()
        mato.update_market_model_data()
        mato.run_market_model()

        # There are two market results loaded into data.results.
        # Specify redisp and market result for analysis
        redisp_result = mato.data.results[next(r for r in list(mato.data.results) if "redispatch" in r)]
        market_result = mato.data.results[next(r for r in list(mato.data.results) if "market_result" in r)]

        redisp_result.default_plots()
        market_result.default_plots()
        print(redisp_result, redisp_result)
        # # Check for Overloaded lines N-0, N-1 (should be non for N-0, but plenty for N-1)
        # df1, df2 = redisp_result.overloaded_lines_n_1()
        # df3, df4 = redisp_result.overloaded_lines_n_0()
        mato.create_geo_plot(name="DE", show=False)


