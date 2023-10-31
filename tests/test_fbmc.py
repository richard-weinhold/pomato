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


# pylint: disable-msg=E1101
class TestFBMCModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)
        copytree(Path.cwd().joinpath("tests/test_data"), cls.wdir)

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.mato = None
        cls.wdir = None
        cls.temp_dir = None

    def test_domain(self):
        mato = pomato.POMATO(wdir=self.wdir, options_file="profiles/nrel118.json",
                                 logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/nrel_118_original.zip')
        
        folder = self.wdir.joinpath("opf_market")
        mato.data.process_results(folder, mato.grid)

        basecase = mato.data.results["opf_market"]
        mato.options["fbmc"]["minram"] = 0.1
        mato.options["fbmc"]["gsk"] = "gmax"
        mato.fbmc.create_flowbased_parameters(basecase)
        mato.options["fbmc"]["gsk"] = "dynamic"

        mato.fbmc.create_flowbased_parameters(basecase)
        mato.logger.handlers[0].close()

        # self.assertRaises(AssertionError, np.testing.assert_almost_equal, 
        #                   fbmc_gridrep_G.loc[:, mato.data.zones.index].values, 
        #                   fbmc_gridrep_Gmax.loc[:, mato.data.zones.index].values)

