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
class TestPomatoMarketModel(unittest.TestCase):
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("domains"), ignore_errors=True)

    def test_run_ieee_init_invalid_option(self):
        mato = pomato.POMATO(wdir=self.wdir, options_file="INVALID_PATH",
                             logging_level=logging.ERROR)
        self.assertTrue(mato.options == pomato.tools.default_options())

    def test_run_ieee_init_no_option(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        self.assertTrue(mato.options == pomato.tools.default_options())

    def test_run_ieee_init_invalid_data(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        self.assertRaises(FileNotFoundError, mato.load_data, "INVALID_PATH")

    def test_init_ieee_mfile(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        mato.load_data('data_input/pglib_opf_case118_ieee.m')

    def test_init_ieee_matfile(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        mato.load_data('data_input/pglib_opf_case118_ieee.mat')

    