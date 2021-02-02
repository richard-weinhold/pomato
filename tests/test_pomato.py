import logging
import random
import shutil
import os
import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from context import pomato, copytree	
           
# pylint: disable-msg=E1101
class TestPomato(unittest.TestCase):
    """Testing instantiation with different input files and options"""
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        pass
    
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
    
    
    def test_init_de_zip(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.INFO)
        mato.load_data('data_input/dataset_de.zip')

    def test_init_nrel_direct_filepath(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        filepath = self.wdir.joinpath('data_input/nrel_118.zip')
        mato.load_data(filepath)
    def test_init_nrel_xlsx(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        mato.load_data('data_input/nrel_118.xlsx')
    
    def test_init_nrel_zip(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        mato.load_data('data_input/nrel_118.zip')

    def test_init_nrel_folder(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR)
        mato.load_data('data_input/nrel_118/')