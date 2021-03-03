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
           
class TestPomato(unittest.TestCase):
    """Testing instantiation with different input files and options"""
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)
        shutil.copyfile(Path.cwd().joinpath("tests/test_data/unsupported_inputformat.xyz"), 
                        cls.wdir.joinpath("data_input/unsupported_inputformat.xyz"))

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.wdir = None
        cls.temp_dir = None
    
    def test_init_pomato_invalid_option(self):
        mato = pomato.POMATO(wdir=self.wdir, options_file="INVALID_PATH",
                             logging_level=logging.ERROR, file_logger=False)
        self.assertTrue(mato.options == pomato.tools.default_options())

    def test_init_pomato_no_option(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        self.assertTrue(mato.options == pomato.tools.default_options())

    def test_init_pomato_invalid_filepath(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        self.assertRaises(FileNotFoundError, mato.load_data, "INVALID_PATH")

    def test_init_pomato_invalid_fileformat(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        self.assertRaises(TypeError, mato.load_data, "data_input/unsupported_inputformat.xyz")
    
    def test_init_ieee_mfile(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/pglib_opf_case118_ieee.m')

    def test_init_ieee_matfile(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/pglib_opf_case118_ieee.mat')
    
    def test_init_de_zip(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/dataset_de.zip')

    def test_init_nrel_direct_filepath(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        filepath = self.wdir.joinpath('data_input/nrel_118.zip')
        mato.load_data(filepath)

    def test_init_nrel_xlsx(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/nrel_118.xlsx')
    
    def test_init_nrel_zip(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/nrel_118.zip')

    def test_init_nrel_folder(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)
        mato.load_data('data_input/nrel_118/')

    def test_save_model_data_to_folder(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False,
                             options_file="profiles/ieee118.json")
        mato.load_data('data_input/pglib_opf_case118_ieee.m')
        folder = self.wdir.joinpath("tmp_dir")
        Path.mkdir(folder)
        mato.create_grid_representation()
        mato.update_market_model_data(folder=folder)

    def test_rename_results(self):
        mato = pomato.POMATO(wdir=self.wdir, logging_level=logging.ERROR, file_logger=False)    
        mato.data.results = {"result_a": None, "result_b": None}
        mato.rename_market_result("result", "newname")
        self.assertTrue(all([r in mato.data.results for r in ["newname_a", "newname_b"]]))