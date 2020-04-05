import logging
import random
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from context import pomato

# pylint: disable-msg=E1101
class TestPomatoData(unittest.TestCase):
    
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")
        self.options = pomato.tools.default_options()
        self.data = pomato.data.DataManagement(self.options, self.wdir)
        self.data.logger.setLevel(logging.ERROR)
        self.data.load_data('data_input/pglib_opf_case118_ieee.m')
    
    def test_plant_data(self):
        self.assertTrue(self.data.plants.loc[self.data.plants.mc_el.isna(), :].empty)

    def test_nodes_lines_data(self):
        
        self.assertEqual(len(self.data.nodes), 118)
        self.assertEqual(len(self.data.lines), 186)
        for data in ["slack"]:
            self.assertTrue(self.data.nodes.loc[self.data.nodes[data].isna(), :].empty)
        for data in ["node_i", "node_j", "b", "maxflow", "contingency"]:
            self.assertTrue(self.data.lines.loc[self.data.lines[data].isna(), :].empty)

if __name__ == '__main__':
    unittest.main()
