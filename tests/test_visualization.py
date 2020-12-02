import json
import logging
import os
import random
import shutil
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from context import pomato

# pylint: disable-msg=E1101
class TestPomatoVisualization(unittest.TestCase):
    
    def setUp(self):
        wdir = Path.cwd().joinpath("examples")
        self.mato = pomato.POMATO(wdir=wdir, options_file="profiles/nrel118.json",
                                  logging_level=logging.ERROR)
        self.mato.load_data('data_input/nrel_118.zip')
        
        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]
        self.mato.data.nodes.loc[R2_to_R3, "zone"] = "R3"

        folder = wdir.parent.joinpath("tests/test_data/nrel_result/nodal_market_results")
        self.mato.data.process_results(folder, self.mato.grid)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def test_visualization_generation_plot(self):
        result = self.mato.data.results["nodal_market_results"]
        self.mato.visualization.create_generation_plot(result)

    def test_visualization_create_installed_capacity_plot(self):
        result = self.mato.data.results["nodal_market_results"]
        self.mato.visualization.create_installed_capacity_plot(result)

    def test_visualization_create_storage_plot(self):
        result = self.mato.data.results["nodal_market_results"]
        self.mato.visualization.create_storage_plot(result)

    def test_geoplot_static(self):
        print(self.mato.data.results)
        self.mato.create_geo_plot()

    def test_geoplot_dynamic(self):

        result = self.mato.data.results["nodal_market_results"]
        self.mato.geo_plot.add_market_result(result, "test_test")
        self.mato.geo_plot.start_server()
        time.sleep(3)
        self.mato.geo_plot.stop_server()

        from pomato.visualization import geoplot_dynamic

if __name__ == '__main__':
    unittest.main()
