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
class TestPomatoGeoPlot(unittest.TestCase):
    
    def setUp(self):
        self.wdir = Path.cwd().joinpath("examples")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_temp"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("data_output"), ignore_errors=True)
        shutil.rmtree(Path.cwd().joinpath("examples").joinpath("logs"), ignore_errors=True)

    def test_dynamic_plot(self):

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

        folder = self.wdir.parent.joinpath("tests/test_data/nrel_result/nodal_result_market")

        mato.data.process_results(folder, mato.grid)
        result = mato.data.results["nodal_result_market"]
        mato.create_geo_plot(plot_type="dynamic")

        mato.geo_plot.add_market_result(result, "test_test")
        mato.geo_plot.start_server()
        time.sleep(3)
        mato.geo_plot.stop_server()

        from pomato.visualization import geoplot_dynamic

if __name__ == '__main__':
    unittest.main()
