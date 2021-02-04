import json
import logging
import os
import random
import shutil
import time
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from context import pomato, copytree

# pylint: disable-msg=E1101
class TestPomatoVisualization(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)
        copytree(Path.cwd().joinpath("tests/test_data/nrel_result"), cls.wdir)

        cls.mato = pomato.POMATO(wdir=cls.wdir, options_file="profiles/nrel118.json",
                                  logging_level=logging.ERROR)
        cls.mato.load_data('data_input/nrel_118.zip')
        
        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]
        cls.mato.data.nodes.loc[R2_to_R3, "zone"] = "R3"
        # For GeoPlot option 3
        cls.mato.data.lines["cb"] = False
        cls.mato.data.lines.loc["line001", "cb"] = True  
        
        market_folder = cls.mato.wdir.joinpath("dispatch_market_results")
        redispatch_folder = cls.mato.wdir.joinpath("dispatch_redispatch")
        cls.mato.initialize_market_results([market_folder, redispatch_folder])
        cls.mato.data.results["dispatch_redispatch"].result_attributes["corresponding_market_result_name"] = "dispatch_market_results"

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.mato = None
        cls.wdir = None
        cls.temp_dir = None

    def test_visualization_generation_plot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/generation_plot.html")
        self.mato.visualization.create_generation_plot(result, show_plot=False, filepath=filepath)

    def test_visualization_create_installed_capacity_plot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/capacity_plot.html")
        self.mato.visualization.create_installed_capacity_plot(result, show_plot=False, filepath=filepath)

    def test_visualization_create_storage_plot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/storage_plot.html")
        self.mato.visualization.create_storage_plot(result, show_plot=False, filepath=filepath)

    def test_visualization_create_generation_overview(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/generation_overview.html")
        self.mato.visualization.create_generation_overview(result, show_plot=False, filepath=filepath)

    def test_fbmc_domain_plot(self):
        folder = self.mato.wdir.joinpath("scopf_market_results")
        self.mato.initialize_market_results([folder])
        basecase = self.mato.data.results["scopf_market_results"]
        self.mato.options["grid"]["minram"] = 0.1
        self.mato.options["grid"]["sensitivity"] = 0.05
        self.mato.fbmc.calculate_parameters()

        fb_parameters = self.mato.fbmc.create_flowbased_parameters(basecase, gsk_strategy="gmax", reduce=False)
        fbmc_domain = pomato.visualization.FBDomainPlots(self.mato.data, fb_parameters)

        fbmc_domain.generate_flowbased_domain(("R1", "R2"), ["R1", "R3"], "t0001", "nrel")
        fbmc_domain.save_all_domain_plots(self.mato.wdir.joinpath("data_output"), include_ntc=True)
        fbmc_domain.save_all_domain_info(self.mato.wdir.joinpath("data_output"))
        

    def test_geoplot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/geoplot.html")
        self.mato.visualization.create_geo_plot(result, show_prices=True, show_redispatch=True, 
                                                show_plot=False, filepath=filepath)

    def test_geoplot_timestep(self):    
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/geoplot_timestep.html")                                     
        self.mato.visualization.create_geo_plot(result, show_prices=True, show_redispatch=True, 
                                                show_plot=False, timestep=0, filepath=filepath)
    def test_zonal_geoplot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/geoplot_zonal.html")
        self.mato.visualization.create_zonal_geoplot(result, show_plot=False, filepath=filepath)

    def test_zonal_geoplot_timestep(self):    
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/geoplot_zonal_timestep.html")                                     
        self.mato.visualization.create_zonal_geoplot(result, show_plot=False, timestep=0, filepath=filepath)

    def test_dashboard(self):
        self.mato.start_dashboard()
        time.sleep(5)
        self.mato.stop_dashboard()

if __name__ == '__main__':
    unittest.main()
