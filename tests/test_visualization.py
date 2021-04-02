import json
import logging
import os
import random
import shutil
import time
import unittest
import tempfile
from pathlib import Path
import plotly 

import numpy as np
import pandas as pd
from context import pomato, copytree

class TestPomatoVisualization(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.wdir = Path(cls.temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), cls.wdir)
        copytree(Path.cwd().joinpath("tests/test_data/nrel_result"), cls.wdir)
        copytree(Path.cwd().joinpath("tests/test_data/cbco_lists"), cls.wdir)

        cls.mato = pomato.POMATO(wdir=cls.wdir, options_file="profiles/nrel118.json",
                                  logging_level=logging.ERROR)
        cls.mato.load_data('data_input/nrel_118.zip')
        cls.mato.options["grid"]["precalc_filename"] = "cbco_nrel_118"

        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]
        cls.mato.data.nodes.loc[R2_to_R3, "zone"] = "R3"
        
        # For GeoPlot option 3
        cls.mato.data.lines["cb"] = False
        cls.mato.data.lines.loc["line001", "cb"] = True  
        
        market_folder = cls.mato.wdir.joinpath("dispatch_market_results")
        redispatch_folder = cls.mato.wdir.joinpath("dispatch_redispatch")
        scopf_folder = cls.mato.wdir.joinpath("scopf_market_results")

        cls.mato.initialize_market_results([market_folder, redispatch_folder])
        cls.mato.data.results["dispatch_redispatch"].result_attributes["corresponding_market_result_name"] = "dispatch_market_results"

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.mato = None
        cls.wdir = None
        cls.temp_dir = None

    # Generation: Dispatch, Storages
    def test_visualization_generation_plot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/generation_plot.html")
        nodes = list(self.mato.data.nodes.index)
        self.mato.visualization.create_generation_plot(result, nodes=nodes[:10], 
                                                       show_plot=False, filepath=filepath)
    def test_visualization_generation_pie(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/generation_plot.html")
        self.mato.visualization.create_generation_pie(result, show_plot=False, filepath=filepath)
    
    def test_visualization_generation_overview(self):
        results = [self.mato.data.results["dispatch_redispatch"], self.mato.data.results["dispatch_market_results"]]
        filepath = self.mato.wdir.joinpath("data_output/generation_overview.html")
        self.mato.visualization.create_generation_overview(results, show_plot=False, filepath=filepath)

    # Capacity and availability
    def test_visualization_available_intermittent_capacity_plot(self):
        data = self.mato.data
        filepath = self.mato.wdir.joinpath("data_output/available_capacity_plot.html")
        self.mato.visualization.create_available_intermittent_capacity_plot(data, show_plot=False, filepath=filepath)

    def test_visualization_installed_capacity_plot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/capacity_plot.html")
        self.mato.visualization.create_installed_capacity_plot(result, show_plot=False, filepath=filepath)

    def test_visualization_merit_order_plot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/capacity_plot.html")
        self.mato.visualization.create_merit_order(result, show_plot=False, filepath=filepath)

    def test_visualization_installed_capacity_plot_exception(self):
        self.assertRaises(TypeError, self.mato.visualization.create_installed_capacity_plot, pd.DataFrame)

    # Misc
    def test_visualization_cost_overview(self):
        results = [self.mato.data.results["dispatch_redispatch"], self.mato.data.results["dispatch_market_results"]]
        filepath = self.mato.wdir.joinpath("data_output/generation_overview.html")
        self.mato.visualization.create_cost_overview(results, show_plot=False, filepath=filepath)

    def test_fbmc_domain_plot(self):
        market_result = self.mato.data.results["dispatch_market_results"]
        folder = self.mato.wdir.joinpath("scopf_market_results")
        self.mato.initialize_market_results([folder])
        basecase = self.mato.data.results["scopf_market_results"]
        self.mato.options["grid"]["minram"] = 0.1
        self.mato.options["grid"]["sensitivity"] = 0.05

        fb_parameters = self.mato.fbmc.create_flowbased_parameters(basecase, gsk_strategy="gmax", reduce=False)
        fbmc_domain = pomato.visualization.FBDomainPlots(self.mato.data, fb_parameters)

        fbmc_domain.generate_flowbased_domains(("R1", "R2"), ["R1", "R3"], timesteps=["t0001"],
                                               commercial_exchange=market_result.EX)

        fbmc_domain.set_xy_limits_forall_plots()
        domain_plot = list(fbmc_domain.fbmc_plots.values())[0]
        #Plotly implementation
        self.mato.visualization.create_fb_domain_plot(domain_plot, show_plot=False)

    # Transmission
    def test_lineflow_plot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/lineplot.html")
        lines = list(self.mato.data.lines.index)[:2]
        self.mato.visualization.create_lineflow_plot(result, lines=lines, show_plot=False, filepath=filepath)
    
    # Geo Plot testing
    def test_geoplot(self):
        result = self.mato.data.results["dispatch_redispatch"]
        filepath = self.mato.wdir.joinpath("data_output/geoplot.html")
        highlight_nodes = list(self.mato.data.nodes.index[:2])
        self.mato.visualization.create_geo_plot(result, show_prices=True, 
                                                show_redispatch=True, 
                                                highlight_nodes=highlight_nodes,
                                                show_plot=False, filepath=filepath)

    def test_geoplot_various_options(self):
        result = self.mato.data.results["dispatch_redispatch"]
        self.mato.visualization.create_geo_plot(result, line_color_option=1, show_plot=False)
        self.mato.visualization.create_geo_plot(result, line_color_option=2, show_plot=False)
        self.mato.visualization.create_geo_plot(result, line_color_option=2, show_plot=False)

        self.mato.visualization.create_geo_plot(result, show_curtailment=True, show_plot=False)
        self.mato.visualization.create_geo_plot(result, show_infeasibility =True, show_plot=False)

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

class TestPomatoDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_dir = tempfile.TemporaryDirectory()
        wdir = Path(temp_dir.name)
        copytree(Path.cwd().joinpath("examples"), wdir)
        copytree(Path.cwd().joinpath("tests/test_data/nrel_result"), wdir)

        mato = pomato.POMATO(wdir=wdir, options_file="profiles/nrel118.json",
                                  logging_level=logging.ERROR)
        mato.load_data('data_input/nrel_118.zip')
        R2_to_R3 = ["bus118", "bus076", "bus077", "bus078", "bus079", 
                    "bus080", "bus081", "bus097", "bus098", "bus099"]
        mato.data.nodes.loc[R2_to_R3, "zone"] = "R3"        
        mato.initialize_market_results([mato.wdir.joinpath("ntc_market_results"), 
                                        mato.wdir.joinpath("ntc_redispatch"),
                                        mato.wdir.joinpath("nodal_market_results")
                                        ])
        mato.data.results["ntc_redispatch"].result_attributes["corresponding_market_result_name"] = "ntc_market_results"

        cls.dashboard = pomato.visualization.Dashboard(mato)
    def setUp(self):
        pass

    @classmethod
    def tearDownClass(cls):
        cls.dashboard = None

    @staticmethod
    def sample_click_line():
        return {
        	'points': [{
        		'customdata': ['line030', '600', '-205.69311257075077', '-461.1482685371414']
        	}]
        }
    
    @staticmethod
    def sample_selection_nodes():
        return {
            'points': [{
                'customdata': ['bus034', 'R2']
            }, {
                'customdata': ['bus037', 'R2']
            }, {
                'customdata': ['bus038', 'R2']
            }],
        }

    @staticmethod
    def sample_click_node():
        return {
            'points': [{
                'customdata': ['bus074', 'R1']
            }]
        }

    @staticmethod
    def sample_click_domain():
        return {
            'points': [{
                'customdata': ['line053', 'line056', '545.8776486981644']
            }]
        }

    def test_update_timestep_slider(self):
        result_name = "ntc_market_results"
        size = {"width": 400}
        slider_max, marks, value = self.dashboard.update_timestep_slider(result_name, size, None)

        self.assertTrue(value == 0)
        self.assertTrue(isinstance(marks, dict))
        self.assertTrue(isinstance(slider_max, int))

    def test_update_result_selection(self):
        
        options, value = self.dashboard.update_result_selection(None)
        self.assertTrue(isinstance(value, str))
        self.assertTrue(all([isinstance(o["value"], str) for o in options]))        

    ### Overview
    def test_update_installed_capacity_figure(self):
        result_name = "ntc_market_results"
        fig = self.dashboard.update_installed_capacity_figure(result_name)
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))
        
    def test_update_generation_overview(self):
        result_names = ["ntc_market_results", "ntc_redispatch"]
        fig = self.dashboard.update_generation_overview(result_names)
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))
    
    def test_update_cost_overview(self):
        result_names = ["ntc_market_results", "ntc_redispatch"]
        fig = self.dashboard.update_cost_overview(result_names)
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))
          
    ### Generation Methods
    def test_update_generation_page_components(self):
        result_name = "ntc_market_results"
        toggle_options, nodes_dropdown_options = self.dashboard.update_components_generation(result_name)
        self.assertTrue(all([isinstance(v, dict) for v in toggle_options]))
        self.assertTrue(all([isinstance(v, dict) for v in nodes_dropdown_options]))


    def test_update_graph_generation(self):
        result_name = "ntc_market_results"
        selection_data = self.sample_selection_nodes()
        fig = self.dashboard.update_graph_generation(result_name, selection_data)
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))

    def test_update_generation_geo_plot(self):
        result_name = "ntc_market_results"
        fig = self.dashboard.update_generation_geo_plot(result_name, [], 50, [])
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))

    def test_display_lineloading(self):
        string = self.dashboard.display_lineloading(50)
        self.assertTrue(isinstance(string, str))
    
    def test_display_plant_data(self):
        click_data = self.sample_click_node()
        result_name = "ntc_market_results"
        columns, data = self.dashboard.display_plant_data(result_name, click_data)
        self.assertTrue(all([isinstance(v, dict) for v in columns]))
        self.assertTrue(all([isinstance(v, dict) for v in data]))

    ### Transmission Methods
    def test_update_components_transmission(self):
        result_name = "ntc_market_results"
        options_lineflow, options_toggles = self.dashboard.update_components_transmission(result_name)
        self.assertTrue(all([isinstance(v, dict) for v in options_lineflow]))
        self.assertTrue(all([isinstance(v, dict) for v in options_toggles]))

    def test_update_transmission_geo_plot(self):
        result_name = "ntc_market_results"
        fig = self.dashboard.update_transmission_geo_plot(result_name, [], 0, 0, 50, [])
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))

    def test_click_lines(self):
        result_name = "ntc_market_results"
        click_data = self.sample_click_line()
        lines = self.dashboard.click_lines(result_name, click_data, [])
        self.assertTrue(isinstance(lines, list))
        self.assertTrue(len(lines) == 1)

    def test_update_graph_lines(self):
        result_name = "ntc_market_results"
        lines = ["line030"]
        fig = self.dashboard.update_graph_lines(result_name, lines)
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))

    def test_display_node_data(self):
        click_data = self.sample_click_node()
        result_name = "ntc_market_results"
        columns, data = self.dashboard.display_node_data(result_name, click_data)
        self.assertTrue(all([isinstance(v, dict) for v in columns]))
        self.assertTrue(all([isinstance(v, dict) for v in data]))

    ### Test Flowbased methods
    def test_update_domain_dropdown(self):
        result_name = "nodal_market_results"
        option_domain, _ = self.dashboard.update_domain_dropdown(result_name)
        self.assertTrue(all([isinstance(v, dict) for v in option_domain]))

    def test_update_domain_plot(self):
        result_name = "nodal_market_results"
        fig = self.dashboard.update_domain_plot(True, result_name, "gmax", 40, 0, 10, 25, 
                                                result_name, 0, "R1-R2", "R2-R3", True, False, 0)
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))
        fig = self.dashboard.update_domain_plot(True, result_name, "gmax", 40, 0, 10, 25, 
                                                result_name, 0, "R1-R2", "R2-R3", True, True, 500)
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))

    def test_update_fb_geo_plot(self):
        result_name = "nodal_market_results"
        click_data = self.sample_click_domain()
        fig = self.dashboard.update_fb_geo_plot(True, click_data, 0, result_name)
        self.assertTrue(isinstance(fig, plotly.graph_objs.Figure))

if __name__ == '__main__':
    unittest.main()
