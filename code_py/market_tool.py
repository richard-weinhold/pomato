import logging
import json
import pandas as pd
import subprocess
from pathlib import Path

from data_management import DataManagement
from grid_model import GridModel
from julia_interface import JuliaInterface
import bokeh_plot_interface as bokeh
import tools


def _logging_setup(wdir):
    # Logging setup
    logger = logging.getLogger('Log.MarketModel')
    logger.setLevel(logging.INFO)
    if len(logger.handlers) < 2:
        # create file handler which logs even debug messages
        if not wdir.joinpath("logs").is_dir():
            wdir.joinpath("logs").mkdir()
        file_handler = logging.FileHandler(wdir.joinpath("logs").joinpath('market_tool.log'))
        file_handler.setLevel(logging.DEBUG)
        # create console handler with a higher log level
#        handler = logging.StreamHandler()
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                         "%d.%m.%Y %H:%M")
        file_handler.setFormatter(fh_formatter)
        # Only message in Console
        ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(ch_formatter)
        # add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

class MarketTool(object):
    """ Main Class"""
    def __init__(self, **kwargs):
        self.wdir = Path.cwd()
        self.logger = _logging_setup(self.wdir)

        self._parse_kwargs(kwargs)
        self.logger.info("Market Tool Initialized")
        self.data = DataManagement()
        self.grid = GridModel(self.wdir)

        tools.create_folder_structure(self.wdir, self.logger)

        ## TODO, [rom]
        ## Not sure what does do, but all objectes should be initalized and
        ## then filled when needed
        ## Core Attributes
        self.grid_representation = None
        self.market_model = None
        self.bokeh_plot = None
        # Hidden option
        self._gams = False

    def load_data(self, filename):
        """init Data Model with loading the fata from file"""
        self.data.load_data(self.wdir, filename)
        if self.grid.is_empty:
            self.grid.build_grid_model(self.data.nodes, self.data.lines)

    def init_market_model(self):
        """init market model"""
        if not self.grid.is_empty:
            self.grid.build_grid_model(self.data.nodes, self.data.lines)
        if not self.grid_representation:
            self.create_grid_representation()

        self.market_model = JuliaInterface(self.wdir, self.data, self.opt_setup,
                                           self.grid_representation, self.model_horizon)

    def run_market_model(self):
        """ Run the model """
        self.market_model.run()

        if self.data.results:
            self.logger.info("Added Grid Model to Results Processing!")
            self.data.results.grid = self.grid

    def clear_data(self):
        self.logger.info("Resetting Data Object")
        self.data = DataManagement()

    def _parse_kwargs(self, kwargs):
        options = ['opt_file', # define location of a .json options file
                   'model_horizon' # restrict the amount of times that are solved
                   ]
        for ka in kwargs.keys():
            if not(ka in options):
                self.logger.warn("Unknown keyword: {}".format(ka))

        if 'opt_file' in kwargs.keys():
            with open(self.wdir.joinpath(kwargs['opt_file'])) as ofile:
                self.opt_setup = json.load(ofile)
                opt_str = "Optimization Options:" + json.dumps(self.opt_setup, indent=2) + "\n"
            self.logger.info(opt_str)
        if 'model_horizon' in kwargs.keys():
            self.model_horizon = kwargs['model_horizon']


    def plot_grid_object(self, name="plotmodel"):
        # TODO: Add all the add market data and grid data stuff to the API
        self.init_bokeh_plot(name)
        self.bokeh_plot.start_server()
        # self.bokeh_plot.stop_server()

    def create_grid_representation(self, precalc_filename=None, add_cbco=None):
        """Grid Representation as property, get recalculated when accessed via dot"""
        self.grid_representation =  self.grid.grid_representation(self.opt_setup["opt"], self.data.ntc, self.data.reference_flows,
                                                                  precalc_filename=precalc_filename, add_cbco=add_cbco)

    def init_bokeh_plot(self, name="default"):
        """init boke plot (saves market result and grid object)"""
        self.bokeh_plot = bokeh.BokehPlot(self.wdir)

        if not self.data.results:
            self.logger.info("No result available form market model!")

        else:
            folder = self.data.result_attributes["source"]
            self.logger.info(f"initializing bokeh plot with from folder: {str(folder)}")
            self.bokeh_plot.add_market_result(self.data.results, name)

