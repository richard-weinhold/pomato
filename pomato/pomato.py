"""
POMATO stands for (POwer MArket TOol) and it is very good.

Actually, POMATO is an easy to use tool for the comprehensive
analysis of the modern power market. It comprises the necessary power
engineering framework to account for power flow physics, thermal transport
constraints and security policies of the underlying transmission
infrastructure, depending on the requirements defined by the user.
POMATO was specifically designed to realistically model Flow-Based
Market-Coupling (FBMC) and is therefore equipped with a fast security
constrained optimal power flow algorithm and allows zonal market clearing
with endogenously generated flow-based parameters, and redispatch.

Model Structure
---------------
The model is structured in three interconnected parts:
    - Data Management: Data input, processing and result analysis.
    - Market Model: Calculation of the economic dispatch based on the
      dataset and chosen grid representation. asd asd
    - Grid Model: Providing grid representation for economic dispatch
      calculation in chosen granularity (N-0, N-1, FBMC, NTC, copperplate)
      and analysis for ex-post analysis of the market result.

Installation
------------
POMATO is written in python and julia. Python takes care of the data processing
and julia runs the economic dispatch and N-1 preprocessing. The folder
``/project_files`` contains environment files for python (3.6, anaconda) and julia (1.3).
Note julia has to be available on the PATH for POMATO to run.

.. code-block:: python

    import sys
    sys.path.append(pomato_path)
    from pomato import POMATO

Examples
--------
This release includes two examples :
    - The IEEE 118 bus network, which contains a singular timestep

          $ python /scripts/run_pomato_ieee.py

    - The DE case study, based on data from DIW DataDoc [insert more description]
      which is more complex and can be run for much longer timeframes

          $ python /scripts/run_pomato_de.py


However, the functionality of POMATO is best utilized when running inside a
IDE (e.g. Spyder) to acces POMATO objects and develop a personal script based
on the provided functionality and its results.

Release Status
--------------

POMATO is part of my PhD and actively developed. I'm not a software engineer,
thus the "program" is not written with robustness in mind. Expect errors,
bug, funky behavior, stupid code structures, hard-coded mess and lack of obvious
features.

"""

from pathlib import Path
import logging
import json

from pomato.data import DataManagement
from pomato.grid import GridModel
from pomato.market_model import MarketModel
from pomato.cbco.cbco_module import CBCOModule
from pomato.visualization.bokeh_interface import BokehPlot
import pomato.tools as tools

def _logging_setup(wdir, webapp):
    # Logging setup
    logger = logging.getLogger('Log.MarketModel')
    logger.setLevel(logging.INFO)
    if len(logger.handlers) < 2:
        # create file handler which logs even debug messages
        if not wdir.joinpath("logs").is_dir():
            wdir.joinpath("logs").mkdir()

        if webapp:
            logfile_path = wdir.joinpath("logs").joinpath('market_tool_webapp.log')
            # Clear Logfile
            with open(logfile_path, 'w'):
                pass
            file_handler = logging.FileHandler(logfile_path)
            file_handler.setLevel(logging.INFO)
            file_handler_formatter = logging.Formatter('%(asctime)s - %(message)s',
                                                       '%d.%m %H:%M')
            file_handler.setFormatter(file_handler_formatter)
            logger.addHandler(file_handler)

        else:
            file_handler = logging.FileHandler(wdir.joinpath("logs").joinpath('market_tool.log'))
            file_handler.setLevel(logging.DEBUG)
            file_handler_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                                       '%d.%m.%Y %H:%M')
            file_handler.setFormatter(file_handler_formatter)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler_formatter = logging.Formatter('%(levelname)s - %(message)s')
            console_handler.setFormatter(console_handler_formatter)

            # add the handlers to the logger
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
    return logger

class POMATO():
    """
    The main module joins POMATO's components, providing accessibility to the user.

    The core attributes are modules that provide specific functionalities to
    the model. At the center is an instance of the DataManagement module, that
    reads and processes input data, makes processed input data acessible to
    other modules and is the container for the results from the market model.


    Attributes
    ----------
    wdir : pathlib.Path
        Working directory, all necessary folders, temporary files and results
        are stored in relation to this directory.
    options : dict
        Dictionalry containing centralized all options relevant for running an economic
        dispatch. The options are roughly cathegorized into:

            - Optimization: defining model type, model horizon, inclusion of
              model features like heat or curtailment, management of infeasibility
              variables and definition of plant types considered.
            - Grid: Setting Capacity multiplier, applying redundancy removal
              for contingency analysis and its sensitivity, for zonal
              representation setting the node-zone weights.
            - Data: data type, data that is stored stacked (opposed to long),
              options to process marginal costs to be unique and default values.
    data : :class:`~pomato.data.DataManagement`
        Instance of DataManagement class containing all data, data processing,
        results and result processing. Is initialized empty, then data
        explicitly loaded.
    grid : :class:`~pomato.grid.GridModel`
        Object containing all grid information. Initializes empty and
        filled based on nodes and lines data when it is loaded and
        processed. Provides the PTDF matrices for N-0 and N-1 load flow analysis.
    cbco_module : :class:`~pomato.cbco.CBCOModule`
        The CBCO module provides the grid representation to the market model,
        based on the chosen configuration. Combines input data and the GridModel
        into a grid representation based on the chosen configuration (N-0, N-1,
        zonal, zonal CBCO, ntc, copper
        plate) and runs the redundancy removal algorithm needed for N-1
        grid representation.
    grid_representation : dict
        The output of the cbco_module. A dictionary containing all information
        for the market model to account for the chosen network representation.
    market_model : :class:`~pomato.market_model.MarketModel`
        Module containing and managing the market model. This includes storing
        the necessary data, running and managing a julia process instance and
        initializing the result object inside ``data``.
    bokeh_plot : :class:`~pomato.visualization.BokehPlot`
        Enabling a geographic visualization of the market results through the
        library ``Bokeh``. This module processes the results and input data to
        create a static map plot containing mean line loadings or a
        dynamic/interactive plot for a timeseries. The latter is implemented
        through a bokeh server which requires some subprocess management.

    Parameters
    ----------
    wdir : pathlib.Path, optional
        Working directory, defaulting to Path.cwd() if not specified, should
        be the root of the POMATO folder.
    options_file : str, optional
        Providing the name of an option file, usually located in the
        ``/profiles`` folder. If not provided, using default options as
        defined in tools.
    webapp : bool, optional
        Optional parameter to set logging settings when initializing POMATO
        as part of the included webapp.

    """

    def __init__(self, options_file=None, webapp=False, wdir=None):
        self.wdir = wdir if wdir else Path.cwd()
        self.logger = _logging_setup(self.wdir, webapp)
        self.logger.info("Market Tool Initialized")
        tools.create_folder_structure(self.wdir, self.logger)

        # Core Attributes
        self.options = None
        self.initialize_options(options_file)

        self.data = DataManagement(self.options, self.wdir)
        self.grid = None
        self.cbco_module = None
        self.grid_representation = None
        self.market_model = None
        self.bokeh_plot = None

    def initialize_options(self, options_file):
        """Initialize options file.

        Parameters
        ----------
        options_file : str, optional
            Providing the name of an option file, usually located in the ``/profiles``
            folder. If not provided, using default options as defined in tools.

        """
        try:
            with open(self.wdir.joinpath(options_file)) as ofile:
                self.options = json.load(ofile)
                opt_str = "Optimization Options:" + json.dumps(self.options, indent=2) + "\n"
            self.logger.info(opt_str)

        except FileNotFoundError:
            self.logger.warning("No or invalid options file provided, using default options")
            self.options = tools.default_options()
            opt_str = "Optimization Options:" + json.dumps(self.options, indent=2) + "\n"
            self.logger.info(opt_str)
        except BaseException as unknown_exception:
            self.logger.exception("Error: %s", unknown_exception)

    def load_data(self, filename):
        """Load data into data_management module.

        The data attribute is initialized empty and explicitly filled with
        this method. When done reading and processing data, the grid model is
        also initialized.

        Parameters
        ----------
        filename : str
            Providing the name of a data file, usually located in the
            ``/input_data`` folder. Excel files and matpower cases are supported.
        """
        self.data.load_data(filename)
        self.grid = GridModel(self.data.nodes, self.data.lines)

    def init_market_model(self):
        """Initialize the market model.

        The Market model is initialized with the data object and a grid
        representation based on the chosen configuration.

        """
        if not self.grid_representation:
            self.create_grid_representation()

        if not self.market_model:
            self.market_model = MarketModel(self.wdir, self.options)
            self.market_model.update_data(self.data, self.options, self.grid_representation)

    def update_market_model_data(self):
        """Update data within an instance of the market model.

        It is possible to change the data in the market mode and re-run without
        explicitly re-initializing the module.
        """
        if not self.market_model:
            self.init_market_model()
        else:
            self.market_model.update_data(self.data, self.options, self.grid_representation)

    def run_market_model(self):
        """Run the market model."""
        if not self.market_model:
            self.init_market_model()

        self.market_model.run()

        if self.data.results:
            self.logger.info("Adding Grid Model to Results Processing!")
            self.data.results.grid = self.grid

    def _clear_data(self):
        """Reset DataManagement Class."""
        self.logger.info("Resetting Data Object")
        self.data = DataManagement()

    def create_grid_representation(self):
        """Grid Representation as property.

        Creates grid representation to be used in the market model.
        """
        self.cbco_module = CBCOModule(self.wdir, self.grid, self.data, self.options)
        self.cbco_module.create_grid_representation()
        self.grid_representation = self.cbco_module.grid_representation

    def init_bokeh_plot(self, name="default", bokeh_type="static"):
        """Initialize bokeh plot based on the dataset and a market result.

        Parameters
        ----------
        name: str, optional
            Name defaults to 'default' is is used to identify the initialized
            market result in the plot itself and to name the folder within
            the ``data_temp/bokeh_files`` folder.

        """
        self.bokeh_plot = BokehPlot(self.wdir, bokeh_type=bokeh_type)

        if not self.data.results:
            self.logger.info("No result available form market model!")

        else:
            folder = self.data.results.result_attributes["source"]
            self.logger.info("initializing bokeh plot with from folder: %s", str(folder))
            self.bokeh_plot.create_static_plot(self.data.results)