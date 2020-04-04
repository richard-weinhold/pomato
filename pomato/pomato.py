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
and julia runs the economic dispatch and N-1 redundancy removal algorithm. 

The recommended way to install POMATO is through *pip* by creating a virtual 
environment and install pomato into it::

    python -m venv pomato && /pomato/Scripts/activate
    pip install git+https://github.com/korpuskel91/pomato.git

After this is completed pomato can be imported in python::

    from pomato import POMATO

Pomato functions from a *working directory*, ideally the project folder including 
the virtual environment, and creates additional folders for results, temporary 
data and logs. The way we use pomato is illustrated by the *examples* folder, 
cloning its contents into the *working directory* allows to run the included examples.

Note: To install pomato in its current state, julia and gurobi must be available on 
the PATH within the venv/project. See [Gurobi.jl](https://github.com/JuliaOpt/Gurobi.jl) 
for notes on the installation. 

Examples
--------
This release includes two examples in the *examples* folder. Including the contents of 
this folder into the pomato working directory will allow their execution:

  - The IEEE 118 bus network, which contains a singular timestep. The data is available under 
    open license at [https://power-grid-lib.github.io/](https://power-grid-lib.github.io/) 
    and rehosted in this repository.::
    
        $ python /run_pomato_ieee.py

  - The DE case study, based on data from [ELMOD-DE](http://www.diw.de/elmod) which is 
    openly available and described in detail in 
    [DIW DataDocumentation 83](https://www.diw.de/documents/publikationen/73/diw_01.c.528927.de/diw_datadoc_2016-083.pdf) which represents a more complex system and can be run for longer model horizon (although 
    shortened to allow to host this data in this git).::

        $ python /run_pomato_de.py


However, the functionality of POMATO is best utilized when running inside a
IDE (e.g. Spyder) to access POMATO objects and develop a personal script based
on the provided functionality and its results.



However, the functionality of POMATO is best utilized when running inside a
IDE (e.g. Spyder) to acces POMATO objects and develop a personal script based
on the provided functionality and its results.

Release Status
--------------

This release covers all features and a big part of the documentation. The FBMCModule is stil
changing very often and is not documented. The julia code also lacks documentation until we figure
out how to include both julia and python code into one shpinx script.

POMATO is part of my PhD and actively developed by Robert and myself. WE are notsoftware engineers,
thus the "program" is not written with robustness in mind. Expect errors, bug, funky behavior,
stupid code structures, hard-coded mess and lack of obvious features.

Related Publications
--------------------

- [Weinhold and Mieth (2019), Fast Security-Constrained Optimal Power Flow through 
   Low-Impact and Redundancy Screening](https://arxiv.org/abs/1910.09034)
- [Schönheit, Weinhold, Dierstein (2020), The impact of different strategies for 
  generation shift keys (GSKs) on the flow-based market coupling domain: A model-based analysis 
  of Central Western Europe](https://www.sciencedirect.com/science/article/pii/S0306261919317544)

Acknowledgments
---------------

Richard and Robert would like to aknowledge the support of Reiner Lemoine-Foundation, the Danish 
Energy Agency and Federal Ministry for Economic Affairs and Energy (BMWi).
Robert Mieth is funded by the Reiner Lemoine-Foundation scholarship. Richard Weinhold is funded 
by the Danish Energy Agency. The development of POMATO and its applications was funded by 
BMWi in the project “Long-term Planning and Short-term Optimization of the German Electricity 
System Within the European Context” (LKD-EU, 03ET4028A).
"""

import json
import logging
from pathlib import Path

import pomato
import pomato.tools as tools
from pomato.cbco.cbco_module import CBCOModule
from pomato.data import DataManagement, ResultProcessing
from pomato.grid import GridModel
from pomato.market_model import MarketModel
from pomato.visualization.bokeh_interface import BokehPlot


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

    def __init__(self, wdir, options_file=None, webapp=False):

        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])

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
            with open(self.wdir.joinpath(options_file)) as opt_file:
                self.options = json.load(opt_file)
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
        self.data = DataManagement(self.options, self.wdir)
        self.data.load_data(filename)
        self.grid = GridModel(self.data.nodes, self.data.lines)
        
        self.cbco_module = CBCOModule(self.wdir, self.package_dir, self.grid, self.data, self.options)

        self.market_model = MarketModel(self.wdir, self.package_dir, self.options)
        self._start_julia_instances()
    def init_market_model(self):
        """Initialize the market model.

        The Market model is initialized with the data object and a grid
        representation based on the chosen configuration.

        """
        if not self.grid_representation:
            self.create_grid_representation()

        if not self.market_model:
            self.market_model = MarketModel(self.wdir, self.package_dir, self.options)
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

    def initialize_market_results(self, result_folders):
        """Initionalizes market results from a list of folders.

        Parameters
        ----------
        result_folders : list
            List of folders containing market resaults.
        """
        for folder in result_folders:
            self.data.results[folder.name] = ResultProcessing(self.data, self.grid, folder)

    def run_market_model(self):
        """Run the market model."""
        if not self.market_model:
            self.init_market_model()

        self.market_model.run()

        if self.market_model.status == "solved":
            self.initialize_market_results(self.market_model.result_folders)
        else:
            self.logger.warning("Market Model not successfully run!")

    def _clear_data(self):
        """Reset DataManagement Class."""
        self.logger.info("Resetting Data Object")
        self.data = DataManagement(self.options, self.wdir)

    def create_grid_representation(self):
        """Grid Representation as property.

        Creates grid representation to be used in the market model.
        """
        if not self.cbco_module:
            self.cbco_module = CBCOModule(self.wdir, self.package_dir, self.grid, self.data, self.options)

        self.cbco_module.create_grid_representation()
        self.grid_representation = self.cbco_module.grid_representation

    def create_geo_plot(self, name="default", bokeh_type="static", results=None):
        """Initialize bokeh plot based on the dataset and a market result.

        Parameters
        ----------
        name : str, optional
            Name defaults to 'default' is is used to identify the initialized
            market result in the plot itself and to name the folder within
            the ``data_temp/bokeh_files`` folder.
        bokeh_type : str, optional
            Specifies if a static or dynamic plot is generated. A dynamic plot
            requires to run a bokeh server, which is generally more involved.
            Defaults to static, which outputs a html version of the map with
            average loads.
        results : dict(str, :obj:`~pomato.data.ResultProcessing`)
            Optionally specify a subset of results to plot.
        """
        self.bokeh_plot = BokehPlot(self.wdir, bokeh_type=bokeh_type)

        if (not self.data.results) and (not results):  # if results dict is empty
            self.logger.info("No result available from market model!")
            self.bokeh_plot.create_empty_static_plot(self.data)
        elif results:
            self.bokeh_plot.create_static_plot(results)
        else:
            self.bokeh_plot.create_static_plot(self.data.results)

    def _join_julia_instances(self):
        self.market_model.julia_model.join()
        self.cbco_module.julia_instance.join()

    def _start_julia_instances(self):
        self.cbco_module._start_julia_daemon()
        self.market_model._start_julia_daemon()
