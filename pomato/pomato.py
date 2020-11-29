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

The recommended way to install POMATO:
  - Install Julia and have it available on the system path
  - Install POMATO through *pip* in python >= 3.6 by creating a virtual environment and install pomato into it::

        python -m venv pomato
        ./pomato/Scripts/activate
        pip install git+https://github.com/richard-weinhold/pomato.git


After this is completed pomato can be imported in python::

    from pomato import POMATO

Pomato functions from a *working directory*, ideally the project folder includes the virtual 
environment, and creates additional folders for results, temporary data and logs. The way we use 
pomato is illustrated by the *examples* folder, cloning its contents as a *working directory* 
allows to run the included examples.

Pomato works with open solvers, if Gurobi is available on the PATH within the venv/project it will 
be used. See `Gurobi.jl <https://github.com/JuliaOpt/Gurobi.jl>`_ for notes on the installation. 
Additionally, the Chance-Constrained model formulation requires MOSEK solver which can be installed
from within Pomato, but requires a licence to use `Mosek.jl <https://github.com/JuliaOpt/Mosek.jl>`_. 


Examples
--------
This release includes two examples in the *examples* folder. Including the contents of 
this folder into the pomato working directory will allow their execution:

  - The IEEE 118 bus network, which contains a singular timestep. The data is available under 
    open license at `https://power-grid-lib.github.io/ <https://power-grid-lib.github.io/>`_
    and re-hosted in this repository::

        $ python /run_pomato_ieee.py

  - The DE case study, based on data from `ELMOD-DE <http://www.diw.de/elmod>`_ which is 
    openly available and described in detail in 
    `DIW DataDocumentation 83 <https://www.diw.de/documents/publikationen/73/diw_01.c.528927.de/diw_datadoc_2016-083.pdf>`_ 
    which represents a more complex system and can be run for longer model horizon (although 
    shortened to allow to host this data in this git)::

        $ python /run_pomato_de.py

However, the functionality of POMATO is best utilized when running inside a
IDE (e.g. Spyder) to access POMATO objects and develop a personal script based
on the provided functionality and its results.

However, the functionality of POMATO is best utilized when running inside a
IDE (e.g. Spyder) to access POMATO objects and develop a personal script based
on the provided functionality and its results.

Release Status
--------------

POMATO is part of my PhD and actively developed by Robert and myself. This means it will keep 
changing to include new functionality or to improve existing features. The existing examples, which
are also part of the Getting Started guide in the documentation, are part of a testing suite to 
ensure some robustness. However, we are not software engineers, thus the "program" is not written 
with robustness in mind and our experience is limited when it comes to common best practices. 
Expect errors, bug, funky behavior and code structures from the minds of two engineering economists.  

Related Publications
--------------------

- `Weinhold and Mieth (2020), Fast Security-Constrained Optimal Power Flow through 
  Low-Impact and Redundancy Screening <https://ieeexplore.ieee.org/document/9094021>`_

- `Schönheit, Weinhold, Dierstein (2020), The impact of different strategies for 
  generation shift keys (GSKs) on the flow-based market coupling domain: A model-based analysis 
  of Central Western Europe <https://www.sciencedirect.com/science/article/pii/S0306261919317544>`_

Acknowledgments
---------------

Richard and Robert would like to acknowledge the support of Reiner Lemoine-Foundation, the Danish 
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
from pomato.data import DataManagement, Results
from pomato.grid import GridTopology, GridModel
from pomato.market_model import MarketModel
from pomato.visualization.geoplot import GeoPlot
from pomato.fbmc import FBMCModule

def _logging_setup(wdir, logging_level=logging.INFO):
    # Logging setup
    logger = logging.getLogger('Log.MarketModel')
    logger.setLevel(logging_level)
    if len(logger.handlers) < 2:
        # create file handler which logs even debug messages
        if not wdir.joinpath("logs").is_dir():
            wdir.joinpath("logs").mkdir()

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
    The main module joins all components of POMATO, providing accessibility to the user.

    The core attributes are modules that provide specific functionalities to
    the model. At the center is an instance of the DataManagement module, that
    reads and processes input data, makes processed input data accessible to
    other modules and is the container for the results from the market model.

    Attributes
    ----------
    wdir : pathlib.Path
        Working directory, all necessary folders, temporary files and results
        are stored in relation to this directory.
    options : dict
        Dictionary containing centralized all options relevant for running an economic
        dispatch. The options are roughly categorized into:

            - Optimization: defining model type, model horizon, inclusion of
              model features like heat or curtailment, management of infeasibility
              variables and definition of plant types considered.
            - Grid: Setting Capacity multiplier, applying redundancy removal
              for contingency analysis and its sensitivity, for zonal
              representation setting the node-zone weights.
            - Data: data type, data that is stored stacked (opposed to long),
              options to process marginal costs to be unique and default values.
        
        Gets initialized with the input json file and receives default options based on the method
        :meth:`~pomato.tools.add_default_options`.

    data : :class:`~pomato.data.DataManagement`
        Instance of DataManagement class containing all data, data processing,
        results and result processing. Is initialized empty, then data
        explicitly loaded.
    grid_topology : :class:`~pomato.grid.GridTopology`
        Object containing all grid information. Initializes empty and
        filled based on nodes and lines data when it is loaded and
        processed. Provides the PTDF matrices for N-0 and N-1 load flow analysis.
    grid_model : :class:`~pomato.grid.GridModel`
        The GridModel provides the grid representation to the market model,
        based on the chosen configuration. Combines input data and the GridModel
        into a grid representation based on the chosen configuration (N-0, N-1,
        zonal, zonal CBCO, ntc, copper
        plate) and runs the redundancy removal algorithm needed for N-1
        grid representation.
    grid_representation : :class:`~pomato.grid.GridModel`.grid_representation
        The output of the grid_module. A namespace containing all information
        for the market model to account for the chosen network representation.
    market_model : :class:`~pomato.market_model.MarketModel`
        Module containing and managing the market model. This includes storing
        the necessary data, running and managing a julia process instance and
        initializing the result object inside ``data``.
    geo_plot : :class:`~pomato.visualization.GeoPlot`
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
    """
   
    def __init__(self, wdir, options_file=None, logging_level=logging.INFO):

        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])

        self.logger = _logging_setup(self.wdir, logging_level)
        self.logger.info("Market Tool Initialized")
        tools.create_folder_structure(self.wdir, self.logger)

        # Core Attributes
        if not options_file:
            self.options = tools.default_options()        
        else: 
            self.initialize_options(options_file)
        
        self.data = DataManagement(self.options, self.wdir)
        self.grid = GridTopology()       
        self.grid_model = GridModel(self.wdir, self.grid, self.data, self.options)
        self.grid_representation = self.grid_model.grid_representation
        self.market_model = MarketModel(self.wdir, self.options, self.data, self.grid_representation)

        # Instances for Result Processing 
        self.geo_plot = GeoPlot(self.wdir, self.data)
        self.fbmc = FBMCModule(self.wdir, self.grid, self.data, self.options)


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
                loaded_options = json.load(opt_file)
            self.options = tools.add_default_options(loaded_options)  
            self.logger.info( "Optimization Options:" + json.dumps(self.options, indent=2) + "\n")

        except FileNotFoundError:
            self.logger.warning("No or invalid options file provided, using default options")
            self.options = tools.default_options()
            self.logger.info("Optimization Options:" + json.dumps(self.options, indent=2) + "\n")
        except BaseException as unknown_exception:
            self.logger.exception("Error: %s", unknown_exception)

    def load_data(self, filename):
        """Load data into :class:`~pomato.data.DataManagement` module.

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
        self.init_grid_model()

    def init_grid_model(self):
        """Initialize the grid model from the data management object"""
        self.grid.calculate_parameters(self.data.nodes, self.data.lines)

    def update_market_model_data(self, folder=None):
        """Update data within an instance of the market model.
        
        It is possible to change the data which is used in the market model and re-run without
        explicitly re-initializing the module.

        Parameters
        ----------
        folder : pathlib.Path, optional
            Saving model data to specified path, defaults to ``data_temp/julia_files/data``.
        """
        
        if folder:
            self.market_model.data_dir = folder
            self.market_model.update_data()
            self.market_model.data_dir = self.wdir.joinpath("data_temp/julia_files/data")
        else:
            self.market_model.update_data()

    def initialize_market_results(self, result_folders):
        """Initializes market results from a list of folders.

        Parameters
        ----------
        result_folders : list
            List of folders containing market results.
        """
        for folder in result_folders:
            self.data.results[folder.name] = Results(self.data, self.grid, folder)
    
    def rename_market_result(self, oldname, newname):
        """Rename market results. 

        The market results are initialized from folders in *data_temp/julia_files/results* in a 
        autogenerated format which consists of date and time. This can be replaced with a more
        appropriate naming. If multiple results exist, all are renamed accordingly. 

        E.g. with two results 01010_market_results, 01010_redispatch_Z1, it makes sense to replace 
        the numeric prefix. 

        Parameters
        ----------
        oldname : String
            The original name
        newname : String
            The new name.
        """ 
        
        for result in self.data.results:
            self.data.results[result.replace(oldname, newname)] = self.data.results.pop(result)


    def run_market_model(self, update_data=True, update_grid_representation=False):
        """Run the market model based on the current state of data and options. 
        
        Parameters
        ----------
        update_data : bool, optional
            Update data before model run. Default: True.
        update_grid_representation : bool, optional
            Update the grid representation before model run. Default: False.
        """

        if update_data:
            self.update_market_model_data()
        
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
        self.grid_model.create_grid_representation()

    def create_geo_plot(self, show=True, **kwargs):
        """Initialize GeoPlot based on the dataset and a market result.

        This is done with the :meth:`~pomato.visualization.geoplot.create_static_plot` method. 
        See the correpsonding documentation for the available conditional arguments.
        
        Parameters
        ----------
        show : bool, optional
            Show the plot after completion, this will open a browser window.
        """
        if (not self.data.results):  # if results dict is empty
            self.logger.info("No result available from market model!")
            self.geo_plot.create_empty_static_plot()
        else:
            self.geo_plot.create_static_plot(**kwargs)
        if show:
            self.geo_plot.show_plot()

    def _instantiate_julia_dev(self, redundancyremoval_path, marketmodel_path):
        """Instantiate the pomato julia environment from local repositories, 

        Parameters
        ----------
        redundancyremoval_path : pathlib.Path   
            Path to the local RedundancyRemoval repository.
        marketmodel_path : pathlib.Path
            Path to the local MarketModel repository.
        """        
        tools.julia_management.instantiate_julia_dev(self.package_dir, 
                                                     str(redundancyremoval_path), 
                                                     str(marketmodel_path))
                                                     
    def _instantiate_julia(self):
        tools.julia_management.instantiate_julia(self.package_dir)

    def _join_julia_instance_market_model(self):
        if self.market_model.julia_model:
            self.market_model.julia_model.join()
            self.market_model.julia_model = None

    def _join_julia_instance_grid_model(self):
        if self.grid_model.julia_instance:
            self.grid_model.julia_instance.join()
            self.grid_model.julia_instance = None

    def _join_julia_instances(self):
        self._join_julia_instance_market_model()
        self._join_julia_instance_grid_model()

    def _start_julia_instances(self):
        self.grid_model._start_julia_daemon()
        self.market_model._start_julia_daemon()

    def __del__(self):
        """Join Julia instances on deletion."""
        self._join_julia_instances()
