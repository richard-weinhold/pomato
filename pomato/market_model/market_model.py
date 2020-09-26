"""The market model of POMATO

This module creates the interface between the data, grid representation and
the market model written in julia. This is done by saving the relevant data as csv,
run the model in a subprocess which provides the results in folder as csv files.

The Modes is initionaled empty and then filled with data seperately. This makes it
easy to change the data and rerun without re-initializing everything again.

"""
import datetime
import json
import logging
import subprocess
from pathlib import Path

import pandas as pd

import pomato
import pomato.tools as tools


class MarketModel():
    """Class to interface the MarketModel in Julia with the python based Data and Grid models.

    This module is initialized empty and only initializes the interactive julia process used to run
    the market model. Once started the model can be easily re-run with changed options or data.
    The option file serves as an argument to distinguish possible market model implementations which
    would be initialized differently.

    Parameters
    ----------
    wdir : pathlib.Path
        Working directory
    options : dict
        The options from POMATO main method persist in the MarketModel.

    Attributes
    ----------
    wdir : pathlib.Path
        Working directory
    options : dict
        The options from POMATO main method persist in the MarketModel.
    data_dir : pathlib.Path
        Subdirectory of working directory to store the data in.
    julia_model : :class:`~pomato.tools.JuliaDaemon`
        Interactive julia process which is used to run the market model.
    data : :class:`~pomato.data.DataManagement`
       An instance of the DataManagement class with processed input data.
    grid_representation : types.SimpleNamespace
        Grid representation resulting from of :class:`~pomato.cbco.GridRepresentation`. Contains a
        suitable grid representation based on the chosen options.
    status : str
        Attribute indicating the model status: Empty, Solved, Error.
    result_folders : list(pathlib.Path)
        List of sub-directories containing the results in case of successful solve.
        Can be multiples for redispatch calculations or FBMC application.
    """

    def __init__(self, wdir, options, data, grid_representation):
        self.logger = logging.getLogger('Log.MarketModel.JuliaInterface')
        self.logger.info("Initializing MarketModel...")
        self.options = options

        # Create Folders
        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])
        self.data_dir = wdir.joinpath("data_temp/julia_files/data")
        self.results_dir = wdir.joinpath("data_temp/julia_files/results")
        self.julia_model = None

        # Make sure all folders exist
        tools.create_folder_structure(self.wdir, self.logger)
        self.data = data
        self.grid_representation = grid_representation

        # attributes to signal successful model run
        self.status = 'empty'
        self.result_folders = None

    def _start_julia_daemon(self):
        """Start julia subprocess."""
        self.julia_model = tools.JuliaDaemon(self.logger, self.wdir, self.package_dir, "market_model")

    def update_data(self):
        """Initialise or update the underlying data of the market model.

        Updates all data used to run the market model: input data, grid representation, options and
        model horizon by running :meth:`~data_to_csv`.
        """

        model_horizon_range = range(self.options["optimization"]["model_horizon"][0],
                                    self.options["optimization"]["model_horizon"][1])

        timesteps = self.data.demand_el.timestep.unique()
        model_horizon = [str(x) for x in timesteps[model_horizon_range]]
        self.data_to_csv(model_horizon)
        self.logger.info("MarketModel Initialized!")

    def run(self):
        """Run the julia program via command Line.

        Uses :class:`~pomato.tools.InteractiveJuliaProcess` from the attribute *julia_model* to
        run the market model. If the model is not initialized, it will be done here once. The
        model run depends on the supplied options.
        In the case of successful completion, the result folders are stores in the *result_folders*
        attribute for further processing the in :class:`~pomato.data.Results` module.

        """
        t_start = datetime.datetime.now()

        if not self.julia_model:
            self.julia_model = tools.JuliaDaemon(self.logger, self.wdir, self.package_dir, "market_model")

        self.logger.info("Start-Time: %s", t_start.strftime("%H:%M:%S"))

        args = {"redispatch": self.options["optimization"]["redispatch"]["include"],
                "chance_constrained": self.options["optimization"]["chance_constrained"]["include"]}
                
        self.julia_model.run(args=args)
        
        # self.julia_model.run(command)
        t_end = datetime.datetime.now()
        self.logger.info("End-Time: %s", t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: %s", str((t_end-t_start).total_seconds()) + " sec")

        if self.julia_model.solved:
            # find latest folders created in julia result folder
            # last for normal dispatch, least 2 for redispatch
            if self.options["optimization"]["redispatch"]["include"]:
                if self.options["optimization"]["redispatch"]["zonal_redispatch"]:
                    num_of_results = len(self.options["optimization"]["redispatch"]["zones"]) + 1
                else:
                    num_of_results = 2
                self.result_folders = tools.newest_file_folder(self.results_dir,
                                                               number_of_elm=num_of_results)
            else:
                self.result_folders = [tools.newest_file_folder(self.results_dir,
                                                                number_of_elm=1)]  

            for folder in self.result_folders:
                with open(folder.joinpath("optionfile.json"), 'w') as file:
                    json.dump(self.options["optimization"], file, indent=2)

            self.status = 'solved'
        else:
            self.logger.warning("Process not terminated successfully!")
            self.status = 'error'
    
    def rolling_horizon_storage_levels(self, model_horizon):
        """Set start/end storage levels for rolling horizon market clearing.

        When market model horizon shorter than total model horizon, specify storage levels to 
        adequately represent storages. Per default storage levels are 65% of total capacity, however
        this can be edited in the storage_level attribute of :class:`~pomato.data.DataManagement`.
        
        """

        splits = int(len(model_horizon)/self.options["optimization"]["timeseries"]["market_horizon"])
        market_model_horizon = self.options["optimization"]["timeseries"]["market_horizon"]

        splits = max(1, int(len(model_horizon)/self.options["optimization"]["timeseries"]["market_horizon"]))
        splits = [model_horizon[t*market_model_horizon] for t in range(0, splits)]
        if not all(t in self.data.storage_level.timestep for t in splits):
            data = []
            for plant in self.data.plants[self.data.plants.plant_type.isin(self.options["optimization"]["plant_types"]["es"])].index:
                for t in splits:
                    data.append([t, plant, self.options["optimization"]["parameters"]["storage_start"]])
            self.data.storage_level = pd.DataFrame(columns=self.data.storage_level.columns, data=data)

       
    def data_to_csv(self, model_horizon):
        """Export input data to csv files in the data_dir sub-directory.

        Writes all data specified in the *model structure* attribute of DataManagement to csv.
        Additionally stores a comprehensive table of plant types, relevant to distinguish between
        certain generation constraints (storages, res etc.), table of slack zones, the grid
        representation and the options.

        """
        if not self.data_dir.is_dir():
            self.data_dir.mkdir()
        
        self.rolling_horizon_storage_levels(model_horizon)
        
        for data in [d for d in self.data.model_structure if d != "lines"]:
            cols = [col for col in self.data.model_structure[data].keys() if col != "index"]
            if "timestep" in cols:
                getattr(self.data, data).loc[getattr(self.data, data)["timestep"].isin(model_horizon), cols] \
                    .to_csv(str(self.data_dir.joinpath(f'{data}.csv')), index_label='index')
            else:
                getattr(self.data, data)[cols].to_csv(str(self.data_dir.joinpath(f'{data}.csv')), index_label='index')

        plant_types = pd.DataFrame(index=self.data.plants.plant_type.unique())
        for ptype in self.options["optimization"]["plant_types"]:
            plant_types[ptype] = 0
            condition = plant_types.index.isin(self.options["optimization"]["plant_types"][ptype])
            plant_types[ptype][condition] = 1
        plant_types.to_csv(str(self.data_dir.joinpath('plant_types.csv')), index_label='index')

        if self.grid_representation.grid.empty:
            pd.DataFrame(columns=["ram"]).to_csv(str(self.data_dir.joinpath('grid.csv')), index_label='index')
        else:
            self.grid_representation.grid \
                .to_csv(str(self.data_dir.joinpath('grid.csv')), index_label='index')

        if self.grid_representation.redispatch_grid.empty:
            pd.DataFrame(columns=["ram"]).to_csv(str(self.data_dir.joinpath('redispatch_grid.csv')), index_label='index')
        else:
            self.grid_representation.redispatch_grid \
                .to_csv(str(self.data_dir.joinpath('redispatch_grid.csv')), index_label='index')

        if not self.grid_representation.ntc.empty:
            self.grid_representation.ntc.to_csv(str(self.data_dir.joinpath('ntc.csv')), index_label='index')

        slack_zones = pd.DataFrame(index=self.data.nodes.index)
        for slack in self.grid_representation.slack_zones:
            slack_zones[slack] = 0
            condition = slack_zones.index.isin(self.grid_representation.slack_zones[slack])
            slack_zones[slack][condition] = 1

        slack_zones.to_csv(str(self.data_dir.joinpath('slack_zones.csv')), index_label='index')

        with open(self.data_dir.joinpath('options.json'), 'w') as file:
            json.dump(self.options["optimization"], file, indent=2)
