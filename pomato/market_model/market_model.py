"""The market model of POMATO

This module creates the interface between the data, grid representaiton and
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

import pomato.tools as tools

# import re


class MarketModel():
    """Class to interface the MarketModel in Julia with the python based Data and Grid models.

    This module is initialized empty and only initializes the interactive julia process used to run
    the market model. Once started the model can be easily re-run with changed options or data.
    The option file serves as an argumant to distinguish possible market model implementations which
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
    julia_model : :class:`~pomato.tools.InteractiveJuliaProcess`
        Interactive julia process which is used to run the market model.
    data : :class:`~pomato.data.DataManagement`
       An instance of the DataManagement class with processed input data.
    grid_representation : dict
        Grid representation resulting from of :class:`~pomato.cbco.CBCOModule`. Contains a
        suitable grid represenation based on the chosen options.
    model_horizon : list
        List containing all timesteps part of the model horizon.
    status : str
        Attribute indicating the model status: Empty, Solved, Error.
    result_folders : list(pathlib.Path)
        List of sub-directories containing the results in case of successful solve.
        Can be multiples for redispatch calculations or FBMC application.
    """

    def __init__(self, wdir, package_dir, options):
        self.logger = logging.getLogger('Log.MarketModel.JuliaInterface')
        self.logger.info("Initializing MarketModel...")
        self.options = options

        # Create Folders
        self.wdir = wdir
        self.package_dir = package_dir
        self.data_dir = wdir.joinpath("data_temp/julia_files")
        self.julia_model = tools.JuliaDeamon(self.logger, self.wdir, self.package_dir, "market_model")

        # Make sure all folders exist
        tools.create_folder_structure(self.wdir, self.logger)
        self.data = None
        self.grid_representation = None
        self.model_horizon = None

        # attributes to signal sucessfull model run
        self.status = 'empty'
        self.result_folders = None

    def update_data(self, data, options, grid_representation):
        """Initialise or update the underlying data of the market model.

        Updates all data used to run the market model: input data, grid representation, options and
        model horizon by running :meth:`~data_to_csv`.

        Parameters
        ----------
        data : :class:`~pomato.data.DataManagement`
           Instance of the DataManagement class with processed input data.
        options : dict
            While already part of the init, re-runs with changed options are often utilized.
        grid_representation : dict
            Grid representation resulting from of :class:`~pomato.cbco.CBCOModule`. Contains a
            suitable grid represenation based on the chosen options.
        """
        self.data = data
        self.grid_representation = grid_representation
        self.options = options

        model_horizon_range = range(options["optimization"]["model_horizon"][0],
                                    options["optimization"]["model_horizon"][1])

        timesteps = self.data.demand_el.timestep.unique()
        self.model_horizon = [str(x) for x in timesteps[model_horizon_range]]

        self.options["optimization"]["t_start"] = self.model_horizon[0]
        self.options["optimization"]["t_end"] = self.model_horizon[-1]
        self.data_to_csv()
        self.logger.info("MarketModel Initialized!")

    def run(self):
        """Run the julia Programm via command Line.

        Uses :class:`~pomato.tools.InteractiveJuliaProcess` from the attribite *julia_model* to
        run the market model. If the model is not initialized, it will be done here onece. The
        model run depends on the supplied options.
        In the case of successful completion, the resul folders are stores in the *result_folders*
        attribute for further processing thje in :class:`~pomato.data.ResultProcessing` module.

        """
        t_start = datetime.datetime.now()

        solved = False

        if not self.julia_model:
            self.julia_model = tools.JuliaDeamon(self.logger, self.wdir, self.package_dir, "market_model")

        self.logger.info("Start-Time: %s", t_start.strftime("%H:%M:%S"))

        args = {"redispatch": self.options["optimization"]["redispatch"]["include"]}
        self.julia_model.run(args=args)
        
        # self.julia_model.run(command)
        t_end = datetime.datetime.now()
        self.logger.info("End-Time: %s", t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: %s", str((t_end-t_start).total_seconds()) + " sec")

        if self.julia_model.solved:
            solved = True

        if solved:
            # find latest folders created in julia result folder
            # last for normal dispatch, last 2 for redispatch
            if self.options["optimization"]["redispatch"]["include"]:
                self.result_folders = tools.newest_file_folder(self.data_dir.joinpath("results"),
                                                               number_of_elm=2)
            else:
                self.result_folders = [tools.newest_file_folder(self.data_dir.joinpath("results"),
                                                                number_of_elm=1)]

            for folder in self.result_folders:
                with open(folder.joinpath("optionfile.json"), 'w') as file:
                    json.dump(self.options["optimization"], file, indent=2)

            self.status = 'solved'
        else:
            self.logger.warning("Process not terminated successfully!")
            self.status = 'error'

    def data_to_csv(self):
        """Export input data to csv files in the ddir sub-directory.

        Writes all data specified in the *model stucture* attribute of DataMangement to csv.
        Additionally stores a comprehensive table of plant types, relevant to distinguish between
        certain generation constraints (storages, res etc.), table of slack zones, the grid
        represenation and the options.

        """
        csv_path = self.data_dir.joinpath('data')
        for data in self.data.model_structure:
            cols = [col for col in self.data.model_structure[data].attributes if col != "index"]
            if "timestep" in cols:
                getattr(self.data, data).loc[getattr(self.data, data)["timestep"].isin(self.model_horizon), cols] \
                    .to_csv(str(csv_path.joinpath(f'{data}.csv')), index_label='index')
            else:
                getattr(self.data, data)[cols].to_csv(str(csv_path.joinpath(f'{data}.csv')), index_label='index')

        plant_types = pd.DataFrame(index=self.data.plants.plant_type.unique())
        for ptype in self.options["optimization"]["plant_types"]:
            plant_types[ptype] = 0
            condition = plant_types.index.isin(self.options["optimization"]["plant_types"][ptype])
            plant_types[ptype][condition] = 1
        plant_types.to_csv(str(csv_path.joinpath('plant_types.csv')), index_label='index')

        # Optional data
        if self.grid_representation["grid"].empty:
            pd.DataFrame(columns=["ram"]).to_csv(str(csv_path.joinpath('grid.csv')), index_label='index')
        else:
            self.grid_representation["grid"] \
                .to_csv(str(csv_path.joinpath('grid.csv')), index_label='index')

        if self.grid_representation["redispatch_grid"].empty:
            pd.DataFrame(columns=["ram"]).to_csv(str(csv_path.joinpath('redispatch_grid.csv')), index_label='index')
        else:
            self.grid_representation["redispatch_grid"] \
                .to_csv(str(csv_path.joinpath('redispatch_grid.csv')), index_label='index')

        slack_zones = pd.DataFrame(index=self.data.nodes.index)
        for slack in self.grid_representation["slack_zones"]:
            slack_zones[slack] = 0
            condition = slack_zones.index.isin(self.grid_representation["slack_zones"][slack])
            slack_zones[slack][condition] = 1

        slack_zones.to_csv(str(csv_path.joinpath('slack_zones.csv')), index_label='index')

        with open(csv_path.joinpath('options.json'), 'w') as file:
            json.dump(self.options["optimization"], file, indent=2)
