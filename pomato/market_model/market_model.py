"""The market model of POMATO

This module creates the interface between the data, grid representation and
the market model written in julia. This is done by saving the relevant data as csv,
run the model in a threaded julia program which provides the results in folder as csv files.

The model is initialized empty and then filled with data separately. This makes it
easy to change the data and rerun without re-initializing everything again.

"""
import datetime
import json
import logging
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
        self.logger = logging.getLogger('log.pomato.market_model.MarketModel')
        self.logger.info("Initializing MarketModel...")
        self.options = options

        # Create Folders
        self.wdir = wdir
        self.package_dir = Path(pomato.__path__[0])
        self.data_dir = wdir.joinpath("data_temp/julia_files/data")
        self.results_dir = wdir.joinpath("data_temp/julia_files/results")
        self.julia_model = None
        self.data = data
        self.grid_representation = grid_representation

        # attributes to signal successful model run
        self.status = 'empty'
        self.result_folders = None

    def _start_julia_daemon(self):
        """Start julia subprocess."""
        self.julia_model = tools.JuliaDaemon(self.logger, self.wdir, self.package_dir, "market_model", self.options["solver"]["name"])

    @property
    def model_horizon(self):
        model_horizon_range = range(self.options["model_horizon"][0],
                                    self.options["model_horizon"][1])
        timesteps = self.data.demand_el.timestep.unique()
        model_horizon = [str(x) for x in timesteps[model_horizon_range]]
        return model_horizon
    
    def update_data(self):
        """Initialise or update the underlying data of the market model.

        Updates all data used to run the market model: input data, grid representation, options and
        model horizon by running :meth:`~data_to_csv`.
        """
        self.data_to_csv()

    def run(self):
        """Run the julia program via command Line.

        Uses :class:`~pomato.tools.JuliaDaemon` that is initialized into the *julia_model* attribute
        to run the market model. The model run depends on the supplied options.
        In the case of successful completion, the result folders are stores in the *result_folders*
        attribute which will be instantiated as :class:`~pomato.data.Results` as part of the 
        DataManagement module.

        """
        t_start = datetime.datetime.now()

        if not self.julia_model:
            self.julia_model = tools.JuliaDaemon(self.logger, self.wdir, self.package_dir, "market_model", self.options["solver"]["name"])
        elif not self.julia_model.is_alive:
            self.logger.info("Joining previous market model thread.")
            self.julia_model.join()
            self.julia_model = tools.JuliaDaemon(self.logger, self.wdir, self.package_dir, "market_model", self.options["solver"]["name"])

        self.logger.info("Start-Time: %s", t_start.strftime("%H:%M:%S"))

        args = {"redispatch": self.options["redispatch"]["include"],
                "chance_constrained": self.options["chance_constrained"]["include"]}
        self.julia_model.run(args=args)
        t_end = datetime.datetime.now()
        self.logger.info("End-Time: %s", t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: %s", str((t_end-t_start).total_seconds()) + " sec")

        if self.julia_model.solved:
            # find latest folders created in julia result folder
            # last for normal dispatch, least 2 for redispatch
            if self.options["redispatch"]["include"]:
                if self.options["redispatch"]["zonal_redispatch"]:
                    num_of_results = len(self.options["redispatch"]["zones"]) + 1
                else:
                    num_of_results = 2
                self.result_folders = tools.newest_file_folder(self.results_dir,
                                                               number_of_elm=num_of_results)
            else:
                self.result_folders = [tools.newest_file_folder(self.results_dir,
                                                                number_of_elm=1)]  

            for folder in self.result_folders:
                with open(folder.joinpath("optionfile.json"), 'w') as file:
                    json.dump(self.options, file, indent=2)

            self.status = 'solved'
        else:
            self.logger.warning("Process not terminated successfully!")
            self.status = 'error'
    
    def save_rolling_horizon_storage_levels(self):
        """Set start/end storage levels for rolling horizon market clearing.
        
        This method alters the *storage_level* attribute of the :class:`~pomato.data.DataManagement`
        instance, when market model horizon shorter than total model horizon or values within the
        supplied data does not exactly match the model segments. Default storage levels can be set as part of the options attribute. 
        """
        
        model_horizon = self.model_horizon
        market_model_horizon = self.options["timeseries"]["market_horizon"]
        splits = max(round(len(model_horizon)/market_model_horizon), 1)
        splits_start = [model_horizon[t*market_model_horizon] for t in range(0, splits)]

        def add_to_timestep(t):
            int_t = int(t[1:]) + market_model_horizon
            return 't' + "{0:0>4}".format(int_t)

        splits_end = [add_to_timestep(model_horizon[t*market_model_horizon]) for t in range(0, splits)]

        t_split_map = {}
        timesteps = list(self.data.storage_level.timestep.unique())
        if len(timesteps) > 0:
            for t in set(splits_start + splits_end):
                if t in timesteps:
                    t_split_map[t] = t
                else:
                    int_t = int(t[1:])
                    delta = [abs(int_t - int(t[1:])) for t in timesteps]
                    t_split_map[t] = timesteps[delta.index(min(delta))]
        else:
            for t in set(splits_start + splits_end):
                t_split_map[t] = t
            
        storage_level = self.data.storage_level.copy()
        storage_level = storage_level.set_index(["timestep", "plant"])
        data = []
        for plant in self.data.plants[self.data.plants.plant_type.isin(self.options["plant_types"]["es"])].index:
            for t_start, t_end in zip(splits_start, splits_end):
                if ((t_split_map[t_start], plant) in storage_level.index): # and (self.data.plants.loc[plant, "plant_type"] == "hydro_res"):
                    data.append([t_start, plant, 
                                storage_level.loc[(t_split_map[t_start], plant), "storage_level"],
                                storage_level.loc[(t_split_map[t_end], plant), "storage_level"]])
                else:
                    data.append([t_start, plant, self.options["storages"]["storage_start"], 
                                self.options["storages"]["storage_end"]])
                    
        tmp_storage_level = pd.DataFrame(columns=["timestep", "plant", "storage_start", "storage_end"], data=data)
        # Apply smoothing 
        if self.options["storages"]["smooth_storage_level"]:
            self.logger.info("Smoothing storage levels for rolling horizon")
            timesteps = pd.DataFrame(index=tmp_storage_level["timestep"].unique())
            timesteps["group"] = 0
            timesteps["t_int"] = [int(t[1:]) for t in timesteps.index]
            counter = 1
            for (i, t) in enumerate(timesteps.index[:-1]):
                timesteps.loc[t, "group"] = counter
                if timesteps.loc[timesteps.index[i + 1], "t_int"] - timesteps.loc[t, "t_int"] > market_model_horizon:
                    counter += 1
            timesteps.loc[timesteps.index[-1], "group"] = counter
            
            for plant in tmp_storage_level[tmp_storage_level.storage_start != self.options["storages"]["storage_start"]].plant.unique():   
                for group in timesteps.group.unique():  
                    condition = (tmp_storage_level.timestep.isin(timesteps[timesteps.group == group].index))&(tmp_storage_level.plant == plant)
                    window=int(168/market_model_horizon)
                    tmp_storage_level.loc[condition, "storage_start"] = tmp_storage_level.loc[condition, "storage_start"].rolling(window, min_periods=1).mean()
                    tmp_storage_level.loc[condition, "storage_end"] = tmp_storage_level.loc[condition, "storage_end"].rolling(window, min_periods=1).mean()

        return tmp_storage_level
    
    
        
    def data_to_csv(self):
        """Export input data to csv files in the data_dir sub-directory.

        Writes all data specified in the *model structure* attribute of DataManagement to csv.
        Additionally stores a comprehensive table of plant types, relevant to distinguish between
        certain generation constraints (storages, res etc.), table of slack zones, the grid
        representation and the options.

        Parameters
        ----------
        model_horizon : list
            List of timesteps that are the model horizon
        """
        
        if not self.data_dir.is_dir():
            self.data_dir.mkdir()

        model_horizon = self.model_horizon

        for data in [d for d in self.data.model_structure]:
            cols = [col for col in self.data.model_structure[data].keys() if col != "index"]
            if "timestep" in cols:
                getattr(self.data, data).loc[getattr(self.data, data)["timestep"].isin(model_horizon), cols] \
                    .to_csv(str(self.data_dir.joinpath(f'{data}.csv')), index_label='index')
            else:
                getattr(self.data, data)[cols].to_csv(str(self.data_dir.joinpath(f'{data}.csv')), index_label='index')

        storages = self.data.plants[self.data.plants.plant_type.isin(self.options["plant_types"]["es"])]
        if self.options["storages"]["storage_model"] or storages.empty:
            if storages.empty:
                self.logger.warning("No Storage Plants, disabling storage model.")
                self.options["storages"]["storage_model"] = False
            storage_level = pd.DataFrame(columns=["timestep", "plant", "storage_start", "storage_end"])
            storage_level.to_csv(str(self.data_dir.joinpath('storage_level.csv')), index_label='index')
        else:
            storage_level = self.save_rolling_horizon_storage_levels()
            storage_level.to_csv(str(self.data_dir.joinpath('storage_level.csv')), index_label='index')

        plant_types = pd.DataFrame(index=self.data.plants.plant_type.unique())
        for ptype in self.options["plant_types"]:
            plant_types[ptype] = 0
            condition = plant_types.index.isin(self.options["plant_types"][ptype])
            plant_types[ptype][condition] = 1
        plant_types.to_csv(str(self.data_dir.joinpath('plant_types.csv')), index_label='index')

        if self.grid_representation.grid.empty:
            pd.DataFrame(columns=["cb", "co", "ram"]).to_csv(str(self.data_dir.joinpath('grid.csv')), index_label='index')
        else:
            self.grid_representation.grid \
                .to_csv(str(self.data_dir.joinpath('grid.csv')), index_label='index')

        if self.grid_representation.redispatch_grid.empty:
            pd.DataFrame(columns=["cb", "co", "ram"]).to_csv(str(self.data_dir.joinpath('redispatch_grid.csv')), index_label='index')
        else:
            self.grid_representation.redispatch_grid \
                .to_csv(str(self.data_dir.joinpath('redispatch_grid.csv')), index_label='index')

        if not self.grid_representation.lines.empty:
            self.grid_representation.lines.to_csv(
                str(self.data_dir.joinpath('lines.csv')), index_label='index'
            )
 
        
        with open(self.data_dir.joinpath('contingency_groups.json'), 'w') as file:
            json.dump(self.grid_representation.contingency_groups, file, indent=2)

        if not self.grid_representation.ntc.empty:
            self.grid_representation.ntc.to_csv(str(self.data_dir.joinpath('ntc.csv')), index_label='index')

        slack_zones = pd.DataFrame(index=self.data.nodes.index)
        for slack in self.grid_representation.slack_zones:
            slack_zones[slack] = 0
            condition = slack_zones.index.isin(self.grid_representation.slack_zones[slack])
            slack_zones[slack][condition] = 1

        slack_zones.to_csv(str(self.data_dir.joinpath('slack_zones.csv')), index_label='index')

        with open(self.data_dir.joinpath('options.json'), 'w') as file:
            json.dump(self.options, file, indent=2)
