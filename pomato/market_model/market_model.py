"""
    This is the Julia Interface. It Does:
    Read and save the relevant data into julia/data

    Object Status:
        - empty: initalized but no data loaded
        - ready_to_solve: data loaded and files build in the respective folders
        - solved: model with the current data has been solved succesfully and results-files
                  have been saved in the Julia folder
        - solve_error: something went wrong while trying to run Julia
 """
import logging
import subprocess
import json
import datetime
import pandas as pd
import pomato.tools as tools
from pathlib import Path
# import re

class MarketModel():
    """ Class to interface the Julia model with the python Market and Grid Model"""
    def __init__(self, wdir, options):
        # Import Logger
        self.status = 'empty'
        self.logger = logging.getLogger('Log.MarketModel.JuliaInterface')
        self.logger.info("Initializing MarketModel...")
        self.options = options

        # Create Folders
        self.wdir = wdir
        if self.options["optimization"]["gams"]:
            self.data_dir = wdir.joinpath("data_temp/gms_files")
        else:
            self.data_dir = wdir.joinpath("data_temp/julia_files")
            self.julia_model = None

        # Make sure all folders exist
        tools.create_folder_structure(self.wdir, self.logger)
        self.data = None
        self.grid_representation = None
        self.model_horizon = None

    def update_data(self, data, options, grid_representation):
        # Init Datamangement, Grid Data and Model Set-up
        self.data = data
        self.grid_representation = grid_representation
        self.option = options

        model_horizon_range = range(options["optimization"]["model_horizon"][0],
                                    options["optimization"]["model_horizon"][1])

        timesteps = self.data.demand_el.timestep.unique()
        self.model_horizon = [str(x) for x in timesteps[model_horizon_range]]

        self.options["optimization"]["t_start"] = self.model_horizon[0]
        self.options["optimization"]["t_end"] = self.model_horizon[-1]
        self.data_to_csv()
        self.logger.info("MarketModel Initialized!")

    def run(self):
        """Run the julia Programm via command Line"""
        t_start = datetime.datetime.now()

        solved = False
        if self.options["optimization"]["gams"]:
            result_folder = self.wdir.joinpath("data_temp/gms_files/results/" + t_start.strftime("%d%m_%H%M"))
            if not Path.is_dir(result_folder):
                Path.mkdir(result_folder)

            args = ["gams", str(self.wdir.joinpath("pomato_gms/model.gms")),
                    # "--wdir=" + str(self.wdir),
                    "--rdir=" + str(result_folder),
                    "--model_type=" + self.options["optimization"]["type"],
                    "--infeasibility_electricity=" + str(self.options["optimization"]["infeasibility"]["electricity"]["include"]),
                    "--infeasibility_lines=" + str(self.options["optimization"]["infeasibility"]["lines"]["include"]),
                    "--infeasibility_heat=" + str(self.options["optimization"]["infeasibility"]["heat"]["include"]),
                    "--infeasibility_bound=" + str(self.options["optimization"]["infeasibility"]["electricity"]["bound"]),
                    "lo=3",
                    "curDir=" + str(result_folder)]

            print(" ".join(args))

            self.logger.info("Start-Time: %s", t_start.strftime("%H:%M:%S"))
            with open(self.wdir.joinpath("logs").joinpath('market_model.log'), 'w') as log:
                # shell=false needed for mac (and for Unix in general I guess)
                with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as programm:
                    for line in programm.stdout:
                        log.write(line.decode())
                        self.logger.info(line.decode().strip())

            t_end = datetime.datetime.now()
            self.logger.info("End-Time: %s", t_end.strftime("%H:%M:%S"))
            self.logger.info("Total Time: %s", str((t_end-t_start).total_seconds()) + " sec")
            if programm.returncode == 0:
                solved = True

        else:
            if not self.julia_model:
                self.julia_model = tools.InteractiveJuliaProcess(self.wdir, self.logger, "market_model")

            if self.options["optimization"]["redispatch"]:
                command = 'run_redispatch_pomato("'+ str(self.wdir.as_posix()) + '", "/data/")'
            else:
                command = 'run_pomato("'+ str(self.wdir.as_posix()) + '", "/data/")'

            self.logger.info("Start-Time: %s", t_start.strftime("%H:%M:%S"))
            self.julia_model.run(command)
            t_end = datetime.datetime.now()
            self.logger.info("End-Time: %s", t_end.strftime("%H:%M:%S"))
            self.logger.info("Total Time: %s", str((t_end-t_start).total_seconds()) + " sec")

            if self.julia_model.solved:
                solved = True

        if solved:
            # find latest folder created in julia result folder
            folders = pd.DataFrame()
            folders["folder"] = [i for i in self.data_dir.joinpath("results").iterdir()]
            folders["time"] = [i.lstat().st_mtime \
                          for i in self.data_dir.joinpath("results").iterdir()]
            folder = folders.folder[folders.time.idxmax()]

            with open(folder.joinpath("optionfile.json"), 'w') as file:
                json.dump(self.options, file, indent=2)

            self.data.process_results(folder, self.options["optimization"])
            self.data.results.check_infeasibilities()
            self.status = 'solved'
        else:
            self.logger.warning("Process not terminated successfully!")

    def data_to_csv(self):
        """
        Export Data to csv files file in the jdir/data
        Some json also needs to be written

        """

        model_structure = self.data.model_structure
        csv_path = self.data_dir.joinpath('data')

        for data in self.data.model_structure:
            cols = [col for col in self.data.model_structure[data].attributes if col != "index"]
            getattr(self.data, data)[cols].to_csv(str(csv_path.joinpath(f'{data}.csv')), index_label='index')

        plant_types = pd.DataFrame(index=self.data.plants.plant_type.unique())
        for ptype in self.options["optimization"]["plant_types"]:
            plant_types[ptype] = 0
            condition = plant_types.index.isin(self.options["optimization"]["plant_types"][ptype])
            plant_types[ptype][condition] = 1
        plant_types.to_csv(str(csv_path.joinpath('plant_types.csv')), index_label='index')

        # Optional data
        self.grid_representation["cbco"] \
            .to_csv(str(csv_path.joinpath('cbco.csv')), index_label='index')

        slack_zones = pd.DataFrame(index=self.data.nodes.index)
        for slack in self.grid_representation["slack_zones"]:
            slack_zones[slack] = 0
            condition = slack_zones.index.isin(self.grid_representation["slack_zones"][slack])
            slack_zones[slack][condition] = 1

        slack_zones.to_csv(str(csv_path.joinpath('slack_zones.csv')), index_label='index')

        with open(csv_path.joinpath('options.json'), 'w') as file:
            json.dump(self.options["optimization"], file, indent=2)

        try:
            with open(csv_path.joinpath('slack_zones.json'), 'w') as file:
                json.dump(self.grid_representation["slack_zones"], file)
        except:
            self.logger.warning("slack_zones.json not found - Check if relevant for the model")

