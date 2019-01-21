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
from pathlib import Path
import shutil
import tools


class JuliaInterface(object):
    """ Class to interface the Julia model with the python Market and Grid Model"""
    def __init__(self, wdir, data, opt_setup, grid_representation, model_horizon):
        # Import Logger
        self.status = 'empty'
        self.logger = logging.getLogger('Log.MarketModel.JuliaInterface')
        self.logger.info("Initializing MarketModel...")

        # Create Folders
        self.wdir = wdir
        self.jl_data_dir = wdir.joinpath("data_temp/julia_files")
        # Make sure all folders exist
        tools.create_folder_structure(self.wdir, self.logger)

        # Init Datamangement, Grid Data and Model Set-up
        self.data = data
        self.model_horizon = [str(x) for x in data.demand_el.index[model_horizon]]
        self.grid_representation = grid_representation

        self.opt_setup = opt_setup
        self.opt_setup["t_start"] = self.model_horizon[0]
        self.opt_setup["t_end"] = self.model_horizon[-1]

        # Extract Model Relevant Data
        self.nodes = data.nodes[["name", "zone", "slack"]]
        self.zones = data.zones
        self.plants = data.plants[['mc', 'tech', 'node', 'eta', 'g_max',
                                   'h_max', 'heatarea']]

        self.heatareas = data.heatareas
        self.demand_el = data.demand_el[data.demand_el.index.isin(self.model_horizon)]
        self.demand_h = data.demand_h[data.demand_h.index.isin(self.model_horizon)]
        self.availability = data.availability
        self.dclines = data.dclines[["node_i", "node_j", "maxflow"]]
        self.ntc = data.ntc

        # fbmc related parameters
        self.net_position = data.net_position
        self.net_export = data.net_export
        self.reference_flows = data.reference_flows

        self.data_to_csv()
        # self.data_to_json()

        self.status = 'ready_to_solve'
        self.results = {}
        self.logger.info("MarketModel Initialized!")

    def run(self):
        """Run the julia Programm via command Line"""
        args = ["julia", str(self.wdir.joinpath("code_jl/main.jl")), str(self.wdir)]

        t_start = datetime.datetime.now()
        self.logger.info("Start-Time: " + t_start.strftime("%H:%M:%S"))
        with open(self.wdir.joinpath("logs").joinpath('julia.log'), 'w') as log:
            # shell=false needed for mac (and for Unix in general I guess)
            with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE) as programm:
                for line in programm.stdout:
                    log.write(line.decode())
                    self.logger.info(line.decode().strip())
#                    self.logger.info(line.decode())

        # if programm.returncode == 1:
            ## have to rerun it to catch the error message :(
            ## there might be a better option
            # self.logger.error("error in Julia Code!\n")
            # programm = subprocess.Popen(args, shell=False, stderr=subprocess.PIPE)
            # _, stderr = programm.communicate()
            # self.logger.error(stderr.decode())
            ## Write Log file
            # with open(self.wdir.joinpath("logs").joinpath('julia.log'), 'a') as log:
            #     log.write(stderr.decode())
            # self.logger.info("julia.log saved!")\

        t_end = datetime.datetime.now()
        self.logger.info("End-Time: " + t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: " + str((t_end-t_start).total_seconds()) + " sec")

        if programm.returncode == 0:
            # find latest folder created in julia result folder
            df = pd.DataFrame()
            df["folder"] = [i for i in self.jl_data_dir.joinpath("results").iterdir()]
            df["time"] = [i.lstat().st_mtime for i in self.jl_data_dir.joinpath("results").iterdir()]
            folder = df.folder[df.time.idxmax()]

            self.data.process_results(folder, self.opt_setup)
            self.data.results.check_infeasibilities()
            self.status = 'solved'
        else:
            self.logger.warning("Process not terminated successfully!")

    def data_to_csv(self):
        """Export Data to csv files file in the jdir + \\data"""
        # Some legacy json also needs to be written here
        csv_path = self.jl_data_dir.joinpath('data')
        self.plants.to_csv(str(csv_path.joinpath('plants.csv')), index_label='index')
        self.nodes.to_csv(str(csv_path.joinpath('nodes.csv')), index_label='index')
        self.zones.to_csv(str(csv_path.joinpath('zones.csv')), index_label='index')
        self.heatareas.to_csv(str(csv_path.joinpath('heatareas.csv')), index_label='index')
        self.demand_el.to_csv(str(csv_path.joinpath('demand_el.csv')), index_label='index')
        self.demand_h.to_csv(str(csv_path.joinpath('demand_h.csv')), index_label='index')
        self.availability.to_csv(str(csv_path.joinpath('availability.csv')), index_label='index')
        self.ntc.to_csv(str(csv_path.joinpath('ntc.csv')), index=False)
        self.dclines.to_csv(str(csv_path.joinpath('dclines.csv')), index_label='index')
        self.net_export.to_csv(str(csv_path.joinpath('net_export.csv')), index_label='index')
        self.net_position.to_csv(str(csv_path.joinpath('net_position.csv')), index_label='index')
        self.reference_flows.to_csv(str(csv_path.joinpath('reference_flows.csv')), index_label='index')

        # Optional data
        self.grid_representation["cbco"].to_csv(str(csv_path.joinpath('cbco.csv')), index_label='index')

        try:
            with open(csv_path.joinpath('slack_zones.json'), 'w') as file:
                json.dump(self.grid_representation["slack_zones"], file)
        except:
            self.logger.warning("slack_zones.json not found - Check if relevant for the model")

        try:
            with open(csv_path.joinpath('opt_setup.json'), 'w') as file:
                json.dump(self.opt_setup, file, indent=2)
        except:
            self.logger.warning("opt_setup.json not found - Check if relevant for the model")

