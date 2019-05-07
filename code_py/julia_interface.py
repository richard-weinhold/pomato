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
    def __init__(self, wdir, data, options, grid_representation):
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
        self.grid_representation = grid_representation

        self.options = options
        model_horizon_range = range(options["model_horizon"][0], options["model_horizon"][1])
        self.model_horizon = [str(x) for x in data.demand_el.index[model_horizon_range]]

        self.options["t_start"] = self.model_horizon[0]
        self.options["t_end"] = self.model_horizon[-1]

        self.data_to_csv()
        # self.data_to_json()

        self.status = 'ready_to_solve'
        self.results = {}
        self.logger.info("MarketModel Initialized!")

    def run(self):
        """Run the julia Programm via command Line"""
        args = ["julia", "--project=project_files/pomato", 
                str(self.wdir.joinpath("code_jl/main.jl")), str(self.wdir), "/data/"]
        t_start = datetime.datetime.now()
        self.logger.info("Start-Time: " + t_start.strftime("%H:%M:%S"))
        with open(self.wdir.joinpath("logs").joinpath('julia.log'), 'w') as log:
            # shell=false needed for mac (and for Unix in general I guess)
            with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE) as programm:
                for line in programm.stdout:
                    log.write(line.decode())
                    self.logger.info(line.decode().strip())

        t_end = datetime.datetime.now()
        self.logger.info("End-Time: " + t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: " + str((t_end-t_start).total_seconds()) + " sec")

        if programm.returncode == 0:
            # find latest folder created in julia result folder
            df = pd.DataFrame()
            df["folder"] = [i for i in self.jl_data_dir.joinpath("results").iterdir()]
            df["time"] = [i.lstat().st_mtime for i in self.jl_data_dir.joinpath("results").iterdir()]
            folder = df.folder[df.time.idxmax()]

            self.data.process_results(folder, self.options)
            self.data.results.check_infeasibilities()
            self.status = 'solved'
        else:
            self.logger.warning("Process not terminated successfully!")

    def data_to_csv(self):
        """Export Data to csv files file in the jdir + \\data"""
        # Some legacy json also needs to be written here
        csv_path = self.jl_data_dir.joinpath('data')
        self.data.plants[['mc', 'tech', 'node', 'eta', 'g_max','h_max', 'heatarea']].to_csv(str(csv_path.joinpath('plants.csv')), index_label='index')
        self.data.nodes[["name", "zone", "slack"]].to_csv(str(csv_path.joinpath('nodes.csv')), index_label='index')
        self.data.zones.to_csv(str(csv_path.joinpath('zones.csv')), index_label='index')
        self.data.heatareas.to_csv(str(csv_path.joinpath('heatareas.csv')), index_label='index')
        self.data.demand_el[self.data.demand_el.index.isin(self.model_horizon)].to_csv(str(csv_path.joinpath('demand_el.csv')), index_label='index')
        self.data.demand_h[self.data.demand_h.index.isin(self.model_horizon)].to_csv(str(csv_path.joinpath('demand_h.csv')), index_label='index')
        self.data.availability.to_csv(str(csv_path.joinpath('availability.csv')), index_label='index')
        self.data.dclines[["node_i", "node_j", "maxflow"]].to_csv(str(csv_path.joinpath('dclines.csv')), index_label='index')
        self.data.ntc.to_csv(str(csv_path.joinpath('ntc.csv')), index=False)
        self.data.net_export.to_csv(str(csv_path.joinpath('net_export.csv')), index_label='index')
        self.data.net_position.to_csv(str(csv_path.joinpath('net_position.csv')), index_label='index')
        self.data.reference_flows.to_csv(str(csv_path.joinpath('reference_flows.csv')), index_label='index')

        plant_types = pd.DataFrame(index=self.data.plants.tech.unique())
        for ptype in self.options["plant_types"]:
            plant_types[ptype] = 0
            plant_types[ptype][plant_types.index.isin(self.options["plant_types"][ptype])] = 1
        plant_types.to_csv(str(csv_path.joinpath('plant_types.csv')), index_label='index')

        # Optional data
        self.grid_representation["cbco"].to_csv(str(csv_path.joinpath('cbco.csv')), index_label='index')
        
        try:
            slack_zones = pd.DataFrame(index=self.data.nodes.index)
            for slack in self.grid_representation["slack_zones"]:
                slack_zones[slack] = 0
                slack_zones[slack][slack_zones.index.isin(self.grid_representation["slack_zones"][slack])] = 1
            slack_zones.to_csv(str(csv_path.joinpath('slack_zones.csv')), index_label='index')
        except:
            self.logger.warning("slack_zones.json not found - Check if relevant for the model")

        try:
            with open(csv_path.joinpath('slack_zones.json'), 'w') as file:
                json.dump(self.grid_representation["slack_zones"], file)
        except:
            self.logger.warning("slack_zones.json not found - Check if relevant for the model")
            
        try:
            with open(csv_path.joinpath('options.json'), 'w') as file:
                json.dump(self.options, file, indent=2)
        except:
            self.logger.warning("options.json not found - Check if relevant for the model")
