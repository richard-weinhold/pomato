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
import tools
from pathlib import Path

class MarketModel():
    """ Class to interface the Julia model with the python Market and Grid Model"""
    def __init__(self, wdir, data, options, grid_representation):
        # Import Logger
        self.status = 'empty'
        self.logger = logging.getLogger('Log.MarketModel.JuliaInterface')
        self.logger.info("Initializing MarketModel...")
        self._gams = True
        # Create Folders
        self.wdir = wdir
        if self._gams:
            self.data_dir = wdir.joinpath("data_temp/gms_files")
        else:
            self.data_dir = wdir.joinpath("data_temp/julia_files")
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
        t_start = datetime.datetime.now()
        if self._gams:
            result_folder = self.wdir.joinpath("data_temp/gms_files/results/" + t_start.strftime("%d%m_%H%M"))
            if not Path.is_dir(result_folder):
                Path.mkdir(result_folder)

            args = ["gams", str(self.wdir.joinpath("code_gms/model.gms")),
                    "--wdir=" + str(self.wdir),
                    "--rdir=" + str(result_folder),
                    "--model_type=" + self.options["type"],
                    "--infeas_el_nodal=" + str(self.options["infeas_el_nodal"]),
                    "--infeas_lines=" + str(self.options["infeas_lines"]),
                    "--infeas_heat=" + str(self.options["infeas_heat"]),
                    "--infeasibility_bound=" + str(self.options["infeasibility_bound"]),
                    "lo=3",
                    "curDir=" + str(result_folder)]
        else:
            args = ["julia", "--project=project_files/pomato",
                    str(self.wdir.joinpath("code_jl/main.jl")), str(self.wdir), "/data/"]

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
            # find latest folder created in julia result folder
            folders = pd.DataFrame()
            folders["folder"] = [i for i in self.data_dir.joinpath("results").iterdir()]
            folders["time"] = [i.lstat().st_mtime \
                          for i in self.data_dir.joinpath("results").iterdir()]
            folder = folders.folder[folders.time.idxmax()]

            self.data.process_results(folder, self.options)
            self.data.results.check_infeasibilities()
            self.status = 'solved'
        else:
            self.logger.warning("Process not terminated successfully!")

    def data_to_csv(self):
        """Export Data to csv files file in the jdir + \\data"""
        # Some legacy json also needs to be written here
        csv_path = self.data_dir.joinpath('data')
        self.data.plants[['mc', 'tech', 'node', 'eta', 'g_max', 'h_max', 'heatarea']] \
            .to_csv(str(csv_path.joinpath('plants.csv')), index_label='index')

        self.data.nodes[["name", "zone", "slack"]] \
            .to_csv(str(csv_path.joinpath('nodes.csv')), index_label='index')

        self.data.zones.to_csv(str(csv_path.joinpath('zones.csv')), index_label='index')
        self.data.heatareas.to_csv(str(csv_path.joinpath('heatareas.csv')), index_label='index')

        self.data.demand_el[self.data.demand_el.index.isin(self.model_horizon)] \
            .to_csv(str(csv_path.joinpath('demand_el.csv')), index_label='index')

        self.data.demand_h[self.data.demand_h.index.isin(self.model_horizon)] \
            .to_csv(str(csv_path.joinpath('demand_h.csv')), index_label='index')
        self.data.dclines[["node_i", "node_j", "maxflow"]] \
            .to_csv(str(csv_path.joinpath('dclines.csv')), index_label='index')

        self.data.ntc.to_csv(str(csv_path.joinpath('ntc.csv')), index=False)

        self.data.availability[self.data.availability.index.isin(self.model_horizon)] \
            .to_csv(str(csv_path.joinpath('availability.csv')), index_label='index')

        self.data.net_export[self.data.net_export.index.isin(self.model_horizon)] \
            .to_csv(str(csv_path.joinpath('net_export.csv')), index_label='index')

        self.data.net_position.loc[self.data.net_position.index.isin(self.model_horizon),
                                   self.data.zones.index] \
            .to_csv(str(csv_path.joinpath('net_position.csv')), index_label='index')

        self.data.reference_flows \
            .to_csv(str(csv_path.joinpath('reference_flows.csv')), index_label='index')

        plant_types = pd.DataFrame(index=self.data.plants.tech.unique())
        for ptype in self.options["plant_types"]:
            plant_types[ptype] = 0
            plant_types[ptype][plant_types.index.isin(self.options["plant_types"][ptype])] = 1
        plant_types.to_csv(str(csv_path.joinpath('plant_types.csv')), index_label='index')

        # Optional data
        self.grid_representation["cbco"] \
            .to_csv(str(csv_path.joinpath('cbco.csv')), index_label='index')

        # try:
        slack_zones = pd.DataFrame(index=self.data.nodes.index)
        for slack in self.grid_representation["slack_zones"]:
            slack_zones[slack] = 0
            condition = slack_zones.index.isin(self.grid_representation["slack_zones"][slack])
            slack_zones[slack][condition] = 1

        slack_zones.to_csv(str(csv_path.joinpath('slack_zones.csv')), index_label='index')
        # except:
        #     self.logger.warning("slack_zones.json not found - Check if relevant for the model")

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
