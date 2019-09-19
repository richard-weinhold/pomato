"""
Data Management
"""
# pylint: disable-msg=E1101

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_worker import DataWorker
from data_input import InputProcessing
from data_results import ResultProcessing

class DataManagement():
    """Data Set Class"""
    def __init__(self, options):
        # import logger
        self.logger = logging.getLogger('Log.MarketModel.DataManagement')
        self.logger.info("Initializing DataObject")

        self.options = options
        self.wdir = None
        # init
        data = {data: False for data in ["lines", "nodes", "zones",
                                         "heatareas", "plants",
                                         "dclines", "tech", "fuel",
                                         "demand_el", "demand_h",
                                         "timeseries", "availability",
                                         "ntc", "net_position",
                                         "reference_flows", "frm_fav",
                                         "net_export"]}

        variables = {variable: False for variable in ["G", "H", "D_es", "L_es",
                                                      "D_hs", "L_hs", "INJ", "F_DC",
                                                      "D_ph", "D_d", "EX"]}

        dual_variables = {variable: False for variable in ["EB_nodal", "EB_zonal"]}

        infeasibility_variables = {variable: False \
                                       for variable in ["INFEAS_H_POS", "INFEAS_H_NEG",
                                                        "INFEAS_EL_N_POS", "INFEAS_EL_N_NEG",
                                                        "INFEAS_EL_Z_POS", "INFEAS_EL_Z_NEG",
                                                        "INFEAS_LINES", "INFEAS_REF_FLOW"]}

        self.data_attributes = {"data": data, "source": None}

        self.result_attributes = {"variables": variables, "dual_variables": dual_variables,
                                  "infeasibility_variables": infeasibility_variables,
                                  "model_horizon": None, "source": None, "status": None,
                                  "objective": None, "t_start": None, "t_end": None
                                  }

        # Input Data as Attributes of DataManagement Class
        for attr in data:
            setattr(self, attr, pd.DataFrame())

        # Results are part of the results processing
        self.results = None
    def save_data(self, wdir, filepath):
        """Write Data to excel file"""
        xls_file = wdir.joinpath(filepath)

        self.logger.info("Writing Data to Excel File %s", str(xls_file))
        with pd.ExcelWriter(xls_file) as writer:
            for data in self.data_attributes["data"]:
                getattr(self, data).to_excel(writer, sheet_name=data)

    def load_data(self, wdir, filepath):
        """
        Load Data from data set at filepath
        Currently xls(x) and mat work
        filepath with or without ext
        """
        ### PATH ARETMETICS INCOMING
        self.wdir = wdir
        ### Make sure wdir/file_path or wdir/data/file_path is a file
        if self.wdir.joinpath(filepath).is_file():
            DataWorker(self, self.wdir.joinpath(filepath))
            self.process_input()


        elif self.wdir.joinpath(f"data_input/{filepath}").is_file():
            DataWorker(self, self.wdir.joinpath(f"data/{filepath}"))
            self.process_input()

        elif self.wdir.joinpath(f"data_input/mp_casedata/{filepath}.mat").is_file():
            DataWorker(self, self.wdir.joinpath(f"data_input/mp_casedata/{filepath}.mat"))
            self.process_input()
        else:
            self.logger.error("Data File not found!")

    def process_input(self):
        """
        Input Processing in Seperate Class
        Will Change data attr based on options["data"]
        """
        if self.options["data"]["process_input"]:
            InputProcessing(self, self.options)
        else:
            self.logger.info("Input Data not processed")

    def process_results(self, opt_folder, opt_setup, grid=None):
        """ Init Results Calss with results_folder and self"""
        self.results = ResultProcessing(self, opt_folder, opt_setup, grid=grid)

    def return_results(self, symb):
        """interface method to allow access to results from ResultsProcessing class"""
        if self.results and symb in self.result.__dict__.keys():
            return_value = getattr(self.results, symb)
        else:
            if not self.results:
                self.logger.error("Results not Initialized")
            else:
                self.logger.error("Symbol not in in results class")
            return_value = None
        return return_value

    def _clear_all_data(self):
        attr = list(self.__dict__.keys())
        attr.remove('logger')
        for att in attr:
            delattr(self, att)

    def visulize_inputdata(self, folder, show_plot=True):
        """Default Plots for Input Data"""
        if not Path.is_dir(folder):
            self.logger.warning("Folder %s does not exist!", folder)
            self.logger.warning("Creating %s", folder)
            Path.mkdir(folder)

        if show_plot:
            plt.ion()
        else:
            plt.ioff()

        # Demand by Zone
        demand_zonal = pd.DataFrame(index=self.demand_el.index)
        for zone in self.zones.index:
            nodes_in_zone = self.nodes.index[self.nodes.zone == zone]
            demand_zonal[zone] = self.demand_el[nodes_in_zone].sum(axis=1)
        fig_demand, ax_demand = plt.subplots()
        demand_zonal.plot.area(ax=ax_demand, xticks=np.arange(0, len(demand_zonal.index), step=10))
        ax_demand.legend(loc='upper right')
        ax_demand.margins(x=0)
        fig_demand.savefig(str(folder.joinpath("zonal_demand.png")))

        # Plot Installed Capacity by....
        plants_zone = pd.merge(self.plants, self.nodes.zone,
                               how="left", left_on="node", right_on="index")

        for elm in ["fuel", "tech"]:
            inst_capacity = plants_zone[["g_max", "zone", elm]].groupby([elm, "zone"],
                                                                        as_index=False).sum()
            fig_gen, ax_gen = plt.subplots()
            inst_capacity.pivot(index="zone",
                                columns=elm,
                                values="g_max").plot.bar(stacked=True, ax=ax_gen)

            ax_gen.legend(loc='upper right')
            ax_gen.margins(x=0)
            fig_gen.savefig(str(folder.joinpath(f"installed_capacity_by_{elm}.png")))
