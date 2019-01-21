import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from data_worker import DataWorker
from data_input import InputProcessing
from data_results import ResultProcessing

pd.options.mode.chained_assignment = None  # default='warn'

class DataManagement(object):
	"""Data Set Class"""
	def __init__(self):
			# import logger
			self.logger = logging.getLogger('Log.MarketModel.DataManagement')
			self.logger.info("Initializing DataObject")

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

			infeasibility_variables = {variable: False for variable in ["INFEAS_H", "INFEAS_EL_N",
                            											"INFEAS_EL_Z", "INFEAS_LINES",
                            											"INFEAS_REF_FLOW"]}

			self.data_attributes = {"data": data, "source": None}

			self.result_attributes = {"variables": variables, "dual_variables": dual_variables,
                                 "infeasibility_variables": infeasibility_variables,
                                 "model_horizon": None, "source": None, "status": None,
                                 "objective": None, "t_start": None, "t_end": None
                                    }

			# Input Data as Attributes of DataManagement Class
			for d in data:
				setattr(self, d, pd.DataFrame())
			# Results are part of the results processing
			self.results = None

	def load_data(self, wdir, filepath):
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

	def process_input(self, set_up=None):
		InputProcessing(self, set_up)

	def process_results(self, opt_folder, opt_setup, grid=None):
		self.results = ResultProcessing(self, opt_folder, opt_setup, grid=grid)
		
		if not grid:
			self.logger.warning("Grid not set in Results Processing! \
								Manually set Grid as attribute to perform load flow analysis")

	def return_results(self, symb):
		"""interface method to allow access to results from ResultsProcessing class"""
		if self.results:
			try:
				return getattr(self.results, symb)
			except:
				self.logger.error("Symbol not in Results")
		else:
			self.logger.error("Results not Initialized")

	def _clear_all_data(self):
		attr = list(self.__dict__.keys())
		attr.remove('logger')
		for at in attr:
			delattr(self, at)
		self.is_empty = True

	def visulize_inputdata(self, folder, show_plot=True):
		"""Default Plots for Input Data"""
		if not Path.is_dir(folder):
			self.logger.warning(f"Folder {folder} does not exist!")
			self.logger.warning(f"Creating {folder}!")
			Path.mkdir(folder)

		if show_plot:
			plt.ion()
		else:
			plt.ioff()

		# Demand by Zone
		demand_zonal = pd.DataFrame(index=self.demand_el.index)
		for zone in self.zones.index:
			demand_zonal[zone] = self.demand_el[self.nodes.index[self.nodes.zone == zone]].sum(axis=1)
		fig, ax = plt.subplots()
		demand_zonal.plot.area(ax=ax, xticks=np.arange(0, len(demand_zonal.index), step=10))
		ax.legend(loc='upper right')
		ax.margins(x=0)
		fig.savefig(str(folder.joinpath("zonal_demand.png")))

		# Plot Installed Capacity by....
		for elm in ["fuel", "tech"]:
			inst_capacity = self.plants[["g_max", "zone", elm]].groupby([elm,"zone"], as_index=False).sum()
			fig, ax = plt.subplots()
			inst_capacity.pivot(index="zone", columns=elm, values="g_max").plot.bar(stacked=True, ax=ax)
			ax.legend(loc='upper right')
			ax.margins(x=0)
			fig.savefig(str(folder.joinpath(f"installed_capacity_by_{elm}.png")))