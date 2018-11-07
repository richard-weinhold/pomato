import sys
import numpy as np
import pandas as pd
import logging

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
											 "net_export", "net_export_nodes"]}

			variables = {variable: False for variable in ["G", "H", "D_es", "L_es",
														  "D_hs", "L_hs", "INJ", "F_DC",
														  "D_ph", "D_d", "EX"]}

			dual_variables = {variable: False for variable in ["EB_nodal", "EB_zonal"]}

			infeasibility_variables = {variable: False for variable in ["INFEAS_H", "INFEAS_EL_N",
                            											"INFEAS_EL_Z", "INFEAS_LINES", 
                            											"INFEAS_REF_FLOW"]}

			self.data_attributes = {"data": data, "source": None}

			self.result_attributes = {"variables": variables,
            						  "dual_variables": dual_variables,
		                              "infeasibility_variables": infeasibility_variables,
		                              "source": None, "status": None, "objective": None,
		            				  "t_start": None, "t_end": None}

			# Input Data as Attributes of DataManagement Class
			for d in data:
				setattr(self, d, pd.DataFrame())
			# Results are part of the results processing
			self.results = None

	def load_data(self, wdir, filepath):
		### PATH ARETMETICS INCOMING
		self.wdir = wdir
		self.create_folder_structure()
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

	def create_folder_structure(self):
		folders = ["data", "output"]
		for folder in folders:
			if not self.wdir.joinpath(folder).is_dir():
				self.wdir.joinpath(folder).mkdir()

	def process_input(self, set_up=None):
		InputProcessing(self, set_up)

	def process_results(self, opt_folder, opt_setup):
		self.results = ResultProcessing(self, opt_folder, opt_setup)

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




