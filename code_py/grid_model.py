"""
GRID Model
"""
import sys
import logging
import numpy as np
import pandas as pd
import tables

class GridModel(object):
	"""GRID Model Class"""
	numpy_settings = np.seterr(divide="raise")
	def __init__(self, wdir):
		self.logger = logging.getLogger('Log.MarketModel.GridModel')
#        self.logger.info("Initializing GridModel..")
		self.wdir = wdir
		self.is_empty = True

	def build_grid_model(self, nodes, lines):
		try:
			# import logger
			self.logger.info("Initializing GridModel..")
			self.nodes = nodes
			self.lines = lines
			self.check_slack()

			self.logger.info("Calculating PTDF and PSDF Matrices!")
			self.ptdf = self.create_ptdf_matrix()
			self.psdf = self.create_psdf_matrix()

			self.check_grid_topology()

			self.logger.info("Calculating LODF Matrix!")
			self.lodf = self.create_lodf_matrix()

			self.logger.info("GridModel initialized!")

			self.is_empty = False
		except:
			self.logger.exception("Error in GridModel!")

	def check_slack(self):
		"""
		Set any node as slack, if no sleck is defined
		Add Unconnected nodes as additional slack
		"""
		if not any(self.nodes.slack):
			self.logger.warning("No slack detected, setting first node as slack!")
			self.nodes.loc[self.nodes.index[0], "slack"] = True

		condition = ~(self.nodes.index.isin(self.lines.node_i) | \
					  self.nodes.index.isin(self.lines.node_j))
		if any(condition):
			self.logger.warning(f"{sum(condition)} unconnected nodes detected, "
								"set as slacks!")

		self.nodes.loc[condition, "slack"] = True

		self.mult_slack = bool(len(self.nodes.index[self.nodes.slack]) > 1)
		if self.mult_slack:
			self.logger.info("Multiple Slacks Detected!")

	def check_grid_topology(self):
		"""Checking grid topology for radial nodes and lines"""
		self.logger.info("Checking Grid Topology...")

		radial_nodes = []
		for node in self.nodes.index:
			if len(self.lines[(self.lines.node_i == node)|(self.lines.node_j == node)]) < 2:
				radial_nodes.append(node)

		radial_lines = []
		for idx, line in enumerate(self.lines.index):
			tmp = np.abs(self.ptdf[idx, :])
			tmp = np.around(tmp, decimals=3)
			if 1 in tmp:
				radial_lines.append(line)
#			elif self.lines.at["l1", "b"] == 0:
#				radial_lines.append(line)

		condition = ((self.lines.node_i.isin(radial_nodes)) | \
					 (self.lines.node_j.isin(radial_nodes))) & \
					(self.lines.contingency)

		if not self.lines[condition].empty:
			self.logger.info("Radial nodes are part of the contingency: " + \
							 ", ".join(list(self.lines.index[condition])))
			self.lines.loc[condition, "contingency"] = False

			self.logger.info("Contingency of radial nodes is set to False")

		condition = self.lines.index.isin(radial_lines) & self.lines.contingency
		if not self.lines.contingency[condition].empty:
			self.logger.info("Radial lines are part of the contingency: " + \
							 ", ".join(list(self.lines.index[condition])))
			self.lines.loc[condition, "contingency"] = False
			self.logger.info("Contingency of radial lines is set to False")

	def slack_zones(self):
		"""
		returns number of nodes that are part of a synchronous area
		for each slack defined
		"""
		## Creates Slack zones, given that the given slacks are well suited
		## Meaning one slack per zone, all zones have a slack.
		# Method: slack -> find Line -> look at ptdf
		# all non-zero elementes are in slack zone.
		slacks = self.nodes.index[self.nodes.slack]
		slack_zones = {}
		for slack in slacks:
			condition = (self.lines.node_i == slack)|(self.lines.node_j == slack)
			if self.lines.index[condition].empty:
				slack_zones[slack] = [slack]
			else:
				slack_line = self.lines.index[condition][0]
				line_index = self.lines.index.get_loc(slack_line)
				pos = self.ptdf[line_index, :] != 0
				tmp = list(self.nodes.index[pos])
				tmp.append(slack)
				slack_zones[slack] = tmp

		return slack_zones

	def create_incedence_matrix(self):
		"""Create incendence matrix"""
		incedence = np.zeros((len(self.lines), len(self.nodes)))
		for i, elem in enumerate(self.lines.index):
			incedence[i, self.nodes.index.get_loc(self.lines.node_i[elem])] = 1
			incedence[i, self.nodes.index.get_loc(self.lines.node_j[elem])] = -1
		return incedence

	def create_susceptance_matrices(self):
		""" Create Line (Bl) and Node (Bn) susceptance matrix """
		suceptance_vector = self.lines.b
		incedence = self.create_incedence_matrix()
		susceptance_diag = np.diag(suceptance_vector)
		line_susceptance = np.dot(susceptance_diag, incedence)
		node_susceptance = np.dot(np.dot(incedence.transpose(1, 0), susceptance_diag), incedence)
		return(line_susceptance, node_susceptance)

	def create_ptdf_matrix(self):
		""" Create ptdf Matrix """
		try:
			#Find slack
			slack = list(self.nodes.index[self.nodes.slack])
			slack_idx = [self.nodes.index.get_loc(s) for s in slack]
			line_susceptance, node_susceptance = self.create_susceptance_matrices()
			#Create List without the slack and invert it
			list_wo_slack = [x for x in range(0, len(self.nodes.index)) \
							if x not in slack_idx]

			node_susceptance_wo_slack = node_susceptance[np.ix_(list_wo_slack, list_wo_slack)]
			inv = np.linalg.inv(node_susceptance_wo_slack)
			#sort slack back in to get nxn
			node_susc_inv = np.zeros((len(self.nodes), len(self.nodes)))
			node_susc_inv[np.ix_(list_wo_slack, list_wo_slack)] = inv
			#calculate ptdfs
			ptdf = np.dot(line_susceptance, node_susc_inv)
			return ptdf
		except:
			self.logger.exception('error:create_ptdf_matrix')

	def create_psdf_matrix(self):
		"""
		Calculate psdf (phase-shifting distribution matrix, LxLL)
		meaning the change of p.u. loadflow
		on a line LL through a phase-shifting by 1rad at line L
		"""
		line_susceptance, _ = self.create_susceptance_matrices()
		psdf = np.diag(self.lines.b) - np.dot(self.ptdf, line_susceptance.T)
		return psdf

	def shift_phase_on_line(self, shift_dict):
		"""
		Shifts the phase on line l by angle a (in rad)
		Recalculates the ptdf matrix and replaces it as the attribute
		"""
		shift = np.zeros(len(self.lines))
		for line in shift_dict:
			shift[self.lines.index.get_loc(line)] = shift_dict[line]
		# recalc the ptdf
		shift_matrix = np.multiply(self.psdf, shift)
		self.ptdf += np.dot(shift_matrix, self.ptdf)
		# subsequently recalc lodf
		self.lodf = self.create_lodf_matrix()

	def create_lodf_matrix_old(self):
		""" Load outage distribution matrix -> Line to line sensitivity """
		try:
			lodf = np.zeros((len(self.lines), len(self.lines)))
			incedence = self.create_incedence_matrix()
			for idx, _ in enumerate(self.lines.index):
				for idy, line in enumerate(self.lines.index):
					## Exception for lines that are not in the contingency
					if line in self.lines.index[~self.lines.contingency]:
						lodf[idx, idy] = 0
					elif idx == idy:
						lodf[idx, idy] = -1
					else:
						lodf[idx, idy] = (np.dot(incedence[idy, :], self.ptdf[idx, :]))/ \
									   (1-np.dot(incedence[idy, :], self.ptdf[idy, :]))
			return lodf
		except:
			self.logger.exception("error in create_lodf_matrix ", sys.exc_info()[0])
			raise ZeroDivisionError('LODFError: Check Slacks, radial Lines/Nodes')

	def create_lodf_matrix(self):
		""" Load outage distribution matrix -> Line to line sensitivity """
		try:
			incedence = self.create_incedence_matrix()
			H = np.dot(self.ptdf, incedence.T)
			h = np.diag(H).reshape(len(self.lines), 1)
			## Avoid division by zero because of radial nodes and lines
			con = self.lines.contingency.values.astype(int).reshape(len(self.lines), 1)
			h = np.multiply(h,con)
			lodf = np.divide(H, (np.ones((len(self.lines), len(self.lines))) - np.dot(np.ones((len(self.lines), 1)), h.T)))
			lodf = lodf - np.diag(np.diag(lodf)) - np.eye(len(self.lines), len(self.lines))
			# explicitly set line-line sensitivity to 0 for contingency==False
			lodf = np.multiply(lodf, np.ones((len(self.lines),1))*self.lines.contingency.values)
			return lodf

		except:
			self.logger.exception("error in create_lodf_matrix ", sys.exc_info()[0])
			raise ZeroDivisionError('LODFError: Check Slacks, radial Lines/Nodes')

	def lodf_filter(self, line, sensitivity=5e-2, as_index=False):
		"""return outages that have a sensitivity to a line > 5%"""
		if not isinstance(line, int):
			line = self.lines.index.get_loc(line)

		cond = abs(np.multiply(self.lodf[line], self.lines.maxflow.values)) >= sensitivity*self.lines.maxflow[line]

		if as_index:
			return [self.lines.index.get_loc(line) for line in self.lines.index[cond]]
		else:
			return self.lines.index[cond]

	def create_n_1_ptdf_line(self, line):
		"""
		Creates N-1 ptdf for one specific line - all outages
		returns LxN Matrix, where ptdf*inj = flows on specified line with all outages
		"""
		try:
			if not isinstance(line, int):
				line = self.lines.index.get_loc(line)

			n_1_ptdf_line =  np.vstack([self.ptdf[line, :]] + \
									   [self.ptdf[line, :] + np.dot(self.lodf[line, outage], self.ptdf[outage, :]) \
										for outage in range(0, len(self.lines))])
			return n_1_ptdf_line
		except:
			self.logger.exception('error:create_n_1_ptdf_cb')

	def create_n_1_ptdf_cbco(self, line, outage):
		"""
		Creates N-1 ptdf for one specific line and outage
		returns LxN Matrix, where ptdf*inj = flow on specified line under specified outages
		"""
		try:
			if not isinstance(outage, int):
				outage = self.lines.index.get_loc(outage)
			if not isinstance(line, int):
				line = self.lines.index.get_loc(line)

			n_1_ptdf_cbco =  self.ptdf[line, :] + np.dot(self.lodf[line, outage], self.ptdf[outage, :])

			return n_1_ptdf_cbco
		except:
			self.logger.exception('error:create_n_1_ptdf_cbco')

	def create_n_1_ptdf_outage(self, outage):
		"""
		Create N-1 ptdf for one specific outage - all lines
		returns LxN Matrix, where ptdf*inj = lineflows with specified outage
		"""
		try:
			if not isinstance(outage, int):
				outage = self.lines.index.get_loc(outage)
			n_1_ptdf = np.array(self.ptdf, dtype=np.float) + np.vstack([np.dot(self.lodf[lx, outage], self.ptdf[outage, :]) for lx in range(0, len(self.lines))])
			return n_1_ptdf

		except:
			self.logger.exception('error:create_n_1_ptdf_co')


	def create_filtered_n_1_ptdf(self, sensitivity=5e-2):
		"""
		Create all relevant N-1 ptdfs in the for of Ax<b (ptdf x < ram):
		For each line as CB add basecase (N-0)
		and COs based on the senstitivity in LODF (default = 5%)
		return ptdf, corresponding ram and df with the relevant info
		"""
		try:
			A = [self.ptdf]
			label_lines = list(self.lines.index)
			label_outages = ["basecase" for i in range(0, len(self.lines.index))]

			for idx, line in enumerate(self.lines.index[self.lines.contingency]):
				outages = list(self.lodf_filter(line, sensitivity))
				label_lines.extend([line for i in range(0, len(outages))])
				label_outages.extend(outages)

			# estimate size of array = nr_elements * bits per element (float32) / (8 * 1e6) MB
			estimate_size = len(label_lines)*len(self.nodes.index)*32/(8*1e6)
			self.logger.info(f"Estimated size in RAM for A is: {estimate_size} MB")
			if estimate_size > 3000:
				raise

			for idx, line in enumerate(self.lines.index[self.lines.contingency]):
				outages = list(self.lodf_filter(line, lodf_sensitivity))
				tmp_ptdf = np.vstack([self.create_n_1_ptdf_cbco(line,o) for o in outages])
				A.append(tmp_ptdf)

			### add x_i > 0 for i in I
			A = np.concatenate(A).reshape(len(label_lines), len(list(self.nodes.index)))
			b = self.lines.maxflow[label_lines].values.reshape(len(label_lines), 1)

			df_info = pd.DataFrame(columns=list(self.nodes.index), data=A)
			df_info["cb"] = label_lines
			df_info["co"] = label_outages
			df_info["ram"] = b
			df_info = cbco_info[["cb", "co", "ram"] + list(list(self.nodes.index))]
			return A, b, df_info
		except:
			self.logger.exception('error:create_n_1_ptdf')

	def slack_zones_index(self):
		"""returns the indecies of nodes per slack_zones
		(aka control area/synchronious area) in the A matrix"""
		slack_zones = self.slack_zones()
		slack_zones_idx = []
		for slack in slack_zones:
			slack_zones_idx.append([self.nodes.index.get_loc(node) \
									for node in slack_zones[slack]])
		slack_zones_idx.append([x for x in range(0, len(self.nodes))])
		return slack_zones_idx

