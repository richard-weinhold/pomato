import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import logging


def _mpc_data_pu_to_real(lines,  base_kv, base_mva):
	v_base = base_kv * 1e3
	s_base = base_mva * 1e6
	z_base = np.power(v_base,2)/s_base
	lines['r'] = np.multiply(lines['r'], z_base)
	lines['x'] = np.multiply(lines['x'], z_base)
	lines['b_other'] = np.divide(lines['b_other'], z_base)
	lines['b'] = np.divide(1, lines['x'])
	# Some numerical correction
	# lines['b'] = np.multiply(lines['b'], 1)
	lines['maxflow'] = np.ones(len(lines['idx']))*600
	return lines

class DataWorker(object):
	"""Data Woker Class"""
	def __init__(self, data, file_path):
		self.logger = logging.getLogger('Log.MarketModel.DataManagement.DataWorker')
		self.data = data

		if ".xls" in str(file_path):
			self.logger.info("Loading data from xls file")
			print(file_path)
			self.read_xls(file_path)

		elif ".mat" in str(file_path):
			self.logger.info("Loading data from matpower case-file")
			self.read_matpower_case(file_path)
		else:
			self.logger.warning("Data Type not supported, only .xls(x) or .mat")

	def read_xls(self, xls_file):
		self.logger.info("Reading Data from Excel File")
		xls = pd.ExcelFile(xls_file)
		self.data.data_attributes["source"] = "xls file"

		for data in self.data.data_attributes["data"]:
			try:
				setattr(self.data, data, xls.parse(data, index_col=0))
				self.data.data_attributes["data"][data] = True
			except:
				self.logger.warning(f"{data} not in excel file")

	def read_matpower_case(self, casefile):
		self.logger.info("Reading MatPower Casefile")
		self.data.data_attributes["source"] = "mpc_casefile"

		MPCOLNAMES = {'bus_keys': np.array(['bus_i', 'b_type', 'Pd',
											'Qd', 'Gs', 'Bs', 'area',
											'Vm', 'Va', 'baseKV',
											'zone', 'Vmax', 'Vmin']),

					  'gen_keys': np.array(['bus', 'Pg', 'Qg', 'Qmax',
					  						'Qmin', 'Vg', 'mBase',
									  		'status', 'Pmax', 'Pmin',
									  		'Pc1', 'Pc2', 'Qc1min',
									  		'Qc1max', 'Qc2min', 'Qc2max',
											'ramp_agc', 'ramp_10',
											'ramp_30', 'ramp_q', 'apf']),

						'branch_keys': np.array(['fbus', 'tbus', 'r', 'x',
												 'b', 'rateA', 'rateB',
										 		 'rateC', 'ratio', 'angle',
										 		 'status', 'angmin', 'angmax']),

						'gencost_keys': np.array(['model', 'startup',
												  'shutdown', 'n'])}

		case_raw = sio.loadmat(casefile)
		mpc = case_raw['mpc']
		bus = mpc['bus'][0,0]
		gen = mpc['gen'][0,0]
		baseMVA = mpc['baseMVA'][0,0]
		branch = mpc['branch'][0,0]
		gencost = mpc['gencost'][0,0]
		try:
			busname = mpc['bus_name'][0,0]
		except:
			busname = np.array([])
		docstring = mpc['docstring'][0,0]
		n = int(gencost[0,3])
		for i in range(n):
			MPCOLNAMES['gencost_keys'] = np.append(MPCOLNAMES['gencost_keys'], 'x{}'.format(n-i-1))
		bus_df = pd.DataFrame(bus, columns=MPCOLNAMES['bus_keys'])
		gen_df = pd.DataFrame(gen, columns=MPCOLNAMES['gen_keys'])
		branch_df = pd.DataFrame(branch, columns=MPCOLNAMES['branch_keys'])
		gencost_df = pd.DataFrame(gencost, columns=MPCOLNAMES['gencost_keys'])
		caseinfo = docstring[0]

		mpc_buses = {
				'idx': bus_df['bus_i'],
				'zone': bus_df['zone'],
				'Pd': bus_df['Pd'],
				'Qd': bus_df['Qd'],
				'baseKV': bus_df['baseKV']
				}

		# find and set slack bus
		if 3.0 in bus_df['b_type']:
			slackbus_idx = bus_df['b_type'][bus_df['b_type'] == 3.0].index[0]
			slackbus = bus_df['bus_i'][slackbus_idx]
			self.logger.info("Slackbus read as {:.0f}".format(slackbus))
		else:
			slackbus_idx = 0
			slackbus = bus_df['bus_i'][0]
			self.logger.info("Slackbus set to default {}".format(slackbus))

		slack = np.zeros(len(bus_df['bus_i']))
		slack[slackbus_idx] = 1
		mpc_buses['slack'] = slack
		mpc_buses['slack'] = mpc_buses['slack'].astype(bool)
		mpc_buses['net_injection'] = np.zeros(len(mpc_buses['idx']))

		# add verbose names if available
		if busname.any():
			b_name = []
			for b in busname:
				b_name.append(b[0][0])
			b_name = np.array(b_name)
			mpc_buses['name'] = b_name

		lineidx = ['l{}'.format(i) for i in range(0,len(branch_df.index))]
		mpc_lines = {
				'idx': lineidx,
				'node_i': branch_df['fbus'],
				'node_j': branch_df['tbus'],
				'maxflow': branch_df['rateA'],
				'b_other': branch_df['b'],
				'r': branch_df['r'],
				'x': branch_df['x']
				}
		mpc_lines = _mpc_data_pu_to_real(mpc_lines, mpc_buses['baseKV'][0], baseMVA[0][0])

		contingency = np.ones(len(mpc_lines['idx']))
		mpc_lines['contingency'] = contingency.astype(bool)

		ng = len(gen_df.index)
		genidx = ['g{}'.format(i) for i in range(ng)]
		mpc_generators = {
					'idx': genidx,
					'g_max': gen_df['Pmax'],
					'g_max_Q': gen_df['Qmax'],
					'node': gen_df['bus'],
					'apf': gen_df['apf'],
					'mc': gencost_df['x2'][list(range(0,ng))],
					'mc_Q': np.zeros(ng)
					}
		if len(gencost_df.index) == 2*ng:
			mpc_generators['mc_Q'] = gencost_df['x2'][list(range(ng,2*ng))].tolist

		self.data.lines = pd.DataFrame(mpc_lines).set_index('idx')
		self.data.nodes = pd.DataFrame(mpc_buses).set_index('idx')
		self.data.plants = pd.DataFrame(mpc_generators).set_index('idx')

		### Make ieee case ready for the market model
		self.data.nodes["name"] = ["n" + str(int(idx)) for idx in self.data.nodes.index]
		self.data.nodes.set_index("name", drop=False, inplace=True)
		self.data.nodes.zone = ["z" + str(int(idx)) for idx in self.data.nodes.zone]

		self.data.lines.node_i = ["n" + str(int(idx)) for idx in self.data.lines.node_i]
		self.data.lines.node_j = ["n" + str(int(idx)) for idx in self.data.lines.node_j]
		self.data.zones = pd.DataFrame(index=set(self.data.nodes.zone.values))
		self.data.plants.node = ["n" + str(int(idx)) for idx in self.data.plants.node]
		self.data.demand_el = pd.DataFrame(index=["t0001"], data=self.data.nodes.Pd.to_dict())

		self.data.plants = self.data.plants[["g_max", "mc", "node"]]

		self.data.plants["tech"] = self.data.plants.index
		self.data.plants["eta"] = 1
		self.data.plants["h_max"] = 0

		# mark read data as true in datamanagement attributes
		self.data.data_attributes["data"]["lines"] = True
		self.data.data_attributes["data"]["nodes"] = True
		self.data.data_attributes["data"]["plants"] = True
		self.data.data_attributes["data"]["zones"] = True
		self.data.data_attributes["data"]["demand_el"] = True

