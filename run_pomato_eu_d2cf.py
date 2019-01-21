import sys
from pathlib import Path
code_py = Path.cwd().joinpath("code_py")
sys.path.append(str(code_py))
import matplotlib.pyplot as plt

from market_tool import MarketTool

import numpy as np
import pandas as pd
import tools
import time

mato = MarketTool(opt_file="profiles/cbco_nodal.json",
                  model_horizon=range(0,1))
#mato.load_data('data_input\dataset_de.xlsx')
mato.load_data('data_input\pglib_casedata\pglib_opf_case118_ieee.m')
#mato.load_data('case118')
#
t = mato.data.lines
p = mato.data.plants

#t = pd.DataFrame(columns=['node_i', 'node_j', 'name_i', 'name_j', 'maxflow'])
#mato.data.lines.maxflow = 0.2
#mato.grid.build_grid_model(mato.data.nodes, mato.data.lines)
#mato.create_grid_representation(precalc_filename="cbco_01_ieee118_2101_1648")

#t = mato.grid_representation
#t = mato.data.nodes
#mato.data.lines.cb = True
#mato.data.visulize_inputdata(mato.wdir)
#mato.data.lines.cb = True
#tmp = mato.data.lines.cb[mato.data.lines.cb]
#mato.data.process_results(mato.wdir.joinpath("data_temp\\julia_files\\results\\1612_2029"),
#                          mato.opt_setup, grid=mato.grid)


#mato.init_market_model()
#mato.run_market_model()

#df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)
#df3 = mato.data.results.n_1_flow()

#mato.data.results.INJ[mato.data.results.INJ.INJ > 1e4]

#mato.data.results.default_plots()

#from bokeh_plot_interface import BokehPlot
#mato.init_bokeh_plot(name="n168")
#mato.bokeh_plot.start_server()
#mato.bokeh_plot.stop_server()
#
#overloaded_lines = mato.check_n_1_for_marketresult()
##add_cbco = []
##for t in overloaded_lines:
##    for nr in overloaded_lines[t]:
##        cbco = [overloaded_lines[t][nr]["Line"], overloaded_lines[t][nr]["Outage"]]
##        if not cbco in add_cbco:
##            add_cbco.append(cbco)
#
##print("OK")

#%%
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
###
#####%%
from cbco_module import CBCOModule
cbco_module = CBCOModule(mato.wdir, mato.grid)
cbco_module.cbco_algorithm(False)
df = cbco_module.return_cbco()
#path = mato.wdir.joinpath("data_temp/julia_files/cbco_data/")
#A, b, info = cbco_module.create_Ab(preprocess=True)
#idx = pd.read_csv(path.joinpath("cbco_01_ieee118_2101_1648.csv"), delimiter=',').constraints.values

#np.savetxt(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("A_ieee118.csv"), np.asarray(A), delimiter=",")
#np.savetxt(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("b_ieee118.csv"), np.asarray(b), delimiter=",")
##
#vertices = np.array(cbco_module.reduce_Ab_convex_hull())
#D = A[vertices]/b[vertices]
#model = PCA(n_components=8).fit(D)
#D_t = model.transform(D)
#k = ConvexHull(D_t, qhull_options="Qx")
###
#convex_hull = vertices[k.vertices]
##
#t = [i for i in vertices if i not in idx]



#np.savetxt(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("I_nl.csv"), convex_hull, fmt='%i', delimiter=",")
#
#gurobi = pd.read_csv(path.joinpath("cbco_01_nl_pre_1001_1535.csv"), delimiter=',').constraints.values
#glpk = pd.read_csv(path.joinpath("cbco_01_nl_pre_1001_1534.csv"), delimiter=',').constraints.values

#%%
#mato.data.nodes["gsk"] = 1
#from fbmc_module import FBMCModule
#fbmc = FBMCModule(mato.wdir, mato.grid, mato.data.results.INJ, mato.data.frm_fav)
####
#
###gsk_sink = {key: 0 for key in mato.data.zones.index}
###for key in ["NL"]:
###    gsk_sink[key] = 1
##
##for timestep in injection.t.unique():
#for timestep in ["t0001", "t0002", "t0003"]:
##for timestep in ["t0001"]:
#    for gsk_strat in ["jao", "flat", "G", "g_max", "g_max_G_flat"]:
##    for gsk_strat in ["flat"]:
#        fbmc.update_plot_setup(timestep, gsk_strat)
#        fbmc.plot_fbmc(["DE", "FR"], ["DE", "NL"])
##        plot = fbmc.plot_fbmc(["DE"], ["FR"], gsk_sink)
#
##fbmc.save_all_domain_plots(mato.data.results.result_folder)
#fbmc.save_all_domain_info(mato.data.results.result_folder)

#%%
#import scipy.io as sio
#
#def _mpc_data_pu_to_real(lines,  base_kv, base_mva):
#    v_base = base_kv * 1e3
#    s_base = base_mva * 1e6
#    z_base = np.power(v_base,2)/s_base
#    lines['r'] = np.multiply(lines['r'], z_base)
#    lines['x'] = np.multiply(lines['x'], z_base)
#    lines['b_other'] = np.divide(lines['b_other'], z_base)
#    lines['b'] = np.divide(1, lines['x'])
#    # Some numerical correction
#    # lines['b'] = np.multiply(lines['b'], 1)
#    # lines['maxflow'] = np.ones(len(lines['idx']))*600
#    return lines
#
#
#		MPCOLNAMES = {'bus_keys': np.array(['bus_i', 'type', 'Pd',
#											'Qd', 'Gs', 'Bs', 'area',
#											'Vm', 'Va', 'baseKV',
#											'zone', 'Vmax', 'Vmin']),
#
#					  'gen_keys': np.array(['bus', 'Pg', 'Qg', 'Qmax',
#					  						'Qmin', 'Vg', 'mBase',
#									  		'status', 'Pmax', 'Pmin',
#									  		'Pc1', 'Pc2', 'Qc1min',
#									  		'Qc1max', 'Qc2min', 'Qc2max',
#											'ramp_agc', 'ramp_10',
#											'ramp_30', 'ramp_q', 'apf']),
#
#						'branch_keys': np.array(['fbus', 'tbus', 'r', 'x',
#												 'b', 'rateA', 'rateB',
#										 		 'rateC', 'ratio', 'angle',
#										 		 'status', 'angmin', 'angmax']),
#
#						'gencost_keys': np.array(['model', 'startup',
#												  'shutdown', 'n'])}
#
#    casefile = 'data_input\mp_casedata\case118.mat'
#
#		case_raw = sio.loadmat(casefile)
#		mpc = case_raw['mpc']
#		bus = mpc['bus'][0,0]
#		gen = mpc['gen'][0,0]
#		baseMVA = mpc['baseMVA'][0,0][0][0]
#		branch = mpc['branch'][0,0]
#		gencost = mpc['gencost'][0,0]
#
#		try:
#			busname = mpc['bus_name'][0,0]
#		except:
#			busname = np.array([])
#		docstring = mpc['docstring'][0,0]
#		n = int(gencost[0,3])
#
#		for i in range(n):
#			MPCOLNAMES['gencost_keys'] = np.append(MPCOLNAMES['gencost_keys'], 'c{}'.format(n-i-1))
#		bus_df = pd.DataFrame(bus, columns=MPCOLNAMES['bus_keys'])
#		gen_df = pd.DataFrame(gen, columns=MPCOLNAMES['gen_keys'])
#		branch_df = pd.DataFrame(branch, columns=MPCOLNAMES['branch_keys'])
#		gencost_df = pd.DataFrame(gencost, columns=MPCOLNAMES['gencost_keys'])
#		caseinfo = docstring[0]
#

#casefile = 'data_input\pglib_casedata\pglib_opf_case118_ieee.m'
#with open(casefile) as mfile:
#    t = mfile.read()
#
#tt = t.splitlines()
#is_table = False
#tables, table = [], []
#for line in tt:
#    if "function mpc" in line:
#        caseinfo = line.split()[-1]
#    if "mpc.baseMVA" in line:
#        baseMVA = float(line.lstrip("mpc.baseMVA = ").rstrip(";"))
#    if "%%" in line:
#        table = []
#        is_table = True
#    if "];" in line:
#        is_table = False
#        tables.append(table)
#    if is_table:
#        table.append(line)
#
#df_dict = {}
#for table in tables:
#    name = table[0].lstrip("%% ")
#    columns = table[1].lstrip("%\t").split()
#    data = [row.split(";")[0].split() for row in table[3:]]
#    df = pd.DataFrame(data=data, columns=columns, dtype=float)
#    df_dict[name] = df
#
#busname = np.array([])
#bus_df = df_dict["bus data"]
#gen_df = df_dict["generator data"]
#gencost_df = df_dict["generator cost data"]
#gencost_columns = ["model", "startup", "shutdown", "n"] + \
#                  ["c" + str(x) for x in range(int(gencost_df.n.values[0]-1), -1, -1)]
#gencost_df.columns = gencost_columns
#
#branch_df = df_dict["branch data"]
#
#caseinfo, busname, baseMVA, bus_df, gen_df, branch_df, gencost_df
#
#
#
#mpc_buses = {
#        'idx': bus_df['bus_i'],
#        'zone': bus_df['zone'],
#        'Pd': bus_df['Pd'],
#        'Qd': bus_df['Qd'],
#        'baseKV': bus_df['baseKV']
#        }
##
## find and set slack bus
#if 3.0 in bus_df['type']:
#    slackbus_idx = bus_df['type'][bus_df['type'] == 3.0].index[0]
#    slackbus = bus_df['bus_i'][slackbus_idx]
#else:
#    slackbus_idx = 0
#    slackbus = bus_df['bus_i'][0]
#
#slack = np.zeros(len(bus_df['bus_i']))
#slack[slackbus_idx] = 1
#mpc_buses['slack'] = slack
#mpc_buses['slack'] = mpc_buses['slack'].astype(bool)
#
## add verbose names if available
#if busname.any():
#    b_name = []
#    for b in busname:
#        b_name.append(b[0][0])
#    b_name = np.array(b_name)
#    mpc_buses['name'] = b_name
#
#lineidx = ['l{}'.format(i) for i in range(0,len(branch_df.index))]
#mpc_lines = {
#        'idx': lineidx,
#        'node_i': branch_df['fbus'],
#        'node_j': branch_df['tbus'],
#        'maxflow': branch_df['rateA'],
#        'b_other': branch_df['b'],
#        'r': branch_df['r'],
#        'x': branch_df['x']
#        }
#
#mpc_lines = _mpc_data_pu_to_real(mpc_lines, mpc_buses['baseKV'][0], baseMVA)
#
#contingency = np.ones(len(mpc_lines['idx']))
#mpc_lines['contingency'] = contingency.astype(bool)
#
#ng = len(gen_df.index)
#genidx = ['g{}'.format(i) for i in range(ng)]
#mpc_generators = {
#            'idx': genidx,
#            'g_max': gen_df['Pmax'],
#            'g_max_Q': gen_df['Qmax'],
#            'node': gen_df['bus'],
##            'apf': gen_df['apf'],
#            'mc': gencost_df['c1'][list(range(0,ng))],
#            'mc_Q': np.zeros(ng)
#            }
#
#if len(gencost_df.index) == 2*ng:
#    mpc_generators['mc_Q'] = gencost_df['c1'][list(range(ng,2*ng))].tolist
#
#lines = pd.DataFrame(mpc_lines).set_index('idx')
#nodes = pd.DataFrame(mpc_buses).set_index('idx')
#plants = pd.DataFrame(mpc_generators).set_index('idx')
