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
                  model_horizon=range(0,10))
mato.load_data('data_input\dataset_de.xlsx')
#mato.load_data('data_input\pglib_casedata\pglib_opf_case118_ieee.m')
##mato.load_data('case118')

#mato.data.lines.maxflow = 0.2
mato.grid.build_grid_model(mato.data.nodes, mato.data.lines)
mato.create_grid_representation(precalc_filename="cbco_01_gurobide_pre_1601_0018") ### 400s
mato.create_grid_representation(cbco_option="full_cbco")

#t = mato.grid_representation
#mato.data.visulize_inputdata(mato.wdir)
#mato.data.lines.cb = True
#tmp = mato.data.lines.cb[mato.data.lines.cb]
#mato.data.process_results(mato.wdir.joinpath("data_temp\\julia_files\\results\\2201_1607"),
#                          mato.opt_setup, grid=mato.grid)

mato.init_market_model()
mato.run_market_model()
#t = mato.data.results.INFEAS_EL_N_POS

df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)
df3 = mato.data.results.n_1_flow()
#mato.data.results.default_plots()

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
#from scipy.spatial import ConvexHull
#from sklearn.decomposition import PCA
####
######%%
#from cbco_module import CBCOModule
#cbco_module = CBCOModule(mato.wdir, mato.grid)
#
#df = cbco_module.return_cbco()
#
#cbco_module.A

#cbco_module.cbco_algorithm(False)

#path = mato.wdir.joinpath("data_temp/julia_files/cbco_data/")
#A, b, info = cbco_module.create_Ab(preprocess=True)
#idx = pd.read_csv(path.joinpath("cbco_01_ieee118_2101_1648.csv"), delimiter=',').constraints.values
#vertices = np.array(cbco_module.reduce_Ab_convex_hull())
#D = A[vertices]/b[vertices]
#np.savetxt(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("A_ieee118.csv"), np.asarray(A), delimiter=",")
#np.savetxt(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("b_ieee118.csv"), np.asarray(b), delimiter=",")
##

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

