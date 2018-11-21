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

mato = MarketTool(opt_file="profiles/d2cf.json", model_horizon=range(0,168))

mato.load_data('data_input/dataset_eu_d2cf.xlsx')

#mato.data.visulize_inputdata(mato.wdir)

#mato.data.availability *= 0.6
#mato.data.lines.cb = True
#tmp = mato.data.lines.cb[mato.data.lines.cb]
#any(mato.data.nodes.slack)

mato.grid.build_grid_model(mato.data.nodes, mato.data.lines)

#mato.data.process_results(mato.wdir.joinpath("data_temp\\julia_files\\results\\1611_1501"), mato.opt_setup)

#mato.data.results.grid = mato.grid

#mato.data.results.grid = mato.grid
#mato.data.results.n_1_flow(["t0001", "t0002"], list(mato.data.lines.index.values), list(mato.data.lines.index.values))
#mato.data.results.n_0_flow(["t0001", "t0002"])

# TODO Make new julia type Grid that has the grid information
# Export the relevant data to csv instead of json
# change the export timeseries from beeing included in the demand parameter
# to an explicit parameter which is either on the nodela or, better, on the zonal energy balance
#t = mato.data.lines.cb

#mato.data.lines.maxflow *= 1


#mato.create_grid_representation()
#t = mato.grid_representation

#mato.data.lines.cb[e.index] = True
#mato.data.lines.cb[mato.data.lines.cb]

mato.init_market_model()
mato.run_market_model()

#e,f = mato.data.results.overloaded_lines_n_1()
#g,h = mato.data.results.overloaded_lines_n_0()


#t = mato.data.results.INFEAS_EL_N
#mato.data.results.default_plots()

#mato.data.process_results(mato.wdir.joinpath("julia-files\\results\\210_1249"), mato.opt_setup)
#mato.data.results.default_plots()

#mato.data.results.commercial_exchange("t0001")
#mato.data.results.net_position()

#ref_flow = mato.data.reference_flows[["l27"]]

#tmp = mato.data.results.INFEAS_LINES

#mato.data.net_export
#mato.data.net_position

#mato.init_bokeh_plot(name="stuffff")
#mato.bokeh_plot.start_server()
##mato.bokeh_plot.stop_server()
#
#
##overloaded_lines = mato.check_n_1_for_marketresult()
##add_cbco = []
##for t in overloaded_lines:
##    for nr in overloaded_lines[t]:
##        cbco = [overloaded_lines[t][nr]["Line"], overloaded_lines[t][nr]["Outage"]]
##        if not cbco in add_cbco:
##            add_cbco.append(cbco)
#
##print("OK")
##
#mato.data.nodes["gsk"] = 1
#from fbmc_module import FBMCModule
#fbmc = FBMCModule(Path.cwd(), mato.grid, injection, mato.data.frm_fav)
#######
##gsk_sink = {key: 0 for key in mato.data.zones.index}
##for key in ["NL"]:
##    gsk_sink[key] = 1
###
#for timestep in injection.t.unique():
##for timestep in ["t0001"]: #, "t0002", "t0003"]:
#    fbmc.create_Ab_from_jao_data(timestep=timestep)
#    gsk_strat = "jao"
#    plot = fbmc.plot_fbmc(["DE", "FR"], ["DE", "NL"])
#    #plot = fbmc.plot_fbmc(["DE"], ["FR"], gsk_sink)
#    for gsk_strat in ["flat", "G", "g_max", "g_max_G_flat"]:
#        fbmc.update_plot_setup(timestep, gsk_strat)
#        plot = fbmc.plot_fbmc(["DE", "FR"], ["DE", "NL"])
##        plot = fbmc.plot_fbmc(["DE"], ["FR"], gsk_sink)
#fbmc.set_xy_limits_forall_plots()
#for plot in fbmc.fbmc_plots:
#    print(plot)
#    fbmc.fbmc_plots[plot].plot_fbmc_domain(folder)


#t["t0001_jao"].plot_fbmc_domain()
#
#t["t0001_jao"].fig.show()
#n_1_flows = pd.DataFrame(index=mato.grid.lines.index)
#INJ = mato.market_model.results["INJ"]
#for t in np.unique(INJ.t.values):
#    mato.grid.nodes.net_injection = INJ.INJ[INJ.t == t].values
#    n_1_flows[t] = mato.grid.max_n_1_flow_per_line()
#
#flows_t = {t: [] for t in np.unique(INJ.t.values)}
#for i in mato.grid.lines.index:
#    n_1_ptdf = mato.grid.create_n_1_ptdf_outage(i)
#    for t in np.unique(INJ.t.values):
#        flows_t[t].append(np.dot(n_1_ptdf, INJ.INJ[INJ.t == t].values))
#
#n_1_flows = pd.DataFrame(index=mato.grid.lines.index)
#for t in np.unique(INJ.t.values):
#    n_1_flows_tmp = np.vstack(flows_t[t])
#    max_values = np.argmax(np.abs(n_1_flows_tmp), axis=0)
#    n_1_flows[t] = [n_1_flows_tmp[column, row] for row, column in enumerate(max_values)]

###
#fbmc.update_net_injections(INJ.INJ[INJ.t == "t0001"].values)
#fbmc.plot_fbmc(["DE"], ["FR"], gsk_sink=gsk_sink)
#

#n_1_flows = mato.grid.n_1_flows_timeseries(mato.market_model.results["INJ"])
#
#mato.grid.lines.type
#for line in n_1_flows["t0001"].index:
#    n_1_flows["t0001"][line]


#flows = mato.grid.n_0_flows_timeseries(injetion).transpose()
#stats = {}
#for line in mato.grid_representation["cbco"]:
#    calc_flows = flows[line]
#    ref_flows = list(mato.grid_representation["cbco"][line]["f_ref"].values())
#    mean_rel_diff = abs(np.subtract(calc_flows, ref_flows).sum())/(len(ref_flows)*mato.data.lines.maxflow[line])
#    max_rel_diff = abs(np.subtract(calc_flows, ref_flows).max())/mato.data.lines.maxflow[line]
#    stats[line] = {"mean relative difference cal-ref": mean_rel_diff,
#                   "max relative difference cal-ref": max_rel_diff,
#                   "mean rel inf": abs(infeasibility_lines.INFEAS_LINES[infeasibility_lines.cb == line].mean()/mato.data.lines.maxflow[line]),
#                    "max rel inf": abs(infeasibility_lines.INFEAS_LINES[infeasibility_lines.cb == line].max()/mato.data.lines.maxflow[line])}
#t = pd.DataFrame.from_dict(stats).transpose().round(3)




