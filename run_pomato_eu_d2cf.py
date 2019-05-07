import sys
from pathlib import Path
code_py = Path.cwd().joinpath("code_py")
sys.path.append(str(code_py))

import matplotlib.pyplot as plt

from market_tool import MarketTool

import numpy as np
import pandas as pd
import tools
import datetime

mato = MarketTool(options_file="profiles/eu_cbco.json")
mato.load_data('data_input\dataset_eu_d2cf.xlsx')

n = mato.data.nodes
p = mato.data.plants
l = mato.grid.lines
dc = mato.data.dclines
f = mato.data.fuel

d = mato.data.demand_el
o = mato.options
a = mato.data.availability

mato.create_grid_representation()
gr = mato.grid_representation["cbco"]

#8*3600 + 53*60

#len(mato.grid.lines) + len(mato.grid.lines)*len(mato.grid.lines[mato.grid.lines.contingency])
#lodf_graph = {}
#for i in range(1, 20):
#    mato.options["grid"]["senstitivity"] = 0.01 + 0.005*i
#    mato.create_grid_representation()
#    lodf_graph[0.01 + 0.005*i] = len(mato.grid_representation["cbco"])
#
#df = pd.DataFrame.from_dict(lodf_graph, orient='index')
#df.to_clipboard()
#
#for i in df.index:
#    print(f"({i*100}, {df.loc[i, 0]})")

#mato.data.visulize_inputdata(mato.data.results.result_folder)

#mato.data.process_results(mato.wdir.joinpath("data_temp\\julia_files\\results\\1402_1953"),
#                          mato.options, grid=mato.grid)

#t = mato.data.results.net_position()

#mato.init_market_model()
#mato.run_market_model()
#t = mato.data.results.check_courtailment()
#from bokeh_plot_interface import BokehPlot
#mato.init_bokeh_plot(name="FBMC")
#print("OK")
# ##  bokeh serve --show .\code_py\bokeh_plot.py --args=data_temp/bokeh_files


#%%
#plt.close("all")
###mato.grid.lines.cb = False
##mato.grid.lines.cb[mato.grid.lines.index == "l0"] = True
###
#mato.grid.lines.loc["l2340", "cb"] = False
#mato.grid.lines.loc["l2340", "cnec"] = False
#
#mato.grid.lines.loc["l1693", "cb"] = False
#mato.grid.lines.loc["l1693", "cnec"] = False
#
#from fbmc_module import FBMCModule
#fbmc = FBMCModule(mato.wdir, mato.grid, mato.data.results.INJ, mato.data.frm_fav,
#                  "data_input\gsk\GSKs_1402_1953_1955\GSK_results_selected_technologies_by_zone_2019_1402_1953_including_G_equal_0.csv")
#######
###
###gsk_sink = {key: 0 for key in mato.data.zones.index}
###for key in ["NL"]:
###    gsk_sink[key] = 1
#
#for timestep in mato.data.results.INJ.t.unique():
##for timestep in ["t0001", "t0002", "t0003"]:
##for timestep in ["t0001"]:
#    for gsk_strat in ["flat", "G", "g_max", "g_max_G_flat", "g_max_G"]:
##    for gsk_strat in ["flat"]:
#        fbmc.update_plot_setup(timestep, gsk_strat)
#        fbmc.plot_fbmc(["DE", "FR"], ["DE", "NL"])
##        plot = fbmc.plot_fbmc(["DE"], ["FR"], gsk_sink)
##
#fbmc.save_all_domain_plots(mato.data.results.result_folder)
#t = fbmc.save_all_domain_info(mato.data.results.result_folder, name_suffix="_including_G_equal_0")

#fbmc.gsk_strat = "flat"
#gsk = fbmc.load_gsk()

#self = fbmc
#domain_info = pd.concat([self.fbmc_plots[plot].domain_data for plot in self.fbmc_plots])
## oder the columns
#columns = ["timestep", "gsk_strategy", "cb", "co"]
#columns.extend(list(self.nodes.zone.unique()))
#columns.extend(["ram", "in_domain"])
#domain_info = domain_info[columns]
#mask = domain_info[['cb','co']].isin({'cb': domain_info[domain_info.in_domain].cb.unique(), 'co': domain_info[domain_info.in_domain].co.unique()}).all(axis=1)
#
#t = domain_info[mask]
#countries = ["DE", "FR", "BE", "NL", "LU"]
#tmp = domain_info[mask][["timestep", "gsk_strategy", "cb", "co", "ram", "in_domain"]].copy()
#for ref_c in countries:
#    other_c = list(countries)
#    other_c.remove(ref_c)
#    for c in other_c:
#        tmp[ref_c + "-" + c] = domain_info[mask][ref_c] - domain_info[mask][c]
#
#tmp.to_csv(mato.wdir.joinpath("domain_info_full_all.csv"))

#



#%%

#tech = mato.data.tech
#has_conv = []
#conv_fuel = ["lignite", "hard coal", "gas", "oil", "waste", "waste", "mixed fossil fuels", "biomass"]
#for node in n.index:
#    condition_fuel = p.index[(p.node == node)&(p.fuel.isin(conv_fuel))].empty
#    condition_psp = p.index[(p.node == node)&(p.tech == "psp")].empty
#    if condition_fuel and condition_psp:
#        has_conv.append(False)
#    else:
#        has_conv.append(True)
#n["has_conv"] = has_conv
#
#n.to_csv(str(mato.wdir.joinpath('nodes_with_conv.csv')), index_label='index')
#n[n.has_conv]

