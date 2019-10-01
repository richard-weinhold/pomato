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

mato = MarketTool(options_file="profiles/ramses.json")
mato.load_data('data_input\\ramses_2019.xlsx')

n = mato.data.nodes
p = mato.data.plants

l = mato.grid.lines
dc = mato.data.dclines
f = mato.data.fuel
d = mato.data.demand_el
o = mato.options
a = mato.data.availability
z = mato.data.zones
i = mato.data.inflows

mato.data.plants["tech"] = mato.data.plants["plant_type"]

mato.create_grid_representation()
gr = mato.grid_representation
n.loc[n.lat.isna(), "lat"], n.loc[n.lon.isna(), "lon"] = 51.787038, 9.551143

mato.init_market_model()
mato.run_market_model()

file = mato.wdir.joinpath("data_input/data_structure.xlsx")
xls = pd.ExcelFile(file)
structure = xls.parse("raw")
columns = [c for c in structure.columns if not "Unnamed:" in c]
data_structure = {}
for c in columns:
   att = "attribute"
   col_pos = structure.columns.get_loc(c)
   cols = list(structure.columns[col_pos:col_pos + 2])
   tmp = structure.loc[1:, cols].copy()
   data_structure[c] = {"attribute": {}, "optional attribute": {}}
   for (t,v) in zip(tmp[cols[0]].astype(str), tmp[cols[1]]):
       if not t == "nan":
           if t == "optional attribute":
               att = "optional attribute"
           else:
               data_structure[c][att][t] = v

# result_path = mato.wdir.joinpath("data_temp\\gms_files\\results\\ramses2019II")
# mato.data.process_results(result_path, mato.options, grid=mato.grid)

# %%

plt.ion()
# mato.data.demand_el["nNorway"]
# mato.data.plants.g_max[mato.data.plants.zone == "Norway"]
# mato.data.inflows.loc[:,"HydroReg_Norway"]
# mato.data.plants.loc["HydroReg_Norway"]
# mato.data.results.G[mato.data.results.G.p == "HydroReg_Norway"].mean()

mato.data.results.L_es[mato.data.results.L_es.p == "HydroReg_Norway"].plot()

# t = mato.data.results.INFEAS_EL_N_POS[mato.data.results.INFEAS_EL_N_POS.INFEAS_EL_N_POS > 0]
# t = mato.data.results.INFEAS_EL_N_NEG

# mato.data.results.G
# t = mato.data.results.check_infeasibilities()
p.g_max[p.plant_type=="WL"].sum()


mato.data.results.check_courtailment()


exchange = (mato.data.results.EX.groupby(["z", "zz"]).sum()/1E6).reset_index()
exchange = exchange[exchange.EX>0]

average_price = mato.data.results.price().groupby("z").mean()

gen = mato.data.results.G
gen_by_type = pd.merge(mato.data.plants[["plant_type", "zone"]], gen, left_index=True, right_on="p")
gen_by_type = (gen_by_type.groupby(["plant_type", "zone"]).sum()/1E6).reset_index()
gen_by_type = gen_by_type.pivot(index="zone", columns="plant_type", values="G")


gen_by_fuel = pd.merge(mato.data.plants[["fuel", "zone"]], gen, left_index=True, right_on="p")
gen_by_fuel = (gen_by_fuel.groupby(["fuel", "zone"]).sum()/1E6).reset_index()
gen_by_fuel = gen_by_fuel.pivot(index="zone", columns="fuel", values="G")

gen_baltic = gen_by_type.loc["EELVLT"]
gen_baltic[~gen_baltic.isna()]

t = p[p.zone == "EELVLT"]

gen_by_node = pd.merge(mato.data.plants[["node", "fuel"]], gen, left_index=True, right_on="p")
gen_by_node = (gen_by_node.groupby(["node", "fuel"]).sum()/1E6).reset_index()

gen_by_fuel.sum(axis=1).sum()

dem_ph = mato.data.results.D_ph
dem_ph.D_ph.sum()/1E6

condition = (p.node == "nPLCZSK")&(p.fuel == "Kul")
gen_by_node.loc[(gen_by_node.node == "nPLCZSK")&(gen_by_node.fuel == "Kul"), "G"]*1E6/(p.loc[condition, "g_max"].sum())

heat = mato.data.results.H
heat_by_type = pd.merge(mato.data.plants[["plant_type", "heatarea"]], heat, left_index=True, right_on="p")
heat_by_type = (heat_by_type.groupby(["plant_type", "heatarea"]).sum()/1E6).reset_index()
heat_by_type = heat_by_type.pivot(index="heatarea", columns="plant_type", values="H")

heat_by_type.sum(axis=1)

#df3 = mato.data.results.n_1_flow()
#df4 = mato.data.results.n_0_flow()

#t = mato.data.results.check_courtailment()

#from bokeh_plot_interface import BokehPlot
# mato.init_bokeh_plot(name="newnew")
# bokeh serve --show .\code_py\bokeh_plot.py --args=data_temp/bokeh_files


