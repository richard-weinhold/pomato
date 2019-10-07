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
mato.load_data('data_input\\ramses_2019_small.xlsx')
# mato.load_data('data_input\\ramses_2019.xlsx')

n = mato.data.nodes
p = mato.data.plants

l = mato.grid.lines
dc = mato.data.dclines
f = mato.data.fuel
d = mato.data.demand_el
d_h = mato.data.demand_h
o = mato.options
a = mato.data.availability
z = mato.data.zones
i = mato.data.inflows
tech = mato.data.tech
ntc = mato.data.ntc
nex = mato.data.net_export
inflows = mato.data.inflows




nex.node.unique()
condition_hs = (mato.data.plants.h_max >0)&(mato.data.plants.storage_capacity.notna())
mato.data.plants.loc[condition_hs, "mc_heat"] = mato.data.plants.loc[condition_hs, "variable_om"]
mato.data.net_export.loc[:, "net_export"] *= -1

# 3.742355039e+09
# 3.734239375e+09
# 3.73423938e+09

# 3.730199018e+09
# 6.920834232e+09
# 6.683816890693237e9
# 9.316599534339179e8
# 1.2777826212290524e10

# mato.data.availability = mato.data.availability.round(3)
# mato.data.net_export = mato.data.net_export.round(3)
# mato.data.demand_el = mato.data.demand_el.round(3)
# mato.data.demand_h = mato.data.demand_h.round(3)
# mato.data.inflows = mato.data.inflows.round(3)
# mato.data.plants = mato.data.plants[(mato.data.plants.g_max > 1)|(mato.data.plants.h_max > 1)]
# mato.data.plants.loc[:, "g_max"] = mato.data.plants.loc[:, "g_max"].round(3)
# mato.data.plants.loc[:, "h_max"] = mato.data.plants.loc[:, "h_max"].round(3)

# mato.data.plants.loc[:, "mc_el"] = mato.data.plants.loc[:, "mc_el"].round(3)
# mato.data.plants.loc[:, "mc_heat"] = mato.data.plants.loc[:, "mc_heat"].round(3)

# mato.data.net_export.loc[:, "net_export"] =  mato.data.net_export.loc[:, "net_export"]*0.9

# mato.data.plants.loc[:, "g_max"].min()

# mato.data.plants[((mato.data.plants.g_max == 0)&(mato.data.plants.h_max == 0))]
# condition_el = mato.data.plants.g_max>0
# condition_he = (mato.data.plants.g_max == 0)&(mato.data.plants.h_max>0)
# mato.data.plants.loc[condition_el, "mc_el"] = mato.data.plants.mc_el + mato.data.plants.mc_heat
# mato.data.plants.loc[condition_el, "mc_heat"] = 0
# mato.data.plants.loc[condition_he, "mc_heat"] = mato.data.plants.mc_el + mato.data.plants.mc_heat

mato.create_grid_representation()
gr = mato.grid_representation

n.loc[n.lat.isna(), "lat"], n.loc[n.lon.isna(), "lon"] = 51.787038, 9.551143

mato.init_market_model()
# mato.update_market_model_data()
mato.run_market_model()

t = mato.data.results.commercial_exchange("t0001")

# result_path = mato.wdir.joinpath("data_temp\\gms_files\\results\\0310_1236")
# mato.data.process_results(result_path, mato.options, grid=mato.grid)

# %%
char_dict = {"ø": "o", "Ø": "O",
             "æ": "ae", "Æ": "AE",
             "å": "a", "Å": "A"}

ramses_result_path = Path("C:/Users/riw/tubCloud/Uni/ENS/ramses_data/2019-07-05_DataR_Master_TVAR2014_2015_2040_YS1_S1_O2_C1506980/RR_2019-07-05_DataR_Master_TVAR2014_2015_2040_YS1_S1_O2.xlsx")
ramses_xls = pd.ExcelFile(ramses_result_path)

results_tables = ramses_xls.parse("2019", usecols=[0], index_col=0)

price_pos = results_tables.index.get_loc("Electricity Price")
ramses_price = ramses_xls.parse("2019", index_col=0, skiprows=price_pos+1, nrows=2).dropna(axis=1)

gen_pos = results_tables.index.get_loc("PlantName")
ramses_generation_raw = ramses_xls.parse("2019", index_col=0, skiprows=gen_pos+1).dropna(axis=1).reset_index()
ramses_generation_raw.replace(char_dict, regex=True, inplace=True)

ramses_generation_fuel_full = ramses_generation_raw.pivot_table(index="ElArea", columns="FuelMix", values="ElProd_TWh", aggfunc ="sum")
ramses_generation_fuel_full.index = ramses_generation_fuel_full.index + "_ramses"

ramses_generation_type_full = ramses_generation_raw.pivot_table(index="ElArea", columns="PlantType", values="ElProd_TWh", aggfunc ="sum")
ramses_generation_type_full.index = ramses_generation_type_full.index + "_ramses"

ramses_heat_full = ramses_generation_raw.pivot_table(index="ElArea", columns="FuelMix", values="HeatProd_TWh", aggfunc ="sum")
ramses_heat_full.index = ramses_heat_full.index + "_ramses"

plt.ion()

mato.data.results.L_es[mato.data.results.L_es.p == "HydroReg_Norway"].plot()
# t = mato.data.results.check_infeasibilities()
# mato.data.results.check_courtailment()

exchange = (mato.data.results.EX.groupby(["z", "zz"]).sum()/1E6).reset_index()
exchange = exchange[exchange.EX>0]

average_price = mato.data.results.price().groupby("z").mean()
average_price["ramses"] = ramses_price.T.iloc[:, 0]
average_price["absolute"] = average_price["marginal"] - average_price["ramses"]
average_price["relative"] = (average_price["marginal"] - average_price["ramses"])*100/average_price["marginal"]
print(average_price)

gen = pd.merge(mato.data.plants[["plant_type", "zone", "fuel", "node"]], mato.data.results.G, left_index=True, right_on="p")
gen_by_type = gen.pivot_table(index="zone", columns="plant_type", values="G", aggfunc="sum")/1E6
gen_by_fuel = gen.pivot_table(index="zone", columns="fuel", values="G", aggfunc="sum")/1E6
gen_by_node = gen.pivot_table(index="zone", columns="node", values="G", aggfunc="sum")/1E6

dem_ph = mato.data.results.D_ph
dem_ph.D_ph.sum()/1E6

heat = pd.merge(mato.data.plants[["plant_type", "heatarea", "fuel", "zone"]], mato.data.results.H, left_index=True, right_on="p")
heat_by_type = heat.pivot_table(index="zone", columns="plant_type", values="H", aggfunc="sum")/1E6
heat_by_fuel = heat.pivot_table(index="zone", columns="fuel", values="H", aggfunc="sum")/1E6

### Plots
# %%
ramses_generation_fuel = ramses_generation_fuel_full.loc[:, ramses_generation_fuel_full.sum() > 15]
ramses_generation_type = ramses_generation_type_full.loc[:, ramses_generation_type_full.sum() > 15]

ramses_generation_fuel_dk = ramses_generation_fuel_full.loc[["DK1_ramses", "DK2_ramses"],
                                                            ramses_generation_fuel_full.loc[["DK1_ramses", "DK2_ramses"]].sum() > 0.5]

ramses_generation_type_dk = ramses_generation_type_full.loc[["DK1_ramses", "DK2_ramses"],
                                                            ramses_generation_type_full.loc[["DK1_ramses", "DK2_ramses"]].sum() > 0]

ramses_heat = ramses_heat_full.loc[["DK1_ramses", "DK2_ramses"], ramses_heat_full.sum()>0]

pomato_generation_fuel = gen_by_fuel.loc[:, gen_by_fuel.sum()>15]
pomato_generation_type = gen_by_type.loc[:, gen_by_type.sum()>15]

pomato_generation_fuel_dk = gen_by_fuel.loc[["DK1", "DK2"], gen_by_fuel.loc[["DK1", "DK2"]].sum()>0.5]
pomato_generation_type_dk = gen_by_type.loc[["DK1", "DK2"]]

pomato_heat = heat_by_fuel.loc[["DK1", "DK2"], heat_by_fuel.loc[["DK1", "DK2"]].sum()>0.1]

print("heat: RAMSES VS POMATO")
print(ramses_heat_full.sum().sum(), heat_by_fuel.loc[["DK1", "DK2"]].sum().sum())

ramses_colors_path = Path("C:/Users/riw/tubCloud/Uni/ENS/data_projects/colors_ramses.xlsx")
ramses_colors = pd.ExcelFile(ramses_colors_path)
fuel_colors = ramses_colors.parse("fuel", index_col=2)["color"].to_dict()
fuel_group = ramses_colors.parse("fuel", index_col=0)["group"].to_dict()
type_colors = ramses_colors.parse("type", index_col=2)["color"].to_dict()
type_group = ramses_colors.parse("type", index_col=0)["group"].to_dict()

ramses_generation_fuel = ramses_generation_fuel.groupby(fuel_group, axis=1).sum()
pomato_generation_fuel = pomato_generation_fuel.groupby(fuel_group, axis=1).sum()
ramses_generation_fuel.plot.bar(stacked=True, color=[fuel_colors[fuel] for fuel in ramses_generation_fuel.columns])
pomato_generation_fuel.plot.bar(stacked=True, color=[fuel_colors[fuel] for fuel in pomato_generation_fuel.columns])

ramses_generation_type = ramses_generation_type.groupby(type_group, axis=1).sum()
pomato_generation_type = pomato_generation_type.groupby(type_group, axis=1).sum()
ramses_generation_type.plot.bar(stacked=True, color=[type_colors[fuel] for fuel in ramses_generation_type.columns])
pomato_generation_type.plot.bar(stacked=True, color=[type_colors[fuel] for fuel in pomato_generation_type.columns])

ramses_generation_fuel_dk = ramses_generation_fuel_dk.groupby(fuel_group, axis=1).sum()
pomato_generation_fuel_dk = pomato_generation_fuel_dk.groupby(fuel_group, axis=1).sum()
ramses_generation_fuel_dk.plot.bar(stacked=True, color=[fuel_colors[fuel] for fuel in ramses_generation_fuel_dk.columns])
pomato_generation_fuel_dk.plot.bar(stacked=True, color=[fuel_colors[fuel] for fuel in pomato_generation_fuel_dk.columns])

ramses_generation_type_dk = ramses_generation_type_dk.groupby(type_group, axis=1).sum()
pomato_generation_type_dk = pomato_generation_type_dk.groupby(type_group, axis=1).sum()
ramses_generation_type_dk.plot.bar(stacked=True, color=[type_colors[fuel] for fuel in ramses_generation_type_dk.columns])
pomato_generation_type_dk.plot.bar(stacked=True, color=[type_colors[fuel] for fuel in pomato_generation_type_dk.columns])

ramses_heat = ramses_heat.groupby(fuel_group, axis=1).sum()
pomato_heat = pomato_heat.groupby(fuel_group, axis=1).sum()
ramses_heat.plot.bar(stacked=True, color=[fuel_colors[fuel] for fuel in ramses_heat.columns])
pomato_heat.plot.bar(stacked=True, color=[fuel_colors[fuel] for fuel in pomato_heat.columns])

#df3 = mato.data.results.n_1_flow()
#df4 = mato.data.results.n_0_flow()

#t = mato.data.results.check_courtailment()
# %%

# from bokeh_plot_interface import BokehPlot
# mato.init_bokeh_plot(name="ramsesnew")
# bokeh serve --show .\code_py\bokeh_plot.py --args=data_temp/bokeh_files

# %%



