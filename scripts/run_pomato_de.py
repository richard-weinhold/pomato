"""DE Test Case."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pomato_path = Path.cwd().parent.joinpath("pomato")
sys.path.append(str(pomato_path))
from pomato import POMATO

mato = POMATO(wdir=Path.cwd().parent, options_file="profiles/de.json")
mato.load_data(r'data_input\dataset_de_short_gridkit.xlsx')
n = mato.data.nodes
p = mato.data.plants
l = mato.grid.lines
dc = mato.data.dclines
f = mato.data.fuel
z = mato.data.zones
d = mato.data.demand_el
o = mato.options
a = mato.data.availability

mato.grid.nodes[mato.grid.nodes.slack]

# %%
# p657 n261 1402 0 0.33 steam  uran  9.252722 8.435436  Kernkraftwerk Philippsburg 2 Philippsburg
# mato.data.plants = mato.data.plants.drop("p657")
# %%

mato.options["optimization"]["type"] = "ntc"
mato.data.results = {}
mato.create_grid_representation()

mato.grid_representation["redispatch_grid"].loc[:, "zone"] = "DE"
mato.update_market_model_data()
mato.run_market_model()

df1, df2 = mato.data.results.overloaded_lines_n_1()
df3, df4 = mato.data.results.overloaded_lines_n_0()

# %% Some Result analysis

t = p[p.node == "n3966"]

redisp_result = mato.data.results[next(r for r in list(mato.data.results) if "redispatch" in r)]
market_result = mato.data.results[next(r for r in list(mato.data.results) if "market_result" in r)]

gen = pd.merge(market_result.data.plants[["plant_type", "fuel", "tech", "g_max", "node"]],
                market_result.G, left_index=True, right_on="p")

# Redispatch Caluclation
gen = pd.merge(gen, redisp_result.G, on=["p", "t"], suffixes=("_market", "_redispatch"))
gen["delta"] = gen["G_redispatch"] - gen["G_market"]
gen["delta_abs"] = gen["delta"].abs()

# Generation PLots
gen[["fuel", "t", "G_market"]].groupby(["t", "fuel"]).sum().reset_index().pivot(index="t", columns="fuel", values="G_market").plot.area(stacked=True)
gen[["fuel", "t", "G_redispatch"]].groupby(["t", "fuel"]).sum().reset_index().pivot(index="t", columns="fuel", values="G_redispatch").plot.area(stacked=True)

# Redispatch Values
gen.delta.sum()
gen.delta_abs.sum()/20


# %% Bokeh PLot
mato.create_geo_plot(name="DE")
# df1, df2 = redisp_result.overloaded_lines_n_1()

# df1["# of overloads"].sum()
