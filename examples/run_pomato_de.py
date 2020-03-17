"""DE Test Case."""
from pathlib import Path
import sys
pomato_path = Path.cwd().parent.joinpath("pomato")
sys.path.append(str(pomato_path))
from pomato import POMATO

import numpy as np
import pandas as pd

# %% Init POMATO with the options file and the dataset
mato = POMATO(wdir=Path.cwd().parent, options_file="profiles/de.json")
mato.load_data(r'data_input\dataset_de.xlsx')

# Acess the data pre-marketmodel.
nodes = mato.data.nodes
lines = mato.grid.lines
dclines = mato.data.dclines
demand = mato.data.demand_el
zones = mato.data.zones
plants = mato.data.plants
availability = mato.data.availability


# %% Potentially alter the input data, e.g. remove Phillipsburg since its offline from Jan 2020.
# p657 n261 1402 0 0.33 steam  uran  9.252722 8.435436  Kernkraftwerk Philippsburg 2 Philippsburg
# mato.data.plants = mato.data.plants.drop("p657")

# %% Run the Market Model, including (costbased) Re-Dispatch.
# The Market Result is determined based on the option file.
# The Redispatrch is done to N-0 per default.
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()

# There are two market results loaded into data.results.
# Specify redisp and market result for analysis
redisp_result = mato.data.results[next(r for r in list(mato.data.results) if "redispatch" in r)]
market_result = mato.data.results[next(r for r in list(mato.data.results) if "market_result" in r)]

# Check for Overloaded lines N-0, N-1 (should be non for N-0, but plenty for N-1)
df1, df2 = redisp_result.overloaded_lines_n_1()
df3, df4 = redisp_result.overloaded_lines_n_0()

# %% Examplary Result Analysis

# Generation comparison between Market Result and Redispatch.
gen = pd.merge(market_result.data.plants[["plant_type", "fuel", "tech", "g_max", "node"]],
               market_result.G, left_index=True, right_on="p")

# Redispatch Caluclation G_redispatch - G_market
gen = pd.merge(gen, redisp_result.G, on=["p", "t"], suffixes=("_market", "_redispatch"))
gen["delta"] = gen["G_redispatch"] - gen["G_market"]
gen["delta_abs"] = gen["delta"].abs()

# Generation Plots
gen[["fuel", "t", "G_market"]].groupby(["t", "fuel"]).sum().reset_index().pivot(index="t", columns="fuel", values="G_market").plot.area(stacked=True)
gen[["fuel", "t", "G_redispatch"]].groupby(["t", "fuel"]).sum().reset_index().pivot(index="t", columns="fuel", values="G_redispatch").plot.area(stacked=True)

# Redispatch Values
gen.delta.sum()
gen.delta_abs.sum()/20

# %% Bokeh PLot
mato.create_geo_plot(name="DE")
