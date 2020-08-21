"""DE Test Case."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# Init POMATO with the options file and the dataset
from pomato import POMATO

wdir = Path(__file__).parent
mato = POMATO(wdir=wdir, options_file="profiles/de.json")
mato.load_data('data_input/dataset_de.zip')

# Acess the data.
nodes = mato.data.nodes
lines = mato.grid.lines
dclines = mato.data.dclines
demand = mato.data.demand_el
zones = mato.data.zones
plants = mato.data.plants
availability = mato.data.availability
net_export = mato.data.net_export

mato.data.visualize_inputdata()
# Run the Market Model, including (costbased) Re-Dispatch.
# The Market Result is determined based on the option file.
# The Redispatrch is done to N-0 per default.

mato.run_market_model()
market_result = next(iter(mato.data.results.values()))

# Some result analysis
df1, df2 = market_result.overloaded_lines_n_0()
gen = pd.merge(mato.data.plants, market_result.G, left_index=True, right_on="p")
util = gen[["plant_type", "g_max", "G"]].groupby("plant_type").sum()
print(util.G / util.g_max)

# Show Geo Plot
mato.create_geo_plot()

# Change options, re-create grid representation and re-run
mato.data.results = {}
mato.options["optimization"]["redispatch"]["include"] = True
mato.options["optimization"]["redispatch"]["zones"] = ["DE"]
mato.options["optimization"]["redispatch"]["cost"] = 50

mato.create_grid_representation()
mato.run_market_model()

market_result, redisp_result = mato.data.return_results()

# Check for Overloaded lines N-0
n0_m, _ = market_result.overloaded_lines_n_0()
print("Number of N-0 Overloads: ", len(n0_m))

n0_r, _  = redisp_result.overloaded_lines_n_0()
print("Number of N-0 Overloads: ", len(n0_r))

# Check for infeasibilities in market / redispatch result.
infeas_redisp = redisp_result.infeasibility()
infeas_market = market_result.infeasibility()

# Generation comparison between Market Result and Redispatch.
gen = pd.merge(market_result.data.plants[["plant_type", "fuel", "tech", "g_max", "node"]],
               market_result.G, left_index=True, right_on="p")

# Redispatch Caluclation G_redispatch - G_market
gen = pd.merge(gen, redisp_result.G, on=["p", "t"], suffixes=("_market", "_redispatch"))
gen["delta"] = gen["G_redispatch"] - gen["G_market"]
gen["delta_abs"] = gen["delta"].abs()

# Redispatch Values
gen.delta.sum()
gen.delta_abs.sum()/24

# Create Geo PLot
mato.create_geo_plot(title="DE: Redispatch")
# mato.bokeh_plot.save_plot(mato.wdir.joinpath("bokeh_redispatch.html")

mato._join_julia_instances()