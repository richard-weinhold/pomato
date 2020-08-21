"""IEEE Test Case."""
#
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Init POMATO with the options file and the dataset
import pomato

wdir = Path(__file__).parent
mato = pomato.POMATO(wdir=wdir, options_file="profiles/ieee118.json")
mato.load_data('data_input/pglib_opf_case118_ieee.m')

# %% Access the data from the main pomato isntance in the data object.
nodes = mato.data.nodes
lines = mato.grid.lines
demand = mato.data.demand_el
zones = mato.data.zones
plants = mato.data.plants

# %% Run Uniform Prcing Market Clearing
mato.options["optimization"]["type"] = "dispatch"
mato.create_grid_representation()
mato.run_market_model()

result_folder = mato.market_model.result_folders[0]
result = mato.data.results[result_folder.name]

# Check Overloaded Lines for N-0 and N-1 contingency cases.
df1, df2 = result.overloaded_lines_n_0()
df3, df4 = result.overloaded_lines_n_1()

print("Number of overloaded lines (Dispatch): ", len(df1))
print("Number of overloaded lines N-1 (Dispatch): ", len(df3))

# mato.create_geo_plot()

# %% Run N-0 Market Clearing
mato.options["optimization"]["type"] = "nodal"
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()

result_folder = mato.market_model.result_folders[0]
result = mato.data.results[result_folder.name]

# Check Overloaded Lines for N-0 and N-1 contingency cases.
df1, df2 = result.overloaded_lines_n_0()
df3, df4 = result.overloaded_lines_n_1()

print("Number of overloaded lines (Nodal): ", len(df1))
print("Number of overloaded lines N-1 (Nodal): ", len(df3))


# %% Rerun the model as SCOPF
mato.options["optimization"]["type"] = "cbco_nodal"
mato.options["grid"]["cbco_option"] = "clarkson_base"

# Requires to presolve the network with the RedundancyRemvoal Algorith
# if no previous set of essential indices is privided in the option file
mato.create_grid_representation()

# # Update the model data
mato.update_market_model_data()
mato.run_market_model()

# # Check for overloaded lines (Should be none for N-0 and N-1)
result_folder = mato.market_model.result_folders[0]
result = mato.data.results[result_folder.name]

df1, df2 = result.overloaded_lines_n_0()
df3, df4 = result.overloaded_lines_n_1()

print("Number of overloaded lines (SCOPF): ", len(df1))
print("Number of overloaded lines N-1 (SCOPF): ", len(df3))

# %% Show a Geo Plot of the market- and the resulting power flows.
mato.create_geo_plot(title="IEEE")
mato._join_julia_instances()
