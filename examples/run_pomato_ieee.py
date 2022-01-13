"""IEEE Test Case."""
from pathlib import Path
import numpy as np
import pandas as pd
import pomato

# Init POMATO with the options file and the dataset
wdir = Path("/examples/") # Change to local copy of examples folder
mato = pomato.POMATO(wdir=wdir, options_file="profiles/ieee118.json")
mato.load_data('data_input/pglib_opf_case118_ieee.m')

# %% Access the data from the main pomato instance in the data object.
nodes = mato.data.nodes
lines = mato.grid.lines
demand = mato.data.demand_el
zones = mato.data.zones
plants = mato.data.plants

# %% Run uniform pricing
mato.options["type"] = "dispatch"
mato.create_grid_representation()
mato.run_market_model()

# Obtain the market result by name and its instance
dispatch_result_name = mato.market_model.result_folders[0].name
dispatch_result = mato.data.results[dispatch_result_name]

# Check Overloaded Lines for N-0 and N-1 contingency cases.
df1, df2 = dispatch_result.overloaded_lines_n_0()
df3, df4 = dispatch_result.overloaded_lines_n_1()
print("Number of overloaded lines (Dispatch): ", len(df1))
print("Number of overloaded lines N-1 (Dispatch): ", len(df3))

# %% Run nodal-prcing market clearing
mato.options["type"] = "nodal"
mato.create_grid_representation()
mato.run_market_model()

# Obtain the market result by name and its instance
nodal_result_name = mato.market_model.result_folders[0].name
nodal_result = mato.data.results[nodal_result_name]

# Check overloaded lines for N-0 and N-1 contingency cases.
df1, df2 = nodal_result.overloaded_lines_n_0()
df3, df4 = nodal_result.overloaded_lines_n_1()

print("Number of overloaded lines (nodal): ", len(df1))
print("Number of overloaded lines N-1 (nodal): ", len(df3))

# Create geoplot for nodal result, including price layer and saving it as .html
mato.visualization.create_geo_plot(nodal_result, 
                                   show_prices=True,
                                   filepath=mato.wdir.joinpath("geoplot_nodal.html"))

# %% Rerun the model as SCOPF
mato.options["type"] = "cbco_nodal"
mato.options["grid"]["redundancy_removal_option"] = "clarkson_base"
# Optionally set a line rating factors
# mato.options["grid"]["short_term_rating_factor"] = 1
# mato.options["grid"]["long_term_rating_factor"] = 0.75

# SCOPF requires to presolve the network with the RedundancyRemoval Algorithm
mato.create_grid_representation()

# Update the model data
mato.update_market_model_data()
mato.run_market_model()

# Check for overloaded lines (Should be none for N-0 and N-1)
scopf_result_name = mato.market_model.result_folders[0].name
scopf_result = mato.data.results[scopf_result_name]

df1, df2 = scopf_result.overloaded_lines_n_0()
df3, df4 = scopf_result.overloaded_lines_n_1()

print("Number of overloaded lines (SCOPF): ", len(df1))
print("Number of overloaded lines N-1 (SCOPF): ", len(df3))

# Create geoplot for scopf result, including price layer and saving it as .html
mato.visualization.create_geo_plot(scopf_result, 
                                   show_prices=True,
                                   filepath=mato.wdir.joinpath("geoplot_scopf.html"))

# Join all subprocesses. 
mato._join_julia_instances()
