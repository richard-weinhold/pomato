"""IEEE Test Case."""

from pathlib import Path
from pomato import POMATO


# %% Init POMATO with the options file and the dataset
mato = POMATO(wdir=Path.cwd(), options_file="profiles/ieee118.json")
mato.load_data('data_input/pglib_opf_case118_ieee.m')

# %% Rerun the model as SCOPF
mato.options["optimization"]["type"] = "cbco_nodal"
# mato.cbco_module.options["grid"]["senstitivity"] = 0
mato.cbco_module.options["grid"]["cbco_option"] = "clarkson_base"
# Requires to presolve the network with the RedundancyRemvoal Algorith
# if no previous set of essential indices is privided in the option file
mato.create_grid_representation()
# Update the model data
mato.update_market_model_data()
mato.run_market_model()

# Check for overloaded lines (Should be none for N-0 and N-1)
result_folder = mato.market_model.result_folders[0]
result = mato.data.results[result_folder.name]

df1, df2 = result.overloaded_lines_n_1(sensitivity=0)
df3, df4 = result.overloaded_lines_n_0()

# %% Show a Geo Plot of the market- and the resulting power flows.
mato.create_geo_plot(name="IEEE")

mato._join_julia_instances()
