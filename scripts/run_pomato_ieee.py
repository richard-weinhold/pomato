"""IEEE Test Case."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
pomato_path = Path.cwd().parent.joinpath("pomato")
sys.path.append(str(pomato_path))
from pomato import POMATO


# %%
mato = POMATO(wdir=Path.cwd().parent, options_file="profiles/ieee118.json")
# mato.load_data('data_input/DEV_pglib_casedata/pglib_opf_case4661_sdet.m')
mato.load_data('data_input/pglib_opf_case118_ieee.m')

nodes = mato.data.nodes
lines = mato.grid.lines
demand = mato.data.demand_el
zones = mato.data.zones
plants = mato.data.plants

# %%
# mato.options["optimization"]["type"] = "nodal"
# mato.create_grid_representation()
# mato.init_market_model()
# mato.run_market_model()

# result_folder = mato.market_model.result_folders[0]
# result = mato.data.results[result_folder.name]

# df1, df2 = result.overloaded_lines_n_1(sensitivity=0)
# df3, df4 = result.overloaded_lines_n_0()


# %%
mato.options["optimization"]["type"] = "cbco_nodal"
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()

result_folder = mato.market_model.result_folders[0]
result = mato.data.results[result_folder.name]
df1, df2 = result.overloaded_lines_n_1()
df3, df4 = result.overloaded_lines_n_0()


# %%
mato.create_geo_plot(name="IEEE")


