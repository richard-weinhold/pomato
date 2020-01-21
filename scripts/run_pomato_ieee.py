import sys
from pathlib import Path

import numpy as np
import pandas as pd

pomato_path = Path.cwd().parent.joinpath("pomato")
sys.path.append(str(pomato_path))
from pomato import POMATO


# %%%
mato = POMATO(wdir=Path.cwd().parent, options_file="profiles/ieee118.json")
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

# df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)
# df3, df4 = mato.data.results.overloaded_lines_n_0()

# %%

mato.options["optimization"]["type"] = "cbco_nodal"
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()
df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)
df3, df4 = mato.data.results.overloaded_lines_n_0()

# %%

mato.init_bokeh_plot(name="IEEE")

# %%

# result_path = Path("result_path")
# mato.data.process_results(result_path, grid=mato.grid)
