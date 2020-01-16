import sys
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import datetime

pomato_path = Path.cwd().parent.joinpath("pomato")
sys.path.append(str(pomato_path))
from pomato import POMATO

mato = POMATO(wdir=Path.cwd().parent, options_file="profiles/ieee118.json")
mato.load_data('data_input/pglib_opf_case118_ieee.m')

n = mato.data.nodes
l = mato.grid.lines
dc = mato.data.dclines
d = mato.data.demand_el
d_h = mato.data.demand_h
o = mato.options
a = mato.data.availability
z = mato.data.zones
i = mato.data.inflows
ntc = mato.data.ntc
nex = mato.data.net_export
inflows = mato.data.inflows
net_position = mato.data.net_position

mato.create_grid_representation()
gr = mato.grid_representation["cbco"]

# %%

# result_path = Path("C:/Users/riw/tubCloud/Uni/Market_Tool/pomato/data_temp/julia_files/results/601_1512")
# mato.data.process_results(result_path, grid=mato.grid)

# %%

mato.init_market_model()
mato.run_market_model()

# %%

# mato.data.lines.maxflow *= 0.7
# mato.create_grid_representation()
# mato.update_market_model_data()
# mato.run_market_model()

# %%

df = mato.data.results.n_1_flow()
df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)
df3, df4 = mato.data.results.overloaded_lines_n_0()

# mato.init_bokeh_plot(name="IEEE")
