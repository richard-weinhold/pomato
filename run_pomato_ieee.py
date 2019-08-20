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

mato = MarketTool(options_file="profiles/ieee118.json")
mato.load_data('data_input\pglib_casedata\pglib_opf_case118_ieee.m')

n = mato.data.nodes
p = mato.data.plants
l = mato.grid.lines
dc = mato.data.dclines
f = mato.data.fuel
d = mato.data.demand_el
o = mato.options
a = mato.data.availability


# %%
mato.create_grid_representation()
gr = mato.grid_representation["cbco"]

# cbco_convexhull = gr
# # cbco_clarkson = gr

# idx_ch = list(cbco_convexhull.index)
# # idx_clarkson = list(cbco_clarkson.index)

# count = 0
# for i in idx_ch:
#     if i not in idx_clarkson:
#         count += 1
#         print(i)
# print(count)

#mato.grid.lines[mato.grid.lines.contingency]
#186 + 186*(186 - 9)
#mato.data.process_results(mato.wdir.joinpath("data_temp\\julia_files\\results\\304_1027"),
#                          mato.options, grid=mato.grid)

mato.init_market_model()
mato.run_market_model()
#

mato.data.results.G
df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)



# df3 = mato.data.results.n_1_flow()
# df4 = mato.data.results.n_0_flow()

#from bokeh_plot_interface import BokehPlot
#mato.init_bokeh_plot(name="IEEE")

# bokeh serve --show .\code_py\bokeh_plot.py --args=data_temp/bokeh_files

