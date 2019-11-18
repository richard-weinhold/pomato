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
mato.load_data('data_input\DEV_pglib_casedata\pglib_opf_case118_ieee.m')

n = mato.data.nodes
p = mato.data.plants
l = mato.grid.lines
dc = mato.data.dclines
f = mato.data.fuel
d = mato.data.demand_el
d_h = mato.data.demand_h
o = mato.options
a = mato.data.availability
z = mato.data.zones
i = mato.data.inflows
tech = mato.data.tech
ntc = mato.data.ntc
nex = mato.data.net_export
inflows = mato.data.inflows
net_position = mato.data.net_position



mato.create_grid_representation()
gr = mato.grid_representation["cbco"]

mato.init_market_model()
# mato.run_market_model()

# df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)
# df3, df4 = mato.data.results.overloaded_lines_n_0()

# from bokeh_plot_interface import BokehPlot
# mato.init_bokeh_plot(name="IEEE")

# bokeh serve --show .\code_py\bokeh_plot.py --args=data_temp/bokeh_files

