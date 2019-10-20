# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:05:16 2019

@author: riw
"""

import sys
from pathlib import Path
code_py = Path.cwd().joinpath("code_py")
sys.path.append(str(code_py))

from market_tool import MarketTool
from bokeh_plot_interface import BokehPlot

import numpy as np
import pandas as pd
import tools
import datetime

mato = MarketTool(options_file="profiles/de.json")
mato.load_data('data_input\dataset_de_oct.xlsx')

n = mato.data.nodes
p = mato.data.plants
l = mato.grid.lines
dc = mato.data.dclines
f = mato.data.fuel
z = mato.data.zones

d = mato.data.demand_el
o = mato.options
a = mato.data.availability
mato.create_grid_representation()
gr = mato.grid_representation["cbco"]
slack = mato.grid_representation["slack_zones"]

mato.init_market_model()
mato.run_market_model()
df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=5e-2)
df3, df4 = mato.data.results.overloaded_lines_n_0()

mato.init_bokeh_plot(name="DE")

# ##  bokeh serve --show .\code_py\bokeh_plot.py --args=data_temp/bokeh_files
#%%

