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

## cbco_py_2108_1611 without base


# %%
mato.create_grid_representation()
gr = mato.grid_representation["cbco"]

# # cbco_convexhull = gr
# # # cbco_clarkson = gr

# # idx_ch = list(cbco_convexhull.index)
# # # idx_clarkson = list(cbco_clarkson.index)

# # count = 0
# # for i in idx_ch:
# #     if i not in idx_clarkson:
# #         count += 1
# #         print(i)
# # print(count)

# #mato.grid.lines[mato.grid.lines.contingency]
# #186 + 186*(186 - 9)
# #mato.data.process_results(mato.wdir.joinpath("data_temp\\julia_files\\results\\304_1027"),
# #                          mato.options, grid=mato.grid)

mato.init_market_model()
mato.run_market_model()


mato.data.results.INFEAS_EL_N_NEG.max()
mato.data.results.INFEAS_EL_N_POS.max()

df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)
df3, df4 = mato.data.results.overloaded_lines_n_0()
# mato.data.result_attributes

# %%
# from bokeh_webapp import create_plot

# mato.data
# timestep = "t0001"
# # inj = mato.data.results.INJ
# # inj = inj.INJ[inj.t == timestep].values

# # flow_n_0 = mato.data.results.n_0_flow()
# # flow_n_1 = mato.data.results.n_1_flow()
# # flow_n_0 = flow_n_0[timestep]

# # flow_n_1 = flow_n_1.drop("co", axis=1)
# # flow_n_1[timestep] = flow_n_1[timestep].abs()
# # flow_n_1 = flow_n_1.groupby("cb").max().reset_index()
# # flow_n_1 = flow_n_1.set_index("cb").reindex(mato.data.lines.index)

# # f_dc = mato.data.results.F_DC

# # mato.data.nodes.lat.max()

# inj = np.array([0 for n in mato.data.nodes.index])

# flow_n_0 = pd.DataFrame(index=mato.data.lines.index)
# flow_n_0["t0001"] = 0
# flow_n_0 = flow_n_0[timestep]

# flow_n_1 = pd.DataFrame(index=mato.data.lines.index)
# flow_n_1["t0001"] = 0
# flow_n_1 = flow_n_1[timestep]

# f_dc = pd.DataFrame()

# fig = create_plot(mato.data.lines, mato.data.nodes, mato.data.dclines, inj, flow_n_0, flow_n_1, f_dc)

# t, tt = components(fig)

# dir(CDN.render())


# from bokeh.io import show
# from bokeh.embed import components
# from bokeh.resources import CDN

# show(fig)
# # if mato.data.results:
    # print("asd")

#%%
# julia_obj = 248206.36926550948

# julia_obj - gms_obj

# # mato.data.results.G
# df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)


# df3 = mato.data.results.n_1_flow()
# df4 = mato.data.results.n_0_flow()

#from bokeh_plot_interface import BokehPlot
#mato.init_bokeh_plot(name="IEEE")

# bokeh serve --show .\code_py\bokeh_plot.py --args=data_temp/bokeh_files

