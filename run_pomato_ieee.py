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
#mato.data.lines.maxflow *= 0.2
mato.create_grid_representation()
gr = mato.grid_representation["cbco"]


#mato.data.process_results(mato.wdir.joinpath("data_temp\\julia_files\\results\\2102_1516"),
#                          mato.options, grid=mato.grid)

mato.init_market_model()
mato.run_market_model()

df1, df2 = mato.data.results.overloaded_lines_n_1(sensitivity=0)
df3 = mato.data.results.n_1_flow()
df4 = mato.data.results.n_0_flow()
#

from bokeh_plot_interface import BokehPlot
mato.init_bokeh_plot(name="IEEE")

# ##  bokeh serve --show .\code_py\bokeh_plot.py --args=data_temp/bokeh_files

#%%
#from scipy.spatial import ConvexHull
#from sklearn.decomposition import PCA
#####
#######%%
#from cbco_module import CBCOModule
#cbco_module = CBCOModule(mato.wdir, mato.grid)
#####df = cbco_module.julia_cbco_algorithm()
###cbco_module.cbco_index = [i for i in range(0, len(cbco_module.b))]
##idx = cbco_module.reduce_Ab_convex_hull()
#path = mato.wdir.joinpath("data_temp/julia_files/cbco_data/")
#A, b, info = cbco_module.create_Ab(preprocess=True)
#
#add_A = np.vstack(nodal_constraints)
#add_b = np.array(nodal_rhs).reshape(len(nodal_rhs), 1)
#
#Aplus = np.vstack([A, add_A])
#bplus = np.vstack([b, add_b])
#Iplus = [i for i in range(len(A), len(Aplus))]
##
#path = mato.wdir.joinpath("data_temp/julia_files/cbco_data")

#cbco = [cb for cb in range(0, len(Aplus) - len(add_A)) if cb in precalc_cbco]
#
#pd.DataFrame(columns=["constraints"], data=cbco).to_csv(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("cbco118_+_test.csv"))
##return_df = info.iloc[precalc_cbco]
#
#np.savetxt(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("A_118+.csv"), np.asarray(Aplus), delimiter=",")
#np.savetxt(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("b_118+.csv"), np.asarray(bplus), delimiter=",")
#np.savetxt(mato.wdir.joinpath("data_temp/julia_files/cbco_data").joinpath("I_118+.csv"), np.array(Iplus).astype(int), fmt='%i', delimiter=",")




