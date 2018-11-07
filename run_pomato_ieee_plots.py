from market_tool import MarketTool
import numpy as np
import pandas as pd

mato = MarketTool(opt_file="opt_setup.json", model_horizon=range(200,204))
mato.opt_setup["opt"] = "cbco_nodal"
mato.load_data_from_file('DEV_riwdata/de_dataset.xlsx', autobuild_gridmodel=True)

#mato.load_matpower_case('case300', autobuild_gridmodel=True)
#nodes = mato.data.nodes
#mato.data.nodes["gsk"] = 1
#mato.data.nodes.zone = "1"
#mato.data.nodes.zone[mato.data.nodes.index.isin([x for x in range(220,230)])] = "2"
#mato.data.nodes.zone[mato.data.nodes.index.isin([x for x in range(20, 27)])] = "3"
#
#gsk_sink = {"1": 1}
#asd = mato.grid.plot_fbmc(["2"], ["3"], gsk_sink=gsk_sink)
##
#asd.show()

#mato.grid.plot_vertecies_of_inequalities(["1"], ["2"], gsk_sink=gsk_sink)

#mato.load_matpower_case('case300', autobuild_gridmodel=True)
#nodes = mato.data.nodes
#lines = mato.data.lines
#plants = mato.data.plants
#mato.data.nodes["gsk"] = 1
##
#gsk_sink = {"z3": 0.75, "z9":0.25}
#mato.grid.plot_fbmc(["z1"], ["z2"], gsk_sink=gsk_sink)
##mato.grid.plot_vertecies_of_inequalities(["1"], ["2"], gsk_sink=gsk_sink)
#print("OK")
