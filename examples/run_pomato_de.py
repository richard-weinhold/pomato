"""DE Test Case."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pomato

# Init POMATO with the options file and the dataset
wdir = Path("/examples/") # Change to local copy of examples folder
mato = pomato.POMATO(wdir=wdir, options_file="profiles/de.json")
mato.load_data('data_input/dataset_de.zip')

# Create the grid representation, the options are specified in the 
# json file supplied in the instantiation. 
# In this case the model runs a uniform pricing with subsequent redispatch.
mato.create_grid_representation()
mato.run_market_model()

# Return the results, in this case two: 
# market result (uniform pricing, no network constraints)
# and redispatch (to account for N-0 network feasibility)
market_result, redisp_result = mato.data.return_results()

# Check for overloaded lines in the market and redispatch results
n0_m, _ = market_result.overloaded_lines_n_0()
print("Number of N-0 Overloads: ", len(n0_m))
n0_r, _  = redisp_result.overloaded_lines_n_0()
print("Number of N-0 Overloads: ", len(n0_r))

# Generation comparison between Market Result and Redispatch.
# Redispatch is calculated G_redispatch - G_market as delta
# The absolute delta represents the total redispatched energy
gen = pd.merge(market_result.data.plants[["plant_type", "fuel", "technology", "g_max", "node"]],
               market_result.G, left_index=True, right_on="p")
gen = pd.merge(gen, redisp_result.G, on=["p", "t"], suffixes=("_market", "_redispatch"))
gen["delta"] = gen["G_redispatch"] - gen["G_market"]
gen["delta_abs"] = gen["delta"].abs()
print("Redispatched energy per hour (abs) [MWh] ", gen.delta_abs.sum()/len(gen.t.unique()))

# Create Geo Plot. DISCLAiMER: The reported prices are the dual 
# in the redispatch result, thus including costs for redispatch.
mato.visualization.create_geo_plot(redisp_result, show_prices=True, show_plot=False,
                                   filepath=mato.wdir.joinpath("geoplot_DE.html"))

# Create visualization of the generation schedule in the market result. 
mato.visualization.create_generation_plot(market_result, show_plot=False, 
                                          filepath=mato.wdir.joinpath("generation_plot.html"))
mato.visualization.create_storage_plot(market_result, show_plot=False, 
                                       filepath=mato.wdir.joinpath("storage_plot.html"))

mato._join_julia_instances()

