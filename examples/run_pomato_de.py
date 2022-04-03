# %%
"""DE Test Case."""
from pathlib import Path
import pandas as pd
import pomato

# %%
# wdir = Path("/examples/") # Change to local copy of examples folder
mato = pomato.POMATO(wdir=wdir, options_file="profiles/de.json")
mato.load_data('data_input/DE_2020.zip')

# %%
# Create the grid representation, the options are specified in the 
# json file supplied in the instantiation. 
# In this case the model runs a zonal pricing with NTC to neighboring countries
# and subsequent redispatch.

mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()

# %%
# Return the results, in this case two: 
# market result (uniform pricing, no network constraints)
# and redispatch (to account for N-0 network feasibility)
market_result, redisp_result = mato.data.return_results()

# %%
# Check for overloaded lines in the market and redispatch results
n0_m, _ = market_result.overloaded_lines_n_0()
print("Number of N-0 Overloads in market result: ", len(n0_m))
n0_r, _  = redisp_result.overloaded_lines_n_0()
print("Number of N-0 Overloads after redispatch: ", len(n0_r))

# Note: Overloads can still occur on cross-border lines, as only DE internal 
# lines are redispatched.


# %%
# Generation comparison between Market Result and Redispatch.
# Redispatch is calculated G_redispatch - G_market as delta
# The absolute delta represents the total redispatched energy
gen = pd.merge(market_result.data.plants[["plant_type", "fuel", "technology", "g_max", "node"]],
               market_result.G, left_index=True, right_on="p")
gen = pd.merge(gen, redisp_result.G, on=["p", "t"], suffixes=("_market", "_redispatch"))
gen["delta"] = gen["G_redispatch"] - gen["G_market"]
gen["delta_abs"] = gen["delta"].abs()
print("Redispatched energy per hour (abs) [MWh] ", gen.delta_abs.sum()/len(gen.t.unique()))

# Many of these functions are implemented as methods of the Result class
gen_2 = redisp_result.redispatch()
print("Redispatched energy per hour (abs) [MWh] ", gen_2.delta_abs.sum()/len(gen_2.t.unique()))


# %%
# Create Geo Plot. DISCLAiMER: The reported prices are the dual 
# in the redispatch result, thus including costs for redispatch.
mato.visualization.create_geo_plot(
    redisp_result, show_redispatch=True, 
    filepath=mato.wdir.joinpath("geoplot_DE.html")
)

# %%
# Create visualization of the generation schedule in the market result. 
mato.visualization.create_generation_plot(
    market_result,
    filepath=mato.wdir.joinpath("generation_plot.html")
)

# %%
# 
mato._join_julia_instances()


