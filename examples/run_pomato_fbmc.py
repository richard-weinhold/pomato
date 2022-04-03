# %%
"""FBMC Test Case."""
from pathlib import Path
import pandas as pd
import pomato

# %%
# wdir = Path("/examples/") # Change to local copy of examples folder
mato = pomato.POMATO(wdir=wdir, options_file="profiles/nrel118.json")
mato.load_data('data_input/nrel_118_original.zip')

mato.options["model_horizon"] = [0, 168]
mato.options["redispatch"]["include"] = False
mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
mato.options["infeasibility"]["electricity"]["bound"] = 200
mato.options["infeasibility"]["electricity"]["cost"] = 1000
mato.options["redispatch"]["cost"] = 20

# %% Nodal Basecase
mato.options["type"] = "opf"
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()
result_name = next(r for r in list(mato.data.results))
basecase = mato.data.results[result_name]

mato.options["fbmc"]["minram"] = 0.2
mato.options["fbmc"]["lodf_sensitivity"] = 0.1
mato.options["fbmc"]["cne_sensitivity"] = 0.2
fb_parameters = mato.create_flowbased_parameters(basecase)
fbmc_domains = pomato.visualization.FBDomainPlots(mato.data, fb_parameters)
fb_domain = fbmc_domains.generate_flowbased_domain(("R1", "R2"), ["R1", "R3"], timestep="t0002")
mato.visualization.create_fb_domain_plot(fb_domain, show_plot=True)

# %% FBMC market clearing
mato.options["redispatch"]["include"] = True
mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
mato.create_grid_representation(flowbased_parameters=fb_parameters)
mato.update_market_model_data()
mato.run_market_model()

mato.visualization.create_generation_overview(list(mato.data.results.values()), show_plot=False)
mato._join_julia_instances()

# Join all subprocesses. 
mato._join_julia_instances()
