"""IEEE Test Case."""
from pathlib import Path
import numpy as np
import pandas as pd
import pomato


mato = pomato.POMATO(wdir=self.wdir, options_file="profiles/nrel118.json")
mato.load_data('data_input/nrel_118_original.zip')

mato.options["model_horizon"] = [0, 168]
mato.options["redispatch"]["include"] = False
mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
mato.options["infeasibility"]["electricity"]["bound"] = 200
mato.options["infeasibility"]["electricity"]["cost"] = 1000
mato.options["redispatch"]["cost"] = 20

# %% Nodal Basecase
mato.data.results = {}
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
fbmc_domain = pomato.visualization.FBDomainPlots(mato.data, fb_parameters)
fbmc_domain.generate_flowbased_domains(("R1", "R2"), ["R1", "R3"], timesteps=["t0001"])
mato.visualization.create_fb_domain_plot(fbmc_domain.fbmc_plots[0], show_plot=False)

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
