# %%
"""Chance-Constrained Test Case."""
from pathlib import Path
import pandas as pd
import pomato
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.io as pio
pio.renderers.default = "browser"

# %%
wdir = Path(r"C:\Users\riw\Documents\environments\pomato_310_testing") # Change to local copy of examples folder

mato = pomato.POMATO(wdir=wdir, options_file="profiles/nrel118.json")
mato.load_data('data_input/nrel_118_high_res.zip')

# %% Calculate N-0 Dispatch

mato.options["title"] = "N-0"
mato.options["type"] = "opf"
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()

# %% Calculate N-0 Chance Constrained
mato.options["title"] = "N-0 CC"
mato.options["chance_constrained"]["include"] = True
mato.options["chance_constrained"]["fixed_alpha"] = True
mato.options["chance_constrained"]["cc_res_mw"] = 0
mato.options["chance_constrained"]["alpha_plants_mw"] = 30
mato.options["chance_constrained"]["epsilon"] = 0.05
mato.options["chance_constrained"]["percent_std"] = 0.1
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()


# %% Calculate N-0 Chance Constrained - Variable Alpha
mato.options["title"] = "N-0 CC - Variable Alpha"
mato.options["chance_constrained"]["fixed_alpha"] = False
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()

# %%

mato.visualization.create_cost_overview(mato.data.results.values())

# %% Join all subprocesses. 
mato._join_julia_instances()
