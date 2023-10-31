# %%
"""FBMC Test Case."""
from pathlib import Path
import pomato
import plotly.io as pio
pio.renderers.default = "browser"

# %%
wdir = Path(r"C:\Users\riw\Documents\environments\pomato_310_testing") # Change to local copy of examples folder

mato = pomato.POMATO(wdir=wdir, options_file="profiles/nrel118.json")
mato.load_data('data_input/nrel_118_high_res.zip')

# %% Nodal Basecase
mato.options["title"] = "Basecase"
mato.options["type"] = "opf"
mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()
result_name = next(r for r in list(mato.data.results))
basecase = mato.data.return_results("Basecase")

# %% Calculate FB Parameters with 40%minRAM

mato.options["fbmc"]["minram"] = 0.4
mato.options["fbmc"]["frm"] = 0.1
mato.options["fbmc"]["cne_sensitivity"] = 0.05
mato.options["fbmc"]["gsk"] = "dynamic"
mato.options["fbmc"]["reduce"] = True
fb_parameters = mato.create_flowbased_parameters(basecase)

# FBMC market clearing
mato.options["title"] = "FB Market Coupling - 40%"
mato.options["redispatch"]["include"] = True
mato.options["redispatch"]["zones"] = list(mato.data.zones.index)
mato.create_grid_representation(flowbased_parameters=fb_parameters)
mato.update_market_model_data()
mato.run_market_model()
fb_market_result, _ = mato.data.return_results(mato.options["title"])

# %% Chance Constrained 

mato.options["title"] = "FB CC Market Coupling - 40%"
mato.options["chance_constrained"]["include"] = True
mato.options["chance_constrained"]["fixed_alpha"] = True
mato.options["chance_constrained"]["cc_res_mw"] = 0
mato.options["chance_constrained"]["alpha_plants_mw"] = 30
mato.options["chance_constrained"]["epsilon"] = 0.05
mato.options["chance_constrained"]["percent_std"] = 0.05
mato.options["fbmc"]["frm"] = 0

mato.update_market_model_data()
mato.run_market_model()
fb_cc_market_result, _ = mato.data.return_results(mato.options["title"])
print("Mean CC Margin:", fb_cc_market_result.CC_LINE_MARGIN["CC_LINE_MARGIN"].mean())

# %% Cost Overview

mato.visualization.create_cost_overview(mato.data.results.values())

# %%
domain_x, domain_y, timestep = ("R2", "R3"), ("R1", "R3"), "t0021"

fbmc_domains = pomato.visualization.FBDomainPlots(mato.data, fb_parameters)
fbmc_domains.generate_flowbased_domain(domain_x, domain_y, timestep=timestep, shift2MCP=True, result=fb_market_result)
fbmc_domains.generate_flowbased_domain(domain_x, domain_y, timestep=timestep, shift2MCP=True, result=fb_cc_market_result)

for elm in fbmc_domains.fbmc_plots:
    elm.x_max, elm.x_min = 1000, -1100
    elm.y_max, elm.y_min = 500, -400
    fig = mato.visualization.create_fb_domain_plot(elm, show_plot=False)
    fig.show()
    fig.write_html(elm.title)

# %% Join all subprocesses. 
mato._join_julia_instances()
