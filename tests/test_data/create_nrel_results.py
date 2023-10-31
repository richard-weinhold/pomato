"""Create test data, i.e. results for the NREL 118 Bus case."""
from pathlib import Path
import numpy as np
import shutil
import pandas as pd
import pomato

# Init POMATO with the options file and the dataset
# wdir = Path("/examples/") # Change to local copy of examples folder
repodir = Path(r"C:\Users\riw\Documents\repositories\pomato") # Change to local copy of examples folder
wdir = repodir.joinpath("tests/test_data/_create_test_data")
if not wdir.is_dir():
    wdir.mkdir()
    
mato = pomato.POMATO(wdir=wdir, options_file=repodir.joinpath("examples/profiles/nrel118.json"))
mato.load_data(repodir.joinpath("examples/data_input/nrel_118_original.zip"))
mato.data.results = {}

def system_balance(result):
    model_horizon = result.result_attributes["model_horizon"]
    condition = result.data.demand_el.timestep.isin(model_horizon)
    return (result.G.G.sum() 
            - result.data.demand_el.loc[condition, "demand_el"].sum()  
            + result.INFEASIBILITY_EL_POS.INFEASIBILITY_EL_POS.sum() 
            - result.INFEASIBILITY_EL_NEG.INFEASIBILITY_EL_NEG.sum())

# %% Run uniform pricing
mato.options["type"] = "uniform"
mato.options["title"] = "Uniform"
mato.options["model_horizon"] = [0, 24]
mato.options["redispatch"]["include"] = True
mato.options["redispatch"]["zones"] = ["R1", "R2", "R3"]

mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()

uniform_market, uniform_redispatch = mato.data.return_results(title="Uniform")
uniform_market.save(wdir.joinpath("data_output/uniform_market"))
uniform_redispatch.save(wdir.joinpath("data_output/uniform_redispatch"))

df1, df2 = uniform_market.overloaded_lines_n_0()
df3, df4 = uniform_redispatch.overloaded_lines_n_0()
print("Number of overloaded lines (Market): ", len(df1))
print("Number of overloaded lines (Redispatch): ", len(df3))

system_balance(uniform_market)
system_balance(uniform_redispatch)

# %% Run nodal-prcing market clearing
mato.options["type"] = "opf"
mato.options["title"] = "Opf"

mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()

opf_market, opf_redispatch = mato.data.return_results(title="Opf")
opf_market.save(wdir.joinpath("data_output/opf_market"))
opf_redispatch.save(wdir.joinpath("data_output/opf_redispatch"))

df1, df2 = opf_market.overloaded_lines_n_0()
df3, df4 = opf_market.overloaded_lines_n_1()
print("Number of overloaded lines (Market): ", len(df1))
print("Number of overloaded lines N-1 (Market): ", len(df3))



# %% Rerun the model as SCOPF

mato.options["type"] = "scopf"
mato.options["title"] = "Scopf"
mato.options["redispatch"]["include"] = False

mato.create_grid_representation()
mato.update_market_model_data()
mato.run_market_model()
mato.options["grid"]
# Gurobi 764

file = pomato.tools.newest_file_folder(mato.grid_model.julia_dir.joinpath("cbco_data"), keyword="cbco")
shutil.copy(file, mato.wdir.joinpath("data_output/nrel_cbco_indices.csv"))
mato.grid_representation.grid[["cb", "co"]].to_csv(wdir.joinpath("data_output/nrel_cbco_table.csv"))

scopf_market = mato.data.return_results(title="Scopf")
scopf_market.save(wdir.joinpath("data_output/scopf_market"))

df1, df2 = scopf_market.overloaded_lines_n_0()
df3, df4 = scopf_market.overloaded_lines_n_1()
print("Number of overloaded lines (Market): ", len(df1))
print("Number of overloaded lines N-1 (Market): ", len(df3))

# Join all subprocesses. 
mato._join_julia_instances()
