# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018 
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v0.6.2 (required)
# ----------------------------------
# 
# This file:
# POMATO optimization kernel
# Called by julia_interface.py, reads pre-processed data from /julia/data/
# Output: Optimization results saved in /julia/results/
# -------------------------------------------------------------------------------------------------

# To use file both in POMATO and as script
if length(ARGS) > 0
    WDIR = ARGS[1]
else
    println("No arguments passed, running as script in pwd()")
    WDIR = pwd()
end

using DataFrames
using CSV
using JSON
using DataStructures
using JuMP
using Clp
using Gurobi


include("tools.jl")
include("typedefinitions.jl")
include("read_data.jl")
include("model.jl")
include("setdefinitions.jl")

model_horizon, opt_setup, 
plants, plants_in_ha, plants_at_node, plants_in_zone, availabilities,
nodes, zones, slack_zones, heatareas, nodes_in_zone,
ntc, dc_lines, cbco = read_model_data(WDIR*"/data_temp/julia_files/data/")


#Run Dispatch Model
out = build_and_run_model(model_horizon, opt_setup, 
                          plants, plants_in_ha, plants_at_node, plants_in_zone, availabilities,
                          nodes, zones, slack_zones, heatareas, nodes_in_zone,
                          ntc, dc_lines, cbco)


println("DONE")
