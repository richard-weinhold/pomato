# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v1.0
# ----------------------------------
#
# This file:
# POMATO optimization kernel
# Called by julia_interface.py, reads pre-processed data from /julia/data/
# Output: Optimization results saved in /julia/results/
# -------------------------------------------------------------------------------------------------

# To use file both in POMATO and as script
if length(ARGS) > 0
    println("Arguments passed, running in: ", ARGS[1], " and data folder: ", ARGS[2])
    WDIR = ARGS[1]
    data_folder = ARGS[2]
else
	println("No arguments passed, running as script in pwd()")
    WDIR = pwd()
    data_folder = "/data/"
end
# WDIR = "C:/Users/riw/tubCloud/Uni/Market_Tool/pomato"
# /net/work/wein/pomato//data_temp/julia_files/data/
using DataFrames
using CSV
using JSON
using DataStructures
using JuMP
using Gurobi
using Dates

include("tools.jl")
include("typedefinitions.jl")
include("read_data.jl")
include("model.jl")
include("setdefinitions.jl")

model_horizon, options,
plants, plants_in_ha, plants_at_node, plants_in_zone, availabilities,
nodes, zones, slack_zones, heatareas, nodes_in_zone,
ntc, dc_lines, grid = read_model_data(WDIR*"/data_temp/julia_files"*data_folder)


# Run Dispatch Model
out = build_and_run_model(model_horizon, options,
                          plants, plants_in_ha, plants_at_node, plants_in_zone, 
                          availabilities,
                          nodes, zones, heatareas, nodes_in_zone,
                          dc_lines, grid, ntc, slack_zones)


println("DONE")
