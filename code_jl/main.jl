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
# if length(ARGS) > 0
#     println("Arguments passed, running in: ", ARGS[1], " and data folder: ", ARGS[2])
#     global WDIR = ARGS[1]
#     global data_dir = ARGS[2]
# else
# 	println("No arguments passed, running as script in pwd()")
#     global WDIR = pwd()
#     global data_dir = "/data/"
# end
# WDIR = "C:/Users/riw/tubCloud/Uni/Market_Tool/pomato"
# /net/work/wein/pomato//data_temp/julia_files/data/
using DataFrames
using CSV
using JSON
using DataStructures
using JuMP
using Gurobi, GLPK
using Dates

include("tools.jl")
include("typedefinitions.jl")
include("read_data.jl")
include("model.jl")
include("setdefinitions.jl")


function run(WDIR=pwd(), DATA_DIR="/data/")
	DATA_DIR="/data/"
	WDIR = "C:/Users/riw/tubCloud/Uni/Market_Tool/pomato/"
	DATA_DIR = WDIR*"/data_temp/julia_files"*DATA_DIR
	model_horizon, options, plant_types,
	plants,
	nodes, zones, heatareas,
	grid, dc_lines = read_model_data(DATA_DIR)

	# Run Dispatch Model
	out = build_and_run_model(WDIR,
							  model_horizon, options, plant_types,
	                          plants,
	                          nodes, zones, heatareas,
	                          grid, dc_lines)

	println("Model Done!")
end


println("Initialized")

