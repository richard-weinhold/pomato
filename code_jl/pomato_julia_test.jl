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
# Testing environment for the Julia Kernel
# -------------------------------------------------------------------------------------------------


# Invoke Required Modules
using DataFrames, CSV, JSON, DataStructures
using JuMP, Clp


# To use file both in POMATO and as script
if length(ARGS) > 0
    WDIR = ARGS[1]
else
    println("No arguments passed, running as script in pwd()")
    WDIR = pwd()
end

# For prototyping Robert
# WDIR = "C:/Users/Robert/Google Drive/My_Documents/_Projects/Market Tool/market_tool_julia_extension/julia"
# WDIR = "/Users/Balu/IEEEGoogleDrive/My_Documents/_Projects/Market Tool/market_tool_julia_extension/julia"
# model_type = "dispatch"

include("tools.jl")
include("typedefinitions.jl")
include("read_data.jl")
include("model.jl")
include("setdefinitions.jl")

plants, plants_in_ha, plants_at_node, plants_in_zone, availabilites, 
dc_lines, nodes, slack_zones, zones, heatareas, cbco, ntc, 
model_horizon, opt_setup = read_model_data(WDIR*"/data/")


#Run Dispatch Model
out = build_and_run_model(plants, plants_in_ha, plants_at_node, plants_in_zone, availabilites, 
                   nodes, zones, heatareas, ntc, dc_lines, slack_zones, cbco,
                   model_horizon, opt_setup)


println("DONE")
