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

using DataFrames, CSV, JSON, Dates, Base.Threads
using LinearAlgebra, Distributions, SparseArrays
using JuMP, Mosek, MosekTools, Gurobi

include("typedefinitions.jl")
include("read_data.jl")
include("tools.jl")
include("model_struct.jl")
include("model_functions.jl")
include("models.jl")
data_dir = "/data/"
wdir = "C:/Users/riw/tubCloud/Uni/Market_Tool/pomato"

function run_pomato(wdir, data_dir)
	data_dir = wdir*"/data_temp/julia_files"*data_dir
	println("Read Model Data..")
	options, data = read_model_data(data_dir)
	data.folders = Dict("wdir" => wdir,
						"data_dir" => data_dir)

	# result_folder = wdir*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM")
	data.folders["result_dir"] = data.folders["wdir"]*"/data_temp/julia_files/results/test"
	create_folder(data.folders["result_dir"])

	options["infeasibility"]["electricity"]["bound"] = 10000
	options["split"] = true

	data.t = data.t[1:3]
	set_model_horizon!(data)
	data_0 = deepcopy(data)

	if options["split"]
		timestep_ranges = [t.index:t.index for t in data.t]
	else
		timestep_ranges = [data.t[1].index:data.t[end].index]
	end

	pomato_results = Dict{String, POMATO}()
	for timesteps in timestep_ranges
		data = deepcopy(data_0)
		data.t = data.t[timesteps]
		set_model_horizon!(data)
		println("Initializing Market Model..")
		pomato_results[data.t[1].name] = run_market_model(data, options)
	end

	# println(typeof(pomato))
	# if options["split"]
	# 	pomato.results = Dict("" => concat_results(pomato_results))
	# end
	save_results(concat_results(pomato_results), data.folders["result_dir"])
	println("Model Done!")
	return pomato_results
end

data_dir = "/data/"
wdir = "C:/Users/riw/tubCloud/Uni/Market_Tool/pomato"
options, data = read_model_data(wdir*"/data_temp/julia_files"*data_dir)


# t = run_pomato(wdir, data_dir)
# r = concat_results(t)

# r.G[r.G[:p] .== "p4770", :]
# println("Read Model Data..")
# options, data = read_model_data(wdir*"/data_temp/julia_files"*data_dir)
# data.folders = Dict("wdir" => wdir,
# 					"data_dir" => data_dir)
#
# 	# result_folder = wdir*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM")
# data.folders["result_dir"] = data.folders["wdir"]*"/data_temp/julia_files/results/test"
# create_folder(data.folders["result_dir"])
# options["infeasibility"]["electricity"]["bound"] = 10000
# data.t = data.t[1:1]
# set_model_horizon!(data)
# pomato = run_market_model(data, options)
# pomato.results

# t = run_pomato(wdir, data_dir)
