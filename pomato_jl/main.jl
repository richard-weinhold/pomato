# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2020
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v1.3
# ----------------------------------
#
# This file:
# POMATO optimization kernel
# Called by julia_interface.py, reads pre-processed data from /julia/data/
# Output: Optimization results saved in /julia/results/
# -------------------------------------------------------------------------------------------------

using DataFrames, CSV, JSON, Dates, Base.Threads
using LinearAlgebra, Distributions, SparseArrays
using JuMP, Mosek, MosekTools, Gurobi, GLPK

include("typedefinitions.jl")
include("model_struct.jl")
include("read_data.jl")
include("tools.jl")
include("model_functions.jl")
include("models.jl")

function run_pomato(wdir, data_dir)
	println("wdir", wdir)
	println("Read Model Data..")
	options, data = read_model_data(wdir*"/data_temp/julia_files"*data_dir)
	data.folders = Dict("wdir" => wdir,
						"data_dir" => wdir*"/data_temp/julia_files"*data_dir,
 						"result_dir" => wdir*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM"))

	create_folder(data.folders["result_dir"])
	if !("split_timeseries" in keys(options))
		options["split_timeseries"] = false
	end
	## Willy Wonka Manual Adjustments
	# options["infeasibility"]["electricity"]["bound"] = 10000
	# options["type"] = "chance_constrained"
	# data.t = data.t[1:3]

	set_model_horizon!(data)
	pomato_results = Dict{String, Result}()
	if options["split_timeseries"]
		data_0 = deepcopy(data)
		for timesteps in [t.index:t.index for t in data.t]
			data = deepcopy(data_0)
			data.t = data.t[timesteps]
			set_model_horizon!(data)
			println("Initializing Market Model for timestep $(data.t[1].name)...")
			pomato_results[data.t[1].name] = run_market_model(data, options).result
		end
	else
		pomato_results[data.t[1].name] = run_market_model(data, options).result
	end

	save_result(concat_results(pomato_results), data.folders["result_dir"])
	println("Everything Done!")
	return pomato_results
end

function run_redispatch_pomato(wdir, data_dir; redispatch_zones=["DE"])
	println("Read Model Data..")
	options, data = read_model_data(wdir*"/data_temp/julia_files"*data_dir)
	data.folders = Dict("wdir" => wdir,
						"data_dir" => wdir*"/data_temp/julia_files"*data_dir,
						"result_dir" => wdir*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM"))
	create_folder(data.folders["result_dir"])

	options["curtailment"]["cost"] = 100
	options["infeasibility"]["electricity"]["bound"] = 10000
	pomato_results = run_redispatch_model(data, options, redispatch_zones)
	println("Everything Done!")
	for result in keys(pomato_results)
		save_result(pomato_results[result], data.folders["result_dir"]*"_"*result)
	end
	return pomato_results
end
println("Initialized")

# data_dir = "/data/redispatch/"
# data_dir = "/data/"
# wdir = "C:/Users/riw/tubCloud/Uni/Market_Tool/pomato"
# # # # #
# # # # # options, data = read_model_data(wdir*"/data_temp/julia_files"*data_dir)
# options, data = read_model_data(wdir*"/data_temp/julia_files"*data_dir)
# t = run_pomato(wdir, data_dir)
# r = concat_results(t)
# save_result(r, wdir*"/julia_files/test")
# save_result(concat_results(t), wdir*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM"))

# t = run_redispatch_pomato(wdir, data_dir)
#
# sum(t["DE"].INFEAS_EL_N_POS[:, :INFEAS_EL_N_POS])
# sum(t["DE"].INFEAS_EL_N_NEG[:, :INFEAS_EL_N_NEG])
# maximum(t["DE"].INFEAS_EL_N_NEG[:, :INFEAS_EL_N_NEG])
# sum(t["market_results"].INFEAS_EL_N_POS[:, :INFEAS_EL_N_POS])
# sum(t["market_results"].INFEAS_EL_N_NEG[:, :INFEAS_EL_N_NEG])
# # save_result(t["DE"], wdir*"/data_temp/julia_files/results/redispatch_test")
# de_plants = [p.name for p in data.plants[data.zones[5].plants]]
# res_plants = [res.name for res in data.renewables[data.zones[5].res_plants]]
# redispatch_G = join(t["market_results"].G, t["DE"].G, on = [:p, :t], makeunique=true)
# redispatch_G[!, :delta] = redispatch_G[:, :G_1] - redispatch_G[:, :G]
# redispatch_G = filter(row -> (row[:p] in de_plants) , redispatch_G)
# redispatch_G[!, :delta_abs] = abs.(redispatch_G[!, :delta])
#
# sum(redispatch_G[!, :delta_abs])
# 222748 # N-0
# 412139 # N-1
# by(redispatch_G, :t, :delta_abs => sum)
#
# redispatch_G_res = join(t["market_results"].G_RES, t["DE"].G_RES, on = [:p, :t], makeunique=true)
# redispatch_G_res[!, :delta] = redispatch_G_res[:, :G_1] - redispatch_G_res[:, :G]
# redispatch_G_res = filter(row -> row[:p] in res_plants , redispatch_G_res)
# redispatch_G_res[!, :delta_abs] = abs.(redispatch_G_res[!, :delta])
#
# sum(redispatch_G_res[!, :delta_abs])
# by(redispatch_G_res, :t, :delta_abs => sum)
#
#
# by(redispatch_G, :t, :G => sum)
# by(redispatch_G, :t, :G_1 => sum)
# by(redispatch_G, :t, :delta => sum)
# sum(redispatch_G[:G_1])
# sum(data.zones[5].demand)
#
# # CSV.write(wdir*"/data_temp/julia_files/results/test_redisp_de/tmp.csv", tmp)
# 20000*1E3/8760 # MWh redispatch
#
# # save_result(t["DE"], wdir*"/data_temp/julia_files/results/test_redisp_de")
# data.zones[findfirst(z -> z.name == "DE", data.zones)].plants
# data.zones[findfirst(z -> z.name == "FR", data.zones)].plants
#
# filter(row -> row.p == data.plants[1260].name, t["market_results"].G)
# filter(row -> row.p == data.plants[1260].name, t["DE"].G)
#
# CSV.write(wdir*"/tmp.csv", redispatch_G)



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
