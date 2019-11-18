# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v0.6.2 (required)
# ----------------------------------------
# This file:
# Central optimization implemented in JuMP
# ----------------------------------------

# using DataFrames, CSV, JSON, Dates, Base.Threads
# using LinearAlgebra, Distributions, SparseArrays
# using JuMP, Mosek, MosekTools, Gurobi
#
# include("typedefinitions.jl")
# include("read_data.jl")
# include("tools.jl")
# include("model_struct.jl")
# include("model_functions.jl")
# include("models.jl")

# data_dir = "/data/"
# wdir = "C:/Users/riw/tubCloud/Uni/Market_Tool/pomato"
# result_name = ""

function run_market_model(data::Data, options::Dict{String, Any}; result_name::String="")

	if in(options["type"] , ["chance_constrained"])
		pomato = POMATO(Model(with_optimizer(Mosek.Optimizer, logFile=data.folders["result_dir"]*"/log.txt")),
					   	data, options)
	else
		pomato = POMATO(Model(with_optimizer(Gurobi.Optimizer, LogFile=data.folders["result_dir"]*"/log.txt")),
					   	data, options)
	end

	println("Adding Variables and Expressions..")
	add_variables_expressions!(pomato)

	println("Add Base Model")
	add_electricity_generation_constraints!(pomato)
	add_electricity_storage_constraints!(pomato)

	if pomato.options["heat_model"]
		println("Adding Heat Model...")
		add_heat_generation_constraints!(pomato)
	end
	if pomato.options["curtailment"]["include"]
		println("Adding Curtailment...")
		add_curtailment_constraints!(pomato)
	end

	if in(pomato.options["type"] , ["ntc", "zonal", "cbco_zonal"])
		println("Adding NTC Constraints...")
		add_ntc_constraints!(pomato)
	end

	if in(pomato.options["type"] , ["cbco_nodal", "nodal"])
		println("Adding Load Flow Constraints...")
		add_dclf_constraints!(pomato)
	end
	if in(pomato.options["type"] , ["chance_constrained"])
		println("Adding Chance Constraints...")
		@time add_chance_constraints!(pomato, fixed_alpha=true)
	end

	println("Adding NEX Constraints...")
	add_net_position_constraints!(pomato)
	println("Adding Objective Function...")
	add_objective!(pomato)

	println("Solving...")
	t_start = time_ns()
	@time JuMP.optimize!(pomato.model)
	println("Objective: $(JuMP.objective_value(pomato.model))")
	println("Objective: $(JuMP.termination_status(pomato.model))")
	t_elapsed = time_ns() - t_start
	println("Solvetime: $(round(t_elapsed*1e-9, digits=2)) seconds")
	add_results!(pomato, result_name)

	return pomato
end

# redispatch_zone = "DE"
function run_redispatch_model(data::Data, redispatch_zone::String)

	pomato = market_model(data; result_name="market_model")
	market_model_results = Dict()

	market_model_results["g_market"] = value.(pomato.model[:G])
	market_model_results["d_es_market"] = value.(pomato.model[:D_es])
	market_model_results["d_ph_market"] = value.(pomato.model[:D_ph])
	market_model_results["infeas_pos_market"] = value.(pomato.model[:INFEAS_EL_N_POS])
	market_model_results["infeas_neg_market"] = value.(pomato.model[:INFEAS_EL_N_NEG])
	redispatch_model!(pomato, market_model_results, redispatch_zone);

	println("Solving...")
	t_start = time_ns()
	@time JuMP.optimize!(pomato.model)
	println("Objective: $(JuMP.objective_value(pomato.model))")
	println("Objective: $(JuMP.termination_status(pomato.model))")
	t_elapsed = time_ns() - t_start
	println("Solvetime: $(round(t_elapsed*1e-9, digits=2)) seconds")

	add_results!(pomato, "redispatch_"*redispatch_zone)

	return pomato
end


# tmp = join(pomato.results[""].G, pomato.results["redispatch_DE"].G, on = [:p, :t], makeunique=true)
# tmp[!, :delta] = tmp[:, :G_1] - tmp[:, :G]
# by(tmp, :t, :delta => x -> sum(abs.(x)))
#
# save_results(pomato)

# Objective: 4.570786289590631e6 var alpha t:1:2
# Objective: 4.583418581323959e6 fixed alpha 1:2
# check_infeasibility(pomato.model)
# println("Saving results to results folder: ", result_folder)

# write_results(pomato, result_folder)


# CSV.write(result_folder*"/"*"H"*".csv", df)


# model_symbol_to_df(:F_DC, result_info, pomato)

function test_model(wdir, data_dir)
	data_dir = wdir*"/data_temp/julia_files"*data_dir
	println("Read Model Data..")
	@time options, data = read_model_data(data_dir)
	data.t = data.t[1:2]
	set_model_horizon!(data)

	println("Initializing Market Model..")
	result_folder = wdir*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM")

	pomato = POMATO(Model(with_optimizer(Gurobi.Optimizer), LogFile=result_folder*"/log.txt"),
				data, options)

	println("Adding Variables and Expressions..")
	add_variables_expressions!(pomato)

	print("Add Base Model")
	add_electricity_generation_constraints!(pomato)
	add_electricity_storage_constraints!(pomato)
	# add_heat_generation_constraints!(pomato)

	# add_curtailment_constraints!(pomato)
	println("Load Flow Constraints...")
	# @time add_dclf_constraints!(pomato)
	# @time add_chance_constraints!(pomato, fixed_alpha=true)

	println("NTC Constraints...")
	add_ntc_constraints!(pomato)
	println("NEX Constraints...")
	add_net_position_constraints!(pomato)
	println("Adding Objective...")
	add_objective!(pomato)

	println("Solving...")
	t_start = time_ns()
	@time JuMP.optimize!(pomato.model)
	println("Objective: $(JuMP.objective_value(pomato.model))")
	println("Objective: $(JuMP.termination_status(pomato.model))")
	t_elapsed = time_ns() - t_start
	println("Solvetime: $(round(t_elapsed*1e-9, digits=2)) seconds")

	add_results!(pomato, "market")
	redispatch_model!(pomato, "DE");

	println("Solving...")
	t_start = time_ns()
	@time JuMP.optimize!(pomato.model)
	println("Objective: $(JuMP.objective_value(pomato.model))")
	println("Objective: $(JuMP.termination_status(pomato.model))")
	t_elapsed = time_ns() - t_start
	println("Solvetime: $(round(t_elapsed*1e-9, digits=2)) seconds")

	add_results!(pomato, "redispatch_DE")

	tmp = join(pomato.results["market"].G, pomato.results["redispatch_DE"].G, on = [:p, :t], makeunique=true)
	tmp[!, :delta] = tmp[:, :G_1] - tmp[:, :G]
	by(tmp, :t, :delta => x -> sum(abs.(x)))

	save_results(pomato)
end

# Objective: 4.570786289590631e6 var alpha t:1:2
# Objective: 4.583418581323959e6 fixed alpha 1:2
# check_infeasibility(pomato.model)
# println("Saving results to results folder: ", result_folder)

# write_results(pomato, result_folder)


# CSV.write(result_folder*"/"*"H"*".csv", df)


# model_symbol_to_df(:F_DC, result_info, pomato)
