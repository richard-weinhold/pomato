# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v1.0 (required)
# ----------------------------------
#
# This file:
# Auxiliary functions for data pre- and postprocessing
# -------------------------------------------------------------------------------------------------

function check_infeasibility(model)
	Gurobi.compute_conflict(model.moi_backend.optimizer.model)
	for constraint_types in list_of_constraint_types(model)
		out = filter(x -> MOI.get(model.moi_backend, Gurobi.ConstraintConflictStatus(), x.index),
			 	     all_constraints(model, constraint_types[1], constraint_types[2]))
		println(out)
	end
end

function set_model_horizon!(data)
	timesteps = [t.index for t in data.t]
	for n in data.nodes
		n.demand = n.demand[timesteps]
		n.net_export = n.net_export[timesteps]
	end
	for z in data.zones
		z.demand = z.demand[timesteps]
		z.net_position = z.net_position[timesteps]
		z.net_export = z.net_export[timesteps]
	end

	for res in data.renewables
		res.mu = res.mu[timesteps]
		res.mu_heat = res.mu_heat[timesteps]
		res.sigma = res.sigma[timesteps]
		res.sigma_heat = res.sigma_heat[timesteps]
	end
end

function create_folder(result_folder)
	if !isdir(result_folder)
	    println("Creating Results Folder")
		mkdir(result_folder)
	end
end

function model_symbol_to_df(v, result_info, pomato)
	if !(v in keys(pomato.model.obj_dict))
		arr = zeros(Int, 0, size(result_info[v].sets, 1))
	elseif result_info[v].dual
		arr = dual.(pomato.model[v])
	else
		arr = value.(pomato.model[v])
	end
	dim_arr = [map(x -> x.name, getfield(pomato.data, s))[i] for (s,i) in zip(result_info[v].sets, result_info[v].indices)]
	dims = size(dim_arr, 1)
	rows = []
	for ind in CartesianIndices(size(arr))
		row_ind = [dim_arr[dim][ind.I[dim]] for dim in 1:dims]
		push!(rows, (row_ind..., arr[ind]))
	end
	dim_names = result_info[v].columns
	df = DataFrame([dim_names[i] => [row[i] for row in rows] for i in 1:length(dim_names)])
	return df
end


#
# function jump_to_df(m::JuMP.Model,
#  				    jump_ref::Symbol,
# 				    dim_names::Array{Symbol, 1},
# 					dual::Bool,
# 					model_horizon::OrderedDict,
# 					result_folder::String="",
# 					)
# 	if dual
# 		arr = JuMP.dual.(getindex(m, jump_ref)).data
# 		dim_arr = JuMP.dual.(getindex(m, jump_ref)).axes
#
# 	else
# 		arr = JuMP.value.(getindex(m, jump_ref)).data
#         dim_arr = JuMP.value.(getindex(m, jump_ref)).axes
# 	end
# 	dims = length(dim_arr)
# 	rows = []
# 	for ind in CartesianIndices(size(arr))
# 		row_ind = [dim_arr[dim][ind.I[dim]] for dim in 1:dims]
# 		push!(rows, (row_ind..., arr[ind]))
# 	end
# 	rows = vcat(rows...)
#
# 	k = [dim_names[i] => [row[i] for row in rows] for i in 1:length(dim_names)]
# 	kv = vcat(k..., jump_ref => [row[length(row)] for row in rows])
#     df = DataFrame(kv...)
#
# 	if :t in dim_names
#         df[!, :t] = [model_horizon[x] for x in df[:, :t]]
#     end
# 	if result_folder == ""
# 		return df
# 	else
# 		CSV.write(result_folder*"/"*String(jump_ref)*".csv", df)
# 	end
# end
