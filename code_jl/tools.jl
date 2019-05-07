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

function jump_to_df(m::JuMP.Model,
 				    jump_ref::Symbol,
				    dim_names::Array{Symbol, 1},
					dual::Bool,
					result_folder::String="",
					)
	if dual
		arr = JuMP.dual.(getindex(m, jump_ref)).data
		dim_arr = JuMP.dual.(getindex(m, jump_ref)).axes

	else
		arr = JuMP.value.(getindex(m, jump_ref)).data
        dim_arr = JuMP.value.(getindex(m, jump_ref)).axes
	end
	dims = length(dim_arr)
	rows = []
	for ind in CartesianIndices(size(arr))
		row_ind = [dim_arr[dim][ind.I[dim]] for dim in 1:dims]
		push!(rows, (row_ind..., arr[ind]))
	end
	rows = vcat(rows...)

	k = [dim_names[i] => [row[i] for row in rows] for i in 1:length(dim_names)]
	kv = vcat(k..., jump_ref => [row[length(row)] for row in rows])
    df = DataFrame(kv...)

	if :t in dim_names
        df[:t] = [model_horizon[x] for x in df[:t]]
    end
	if result_folder == ""
		return df
	else
		CSV.write(result_folder*"/"*String(jump_ref)*".csv", df)
	end
end
