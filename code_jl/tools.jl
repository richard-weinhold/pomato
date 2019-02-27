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

function indexdf_to_indexdict(df::DataFrame, indexcol::Symbol)
    out_dict = Dict()
    for row in 1:nrow(df)
        tmp = Dict()
        for col in names(df)
            tmp[col] = df[row, col]
            println
        end
        out_dict[df[row, indexcol]] = deepcopy(tmp)
    end
    return out_dict
end

function write_to_csv(dir::String, fname::String, df::DataFrame; time_stamp=false)

    if isdir(dir) == false
        mkdir(dir)
    end
    if time_stamp
        rn = now()
        fname = "$fname$(rn)"
        fname = replace(fname, ":", "-")
    end

    if !contains(fname, ".csv")
        fname = "$fname.csv"
    end

    try
        writetable("$dir/$fname", df)
    catch
        warn("Can't overwrite $fname. Check if the file is opened, if so close it.")
    end
end

function progress(t, max_t)
    max_bars = 35
    r = t/max_t
    n = Int(floor(r*35))
    p = Int(floor(r*100))
    bar = repeat("=", n)
    res = repeat(" ", max_bars - n)
    out = "[" * bar * res * "] " * string(p) * "%"
    (t == max_t ? print("\r" * out * "\n") : print("\r" * out))
    flush(STDOUT)
end

function jump_to_df(m::JuMP.Model,
 				    jump_ref::Symbol,
				    dim_names::Array{Symbol, 1},
					dual::Bool,
					result_folder::String="",
					)
	if dual
		arr = getdual(getindex(m, jump_ref)).innerArray
		dim_arr = getdual(getindex(m, jump_ref)).indexsets
	else
		arr = getvalue(getindex(m, jump_ref)).innerArray
		dim_arr = getvalue(getindex(m, jump_ref)).indexsets
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
