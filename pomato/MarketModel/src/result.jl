"""Result related functions."""

function get_result_info(pomato::POMATO)
	n, map = pomato.n, pomato.map
	var_info(x) = NamedTuple{(:sets, :indices, :columns, :dual), Tuple{Vector{Symbol}, Vector{AbstractArray{Int, 1}}, Vector{Symbol}, Bool}}(x)
	return Dict(:G => var_info(([:t, :plants], [1:n.t, 1:n.plants], [:t, :p, :G], false)),
			    :H => var_info(([:t, :plants], [1:n.t, map.he], [:t, :p, :H], false)),
			    :INJ => var_info(([:t, :nodes], [1:n.t, 1:n.nodes], [:t, :n, :INJ], false)),
			    :F_DC => var_info(([:t, :dc_lines], [1:n.t, 1:n.dc], [:t, :dc, :F_DC], false)),
			    :EX => var_info(([:t, :zones, :zones], [1:n.t, 1:n.zones, 1:n.zones], [:t, :z, :zz, :EX], false)),
			    :D_es => var_info(([:t, :plants], [1:n.t, map.es], [:t, :p, :D_es], false)),
			    :L_es => var_info(([:t, :plants], [1:n.t, map.es], [:t, :p, :L_es], false)),
			    :D_hs => var_info(([:t, :plants], [1:n.t, map.he[map.hs]], [:t, :p, :D_hs], false)),
			    :L_hs => var_info(([:t, :plants], [1:n.t, map.he[map.hs]], [:t, :p, :L_hs], false)),
			    :D_ph => var_info(([:t, :plants], [1:n.t, map.he[map.ph]], [:t, :p, :D_ph], false)),
			    :INFEAS_H_POS => var_info(([:t, :heatareas], [1:n.t, 1:n.heatareas], [:t, :ha, :INFEAS_H_POS], false)),
			    :INFEAS_H_NEG => var_info(([:t, :heatareas], [1:n.t, 1:n.heatareas], [:t, :ha, :INFEAS_H_NEG], false)),
			    :INFEAS_EL_N_POS => var_info(([:t, :nodes], [1:n.t, 1:n.nodes], [:t, :n, :INFEAS_EL_N_POS], false)),
			    :INFEAS_EL_N_NEG => var_info(([:t, :nodes], [1:n.t, 1:n.nodes], [:t, :n, :INFEAS_EL_N_NEG], false)),
			    :EB_nodal => var_info(([:t, :nodes], [1:n.t, 1:n.nodes], [:t, :n, :EB_zodal], true)),
			    :EB_zonal => var_info(([:t, :zones], [1:n.t, 1:n.zones], [:t, :n, :EB_zonal], true)),
			    :CURT => var_info(([:t, :renewables], [1:n.t, 1:n.res], [:t, :n, :CURT], false)),
			    :Alpha => var_info(([:t, :plants], [1:n.t, map.alpha], [:t, :n, :Alpha], false)),
			    :G_RES => var_info(([:t, :renewables], [1:n.t, 1:n.res], [:t, :n, :G_RES], false)),
			    :H_RES => var_info(([:t, :renewables], [1:n.t, 1:n.res], [:t, :n, :H_RES], false)),
				)
end

function Result(pomato::POMATO)
	result_info = get_result_info(pomato)
	result = Result()
	for v in keys(result_info)
		setfield!(result, v, model_symbol_to_df(v, result_info, pomato))
	end

	setfield!(result, :G, vcat(result.G, rename!(result.G_RES, names(result.G))))
	setfield!(result, :H, vcat(result.H, rename!(result.H_RES, names(result.H))))
	# Misc Results or Data
	result.misc_results = Dict()
	result.misc_results["Objective Value"] = JuMP.objective_value(pomato.model)
	result.misc_results["COST_G"] = JuMP.value(pomato.model[:COST_G])
	result.misc_results["COST_H"] = typeof(pomato.model[:COST_H]) == GenericAffExpr{Float64, VariableRef} ?  JuMP.value(pomato.model[:COST_H]) : 0
	result.misc_results["COST_EX"] = JuMP.value(pomato.model[:COST_EX])
	result.misc_results["COST_CURT"] = JuMP.value(pomato.model[:COST_CURT])
	result.misc_results["COST_INEAS_EL"] = JuMP.value(pomato.model[:COST_INFEAS_EL])
	result.misc_results["COST_INEAS_H"] = JuMP.value(pomato.model[:COST_INFEAS_H])
	result.misc_results["Solve Status"] = JuMP.termination_status(pomato.model)
	return result
end

function concat_results(results::Dict{String, Result})
	r = Result()
	for (field, fieldtype) in zip(fieldnames(Result), fieldtypes(Result))
		if fieldtype == DataFrame
			setfield!(r, field, vcat([getfield(results[k], field) for k in keys(results)]...))
		end
	end

	r.misc_results = Dict()
	r.misc_results["Objective Value"] = sum([results[k].misc_results["Objective Value"] for k in keys(results)])

	r.misc_results["COST_G"] = sum([results[k].misc_results["COST_G"] for k in keys(results)])
	r.misc_results["COST_H"] = sum([results[k].misc_results["COST_H"] for k in keys(results)])
	r.misc_results["COST_EX"] = sum([results[k].misc_results["COST_EX"] for k in keys(results)])
	r.misc_results["COST_CURT"] = sum([results[k].misc_results["COST_CURT"] for k in keys(results)])
	r.misc_results["COST_INEAS_EL"] = sum([results[k].misc_results["COST_INEAS_EL"] for k in keys(results)])
	r.misc_results["COST_INEAS_H"] = sum([results[k].misc_results["COST_INEAS_H"] for k in keys(results)])

	solved_to_opt = [results[k].misc_results["Solve Status"] == MOI.OPTIMAL for k in keys(results)]
	if all(solved_to_opt)
		r.misc_results["Solve Status"] = MOI.OPTIMAL
	else
		println("Not all timesteps solved to optimality!")
		println("Suboptimal Timesteps: ", join(filter((k,v) -> v.misc_results["Solve Status"] == MOI.OPTIMAL, pomato_results) |> keys |> collect, ", "))
	end
	return r
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
