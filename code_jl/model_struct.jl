

mutable struct Results
	G::DataFrame
	H::DataFrame
	INJ::DataFrame
	F_DC::DataFrame
	EX::DataFrame
	D_es::DataFrame
	L_es::DataFrame
	D_hs::DataFrame
	L_hs::DataFrame
	D_ph::DataFrame
	INFEAS_H_POS::DataFrame
	INFEAS_H_NEG::DataFrame
	INFEAS_EL_N_POS::DataFrame
	INFEAS_EL_N_NEG::DataFrame
	EB_nodal::DataFrame
	EB_zonal::DataFrame
	CURT::DataFrame
	Alpha::DataFrame
	G_RES::DataFrame
	H_RES::DataFrame
	misc_results::Dict
	function Results()
		return new()
	end
end

mutable struct POMATO
    ### Main Attributes
    model::Model
    data::Data
    options::Dict
	results::Dict{String, Results}

    ### Maps and Sets
    n::NamedTuple{(:t, :zones, :nodes, :heatareas,
                   :plants, :res, :dc, :cb,
                   :he, :chp, :es, :hs, :ph, :alpha, :cc_res)
                    ,Tuple{Vararg{Int, 15}}}

    ## Plant Mappings
    map::NamedTuple{(:slack, # slacks to 1:n_nodes
                     :he, # 1:N.he index to 1:N.plants
                     :chp, # 1:N.chp to 1:N.he
                     :es, # 1:N.es to 1:N.plants
                     :hs, # 1:N.hs to 1:N.he
                     :ph, # 1:N.ph to 1:N.he
                     :alpha, # 1:N.alpha to 1:N.he
                     :cc_res), # map 1:cc_res to 1:n_res
                    Tuple{Vararg{Vector{Int}, 8}}}
		function POMATO()
			return new()
		end
end

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
			    :INFEAS_EL_N_POS => var_info(([:t, :nodes], [1:n.t, 1:n.nodes], [:t, :ha, :INFEAS_EL_N_POS], false)),
			    :INFEAS_EL_N_NEG => var_info(([:t, :nodes], [1:n.t, 1:n.nodes], [:t, :ha, :INFEAS_EL_N_NEG], false)),
			    :EB_nodal => var_info(([:t, :nodes], [1:n.t, 1:n.nodes], [:t, :n, :EB_zodal], true)),
			    :EB_zonal => var_info(([:t, :zones], [1:n.t, 1:n.zones], [:t, :n, :EB_zonal], true)),
			    :CURT => var_info(([:t, :renewables], [1:n.t, 1:n.res], [:t, :n, :CURT], false)),
			    :Alpha => var_info(([:t, :plants], [1:n.t, map.alpha], [:t, :n, :Alpha], false)),
			    :G_RES => var_info(([:t, :renewables], [1:n.t, 1:n.res], [:t, :n, :G_RES], false)),
			    :H_RES => var_info(([:t, :renewables], [1:n.t, 1:n.res], [:t, :n, :H_RES], false)),
				)
end

function Results(pomato::POMATO)
	result_info = get_result_info(pomato)
	r = Results()
	for v in keys(result_info)
		setfield!(r, v, model_symbol_to_df(v, result_info, pomato))
	end

	setfield!(r, :G, vcat(r.G, names!(r.G_RES, names(r.G))))
	setfield!(r, :H, vcat(r.H, names!(r.H_RES, names(r.H))))
	# Misc Results or Data
	r.misc_results = Dict()
	r.misc_results["Objective Value"] = JuMP.objective_value(pomato.model)
	r.misc_results["COST_G"] = JuMP.value(pomato.model[:COST_G])
	r.misc_results["COST_H"] = JuMP.value(pomato.model[:COST_H])
	r.misc_results["COST_EX"] = JuMP.value(pomato.model[:COST_EX])
	r.misc_results["COST_CURT"] = JuMP.value(pomato.model[:COST_CURT])
	r.misc_results["COST_INEAS_EL"] = JuMP.value(pomato.model[:COST_INFEAS_EL])
	r.misc_results["COST_INEAS_H"] = JuMP.value(pomato.model[:COST_INFEAS_H])
	r.misc_results["Solve Status"] = JuMP.termination_status(pomato.model)
	return r
end

function concat_results(results::Dict{String, Results})
	r = Results()
	for (field, fieldtype) in zip(fieldnames(Results), fieldtypes(Results))
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

function concat_results(results::Dict{String, POMATO})
	r = Results()
	for (field, fieldtype) in zip(fieldnames(Results), fieldtypes(Results))
		if fieldtype == DataFrame
			setfield!(r, field, vcat([getfield(getfield(results[k], :results)[""], field) for k in keys(results)]...))
		end
	end
	r.misc_results = Dict()
	r.misc_results["Objective Value"] = sum([getfield(results[k], :results)[""].misc_results["Objective Value"] for k in keys(results)])

	r.misc_results["COST_G"] = sum([getfield(results[k], :results)[""].misc_results["COST_G"] for k in keys(results)])
	r.misc_results["COST_H"] = sum([getfield(results[k], :results)[""].misc_results["COST_H"] for k in keys(results)])
	r.misc_results["COST_EX"] = sum([getfield(results[k], :results)[""].misc_results["COST_EX"] for k in keys(results)])
	r.misc_results["COST_CURT"] = sum([getfield(results[k], :results)[""].misc_results["COST_CURT"] for k in keys(results)])
	r.misc_results["COST_INEAS_EL"] = sum([getfield(results[k], :results)[""].misc_results["COST_INEAS_EL"] for k in keys(results)])
	r.misc_results["COST_INEAS_H"] = sum([getfield(results[k], :results)[""].misc_results["COST_INEAS_H"] for k in keys(results)])

	solved_to_opt = [getfield(results[k], :results)[""].misc_results["Solve Status"] == MOI.OPTIMAL for k in keys(results)]
	if all(solved_to_opt)
		r.misc_results["Solve Status"] = MOI.OPTIMAL
	else
		println("Not all timesteps solved to optimality!")
		println("Suboptimal Timesteps: ", join(filter((k,v) -> getfield(v, :results)[""].misc_results["Solve Status"] == MOI.OPTIMAL, pomato_results) |> keys |> collect, ", "))
	end
	return r
end

function save_results(pomato::POMATO)
	for result in keys(pomato.results)
		if result == ""
			folder = pomato.data.folders["result_dir"]
		else
			folder = result_folder*"_"*result
		end
		create_folder(folder)
		println("Saving results "*result*" to folder "*folder*"...")
		for (field, fieldtype) in zip(fieldnames(Results), fieldtypes(Results))
			if fieldtype == DataFrame
				CSV.write(folder*"/"*String(field)*".csv",
						  DataFrame(getfield(pomato.results[result], field)))
			elseif fieldtype == Dict
				open(folder*"/"*String(field)*".json", "w") do f
						write(f, JSON.json(getfield(pomato.results[result], field), 2))
				end
			else
				println(field, " not Dict or DataFrame, cant save!")
			end
		end
	end
	println("All Results Saved!")
end

function save_results(result::Results, folder::String)
	create_folder(folder)
	println("Saving results to folder "*folder*"...")
	for (field, fieldtype) in zip(fieldnames(Results), fieldtypes(Results))
		if fieldtype == DataFrame
			CSV.write(folder*"/"*String(field)*".csv",
					  DataFrame(getfield(result, field)))
		elseif fieldtype == Dict
			open(folder*"/"*String(field)*".json", "w") do f
					write(f, JSON.json(getfield(result, field), 2))
			end
		else
			println(field, " not Dict or DataFrame, cant save!")
		end
	end
	println("All Results Saved!")
end

function POMATO(model::Model,
				data::Data,
				options::Dict{String, Any})

	m = POMATO()
	m.model = model
	m.data = data
	m.options = options
	m.results = Dict{String, Results}()
	## Plant Mappings
	# map heat index to G index
	map_he = findall(plant -> plant.h_max > 0, data.plants)
	m.map = (slack = findall(node -> node.slack, data.nodes),
			 he = map_he,
			 chp = findall(plant -> ((plant.h_max > 0)&(plant.g_max > 0)), data.plants[map_he]),
			 es = findall(plant -> plant.plant_type in options["plant_types"]["es"], data.plants),
			 hs = findall(plant -> plant.plant_type in options["plant_types"]["hs"], data.plants[map_he]),
			 ph = findall(plant -> plant.plant_type in options["plant_types"]["ph"], data.plants[map_he]),
			 alpha = findall(plant -> plant.g_max > 1000, data.plants),
			 cc_res = findall(res_plants -> res_plants.g_max > 50, data.renewables))

	m.n = (t = size(data.t, 1),
		   zones = size(data.zones, 1),
		   nodes = size(data.nodes, 1),
		   heatareas = size(data.heatareas, 1),
		   plants = size(data.plants, 1),
		   res = size(data.renewables, 1),
		   dc = size(data.dc_lines, 1),
		   cb = size(data.grid, 1),
		   he = size(m.map.he, 1),
		   chp = size(m.map.chp, 1),
		   es = size(m.map.es, 1),
		   hs = size(m.map.hs, 1),
		   ph = size(m.map.ph, 1),
		   alpha = size(m.map.alpha, 1),
		   cc_res = size(m.map.cc_res, 1))
	return m
end
