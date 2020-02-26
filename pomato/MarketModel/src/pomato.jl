mutable struct Result
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
	function Result()
		return new()
	end
end

mutable struct POMATO
    ### Main Attributes
    model::Model
    data::Data
    options::Dict
	result::Result

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

function POMATO(model::Model,
				data::Data,
				options::Dict{String, Any})

	m = POMATO()
	m.model = model
	m.data = data
	m.options = options

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

function check_infeasibility(model::Model)
	Gurobi.compute_conflict(model.moi_backend.optimizer.model)
	for constraint_types in list_of_constraint_types(model)
		out = filter(x -> MOI.get(model.moi_backend, Gurobi.ConstraintConflictStatus(), x.index),
			 	     all_constraints(model, constraint_types[1], constraint_types[2]))
		println(out)
	end
end
