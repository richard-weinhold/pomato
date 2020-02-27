"""
asd

"""

function run_market_model(data::Data, options::Dict{String, Any})

	if in(options["type"] , ["chance_constrained"])
		pomato = POMATO(Model(optimizer_with_attributes(Mosek.Optimizer)), # logFile=data.folders["result_dir"]*"/log.txt")),
					   	data, options)
	else
		pomato = POMATO(Model(with_optimizer(Gurobi.Optimizer, LogFile=data.folders["result_dir"]*"/log.txt")),
					    data, options)

		# pomato = POMATO(Model(optimizer_with_attributes(Gurobi.Optimizer, "LogFile" => data.folders["result_dir"]*"/log.txt")),
		# 			   	data, options)

		# pomato = POMATO(Model(with_optimizer(GLPK.Optimizer)), #LogFile=data.folders["result_dir"]*"/log.txt")),
		# 			   	data, options)
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

	if in(pomato.options["type"] , ["zonal", "cbco_zonal"])
		println("Adding FlowBased Constraints...")
		add_flowbased_constraints!(pomato)
	end

	if in(pomato.options["type"] , ["cbco_nodal", "nodal"])
		println("Adding Load Flow Constraints...")
		add_dclf_constraints!(pomato)
	end

	if in(pomato.options["type"] , ["chance_constrained"])
		println("Adding Chance Constraints...")
		@time add_chance_constraints!(pomato, fixed_alpha=true)
	end
	# if any([isdefined(z, :net_position) for z in pomato.data.zones]) & !(pomato.options["redispatch"])
	# 	println("Adding NEX Constraints...")
	# 	add_net_position_constraints!(pomato)
	# end
	add_electricity_energy_balance!(pomato::POMATO)

	println("Adding Objective Function...")
	add_objective!(pomato)

	println("Solving...")
	t_start = time_ns()
	@time JuMP.optimize!(pomato.model)
	println("Objective: $(JuMP.objective_value(pomato.model))")
	println("Objective: $(JuMP.termination_status(pomato.model))")
	t_elapsed = time_ns() - t_start
	println("Solvetime: $(round(t_elapsed*1e-9, digits=2)) seconds")
	if JuMP.termination_status(pomato.model) != MOI.OPTIMAL
		check_infeasibility(pomato.model)
	end
	add_result!(pomato)
	println("Model Done!")
	return pomato
end

function run_redispatch_model(data::Data, options::Dict{String, Any}, redispatch_zones::Vector{String})
	pomato = run_market_model(data, options)
	redispatch_results = Dict{String, Result}()
	redispatch_results["market_results"] = pomato.result

	market_result = Dict{String, Array{Float64, 2}}()
	market_result["g_market"] = value.(pomato.model[:G])
	market_result["d_es_market"] = value.(pomato.model[:D_es])
	market_result["d_ph_market"] = value.(pomato.model[:D_ph])
	market_result["infeas_pos_market"] = value.(pomato.model[:INFEAS_EL_N_POS])
	market_result["infeas_neg_market"] = value.(pomato.model[:INFEAS_EL_N_NEG])

	load_redispatch_grid!(pomato)
	data_copy = deepcopy(pomato.data)
	market_result_copy = deepcopy(market_result)
	for redispatch_zone in redispatch_zones
		tmp_results = Dict{String, Result}()
		for timesteps in [t.index:t.index for t in data_copy.t]
			data = deepcopy(data_copy)
			data.t = data.t[timesteps]
			set_model_horizon!(data)
			market_result = Dict()
			for key in keys(market_result_copy)
				market_result[key] = market_result_copy[key][timesteps, :]
			end
			println("Initializing Redispatch Model for zone $(redispatch_zone) Timestep $(data.t[1].name)")
			pomato = POMATO(Model(with_optimizer(Gurobi.Optimizer, LogFile=data.folders["result_dir"]*"/log.txt")),
						    data, options)
			# pomato = POMATO(Model(optimizer_with_attributes(Gurobi.Optimizer,
			# 					  "LogFile" => data.folders["result_dir"]*"/log.txt")),
			# 			   	data, options)

			redispatch_model!(pomato, market_result, redispatch_zone);
			add_curtailment_constraints!(pomato, redispatch_zone);
			add_electricity_energy_balance!(pomato);

			# add_objective!(pomato)
			println("Solving...")
			t_start = time_ns()
			@time JuMP.optimize!(pomato.model)
			println("Objective: $(JuMP.objective_value(pomato.model))")
			println("Objective: $(JuMP.termination_status(pomato.model))")
			t_elapsed = time_ns() - t_start
			println("Solvetime: $(round(t_elapsed*1e-9, digits=2)) seconds")

			if JuMP.termination_status(pomato.model) != MOI.OPTIMAL
				check_infeasibility(pomato.model)
			end
			tmp_results[data.t[1].name] = add_result!(pomato)
		end
		redispatch_results["redispatch_"*redispatch_zone] = concat_results(tmp_results)
	end
	return redispatch_results
end
