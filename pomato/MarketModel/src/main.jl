""" Run the models, defined in create_model.jl"""

function run_market_model(wdir, data_dir, return_result=false)
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
	if return_result
		return pomato_results
	end
end

function run_market_model_redispatch(wdir, data_dir; return_result=false)
	println("Read Model Data..")
	options, data = read_model_data(wdir*"/data_temp/julia_files"*data_dir)
	data.folders = Dict("wdir" => wdir,
						"data_dir" => wdir*"/data_temp/julia_files"*data_dir,
						"result_dir" => wdir*"/data_temp/julia_files/results/"*Dates.format(now(), "dmm_HHMM"))
	create_folder(data.folders["result_dir"])
	pomato_results = run_redispatch_model(data, options)
	for result in keys(pomato_results)
		save_result(pomato_results[result], data.folders["result_dir"]*"_"*result)
	end
	println("Everything Done!")
	if return_result
		return pomato_results
	end
end
