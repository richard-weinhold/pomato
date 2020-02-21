
mutable struct RAW
    options::Dict{String, Any}
    model_horizon::DataFrame
    plant_types::Dict{String, Any}
    nodes::DataFrame
    zones::DataFrame
    heatareas::DataFrame
    plants::DataFrame
    res_plants::DataFrame
    availability::DataFrame
    demand_el::DataFrame
    demand_h::DataFrame
    dc_lines::DataFrame
    ntc::DataFrame
    net_position::DataFrame
    net_export::DataFrame
    inflows::DataFrame
    reference_flows::DataFrame
    grid::DataFrame
    slack_zones::DataFrame
    function RAW(data_dir)
        raw = new()
        raw.options = JSON.parsefile(data_dir*"options.json"; dicttype=Dict)
        raw.plant_types = raw.options["plant_types"]

        raw.nodes = CSV.read(data_dir*"nodes.csv")
        raw.nodes[!, :int_idx] = collect(1:nrow(raw.nodes))
        raw.zones = CSV.read(data_dir*"zones.csv")
        raw.zones[!, :int_idx] = collect(1:nrow(raw.zones))
        raw.heatareas = CSV.read(data_dir*"heatareas.csv")
        raw.heatareas[!, :int_idx] = collect(1:nrow(raw.heatareas))
        plants = CSV.read(data_dir*"plants.csv")
        raw.plants =  filter(row -> !(row[:plant_type] in raw.plant_types["ts"]), plants)
        raw.res_plants =  filter(row -> row[:plant_type] in raw.plant_types["ts"], plants)
        raw.plants[!, :int_idx] = collect(1:nrow(raw.plants))
        raw.res_plants[!, :int_idx] = collect(1:nrow(raw.res_plants))
        raw.availability = CSV.read(data_dir*"availability.csv")
        raw.demand_el = CSV.read(data_dir*"demand_el.csv")
        raw.demand_h = CSV.read(data_dir*"demand_h.csv")
        raw.dc_lines = CSV.read(data_dir*"dclines.csv")
        raw.ntc = CSV.read(data_dir*"ntc.csv")
        raw.net_position = CSV.read(data_dir*"net_position.csv")
        raw.net_export = CSV.read(data_dir*"net_export.csv")
        raw.inflows = CSV.read(data_dir*"inflows.csv")
        # raw.reference_flows = CSV.read(data_dir*"reference_flows.csv");
        raw.grid = CSV.read(data_dir*"grid.csv")
        raw.slack_zones = CSV.read(data_dir*"slack_zones.csv")
        raw.model_horizon = DataFrame(index=collect(1:size(unique(raw.demand_el[:, :timestep]), 1)),
                                      timesteps=unique(raw.demand_el[:, :timestep]))
        return raw
    end
end

function load_redispatch_grid!(pomato::POMATO)
    grid = Vector{Grid}()
    grid_data = CSV.read(pomato.data.folders["data_dir"]*"redispatch_grid.csv")
    # grid_data = CSV.read(pomato.data.folders["data_dir"]*"redispatch_nodal.csv")
    for cbco in 1:nrow(grid_data)
        index = cbco
        name = grid_data[cbco, :index]
        ptdf = [grid_data[cbco, Symbol(node.name)] for node in pomato.data.nodes]
        ram = grid_data[cbco, :ram]*1.
        newcbco = Grid(index, name, ptdf, ram)
        newcbco.zone = coalesce(grid_data[cbco, :zone], nothing)
        push!(grid, newcbco)
    end
    pomato.data.grid = grid
    return grid
end

function save_result(result::Result, folder::String)
	create_folder(folder)
	println("Saving results to folder "*folder*"...")
	for (field, fieldtype) in zip(fieldnames(Result), fieldtypes(Result))
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

function create_folder(result_folder)
	if !isdir(result_folder)
	    println("Creating Results Folder")
		mkdir(result_folder)
	end
end
