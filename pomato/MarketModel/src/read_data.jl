# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v1.3. (required)
# ----------------------------------
#
# This file:
# POMATO optimization kernel
# Called by julia_interface.py, reads pre-processed data from /julia/data/
# Output: Optimization results saved in /julia/results/
# -----------------------------------------------------------------------

# Read Data into Data Frame
# Input: Pre-Processed data from /julia/data/
# Output: Ordered Dicts of Types as definded in typedefinitions.jl (Plants, Node, Heatareas, Grid etc.)
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

function read_model_data(data_dir::String)
    println("Reading Model Data from: ", data_dir)
    raw = RAW(data_dir)

    task_zones = Threads.@spawn populate_zones(raw)
    task_nodes = Threads.@spawn populate_nodes(raw)
    task_heatareas = Threads.@spawn populate_heatareas(raw)
    task_plants = Threads.@spawn populate_plants(raw)
    task_res_plants = Threads.@spawn populate_res_plants(raw)
    task_dc_lines = Threads.@spawn populate_dclines(raw)
    task_grid = Threads.@spawn populate_grid(raw)

    zones = fetch(task_zones)
    nodes = fetch(task_nodes)
    heatareas = fetch(task_heatareas)
    plants = fetch(task_plants)
    res_plants = fetch(task_res_plants)
    dc_lines = fetch(task_dc_lines)
    grid = fetch(task_grid)

    timesteps = popolate_timesteps(raw)

    println("Data Prepared")
    return raw.options, Data(nodes, zones, heatareas, plants, res_plants, grid, dc_lines, timesteps)
end # end of function

function popolate_timesteps(raw::RAW)
    timesteps = Vector{Timestep}()
    for t in 1:nrow(raw.model_horizon)
        index = t
        name = raw.model_horizon[t, :timesteps]
        push!(timesteps, Timestep(index, name))
    end
    return timesteps
end

function populate_zones(raw::RAW)
    zones = Vector{Zone}()
    for z in 1:nrow(raw.zones)
        index = z
        name = raw.zones[z, :index]
        nodes_idx = raw.nodes[raw.nodes[:, :zone] .== name, :int_idx]
        nodes_name = raw.nodes[raw.nodes[:, :zone] .== name, :index]
        plants = filter(row -> row[:node] in nodes_name, raw.plants)[:, :int_idx]
        res_plants = filter(row -> row[:node] in nodes_name, raw.res_plants)[:, :int_idx]
        demand = by(filter(col -> col[:node] in nodes_name, raw.demand_el), :timestep, sort=true, :demand_el => sum)
        newz = Zone(index, name, demand[:, :demand_el_sum], nodes_idx, plants, res_plants)
        if (size(raw.ntc, 2) > 1)
            ntc = filter(row -> row[:zone_i] == name, raw.ntc)
            newz.ntc = [zone in ntc[:, :zone_j] ?
                        ntc[ntc[:, :zone_j] .== zone, :ntc][1] :
                        0 for zone in raw.zones[:, :index]]
        end

        net_export = by(filter(col -> col[:node] in nodes_name, raw.net_export), :timestep, sort=true, :net_export => sum)
        newz.net_export = net_export[:, :net_export_sum]

        net_position = by(filter(col -> col[:zone] == name, raw.net_position), :timestep, sort=true, :net_position => sum)
        if size(net_position, 1) > 0
            newz.net_position = net_position[:, :net_position_sum]
        end
        push!(zones, newz)
    end
    return zones
end

function populate_nodes(raw::RAW)
    nodes = Vector{Node}()
    for n in 1:nrow(raw.nodes)
        index = n
        name = raw.nodes[n, :index]
        slack = uppercase(raw.nodes[n, :slack]) == "TRUE"
        zone_name = raw.nodes[n, :zone]
        zone_idx = raw.zones[raw.zones[:, :index] .== zone_name, :int_idx][1]
        plants = raw.plants[raw.plants[:, :node] .== name, :int_idx]
        res_plants = raw.res_plants[raw.res_plants[:, :node] .== name, :int_idx]
        demand = by(filter(col -> col[:node] == name, raw.demand_el), :timestep, sort=true, :demand_el => sum)
        newn = Node(index, name, zone_idx, demand[:, :demand_el_sum], slack, plants, res_plants)
        if slack
            # newn.slack_zone = slack_zones[index]
            slack_zone = raw.slack_zones[:, :index][raw.slack_zones[:, Symbol(name)] .== 1]
            newn.slack_zone = filter(col -> col[:index] in slack_zone, raw.nodes)[:, :int_idx]
        end
        net_export = by(filter(col -> col[:node] == name, raw.net_export), :timestep, sort=true, :net_export => sum)
        newn.net_export = net_export[:, :net_export_sum]
        push!(nodes, newn)
    end
    return nodes
end

function populate_heatareas(raw::RAW)
    heatareas = Vector{Heatarea}()
    for h in 1:nrow(raw.heatareas)
        index = h
        name = raw.heatareas[h, :index]
        demand = by(filter(col -> col[:heatarea] == name, raw.demand_h), :timestep, sort=true, :demand_h => sum)
        plants = raw.plants[(raw.plants[:, :heatarea] .=== name).&(raw.plants[:, :h_max] .> 0), :int_idx]
        res_plants = raw.res_plants[(raw.res_plants[:, :heatarea] .=== name).&(raw.res_plants[:, :h_max] .> 0), :int_idx]
        newh = Heatarea(index, name, demand[:, :demand_h_sum], plants, res_plants)
        push!(heatareas, newh)
    end
    return heatareas
end

function populate_plants(raw::RAW)
    plants =  Vector{Plant}()
    for p in 1:nrow(raw.plants)
        index = p
        name = string(raw.plants[p, :index])
        node_name = raw.plants[p, :node]
        node_idx = raw.nodes[raw.nodes[:, :index] .== node_name, :int_idx][1]
        eta = raw.plants[p, :eta]*1.
        g_max = raw.plants[p, :g_max]*1.
        h_max = raw.plants[p, :h_max]*1.
        mc_el = raw.plants[p, :mc_el]*1.
        mc_heat = raw.plants[p, :mc_heat]*1.
        plant_type = raw.plants[p, :plant_type]
        newp = Plant(index, name, node_idx, mc_el,
                     mc_heat, eta, g_max, h_max, plant_type)
        if plant_type in union(raw.plant_types["hs"], raw.plant_types["es"])
            newp.inflow = raw.inflows[raw.inflows[:, :plant] .== name, :inflow]
            newp.storage_capacity = raw.plants[p, :storage_capacity]
        end
        push!(plants, newp)
    end
    return plants
end

function populate_res_plants(raw::RAW)
    res_plants = Vector{Renewables}()
    for res in 1:nrow(raw.res_plants)
        index = res
        name = string(raw.res_plants[res, :index])
        node_name = raw.res_plants[res, :node]
        node_idx = raw.nodes[raw.nodes[:, :index] .== node_name, :int_idx][1]
        g_max = raw.res_plants[res, :g_max]*1.
        h_max = raw.res_plants[res, :h_max]*1.
        mc_el = raw.res_plants[res, :mc_el]*1.
        mc_heat = raw.res_plants[res, :mc_heat]*1.
        plant_type = raw.res_plants[res, :plant_type]
        availability = by(filter(col -> col[:plant] == name, raw.availability),
                          :timestep, sort=true, :availability => sum)
        newres = Renewables(index, name, g_max, h_max, mc_el, mc_heat,
                            availability[:, :availability_sum],
                            node_idx, plant_type)
        push!(res_plants, newres)
    end
    return res_plants
end

function populate_dclines(raw::RAW)
    dc_lines = Vector{DC_Line}()
    for dc in 1:nrow(raw.dc_lines)
        index = dc
        name = raw.dc_lines[dc, :index]
        node_i = raw.dc_lines[dc, :node_i]
        node_j = raw.dc_lines[dc, :node_j]
        node_i_idx = raw.nodes[raw.nodes[:, :index] .== node_i, :int_idx][1]
        node_j_idx = raw.nodes[raw.nodes[:, :index] .== node_j, :int_idx][1]
        maxflow = raw.dc_lines[dc, :maxflow]*1.
        newdc = DC_Line(index, name, node_i_idx, node_j_idx, maxflow)
        push!(dc_lines, newdc)
    end
    return dc_lines
end

function populate_grid(raw::RAW)
    grid = Vector{Grid}()
    for cbco in 1:nrow(raw.grid)
        index = cbco
        name = raw.grid[cbco, :index]
        if in(raw.options["type"], ["cbco_zonal", "zonal"])
            ptdf = [x for x in raw.grid[cbco, Symbol.(collect(raw.zones[:,:index]))]]
        else
            ptdf = [x for x in raw.grid[cbco, Symbol.(collect(raw.nodes[:,:index]))]]
        end
        ram = raw.grid[cbco, :ram]*1.
        newcbco = Grid(index, name, ptdf, ram)
        if in(raw.options["type"], ["d2cf"])
            newcbco.reference_flow = Dict(collect(zip(raw.reference_flows[:, :index],
                                                      raw.reference_flows[:, Symbol(index)])))
        end
        if :zone in names(raw.grid)
            newcbco.zone = coalesce(raw.grid[cbco, :zone], nothing)
        end
        if :timestep in names(raw.grid)
            newcbco.timestep = raw.grid[cbco, :timestep]
        end
        push!(grid, newcbco)
    end
    return grid
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
