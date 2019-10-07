# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v1.1. (required)
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

function read_model_data(data_dir::String)
println("Reading Model Data from: ", data_dir)

nodes_mat = CSV.read(data_dir*"nodes.csv")
zones_mat = CSV.read(data_dir*"zones.csv")
heatareas_mat = CSV.read(data_dir*"heatareas.csv")
plants_mat = CSV.read(data_dir*"plants.csv")
availability_mat = CSV.read(data_dir*"availability.csv")
demand_el_mat = CSV.read(data_dir*"demand_el.csv")
demand_h_mat = CSV.read(data_dir*"demand_h.csv")
dc_lines_mat = CSV.read(data_dir*"dclines.csv")
ntc_mat = CSV.read(data_dir*"ntc.csv")
net_position_mat = CSV.read(data_dir*"net_position.csv")
net_export_mat = CSV.read(data_dir*"net_export.csv")
inflows_mat = CSV.read(data_dir*"inflows.csv")
reference_flows_mat = CSV.read(data_dir*"reference_flows.csv");
grid_mat = CSV.read(data_dir*"cbco.csv")

slack_zones = CSV.read(data_dir*"slack_zones.csv")
options = JSON.parsefile(data_dir*"options.json"; dicttype=Dict)
plant_types = options["plant_types"]
model_type = options["type"]

println("Model Type: ", model_type)
# Prepare model_horizon
model_horizon = OrderedDict{Int, String}()
for t in unique(demand_el_mat[:, :timestep])
    model_horizon[Meta.parse(t[2:5])] = t
end
# Prepare Zones
# Ordered Dict nessesary for Load Flow Calculation
zones = OrderedDict{String, Zone}()
for z in 1:nrow(zones_mat)

    index = zones_mat[z, :index]
    nodes = nodes_mat[nodes_mat[:, :zone] .== index, :index]
    plants = filter(row -> row[:node] in nodes, plants_mat)[:, :index]
    demand = by(filter(col -> col[:node] in nodes, demand_el_mat), :timestep, sort=true, :demand_el => sum)
    demand_dict = Dict(zip(eachcol(demand)...))
    newz = Zone(index, demand_dict, nodes, plants)
    if (size(ntc_mat, 2) > 1)
        ntc = filter(row -> row[:zone_i] == index, ntc_mat)
        ntc_dict = Dict(zip(ntc[:, :zone_j], ntc[:, :ntc]))
        for zone in setdiff(zones_mat[:, :index], ntc[:, :zone_j])
            ntc_dict[zone] = 0
        end
        newz.ntc = ntc_dict
    end
    net_export = by(filter(col -> col[:node] in nodes, net_export_mat), :timestep, sort=true, :net_export => sum)
    newz.net_export = Dict(zip(eachcol(net_export)...))
    # if in(model_type, ["d2cf"])
    net_position = by(filter(col -> col[:zone] == index, net_position_mat), :timestep, sort=true, :net_position => sum)
    newz.net_position = Dict(zip(eachcol(net_position)...))
    # end
    zones[newz.index] = newz
end
# Prepare Nodes
# Ordered Dict nessesary for Load Flow Calculation
nodes = OrderedDict{String, Node}()
for n in 1:nrow(nodes_mat)
    index = nodes_mat[n, :index]
    slack = uppercase(nodes_mat[n, :slack]) == "TRUE"
    name = nodes_mat[n, :name]
    zone = nodes_mat[n, :zone]
    plants = plants_mat[plants_mat[:, :node] .== index, :index]

    demand = by(filter(col -> col[:node] == index, demand_el_mat), :timestep, sort=true, :demand_el => sum)
    demand_dict = Dict(zip(eachcol(demand)...))
    newn = Node(index, zone, demand_dict, slack, plants)
    if slack
        # newn.slack_zone = slack_zones[index]
        newn.slack_zone = slack_zones[:, :index][slack_zones[:, Symbol(index)] .== 1]
    end
    net_export = by(filter(col -> col[:node] == index, net_export_mat), :timestep, sort=true, :net_export => sum)
    newn.net_export = Dict(zip(eachcol(net_export)...))
    nodes[newn.index] = newn
end
#Prepare Heatareas
heatareas = Dict{String, Heatarea}()
for h in 1:nrow(heatareas_mat)
    index = heatareas_mat[h, :index]
    demand = by(filter(col -> col[:heatarea] == index, demand_h_mat), :timestep, sort=true, :demand_h => sum)
    demand_dict = Dict(zip(eachcol(demand)...))
    plants = plants_mat[plants_mat[:, :heatarea] .=== index, :index]
    newh = Heatarea(index, demand_dict, plants)
    heatareas[newh.index] = newh
end


# Prepare Plants
plants = Dict{String, Plant}()
for p in 1:nrow(plants_mat)
    index = string(plants_mat[p, :index])
    node = plants_mat[p, :node]
    eta = plants_mat[p, :eta]*1.
    g_max = plants_mat[p, :g_max]*1.
    h_max = plants_mat[p, :h_max]*1.
    mc_el = plants_mat[p, :mc_el]*1.
    mc_heat = plants_mat[p, :mc_heat]*1.
    plant_type = plants_mat[p, :plant_type]
    newp = Plant(index, node, mc_el, mc_heat, eta, g_max, h_max, plant_type)
    if index in availability_mat[:, :plant]
        availability = by(filter(col -> col[:plant] == index, availability_mat),
                          :timestep, sort=true, :availability => sum)
        newp.availability = Dict(zip(eachcol(availability)...))
    end
    if index in inflows_mat[:, :plant]
        inflow = by(filter(col -> col[:plant] == index, inflows_mat),
                          :timestep, sort=true, :inflow => sum)
        newp.inflow = Dict(zip(eachcol(inflow)...))
    end

    if index in plants_mat[.!(ismissing.(plants_mat[:, :storage_capacity])),:index]
        newp.storage_capacity = plants_mat[p, :storage_capacity]
    end
    plants[newp.index] = newp
end
# Prepare dc_lines
dc_lines = Dict{String, DC_Line}()
for l in 1:nrow(dc_lines_mat)
    index = dc_lines_mat[l, :index]
    node_i = dc_lines_mat[l, :node_i]
    node_j = dc_lines_mat[l, :node_j]
    maxflow = dc_lines_mat[l, :maxflow]*1.
    newl = DC_Line(index, node_i, node_j, maxflow)
    dc_lines[newl.index] = newl
end

# Prepare Grid Representation
grid = Dict{String, Grid}()
for cbco in 1:nrow(grid_mat)
    index = grid_mat[cbco, :index]

    if in(model_type, ["cbco_zonal", "zonal"])
        ptdf = [x for x in grid_mat[cbco, Symbol.(collect(keys(zones)))]]
    else
        ptdf = [x for x in grid_mat[cbco, Symbol.(collect(keys(nodes)))]]
    end
    ram = grid_mat[cbco, :ram]*1.
    newcbco = Grid(index, ptdf, ram)
    if in(model_type, ["d2cf"])
        newcbco.reference_flow = Dict(collect(zip(reference_flows_mat[:, :index],
                                                  reference_flows_mat[:, Symbol(index)])))
    end
    grid[newcbco.index] = newcbco
end

println("Data Prepared")
return model_horizon, options, plant_types,
       plants,
       nodes, zones, heatareas,
       grid, dc_lines
end # end of function
