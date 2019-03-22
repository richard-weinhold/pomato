# -------------------------------------------------------------------------------------------------
# POMATO - Power Market Tool (C) 2018
# Current Version: Pre-Release-2018
# Created by Robert Mieth and Richard Weinhold
# Licensed under LGPL v3
#
# Language: Julia, v0.6.2 (required)
# ----------------------------------
#
# This file:
# POMATO optimization kernel
# Called by julia_interface.py, reads pre-processed data from /julia/data/
# Output: Optimization results saved in /julia/results/
# -------------------------------------------------------------------------------------------------


# Read Excel Data into Data Frame
# Input: Pre-Processed data from /julia/data/
# Output: Ordered Dicts of Plants, Node, Heatareas, Lines etc.,
#         as well as PTDF matrix, potential cbcos and necessary mappings
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
reference_flows_mat = CSV.read(data_dir*"reference_flows.csv");

grid_mat = CSV.read(data_dir*"cbco.csv")

slack_zones = JSON.parsefile(data_dir*"slack_zones.json"; dicttype=Dict)
options = JSON.parsefile(data_dir*"options.json"; dicttype=Dict)
model_type = options["type"]

println("Model Type: ", model_type)
# Prepare Zones
zones = OrderedDict{String, Zone}()
for z in 1:nrow(zones_mat)
    index = zones_mat[z, :index]
    demand_sum = Dict()
    net_export_sum = Dict()
    for t in 1:nrow(demand_el_mat)
        d_sum = sum(demand_el_mat[t, Symbol(n)] for n in nodes_mat[nodes_mat[:zone] .== index, :index])
        nex_sum = sum(net_export_mat[t, Symbol(n)] for n in nodes_mat[nodes_mat[:zone] .== index, :index])
        demand_sum[demand_el_mat[t, :index]] = d_sum
        net_export_sum[demand_el_mat[t, :index]] = nex_sum
    end
    nodes = nodes_mat[nodes_mat[:zone] .== index, :index]
    plants = filter(row -> row[:node] in nodes, plants_mat)[:index]

    newz = Zone(index, demand_sum, nodes, plants)
    newz.net_export = net_export_sum
    if in(model_type, ["d2cf"])
        newz.net_position = Dict(zip(net_position_mat[:index],  net_position_mat[Symbol(newz.index)]))
    end

    zones[newz.index] = newz
end


nodes = OrderedDict{String, Node}()
for n in 1:nrow(nodes_mat)
    index = nodes_mat[n, :index]
    slack = uppercase(nodes_mat[n, :slack]) == "TRUE"
    name = nodes_mat[n, :name]
    zone = nodes_mat[n, :zone]
    zone = zones[zone]
    plants = plants_mat[plants_mat[:node] .== index, :index]

    newn = Node(index, zone, slack, name, plants)

    demand_time = demand_el_mat[:index]
    demand_at_node = demand_el_mat[Symbol(newn.index)]
    demand_dict = Dict(zip(demand_time, demand_at_node))
    newn.demand = demand_dict

    net_export_time = net_export_mat[:index]
    net_export_at_node = net_export_mat[Symbol(newn.index)]
    net_export_dict = Dict(zip(net_export_time, net_export_at_node))
    newn.net_export = net_export_dict

    nodes[newn.index] = newn
end

#Prepare Heatareas
heatareas = Dict{String, Heatarea}()
for h in 1:nrow(heatareas_mat)
    index = heatareas_mat[h, :index]
    demand_time = demand_h_mat[:index]
    demand_in_ha = demand_h_mat[Symbol(index)]
    demand_dict = Dict(zip(demand_time, demand_in_ha))
    newh = Heatarea(index, demand_dict)
    heatareas[newh.index] = newh
end

# Prepare Plants
plants = Dict{String, Plant}()
for p in 1:nrow(plants_mat)
    index = string(plants_mat[p, :index])
    efficiency = plants_mat[p, :eta]
    g_max = plants_mat[p, :g_max]
    h_max = plants_mat[p, :h_max]
    mc = plants_mat[p, :mc]
    tech = plants_mat[p, :tech]
    newp = Plant(index, efficiency, g_max, h_max, tech, mc)
    try
        newp.node = nodes[plants_mat[p, :node]]
    catch
        println("Warning: Node $(plants_mat[p, :node]) not found in node-list for plant $(newp.index).")
    end

    if !(isa(plants_mat[p, :heatarea], Missing))
        try
            newp.heatarea = heatareas[plants_mat[p, :heatarea]]
            plants_in_ha[newp.heatarea.index][newp.index] = newp
        catch
            println("Warning: Heatarea $(heatareas[plants_mat[p, :heatarea]].index)", 
                    " not found in heatarea list for plant $(newp.index)")
        end
    end
    # Cast to string necessary to recast from symbol belwo
    plants[newp.index] = newp
end

# Prepare Availability
availabilities = Dict{String, Availability}()
time_data = availability_mat[:index]
for p in setdiff(names(availability_mat), [:index])
    plant = plants[String(p)]
    avail_dict = Dict(zip(time_data, availability_mat[p]))
    newa = Availability(plant, avail_dict)
    availabilities[String(p)] = newa
end

# Prepare dc_lines
dc_lines = Dict{String, DC_Line}()
for l in 1:nrow(dc_lines_mat)
    index = dc_lines_mat[l, :index]
    node_i = nodes[dc_lines_mat[l, :node_i]]
    node_j = nodes[dc_lines_mat[l, :node_j]]
    maxflow = dc_lines_mat[l, :maxflow]
    newl = DC_Line(index, node_i, node_j, maxflow)
    dc_lines[newl.index] = newl
end

# Build NTC Matrix
ntc = Dict{Tuple, Number}()
for n in 1:nrow(ntc_mat)
    i = ntc_mat[n, :zone_i]
    j = ntc_mat[n, :zone_j]
    ntc[(i, j)] = ntc_mat[n, :ntc]
end
for z in collect(keys(zones))
    ntc[(z, z)] = 0
end

# Prepare Grid Representation
grid = Dict{String, Grid}()
for cbco in 1:nrow(grid_mat)
    index = grid_mat[cbco, :index]

    if in(model_type, ["cbco_zonal"])
        ptdf = [x for x in grid_mat[cbco, Symbol.(collect(keys(zones)))]]
    else
        ptdf = [x for x in grid_mat[cbco, Symbol.(collect(keys(nodes)))]]
    end
    ram = grid_mat[cbco, :ram]
    newcbco = Grid(index, ptdf, ram)
    if in(model_type, ["d2cf"])
        newcbco.reference_flow = Dict(collect(zip(reference_flows_mat[:, :index],
                                                  reference_flows_mat[:, Symbol(index)])))
    end
    grid[newcbco.index] = newcbco
end

# Prepare model_horizon
model_horizon = OrderedDict{Int, String}()
for t in demand_el_mat[:index]
    model_horizon[Meta.parse(t[2:5])] = t
end

println("Data Prepared")
return model_horizon, options,
       plants, availabilities,
       nodes, zones, heatareas,
       grid, dc_lines, slack_zones, ntc
end # end of function
