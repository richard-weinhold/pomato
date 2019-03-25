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
# Output: Ordered Dicts of Types as definded in typedefinitions.jl (Plants, Node, Heatareas, Lines etc.)

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
# Ordered Dict nessesary for Load Flow Calculation
zones = OrderedDict{String, Zone}()
for z in 1:nrow(zones_mat)
    index = zones_mat[z, :index]
    nodes = nodes_mat[nodes_mat[:zone] .== index, :index]
    plants = filter(row -> row[:node] in nodes, plants_mat)[:index]

    demand_sum = Dict(zip(demand_el_mat[:index],
                          sum(demand_el_mat[Symbol(n)] for n in nodes)))

    newz = Zone(index, demand_sum, nodes, plants)

    if (size(ntc_mat, 2) > 1)
        ntc = filter(row -> row[:zone_i] == index, ntc_mat)
        newz.ntc = Dict(zip(ntc[:zone_j], ntc[:ntc]))
        newz.ntc[index] = 0
    end
    net_export_sum = Dict(zip(demand_el_mat[:index],
                          sum(net_export_mat[Symbol(n)] for n in nodes)))
    newz.net_export = net_export_sum

    if in(model_type, ["d2cf"])
        newz.net_position = Dict(zip(net_position_mat[:index],  
                                     net_position_mat[Symbol(index)]))
    end
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
    plants = plants_mat[plants_mat[:node] .== index, :index]
    demand_time = demand_el_mat[:index]
    demand_at_node = demand_el_mat[Symbol(index)]
    demand_dict = Dict(zip(demand_time, demand_at_node))
    newn = Node(index, zone, demand_dict, slack, plants)
    if slack
        newn.slack_zone = slack_zones[index]
    end
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
    plants = plants_mat[plants_mat[:heatarea] .=== index, :index]
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
    mc = plants_mat[p, :mc]*1.
    tech = plants_mat[p, :tech]
    newp = Plant(index, node, mc, eta, g_max, h_max, tech) 

    if Symbol(index) in names(availability_mat)
        newp.availability = Dict(zip(availability_mat[:index], 
                                     availability_mat[Symbol(index)]))
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
       plants,
       nodes, zones, heatareas,
       grid, dc_lines
end # end of function
