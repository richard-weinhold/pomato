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
function read_model_data(data_dir)
println("Reading Model Data")

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
net_export_nodes_mat = CSV.read(data_dir*"net_export_nodes.csv") 

slack_zones = JSON.parsefile(data_dir*"slack_zones.json"; dicttype=Dict)
opt_setup = JSON.parsefile(data_dir*"opt_setup.json"; dicttype=Dict)

model_type = opt_setup["opt"]

# Prepare Zones
zones = OrderedDict()
for z in 1:nrow(zones_mat)
    index = zones_mat[z, :index]
    demand_sum = Dict()
    net_export_sum = Dict()
    for t in 1:nrow(demand_el_mat)
        d_sum = sum(demand_el_mat[t, Symbol(n)] for n in nodes_mat[nodes_mat[:zone] .== index, :index])
        nex_sum = sum(net_export_nodes_mat[t, Symbol(n)] for n in nodes_mat[nodes_mat[:zone] .== index, :index])
        demand_sum[demand_el_mat[t, :index]] = d_sum
        net_export_sum[demand_el_mat[t, :index]] = nex_sum
    end
    nodes = nodes_mat[nodes_mat[:zone] .== index, :index]
    newz = Zone(index, demand_sum, nodes)
    newz.net_export = net_export_sum
    zones[newz.index] = newz
end

# Prapare Nodes
nodes_in_zone = Dict()
for z in collect(keys(zones))
    nodes_in_zone[z] = Dict()
end

nodes = OrderedDict()
for n in 1:nrow(nodes_mat)
    index = nodes_mat[n, :index]
    slack = uppercase(nodes_mat[n, :slack]) == "TRUE"
    name = nodes_mat[n, :name]
    zone = nodes_mat[n, :zone]
    zone = zones[zone]

    newn = Node(index, zone, slack, name)

    demand_time = demand_el_mat[:index]
    demand_at_node = demand_el_mat[Symbol(newn.index)]
    demand_dict = Dict(zip(demand_time, demand_at_node))
    newn.demand = demand_dict

    net_export_time = net_export_nodes_mat[:index]
    net_export_at_node = net_export_nodes_mat[Symbol(newn.index)]
    net_export_dict = Dict(zip(net_export_time, net_export_at_node))
    newn.net_export = net_export_dict

    nodes[newn.index] = newn
    nodes_in_zone[newn.zone.index][newn.index] = newn
end

#Prepare Heatareas
heatareas = Dict()
for h in 1:nrow(heatareas_mat)
    index = heatareas_mat[h, :index]
    demand_time = demand_h_mat[:index]
    demand_in_ha = demand_h_mat[Symbol(index)]
    demand_dict = Dict(zip(demand_time, demand_in_ha))
    newh = Heatarea(index, demand_dict)
    heatareas[newh.index] = newh
end

# Prepare Plants
plants = Dict()
plants_in_zone = Dict()
for z in collect(keys(zones))
    plants_in_zone[z] = Dict()
end
plants_in_ha = Dict()
for h in collect(keys(heatareas))
    plants_in_ha[h] = Dict()
end

plants_at_node = Dict()
for n in collect(keys(nodes))
    plants_at_node[n] = Dict()
end

p_with_avail = names(availability_mat)
p_with_avail = setdiff(p_with_avail, [:index])
p_with_avail = [String(p) for p in p_with_avail]

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
            println("Warning: Heatarea $(heatareas[plants_mat[p, :heatarea]].index) not found in heatarea list for plant $(newp.index)")
        end
    end
    # Cast to string necessary to recast from symbol belwo
    plants[newp.index] = newp
    plants_at_node[newp.node.index][newp.index] = newp
    plants_in_zone[newp.node.zone.index][newp.index] = newp
end

# Prepare Availability
availabilities = Dict()
time_data = availability_mat[:index]
for p in setdiff(names(availability_mat), [:index])
    plant = plants[String(p)]
    avail_dict = Dict(zip(time_data, availability_mat[p]))
    newa = Availability(plant, avail_dict)
    availabilities[String(p)] = newa
end

# Prepare dc_lines
dc_lines = Dict()
for l in 1:nrow(dc_lines_mat)
    index = dc_lines_mat[l, :index]
    node_i = nodes[dc_lines_mat[l, :node_i]]
    node_j = nodes[dc_lines_mat[l, :node_j]]
    capacity = dc_lines_mat[l, :capacity]

    newl = DC_Line(index, node_i, node_j, capacity)
    dc_lines[newl.index] = newl
end

# Build NTC Matrix
ntc = Dict()
for n in 1:nrow(ntc_mat)
    i = ntc_mat[n, :zone_i]
    j = ntc_mat[n, :zone_j]
    ntc[(i, j)] = ntc_mat[n, :ntc]
end
for z in collect(keys(zones))
    ntc[(z, z)] = 0
end

# Load CBCOs if available
if in(model_type,  ["cbco_nodal", "cbco_zonal", "nodal"])
    println("Loading CBCOs")
    cbco = JSON.parsefile(data_dir*"cbco.json")

elseif in(model_type,  ["d2cf"])
    cbco = JSON.parsefile(data_dir*"cbco.json")
    for z in collect(keys(zones))
        net_position_time = net_position_mat[:index]
        net_position_of_zone = net_position_mat[Symbol(z)]
        net_position_dict = Dict(zip(net_position_time, net_position_of_zone))
        zones[z].net_position = net_position_dict
    end
else
    cbco = Dict()
end

# Prepare model_horizon
model_horizon = OrderedDict()
for t in demand_el_mat[:index]
    model_horizon[parse(t[2:5])] = t
end

println("Data Prepared")
return model_horizon, opt_setup, 
       plants, plants_in_ha, plants_at_node, plants_in_zone, availabilities,
       nodes, zones, slack_zones, heatareas, nodes_in_zone,
       ntc, dc_lines, cbco
end # end of function
